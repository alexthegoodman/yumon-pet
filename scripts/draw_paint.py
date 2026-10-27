#!/usr/bin/env python3
"""
Interactive 16x16 Coarse-to-Fine Studio for Yumon Paint.

Allows drawing an artwork concept on a 16x16 grid, typing an art prompt,
and feeding the manual 16x16 drawing directly into Pass 3 (32x32),
which cascades into Pass 4 (64x64), and finishes with the Neural Upscaler (256x256).

Usage:
    python scripts/draw_paint.py
    (or uv run python scripts/draw_paint.py)
"""

import os
import sys
import json
import time
import socket
import base64
import uuid
import shutil
import webbrowser
import subprocess
from pathlib import Path
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TMP_DIR = PROJECT_ROOT / "tmp" / "draw_paint"
TMP_DIR.mkdir(parents=True, exist_ok=True)

RELEASE_EXE = PROJECT_ROOT / "target" / "release" / "yumon_paint.exe"
DEBUG_EXE = PROJECT_ROOT / "target" / "debug" / "yumon_paint.exe"

def find_paint_binary() -> Path:
    if RELEASE_EXE.exists():
        return RELEASE_EXE
    if DEBUG_EXE.exists():
        return DEBUG_EXE
    return RELEASE_EXE

def palette_to_rgb(idx: int):
    idx = max(0, min(255, idx))
    r_bin = (idx >> 5) & 0x07
    g_bin = (idx >> 2) & 0x07
    b_bin = idx & 0x03
    r = (r_bin * 255) // 7
    g = (g_bin * 255) // 7
    b = (b_bin * 255) // 3
    return [r, g, b]

def rgb_to_palette(r: int, g: int, b: int) -> int:
    r_bin = min(7, (r * 7 + 127) // 255)
    g_bin = min(7, (g * 7 + 127) // 255)
    b_bin = min(3, (b * 3 + 127) // 255)
    return r_bin * 32 + g_bin * 4 + b_bin

def file_to_base64_data_uri(path: Path) -> str:
    if not path.is_file():
        return ""
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

class StudioServer(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Keep terminal log concise
        return

    def send_json(self, status_code: int, data: dict):
        try:
            body = json.dumps(data).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path

            if path in ("/", "/index.html"):
                body = HTML_CONTENT.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            elif path == "/api/status":
                bin_path = find_paint_binary()
                self.send_json(200, {
                    "status": "ready" if bin_path.exists() else "building",
                    "binary": str(bin_path),
                    "binary_exists": bin_path.exists(),
                    "vocab_exists": (PROJECT_ROOT / "training-cache" / "diffusion_talk_vocab.json").exists(),
                    "pass3_ckpt": (PROJECT_ROOT / "checkpoints" / "yumon-art-pass3-db").exists(),
                    "pass4_ckpt": (PROJECT_ROOT / "checkpoints" / "yumon-art-pass4-db").exists(),
                    "upscaler_ckpt": (PROJECT_ROOT / "checkpoints" / "yumon-upscaler-64").exists(),
                })
                return

            self.send_error(404, "Not Found")
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/generate":
            content_len = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_len).decode("utf-8")
            try:
                payload = json.loads(raw_body)
            except Exception as e:
                self.send_json(400, {"error": f"Invalid JSON payload: {e}"})
                return

            prompt = payload.get("prompt", "").strip()
            if not prompt:
                prompt = "a cute pixel art creature in an enchanted magical forest"

            pixels = payload.get("pixels", [])
            if len(pixels) != 256:
                self.send_json(400, {"error": f"Pixels array must have exactly 256 items (16x16 grid), got {len(pixels)}"})
                return

            # Extract generation parameters
            shape_guidance = float(payload.get("shape_guidance", 0.25))
            shape_guidance_p4 = float(payload.get("shape_guidance_pass4", 5.5))
            temp_p3 = float(payload.get("temperature_pass3", 1.25))
            temp_p4 = float(payload.get("temperature_pass4", 1.20))
            top_k = int(payload.get("top_k", 24))
            top_p = float(payload.get("top_p", 0.73))

            run_id = f"gen_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            input_json_path = TMP_DIR / f"{run_id}_input.json"
            output_png_path = TMP_DIR / f"{run_id}.png"

            # Write input drawing JSON
            with open(input_json_path, "w", encoding="utf-8") as f:
                json.dump({"prompt": prompt, "pixels": pixels}, f)

            bin_path = find_paint_binary()
            if not bin_path.exists():
                self.send_json(500, {
                    "error": f"Yumon Paint binary not found at {bin_path}. Run `cargo build --release --bin yumon_paint` first."
                })
                return

            cmd = [
                str(bin_path),
                "--prompt", prompt,
                "--input-pass2", str(input_json_path),
                "--output", str(output_png_path),
                "--save-intermediate",
                "--shape-guidance", str(shape_guidance),
                "--shape-guidance-pass4", str(shape_guidance_p4),
                "--temperature-pass3", str(temp_p3),
                "--temperature-pass4", str(temp_p4),
                "--top-k", str(top_k),
                "--top-p", str(top_p),
                "--passes", "5",
            ]

            t_start = time.time()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    encoding="utf-8",
                    errors="replace",
                    timeout=360,
                )
            except subprocess.TimeoutExpired:
                self.send_json(504, {"error": "Generation timed out after 360 seconds."})
                return
            except Exception as e:
                self.send_json(500, {"error": f"Failed running yumon_paint: {e}"})
                return

            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            elapsed = round(time.time() - t_start, 2)

            if proc.returncode != 0:
                print(f"[Error] yumon_paint stderr:\n{stderr}", file=sys.stderr)
                self.send_json(500, {
                    "error": "yumon_paint process failed",
                    "stderr": stderr[-2000:],
                    "stdout": stdout[-2000:],
                })
                return

            # Read intermediate outputs
            stem = run_id
            p2_file = TMP_DIR / f"{stem}_pass2.png"
            p3_file = TMP_DIR / f"{stem}_pass3.png"
            p4_file = TMP_DIR / f"{stem}_pass4.png"
            upscaled_file = output_png_path

            p2_b64 = file_to_base64_data_uri(p2_file)
            p3_b64 = file_to_base64_data_uri(p3_file)
            p4_b64 = file_to_base64_data_uri(p4_file)
            up_b64 = file_to_base64_data_uri(upscaled_file)

            self.send_json(200, {
                "status": "ok",
                "run_id": run_id,
                "duration_seconds": elapsed,
                "prompt": prompt,
                "images": {
                    "pass2": p2_b64,
                    "pass3": p3_b64,
                    "pass4": p4_b64,
                    "upscaled": up_b64,
                },
                "log": stdout[-1500:],
            })
            return

        self.send_error(404, "Not Found")

HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Yumon Paint Studio — 16x16 Interactive Coarse-to-Fine Generator</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/firacode/6.2.0/fira_code.min.css"/>
<style>
  :root {
    --bg-dark: #0f1115;
    --panel-bg: #181b22;
    --panel-sub: #1f232c;
    --border-color: #2c3340;
    --border-focus: #00e5ff;
    --accent-cyan: #00e5ff;
    --accent-magenta: #ff007f;
    --accent-gold: #ffb703;
    --accent-green: #06d6a0;
    --accent-purple: #9d4edd;
    --text-primary: #f0f4f8;
    --text-muted: #8b9bb4;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { background: var(--bg-dark); color: var(--text-primary); display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

  header {
    background: var(--panel-bg);
    border-bottom: 1px solid var(--border-color);
    padding: 10px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.4);
    z-index: 10;
  }
  .title-group { display: flex; align-items: center; gap: 12px; }
  .logo-badge {
    background: linear-gradient(135deg, #00e5ff 0%, #ff007f 100%);
    width: 32px; height: 32px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; box-shadow: 0 0 14px rgba(0,229,255,0.4);
  }
  .title-text h1 { font-size: 1.15rem; font-weight: 700; color: #fff; letter-spacing: 0.3px; }
  .title-text span { font-size: 0.78rem; color: var(--text-muted); font-family: "Fira Code", monospace; }

  .header-actions { display: flex; gap: 14px; align-items: center; }
  .status-pill {
    font-size: 0.78rem; font-weight: 600; padding: 5px 12px; border-radius: 20px;
    background: #1e2430; border: 1px solid var(--border-color); display: flex; align-items: center; gap: 6px;
  }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent-green); box-shadow: 0 0 8px var(--accent-green); }
  .status-dot.busy { background: var(--accent-gold); box-shadow: 0 0 8px var(--accent-gold); }

  /* Main Workspace */
  main { flex: 1; display: grid; grid-template-columns: 460px 1fr; overflow: hidden; }

  /* Left Column: Canvas & Controls */
  .editor-pane {
    background: var(--panel-bg);
    border-right: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    padding: 18px 20px;
    overflow-y: auto;
    gap: 16px;
  }

  /* Right Column: Cascaded Results & Gallery */
  .results-pane {
    background: radial-gradient(circle at top right, #1a202c 0%, #0d1015 100%);
    display: flex;
    flex-direction: column;
    padding: 22px 26px;
    overflow-y: auto;
    gap: 20px;
  }

  .section-label {
    font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted);
    font-weight: 700; letter-spacing: 1px; display: flex; justify-content: space-between;
    align-items: center; margin-bottom: 6px;
  }

  /* Canvas Section */
  .canvas-box {
    display: flex;
    flex-direction: column;
    align-items: center;
    background: var(--panel-sub);
    padding: 14px;
    border-radius: 12px;
    border: 1px solid var(--border-color);
  }

  .canvas-container {
    position: relative;
    border: 2px solid #3a4454;
    border-radius: 8px;
    box-shadow: 0 8px 28px rgba(0,0,0,0.6);
    background: #111;
    cursor: crosshair;
    user-select: none;
    line-height: 0;
  }
  canvas#pixelCanvas {
    display: block;
    image-rendering: pixelated;
  }

  .canvas-hud {
    display: flex;
    justify-content: space-between;
    width: 100%;
    margin-top: 10px;
    font-size: 0.78rem;
    color: var(--text-muted);
    font-family: "Fira Code", monospace;
  }

  /* Tools & Toolbar */
  .toolbar {
    display: flex;
    gap: 6px;
    margin-top: 10px;
    width: 100%;
    justify-content: center;
  }
  .tool-btn {
    background: #252b37;
    border: 1px solid var(--border-color);
    color: var(--text-primary);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 5px;
    transition: all 0.15s ease;
  }
  .tool-btn:hover { background: #323a4a; border-color: #4f5b73; }
  .tool-btn.active {
    background: #153c48;
    border-color: var(--accent-cyan);
    color: var(--accent-cyan);
    box-shadow: 0 0 10px rgba(0,229,255,0.25);
  }

  /* Palette UI */
  .palette-section {
    background: var(--panel-sub);
    padding: 12px;
    border-radius: 10px;
    border: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .active-color-bar {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .active-color-preview {
    width: 32px;
    height: 32px;
    border-radius: 6px;
    border: 2px solid #fff;
    box-shadow: 0 0 8px rgba(0,0,0,0.5);
  }
  .active-color-info {
    font-size: 0.78rem;
    font-family: "Fira Code", monospace;
    color: var(--text-muted);
  }
  .active-color-info strong { color: #fff; font-size: 0.85rem; }

  /* Quick Swatches */
  .swatch-grid {
    display: grid;
    grid-template-columns: repeat(12, 1fr);
    gap: 4px;
  }
  .swatch-cell {
    aspect-ratio: 1;
    border-radius: 4px;
    cursor: pointer;
    border: 1px solid rgba(255,255,255,0.12);
    transition: transform 0.1s, border-color 0.1s;
  }
  .swatch-cell:hover { transform: scale(1.15); z-index: 2; border-color: #fff; }
  .swatch-cell.selected { border: 2px solid #fff; box-shadow: 0 0 8px rgba(255,255,255,0.6); }

  /* 256 Color Popover Toggle */
  .palette-extras {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.75rem;
  }
  .palette-toggle-btn {
    background: transparent;
    border: none;
    color: var(--accent-cyan);
    cursor: pointer;
    font-size: 0.75rem;
    font-weight: 600;
  }
  .palette-toggle-btn:hover { text-decoration: underline; }

  .full-palette-matrix {
    display: none;
    grid-template-columns: repeat(32, 1fr);
    gap: 2px;
    background: #111;
    padding: 6px;
    border-radius: 6px;
    border: 1px solid var(--border-color);
    max-height: 120px;
    overflow-y: auto;
  }
  .full-palette-matrix.show { display: grid; }
  .matrix-cell {
    aspect-ratio: 1;
    cursor: pointer;
  }
  .matrix-cell:hover { outline: 1px solid #fff; z-index: 2; }

  /* Presets Bar */
  .presets-bar {
    display: flex;
    gap: 6px;
    overflow-x: auto;
    padding-bottom: 4px;
  }
  .preset-chip {
    background: #232936;
    border: 1px solid var(--border-color);
    color: var(--text-muted);
    font-size: 0.74rem;
    padding: 5px 9px;
    border-radius: 6px;
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.15s ease;
  }
  .preset-chip:hover {
    background: #2d3546;
    color: #fff;
    border-color: var(--accent-purple);
  }

  /* Prompt Input */
  .prompt-group {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  textarea#promptInput {
    background: #13161c;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    color: #fff;
    padding: 10px 12px;
    font-size: 0.88rem;
    line-height: 1.45;
    resize: vertical;
    min-height: 72px;
    outline: none;
    transition: border-color 0.2s;
  }
  textarea#promptInput:focus {
    border-color: var(--accent-cyan);
    box-shadow: 0 0 10px rgba(0,229,255,0.2);
  }

  /* Quick Suggestion Chips */
  .suggestion-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
  }
  .chip {
    background: #1c222d;
    border: 1px solid var(--border-color);
    color: var(--text-muted);
    font-size: 0.72rem;
    padding: 3px 8px;
    border-radius: 12px;
    cursor: pointer;
  }
  .chip:hover { color: var(--accent-cyan); border-color: var(--accent-cyan); }

  /* Advanced Controls (Collapsible) */
  details.advanced-box {
    background: var(--panel-sub);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 0.8rem;
  }
  details.advanced-box summary {
    cursor: pointer;
    font-weight: 600;
    color: var(--text-muted);
    outline: none;
  }
  details.advanced-box summary:hover { color: #fff; }
  .sliders-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 10px;
  }
  .slider-item { display: flex; flex-direction: column; gap: 3px; font-size: 0.75rem; }
  .slider-item label { display: flex; justify-content: space-between; color: var(--text-muted); }
  .slider-item input[type="range"] { accent-color: var(--accent-cyan); cursor: pointer; }

  /* Generate Button */
  .btn-generate {
    background: linear-gradient(135deg, #00b4d8 0%, #0077b6 50%, #7209b7 100%);
    border: none;
    color: #fff;
    padding: 14px 20px;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    cursor: pointer;
    box-shadow: 0 6px 20px rgba(0, 180, 216, 0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    transition: all 0.2s ease;
  }
  .btn-generate:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 8px 26px rgba(0, 180, 216, 0.5);
    background: linear-gradient(135deg, #00e5ff 0%, #0096c7 50%, #9d4edd 100%);
  }
  .btn-generate:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    filter: grayscale(0.6);
  }

  /* Cascaded Progression Cards */
  .cascade-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .cascade-header h2 { font-size: 1.1rem; font-weight: 700; color: #fff; }
  .badge-time {
    font-size: 0.78rem; font-family: monospace; padding: 4px 10px; border-radius: 6px;
    background: #202735; color: var(--accent-cyan); border: 1px solid var(--border-color);
  }

  .stages-track {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
  }
  .stage-card {
    background: var(--panel-bg);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    position: relative;
    transition: transform 0.2s, border-color 0.2s;
  }
  .stage-card:hover {
    transform: translateY(-3px);
    border-color: #4a5568;
  }
  .stage-card.highlight {
    border-color: var(--accent-cyan);
    box-shadow: 0 0 20px rgba(0, 229, 255, 0.25);
  }
  .stage-tag {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
  }
  .stage-card.highlight .stage-tag { color: var(--accent-cyan); }

  .stage-frame {
    width: 100%;
    aspect-ratio: 1;
    background: #0d0f13;
    border-radius: 8px;
    border: 1px solid #282f3d;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    position: relative;
  }
  .stage-frame img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    image-rendering: pixelated;
    display: block;
  }
  .stage-frame .empty-hint {
    font-size: 0.75rem;
    color: #4b5568;
    text-align: center;
    padding: 10px;
  }
  .stage-meta {
    font-size: 0.74rem;
    color: var(--text-muted);
    font-family: monospace;
    display: flex;
    justify-content: space-between;
    width: 100%;
  }

  /* History Reel */
  .history-section {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-top: 10px;
  }
  .history-list {
    display: flex;
    gap: 12px;
    overflow-x: auto;
    padding-bottom: 8px;
  }
  .history-item {
    background: var(--panel-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    cursor: pointer;
    min-width: 120px;
    max-width: 130px;
    transition: all 0.15s ease;
  }
  .history-item:hover { border-color: var(--accent-cyan); transform: scale(1.03); }
  .history-item img {
    width: 100%;
    aspect-ratio: 1;
    border-radius: 4px;
    image-rendering: pixelated;
    object-fit: cover;
  }
  .history-item span {
    font-size: 0.7rem;
    color: var(--text-muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  /* Modal Zoom View */
  .modal-overlay {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.85);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 100;
    backdrop-filter: blur(4px);
  }
  .modal-overlay.open { display: flex; }
  .modal-content {
    background: var(--panel-bg);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    max-width: 90vw;
    max-height: 90vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 14px;
  }
  .modal-content img {
    max-width: 512px;
    max-height: 512px;
    image-rendering: pixelated;
    border-radius: 6px;
    box-shadow: 0 0 30px rgba(0,0,0,0.8);
  }
</style>
</head>
<body>

<header>
  <div class="title-group">
    <div class="logo-badge">🎨</div>
    <div class="title-text">
      <h1>Yumon Paint Studio</h1>
      <span>Manual 16x16 Coarse-to-Fine Neural Synthesis</span>
    </div>
  </div>
  <div class="header-actions">
    <div class="status-pill" id="statusPill">
      <div class="status-dot" id="statusDot"></div>
      <span id="statusText">Ready</span>
    </div>
  </div>
</header>

<main>
  <!-- Left Column: 16x16 Canvas & Inputs -->
  <section class="editor-pane">
    <div class="section-label">
      <span>16×16 Coarse Drawing Grid</span>
      <span id="coordHud">(0, 0)</span>
    </div>

    <!-- Interactive Canvas Box -->
    <div class="canvas-box">
      <div class="canvas-container">
        <canvas id="pixelCanvas" width="384" height="384"></canvas>
      </div>
      <div class="canvas-hud">
        <span>Click / Drag to Paint</span>
        <span>Right-Click: Erase</span>
      </div>
      <div class="toolbar">
        <button class="tool-btn active" id="toolPencil" title="Draw (P)">✏️ Pencil</button>
        <button class="tool-btn" id="toolEraser" title="Eraser (E)">🧹 Eraser</button>
        <button class="tool-btn" id="toolFill" title="Bucket Fill (B)">🪣 Fill</button>
        <button class="tool-btn" id="toolPicker" title="Eyedropper (I)">💧 Pick</button>
        <button class="tool-btn" id="toolUndo" title="Undo (Ctrl+Z)">↩️</button>
        <button class="tool-btn" id="toolRedo" title="Redo (Ctrl+Y)">↪️</button>
        <button class="tool-btn" id="toolGrid" title="Toggle Grid">🔲 Grid</button>
        <button class="tool-btn" id="toolClear" title="Clear Canvas">🗑️ Clear</button>
      </div>
    </div>

    <!-- Palette Section -->
    <div class="palette-section">
      <div class="active-color-bar">
        <div class="active-color-preview" id="activeColorPreview" style="background: #ffffff;"></div>
        <div class="active-color-info">
          <div>Selected: <strong id="activeHexLabel">#FFFFFF</strong></div>
          <div id="activePaletteLabel">Palette Token: 255 (RGB: 255, 255, 255)</div>
        </div>
        <input type="color" id="nativeColorPicker" value="#ffffff" style="margin-left: auto; width: 34px; height: 34px; border: none; background: transparent; cursor: pointer;"/>
      </div>

      <!-- Quick 24 Color Swatches -->
      <div class="swatch-grid" id="quickSwatches"></div>

      <div class="palette-extras">
        <span>Full 256 Yumon Palette (3-3-2 RGB)</span>
        <button class="palette-toggle-btn" id="togglePaletteBtn">Show All 256 Colors ▾</button>
      </div>
      <div class="full-palette-matrix" id="fullPaletteMatrix"></div>
    </div>

    <!-- Template Presets -->
    <div>
      <div class="section-label">Starter Templates</div>
      <div class="presets-bar" id="presetsBar"></div>
    </div>

    <!-- Prompt Box -->
    <div class="prompt-group">
      <div class="section-label">Art Prompt Guiding Yumon</div>
      <textarea id="promptInput" placeholder="Describe the scene or subject in detail...">a cute friendly baby dragon with glowing emerald eyes and golden horns, vibrant digital painting, enchanted forest</textarea>
      <div class="suggestion-chips">
        <span class="chip" data-append=", vibrant oil painting">🎨 Oil Painting</span>
        <span class="chip" data-append=", cyberpunk neon aesthetic">🌌 Cyberpunk</span>
        <span class="chip" data-append=", dramatic studio lighting">✨ Studio Lighting</span>
        <span class="chip" data-append=", detailed fantasy masterpiece">🗡️ Fantasy</span>
        <span class="chip" data-append=", cute tabletop pet mascot">🐾 Companion</span>
        <span class="chip" data-append=", rich atmospheric backlight">🌅 Backlight</span>
      </div>
    </div>

    <!-- Advanced Guidance Controls -->
    <details class="advanced-box">
      <summary>⚙️ Generation & Guidance Controls</summary>
      <div class="sliders-grid">
        <div class="slider-item">
          <label><span>Pass 3 Guidance (16x16 anchor)</span><strong id="valGuideP3">8</strong></label>
          <input type="range" id="guideP3" min="0" max="10" step="0.25" value="8"/>
        </div>
        <div class="slider-item">
          <label><span>Pass 4 Guidance (32x32 anchor)</span><strong id="valGuideP4">8</strong></label>
          <input type="range" id="guideP4" min="0" max="10" step="0.25" value="8"/>
        </div>
        <div class="slider-item">
          <label><span>Pass 3 Temp (Anatomy)</span><strong id="valTempP3">2.2</strong></label>
          <input type="range" id="tempP3" min="0.4" max="3.0" step="0.05" value="2.2"/>
        </div>
        <div class="slider-item">
          <label><span>Pass 4 Temp (Details)</span><strong id="valTempP4">0.5</strong></label>
          <input type="range" id="tempP4" min="0.4" max="3.0" step="0.05" value="0.5"/>
        </div>
      </div>
    </details>

    <!-- Generate Action -->
    <button class="btn-generate" id="btnGenerate">
      <span id="btnIcon">🎨</span>
      <span id="btnText">Synthesize Artwork with Yumon</span>
    </button>
  </section>

  <!-- Right Column: Cascaded Results -->
  <section class="results-pane">
    <div class="cascade-header">
      <h2>Cascaded Synthesis Progression</h2>
      <div class="badge-time" id="durationBadge">Awaiting generation...</div>
    </div>

    <div class="stages-track">
      <!-- Stage 1: User 16x16 -->
      <div class="stage-card">
        <div class="stage-tag">Manual Input (Pass 2)</div>
        <div class="stage-frame">
          <img id="imgStage1" src="" alt="" style="display:none;"/>
          <div class="empty-hint" id="hintStage1">Draw on the 16×16 canvas</div>
        </div>
        <div class="stage-meta">
          <span>16×16 Grid</span>
          <span>User Drawing</span>
        </div>
      </div>

      <!-- Stage 2: Pass 3 (32x32) -->
      <div class="stage-card">
        <div class="stage-tag">Pass 3: Anatomy & Features</div>
        <div class="stage-frame">
          <img id="imgStage2" src="" alt="" style="display:none;"/>
          <div class="empty-hint" id="hintStage2">Waiting for Pass 3...</div>
        </div>
        <div class="stage-meta">
          <span>32×32 Patches</span>
          <span>1,024 Tokens</span>
        </div>
      </div>

      <!-- Stage 3: Pass 4 (64x64) -->
      <div class="stage-card">
        <div class="stage-tag">Pass 4: Fine Color Details</div>
        <div class="stage-frame">
          <img id="imgStage3" src="" alt="" style="display:none;"/>
          <div class="empty-hint" id="hintStage3">Waiting for Pass 4...</div>
        </div>
        <div class="stage-meta">
          <span>64×64 Pixels</span>
          <span>4,096 Tokens</span>
        </div>
      </div>

      <!-- Stage 4: Neural Super-Resolution (256x256) -->
      <div class="stage-card highlight">
        <div class="stage-tag">Pass 5: Neural Upscaler</div>
        <div class="stage-frame">
          <img id="imgStage4" src="" alt="" style="display:none;"/>
          <div class="empty-hint" id="hintStage4">Waiting for Super-Res...</div>
        </div>
        <div class="stage-meta">
          <span>256×256 Super-Res</span>
          <span>Final TrueColor</span>
        </div>
      </div>
    </div>

    <!-- History Reel -->
    <div class="history-section">
      <div class="section-label">Session Artwork History</div>
      <div class="history-list" id="historyList">
        <span style="font-size:0.75rem; color:var(--text-muted); font-style:italic;">No past generations yet in this session.</span>
      </div>
    </div>
  </section>
</main>

<!-- Modal Zoom View -->
<div class="modal-overlay" id="modalOverlay">
  <div class="modal-content">
    <h3 id="modalTitle" style="color:var(--accent-cyan); font-size:1rem;">Artwork Zoom</h3>
    <img id="modalImg" src="" alt=""/>
    <div style="display:flex; gap:10px;">
      <a id="modalDownload" href="#" download="yumon_art.png" class="tool-btn" style="text-decoration:none;">💾 Download PNG</a>
      <button class="tool-btn" id="modalClose">Close</button>
    </div>
  </div>
</div>

<script>
// --- Palette Utilities (Exact match with Rust Yumon paint_action) ---
function paletteToRgb(idx) {
  const i = Math.max(0, Math.min(255, idx | 0));
  const r_bin = (i >> 5) & 0x07;
  const g_bin = (i >> 2) & 0x07;
  const b_bin = i & 0x03;
  const r = Math.floor((r_bin * 255) / 7);
  const g = Math.floor((g_bin * 255) / 7);
  const b = Math.floor((b_bin * 255) / 3);
  return [r, g, b];
}

function rgbToPalette(r, g, b) {
  const r_bin = Math.min(7, Math.floor((r * 7 + 127) / 255));
  const g_bin = Math.min(7, Math.floor((g * 7 + 127) / 255));
  const b_bin = Math.min(3, Math.floor((b * 3 + 127) / 255));
  return (r_bin * 32 + g_bin * 4 + b_bin);
}

function rgbToHex(r, g, b) {
  return "#" + [r, g, b].map(x => x.toString(16).padStart(2, "0")).join("").toUpperCase();
}

function hexToRgb(hex) {
  hex = hex.replace("#", "");
  if (hex.length === 3) hex = hex.split("").map(c => c + c).join("");
  const num = parseInt(hex, 16);
  return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
}

// 16x16 Grid State
const GRID_SIZE = 16;
let grid = new Array(GRID_SIZE * GRID_SIZE).fill(255); // Default white/canvas background
let activePaletteIdx = 0; // Default black brush
let currentTool = "pencil"; // pencil, eraser, fill, picker
let showGrid = true;
let isMouseDown = false;
let undoStack = [];
let redoStack = [];

const canvas = document.getElementById("pixelCanvas");
const ctx = canvas.getContext("2d");
const CELL_SIZE = canvas.width / GRID_SIZE; // 24px per cell

// Setup Quick Swatches
const quickPaletteIndices = [
  0,   // Black
  255, // White
  73,  // Dark Gray
  146, // Mid Gray
  219, // Light Gray
  224, // Bright Red
  160, // Crimson
  240, // Orange
  252, // Amber / Yellow
  28,  // Lime Green
  16,  // Emerald
  8,   // Forest Green
  31,  // Cyan
  23,  // Sky Blue
  3,   // Deep Blue
  99,  // Purple
  227, // Magenta
  231, // Pink
  246, // Tan / Flesh
  104, // Warm Brown
  68,  // Earth Umber
  195, // Indigo
  59,  // Teal
  128  // Maroon
];

function initSwatches() {
  const container = document.getElementById("quickSwatches");
  container.innerHTML = "";
  quickPaletteIndices.forEach(idx => {
    const [r, g, b] = paletteToRgb(idx);
    const cell = document.createElement("div");
    cell.className = "swatch-cell" + (idx === activePaletteIdx ? " selected" : "");
    cell.style.backgroundColor = rgbToHex(r, g, b);
    cell.title = `Color #${idx} (${r}, ${g}, ${b})`;
    cell.addEventListener("click", () => selectPaletteIndex(idx));
    container.appendChild(cell);
  });

  // Full 256 matrix
  const matrix = document.getElementById("fullPaletteMatrix");
  matrix.innerHTML = "";
  for (let i = 0; i < 256; i++) {
    const [r, g, b] = paletteToRgb(i);
    const cell = document.createElement("div");
    cell.className = "matrix-cell";
    cell.style.backgroundColor = rgbToHex(r, g, b);
    cell.title = `Palette #${i} (RGB: ${r}, ${g}, ${b})`;
    cell.addEventListener("click", () => selectPaletteIndex(i));
    matrix.appendChild(cell);
  }
}

function selectPaletteIndex(idx) {
  activePaletteIdx = Math.max(0, Math.min(255, idx));
  const [r, g, b] = paletteToRgb(activePaletteIdx);
  const hex = rgbToHex(r, g, b);
  document.getElementById("activeColorPreview").style.backgroundColor = hex;
  document.getElementById("activeHexLabel").innerText = hex;
  document.getElementById("activePaletteLabel").innerText = `Palette Token: ${activePaletteIdx} (RGB: ${r}, ${g}, ${b})`;
  document.getElementById("nativeColorPicker").value = hex;

  document.querySelectorAll(".swatch-cell").forEach(cell => {
    cell.classList.toggle("selected", cell.style.backgroundColor.toLowerCase() === hex.toLowerCase());
  });
}

// Preset Shapes (16x16)
const PRESETS = {
  "🐉 Dragon Head": {
    prompt: "a cute friendly baby dragon with glowing emerald eyes and golden horns, vibrant digital painting, enchanted forest",
    data: [
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,252,0,0,0,0,0,252,0,0,0,0,0,
      0,0,0,252,240,0,0,0,0,240,252,0,0,0,0,0,
      0,0,0,0,240,16,16,16,16,240,0,0,0,0,0,0,
      0,0,0,16,16,28,28,28,28,16,16,0,0,0,0,0,
      0,0,16,28,28,252,28,28,252,28,28,16,0,0,0,0,
      0,0,16,28,28,0,28,28,0,28,28,16,0,0,0,0,
      0,16,28,28,28,28,28,28,28,28,28,28,16,0,0,0,
      0,16,28,28,28,28,240,240,28,28,28,28,16,0,0,0,
      0,0,16,28,28,28,28,28,28,28,28,16,0,0,0,0,
      0,0,0,16,28,28,28,28,28,28,16,0,0,0,0,0,
      0,0,0,0,16,16,224,224,16,16,0,0,0,0,0,0,
      0,0,0,0,0,0,16,16,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    ]
  },
  "🗡️ Sword": {
    prompt: "a legendary glowing crystalline broadsword with ornate gold guard and azure blade, fantasy pixel art masterpiece",
    data: [
      0,0,0,0,0,0,0,0,0,0,0,0,0,31,255,255,
      0,0,0,0,0,0,0,0,0,0,0,0,31,255,31,0,
      0,0,0,0,0,0,0,0,0,0,0,31,255,31,0,0,
      0,0,0,0,0,0,0,0,0,0,31,255,31,0,0,0,
      0,0,0,0,0,0,0,0,0,31,255,31,0,0,0,0,
      0,0,0,0,0,0,0,0,31,255,31,0,0,0,0,0,
      0,0,0,0,0,0,0,31,255,31,0,0,0,0,0,0,
      0,0,0,0,0,0,31,255,31,0,0,0,0,0,0,0,
      0,0,0,0,252,252,255,31,0,0,0,0,0,0,0,0,
      0,0,0,252,252,252,252,252,0,0,0,0,0,0,0,0,
      0,0,252,0,0,240,252,0,0,0,0,0,0,0,0,0,
      0,0,0,0,240,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,252,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    ]
  },
  "🧪 Potion": {
    prompt: "a magical glass flask filled with swirling glowing violet elixir, bubbling fantasy potion asset, cinematic lighting",
    data: [
      0,0,0,0,0,0,104,104,104,104,0,0,0,0,0,0,
      0,0,0,0,0,0,219,219,219,219,0,0,0,0,0,0,
      0,0,0,0,0,0,0,219,219,0,0,0,0,0,0,0,
      0,0,0,0,0,219,219,219,219,219,219,0,0,0,0,0,
      0,0,0,0,219,99,99,99,99,99,219,0,0,0,0,0,
      0,0,0,219,99,227,99,99,227,99,99,219,0,0,0,0,
      0,0,0,219,227,227,227,227,227,227,99,219,0,0,0,0,
      0,0,219,99,227,255,227,227,227,227,99,219,0,0,0,0,
      0,0,219,99,227,227,227,227,227,227,99,219,0,0,0,0,
      0,0,219,99,99,227,227,227,227,99,99,219,0,0,0,0,
      0,0,219,99,99,99,99,99,99,99,99,219,0,0,0,0,
      0,0,0,219,219,219,219,219,219,219,219,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    ]
  },
  "❤️ Heart": {
    prompt: "a glowing neon pixel heart pulsing with warm radiant crimson and gold light, retro game item",
    data: [
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,224,224,0,0,0,0,0,224,224,0,0,0,0,0,
      0,224,231,231,224,0,0,0,224,231,231,224,0,0,0,0,
      224,231,255,231,224,224,0,224,224,231,231,231,224,0,0,0,
      224,231,231,224,224,224,224,224,224,224,231,231,224,0,0,0,
      224,224,224,224,224,224,224,224,224,224,224,224,224,0,0,0,
      0,224,224,224,224,224,224,224,224,224,224,224,0,0,0,0,
      0,0,224,224,224,224,224,224,224,224,224,0,0,0,0,0,
      0,0,0,224,224,224,224,224,224,224,0,0,0,0,0,0,
      0,0,0,0,224,224,224,224,224,0,0,0,0,0,0,0,
      0,0,0,0,0,224,224,224,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,224,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    ]
  },
  "🌳 Tree": {
    prompt: "a mystical ancient bonsai willow tree with luminous green foliage, enchanted twilight forest, ethereal lighting",
    data: [
      0,0,0,0,0,0,28,28,28,0,0,0,0,0,0,0,
      0,0,0,0,28,28,28,28,28,28,0,0,0,0,0,0,
      0,0,0,28,28,16,16,28,28,28,28,0,0,0,0,0,
      0,0,28,28,16,16,16,16,28,28,28,28,0,0,0,0,
      0,28,28,16,16,16,16,16,16,28,28,28,28,0,0,0,
      0,28,28,16,16,16,16,16,16,16,28,28,28,0,0,0,
      0,0,28,28,16,16,16,16,16,16,28,28,0,0,0,0,
      0,0,0,28,28,16,16,16,16,28,28,0,0,0,0,0,
      0,0,0,0,0,0,104,104,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,104,104,0,0,0,0,0,0,0,0,
      0,0,0,0,0,104,104,104,0,0,0,0,0,0,0,0,
      0,0,0,0,0,104,104,104,104,0,0,0,0,0,0,0,
      0,0,0,0,104,104,104,104,104,0,0,0,0,0,0,0,
      0,0,0,104,104,104,104,104,104,104,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    ]
  }
};

function initPresets() {
  const container = document.getElementById("presetsBar");
  container.innerHTML = "";
  Object.keys(PRESETS).forEach(name => {
    const chip = document.createElement("button");
    chip.className = "preset-chip";
    chip.innerText = name;
    chip.addEventListener("click", () => {
      saveStateForUndo();
      const p = PRESETS[name];
      grid = [...p.data];
      document.getElementById("promptInput").value = p.prompt;
      renderCanvas();
      updateStage1Preview();
    });
    container.appendChild(chip);
  });
}

// Canvas Rendering
function renderCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let y = 0; y < GRID_SIZE; y++) {
    for (let x = 0; x < GRID_SIZE; x++) {
      const idx = grid[y * GRID_SIZE + x];
      const [r, g, b] = paletteToRgb(idx);
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);

      if (showGrid) {
        ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
        ctx.lineWidth = 1;
        ctx.strokeRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
      }
    }
  }
}

function getGridCoords(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = Math.floor(((e.clientX - rect.left) * scaleX) / CELL_SIZE);
  const y = Math.floor(((e.clientY - rect.top) * scaleY) / CELL_SIZE);
  return [Math.max(0, Math.min(15, x)), Math.max(0, Math.min(15, y))];
}

function saveStateForUndo() {
  undoStack.push([...grid]);
  if (undoStack.length > 50) undoStack.shift();
  redoStack = [];
}

function floodFill(startX, startY, targetVal, replaceVal) {
  if (targetVal === replaceVal) return;
  const queue = [[startX, startY]];
  const visited = new Uint8Array(GRID_SIZE * GRID_SIZE);

  while (queue.length > 0) {
    const [x, y] = queue.pop();
    const idx = y * GRID_SIZE + x;
    if (visited[idx]) continue;
    visited[idx] = 1;

    if (grid[idx] === targetVal) {
      grid[idx] = replaceVal;
      if (x > 0) queue.push([x - 1, y]);
      if (x < GRID_SIZE - 1) queue.push([x + 1, y]);
      if (y > 0) queue.push([x, y - 1]);
      if (y < GRID_SIZE - 1) queue.push([x, y + 1]);
    }
  }
}

function applyToolAt(x, y, isRightClick) {
  const idx = y * GRID_SIZE + x;
  if (currentTool === "picker" || isRightClick && currentTool === "picker") {
    selectPaletteIndex(grid[idx]);
    return;
  }
  if (currentTool === "fill") {
    saveStateForUndo();
    const fillVal = isRightClick ? 255 : activePaletteIdx;
    floodFill(x, y, grid[idx], fillVal);
    renderCanvas();
    updateStage1Preview();
    return;
  }
  const colorToSet = (currentTool === "eraser" || isRightClick) ? 255 : activePaletteIdx;
  if (grid[idx] !== colorToSet) {
    grid[idx] = colorToSet;
    renderCanvas();
    updateStage1Preview();
  }
}

// Canvas Mouse / Touch Handlers
canvas.addEventListener("contextmenu", e => e.preventDefault());

canvas.addEventListener("mousedown", e => {
  isMouseDown = true;
  saveStateForUndo();
  const [x, y] = getGridCoords(e);
  applyToolAt(x, y, e.button === 2);
});

canvas.addEventListener("mousemove", e => {
  const [x, y] = getGridCoords(e);
  document.getElementById("coordHud").innerText = `(${x}, ${y}) | Token: ${grid[y * GRID_SIZE + x]}`;
  if (isMouseDown) {
    applyToolAt(x, y, e.buttons === 2);
  }
});

window.addEventListener("mouseup", () => {
  isMouseDown = false;
});

// Update the 16x16 Stage 1 preview card image
function updateStage1Preview() {
  const offscreen = document.createElement("canvas");
  offscreen.width = 16;
  offscreen.height = 16;
  const offCtx = offscreen.getContext("2d");
  const imgData = offCtx.createImageData(16, 16);
  for (let i = 0; i < 256; i++) {
    const [r, g, b] = paletteToRgb(grid[i]);
    imgData.data[i * 4] = r;
    imgData.data[i * 4 + 1] = g;
    imgData.data[i * 4 + 2] = b;
    imgData.data[i * 4 + 3] = 255;
  }
  offCtx.putImageData(imgData, 0, 0);
  const dataUrl = offscreen.toDataURL("image/png");

  const img1 = document.getElementById("imgStage1");
  img1.src = dataUrl;
  img1.style.display = "block";
  document.getElementById("hintStage1").style.display = "none";
}

// Tool Buttons
const toolButtons = {
  pencil: document.getElementById("toolPencil"),
  eraser: document.getElementById("toolEraser"),
  fill: document.getElementById("toolFill"),
  picker: document.getElementById("toolPicker"),
};

function setTool(toolName) {
  currentTool = toolName;
  Object.keys(toolButtons).forEach(name => {
    toolButtons[name].classList.toggle("active", name === toolName);
  });
}

document.getElementById("toolPencil").addEventListener("click", () => setTool("pencil"));
document.getElementById("toolEraser").addEventListener("click", () => setTool("eraser"));
document.getElementById("toolFill").addEventListener("click", () => setTool("fill"));
document.getElementById("toolPicker").addEventListener("click", () => setTool("picker"));

document.getElementById("toolGrid").addEventListener("click", () => {
  showGrid = !showGrid;
  document.getElementById("toolGrid").classList.toggle("active", showGrid);
  renderCanvas();
});

document.getElementById("toolClear").addEventListener("click", () => {
  if (confirm("Clear the drawing grid?")) {
    saveStateForUndo();
    grid.fill(255);
    renderCanvas();
    updateStage1Preview();
  }
});

document.getElementById("toolUndo").addEventListener("click", () => {
  if (undoStack.length > 0) {
    redoStack.push([...grid]);
    grid = undoStack.pop();
    renderCanvas();
    updateStage1Preview();
  }
});

document.getElementById("toolRedo").addEventListener("click", () => {
  if (redoStack.length > 0) {
    undoStack.push([...grid]);
    grid = redoStack.pop();
    renderCanvas();
    updateStage1Preview();
  }
});

// Color picker inputs
document.getElementById("nativeColorPicker").addEventListener("input", e => {
  const [r, g, b] = hexToRgb(e.target.value);
  const pal = rgbToPalette(r, g, b);
  selectPaletteIndex(pal);
});

document.getElementById("togglePaletteBtn").addEventListener("click", () => {
  const matrix = document.getElementById("fullPaletteMatrix");
  matrix.classList.toggle("show");
  document.getElementById("togglePaletteBtn").innerText = matrix.classList.contains("show")
    ? "Hide 256 Colors ▴"
    : "Show All 256 Colors ▾";
});

// Prompt Chips
document.querySelectorAll(".chip").forEach(chip => {
  chip.addEventListener("click", () => {
    const input = document.getElementById("promptInput");
    input.value = input.value.trim() + chip.dataset.append;
  });
});

// Sliders live label update
const sliderDefs = [
  { id: "guideP3", label: "valGuideP3" },
  { id: "guideP4", label: "valGuideP4" },
  { id: "tempP3", label: "valTempP3" },
  { id: "tempP4", label: "valTempP4" },
];
sliderDefs.forEach(s => {
  const el = document.getElementById(s.id);
  const lbl = document.getElementById(s.label);
  el.addEventListener("input", () => { lbl.innerText = el.value; });
});

// Modal Zoom
function openModal(imgSrc, title) {
  document.getElementById("modalImg").src = imgSrc;
  document.getElementById("modalTitle").innerText = title;
  document.getElementById("modalDownload").href = imgSrc;
  document.getElementById("modalOverlay").classList.add("open");
}
document.getElementById("modalClose").addEventListener("click", () => {
  document.getElementById("modalOverlay").classList.remove("open");
});
document.getElementById("modalOverlay").addEventListener("click", e => {
  if (e.target.id === "modalOverlay") {
    document.getElementById("modalOverlay").classList.remove("open");
  }
});

["imgStage1", "imgStage2", "imgStage3", "imgStage4"].forEach(id => {
  const img = document.getElementById(id);
  img.addEventListener("click", () => {
    if (img.src) {
      const parentTag = img.closest(".stage-card").querySelector(".stage-tag").innerText;
      openModal(img.src, parentTag);
    }
  });
});

// History Reel
const sessionHistory = [];
function addHistoryItem(runId, prompt, upscaledUrl, p2Url, p3Url, p4Url, duration) {
  sessionHistory.unshift({ runId, prompt, upscaledUrl, p2Url, p3Url, p4Url, duration });
  const container = document.getElementById("historyList");
  container.innerHTML = "";
  sessionHistory.forEach(item => {
    const card = document.createElement("div");
    card.className = "history-item";
    card.innerHTML = `
      <img src="${item.upscaledUrl}" alt=""/>
      <span title="${item.prompt}">${item.prompt}</span>
      <span style="color:var(--accent-cyan);">${item.duration}s</span>
    `;
    card.addEventListener("click", () => {
      document.getElementById("imgStage1").src = item.p2Url;
      document.getElementById("imgStage1").style.display = "block";
      document.getElementById("hintStage1").style.display = "none";

      document.getElementById("imgStage2").src = item.p3Url;
      document.getElementById("imgStage2").style.display = "block";
      document.getElementById("hintStage2").style.display = "none";

      document.getElementById("imgStage3").src = item.p4Url;
      document.getElementById("imgStage3").style.display = "block";
      document.getElementById("hintStage3").style.display = "none";

      document.getElementById("imgStage4").src = item.upscaledUrl;
      document.getElementById("imgStage4").style.display = "block";
      document.getElementById("hintStage4").style.display = "none";

      document.getElementById("durationBadge").innerText = `Restored (${item.duration}s)`;
    });
    container.appendChild(card);
  });
}

// Generate Action
const btnGenerate = document.getElementById("btnGenerate");
btnGenerate.addEventListener("click", async () => {
  const prompt = document.getElementById("promptInput").value.trim();
  if (!prompt) {
    alert("Please enter an art prompt to guide Yumon.");
    return;
  }

  btnGenerate.disabled = true;
  document.getElementById("btnIcon").innerText = "⏳";
  document.getElementById("btnText").innerText = "Cascading Passes 3, 4 & Super-Resolution...";
  document.getElementById("statusText").innerText = "Generating...";
  document.getElementById("statusDot").classList.add("busy");
  document.getElementById("durationBadge").innerText = "Processing on GPU...";

  const payload = {
    prompt: prompt,
    pixels: grid,
    shape_guidance: parseFloat(document.getElementById("guideP3").value),
    shape_guidance_pass4: parseFloat(document.getElementById("guideP4").value),
    temperature_pass3: parseFloat(document.getElementById("tempP3").value),
    temperature_pass4: parseFloat(document.getElementById("tempP4").value),
  };

  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    if (!res.ok || data.error) {
      alert("Generation error: " + (data.error || "Unknown error"));
      document.getElementById("durationBadge").innerText = "Failed";
      return;
    }

    // Populate stage images
    if (data.images.pass2) {
      document.getElementById("imgStage1").src = data.images.pass2;
      document.getElementById("imgStage1").style.display = "block";
      document.getElementById("hintStage1").style.display = "none";
    }
    if (data.images.pass3) {
      document.getElementById("imgStage2").src = data.images.pass3;
      document.getElementById("imgStage2").style.display = "block";
      document.getElementById("hintStage2").style.display = "none";
    }
    if (data.images.pass4) {
      document.getElementById("imgStage3").src = data.images.pass4;
      document.getElementById("imgStage3").style.display = "block";
      document.getElementById("hintStage3").style.display = "none";
    }
    if (data.images.upscaled) {
      document.getElementById("imgStage4").src = data.images.upscaled;
      document.getElementById("imgStage4").style.display = "block";
      document.getElementById("hintStage4").style.display = "none";
    }

    document.getElementById("durationBadge").innerText = `Completed in ${data.duration_seconds}s`;

    addHistoryItem(
      data.run_id,
      prompt,
      data.images.upscaled,
      data.images.pass2,
      data.images.pass3,
      data.images.pass4,
      data.duration_seconds
    );

  } catch (err) {
    alert("Connection error: " + err.message);
    document.getElementById("durationBadge").innerText = "Error";
  } finally {
    btnGenerate.disabled = false;
    document.getElementById("btnIcon").innerText = "🎨";
    document.getElementById("btnText").innerText = "Synthesize Artwork with Yumon";
    document.getElementById("statusText").innerText = "Ready";
    document.getElementById("statusDot").classList.remove("busy");
  }
});

// Keyboard shortcuts
window.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return;
  if (e.key === "p" || e.key === "P") setTool("pencil");
  if (e.key === "e" || e.key === "E") setTool("eraser");
  if (e.key === "b" || e.key === "B") setTool("fill");
  if (e.key === "i" || e.key === "I") setTool("picker");
  if (e.ctrlKey && (e.key === "z" || e.key === "Z")) {
    document.getElementById("toolUndo").click();
  }
  if (e.ctrlKey && (e.key === "y" || e.key === "Y")) {
    document.getElementById("toolRedo").click();
  }
  if (e.ctrlKey && e.key === "Enter") {
    btnGenerate.click();
  }
});

// Initialization
initSwatches();
initPresets();
selectPaletteIndex(0); // Black brush to start
renderCanvas();
updateStage1Preview();
</script>

</body>
</html>
"""

def main():
    port = 8088
    # Try finding an open port starting from 8088
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                break
            port += 1

    server = ThreadingHTTPServer(("localhost", port), StudioServer)
    url = f"http://localhost:{port}"

    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║   🎨 Yumon Paint Studio: 16x16 Interactive Coarse-to-Fine Generator  ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print(f"  • Web UI running at:  {url}")
    print(f"  • Project Root:       {PROJECT_ROOT}")
    print(f"  • Binary target:      {find_paint_binary()}")
    print("  • Press Ctrl+C in terminal to stop server.\n")

    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"  (Could not auto-launch browser: {e})")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStudio server shutting down. Have a great day!")
    finally:
        server.server_close()

if __name__ == "__main__":
    main()
