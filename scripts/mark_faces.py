#!/usr/bin/env python3
"""
Interactive ROI Face & Eye Annotator for DiffusionDB in Yumon Pet.

Usage:
    python scripts/mark_faces.py
    (or uv run python scripts/mark_faces.py)

Opens a local web tool in your default browser at http://localhost:8080:
- View 64x64 DiffusionDB images (with optional 256/512 high-res reference).
- Click Eye 1 (Left Eye), Eye 2 (Right Eye), and drag/click Face Box.
- Press [S] to mark 'No Face' / Skip.
- Press [Space] or [Right Arrow] to advance.
- Auto-saves on every action to `training-cache/diffusion_face_rois.json`.
"""

import os
import sys
import glob
import json
import socket
import webbrowser
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, unquote

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_64_DIR = PROJECT_ROOT / "data" / "DiffusionDb" / "images-64"
IMAGES_256_DIR = Path("D:/DiffusionDb/images-256")
IMAGES_ORIG_DIR = Path("D:/DiffusionDb/images")
OUTPUT_JSON_PATH = PROJECT_ROOT / "training-cache" / "diffusion_face_rois.json"

def load_diffusiondb_index():
    manifests = sorted(glob.glob(str(IMAGES_64_DIR / "part-*.json")))
    if not manifests:
        print(f"Error: No part-*.json manifests found in {IMAGES_64_DIR}", file=sys.stderr)
        return []
    
    items = []
    for m in manifests:
        try:
            with open(m, "r", encoding="utf-8") as f:
                data = json.load(f)
                for filename, meta in data.items():
                    items.append({
                        "file": filename,
                        "prompt": meta.get("p", ""),
                    })
        except Exception as e:
            print(f"Warning: Failed reading {m}: {e}", file=sys.stderr)
            
    print(f"Loaded {len(items)} DiffusionDB image records from {len(manifests)} manifests.")
    return items

def load_existing_annotations():
    if OUTPUT_JSON_PATH.exists():
        try:
            with open(OUTPUT_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("annotations", {})
        except Exception as e:
            print(f"Warning: Could not load existing annotations: {e}", file=sys.stderr)
    return {}

def save_annotations(annotations):
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "canvas_size": 64,
        "total_annotated": len(annotations),
        "annotations": annotations,
    }
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

class AnnotationServer(SimpleHTTPRequestHandler):
    images_index = []
    annotations_db = {}

    def log_message(self, format, *args):
        # Silence default request logging to keep console clean
        return

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
            return

        elif path == "/api/init":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {
                "total_images": len(self.images_index),
                "images": self.images_index,
                "annotations": self.annotations_db,
                "output_path": str(OUTPUT_JSON_PATH),
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        elif path.startswith("/image/64/"):
            filename = unquote(path[len("/image/64/"):])
            filepath = IMAGES_64_DIR / filename
            if filepath.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                with open(filepath, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, f"64x64 Image not found: {filename}")
                return

        elif path.startswith("/image/hires/"):
            filename = unquote(path[len("/image/hires/"):])
            # Check 256 then original then fallback to 64
            cand1 = IMAGES_256_DIR / filename
            cand2 = IMAGES_ORIG_DIR / filename
            cand3 = IMAGES_64_DIR / filename
            target = cand1 if cand1.is_file() else (cand2 if cand2.is_file() else cand3)
            if target.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                with open(target, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, f"Image not found: {filename}")
                return

        elif path == "/api/export":
            if OUTPUT_JSON_PATH.exists():
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Disposition", 'attachment; filename="diffusion_face_rois.json"')
                self.end_headers()
                with open(OUTPUT_JSON_PATH, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, "No annotations file yet")
                return

        self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/save":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body)
                filename = data.get("file")
                if filename:
                    self.annotations_db[filename] = {
                        "eyes": data.get("eyes", []),
                        "face": data.get("face"),
                        "no_face": data.get("no_face", False),
                    }
                    save_annotations(self.annotations_db)
                    
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "ok",
                    "total_annotated": len(self.annotations_db),
                }).encode("utf-8"))
            except Exception as e:
                self.send_error(400, f"Error saving annotation: {e}")
            return

        self.send_error(404, "Not Found")

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>DiffusionDB Face & Eye ROI Marker - Yumon Pet</title>
<style>
  :root {
    --bg-dark: #121418;
    --panel-bg: #1c1f26;
    --border-color: #2e3440;
    --accent-cyan: #00e5ff;
    --accent-magenta: #ff007f;
    --accent-gold: #ffb703;
    --accent-green: #06d6a0;
    --text-primary: #eceff4;
    --text-muted: #8892b0;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  body { background: var(--bg-dark); color: var(--text-primary); display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

  header {
    background: var(--panel-bg);
    border-bottom: 1px solid var(--border-color);
    padding: 10px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .title { font-size: 1.15rem; font-weight: 700; color: var(--accent-cyan); display: flex; align-items: center; gap: 8px; }
  .stats-bar { display: flex; gap: 15px; font-size: 0.9rem; font-weight: 500; }
  .badge { background: #262c38; padding: 4px 10px; border-radius: 6px; border: 1px solid var(--border-color); }
  .badge span { font-weight: bold; }
  .badge.marked span { color: var(--accent-cyan); }
  .badge.skipped span { color: var(--accent-gold); }

  main {
    flex: 1;
    display: flex;
    overflow: hidden;
  }

  /* Left Canvas Area */
  .canvas-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 20px;
    background: radial-gradient(circle at center, #1b2028 0%, #101216 100%);
    position: relative;
  }

  .canvas-wrapper {
    position: relative;
    border: 3px solid #3b4252;
    border-radius: 8px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.6);
    background: #000;
    cursor: crosshair;
    user-select: none;
  }

  canvas {
    display: block;
    image-rendering: pixelated;
  }

  .coords-hud {
    margin-top: 12px;
    font-size: 0.85rem;
    color: var(--text-muted);
    display: flex;
    gap: 15px;
    background: rgba(28, 31, 38, 0.8);
    padding: 6px 14px;
    border-radius: 20px;
    border: 1px solid var(--border-color);
  }

  /* Right Control Sidebar */
  .control-sidebar {
    width: 380px;
    background: var(--panel-bg);
    border-left: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    padding: 20px;
    overflow-y: auto;
  }

  .section { margin-bottom: 20px; }
  .section-title { font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); letter-spacing: 1px; margin-bottom: 8px; font-weight: 700; }

  .prompt-box {
    background: #15181e;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 12px;
    font-size: 0.88rem;
    line-height: 1.45;
    color: #e2e8f0;
    min-height: 60px;
  }
  .prompt-filename {
    font-size: 0.75rem;
    color: var(--accent-cyan);
    margin-bottom: 6px;
    font-family: monospace;
    word-break: break-all;
  }

  /* Mode Selectors */
  .mode-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 8px;
    margin-bottom: 12px;
  }
  .mode-btn {
    background: #252a35;
    border: 1px solid var(--border-color);
    color: var(--text-muted);
    border-radius: 6px;
    padding: 10px 4px;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    text-align: center;
    transition: all 0.15s ease;
  }
  .mode-btn.active {
    background: #1e3a47;
    border-color: var(--accent-cyan);
    color: var(--accent-cyan);
    box-shadow: 0 0 10px rgba(0,229,255,0.2);
  }
  .mode-btn[data-mode="eye2"].active {
    background: #421e33;
    border-color: var(--accent-magenta);
    color: var(--accent-magenta);
    box-shadow: 0 0 10px rgba(255,0,127,0.2);
  }
  .mode-btn[data-mode="face"].active {
    background: #42381e;
    border-color: var(--accent-gold);
    color: var(--accent-gold);
    box-shadow: 0 0 10px rgba(255,183,3,0.2);
  }

  .status-step {
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 0.86rem;
    font-weight: 500;
    background: #232834;
    border-left: 4px solid var(--accent-cyan);
    margin-bottom: 15px;
  }

  .action-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 15px;
  }
  button.action-btn {
    border: none;
    border-radius: 6px;
    padding: 12px;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.15s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
  }
  button.action-btn:hover { opacity: 0.9; }
  .btn-skip { background: #3b4252; color: #eceff4; }
  .btn-next { background: var(--accent-green); color: #0b2219; }
  .btn-clear { background: #bf616a; color: #fff; }
  .btn-hires { background: #4c566a; color: #eceff4; }

  .kbd-help {
    font-size: 0.78rem;
    color: var(--text-muted);
    background: #171a21;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 12px;
    line-height: 1.6;
  }
  kbd {
    background: #2e3440;
    color: #eceff4;
    padding: 2px 5px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 0.75rem;
    border: 1px solid #434c5e;
  }

  .nav-row {
    display: flex;
    gap: 10px;
    margin-top: 10px;
  }
  .nav-btn {
    flex: 1;
    background: #2e3440;
    color: var(--text-primary);
    border: 1px solid var(--border-color);
    padding: 8px;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 600;
  }

  .preview-hires-box {
    margin-top: 15px;
    text-align: center;
    display: none;
  }
  .preview-hires-box img {
    max-width: 160px;
    border-radius: 6px;
    border: 1px solid var(--border-color);
  }
</style>
</head>
<body>

<header>
  <div class="title">
    <span>🎨</span> DiffusionDB Face & Eye ROI Marker
  </div>
  <div class="stats-bar">
    <div class="badge">Image: <span id="idx-counter">0 / 0</span></div>
    <div class="badge marked">Marked: <span id="marked-counter">0</span></div>
    <div class="badge skipped">No Face: <span id="skipped-counter">0</span></div>
  </div>
</header>

<main>
  <!-- Left Canvas Area -->
  <div class="canvas-panel">
    <div class="canvas-wrapper">
      <canvas id="view-canvas" width="512" height="512"></canvas>
    </div>
    <div class="coords-hud">
      <div>Grid: <span id="hud-coords">(0, 0)</span></div>
      <div>Scale: <span>8x (64x64 Canvas)</span></div>
      <div>Step: <span id="hud-step">Eye 1</span></div>
    </div>
  </div>

  <!-- Right Controls -->
  <div class="control-sidebar">
    <div class="section">
      <div class="section-title">Image Prompt</div>
      <div class="prompt-box">
        <div class="prompt-filename" id="prompt-filename">Loading...</div>
        <div id="prompt-text">...</div>
      </div>
    </div>

    <div class="section">
      <div class="section-title">Marking Mode</div>
      <div class="mode-row">
        <div class="mode-btn active" data-mode="eye1" onclick="setMode('eye1')">[1] Eye 1</div>
        <div class="mode-btn" data-mode="eye2" onclick="setMode('eye2')">[2] Eye 2</div>
        <div class="mode-btn" data-mode="face" onclick="setMode('face')">[3] Face Box</div>
      </div>
      <div class="status-step" id="step-instruction">
        👉 Click <strong>Eye 1 (Left)</strong>, or press <kbd>S</kbd> if no face
      </div>
    </div>

    <div class="section">
      <div class="section-title">Actions</div>
      <div class="action-grid">
        <button class="action-btn btn-skip" onclick="skipNoFace()">[S] No Face</button>
        <button class="action-btn btn-next" onclick="saveAndNext()">[Space] Next ➔</button>
        <button class="action-btn btn-clear" onclick="clearCurrent()">[R] Reset</button>
        <button class="action-btn btn-hires" onclick="toggleHires()">[H] Ref View</button>
      </div>

      <div class="nav-row">
        <button class="nav-btn" onclick="prevImage()">⬅ Prev</button>
        <button class="nav-btn" onclick="nextImage()">Next ➡</button>
      </div>
    </div>

    <div class="section preview-hires-box" id="hires-box">
      <div class="section-title">High-Res Reference</div>
      <img id="hires-img" src="" alt="High-Res Reference"/>
    </div>

    <div class="section" style="margin-top: auto;">
      <div class="kbd-help">
        <strong>⚡ Shortcuts:</strong><br/>
        • <kbd>Click</kbd> places Eye 1, Eye 2, then drag/click Face<br/>
        • <kbd>S</kbd> = Mark 'No Face' & Advance<br/>
        • <kbd>Space</kbd> or <kbd>→</kbd> = Save & Next Image<br/>
        • <kbd>←</kbd> = Previous Image<br/>
        • <kbd>R</kbd> = Reset Image Annotations<br/>
        • <kbd>H</kbd> = Toggle High-Res Reference Thumbnail<br/>
        • <kbd>G</kbd> = Toggle 64x64 Grid Lines
      </div>
    </div>
  </div>
</main>

<script>
  const CANVAS_SIZE = 64;
  const DISPLAY_SIZE = 512;
  const SCALE = DISPLAY_SIZE / CANVAS_SIZE; // 8px per pixel

  let imagesList = [];
  let annotationsDB = {};
  let currentIndex = 0;

  let currentEyes = [];      // [[x, y], [x, y]]
  let currentFace = null;     // [x1, y1, x2, y2]
  let currentMode = "eye1";   // "eye1" | "eye2" | "face"
  let isDraggingFace = false;
  let faceDragStart = null;
  let showGrid = true;
  let showHires = false;

  const canvas = document.getElementById("view-canvas");
  const ctx = canvas.getContext("2d");
  const img64 = new Image();

  async function init() {
    try {
      const res = await fetch("/api/init");
      const data = await res.json();
      imagesList = data.images;
      annotationsDB = data.annotations || {};

      // Resume at first unannotated image
      let firstUnannotated = 0;
      for (let i = 0; i < imagesList.length; i++) {
        if (!annotationsDB[imagesList[i].file]) {
          firstUnannotated = i;
          break;
        }
      }
      currentIndex = firstUnannotated;
      loadImage(currentIndex);
      updateCounters();
    } catch (e) {
      alert("Failed to load images from server: " + e);
    }
  }

  function updateCounters() {
    document.getElementById("idx-counter").textContent = `${currentIndex + 1} / ${imagesList.length}`;
    let marked = 0;
    let skipped = 0;
    for (const k in annotationsDB) {
      if (annotationsDB[k].no_face) skipped++;
      else if (annotationsDB[k].eyes && annotationsDB[k].eyes.length > 0) marked++;
    }
    document.getElementById("marked-counter").textContent = marked;
    document.getElementById("skipped-counter").textContent = skipped;
  }

  function loadImage(idx) {
    if (idx < 0 || idx >= imagesList.length) return;
    currentIndex = idx;
    const item = imagesList[currentIndex];

    document.getElementById("prompt-filename").textContent = item.file;
    document.getElementById("prompt-text").textContent = item.prompt || "(No prompt text)";

    // Load existing annotation if available
    const existing = annotationsDB[item.file];
    if (existing) {
      currentEyes = existing.eyes ? [...existing.eyes] : [];
      currentFace = existing.face ? [...existing.face] : null;
    } else {
      currentEyes = [];
      currentFace = null;
    }

    if (currentEyes.length === 0) setMode("eye1");
    else if (currentEyes.length === 1) setMode("eye2");
    else if (!currentFace) setMode("face");
    else setMode("face");

    img64.onload = () => render();
    img64.src = `/image/64/${encodeURIComponent(item.file)}`;

    const hiresImg = document.getElementById("hires-img");
    hiresImg.src = `/image/hires/${encodeURIComponent(item.file)}`;

    updateCounters();
  }

  function setMode(mode) {
    currentMode = mode;
    document.querySelectorAll(".mode-btn").forEach(b => {
      b.classList.toggle("active", b.dataset.mode === mode);
    });

    const instr = document.getElementById("step-instruction");
    const hudStep = document.getElementById("hud-step");
    if (mode === "eye1") {
      instr.innerHTML = "👉 Click <strong>Eye 1 (Left)</strong>, or press <kbd>S</kbd> if no face";
      hudStep.textContent = "Eye 1";
    } else if (mode === "eye2") {
      instr.innerHTML = "👉 Click <strong>Eye 2 (Right)</strong>";
      hudStep.textContent = "Eye 2";
    } else {
      instr.innerHTML = "👉 Click & Drag (or click 2 corners) for <strong>Face Box</strong>";
      hudStep.textContent = "Face Box";
    }
    render();
  }

  function render() {
    ctx.clearRect(0, 0, DISPLAY_SIZE, DISPLAY_SIZE);

    // Draw pixelated 64x64 image
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(img64, 0, 0, DISPLAY_SIZE, DISPLAY_SIZE);

    // Draw Grid Lines (Subtle)
    if (showGrid) {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
      ctx.lineWidth = 1;
      for (let i = 0; i <= CANVAS_SIZE; i += 4) {
        ctx.beginPath();
        ctx.moveTo(i * SCALE, 0);
        ctx.lineTo(i * SCALE, DISPLAY_SIZE);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(0, i * SCALE);
        ctx.lineTo(DISPLAY_SIZE, i * SCALE);
        ctx.stroke();
      }
    }

    // Draw Face Box
    if (currentFace) {
      const [x1, y1, x2, y2] = currentFace;
      const minX = Math.min(x1, x2) * SCALE;
      const minY = Math.min(y1, y2) * SCALE;
      const w = (Math.abs(x2 - x1) + 1) * SCALE;
      const h = (Math.abs(y2 - y1) + 1) * SCALE;

      ctx.strokeStyle = "#ffb703";
      ctx.lineWidth = 3;
      ctx.strokeRect(minX, minY, w, h);

      ctx.fillStyle = "rgba(255, 183, 3, 0.15)";
      ctx.fillRect(minX, minY, w, h);

      ctx.fillStyle = "#ffb703";
      ctx.font = "bold 12px sans-serif";
      ctx.fillText("FACE", minX + 4, minY + 14);
    }

    // Draw Eyes
    currentEyes.forEach((pt, i) => {
      const px = pt[0] * SCALE + SCALE / 2;
      const py = pt[1] * SCALE + SCALE / 2;
      const color = (i === 0) ? "#00e5ff" : "#ff007f";

      // Outer Glow
      ctx.beginPath();
      ctx.arc(px, py, 7, 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.stroke();

      // Center Dot
      ctx.beginPath();
      ctx.arc(px, py, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = "#fff";
      ctx.fill();

      // Label
      ctx.fillStyle = color;
      ctx.font = "bold 11px sans-serif";
      ctx.fillText((i === 0 ? "L" : "R"), px + 8, py - 6);
    });
  }

  // Mouse Coordinate Handling
  function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const clientY = e.clientY - rect.top;
    const gridX = Math.min(CANVAS_SIZE - 1, Math.max(0, Math.floor(clientX / SCALE)));
    const gridY = Math.min(CANVAS_SIZE - 1, Math.max(0, Math.floor(clientY / SCALE)));
    return [gridX, gridY];
  }

  canvas.addEventListener("mousemove", (e) => {
    const [gx, gy] = getCanvasCoords(e);
    document.getElementById("hud-coords").textContent = `(${gx}, ${gy})`;

    if (isDraggingFace && faceDragStart) {
      currentFace = [faceDragStart[0], faceDragStart[1], gx, gy];
      render();
    }
  });

  canvas.addEventListener("mousedown", (e) => {
    const [gx, gy] = getCanvasCoords(e);

    if (currentMode === "eye1") {
      currentEyes[0] = [gx, gy];
      setMode("eye2");
    } else if (currentMode === "eye2") {
      currentEyes[1] = [gx, gy];
      setMode("face");
    } else if (currentMode === "face") {
      isDraggingFace = true;
      faceDragStart = [gx, gy];
      currentFace = [gx, gy, gx, gy];
      render();
    }
  });

  window.addEventListener("mouseup", () => {
    if (isDraggingFace) {
      isDraggingFace = false;
      faceDragStart = null;
    }
  });

  async function postSave(annotationData) {
    const item = imagesList[currentIndex];
    annotationsDB[item.file] = annotationData;

    try {
      await fetch("/api/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          file: item.file,
          ...annotationData
        })
      });
    } catch (e) {
      console.error("Save error:", e);
    }
    updateCounters();
  }

  function saveAndNext() {
    const data = {
      eyes: currentEyes,
      face: currentFace,
      no_face: (currentEyes.length === 0 && !currentFace)
    };
    postSave(data);
    nextImage();
  }

  function skipNoFace() {
    currentEyes = [];
    currentFace = null;
    postSave({ eyes: [], face: null, no_face: true });
    nextImage();
  }

  function clearCurrent() {
    currentEyes = [];
    currentFace = null;
    setMode("eye1");
    render();
  }

  function prevImage() {
    if (currentIndex > 0) loadImage(currentIndex - 1);
  }

  function nextImage() {
    if (currentIndex < imagesList.length - 1) loadImage(currentIndex + 1);
  }

  function toggleHires() {
    showHires = !showHires;
    document.getElementById("hires-box").style.display = showHires ? "block" : "none";
  }

  // Keyboard Shortcuts
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

    if (e.code === "Space") {
      e.preventDefault();
      saveAndNext();
    } else if (e.key === "s" || e.key === "S") {
      e.preventDefault();
      skipNoFace();
    } else if (e.key === "r" || e.key === "R" || e.key === "Backspace") {
      e.preventDefault();
      clearCurrent();
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      nextImage();
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      prevImage();
    } else if (e.key === "1") {
      setMode("eye1");
    } else if (e.key === "2") {
      setMode("eye2");
    } else if (e.key === "3") {
      setMode("face");
    } else if (e.key === "h" || e.key === "H") {
      toggleHires();
    } else if (e.key === "g" || e.key === "G") {
      showGrid = !showGrid;
      render();
    }
  });

  window.onload = init;
</script>
</body>
</html>
"""

def main():
    print("=" * 60)
    print("=== Yumon Pet: DiffusionDB ROI Face & Eye Annotator ===")
    print("=" * 60)

    images = load_diffusiondb_index()
    if not images:
        sys.exit(1)

    existing = load_existing_annotations()
    print(f"Existing annotations in {OUTPUT_JSON_PATH}: {len(existing)}")

    AnnotationServer.images_index = images
    AnnotationServer.annotations_db = existing

    port = 8080
    for test_port in range(8080, 8095):
        try:
            server = HTTPServer(("127.0.0.1", test_port), AnnotationServer)
            port = test_port
            break
        except OSError:
            continue

    url = f"http://127.0.0.1:{port}"
    print(f"\nServer running at: {url}")
    print(f"Annotations will automatically save to: {OUTPUT_JSON_PATH}")
    print("Press Ctrl+C in this terminal when finished marking.\n")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSaving and shutting down annotator server...")
        save_annotations(AnnotationServer.annotations_db)
        print("Done! Have a great day!")

if __name__ == "__main__":
    main()
