//! Export and visualize diagnostic samples from the multimodal art cache.
//!
//! Dumps PNG images, a text manifest, and an interactive HTML gallery to inspect
//! progressive canvas states and prompts.

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use std::{
    fs::{self, File},
    io::{BufReader, Read},
    path::PathBuf,
};
use yumon_pet::vision::paint_action::{
    CANVAS_SIZE, PaintAction, palette_to_rgb,
};

#[derive(Deserialize, Debug)]
pub struct ArtCacheMetadata {
    pub version: u32,
    pub total_samples: usize,
    pub images_processed: usize,
    pub canvas_size: usize,
    pub vocab_size: usize,
    pub trajectory_steps: usize,
}

#[derive(Deserialize, Clone, Debug)]
pub struct ArtSample {
    pub prompt_text: String,
    pub prompt_tokens: Vec<u32>,
    pub canvas_hwc: Vec<u8>,
    pub action_tokens: Vec<u32>,
    pub is_done: bool,
}

#[derive(Parser, Debug)]
#[command(about = "Export diagnostic visual samples and prompts from art training cache")]
struct Args {
    /// Path to input binary cache.
    #[arg(long, default_value = "training-cache/diffusion_kilo_samples.bin")]
    cache: PathBuf,

    /// Output folder for exported images and gallery.
    #[arg(long, default_value = "training-cache/sample_previews")]
    output_dir: PathBuf,

    /// Number of samples to export.
    #[arg(long, default_value_t = 100)]
    count: usize,

    /// Upscale factor for exported PNGs (default 4 -> 256x256 crisp pixel art).
    #[arg(long, default_value_t = 4)]
    scale: usize,
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn upscale_nn(src: &[u8], width: usize, height: usize, scale: usize) -> Vec<u8> {
    let out_w = width * scale;
    let out_h = height * scale;
    let mut out = vec![0u8; out_w * out_h * 3];

    for y in 0..out_h {
        let src_y = y / scale;
        for x in 0..out_w {
            let src_x = x / scale;
            let src_idx = (src_y * width + src_x) * 3;
            let out_idx = (y * out_w + x) * 3;
            out[out_idx] = src[src_idx];
            out[out_idx + 1] = src[src_idx + 1];
            out[out_idx + 2] = src[src_idx + 2];
        }
    }

    out
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("=== Yumon Art Cache Diagnostic Exporter ===");
    println!("Reading cache: {}", args.cache.display());
    println!("Target output directory: {}", args.output_dir.display());
    println!("Exporting {} samples (upscale {}x)...", args.count, args.scale);

    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("creating directory {}", args.output_dir.display()))?;

    let file = File::open(&args.cache)
        .with_context(|| format!("opening cache file {}", args.cache.display()))?;
    let mut reader = BufReader::new(file);

    // 1. Read header magic and metadata
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    let mut meta_len_bytes = [0u8; 4];
    reader.read_exact(&mut meta_len_bytes)?;
    let meta_len = u32::from_le_bytes(meta_len_bytes) as usize;
    let mut meta_bytes = vec![0u8; meta_len];
    reader.read_exact(&mut meta_bytes)?;
    let meta: ArtCacheMetadata = bincode::deserialize(&meta_bytes)?;

    println!("Cache info: {} total samples across {} images", meta.total_samples, meta.images_processed);

    let export_count = args.count.min(meta.total_samples);

    let mut html_cards = String::new();
    let mut manifest_lines = Vec::new();

    manifest_lines.push(format!(
        "# Yumon Art Cache Export: {} samples from {}\n",
        export_count,
        args.cache.display()
    ));

    for idx in 0..export_count {
        let mut sample_len_bytes = [0u8; 4];
        if reader.read_exact(&mut sample_len_bytes).is_err() {
            println!("Reached end of file at sample {}", idx);
            break;
        }
        let sample_len = u32::from_le_bytes(sample_len_bytes) as usize;
        let mut sample_bytes = vec![0u8; sample_len];
        reader.read_exact(&mut sample_bytes)?;
        let sample: ArtSample = bincode::deserialize(&sample_bytes)?;

        // File names
        let img_filename = format!("sample_{:03}.png", idx);
        let raw_filename = format!("sample_{:03}_64.png", idx);
        let img_path = args.output_dir.join(&img_filename);
        let raw_path = args.output_dir.join(&raw_filename);

        // Save raw 64x64 PNG
        let raw_img = image::RgbImage::from_raw(
            CANVAS_SIZE as u32,
            CANVAS_SIZE as u32,
            sample.canvas_hwc.clone(),
        ).context("creating 64x64 ImageBuffer")?;
        raw_img.save(&raw_path)?;

        // Save upscaled 256x256 PNG for crisp display
        let upscaled_bytes = upscale_nn(&sample.canvas_hwc, CANVAS_SIZE, CANVAS_SIZE, args.scale);
        let upscaled_img = image::RgbImage::from_raw(
            (CANVAS_SIZE * args.scale) as u32,
            (CANVAS_SIZE * args.scale) as u32,
            upscaled_bytes,
        ).context("creating upscaled ImageBuffer")?;
        upscaled_img.save(&img_path)?;

        // Format action description
        let (action_desc, swatch_html) = if sample.action_tokens.len() >= 3 {
            let x = sample.action_tokens[0] as usize - yumon_pet::vision::paint_action::COORD_START_ID;
            let y = sample.action_tokens[1] as usize - yumon_pet::vision::paint_action::COORD_START_ID;
            let c = (sample.action_tokens[2] as usize - yumon_pet::vision::paint_action::COLOR_START_ID) as u8;
            let [r, g, b] = palette_to_rgb(c);
            let pass_name = if idx % 48 < 16 {
                "Pass 1 (8x8 Composition)"
            } else if idx % 48 < 32 {
                "Pass 2 (16x16 Shapes)"
            } else {
                "Pass 3 (64x64 Details)"
            };
            (
                format!("{} • ({}, {})", pass_name, x, y),
                format!(
                    "<span class=\"swatch\" style=\"background: rgb({}, {}, {});\"></span> Color #{} RGB({}, {}, {})",
                    r, g, b, c, r, g, b
                ),
            )
        } else {
            let tok_usize: Vec<usize> = sample.action_tokens.iter().map(|&t| t as usize).collect();
            let action = PaintAction::from_tokens(&tok_usize);
            match action {
                Some(PaintAction::Paint { x, y, color, radius }) => {
                    let [r, g, b] = palette_to_rgb(color);
                    (
                        format!("Paint dab at ({}, {}) | Color #{} | Radius {}", x, y, color, radius),
                        format!(
                            "<span class=\"swatch\" style=\"background: rgb({}, {}, {});\"></span> RGB({}, {}, {})",
                            r, g, b, r, g, b
                        ),
                    )
                }
                Some(PaintAction::Done) => (
                    "Done (Artwork Complete)".to_string(),
                    "<span class=\"done-badge\">DONE</span>".to_string(),
                ),
                None => ("Pixel Stream Sample".to_string(), "".to_string()),
            }
        };

        // Add to text manifest
        manifest_lines.push(format!(
            "Sample #{:03}\n  Image  : {}\n  Action : {}\n  Prompt : {}\n",
            idx, img_filename, action_desc, sample.prompt_text
        ));

        // Add to HTML card
        let escaped_prompt = escape_html(&sample.prompt_text);
        html_cards.push_str(&format!(
            r#"    <div class="card">
      <div class="card-header">
        <span class="sample-id">#{:03}</span>
        <span class="action-tag">{}</span>
      </div>
      <div class="img-wrapper">
        <a href="{}" target="_blank">
          <img src="{}" alt="Sample {:03}" />
        </a>
      </div>
      <div class="details">
        <div class="swatch-row">{}</div>
        <p class="prompt">"{}"</p>
      </div>
    </div>
"#,
            idx, action_desc, img_filename, img_filename, idx, swatch_html, escaped_prompt
        ));
    }

    // Write text manifest
    let manifest_path = args.output_dir.join("manifest.txt");
    fs::write(&manifest_path, manifest_lines.join("\n"))?;
    println!("Saved text manifest to {}", manifest_path.display());

    // Generate HTML gallery
    let html_content = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Yumon Art Cache: 100 Sample Diagnostics</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #0f1117;
      color: #e6edf3;
      margin: 0;
      padding: 24px;
    }}
    header {{
      max-width: 1400px;
      margin: 0 auto 24px auto;
      border-bottom: 1px solid #30363d;
      padding-bottom: 16px;
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 24px;
      color: #58a6ff;
    }}
    p.subtitle {{
      margin: 0;
      color: #8b949e;
      font-size: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 20px;
      max-width: 1400px;
      margin: 0 auto;
    }}
    .card {{
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 14px;
      background: #21262d;
      border-bottom: 1px solid #30363d;
      font-size: 13px;
    }}
    .sample-id {{
      font-weight: bold;
      color: #79c0ff;
    }}
    .action-tag {{
      color: #d2a8ff;
      font-size: 12px;
    }}
    .img-wrapper {{
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 16px;
      background: #0d1117;
    }}
    img {{
      width: 256px;
      height: 256px;
      image-rendering: pixelated;
      border-radius: 4px;
      border: 1px solid #30363d;
    }}
    .details {{
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      flex-grow: 1;
    }}
    .swatch-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: #8b949e;
    }}
    .swatch {{
      width: 16px;
      height: 16px;
      border-radius: 3px;
      display: inline-block;
      border: 1px solid #8b949e;
    }}
    .done-badge {{
      background: #238636;
      color: #ffffff;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: bold;
    }}
    p.prompt {{
      margin: 0;
      font-size: 13px;
      line-height: 1.4;
      color: #c9d1d9;
    }}
  </style>
</head>
<body>
  <header>
    <h1>🎨 Yumon Art Multimodal Cache: Sample Inspection</h1>
    <p class="subtitle">Inspecting {} progressive canvas states and corresponding DiffusionDb prompts from <code>{}</code></p>
  </header>
  <div class="grid">
{}
  </div>
</body>
</html>
"#,
        export_count,
        args.cache.display(),
        html_cards
    );

    let html_path = args.output_dir.join("index.html");
    fs::write(&html_path, html_content)?;
    println!("Saved interactive HTML gallery to {}", html_path.display());

    println!("\n✅ Successfully exported {} samples to {}!", export_count, args.output_dir.display());
    Ok(())
}
