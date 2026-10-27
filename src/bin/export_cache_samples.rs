//! Export and visualize diagnostic samples from the multimodal art cache.
//!
//! Dumps PNG images, a text manifest, and an interactive HTML gallery to inspect
//! progressive canvas states and prompts.

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use std::{
    fs::{self, File},
    io::{BufReader, Read, Seek, SeekFrom},
    path::PathBuf,
};
use yumon_pet::vision::paint_action::{
    CANVAS_SIZE, PASS_1_ID, PASS_2_ID, PASS_3_ID, PASS_4_ID,
    pass1_tokens_to_image, pass2_tokens_to_image, pass3_tokens_to_image, pass4_tokens_to_image,
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
    pub pass: u8,
    pub tokens: Vec<u32>,
    pub target_start: usize,
    #[serde(default)]
    pub spatial_weights: Option<Vec<f32>>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct ArtSampleV3 {
    pub prompt_text: String,
    pub pass: u8,
    pub tokens: Vec<u32>,
    pub target_start: usize,
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

    /// Number of samples to skip from the beginning of the cache.
    #[arg(long, default_value_t = 0)]
    offset: usize,

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

fn make_pair_image(input: Option<&[u8]>, target: &[u8], scale: usize) -> (usize, usize, Vec<u8>) {
    let card_w = CANVAS_SIZE * scale;
    let card_h = CANVAS_SIZE * scale;
    let gap = 16;
    let total_w = card_w * 2 + gap;
    let total_h = card_h;
    let mut out = vec![24u8; total_w * total_h * 3]; // dark background

    let target_up = upscale_nn(target, CANVAS_SIZE, CANVAS_SIZE, scale);
    let input_up = input.map(|inp| upscale_nn(inp, CANVAS_SIZE, CANVAS_SIZE, scale));

    for y in 0..card_h {
        // Left: Input conditioning (or checkerboard placeholder if None)
        for x in 0..card_w {
            let out_idx = (y * total_w + x) * 3;
            if let Some(ref inp) = input_up {
                let in_idx = (y * card_w + x) * 3;
                out[out_idx] = inp[in_idx];
                out[out_idx + 1] = inp[in_idx + 1];
                out[out_idx + 2] = inp[in_idx + 2];
            } else {
                let check = if (x / (8 * scale) + y / (8 * scale)) % 2 == 0 { 40 } else { 28 };
                out[out_idx] = check;
                out[out_idx + 1] = check;
                out[out_idx + 2] = check;
            }
        }

        // Right: Target Ground Truth
        for x in 0..card_w {
            let out_idx = (y * total_w + (card_w + gap + x)) * 3;
            let tgt_idx = (y * card_w + x) * 3;
            out[out_idx] = target_up[tgt_idx];
            out[out_idx + 1] = target_up[tgt_idx + 1];
            out[out_idx + 2] = target_up[tgt_idx + 2];
        }
    }

    (total_w, total_h, out)
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("=== Yumon Art Cache Diagnostic Exporter ===");
    println!("Reading cache: {}", args.cache.display());
    println!("Target output directory: {}", args.output_dir.display());
    println!("Exporting {} samples starting at offset {} (upscale {}x)...", args.count, args.offset, args.scale);

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

    if args.offset >= meta.total_samples {
        println!("Offset {} is beyond total samples in cache ({}). Nothing to export.", args.offset, meta.total_samples);
        return Ok(());
    }

    // 2. Skip offset samples if requested
    if args.offset > 0 {
        println!("Skipping first {} samples...", args.offset);
        for i in 0..args.offset {
            let mut sample_len_bytes = [0u8; 4];
            if reader.read_exact(&mut sample_len_bytes).is_err() {
                println!("Reached end of file while seeking at sample {}", i);
                return Ok(());
            }
            let sample_len = u32::from_le_bytes(sample_len_bytes) as i64;
            reader.seek(SeekFrom::Current(sample_len))
                .with_context(|| format!("seeking past sample {}", i))?;
        }
        println!("Successfully skipped {} samples.", args.offset);
    }

    let export_count = args.count.min(meta.total_samples.saturating_sub(args.offset));

    let mut html_cards = String::new();
    let mut manifest_lines = Vec::new();

    manifest_lines.push(format!(
        "# Yumon Art Cache Export: {} samples (offset {}) from {}\n",
        export_count,
        args.offset,
        args.cache.display()
    ));

    let mut pass_counts = [0usize; 4];

    for idx in 0..export_count {
        let global_idx = args.offset + idx;
        let mut sample_len_bytes = [0u8; 4];
        if reader.read_exact(&mut sample_len_bytes).is_err() {
            println!("Reached end of file at sample {}", global_idx);
            break;
        }
        let sample_len = u32::from_le_bytes(sample_len_bytes) as usize;
        let mut sample_bytes = vec![0u8; sample_len];
        reader.read_exact(&mut sample_bytes)?;
        let sample: ArtSample = if meta.version == 3 {
            let v3: ArtSampleV3 = bincode::deserialize(&sample_bytes)?;
            ArtSample {
                prompt_text: v3.prompt_text,
                pass: v3.pass,
                tokens: v3.tokens,
                target_start: v3.target_start,
                spatial_weights: None,
            }
        } else {
            bincode::deserialize(&sample_bytes)?
        };

        // Target tokens & canvas (the ground-truth output the model learns to produce)
        let target_colors = if sample.target_start < sample.tokens.len() {
            let end = if sample.tokens.last() == Some(&4099) {
                sample.tokens.len() - 1
            } else {
                sample.tokens.len()
            };
            &sample.tokens[sample.target_start..end]
        } else {
            &[]
        };

        let target_canvas = match sample.pass {
            1 => pass1_tokens_to_image(target_colors),
            2 => pass2_tokens_to_image(target_colors),
            3 => pass3_tokens_to_image(target_colors),
            4 => pass4_tokens_to_image(target_colors),
            _ => pass4_tokens_to_image(target_colors),
        };

        // Extract conditioning input canvas (what the model actually ingests)
        let (input_canvas, cond_token_count) = match sample.pass {
            1 => (None, 0),
            2 => {
                let p1_pos = sample.tokens.iter().position(|&t| t == PASS_1_ID as u32);
                let p2_pos = sample.tokens.iter().position(|&t| t == PASS_2_ID as u32);
                if let (Some(p1), Some(p2)) = (p1_pos, p2_pos) {
                    if p2 > p1 + 1 {
                        let cond_tokens = &sample.tokens[p1 + 1..p2];
                        (Some(pass1_tokens_to_image(cond_tokens)), cond_tokens.len())
                    } else {
                        (None, 0)
                    }
                } else {
                    (None, 0)
                }
            }
            3 => {
                let p2_pos = sample.tokens.iter().position(|&t| t == PASS_2_ID as u32);
                let p3_pos = sample.tokens.iter().position(|&t| t == PASS_3_ID as u32);
                if let (Some(p2), Some(p3)) = (p2_pos, p3_pos) {
                    if p3 > p2 + 1 {
                        let cond_tokens = &sample.tokens[p2 + 1..p3];
                        (Some(pass2_tokens_to_image(cond_tokens)), cond_tokens.len())
                    } else {
                        (None, 0)
                    }
                } else {
                    (None, 0)
                }
            }
            4 => {
                let p3_pos = sample.tokens.iter().position(|&t| t == PASS_3_ID as u32);
                let p4_pos = sample.tokens.iter().position(|&t| t == PASS_4_ID as u32);
                if let (Some(p3), Some(p4)) = (p3_pos, p4_pos) {
                    if p4 > p3 + 1 {
                        let cond_tokens = &sample.tokens[p3 + 1..p4];
                        (Some(pass3_tokens_to_image(cond_tokens)), cond_tokens.len())
                    } else {
                        (None, 0)
                    }
                } else {
                    (None, 0)
                }
            }
            _ => (None, 0),
        };

        if sample.pass >= 1 && (sample.pass as usize) <= 4 {
            pass_counts[sample.pass as usize - 1] += 1;
        }

        // File names
        let pair_filename = format!("sample_{:05}.png", global_idx);
        let pair_path = args.output_dir.join(&pair_filename);
        let target_filename = format!("sample_{:05}_target.png", global_idx);
        let target_path = args.output_dir.join(&target_filename);

        // Save target image
        let upscaled_target = upscale_nn(&target_canvas, CANVAS_SIZE, CANVAS_SIZE, args.scale);
        let target_img = image::RgbImage::from_raw(
            (CANVAS_SIZE * args.scale) as u32,
            (CANVAS_SIZE * args.scale) as u32,
            upscaled_target,
        ).context("creating target ImageBuffer")?;
        target_img.save(&target_path)?;

        // Save conditioning input image if present
        if let Some(ref inp) = input_canvas {
            let input_filename = format!("sample_{:05}_input.png", global_idx);
            let input_path = args.output_dir.join(&input_filename);
            let upscaled_input = upscale_nn(inp, CANVAS_SIZE, CANVAS_SIZE, args.scale);
            let input_img = image::RgbImage::from_raw(
                (CANVAS_SIZE * args.scale) as u32,
                (CANVAS_SIZE * args.scale) as u32,
                upscaled_input,
            ).context("creating input ImageBuffer")?;
            input_img.save(&input_path)?;
        }

        // Save side-by-side composite pair: [Ingested Conditioning] -> [Target Output]
        let (pair_w, pair_h, pair_bytes) = make_pair_image(input_canvas.as_deref(), &target_canvas, args.scale);
        let pair_img = image::RgbImage::from_raw(pair_w as u32, pair_h as u32, pair_bytes)
            .context("creating pair ImageBuffer")?;
        pair_img.save(&pair_path)?;

        let pass_name = match sample.pass {
            1 => "Pass 1 (8x8 Composition)",
            2 => "Pass 2 (16x16 Shapes)",
            3 => "Pass 3 (32x32 Features)",
            4 => "Pass 4 (64x64 Pixel Art)",
            _ => "Unknown Pass",
        };
        let action_desc = format!("{} • Target: {} tokens", pass_name, target_colors.len());
        let swatch_html = format!("<span class=\"action-tag\">{}</span>", pass_name);

        let input_tag = if sample.pass == 1 {
            "<span class=\"noise-badge prompt-only\">✎ Prompt Only</span>".to_string()
        } else {
            format!("<span class=\"noise-badge cond\">⇦ {} Ingested Tokens</span>", cond_token_count)
        };

        let cond_desc = if sample.pass == 1 {
            "Prompt Only (Pass 1)".to_string()
        } else {
            format!("Cascaded Conditioning ({} input tokens)", cond_token_count)
        };

        // Add to text manifest
        manifest_lines.push(format!(
            "Sample #{:05}\n  Image       : {}\n  Pass        : {}\n  Conditioning: {}\n  Prompt      : {}\n",
            global_idx, pair_filename, action_desc, cond_desc, sample.prompt_text
        ));

        // Add to HTML card
        let escaped_prompt = escape_html(&sample.prompt_text);
        html_cards.push_str(&format!(
            r#"    <div class="card">
      <div class="card-header">
        <span class="sample-id">#{:05}</span>
        <span class="action-tag">{}</span>
      </div>
      <div class="img-wrapper">
        <a href="{}" target="_blank">
          <img src="{}" alt="Sample {:05}" />
        </a>
      </div>
      <div class="labels-row">
        <span>⇦ Ingested Conditioning</span>
        <span>Target Output ⇨</span>
      </div>
      <div class="details">
        <div class="swatch-row">{} {}</div>
        <p class="prompt">"{}"</p>
      </div>
    </div>
"#,
            global_idx, action_desc, pair_filename, pair_filename, global_idx, swatch_html, input_tag, escaped_prompt
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
      grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
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
      width: 100%;
      max-width: 440px;
      height: auto;
      image-rendering: pixelated;
      border-radius: 4px;
      border: 1px solid #30363d;
    }}
    .labels-row {{
      display: flex;
      justify-content: space-around;
      font-size: 11px;
      color: #8b949e;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin: 4px 14px 0 14px;
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
    .noise-badge {{
      display: inline-block;
      padding: 2px 7px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
    }}
    .noise-badge.noisy {{
      background: #9e6a03;
      color: #fff;
    }}
    .noise-badge.clean {{
      background: #238636;
      color: #fff;
    }}
    .noise-badge.prompt-only {{
      background: #1f6feb;
      color: #fff;
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
    <p class="subtitle">Inspecting {} progressive canvas states (offset {}, samples {}..{}) and corresponding prompts from <code>{}</code></p>
  </header>
  <div class="grid">
{}
  </div>
</body>
</html>
"#,
        export_count,
        args.offset,
        args.offset,
        args.offset + export_count.saturating_sub(1),
        args.cache.display(),
        html_cards
    );

    let html_path = args.output_dir.join("index.html");
    fs::write(&html_path, html_content)?;
    println!("Saved interactive HTML gallery to {}", html_path.display());

    println!("\n✅ Successfully exported {} samples to {}!", export_count, args.output_dir.display());
    println!("\nExport Summary ({} samples exported):", export_count);
    println!("  - Pass 1 (Prompt -> 8x8) : {}", pass_counts[0]);
    println!("  - Pass 2 (8x8 -> 16x16)  : {}", pass_counts[1]);
    println!("  - Pass 3 (16x16 -> 32x32): {}", pass_counts[2]);
    println!("  - Pass 4 (32x32 -> 64x64): {}", pass_counts[3]);
    println!("\n🖼  Each sample exported as side-by-side composite: [Ingested Conditioning Input] ➔ [Target Predicted Output]");
    println!("    - Combined Pair : sample_XXXXX.png");
    println!("    - Standalone In : sample_XXXXX_input.png (Passes 2-4)");
    println!("    - Standalone Tgt: sample_XXXXX_target.png");
    println!("    - HTML Gallery  : index.html");
    Ok(())
}
