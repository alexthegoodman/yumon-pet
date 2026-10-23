//! Interactive Studio: Watch Yumon create art on a 64x64 canvas.
//!
//! Yumon views the canvas, decides which pixels/strokes to paint,
//! updates the canvas, and loops until satisfied.

#[cfg(not(target_arch = "wasm32"))]
use {
    anyhow::{Context, Result},
    burn::{
        prelude::*,
        tensor::{Int, TensorData},
    },
    clap::Parser,
    std::{path::PathBuf, time::Instant},
    yumon_pet::brain::{
        kilo_vocab::KiloVocab,
        moe_model::{YumonMoeBrain, YumonMoeBrainConfig},
        samples::TrainingStage,
        tokenizer::BOS_TOKEN,
        yumon_talk::TranslationContext,
    },
    yumon_pet::vision::paint_action::{
        ART_VOCAB_SIZE, CANVAS_CHANNELS, CANVAS_SIZE, CANVAS_START_ID,
        PASS_1_ID, PASS_2_ID, PASS_3_ID,
        ROW_START_ID, COL_START_ID, COLOR_START_ID, PALETTE_SIZE,
        hwc_to_chw_floats, palette_to_rgb,
    },
};

type BackendType = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser, Debug)]
#[command(about = "Interactive Art Creation: Watch Yumon paint a 64x64 canvas")]
struct Args {
    /// Art prompt to guide Yumon.
    #[arg(long, default_value = "a cute friendly dragon in an enchanted forest, vibrant colors, oil painting")]
    prompt: String,

    /// Optional checkpoint directory.
    #[arg(long)]
    checkpoint: Option<PathBuf>,

    /// Path to Kilo Talk vocabulary.
    #[arg(long, default_value = "training-cache/kilo_talk_vocab.json")]
    vocab: PathBuf,

    /// Sampling temperature (0.0 = greedy argmax, >0.0 = stochastic sampling).
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    /// Number of multiscale passes to execute (1 = Composition, 2 = Shapes, 3 = Fine Details).
    #[arg(long, default_value_t = 3)]
    passes: usize,

    /// How often Yumon re-glances at the visual canvas during Pass 3 (every N pixels).
    #[arg(long, default_value_t = 32)]
    glance_interval: usize,

    /// Output artwork PNG file path.
    #[arg(long, default_value = "yumon_artwork.png")]
    output: PathBuf,

    /// Upscale factor for saved PNG image (default 4 -> 256x256 crisp pixel art).
    #[arg(long, default_value_t = 4)]
    scale: usize,
}

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
fn sample_token(logits: &[f32], temperature: f32) -> usize {
    if temperature <= 0.001 {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    } else {
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = logits.iter().map(|&l| ((l - max_val) / temperature).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        if sum <= 0.0 {
            return 0;
        }
        let threshold = rand::random::<f32>() * sum;
        let mut acc = 0.0;
        for (i, &val) in exp_vals.iter().enumerate() {
            acc += val;
            if acc >= threshold {
                return i;
            }
        }
        exp_vals.len() - 1
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn print_terminal_canvas(canvas: &[u8]) {
    // Print a compact 32x16 ANSI-colored terminal preview
    println!("\n  ┌────────────────────────────────┐");
    for y in (0..CANVAS_SIZE).step_by(4) {
        print!("  │");
        for x in (0..CANVAS_SIZE).step_by(2) {
            let idx = (y * CANVAS_SIZE + x) * 3;
            let r = canvas[idx];
            let g = canvas[idx + 1];
            let b = canvas[idx + 2];
            // 24-bit ANSI truecolor block
            print!("\x1b[38;2;{};{};{}m█\x1b[0m", r, g, b);
        }
        println!("│");
    }
    println!("  └────────────────────────────────┘");
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<()> {
    let args = Args::parse();
    let device: <BackendType as Backend>::Device = Default::default();

    println!("╔═══════════════════════════════════════════════════════╗");
    println!("║       🎨 Yumon Art Studio: Pixel-Aware Vision MoE    ║");
    println!("╚═══════════════════════════════════════════════════════╝");
    println!("Prompt: \"{}\"\n", args.prompt);

    // 1. Load Kilo Talk vocabulary
    let vocab = KiloVocab::load_from_json(&args.vocab)
        .with_context(|| format!("loading vocabulary from {}", args.vocab.display()))?;

    // 2. Tokenize prompt
    let mut ctx = TranslationContext::new();
    let prompt_tokens = vocab.encode(&args.prompt, &mut ctx)?;
    println!("Kilo Talk tokens ({}): {:?}", prompt_tokens.len(), prompt_tokens);

    // 3. Initialize / Load Multimodal MoE Brain
    let brain = if let Some(ckpt) = &args.checkpoint {
        println!("Loading checkpoint from {}...", ckpt.display());
        let (model, _, _) = YumonMoeBrain::<BackendType>::load(ckpt.to_str().unwrap(), &device)?;
        model
    } else {
        println!("No checkpoint specified — initializing compact untrained MoE architecture for studio demonstration...");
        let cfg = YumonMoeBrainConfig::new(ART_VOCAB_SIZE, TrainingStage::Language)
            .with_embed_dim(64)
            .with_hidden_units(64)
            .with_n_layers(4)
            .with_attn_heads(4)
            .with_ff_dim(256)
            .with_max_seq_len(512)
            .with_num_experts(4)
            .with_top_k(1)
            .with_is_multimodal(true);
        cfg.init(&device)
    };

    // 4. Initialize Blank Canvas
    let mut canvas_hwc = vec![240u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS];
    println!("\n🖼  Initial Canvas State (Blank):");
    print_terminal_canvas(&canvas_hwc);

    // 5. The Art Creation Loop: 3-Pass Multiscale Studio
    println!("\n🎨 Yumon is picking up the stylus to paint the 64x64 canvas across 3 multiscale passes...\n");

    let mut canvas_tensor = Tensor::<BackendType, 4>::from_floats(
        TensorData::new(hwc_to_chw_floats(&canvas_hwc), [1, 3, CANVAS_SIZE, CANVAS_SIZE]),
        &device,
    );

    let mut prefix_tokens = vec![BOS_TOKEN as i32];
    for &tok in &prompt_tokens {
        prefix_tokens.push(tok as i32);
    }
    prefix_tokens.push(CANVAS_START_ID as i32);

    // PASS 1: Broad Composition (8x8 Blocks, 64 queries)
    println!("\n═══════════════════════════════════════════════════════");
    println!("  🎨 Pass 1 / 3: Broad Composition (8x8 Blocks)");
    println!("═══════════════════════════════════════════════════════");
    let t_pass1 = Instant::now();

    for by in 0..8 {
        for bx in 0..8 {
            let x = (bx * 8 + 4) as u8;
            let y = (by * 8 + 4) as u8;

            let mut query_tokens = prefix_tokens.clone();
            query_tokens.push(PASS_1_ID as i32);
            query_tokens.push((ROW_START_ID + y as usize) as i32);
            query_tokens.push((COL_START_ID + x as usize) as i32);

            let token_tensor = Tensor::<BackendType, 2, Int>::from_ints(
                TensorData::new(query_tokens.clone(), [1, query_tokens.len()]),
                &device,
            );

            let logits = brain.forward_multimodal(token_tensor, Some(canvas_tensor.clone()));
            let last_pos = 256 + query_tokens.len() - 1;
            let logit_vec: Vec<f32> = logits
                .slice([0..1, last_pos..last_pos + 1, 0..ART_VOCAB_SIZE])
                .into_data()
                .convert::<f32>()
                .to_vec()
                .unwrap();

            let color_logits = &logit_vec[COLOR_START_ID..COLOR_START_ID + PALETTE_SIZE];
            let best_color = sample_token(color_logits, args.temperature) as u8;
            let [r, g, b] = palette_to_rgb(best_color);

            for py in 0..8 {
                for px in 0..8 {
                    let cy = by * 8 + py;
                    let cx = bx * 8 + px;
                    let idx = (cy * CANVAS_SIZE + cx) * 3;
                    canvas_hwc[idx] = r;
                    canvas_hwc[idx + 1] = g;
                    canvas_hwc[idx + 2] = b;
                }
            }
        }
    }

    canvas_tensor = Tensor::<BackendType, 4>::from_floats(
        TensorData::new(hwc_to_chw_floats(&canvas_hwc), [1, 3, CANVAS_SIZE, CANVAS_SIZE]),
        &device,
    );
    println!("  ✓ Pass 1 Complete ({:.2?}): Global composition established.", t_pass1.elapsed());
    print_terminal_canvas(&canvas_hwc);

    // PASS 2: Shapes & Silhouettes (16x16 Patches, 256 queries)
    if args.passes >= 2 {
        println!("\n═══════════════════════════════════════════════════════");
        println!("  🎨 Pass 2 / 3: Shapes & Silhouettes (16x16 Patches)");
        println!("═══════════════════════════════════════════════════════");
        let t_pass2 = Instant::now();

        for py in 0..16 {
            for px in 0..16 {
                let x = (px * 4 + 2) as u8;
                let y = (py * 4 + 2) as u8;

                let mut query_tokens = prefix_tokens.clone();
                query_tokens.push(PASS_2_ID as i32);
                query_tokens.push((ROW_START_ID + y as usize) as i32);
                query_tokens.push((COL_START_ID + x as usize) as i32);

                let token_tensor = Tensor::<BackendType, 2, Int>::from_ints(
                    TensorData::new(query_tokens.clone(), [1, query_tokens.len()]),
                    &device,
                );

                let logits = brain.forward_multimodal(token_tensor, Some(canvas_tensor.clone()));
                let last_pos = 256 + query_tokens.len() - 1;
                let logit_vec: Vec<f32> = logits
                    .slice([0..1, last_pos..last_pos + 1, 0..ART_VOCAB_SIZE])
                    .into_data()
                    .convert::<f32>()
                    .to_vec()
                    .unwrap();

                let color_logits = &logit_vec[COLOR_START_ID..COLOR_START_ID + PALETTE_SIZE];
                let best_color = sample_token(color_logits, args.temperature) as u8;
                let [r, g, b] = palette_to_rgb(best_color);

                for dy in 0..4 {
                    for dx in 0..4 {
                        let cy = py * 4 + dy;
                        let cx = px * 4 + dx;
                        let idx = (cy * CANVAS_SIZE + cx) * 3;
                        canvas_hwc[idx] = r;
                        canvas_hwc[idx + 1] = g;
                        canvas_hwc[idx + 2] = b;
                    }
                }
            }

            if (py + 1) % 4 == 0 {
                canvas_tensor = Tensor::<BackendType, 4>::from_floats(
                    TensorData::new(hwc_to_chw_floats(&canvas_hwc), [1, 3, CANVAS_SIZE, CANVAS_SIZE]),
                    &device,
                );
                println!("  [Pass 2 Progress] Row {:>2}/16 refined...", py + 1);
            }
        }

        canvas_tensor = Tensor::<BackendType, 4>::from_floats(
            TensorData::new(hwc_to_chw_floats(&canvas_hwc), [1, 3, CANVAS_SIZE, CANVAS_SIZE]),
            &device,
        );
        println!("  ✓ Pass 2 Complete ({:.2?}): Subject shapes and silhouettes refined.", t_pass2.elapsed());
        print_terminal_canvas(&canvas_hwc);
    }

    // PASS 3: Fine Pixel Art Details (64x64 Pixels, 4,096 queries)
    if args.passes >= 3 {
        println!("\n═══════════════════════════════════════════════════════");
        println!("  🎨 Pass 3 / 3: Fine Pixel Art Details (64x64 Pixels)");
        println!("═══════════════════════════════════════════════════════");
        let t_pass3 = Instant::now();
        let total_pixels = CANVAS_SIZE * CANVAS_SIZE;
        let glance_interval = args.glance_interval.max(1);

        for p in 0..total_pixels {
            let x = (p % CANVAS_SIZE) as u8;
            let y = (p / CANVAS_SIZE) as u8;

            if p > 0 && p % glance_interval == 0 {
                let chw = hwc_to_chw_floats(&canvas_hwc);
                canvas_tensor = Tensor::<BackendType, 4>::from_floats(
                    TensorData::new(chw, [1, 3, CANVAS_SIZE, CANVAS_SIZE]),
                    &device,
                );
            }

            let mut query_tokens = prefix_tokens.clone();
            query_tokens.push(PASS_3_ID as i32);
            query_tokens.push((ROW_START_ID + y as usize) as i32);
            query_tokens.push((COL_START_ID + x as usize) as i32);

            let token_tensor = Tensor::<BackendType, 2, Int>::from_ints(
                TensorData::new(query_tokens.clone(), [1, query_tokens.len()]),
                &device,
            );

            let logits = brain.forward_multimodal(token_tensor, Some(canvas_tensor.clone()));
            let last_pos = 256 + query_tokens.len() - 1;
            let logit_vec: Vec<f32> = logits
                .slice([0..1, last_pos..last_pos + 1, 0..ART_VOCAB_SIZE])
                .into_data()
                .convert::<f32>()
                .to_vec()
                .unwrap();

            let color_logits = &logit_vec[COLOR_START_ID..COLOR_START_ID + PALETTE_SIZE];
            let best_color = sample_token(color_logits, args.temperature) as u8;
            let [r, g, b] = palette_to_rgb(best_color);

            let idx = (y as usize * CANVAS_SIZE + x as usize) * 3;
            canvas_hwc[idx] = r;
            canvas_hwc[idx + 1] = g;
            canvas_hwc[idx + 2] = b;

            if x == (CANVAS_SIZE - 1) as u8 && ((y + 1) % 8 == 0 || y == (CANVAS_SIZE - 1) as u8) {
                println!("  [Pass 3 Progress] Row {:>2}/64 rendered...", y + 1);
                print_terminal_canvas(&canvas_hwc);
            }
        }

        println!("  ✓ Pass 3 Complete ({:.2?}): Full pixel-level shading & highlights applied.", t_pass3.elapsed());
    }

    println!("\n  ✨ Yumon steps back, smiles, and says: \"Done! Artwork complete.\"");

    // Save final raw 64x64 PNG
    let raw_output = if let Some(ext) = args.output.extension() {
        let stem = args.output.file_stem().unwrap().to_string_lossy();
        args.output.with_file_name(format!("{}_64.{}", stem, ext.to_string_lossy()))
    } else {
        PathBuf::from("yumon_artwork_64.png")
    };

    let raw_buf = image::RgbImage::from_raw(
        CANVAS_SIZE as u32,
        CANVAS_SIZE as u32,
        canvas_hwc.clone(),
    ).context("creating raw 64x64 ImageBuffer")?;
    raw_buf.save(&raw_output)?;

    // Save upscaled PNG (e.g. 256x256)
    let upscaled_bytes = upscale_nn(&canvas_hwc, CANVAS_SIZE, CANVAS_SIZE, args.scale);
    let upscaled_buf = image::RgbImage::from_raw(
        (CANVAS_SIZE * args.scale) as u32,
        (CANVAS_SIZE * args.scale) as u32,
        upscaled_bytes,
    ).context("creating upscaled ImageBuffer")?;
    upscaled_buf.save(&args.output)?;

    println!("\n🎨 Artwork saved successfully:");
    println!("   - Upscaled ({}x{}): {}", CANVAS_SIZE * args.scale, CANVAS_SIZE * args.scale, args.output.display());
    println!("   - Raw      (64x64):   {}", raw_output.display());

    Ok(())
}
