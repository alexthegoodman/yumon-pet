//! Interactive Studio: Watch Yumon create art with bounding boxes on a 64x64 canvas.
//!
//! Given an art prompt, Yumon outputs a sequence of bounding boxes:
//!   [CANVAS_START, BOX, x1, y1, x2, y2, color, BOX, ..., DONE]
//! Each box is rendered live to the terminal using 24-bit ANSI truecolor.

#[cfg(not(target_arch = "wasm32"))]
use {
    anyhow::{Context, Result},
    burn::{
        prelude::*,
        tensor::{Int, TensorData},
    },
    clap::Parser,
    std::{path::PathBuf, thread::sleep, time::{Duration, Instant}},
    yumon_pet::brain::{
        kilo_vocab::KiloVocab,
        moe_model::{YumonMoeBrain, YumonMoeBrainConfig},
        samples::TrainingStage,
        tokenizer::{BOS_TOKEN, EOS_TOKEN},
        yumon_talk::TranslationContext,
    },
    yumon_pet::vision::{
        BOX_TOKEN_ID, BOX_VOCAB_SIZE, CANVAS_SIZE, CANVAS_START_ID, COLOR_START_ID,
        DONE_TOKEN_ID, LayerBox, X_START_ID, Y_START_ID,
        palette_to_rgb, render_boxes_to_hwc,
    },
};

type BackendType = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser, Debug)]
#[command(about = "Interactive Art Creation: Watch Yumon paint a 64x64 canvas using Bounding Boxes")]
struct Args {
    /// Art prompt to guide Yumon.
    #[arg(long, default_value = "a cute friendly dragon in an enchanted forest, vibrant colors, oil painting")]
    prompt: String,

    /// Optional checkpoint directory.
    #[arg(long)]
    checkpoint: Option<PathBuf>,

    /// Path to Kilo Talk vocabulary.
    #[arg(long, default_value = "training-cache/diffusion_talk_vocab.json")]
    vocab: PathBuf,

    /// Sampling temperature (0.0 = greedy argmax, >0.0 = stochastic sampling).
    #[arg(long, default_value_t = 0.5)]
    temperature: f32,

    /// Maximum bounding boxes Yumon can place.
    #[arg(long, default_value_t = 100)]
    max_boxes: usize,

    /// Milliseconds to pause between rendering each box for live animation.
    #[arg(long, default_value_t = 60)]
    box_delay_ms: u64,

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
    // Print a compact 32x16 ANSI-colored terminal preview (downsampling 2x2 pixels)
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
    println!("║     🎨 Yumon Art Studio: Bounding Box Painter         ║");
    println!("╚═══════════════════════════════════════════════════════╝");
    println!("Prompt: \"{}\"\n", args.prompt);

    // 1. Load Kilo Talk vocabulary
    let vocab = KiloVocab::load_from_json(&args.vocab)
        .with_context(|| format!("loading vocabulary from {}", args.vocab.display()))?;

    // 2. Tokenize prompt
    let mut ctx = TranslationContext::new();
    let prompt_tokens = vocab.encode(&args.prompt, &mut ctx)?;
    println!("Kilo Talk tokens ({}): {:?}", prompt_tokens.len(), prompt_tokens);

    // 3. Initialize / Load MoE Brain
    let brain = if let Some(ckpt) = &args.checkpoint {
        println!("Loading checkpoint from {}...", ckpt.display());
        let (model, _, _) = YumonMoeBrain::<BackendType>::load(ckpt.to_str().unwrap(), &device)?;
        model
    } else {
        println!("No checkpoint specified — initializing compact untrained MoE architecture for studio demonstration...");
        let cfg = YumonMoeBrainConfig::new(BOX_VOCAB_SIZE, TrainingStage::Language)
            .with_embed_dim(64)
            .with_hidden_units(64)
            .with_n_layers(4)
            .with_attn_heads(4)
            .with_ff_dim(256)
            .with_max_seq_len(768)
            .with_num_experts(4)
            .with_top_k(1)
            .with_is_multimodal(false);
        cfg.init(&device)
    };

    // 4. Initialize Blank Canvas
    let mut canvas_hwc = vec![240u8; CANVAS_SIZE * CANVAS_SIZE * 3];
    println!("\n🖼  Initial Canvas State (Blank):");
    print_terminal_canvas(&canvas_hwc);

    // 5. The Art Creation Loop: Autoregressive Box Emission
    println!("\n🎨 Yumon is picking up the stylus to paint bounding boxes...\n");
    let t_start = Instant::now();

    let mut generated_tokens: Vec<usize> = Vec::new();
    generated_tokens.push(BOS_TOKEN);
    for &tok in &prompt_tokens {
        generated_tokens.push(tok);
    }
    generated_tokens.push(CANVAS_START_ID);

    let mut boxes: Vec<LayerBox> = Vec::new();
    let max_gen_steps = args.max_boxes * 6 + 10;

    // Current box builder accumulator: [BOX, x1, y1, x2, y2, color]
    let mut box_buf: Vec<usize> = Vec::new();

    for step in 0..max_gen_steps {
        if generated_tokens.len() >= 768 {
            println!("  [Context window full: 768 tokens reached]");
            break;
        }

        let token_tensor = Tensor::<BackendType, 2, Int>::from_ints(
            TensorData::new(
                generated_tokens.iter().map(|&t| t as i32).collect::<Vec<i32>>(),
                [1, generated_tokens.len()],
            ),
            &device,
        );

        let (logits, _, _, _) = brain.forward_with_depth(token_tensor);
        let last_pos = generated_tokens.len() - 1;
        let logit_vec: Vec<f32> = logits
            .slice([0..1, last_pos..last_pos + 1, 0..BOX_VOCAB_SIZE])
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();

        // Constrain logits depending on what slot in the 6-token box tuple is expected
        // Slot 0: BOX or DONE
        // Slot 1: X1 (4100..4163)
        // Slot 2: Y1 (4164..4227)
        // Slot 3: X2 (4100..4163)
        // Slot 4: Y2 (4164..4227)
        // Slot 5: COLOR (4228..4483)
        let next_tok = match box_buf.len() {
            0 => {
                // Can emit BOX_TOKEN_ID or DONE_TOKEN_ID
                let p_box = logit_vec[BOX_TOKEN_ID];
                let p_done = logit_vec[DONE_TOKEN_ID];
                if p_done > p_box && !boxes.is_empty() {
                    DONE_TOKEN_ID
                } else {
                    BOX_TOKEN_ID
                }
            }
            1 | 3 => {
                // X coordinate (0..63)
                let x_logits = &logit_vec[X_START_ID..X_START_ID + 64];
                let x_val = sample_token(x_logits, args.temperature);
                X_START_ID + x_val
            }
            2 | 4 => {
                // Y coordinate (0..63)
                let y_logits = &logit_vec[Y_START_ID..Y_START_ID + 64];
                let y_val = sample_token(y_logits, args.temperature);
                Y_START_ID + y_val
            }
            5 => {
                // Color (0..255)
                let c_logits = &logit_vec[COLOR_START_ID..COLOR_START_ID + 256];
                let c_val = sample_token(c_logits, args.temperature);
                COLOR_START_ID + c_val
            }
            _ => DONE_TOKEN_ID,
        };

        if next_tok == DONE_TOKEN_ID || next_tok == EOS_TOKEN {
            println!("\n  ✨ Yumon finished the composition! Emitted <DONE> at step {}.", step);
            break;
        }

        generated_tokens.push(next_tok);
        box_buf.push(next_tok);

        // When 6 tokens are assembled, decode the box and render it!
        if box_buf.len() == 6 {
            let raw_x1 = (box_buf[1] - X_START_ID) as u8;
            let raw_y1 = (box_buf[2] - Y_START_ID) as u8;
            let raw_x2 = (box_buf[3] - X_START_ID) as u8;
            let raw_y2 = (box_buf[4] - Y_START_ID) as u8;
            let color = (box_buf[5] - COLOR_START_ID) as u8;

            let x1 = raw_x1.min(raw_x2);
            let x2 = raw_x1.max(raw_x2);
            let y1 = raw_y1.min(raw_y2);
            let y2 = raw_y1.max(raw_y2);

            let new_box = LayerBox { x1, y1, x2, y2, color };
            boxes.push(new_box);

            // Re-render canvas with new box
            render_boxes_to_hwc(&[new_box], &mut canvas_hwc);

            let [r, g, b] = palette_to_rgb(color);
            let w = x2 - x1 + 1;
            let h = y2 - y1 + 1;

            print!(
                "  Box #{:>2}: [{}, {}] -> [{}, {}] ({}x{}) | Color #{:>3} \x1b[38;2;{};{};{}m████\x1b[0m\r\n",
                boxes.len(), x1, y1, x2, y2, w, h, color, r, g, b
            );
            print_terminal_canvas(&canvas_hwc);

            if args.box_delay_ms > 0 {
                sleep(Duration::from_millis(args.box_delay_ms));
            }

            box_buf.clear();

            if boxes.len() >= args.max_boxes {
                println!("\n  ✨ Reached max boxes limit ({}).", args.max_boxes);
                break;
            }
        }
    }

    println!("\n═══════════════════════════════════════════════════════");
    println!("  🎨 Artwork Summary: {} bounding boxes placed in {:.2?}", boxes.len(), t_start.elapsed());
    println!("═══════════════════════════════════════════════════════");
    print_terminal_canvas(&canvas_hwc);

    // Save final PNG
    let upscaled = upscale_nn(&canvas_hwc, CANVAS_SIZE, CANVAS_SIZE, args.scale);
    let final_img = image::RgbImage::from_raw(
        (CANVAS_SIZE * args.scale) as u32,
        (CANVAS_SIZE * args.scale) as u32,
        upscaled,
    ).context("creating final ImageBuffer")?;
    final_img.save(&args.output)?;
    println!("\n💾 Saved final artwork to: {}", args.output.display());

    Ok(())
}
