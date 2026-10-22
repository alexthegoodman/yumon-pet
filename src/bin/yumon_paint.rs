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
    std::path::PathBuf,
    yumon_pet::brain::{
        kilo_vocab::KiloVocab,
        moe_model::{YumonMoeBrain, YumonMoeBrainConfig},
        samples::TrainingStage,
        tokenizer::BOS_TOKEN,
        yumon_talk::TranslationContext,
    },
    yumon_pet::vision::paint_action::{
        ART_VOCAB_SIZE, CANVAS_CHANNELS, CANVAS_SIZE, CANVAS_START_ID, CANVAS_END_ID,
        COORD_START_ID, COLOR_START_ID,
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

    /// Maximum painting loop iterations.
    #[arg(long, default_value_t = 12)]
    max_steps: usize,

    /// Sampling temperature (0.0 = greedy argmax, >0.0 = stochastic sampling).
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    /// How often Yumon re-glances at the visual canvas (every N pixels).
    #[arg(long, default_value_t = 8)]
    glance_interval: usize,

    /// Output artwork PNG file path.
    #[arg(long, default_value = "yumon_artwork.png")]
    output: PathBuf,
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

    // 5. The Art Creation Loop: Pixel-by-Pixel Stream (4,096 Pixels)
    println!("\n🎨 Yumon is picking up the stylus to paint the 64x64 canvas pixel-by-pixel...\n");

    let total_pixels = CANVAS_SIZE * CANVAS_SIZE; // 4096
    let glance_interval = args.glance_interval.max(1);

    let mut canvas_tensor = Tensor::<BackendType, 4>::from_floats(
        TensorData::new(hwc_to_chw_floats(&canvas_hwc), [1, 3, CANVAS_SIZE, CANVAS_SIZE]),
        &device,
    );

    let mut prefix_tokens = vec![BOS_TOKEN as i32];
    for &tok in &prompt_tokens {
        prefix_tokens.push(tok as i32);
    }
    prefix_tokens.push(CANVAS_START_ID as i32);
    prefix_tokens.push(CANVAS_END_ID as i32);

    for p in 0..total_pixels {
        let x = (p % CANVAS_SIZE) as u8;
        let y = (p / CANVAS_SIZE) as u8;

        // Re-glance at the canvas according to glance_interval
        if p > 0 && p % glance_interval == 0 {
            let chw = hwc_to_chw_floats(&canvas_hwc);
            canvas_tensor = Tensor::<BackendType, 4>::from_floats(
                TensorData::new(chw, [1, 3, CANVAS_SIZE, CANVAS_SIZE]),
                &device,
            );
        }

        // Form query sequence: [prefix, x_tok, y_tok]
        let mut query_tokens = prefix_tokens.clone();
        query_tokens.push((COORD_START_ID + x as usize) as i32);
        query_tokens.push((COORD_START_ID + y as usize) as i32);

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

        let color_logits = &logit_vec[COLOR_START_ID..COLOR_START_ID + 64];
        let best_color = sample_token(color_logits, args.temperature) as u8;

        let [r, g, b] = palette_to_rgb(best_color);
        let idx = (y as usize * CANVAS_SIZE + x as usize) * 3;
        canvas_hwc[idx] = r;
        canvas_hwc[idx + 1] = g;
        canvas_hwc[idx + 2] = b;

        // Progress rendering in terminal
        if x == (CANVAS_SIZE - 1) as u8 {
            println!("  [Row {:>2}/64] Rendered pixels {:>4}/4096...", y + 1, p + 1);
            if (y + 1) % 8 == 0 || y == (CANVAS_SIZE - 1) as u8 {
                print_terminal_canvas(&canvas_hwc);
            }
        }
    }

    println!("\n  ✨ Yumon steps back, smiles, and says: \"Done! Artwork complete.\"");

    // 6. Save final image to PNG
    let img_buf = image::RgbImage::from_raw(
        CANVAS_SIZE as u32,
        CANVAS_SIZE as u32,
        canvas_hwc,
    ).context("creating final ImageBuffer")?;

    img_buf.save(&args.output)?;
    println!("\n🎨 Artwork saved successfully to: {}", args.output.display());

    Ok(())
}
