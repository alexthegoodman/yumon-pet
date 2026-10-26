//! Interactive Studio: Watch Yumon create art on a 64x64 canvas.
//!
//! Uses cascaded autoregressive causal Transformers with Flash Attention
//! and KV-caching across 3 coarse-to-fine stages:
//! - Pass 1: Global composition (64 8x8 block color tokens)
//! - Pass 2: Shapes & Silhouettes (256 4x4 patch color tokens)
//! - Pass 3: Fine pixel art details (4,096 1x1 pixel color tokens)

#[cfg(not(target_arch = "wasm32"))]
use {
    anyhow::{Context, Result},
    burn::{
        prelude::*,
        record::{BinFileRecorder, FullPrecisionSettings},
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
    yumon_pet::vision::{
        paint_action::{
            ART_VOCAB_SIZE, CANVAS_SIZE, CANVAS_CHANNELS,
            PASS_1_ID, PASS_2_ID, PASS_3_ID, PASS_4_ID,
            COLOR_START_ID, PALETTE_SIZE,
            build_cascaded_inference_coords,
            palette_to_rgb,
            pass1_tokens_to_image, pass2_tokens_to_image, pass3_tokens_to_image, pass4_tokens_to_image,
        },
        upscaler::{UpscalerModel, UpscalerModelConfig},
    },
};

use chrono::Local;
use std::fs;

type BackendType = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser, Debug)]
#[command(about = "Interactive Art Creation: Watch Yumon paint a 64x64 canvas")]
struct Args {
    /// Art prompt to guide Yumon.
    #[arg(long, default_value = "a cute friendly dragon in an enchanted forest, vibrant colors, oil painting")]
    prompt: String,

    /// Optional checkpoint directory (fallback for all passes if specific checkpoint is omitted).
    #[arg(long)]
    checkpoint: Option<PathBuf>,

    /// Pass 1 checkpoint directory (Broad Composition, e.g. checkpoints/yumon-art-pass1).
    #[arg(long)]
    checkpoint_pass1: Option<PathBuf>,

    /// Pass 2 checkpoint directory (Shapes & Silhouettes, e.g. checkpoints/yumon-art-pass2).
    #[arg(long)]
    checkpoint_pass2: Option<PathBuf>,

    /// Pass 3 checkpoint directory (Anatomy & Features, e.g. checkpoints/yumon-art-pass3).
    #[arg(long)]
    checkpoint_pass3: Option<PathBuf>,

    /// Pass 4 MoE checkpoint directory (Fine Grayscale Details, e.g. checkpoints/yumon-art-pass4-db).
    #[arg(long)]
    checkpoint_pass4: Option<PathBuf>,

    /// Pass 5 Neural Upscaler checkpoint directory (e.g. checkpoints/yumon-upscaler-64).
    #[arg(long, default_value = "checkpoints/yumon-upscaler-64")]
    upscaler_checkpoint: PathBuf,

    /// Number of cascaded generation passes to execute (1 = 8x8, 2 = 16x16, 3 = 32x32, 4 = 64x64, 5 = 256x256).
    #[arg(long, default_value_t = 5)]
    passes: usize,

    /// Path to Kilo Talk vocabulary.
    #[arg(long, default_value = "training-cache/diffusion_talk_vocab.json")]
    vocab: PathBuf,

    /// Sampling temperature (0.0 = greedy argmax, >0.0 = stochastic sampling).
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    /// Top-K sampling filter (0 to disable, e.g. 12).
    #[arg(long, default_value_t = 12)]
    top_k: usize,

    /// Top-P (nucleus) sampling filter (1.0 to disable, e.g. 0.90).
    #[arg(long, default_value_t = 0.90)]
    top_p: f32,

    /// Repetition penalty applied to runs of identical consecutive colors (0.0 to disable).
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,

    /// Optional shape guidance weight from prior silhouette pass (0.0 to disable).
    #[arg(long, default_value_t = 0.5)]
    shape_guidance: f32,

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
fn sample_color_token(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    last_color: Option<usize>,
    consecutive_repeats: usize,
    repeat_penalty: f32,
) -> usize {
    let mut working_logits = logits.to_vec();

    // Consecutive repetition penalty:
    // If the exact same color has been emitted 4+ times consecutively in a raster row,
    // apply a penalty to break out of runaway absorbing loops.
    if let Some(c) = last_color {
        if consecutive_repeats >= 4 && c < working_logits.len() && repeat_penalty > 0.0 {
            let penalty = (consecutive_repeats as f32 - 3.0) * repeat_penalty;
            working_logits[c] -= penalty;
        }
    }

    if temperature <= 0.001 {
        return working_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
    }

    // Sort candidates descending by logit
    let mut candidates: Vec<(usize, f32)> = working_logits.iter().cloned().enumerate().collect();
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Top-K truncation
    if top_k > 0 && top_k < candidates.len() {
        candidates.truncate(top_k);
    }

    // Softmax with temperature
    let max_logit = candidates[0].1;
    let mut exp_probs: Vec<(usize, f32)> = candidates
        .into_iter()
        .map(|(idx, l)| (idx, ((l - max_logit) / temperature).exp()))
        .collect();
    let sum_exp: f32 = exp_probs.iter().map(|(_, p)| p).sum();
    if sum_exp <= 0.0 {
        return exp_probs.first().map(|(idx, _)| *idx).unwrap_or(0);
    }
    for (_, p) in &mut exp_probs {
        *p /= sum_exp;
    }

    // Top-P (Nucleus) truncation
    if top_p > 0.0 && top_p < 1.0 {
        let mut cum_p = 0.0;
        let mut cut = exp_probs.len();
        for (i, &(_, p)) in exp_probs.iter().enumerate() {
            cum_p += p;
            if cum_p >= top_p {
                cut = (i + 1).max(1);
                break;
            }
        }
        exp_probs.truncate(cut);
        let renorm_sum: f32 = exp_probs.iter().map(|(_, p)| p).sum();
        if renorm_sum > 0.0 {
            for (_, p) in &mut exp_probs {
                *p /= renorm_sum;
            }
        }
    }

    // Stochastic sample from remaining candidates
    let r = rand::random::<f32>();
    let mut acc = 0.0;
    for &(idx, p) in &exp_probs {
        acc += p;
        if acc >= r {
            return idx;
        }
    }
    exp_probs.last().map(|&(idx, _)| idx).unwrap_or(0)
}

#[cfg(not(target_arch = "wasm32"))]
fn print_terminal_canvas(canvas: &[u8]) {
    println!("\n  ┌────────────────────────────────┐");
    for y in (0..CANVAS_SIZE).step_by(4) {
        print!("  │");
        for x in (0..CANVAS_SIZE).step_by(2) {
            let idx = (y * CANVAS_SIZE + x) * 3;
            let r = canvas[idx];
            let g = canvas[idx + 1];
            let b = canvas[idx + 2];
            print!("\x1b[38;2;{};{};{}m█\x1b[0m", r, g, b);
        }
        println!("│");
    }
    println!("  └────────────────────────────────┘");
}

#[cfg(not(target_arch = "wasm32"))]
fn load_or_init_pass_brain(
    pass_name: &str,
    pass_ckpt: Option<&PathBuf>,
    fallback_ckpt: Option<&PathBuf>,
    max_seq_len: usize,
    device: &<BackendType as Backend>::Device,
) -> Result<YumonMoeBrain<BackendType>> {
    let target = pass_ckpt.or(fallback_ckpt);
    if let Some(ckpt) = target {
        println!("Loading {} checkpoint from {}...", pass_name, ckpt.display());
        let (model, _, _) = YumonMoeBrain::<BackendType>::load(ckpt.to_str().unwrap(), device)?;
        Ok(model)
    } else {
        println!("No {} checkpoint specified — initializing compact untrained MoE architecture...", pass_name);
        let cfg = YumonMoeBrainConfig::new(ART_VOCAB_SIZE, TrainingStage::Language)
            .with_embed_dim(64)
            .with_hidden_units(64)
            .with_n_layers(4)
            .with_attn_heads(4)
            .with_ff_dim(256)
            .with_max_seq_len(max_seq_len)
            .with_num_experts(4)
            .with_top_k(1)
            .with_is_multimodal(false);
        Ok(cfg.init(device))
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<()> {
    let args = Args::parse();
    let device: <BackendType as Backend>::Device = Default::default();

    println!("╔═══════════════════════════════════════════════════════╗");
    println!("║   🎨 Yumon Art Studio: Autoregressive Image LM        ║");
    println!("╚═══════════════════════════════════════════════════════╝");
    println!("Prompt: \"{}\"\n", args.prompt);

    // 1. Load Kilo Talk vocabulary
    let vocab = KiloVocab::load_from_json(&args.vocab)
        .with_context(|| format!("loading vocabulary from {}", args.vocab.display()))?;

    // 2. Tokenize prompt
    let mut ctx = TranslationContext::new();
    let prompt_tokens = vocab.encode(&args.prompt, &mut ctx)?;
    println!("Kilo Talk tokens ({}): {:?}", prompt_tokens.len(), prompt_tokens);

    // 3. Initialize / Load Causal MoE Brains for Cascaded Passes
    let brain_p1 = load_or_init_pass_brain(
        "Pass 1",
        args.checkpoint_pass1.as_ref(),
        args.checkpoint.as_ref(),
        128,
        &device,
    )?;

    let brain_p2 = if args.passes >= 2 {
        Some(load_or_init_pass_brain(
            "Pass 2",
            args.checkpoint_pass2.as_ref(),
            args.checkpoint.as_ref(),
            512,
            &device,
        )?)
    } else {
        None
    };

    let brain_p3 = if args.passes >= 3 {
        Some(load_or_init_pass_brain(
            "Pass 3",
            args.checkpoint_pass3.as_ref(),
            args.checkpoint.as_ref(),
            1536,
            &device,
        )?)
    } else {
        None
    };

    let brain_p4 = if args.passes >= 4 {
        Some(load_or_init_pass_brain(
            "Pass 4",
            args.checkpoint_pass4.as_ref(),
            args.checkpoint.as_ref(),
            5500,
            &device,
        )?)
    } else {
        None
    };

    let upscaler = if args.passes >= 5 {
        let ckpt_dir = &args.upscaler_checkpoint;
        let model_path = ckpt_dir.join("model");
        if model_path.exists() || ckpt_dir.join("model.bin").exists() || ckpt_dir.join("model.mpk").exists() {
            println!("Loading Neural Upscaler from {}...", ckpt_dir.display());
            let config = UpscalerModelConfig::new()
                .with_channels(32);
            let model: UpscalerModel<BackendType> = config.init(&device);
            let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
            match model.load_file(&model_path, &recorder, &device) {
                Ok(m) => Some(m),
                Err(e) => {
                    eprintln!("Warning: Failed to load upscaler weights: {}. Will fallback to nearest-neighbor.", e);
                    None
                }
            }
        } else {
            None
        }
    } else {
        None
    };

    // 4. Initialize Blank Canvas
    let mut canvas_hwc = vec![240u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS];
    println!("\n🖼  Initial Canvas State (Blank):");
    print_terminal_canvas(&canvas_hwc);

    // 5. The Cascaded Art Generation Loop
    println!("\n🎨 Yumon is generating the artwork across {} cascaded passes...\n", args.passes);

    // PASS 1: Broad Composition (8x8 Blocks, 64 color tokens)
    println!("\n═══════════════════════════════════════════════════════");
    println!("  🎨 Pass 1 / {}: Broad Composition (64 Blocks)", args.passes);
    println!("═══════════════════════════════════════════════════════");
    let t_pass1 = Instant::now();

    let mut p1_prefix = vec![BOS_TOKEN as i32];
    for &tok in &prompt_tokens {
        p1_prefix.push(tok as i32);
    }
    p1_prefix.push(PASS_1_ID as i32);

    let p1_coords = build_cascaded_inference_coords(&p1_prefix, 1, 64);
    let mut cache_p1 = brain_p1.new_kv_cache_mrope(&[p1_coords], &device);
    let prefill_tensor = Tensor::<BackendType, 2, Int>::from_ints(
        TensorData::new(p1_prefix.clone(), [1, p1_prefix.len()]),
        &device,
    );
    let logits = brain_p1.forward_cached(prefill_tensor, &mut cache_p1);
    let mut last_logit: Vec<f32> = logits
        .slice([0..1, p1_prefix.len() - 1..p1_prefix.len(), 0..ART_VOCAB_SIZE])
        .into_data()
        .convert::<f32>()
        .to_vec()
        .unwrap();

    let mut pass1_colors = Vec::with_capacity(64);
    let mut p1_last_color = None;
    let mut p1_consecutive = 0;
    for _ in 0..64 {
        let color_logits = &last_logit[COLOR_START_ID..COLOR_START_ID + PALETTE_SIZE];
        let best_palette_idx = sample_color_token(
            color_logits,
            args.temperature,
            args.top_k,
            args.top_p,
            p1_last_color,
            p1_consecutive,
            args.repeat_penalty,
        ) as u8;
        if Some(best_palette_idx as usize) == p1_last_color {
            p1_consecutive += 1;
        } else {
            p1_last_color = Some(best_palette_idx as usize);
            p1_consecutive = 1;
        }
        let next_token = (COLOR_START_ID + best_palette_idx as usize) as u32;
        pass1_colors.push(next_token);

        let step_tensor = Tensor::<BackendType, 2, Int>::from_ints(
            TensorData::new(vec![next_token as i32], [1, 1]),
            &device,
        );
        let step_logits = brain_p1.forward_cached(step_tensor, &mut cache_p1);
        last_logit = step_logits.reshape([ART_VOCAB_SIZE]).into_data().convert::<f32>().to_vec().unwrap();
    }

    canvas_hwc = pass1_tokens_to_image(&pass1_colors);
    println!("  ✓ Pass 1 Complete ({:.2?}): Global composition established.", t_pass1.elapsed());
    print_terminal_canvas(&canvas_hwc);

    // PASS 2: Shapes & Silhouettes (16x16 Patches, 256 color tokens)
    let mut pass2_colors = Vec::with_capacity(256);
    if let Some(brain_p2) = &brain_p2 {
        println!("\n═══════════════════════════════════════════════════════");
        println!("  🎨 Pass 2 / {}: Shapes & Silhouettes (256 Patches)", args.passes);
        println!("═══════════════════════════════════════════════════════");
        let t_pass2 = Instant::now();

        let mut p2_prefix = vec![BOS_TOKEN as i32];
        for &tok in &prompt_tokens {
            p2_prefix.push(tok as i32);
        }
        p2_prefix.push(PASS_1_ID as i32);
        for &tok in &pass1_colors {
            p2_prefix.push(tok as i32);
        }
        p2_prefix.push(PASS_2_ID as i32);

        let p2_coords = build_cascaded_inference_coords(&p2_prefix, 2, 256);
        let mut cache_p2 = brain_p2.new_kv_cache_mrope(&[p2_coords], &device);
        let prefill_tensor = Tensor::<BackendType, 2, Int>::from_ints(
            TensorData::new(p2_prefix.clone(), [1, p2_prefix.len()]),
            &device,
        );
        let logits = brain_p2.forward_cached(prefill_tensor, &mut cache_p2);
        let mut last_logit: Vec<f32> = logits
            .slice([0..1, p2_prefix.len() - 1..p2_prefix.len(), 0..ART_VOCAB_SIZE])
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();

        let mut p2_last_color = None;
        let mut p2_consecutive = 0;
        for i in 0..256 {
            let color_logits = &last_logit[COLOR_START_ID..COLOR_START_ID + PALETTE_SIZE];
            let best_palette_idx = sample_color_token(
                color_logits,
                args.temperature,
                args.top_k,
                args.top_p,
                p2_last_color,
                p2_consecutive,
                args.repeat_penalty,
            ) as u8;
            if Some(best_palette_idx as usize) == p2_last_color {
                p2_consecutive += 1;
            } else {
                p2_last_color = Some(best_palette_idx as usize);
                p2_consecutive = 1;
            }
            let next_token = (COLOR_START_ID + best_palette_idx as usize) as u32;
            pass2_colors.push(next_token);

            let step_tensor = Tensor::<BackendType, 2, Int>::from_ints(
                TensorData::new(vec![next_token as i32], [1, 1]),
                &device,
            );
            let step_logits = brain_p2.forward_cached(step_tensor, &mut cache_p2);
            last_logit = step_logits.reshape([ART_VOCAB_SIZE]).into_data().convert::<f32>().to_vec().unwrap();

            if (i + 1) % 64 == 0 {
                println!("  [Pass 2 Progress] Patch {:>3}/256 refined...", i + 1);
            }
        }

        canvas_hwc = pass2_tokens_to_image(&pass2_colors);
        println!("  ✓ Pass 2 Complete ({:.2?}): Subject shapes and silhouettes refined.", t_pass2.elapsed());
        print_terminal_canvas(&canvas_hwc);
    }

    // PASS 3: Anatomy & Features (32x32 Sub-Patches, 1,024 color tokens)
    let mut pass3_colors = Vec::with_capacity(1024);
    if let Some(brain_p3) = &brain_p3 {
        println!("\n═══════════════════════════════════════════════════════");
        println!("  🎨 Pass 3 / {}: Anatomy & Features (1,024 Sub-Patches)", args.passes);
        println!("═══════════════════════════════════════════════════════");
        let t_pass3 = Instant::now();

        let mut p3_prefix = vec![BOS_TOKEN as i32];
        for &tok in &prompt_tokens {
            p3_prefix.push(tok as i32);
        }
        p3_prefix.push(PASS_2_ID as i32);
        for &tok in &pass2_colors {
            p3_prefix.push(tok as i32);
        }
        p3_prefix.push(PASS_3_ID as i32);

        let p3_coords = build_cascaded_inference_coords(&p3_prefix, 3, 1024);
        let mut cache_p3 = brain_p3.new_kv_cache_mrope(&[p3_coords], &device);
        let prefill_tensor = Tensor::<BackendType, 2, Int>::from_ints(
            TensorData::new(p3_prefix.clone(), [1, p3_prefix.len()]),
            &device,
        );
        let logits = brain_p3.forward_cached(prefill_tensor, &mut cache_p3);
        let mut last_logit: Vec<f32> = logits
            .slice([0..1, p3_prefix.len() - 1..p3_prefix.len(), 0..ART_VOCAB_SIZE])
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();

        let mut p3_last_color = None;
        let mut p3_consecutive = 0;
        for i in 0..1024 {
            let color_logits = &last_logit[COLOR_START_ID..COLOR_START_ID + PALETTE_SIZE];
            let py = i / 32;
            let px = i % 32;
            let patch_idx = (py / 2) * 16 + (px / 2);

            let mut working_logits = color_logits.to_vec();
            if args.shape_guidance > 0.0 && patch_idx < pass2_colors.len() {
                let shape_token = pass2_colors[patch_idx];
                let shape_color = (shape_token as usize).saturating_sub(COLOR_START_ID);
                if shape_color < PALETTE_SIZE {
                    working_logits[shape_color] += args.shape_guidance;
                }
            }

            let best_palette_idx = sample_color_token(
                &working_logits,
                args.temperature,
                args.top_k,
                args.top_p,
                p3_last_color,
                p3_consecutive,
                args.repeat_penalty,
            ) as u8;

            if Some(best_palette_idx as usize) == p3_last_color {
                p3_consecutive += 1;
            } else {
                p3_last_color = Some(best_palette_idx as usize);
                p3_consecutive = 1;
            }

            let next_token = (COLOR_START_ID + best_palette_idx as usize) as u32;
            pass3_colors.push(next_token);

            let step_tensor = Tensor::<BackendType, 2, Int>::from_ints(
                TensorData::new(vec![next_token as i32], [1, 1]),
                &device,
            );
            let step_logits = brain_p3.forward_cached(step_tensor, &mut cache_p3);
            last_logit = step_logits.reshape([ART_VOCAB_SIZE]).into_data().convert::<f32>().to_vec().unwrap();

            if (i + 1) % 256 == 0 || i == 1023 {
                let y = (i + 1) / 32;
                println!("  [Pass 3 Progress] Sub-patch {:>4}/1024 (Row {:>2}/32 refined)...", i + 1, y);
            }
        }

        canvas_hwc = pass3_tokens_to_image(&pass3_colors);
        println!("  ✓ Pass 3 Complete ({:.2?}): Subject anatomy, features, and contours established.", t_pass3.elapsed());
        print_terminal_canvas(&canvas_hwc);
    }

    // PASS 4: Fine Grayscale Details (64x64 Pixels, 4,096 color tokens)
    let mut pass4_colors = Vec::with_capacity(4096);
    if let Some(brain_p4) = &brain_p4 {
        println!("\n═══════════════════════════════════════════════════════");
        println!("  🎨 Pass 4 / {}: Fine Grayscale Details (4,096 Pixels)", args.passes);
        println!("═══════════════════════════════════════════════════════");
        let t_pass4 = Instant::now();

        let mut p4_prefix = vec![BOS_TOKEN as i32];
        for &tok in &prompt_tokens {
            p4_prefix.push(tok as i32);
        }
        p4_prefix.push(PASS_3_ID as i32);
        for &tok in &pass3_colors {
            p4_prefix.push(tok as i32);
        }
        p4_prefix.push(PASS_4_ID as i32);

        let p4_coords = build_cascaded_inference_coords(&p4_prefix, 4, 4096);
        let mut cache_p4 = brain_p4.new_kv_cache_mrope(&[p4_coords], &device);
        let prefill_tensor = Tensor::<BackendType, 2, Int>::from_ints(
            TensorData::new(p4_prefix.clone(), [1, p4_prefix.len()]),
            &device,
        );
        let logits = brain_p4.forward_cached(prefill_tensor, &mut cache_p4);
        let mut last_logit: Vec<f32> = logits
            .slice([0..1, p4_prefix.len() - 1..p4_prefix.len(), 0..ART_VOCAB_SIZE])
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();

        let mut p4_last_color = None;
        let mut p4_consecutive = 0;
        for i in 0..4096 {
            let color_logits = &last_logit[COLOR_START_ID..COLOR_START_ID + PALETTE_SIZE];
            let py = i / 64;
            let px = i % 64;
            let sub_idx = (py / 2) * 32 + (px / 2);

            let mut working_logits = color_logits.to_vec();
            if args.shape_guidance > 0.0 && sub_idx < pass3_colors.len() {
                let sub_token = pass3_colors[sub_idx];
                let sub_color = (sub_token as usize).saturating_sub(COLOR_START_ID);
                if sub_color < PALETTE_SIZE {
                    working_logits[sub_color] += args.shape_guidance;
                }
            }

            let best_palette_idx = sample_color_token(
                &working_logits,
                args.temperature,
                args.top_k,
                args.top_p,
                p4_last_color,
                p4_consecutive,
                args.repeat_penalty,
            ) as u8;

            if Some(best_palette_idx as usize) == p4_last_color {
                p4_consecutive += 1;
            } else {
                p4_last_color = Some(best_palette_idx as usize);
                p4_consecutive = 1;
            }

            let next_token = (COLOR_START_ID + best_palette_idx as usize) as u32;
            pass4_colors.push(next_token);

            let step_tensor = Tensor::<BackendType, 2, Int>::from_ints(
                TensorData::new(vec![next_token as i32], [1, 1]),
                &device,
            );
            let step_logits = brain_p4.forward_cached(step_tensor, &mut cache_p4);
            last_logit = step_logits.reshape([ART_VOCAB_SIZE]).into_data().convert::<f32>().to_vec().unwrap();

            if (i + 1) % 512 == 0 || i == 4095 {
                let y = (i + 1) / 64;
                println!("  [Pass 4 Progress] Pixel {:>4}/4096 (Row {:>2}/64 refined)...", i + 1, y);
            }
        }

        canvas_hwc = pass4_tokens_to_image(&pass4_colors);
        println!("  ✓ Pass 4 Complete ({:.2?}): High-resolution 64x64 grayscale details established.", t_pass4.elapsed());
        print_terminal_canvas(&canvas_hwc);
    }

    // PASS 5: Deep Neural Super-Resolution (64x64 -> 256x256 Grayscale)
    let mut final_neural_256: Option<Vec<u8>> = None;
    if args.passes >= 5 {
        println!("\n═══════════════════════════════════════════════════════");
        println!("  🎨 Pass 5 / {}: Neural Super-Resolution (64x64 -> 256x256)", args.passes);
        println!("═══════════════════════════════════════════════════════");
        let t_pass5 = Instant::now();

        if let Some(upscaler) = &upscaler {
            let plane = CANVAS_SIZE * CANVAS_SIZE; // 4096
            let mut flat = vec![0.0f32; plane];
            if !pass4_colors.is_empty() {
                for (i, &tok) in pass4_colors.iter().take(plane).enumerate() {
                    let c = (tok as usize).saturating_sub(COLOR_START_ID).min(255) as u8;
                    flat[i] = (c as f32 / 255.0) * 2.0 - 1.0;
                }
            } else {
                for y in 0..CANVAS_SIZE {
                    for x in 0..CANVAS_SIZE {
                        let idx = (y * CANVAS_SIZE + x) * 3;
                        flat[y * CANVAS_SIZE + x] = (canvas_hwc[idx] as f32 / 255.0) * 2.0 - 1.0;
                    }
                }
            }

            let input_tensor = Tensor::<BackendType, 4>::from_floats(
                TensorData::new(flat, [1, 1, CANVAS_SIZE, CANVAS_SIZE]),
                &device,
            );
            let out_tensor = upscaler.forward(input_tensor);
            let out_data: Vec<f32> = out_tensor.into_data().convert::<f32>().to_vec().unwrap();
            let mut rgb_256 = vec![0u8; 256 * 256 * 3];
            for y in 0..256 {
                for x in 0..256 {
                    let idx = y * 256 + x;
                    let val = (((out_data[idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                    let out_idx = idx * 3;
                    rgb_256[out_idx] = val;
                    rgb_256[out_idx + 1] = val;
                    rgb_256[out_idx + 2] = val;
                }
            }
            final_neural_256 = Some(rgb_256);
            println!("  ✓ Pass 5 Complete ({:.2?}): Neural super-resolution synthesized 256x256 grayscale art.", t_pass5.elapsed());
        } else {
            println!("  Neural upscaler checkpoint not loaded; saving artwork using Pass 4 resolution.");
        }
    }

    println!("\n  ✨ Yumon steps back, smiles, and says: \"Done! Artwork complete.\"");

    // Get local date and hour.
    let now = Local::now();

    let day_dir = now.format("%Y-%m-%d").to_string();
    let hour_dir = now.format("%H").to_string();
    let timestamp = now.format("%Y-%m-%d_%H-%M-%S").to_string();

    // Build inference-cache/YYYY-MM-DD/HH/
    let cache_dir = PathBuf::from("inference-cache")
        .join(&day_dir)
        .join(&hour_dir);

    fs::create_dir_all(&cache_dir)
        .context("creating inference-cache directory")?;

    let stem = args.output
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();

    // Save raw 64x64 PNG.
    let raw_output = cache_dir.join(format!(
        "{}_64_{}.png",
        stem, timestamp
    ));

    let raw_buf = image::RgbImage::from_raw(
        CANVAS_SIZE as u32,
        CANVAS_SIZE as u32,
        canvas_hwc.clone(),
    )
    .context("creating raw 64x64 ImageBuffer")?;

    raw_buf.save(&raw_output)?;

    // Save 256x256 (Neural Super-Resolution or Nearest Neighbor fallback)
    let upscaled_output = cache_dir.join(format!(
        "{}_{}.png",
        stem, timestamp
    ));

    let (out_w, out_h, upscaled_bytes) = if let Some(neural_bytes) = final_neural_256 {
        (256, 256, neural_bytes)
    } else {
        (
            CANVAS_SIZE * args.scale,
            CANVAS_SIZE * args.scale,
            upscale_nn(&canvas_hwc, CANVAS_SIZE, CANVAS_SIZE, args.scale),
        )
    };

    let upscaled_buf = image::RgbImage::from_raw(
        out_w as u32,
        out_h as u32,
        upscaled_bytes,
    )
    .context("creating upscaled ImageBuffer")?;

    upscaled_buf.save(&upscaled_output)?;
    upscaled_buf.save(&args.output)?;

    println!("\n🎨 Artwork saved successfully:");
    println!("   - Final ({}x{}):  {}", out_w, out_h, args.output.display());
    println!("   - Cached:          {}", upscaled_output.display());
    println!("   - Raw    (64x64):  {}", raw_output.display());

    Ok(())
}
