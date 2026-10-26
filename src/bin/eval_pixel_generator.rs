//! Evaluate and Visualize Autoregressive Pixel Generator Passes (1 through 4).
//!
//! Generates side-by-side comparison grids of image samples for subjective accuracy:
//! - Pass 1: 8x8 Composition (64 block color tokens)
//! - Pass 2: 16x16 Shapes & Silhouettes (256 patch color tokens)
//! - Pass 3: 32x32 Anatomy & Features (1,024 sub-patch color tokens)
//! - Pass 4: 64x64 Fine Pixel Art (4,096 pixel color tokens)
//!
//! Features:
//! - Autoregressive causal MoE generation with KV caching and 3D M-RoPE coordinates
//! - Direct cache seeking from `diffusion_kilo_samples.bin`
//! - Multiple layout options: Triplets, Pairs, Grid, Panels, Vertical, Wide
//! - Configurable sampling: temperature, top-k, top-p, repeat penalty, shape guidance
//! - Objective quality metrics: Token Accuracy %, RGB MAE, and PSNR alongside visual grid
//! - Custom prompt mode for generating visual variation grids from arbitrary text

use anyhow::{bail, ensure, Context, Result};
use burn::{
    prelude::*,
    tensor::{Int, Tensor, TensorData},
};
use clap::{Parser, ValueEnum};
use rand::{seq::SliceRandom, Rng, SeedableRng};
use std::{
    fs::File,
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use yumon_pet::brain::{
    kilo_vocab::KiloVocab,
    moe_model::YumonMoeBrain,
    tokenizer::BOS_TOKEN,
    yumon_talk::TranslationContext,
};
use yumon_pet::vision::paint_action::{
    build_cascaded_inference_coords,
    pass1_tokens_to_image, pass2_tokens_to_image, pass3_tokens_to_image, pass4_tokens_to_image,
    ART_VOCAB_SIZE, CANVAS_CHANNELS, CANVAS_SIZE, COLOR_START_ID, PALETTE_SIZE,
    PASS_1_ID, PASS_2_ID, PASS_3_ID, PASS_4_ID,
};

pub type EvalBackend = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridLayout {
    /// Automatic layout: Pairs for Pass 1 or prompt mode; Triplets for Passes 2-4.
    Auto,
    /// Side-by-side [ Conditioning (Prior Pass) | Model Predicted | Ground Truth ].
    Triplets,
    /// Side-by-side [ Model Predicted | Ground Truth ].
    Pairs,
    /// Grid of Model Predicted images only.
    Grid,
    /// Three separate sheets: [ All Conditioning | All Predicted | All Ground Truth ].
    Panels,
    /// Single vertical strip of stacked comparison cells.
    Vertical,
    /// 2 comparison cells per row.
    Wide,
}

#[derive(Parser, Debug)]
#[command(
    name = "eval_pixel_generator",
    about = "Evaluate and save visual comparison grids for Yumon Pixel Generator Passes (1 through 4)"
)]
struct Args {
    /// Target pass to evaluate (1 = 8x8, 2 = 16x16, 3 = 32x32, 4 = 64x64).
    #[arg(short, long, default_value_t = 1)]
    pass: u8,

    /// Checkpoint directory (auto-detected based on pass if omitted).
    #[arg(short, long)]
    checkpoint: Option<PathBuf>,

    /// Path to input binary art cache file.
    #[arg(long, default_value = "training-cache/diffusion_kilo_samples.bin")]
    cache: PathBuf,

    /// Output path for generated comparison grid PNG.
    #[arg(short, long)]
    out: Option<PathBuf>,

    /// Number of evaluation samples to generate and compare (default 16).
    #[arg(short = 'n', long, default_value_t = 16, alias = "samples")]
    count: usize,

    /// Number of columns in the output grid (default 4).
    #[arg(long)]
    cols: Option<usize>,

    /// Layout style for arranging images.
    #[arg(long, value_enum, default_value_t = GridLayout::Auto)]
    layout: GridLayout,

    /// Nearest-neighbor upscale factor for crisp pixel visualization (default 4 -> 256x256).
    #[arg(long, default_value_t = 4)]
    scale: usize,

    /// Sampling temperature (0.0 = deterministic argmax greedy, >0.0 = stochastic sampling).
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    /// Top-K sampling filter (0 to disable).
    #[arg(long, default_value_t = 12)]
    top_k: usize,

    /// Top-P (nucleus) sampling filter (1.0 to disable).
    #[arg(long, default_value_t = 0.90)]
    top_p: f32,

    /// Repetition penalty applied to runs of identical consecutive colors (0.0 to disable).
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,

    /// Optional shape guidance weight from prior pass tokens for Pass 3 & 4 (0.0 to disable).
    #[arg(long, default_value_t = 0.5)]
    shape_guidance: f32,

    /// Optional custom text prompt (generates variation grid without cache ground-truth).
    #[arg(long)]
    prompt: Option<String>,

    /// Path to Kilo Talk vocabulary JSON (used when --prompt is provided).
    #[arg(long, default_value = "training-cache/diffusion_talk_vocab.json")]
    vocab: PathBuf,

    /// Offset index into cache samples (when shuffle is disabled).
    #[arg(long, default_value_t = 0)]
    offset: usize,

    /// Random seed for sample selection and sampling.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Shuffle sample selection from cache.
    #[arg(long, default_value_t = true)]
    shuffle: bool,

    /// Optional path to write a text evaluation manifest with prompt texts & metrics.
    #[arg(long)]
    manifest: Option<PathBuf>,
}

#[derive(serde::Deserialize, Debug)]
pub struct ArtCacheMetadata {
    pub version: u32,
    pub total_samples: usize,
    pub images_processed: usize,
    pub canvas_size: usize,
    pub vocab_size: usize,
    pub trajectory_steps: usize,
}

#[derive(serde::Deserialize, Clone, Debug)]
pub struct ArtSample {
    pub prompt_text: String,
    pub pass: u8,
    pub tokens: Vec<u32>,
    pub target_start: usize,
}

struct EvalResult {
    prompt_text: String,
    cond_rgb: Vec<u8>,
    pred_rgb: Vec<u8>,
    gt_rgb: Option<Vec<u8>>,
    token_acc: Option<f32>,
    rgb_mae: Option<f32>,
    psnr: Option<f32>,
    gen_time_ms: u128,
}

fn resolve_checkpoint_dir(pass: u8, explicit: Option<&PathBuf>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        if path.exists() {
            return Ok(path.clone());
        }
        bail!("Specified checkpoint path does not exist: {}", path.display());
    }

    let primary = format!("checkpoints/yumon-art-pass{}-db", pass);
    let fallbacks = [
        primary.as_str(),
        &format!("checkpoints/yumon-art-pass{}-db-512", pass),
        &format!("checkpoints/yumon-art-pass{}-db-color", pass),
        &format!("checkpoints/yumon-art-pass{}", pass),
    ];

    for candidate in &fallbacks {
        let p = Path::new(candidate);
        if p.exists() {
            return Ok(p.to_path_buf());
        }
    }

    bail!(
        "No checkpoint found for Pass {}. Looked in: {:?}. Use --checkpoint to specify path.",
        pass,
        fallbacks
    );
}

fn load_pass_samples_from_cache(path: &Path, target_pass: u8) -> Result<Vec<ArtSample>> {
    let t0 = Instant::now();
    let file = File::open(path)
        .with_context(|| format!("Failed to open cache file: {}", path.display()))?;
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    ensure!(&magic == b"YUMONART", "Invalid cache magic header in {}", path.display());

    let mut meta_len_buf = [0u8; 4];
    reader.read_exact(&mut meta_len_buf)?;
    let meta_len = u32::from_le_bytes(meta_len_buf) as usize;
    let mut meta_bytes = vec![0u8; meta_len];
    reader.read_exact(&mut meta_bytes)?;
    let _meta: ArtCacheMetadata = bincode::deserialize(&meta_bytes)?;

    let mut samples = Vec::new();
    let mut len_buf = [0u8; 4];
    while reader.read_exact(&mut len_buf).is_ok() {
        let sample_len = u32::from_le_bytes(len_buf) as usize;
        let mut sample_bytes = vec![0u8; sample_len];
        reader.read_exact(&mut sample_bytes)?;
        let sample: ArtSample = bincode::deserialize(&sample_bytes)?;
        if sample.pass == target_pass {
            samples.push(sample);
        }
    }

    println!(
        "  ✓ Loaded {} Pass {} samples from cache in {:.2?}",
        samples.len(),
        target_pass,
        t0.elapsed()
    );
    Ok(samples)
}

fn extract_prior_tokens(sample: &ArtSample, pass: u8) -> Vec<u32> {
    let prior_count = match pass {
        2 => 64,
        3 => 256,
        4 => 1024,
        _ => 0,
    };
    if prior_count == 0 {
        return Vec::new();
    }

    if sample.target_start >= prior_count + 1 {
        let start = sample.target_start - 1 - prior_count;
        let end = sample.target_start - 1;
        sample.tokens[start..end].to_vec()
    } else {
        let colors: Vec<u32> = sample.tokens[..sample.target_start]
            .iter()
            .filter(|&&t| t >= COLOR_START_ID as u32 && t < (COLOR_START_ID + PALETTE_SIZE) as u32)
            .cloned()
            .collect();
        if colors.len() >= prior_count {
            colors[colors.len() - prior_count..].to_vec()
        } else {
            colors
        }
    }
}

fn render_pass_tokens_to_rgb(tokens: &[u32], pass: u8) -> Vec<u8> {
    match pass {
        1 => pass1_tokens_to_image(tokens),
        2 => pass2_tokens_to_image(tokens),
        3 => pass3_tokens_to_image(tokens),
        4 => pass4_tokens_to_image(tokens),
        _ => pass4_tokens_to_image(tokens),
    }
}

fn sample_color_token(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    last_color: Option<usize>,
    consecutive_repeats: usize,
    repeat_penalty: f32,
    min_repeat_threshold: usize,
    rng: &mut impl Rng,
) -> usize {
    let mut working_logits = logits.to_vec();

    if let Some(c) = last_color {
        if consecutive_repeats >= min_repeat_threshold && c < working_logits.len() && repeat_penalty > 0.0 {
            let penalty = (consecutive_repeats as f32 - (min_repeat_threshold as f32 - 1.0)) * repeat_penalty;
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

    let mut candidates: Vec<(usize, f32)> = working_logits.iter().cloned().enumerate().collect();
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if top_k > 0 && top_k < candidates.len() {
        candidates.truncate(top_k);
    }

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

    let r: f32 = rng.r#gen();
    let mut acc = 0.0;
    for &(idx, p) in &exp_probs {
        acc += p;
        if acc >= r {
            return idx;
        }
    }
    exp_probs.last().map(|&(idx, _)| idx).unwrap_or(0)
}

fn generate_tokens_for_prefix(
    model: &YumonMoeBrain<EvalBackend>,
    device: &<EvalBackend as Backend>::Device,
    prefix_tokens: &[u32],
    pass: u8,
    gen_count: usize,
    prior_colors: &[u32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repeat_penalty: f32,
    shape_guidance: f32,
    rng: &mut impl Rng,
) -> Vec<u32> {
    let prefix_i32: Vec<i32> = prefix_tokens.iter().map(|&t| t as i32).collect();
    let coords = build_cascaded_inference_coords(&prefix_i32, pass, gen_count);
    let mut cache = model.new_kv_cache_mrope(&[coords], device);

    let prefill_tensor = Tensor::<EvalBackend, 2, Int>::from_ints(
        TensorData::new(prefix_i32.clone(), [1, prefix_i32.len()]),
        device,
    );
    let logits = model.forward_cached(prefill_tensor, &mut cache);
    let mut last_logit: Vec<f32> = logits
        .slice([0..1, prefix_i32.len() - 1..prefix_i32.len(), 0..ART_VOCAB_SIZE])
        .into_data()
        .convert::<f32>()
        .to_vec()
        .unwrap();

    let min_repeats = match pass {
        1 => 4,
        2 => 8,
        3 => 16,
        4 => 32,
        _ => 4,
    };
    let effective_guidance = if pass == 4 && (shape_guidance - 0.5).abs() < 1e-4 {
        3.5
    } else {
        shape_guidance
    };
    let effective_temp = if pass == 4 && (temperature - 0.7).abs() < 1e-4 {
        0.35
    } else {
        temperature
    };

    let mut pred_colors = Vec::with_capacity(gen_count);
    let mut last_color = None;
    let mut consecutive_repeats = 0;

    for step in 0..gen_count {
        let color_logits = &last_logit[COLOR_START_ID..COLOR_START_ID + PALETTE_SIZE];
        let mut working_logits = color_logits.to_vec();

        // Apply shape guidance from prior pass if enabled
        if effective_guidance > 0.0 && !prior_colors.is_empty() {
            let guide_color_opt = match pass {
                3 => {
                    let py = step / 32;
                    let px = step % 32;
                    let patch_idx = (py / 2) * 16 + (px / 2);
                    if patch_idx < prior_colors.len() {
                        Some((prior_colors[patch_idx] as usize).saturating_sub(COLOR_START_ID))
                    } else {
                        None
                    }
                }
                4 => {
                    let py = step / 64;
                    let px = step % 64;
                    let sub_idx = (py / 2) * 32 + (px / 2);
                    if sub_idx < prior_colors.len() {
                        Some((prior_colors[sub_idx] as usize).saturating_sub(COLOR_START_ID))
                    } else {
                        None
                    }
                }
                _ => None,
            };

            if let Some(gc) = guide_color_opt {
                if gc < PALETTE_SIZE {
                    working_logits[gc] += effective_guidance;
                }
            }
        }

        let best_palette_idx = sample_color_token(
            &working_logits,
            effective_temp,
            top_k,
            top_p,
            last_color,
            consecutive_repeats,
            repeat_penalty,
            min_repeats,
            rng,
        ) as u8;

        if Some(best_palette_idx as usize) == last_color {
            consecutive_repeats += 1;
        } else {
            last_color = Some(best_palette_idx as usize);
            consecutive_repeats = 1;
        }

        let next_token = (COLOR_START_ID + best_palette_idx as usize) as u32;
        pred_colors.push(next_token);

        if step + 1 < gen_count {
            let step_tensor = Tensor::<EvalBackend, 2, Int>::from_ints(
                TensorData::new(vec![next_token as i32], [1, 1]),
                device,
            );
            let step_logits = model.forward_cached(step_tensor, &mut cache);
            last_logit = step_logits.reshape([ART_VOCAB_SIZE]).into_data().convert::<f32>().to_vec().unwrap();
        }
    }

    pred_colors
}

fn blit_upscaled(
    dest: &mut image::RgbImage,
    x_dest: u32,
    y_dest: u32,
    src_rgb: &[u8],
    src_w: usize,
    src_h: usize,
    scale: usize,
) {
    for y in 0..src_h {
        let y_start = y_dest + (y * scale) as u32;
        for x in 0..src_w {
            let x_start = x_dest + (x * scale) as u32;
            let src_idx = (y * src_w + x) * 3;
            let pixel = image::Rgb([src_rgb[src_idx], src_rgb[src_idx + 1], src_rgb[src_idx + 2]]);
            for dy in 0..scale as u32 {
                for dx in 0..scale as u32 {
                    dest.put_pixel(x_start + dx, y_start + dy, pixel);
                }
            }
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t_start = Instant::now();

    ensure!(
        args.pass >= 1 && args.pass <= 4,
        "Pass must be between 1 and 4, got {}",
        args.pass
    );

    let pass_desc = match args.pass {
        1 => "Pass 1 (8x8 Composition, 64 tokens)",
        2 => "Pass 2 (16x16 Shapes & Silhouettes, 256 tokens)",
        3 => "Pass 3 (32x32 Features & Anatomy, 1,024 tokens)",
        4 => "Pass 4 (64x64 Fine Pixel Art, 4,096 tokens)",
        _ => unreachable!(),
    };

    let gen_count = match args.pass {
        1 => 64,
        2 => 256,
        3 => 1024,
        4 => 4096,
        _ => unreachable!(),
    };

    let ckpt_dir = resolve_checkpoint_dir(args.pass, args.checkpoint.as_ref())?;
    let out_path = args.out.unwrap_or_else(|| PathBuf::from(format!("eval_grid_pass{}.png", args.pass)));

    println!("==================================================");
    println!("   🎨 Yumon Pixel Generator Evaluator");
    println!("==================================================");
    println!("  Stage Target   : {}", pass_desc);
    println!("  Checkpoint     : {}", ckpt_dir.display());
    println!("  Output PNG     : {}", out_path.display());
    println!("  Scale Factor   : {}x ({}x{} pixels per tile)", args.scale, CANVAS_SIZE * args.scale, CANVAS_SIZE * args.scale);
    println!("  Temperature    : {:.2} (top_k={}, top_p={:.2})", args.temperature, args.top_k, args.top_p);
    println!("  Repeat penalty : {:.2}", args.repeat_penalty);
    if args.pass >= 3 {
        println!("  Shape guidance : {:.2}", args.shape_guidance);
    }

    let device: <EvalBackend as Backend>::Device = Default::default();

    println!("\n  Loading model from {}...", ckpt_dir.display());
    let (model, _, cfg) = YumonMoeBrain::<EvalBackend>::load(ckpt_dir.to_str().unwrap(), &device)
        .with_context(|| format!("Failed to load model from {}", ckpt_dir.display()))?;
    println!(
        "  ✓ Loaded MoE Transformer (dim={}, heads={}, layers={}, max_seq={})",
        cfg.embed_dim, cfg.attn_heads, cfg.n_layers, cfg.max_seq_len
    );

    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);

    let mut results: Vec<EvalResult> = Vec::new();

    // Mode A: Custom Prompt Mode
    if let Some(ref prompt) = args.prompt {
        println!("\n  Evaluating custom prompt: \"{}\"", prompt);
        let vocab = KiloVocab::load_from_json(&args.vocab)
            .with_context(|| format!("loading vocab from {}", args.vocab.display()))?;
        let mut ctx = TranslationContext::new();
        let prompt_tokens = vocab.encode(prompt, &mut ctx)?;

        let mut prefix = vec![BOS_TOKEN as u32];
        prefix.extend(prompt_tokens.iter().map(|&t| t as u32));
        let pass_marker = match args.pass {
            1 => PASS_1_ID,
            2 => PASS_2_ID,
            3 => PASS_3_ID,
            4 => PASS_4_ID,
            _ => unreachable!(),
        };
        prefix.push(pass_marker as u32);

        let num_variations = args.count.max(1);
        println!("  Generating {} variation(s)...", num_variations);

        let blank_canvas = vec![240u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS];

        for v in 0..num_variations {
            let t0 = Instant::now();
            let pred_colors = generate_tokens_for_prefix(
                &model,
                &device,
                &prefix,
                args.pass,
                gen_count,
                &[],
                args.temperature,
                args.top_k,
                args.top_p,
                args.repeat_penalty,
                args.shape_guidance,
                &mut rng,
            );
            let gen_time_ms = t0.elapsed().as_millis();
            let pred_rgb = render_pass_tokens_to_rgb(&pred_colors, args.pass);

            println!(
                "  [{:>2}/{}] Variation #{} generated in {:.2?}",
                v + 1,
                num_variations,
                v + 1,
                t0.elapsed()
            );

            results.push(EvalResult {
                prompt_text: prompt.clone(),
                cond_rgb: blank_canvas.clone(),
                pred_rgb,
                gt_rgb: None,
                token_acc: None,
                rgb_mae: None,
                psnr: None,
                gen_time_ms,
            });
        }
    } else {
        // Mode B: Cache Evaluation Mode (Ground-Truth Comparison)
        println!("\n  Reading cache from: {}", args.cache.display());
        let all_pass_samples = load_pass_samples_from_cache(&args.cache, args.pass)?;
        ensure!(
            !all_pass_samples.is_empty(),
            "No Pass {} samples found in {}",
            args.pass,
            args.cache.display()
        );

        let mut selected_samples: Vec<ArtSample> = if args.shuffle {
            let mut pool = all_pass_samples;
            pool.shuffle(&mut rng);
            pool.into_iter().take(args.count).collect()
        } else {
            all_pass_samples
                .into_iter()
                .skip(args.offset)
                .take(args.count)
                .collect()
        };
        selected_samples.truncate(args.count);

        let total_eval = selected_samples.len();
        println!("  Evaluating {} samples against ground-truth targets...\n", total_eval);

        let blank_canvas = vec![240u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS];

        for (idx, sample) in selected_samples.iter().enumerate() {
            let prefix_tokens = &sample.tokens[..sample.target_start];
            let prior_colors = extract_prior_tokens(sample, args.pass);

            let cond_rgb = match args.pass {
                1 => blank_canvas.clone(),
                2 => render_pass_tokens_to_rgb(&prior_colors, 1),
                3 => render_pass_tokens_to_rgb(&prior_colors, 2),
                4 => render_pass_tokens_to_rgb(&prior_colors, 3),
                _ => blank_canvas.clone(),
            };

            let gt_end = (sample.target_start + gen_count).min(sample.tokens.len());
            let gt_colors = &sample.tokens[sample.target_start..gt_end];
            let gt_rgb = render_pass_tokens_to_rgb(gt_colors, args.pass);

            let t0 = Instant::now();
            let pred_colors = generate_tokens_for_prefix(
                &model,
                &device,
                prefix_tokens,
                args.pass,
                gen_count,
                &prior_colors,
                args.temperature,
                args.top_k,
                args.top_p,
                args.repeat_penalty,
                args.shape_guidance,
                &mut rng,
            );
            let gen_time_ms = t0.elapsed().as_millis();
            let pred_rgb = render_pass_tokens_to_rgb(&pred_colors, args.pass);

            // Compute metrics
            let correct_tokens = pred_colors
                .iter()
                .zip(gt_colors.iter())
                .filter(|(p, g)| *p == *g)
                .count();
            let token_acc = (correct_tokens as f32 / gen_count as f32) * 100.0;

            let mut total_abs_diff = 0u64;
            let mut total_sq_diff = 0u64;
            let total_pixels = CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS;
            for (p_byte, g_byte) in pred_rgb.iter().zip(gt_rgb.iter()) {
                let diff = *p_byte as i32 - *g_byte as i32;
                total_abs_diff += diff.unsigned_abs() as u64;
                total_sq_diff += (diff * diff) as u64;
            }
            let rgb_mae = total_abs_diff as f32 / total_pixels as f32;
            let mse = total_sq_diff as f32 / total_pixels as f32;
            let psnr = if mse <= 0.0001 {
                99.0f32
            } else {
                20.0 * (255.0f32).log10() - 10.0 * mse.log10()
            };

            let prompt_preview = if sample.prompt_text.len() > 45 {
                format!("{}...", &sample.prompt_text[..42])
            } else {
                sample.prompt_text.clone()
            };

            println!(
                "  [{:>2}/{}] Acc: {:>5.1}% | MAE: {:>5.2} | PSNR: {:>4.1}dB | {:>5}ms | \"{}\"",
                idx + 1,
                total_eval,
                token_acc,
                rgb_mae,
                psnr,
                gen_time_ms,
                prompt_preview
            );

            results.push(EvalResult {
                prompt_text: sample.prompt_text.clone(),
                cond_rgb,
                pred_rgb,
                gt_rgb: Some(gt_rgb),
                token_acc: Some(token_acc),
                rgb_mae: Some(rgb_mae),
                psnr: Some(psnr),
                gen_time_ms,
            });
        }
    }

    let b = results.len();
    if b == 0 {
        bail!("No images to render.");
    }

    // Print summary stats if Ground Truth was available
    let has_gt = results.iter().all(|r| r.gt_rgb.is_some());
    if has_gt {
        let mean_acc: f32 = results.iter().filter_map(|r| r.token_acc).sum::<f32>() / b as f32;
        let mean_mae: f32 = results.iter().filter_map(|r| r.rgb_mae).sum::<f32>() / b as f32;
        let mean_psnr: f32 = results.iter().filter_map(|r| r.psnr).sum::<f32>() / b as f32;
        let total_time_ms: u128 = results.iter().map(|r| r.gen_time_ms).sum();
        let avg_time_ms = total_time_ms / b as u128;

        println!("\n  ================ Summary Stats ({} samples) ================", b);
        println!("  Mean Token Accuracy : {:.2}%", mean_acc);
        println!("  Mean RGB MAE        : {:.2} (scale 0-255)", mean_mae);
        println!("  Mean RGB PSNR       : {:.2} dB", mean_psnr);
        println!("  Total Inference Time: {:.2?} (avg {} ms/sample)", Instant::now().duration_since(t_start), avg_time_ms);
        println!("  =============================================================");
    }

    // Determine layout
    let layout = match args.layout {
        GridLayout::Auto => {
            if !has_gt {
                GridLayout::Grid
            } else if args.pass == 1 {
                GridLayout::Pairs
            } else {
                GridLayout::Triplets
            }
        }
        other => other,
    };

    println!("\n  Assembling output image (Layout: {:?})...", layout);

    let tile_size = (CANVAS_SIZE * args.scale) as u32;
    let cell_margin = 8u32;
    let bg_color = image::Rgb([18, 18, 26]);
    let divider_color = image::Rgb([30, 30, 40]);

    let default_cols = match layout {
        GridLayout::Vertical => 1u32,
        GridLayout::Wide => 2u32,
        _ => 4u32.min(b as u32),
    };
    let cols = args.cols.map(|c| c as u32).unwrap_or(default_cols).max(1);
    let rows = ((b as u32) + cols - 1) / cols;

    match layout {
        GridLayout::Panels => {
            let panel_w = cols * tile_size + (cols + 1) * cell_margin;
            let panel_h = rows * tile_size + (rows + 1) * cell_margin;
            let num_panels = if has_gt { 3u32 } else { 1u32 };
            let total_w = num_panels * panel_w + (num_panels - 1) * cell_margin;
            let total_h = panel_h;

            let mut grid_img = image::RgbImage::from_pixel(total_w, total_h, bg_color);

            for (i, r) in results.iter().enumerate() {
                let row = (i as u32) / cols;
                let col = (i as u32) % cols;
                let x_local = cell_margin + col * (tile_size + cell_margin);
                let y_local = cell_margin + row * (tile_size + cell_margin);

                if has_gt {
                    // Panel 1: Conditioning
                    blit_upscaled(&mut grid_img, x_local, y_local, &r.cond_rgb, CANVAS_SIZE, CANVAS_SIZE, args.scale);

                    // Panel 2: Predicted
                    let x_pred = panel_w + cell_margin + x_local;
                    blit_upscaled(&mut grid_img, x_pred, y_local, &r.pred_rgb, CANVAS_SIZE, CANVAS_SIZE, args.scale);

                    // Panel 3: Ground Truth
                    if let Some(ref gt) = r.gt_rgb {
                        let x_gt = (panel_w + cell_margin) * 2 + x_local;
                        blit_upscaled(&mut grid_img, x_gt, y_local, gt, CANVAS_SIZE, CANVAS_SIZE, args.scale);
                    }
                } else {
                    blit_upscaled(&mut grid_img, x_local, y_local, &r.pred_rgb, CANVAS_SIZE, CANVAS_SIZE, args.scale);
                }
            }

            grid_img.save(&out_path)?;
            println!("  ✓ Grid image saved to: {}", out_path.display());
        }

        GridLayout::Grid => {
            let cell_w = tile_size;
            let cell_h = tile_size;
            let img_w = cell_margin + cols * (cell_w + cell_margin);
            let img_h = cell_margin + rows * (cell_h + cell_margin);
            let mut grid_img = image::RgbImage::from_pixel(img_w, img_h, bg_color);

            for (i, r) in results.iter().enumerate() {
                let row = (i as u32) / cols;
                let col = (i as u32) % cols;
                let x_pos = cell_margin + col * (cell_w + cell_margin);
                let y_pos = cell_margin + row * (cell_h + cell_margin);
                blit_upscaled(&mut grid_img, x_pos, y_pos, &r.pred_rgb, CANVAS_SIZE, CANVAS_SIZE, args.scale);
            }

            grid_img.save(&out_path)?;
            println!("  ✓ Grid image saved to: {}", out_path.display());
        }

        GridLayout::Pairs => {
            let pair_w = tile_size * 2 + 1;
            let pair_h = tile_size;
            let img_w = cell_margin + cols * (pair_w + cell_margin);
            let img_h = cell_margin + rows * (pair_h + cell_margin);
            let mut grid_img = image::RgbImage::from_pixel(img_w, img_h, bg_color);

            for (i, r) in results.iter().enumerate() {
                let row = (i as u32) / cols;
                let col = (i as u32) % cols;
                let x_pos = cell_margin + col * (pair_w + cell_margin);
                let y_pos = cell_margin + row * (pair_h + cell_margin);

                // 1. Model Predicted
                blit_upscaled(&mut grid_img, x_pos, y_pos, &r.pred_rgb, CANVAS_SIZE, CANVAS_SIZE, args.scale);

                // Divider line
                let x_div = x_pos + tile_size;
                for y in 0..tile_size {
                    grid_img.put_pixel(x_div, y_pos + y, divider_color);
                }

                // 2. Ground Truth (or Conditioning if no GT)
                let x_gt = x_div + 1;
                let target_buf = r.gt_rgb.as_ref().unwrap_or(&r.cond_rgb);
                blit_upscaled(&mut grid_img, x_gt, y_pos, target_buf, CANVAS_SIZE, CANVAS_SIZE, args.scale);
            }

            grid_img.save(&out_path)?;
            println!("  ✓ Grid image saved to: {}", out_path.display());
        }

        GridLayout::Triplets | GridLayout::Vertical | GridLayout::Wide | GridLayout::Auto => {
            let triplet_w = tile_size * 3 + 2;
            let triplet_h = tile_size;
            let img_w = cell_margin + cols * (triplet_w + cell_margin);
            let img_h = cell_margin + rows * (triplet_h + cell_margin);
            let mut grid_img = image::RgbImage::from_pixel(img_w, img_h, bg_color);

            for (i, r) in results.iter().enumerate() {
                let row = (i as u32) / cols;
                let col = (i as u32) % cols;
                let x_pos = cell_margin + col * (triplet_w + cell_margin);
                let y_pos = cell_margin + row * (triplet_h + cell_margin);

                // 1. Conditioning
                blit_upscaled(&mut grid_img, x_pos, y_pos, &r.cond_rgb, CANVAS_SIZE, CANVAS_SIZE, args.scale);

                // Divider line 1
                let x_div1 = x_pos + tile_size;
                for y in 0..tile_size {
                    grid_img.put_pixel(x_div1, y_pos + y, divider_color);
                }

                // 2. Model Predicted
                let x_pred = x_div1 + 1;
                blit_upscaled(&mut grid_img, x_pred, y_pos, &r.pred_rgb, CANVAS_SIZE, CANVAS_SIZE, args.scale);

                // Divider line 2
                let x_div2 = x_pred + tile_size;
                for y in 0..tile_size {
                    grid_img.put_pixel(x_div2, y_pos + y, divider_color);
                }

                // 3. Ground Truth
                let x_gt = x_div2 + 1;
                let target_buf = r.gt_rgb.as_ref().unwrap_or(&r.cond_rgb);
                blit_upscaled(&mut grid_img, x_gt, y_pos, target_buf, CANVAS_SIZE, CANVAS_SIZE, args.scale);
            }

            grid_img.save(&out_path)?;
            println!("  ✓ Grid image saved to: {}", out_path.display());
        }
    }

    // Optional manifest export
    if let Some(ref m_path) = args.manifest {
        let mut f = File::create(m_path)?;
        writeln!(f, "# Yumon Pixel Generator Eval Manifest - Pass {}", args.pass)?;
        writeln!(f, "# Checkpoint: {}", ckpt_dir.display())?;
        writeln!(f, "# Samples: {}", b)?;
        writeln!(f)?;
        for (i, r) in results.iter().enumerate() {
            writeln!(f, "Sample #{:03}:", i + 1)?;
            writeln!(f, "  Prompt: {}", r.prompt_text)?;
            if let Some(acc) = r.token_acc {
                writeln!(f, "  Token Accuracy: {:.2}%", acc)?;
            }
            if let Some(mae) = r.rgb_mae {
                writeln!(f, "  RGB MAE: {:.2}", mae)?;
            }
            if let Some(p) = r.psnr {
                writeln!(f, "  PSNR: {:.2} dB", p)?;
            }
            writeln!(f, "  Time: {} ms", r.gen_time_ms)?;
            writeln!(f)?;
        }
        println!("  ✓ Saved text manifest to: {}", m_path.display());
    }

    println!("  Total runtime: {:.2?}", t_start.elapsed());
    Ok(())
}
