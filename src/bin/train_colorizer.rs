//! Train Pass 6 Colorizer (Full Zhang et al. ECCV 2016) using Burn WGPU Backend.
//!
//! Features:
//! - CIE $L^*a^*b^*$ formulation: Takes 1-channel 256x256 grayscale $L \in [-1.0, 1.0]$.
//! - Rebalanced multinomial cross-entropy over 313 color cluster bins (`COLOR_BINS_AB`).
//! - Temperature-annealed softmax ($T = 0.38$) visual evaluation at each epoch.
//! - Non-blocking GPU execution with interactive Ratatui dashboard.

use anyhow::{Context, Result, ensure};
use burn::{
    backend::Autodiff,
    module::Module,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{Int, Tensor, TensorData},
};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rand::{seq::SliceRandom, thread_rng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::{stdout, IsTerminal, Read},
    path::PathBuf,
    time::Instant,
};
use ratatui::{
    backend::CrosstermBackend,
    Terminal, TerminalOptions, Viewport,
};

use yumon_pet::vision::{
    chart::{render_colorizer_chart, ColorizerTrainingState},
    colorizer::{
        ColorizerModel, ColorizerModelConfig, COLOR_BINS_AB, NUM_COLOR_BINS,
        DEFAULT_TEMPERATURE, DEFAULT_CHROMA_BOOST, rebalanced_cross_entropy,
    },
};

pub type TrainBackend = Autodiff<
    burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>,
>;

pub const COLORIZER_CACHE_MAGIC: &[u8; 8] = b"YUMON_CL";
pub const INPUT_SIZE: usize = 256;
pub const INPUT_LEN: usize = INPUT_SIZE * INPUT_SIZE;     // 65,536 bytes
pub const TARGET_GRID: usize = 64;
pub const TARGET_LEN: usize = TARGET_GRID * TARGET_GRID;  // 4,096 indices
pub const RECORD_SIZE: usize = INPUT_LEN + (TARGET_LEN * 2); // 73,728 bytes

#[derive(Parser, Debug)]
#[command(about = "Train Pass 6 Colorizer (Full Zhang et al. 2016)")]
struct Args {
    /// Optional path to JSON config file.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Path to binary cache file.
    #[arg(long, default_value = "training-cache/colorizer_256.bin")]
    cache: PathBuf,

    /// Output checkpoint directory.
    #[arg(long, default_value = "checkpoints/yumon-colorizer-256")]
    out_dir: PathBuf,

    /// Number of training epochs.
    #[arg(long, default_value_t = 10)]
    epochs: usize,

    /// Batch size for training.
    #[arg(long, default_value_t = 8)]
    batch_size: usize,

    /// Initial learning rate.
    #[arg(long, default_value_t = 3e-4)]
    lr: f64,

    /// Final learning rate (cosine decay).
    #[arg(long, default_value_t = 3e-5)]
    min_lr: f64,

    /// Feature channels in the residual backbone.
    #[arg(long, default_value_t = 64)]
    channels: usize,

    /// Number of residual blocks in backbone.
    #[arg(long, default_value_t = 5)]
    num_res_blocks: usize,

    /// Fraction of samples used for validation.
    #[arg(long, default_value_t = 0.02)]
    val_fraction: f64,

    /// Inference temperature for annealed mean visual preview.
    #[arg(long, default_value_t = DEFAULT_TEMPERATURE)]
    temperature: f32,

    /// Saturation chroma boost multiplier to counteract expectation contraction (default 1.8).
    #[arg(long, default_value_t = DEFAULT_CHROMA_BOOST)]
    chroma_boost: f32,

    /// Disable interactive Ratatui TUI dashboard.
    #[arg(long, default_value_t = false)]
    no_tui: bool,
}

#[derive(Serialize, Deserialize, Debug)]
struct ColorizerConfigFile {
    #[serde(default = "default_cache")]
    cache: String,
    #[serde(default = "default_out_dir")]
    out_dir: String,
    #[serde(default = "default_epochs")]
    epochs: usize,
    #[serde(default = "default_batch_size")]
    batch_size: usize,
    #[serde(default = "default_lr")]
    lr: f64,
    #[serde(default = "default_min_lr")]
    min_lr: f64,
    #[serde(default = "default_channels")]
    channels: usize,
    #[serde(default = "default_num_res_blocks")]
    num_res_blocks: usize,
    #[serde(default = "default_val_fraction")]
    val_fraction: f64,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_chroma_boost")]
    chroma_boost: f32,
    #[serde(default)]
    no_tui: bool,
}

fn default_cache() -> String { "training-cache/colorizer_256.bin".into() }
fn default_out_dir() -> String { "checkpoints/yumon-colorizer-256".into() }
fn default_epochs() -> usize { 10 }
fn default_batch_size() -> usize { 8 }
fn default_lr() -> f64 { 3e-4 }
fn default_min_lr() -> f64 { 3e-5 }
fn default_channels() -> usize { 64 }
fn default_num_res_blocks() -> usize { 5 }
fn default_val_fraction() -> f64 { 0.02 }
fn default_temperature() -> f32 { DEFAULT_TEMPERATURE }
fn default_chroma_boost() -> f32 { DEFAULT_CHROMA_BOOST }

#[derive(Serialize, Deserialize, Debug)]
struct ColorizerMetadata {
    epoch: usize,
    epochs: usize,
    train_loss: f32,
    val_loss: f32,
    train_acc: f32,
    train_top5: f32,
    val_acc: f32,
    val_top5: f32,
    channels: usize,
    temperature: f32,
    chroma_boost: f32,
    timestamp: String,
}

/// Precomputed in-memory dataset of normalized planar f32 grayscale tensors and u16 target bin IDs.
struct InMemDataset {
    total_samples: usize,
    weights: Vec<f32>,
    inputs_f32: Vec<f32>,
    raw_l_bytes: Vec<u8>,
    targets: Vec<i32>,
}

impl InMemDataset {
    fn load_from_cache(path: &PathBuf) -> Result<Self> {
        let t_load = Instant::now();
        let mut file = File::open(path)
            .with_context(|| format!("Failed to open cache file: {}", path.display()))?;

        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        ensure!(&magic == COLORIZER_CACHE_MAGIC, "Invalid cache magic header");

        let mut ver_bytes = [0u8; 4];
        file.read_exact(&mut ver_bytes)?;
        let version = u32::from_le_bytes(ver_bytes);
        ensure!(version == 1, "Unsupported cache version: {}", version);

        let mut counts_bytes = [0u8; 20];
        file.read_exact(&mut counts_bytes)?;
        let total_samples = u32::from_le_bytes(counts_bytes[0..4].try_into().unwrap()) as usize;
        let record_size = u32::from_le_bytes(counts_bytes[4..8].try_into().unwrap()) as usize;
        let in_size = u32::from_le_bytes(counts_bytes[8..12].try_into().unwrap()) as usize;
        let tgt_grid = u32::from_le_bytes(counts_bytes[12..16].try_into().unwrap()) as usize;

        ensure!(
            record_size == RECORD_SIZE && in_size == INPUT_SIZE && tgt_grid == TARGET_GRID,
            "Cache dimensions mismatch"
        );

        // Read 313 weights
        let mut weights_bytes = vec![0u8; NUM_COLOR_BINS * 4];
        file.read_exact(&mut weights_bytes)?;
        let mut weights = vec![0.0f32; NUM_COLOR_BINS];
        for i in 0..NUM_COLOR_BINS {
            weights[i] = f32::from_le_bytes(weights_bytes[i*4..i*4+4].try_into().unwrap());
        }

        println!(
            "  Cache metadata: {} samples, 1x{}x{} input -> {}x{} target (313 color classes)",
            total_samples, in_size, in_size, tgt_grid, tgt_grid
        );
        println!(
            "  Loading {} records ({:.2} MB) from disk into RAM...",
            total_samples,
            (total_samples * RECORD_SIZE) as f64 / 1e6
        );

        let mut raw_bytes = vec![0u8; total_samples * RECORD_SIZE];
        file.read_exact(&mut raw_bytes)?;

        println!("  Precomputing planar f32 tensors in parallel across CPU cores...");
        let mut inputs_f32 = vec![0.0f32; total_samples * INPUT_LEN];
        let mut raw_l_bytes = vec![0u8; total_samples * INPUT_LEN];
        let mut targets = vec![0i32; total_samples * TARGET_LEN];

        // Lookup table: byte [0..255] maps L* [0..100] to normalized range [-0.5, 0.5] from Zhang et al.
        let mut lut = [0.0f32; 256];
        for i in 0..256 {
            let l_star = (i as f32 / 255.0) * 100.0;
            lut[i] = (l_star - 50.0) / 100.0;
        }

        inputs_f32
            .par_chunks_exact_mut(INPUT_LEN)
            .zip(raw_l_bytes.par_chunks_exact_mut(INPUT_LEN))
            .zip(targets.par_chunks_exact_mut(TARGET_LEN))
            .enumerate()
            .for_each(|(idx, ((in_slice, raw_l_slice), target_slice))| {
                let rec_offset = idx * RECORD_SIZE;
                let in_bytes = &raw_bytes[rec_offset..rec_offset + INPUT_LEN];
                let tgt_bytes = &raw_bytes[rec_offset + INPUT_LEN..rec_offset + RECORD_SIZE];

                for i in 0..INPUT_LEN {
                    raw_l_slice[i] = in_bytes[i];
                    in_slice[i] = lut[in_bytes[i] as usize];
                }

                for i in 0..TARGET_LEN {
                    let bin_id = u16::from_le_bytes([tgt_bytes[i * 2], tgt_bytes[i * 2 + 1]]);
                    target_slice[i] = bin_id as i32;
                }
            });

        println!(
            "  ✓ Colorizer dataset ready in RAM in {:.2?} (Zero disk I/O during training!)",
            t_load.elapsed()
        );

        Ok(Self {
            total_samples,
            weights,
            inputs_f32,
            raw_l_bytes,
            targets,
        })
    }

    fn make_batch_tensors<B: Backend>(
        &self,
        indices: &[usize],
        device: &B::Device,
    ) -> (Tensor<B, 4>, Tensor<B, 2, Int>) {
        let b = indices.len();
        let mut flat_in = Vec::with_capacity(b * INPUT_LEN);
        let mut flat_tgt = Vec::with_capacity(b * TARGET_LEN);

        for &idx in indices {
            let in_start = idx * INPUT_LEN;
            flat_in.extend_from_slice(&self.inputs_f32[in_start..in_start + INPUT_LEN]);

            let tgt_start = idx * TARGET_LEN;
            flat_tgt.extend_from_slice(&self.targets[tgt_start..tgt_start + TARGET_LEN]);
        }

        let in_tensor = Tensor::from_floats(
            TensorData::new(flat_in, [b, 1, INPUT_SIZE, INPUT_SIZE]),
            device,
        );
        let tgt_tensor = Tensor::from_ints(
            TensorData::new(flat_tgt, [b, TARGET_LEN]),
            device,
        );

        (in_tensor, tgt_tensor)
    }
}

/// Save visual comparison grid: Input Grayscale L | Predicted Colorized RGB | Target Colorized RGB.
fn save_colorizer_preview_grid(
    dataset: &InMemDataset,
    indices: &[usize],
    model: &ColorizerModel<TrainBackend>,
    device: &<TrainBackend as Backend>::Device,
    temperature: f32,
    chroma_boost: f32,
    out_path: &PathBuf,
) -> Result<()> {
    let b = indices.len();
    let cols = 4u32.min(b as u32);
    let rows = ((b as u32) + cols - 1) / cols;

    let triplet_w = (INPUT_SIZE * 3 + 2) as u32; // 770 px
    let triplet_h = INPUT_SIZE as u32;           // 256 px
    let cell_margin = 8u32;
    let bg_color = image::Rgb([18, 18, 26]);

    let img_w = cell_margin + cols * (triplet_w + cell_margin);
    let img_h = cell_margin + rows * (triplet_h + cell_margin);

    let mut grid_img = image::RgbImage::from_pixel(img_w, img_h, bg_color);
    let pts = &COLOR_BINS_AB;

    for (i, &idx) in indices.iter().enumerate() {
        let r = (i as u32) / cols;
        let c = (i as u32) % cols;
        let x_offset = cell_margin + c * (triplet_w + cell_margin);
        let y_offset = cell_margin + r * (triplet_h + cell_margin);

        let in_slice = &dataset.inputs_f32[idx * INPUT_LEN..(idx + 1) * INPUT_LEN];
        let raw_l_slice = &dataset.raw_l_bytes[idx * INPUT_LEN..(idx + 1) * INPUT_LEN];
        let target_slice = &dataset.targets[idx * TARGET_LEN..(idx + 1) * TARGET_LEN];

        // 1. Input Grayscale
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                let p_idx = y * INPUT_SIZE + x;
                let val = raw_l_slice[p_idx];
                grid_img.put_pixel(x_offset + x as u32, y_offset + y as u32, image::Rgb([val, val, val]));
            }
        }

        // Separator 1
        for y in 0..INPUT_SIZE {
            grid_img.put_pixel(x_offset + INPUT_SIZE as u32, y_offset + y as u32, image::Rgb([40, 40, 50]));
        }

        // 2. Model Predicted Colorized (Annealed Mean Softmax + Chroma Boost)
        let single_in = Tensor::from_floats(
            TensorData::new(in_slice.to_vec(), [1, 1, INPUT_SIZE, INPUT_SIZE]),
            device,
        );
        let (pred_a, pred_b) = model.predict_ab_annealed(single_in, temperature, chroma_boost);
        let pred_img = yumon_pet::vision::colorizer::colorize_fused_rgb(raw_l_slice, &pred_a, &pred_b, INPUT_SIZE);

        let x_pred_base = x_offset + INPUT_SIZE as u32 + 1;
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                let p = pred_img.get_pixel(x as u32, y as u32);
                grid_img.put_pixel(x_pred_base + x as u32, y_offset + y as u32, *p);
            }
        }

        // Separator 2
        for y in 0..INPUT_SIZE {
            grid_img.put_pixel(x_pred_base + INPUT_SIZE as u32, y_offset + y as u32, image::Rgb([40, 40, 50]));
        }

        // 3. Ground Truth Target Bins reconstructed
        let mut target_a = vec![0.0f32; TARGET_LEN];
        let mut target_b = vec![0.0f32; TARGET_LEN];
        for k in 0..TARGET_LEN {
            let bin = target_slice[k] as usize;
            target_a[k] = pts[bin][0];
            target_b[k] = pts[bin][1];
        }
        let target_img = yumon_pet::vision::colorizer::colorize_fused_rgb(raw_l_slice, &target_a, &target_b, INPUT_SIZE);

        let x_target_base = x_pred_base + INPUT_SIZE as u32 + 1;
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                let p = target_img.get_pixel(x as u32, y as u32);
                grid_img.put_pixel(x_target_base + x as u32, y_offset + y as u32, *p);
            }
        }
    }

    grid_img.save(out_path)?;
    Ok(())
}

fn main() -> Result<()> {
    let mut args = Args::parse();
    if let Some(config_path) = &args.config {
        let content = fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;
        let cfg: ColorizerConfigFile = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {}", config_path.display()))?;

        args.cache = PathBuf::from(cfg.cache);
        args.out_dir = PathBuf::from(cfg.out_dir);
        args.epochs = cfg.epochs;
        args.batch_size = cfg.batch_size;
        args.lr = cfg.lr;
        args.min_lr = cfg.min_lr;
        args.channels = cfg.channels;
        args.num_res_blocks = cfg.num_res_blocks;
        args.val_fraction = cfg.val_fraction;
        args.temperature = cfg.temperature;
        args.chroma_boost = cfg.chroma_boost;
        args.no_tui = cfg.no_tui;
    }

    let device: <TrainBackend as Backend>::Device = Default::default();

    println!("==================================================");
    println!("     Yumon Pass 6 Colorizer Trainer (Zhang 2016)");
    println!("==================================================");
    println!("  Cache binary    : {}", args.cache.display());
    println!("  Output directory: {}", args.out_dir.display());
    println!("  Epochs          : {}", args.epochs);
    println!("  Batch size      : {}", args.batch_size);
    println!("  Channels        : {}", args.channels);
    println!("  ResBlocks       : {}", args.num_res_blocks);
    println!("  Learning rate   : {:.2e} -> {:.2e}", args.lr, args.min_lr);
    println!("  Temperature (T) : {:.2}", args.temperature);
    println!("  Chroma boost    : {:.2}", args.chroma_boost);

    fs::create_dir_all(&args.out_dir)?;

    // 1. Load dataset into RAM
    let dataset = InMemDataset::load_from_cache(&args.cache)?;
    let total_samples = dataset.total_samples;
    println!("  Total cache samples: {}", total_samples);

    let val_count = ((total_samples as f64) * args.val_fraction).max(10.0) as usize;
    let train_count = total_samples.saturating_sub(val_count);

    let mut indices: Vec<usize> = (0..total_samples).collect();
    let mut rng = thread_rng();
    indices.shuffle(&mut rng);

    let val_indices = indices[train_count..].to_vec();
    let train_indices = indices[..train_count].to_vec();

    println!("  Train samples   : {}", train_indices.len());
    println!("  Val samples     : {}", val_indices.len());

    // 2. Weights tensor on GPU
    let weights_tensor: Tensor<TrainBackend, 1> = Tensor::from_floats(
        TensorData::new(dataset.weights.clone(), [NUM_COLOR_BINS]),
        &device,
    );

    // 3. Initialize Model and Optimizer
    let config = ColorizerModelConfig::new()
        .with_scales_from(INPUT_SIZE)
        .with_channels(args.channels)
        .with_num_res_blocks(args.num_res_blocks);

    let mut model: ColorizerModel<TrainBackend> = config.init(&device);

    let mut optimizer = AdamWConfig::new()
        .with_epsilon(1e-8)
        .with_weight_decay(1e-4)
        .init();

    let num_batches = train_indices.len() / args.batch_size;
    let t_train_start = Instant::now();

    let mut tui_state = ColorizerTrainingState::new(args.epochs, num_batches);

    let mut terminal = if !args.no_tui && stdout().is_terminal() {
        let backend = CrosstermBackend::new(stdout());
        match Terminal::with_options(
            backend,
            TerminalOptions {
                viewport: Viewport::Inline(24),
            },
        ) {
            Ok(term) => Some(term),
            Err(e) => {
                eprintln!("Warning: Failed to initialize inline TUI terminal: {e}. Falling back to stdout.");
                None
            }
        }
    } else {
        None
    };

    // 4. Training Loop
    for epoch in 1..=args.epochs {
        let t_epoch = Instant::now();
        let mut epoch_indices = train_indices.clone();
        epoch_indices.shuffle(&mut rng);

        // Cosine learning rate decay
        let progress = (epoch - 1) as f64 / args.epochs.max(1) as f64;
        let lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + (progress * std::f64::consts::PI).cos());

        let pb = if terminal.is_none() {
            let bar = ProgressBar::new(num_batches as u64);
            bar.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Epoch {msg}] [{bar:30.magenta/blue}] {pos}/{len} ({eta}) loss: {prefix}")
                    .unwrap()
                    .progress_chars("#>-"),
            );
            bar.set_message(format!("{}/{}", epoch, args.epochs));
            Some(bar)
        } else {
            None
        };

        let mut batch_count = 0;
        let mut last_loss_val = 0.0f32;
        let mut last_top1_val = 0.0f32;
        let mut last_top5_val = 0.0f32;
        let mut last_delta_e_val = 0.0f32;

        for chunk in epoch_indices.chunks(args.batch_size) {
            if chunk.len() < args.batch_size {
                continue;
            }

            let should_inspect = (batch_count + 1) % 10 == 0 || (batch_count + 1) == num_batches;

            // Zero-copy RAM batch tensor creation
            let (input_tensor, target_tensor) = dataset.make_batch_tensors(chunk, &device);

            // Forward pass: [B, 313, 64, 64]
            let logits = model.forward(input_tensor);

            // Rebalanced cross-entropy loss properly normalized by batch target weights
            let loss = rebalanced_cross_entropy(
                logits.clone(),
                target_tensor.clone(),
                weights_tensor.clone(),
            );

            // Compute Top-1, Top-5, and ΔE metrics on inspection batches
            let (top1_val, top5_val, delta_e_val) = if should_inspect {
                let [b, q, h, w] = logits.dims();
                let spatial_len = h * w;
                let flat_logits_data: Vec<f32> = logits
                    .swap_dims(1, 2)
                    .swap_dims(2, 3)
                    .into_data()
                    .convert::<f32>()
                    .to_vec()
                    .unwrap();
                let target_data: Vec<i32> = target_tensor.clone().into_data().convert::<i32>().to_vec().unwrap();

                let mut top1_correct = 0usize;
                let mut top5_correct = 0usize;
                let mut total_delta_e = 0.0f32;
                let pts = &COLOR_BINS_AB;
                let total_pixels = b * spatial_len;

                for p in 0..total_pixels {
                    let tgt_bin = target_data[p] as usize;
                    let tgt_a = pts[tgt_bin][0];
                    let tgt_b = pts[tgt_bin][1];

                    let logit_offset = p * q;
                    let tgt_logit = flat_logits_data[logit_offset + tgt_bin];

                    let mut count_higher = 0;
                    let mut best_bin = 0;
                    let mut max_l = f32::NEG_INFINITY;

                    for bin in 0..q {
                        let l = flat_logits_data[logit_offset + bin];
                        if l > max_l {
                            max_l = l;
                            best_bin = bin;
                        }
                        if l > tgt_logit {
                            count_higher += 1;
                        }
                    }

                    if best_bin == tgt_bin {
                        top1_correct += 1;
                    }
                    if count_higher < 5 {
                        top5_correct += 1;
                    }

                    let pred_a = pts[best_bin][0];
                    let pred_b = pts[best_bin][1];
                    let de = ((pred_a - tgt_a).powi(2) + (pred_b - tgt_b).powi(2)).sqrt();
                    total_delta_e += de;
                }

                let t1 = (top1_correct as f32 / total_pixels as f32) * 100.0;
                let t5 = (top5_correct as f32 / total_pixels as f32) * 100.0;
                let de = total_delta_e / total_pixels as f32;
                (t1, t5, de)
            } else {
                (last_top1_val, last_top5_val, last_delta_e_val)
            };

            batch_count += 1;

            if should_inspect {
                let loss_val: f32 = loss.clone().into_data().convert::<f32>().to_vec().unwrap()[0];
                last_loss_val = loss_val;
                last_top1_val = top1_val;
                last_top5_val = top5_val;
                last_delta_e_val = delta_e_val;

                let global_step = (epoch - 1) * num_batches + batch_count;
                tui_state.epoch = epoch;
                tui_state.batch = batch_count;
                tui_state.global_step = global_step;
                tui_state.current_lr = lr;
                tui_state.current_loss = loss_val;
                tui_state.current_acc = top1_val;
                tui_state.current_top5_acc = top5_val;
                tui_state.current_ab_error = delta_e_val;

                tui_state.loss_history.push((global_step as f64, loss_val as f64));
                tui_state.acc_history.push((global_step as f64, top1_val as f64));
                tui_state.top5_history.push((global_step as f64, top5_val as f64));
                tui_state.lr_history.push((global_step as f64, lr));

                if let Some(term) = terminal.as_mut() {
                    let _ = term.draw(|f| render_colorizer_chart(f, &tui_state));
                } else if let Some(ref p) = pb {
                    p.set_prefix(format!("{:.4} (T1: {:.1}%, T5: {:.1}%, ΔE: {:.1})", loss_val, top1_val, top5_val, delta_e_val));
                }
            }

            // Backward pass & optimizer step
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(lr, model, grads);

            if let Some(ref p) = pb {
                p.inc(1);
            }
        }

        if let Some(p) = pb {
            p.finish_with_message(format!(
                "{}/{} [Loss: {:.4} | T1: {:.1}% | T5: {:.1}% | ΔE: {:.1}]",
                epoch, args.epochs, last_loss_val, last_top1_val, last_top5_val, last_delta_e_val
            ));
        }

        // 5. Validation Loop
        let eval_count = val_indices.len().min(128);
        let mut val_loss_sum = 0.0f32;
        let mut val_top1_sum = 0.0f32;
        let mut val_top5_sum = 0.0f32;
        let mut val_de_sum = 0.0f32;
        let mut val_batches = 0;

        for chunk in val_indices[..eval_count].chunks(args.batch_size) {
            let (input_tensor, target_tensor) = dataset.make_batch_tensors(chunk, &device);
            let logits = model.forward(input_tensor);
            let loss = rebalanced_cross_entropy(
                logits.clone(),
                target_tensor.clone(),
                weights_tensor.clone(),
            );
            let loss_val: f32 = loss.into_data().convert::<f32>().to_vec().unwrap()[0];
            val_loss_sum += loss_val;

            let [b, q, h, w] = logits.dims();
            let spatial_len = h * w;
            let flat_logits_data: Vec<f32> = logits
                .swap_dims(1, 2)
                .swap_dims(2, 3)
                .into_data()
                .convert::<f32>()
                .to_vec()
                .unwrap();
            let target_data: Vec<i32> = target_tensor.into_data().convert::<i32>().to_vec().unwrap();

            let mut top1_correct = 0usize;
            let mut top5_correct = 0usize;
            let mut total_delta_e = 0.0f32;
            let pts = &COLOR_BINS_AB;
            let total_pixels = b * spatial_len;

            for p in 0..total_pixels {
                let tgt_bin = target_data[p] as usize;
                let tgt_a = pts[tgt_bin][0];
                let tgt_b = pts[tgt_bin][1];

                let logit_offset = p * q;
                let tgt_logit = flat_logits_data[logit_offset + tgt_bin];

                let mut count_higher = 0;
                let mut best_bin = 0;
                let mut max_l = f32::NEG_INFINITY;

                for bin in 0..q {
                    let l = flat_logits_data[logit_offset + bin];
                    if l > max_l {
                        max_l = l;
                        best_bin = bin;
                    }
                    if l > tgt_logit {
                        count_higher += 1;
                    }
                }

                if best_bin == tgt_bin {
                    top1_correct += 1;
                }
                if count_higher < 5 {
                    top5_correct += 1;
                }

                let pred_a = pts[best_bin][0];
                let pred_b = pts[best_bin][1];
                let de = ((pred_a - tgt_a).powi(2) + (pred_b - tgt_b).powi(2)).sqrt();
                total_delta_e += de;
            }

            let t1 = (top1_correct as f32 / total_pixels as f32) * 100.0;
            let t5 = (top5_correct as f32 / total_pixels as f32) * 100.0;
            let de = total_delta_e / total_pixels as f32;

            val_top1_sum += t1;
            val_top5_sum += t5;
            val_de_sum += de;
            val_batches += 1;
        }

        let avg_val_loss = if val_batches > 0 { val_loss_sum / val_batches as f32 } else { 0.0 };
        let avg_val_top1 = if val_batches > 0 { val_top1_sum / val_batches as f32 } else { 0.0 };
        let avg_val_top5 = if val_batches > 0 { val_top5_sum / val_batches as f32 } else { 0.0 };
        let avg_val_de = if val_batches > 0 { val_de_sum / val_batches as f32 } else { 0.0 };

        let global_step = epoch * num_batches;
        tui_state.current_val_loss = Some(avg_val_loss);
        tui_state.current_val_acc = Some(avg_val_top5);
        tui_state.val_loss_history.push((global_step as f64, avg_val_loss as f64));
        let _ = tui_state.save_csv(&args.out_dir.join("loss_history.csv"));

        if let Some(term) = terminal.as_mut() {
            let _ = term.draw(|f| render_colorizer_chart(f, &tui_state));
        } else {
            println!(
                "  Epoch {:>2}/{} ({:.1?}): Val Loss = {:.4} | Top-1 = {:.1}% | Top-5 = {:.1}% | ΔE = {:.1} | LR = {:.2e}",
                epoch, args.epochs, t_epoch.elapsed(), avg_val_loss, avg_val_top1, avg_val_top5, avg_val_de, lr
            );
        }

        // 6. Save visual comparison preview grid (16 samples)
        let preview_count = 16.min(val_indices.len());
        let preview_indices = &val_indices[..preview_count];
        let preview_path = args.out_dir.join(format!("preview_epoch_{:02}.png", epoch));

        if let Err(e) = save_colorizer_preview_grid(
            &dataset,
            preview_indices,
            &model,
            &device,
            args.temperature,
            args.chroma_boost,
            &preview_path,
        ) {
            eprintln!("Warning: Failed to save preview image grid: {}", e);
        } else {
            tui_state.last_preview_path = Some(preview_path.to_string_lossy().to_string());
            if terminal.is_none() {
                println!("  [Preview] Saved 16-sample comparison grid to {}", preview_path.display());
            }
        }

        // 7. Save checkpoint
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        let model_path = args.out_dir.join("model");
        model.clone().save_file(&model_path, &recorder)?;

        let meta = ColorizerMetadata {
            epoch,
            epochs: args.epochs,
            train_loss: last_loss_val,
            val_loss: avg_val_loss,
            train_acc: last_top1_val,
            train_top5: last_top5_val,
            val_acc: avg_val_top1,
            val_top5: avg_val_top5,
            channels: args.channels,
            temperature: args.temperature,
            chroma_boost: args.chroma_boost,
            timestamp: chrono::Local::now().to_rfc3339(),
        };
        let meta_json = serde_json::to_string_pretty(&meta)?;
        fs::write(args.out_dir.join("metadata.json"), meta_json)?;
    }

    println!("\nTraining completed in {:.2?}. Model saved to {}", t_train_start.elapsed(), args.out_dir.display());
    Ok(())
}
