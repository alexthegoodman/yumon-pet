//! Train Pass 4 Generative Upscaler (32x32 -> 256x256) using Burn WGPU Backend.
//!
//! Optimizations:
//! - Full in-RAM pre-computed planar float dataset (zero disk I/O during training)
//! - Non-blocking GPU execution: periodic asynchronous loss reporting instead of per-batch stalls
//! - Side-by-side visual preview image generation at each epoch

use anyhow::{Context, Result, ensure};
use burn::{
    backend::Autodiff,
    module::Module,
    nn::loss::{HuberLossConfig, Reduction},
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{Tensor, TensorData},
};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rand::{seq::SliceRandom, thread_rng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::Read,
    path::PathBuf,
    time::Instant,
};

use yumon_pet::vision::upscaler::{UpscalerModel, UpscalerModelConfig};

pub type TrainBackend = Autodiff<
    burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>,
>;

pub const UPSCALER_CACHE_MAGIC: &[u8; 8] = b"YUMON_UP";
pub const INPUT_GRID_SIZE: usize = 32;
pub const INPUT_CHW_LEN: usize = 3 * INPUT_GRID_SIZE * INPUT_GRID_SIZE; // 3,072
pub const TARGET_SIZE: usize = 256;
pub const TARGET_CHW_LEN: usize = 3 * TARGET_SIZE * TARGET_SIZE; // 196,608
pub const RECORD_SIZE: usize = INPUT_CHW_LEN + TARGET_CHW_LEN; // 199,680 bytes
pub const HEADER_SIZE: usize = 32;

#[derive(Parser, Debug)]
#[command(about = "Train Pass 4 Deep Residual Upscaler (32x32 -> 256x256)")]
struct Args {
    /// Optional path to JSON config file.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Path to binary cache file.
    #[arg(long, default_value = "training-cache/upscaler_256_samples.bin")]
    cache: PathBuf,

    /// Output checkpoint directory.
    #[arg(long, default_value = "checkpoints/yumon-upscaler-256")]
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
    #[arg(long, default_value_t = 32)]
    channels: usize,

    /// Fraction of samples used for validation.
    #[arg(long, default_value_t = 0.02)]
    val_fraction: f64,
}

#[derive(Serialize, Deserialize, Debug)]
struct UpscalerConfigFile {
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
    #[serde(default = "default_val_fraction")]
    val_fraction: f64,
}

fn default_cache() -> String { "training-cache/upscaler_256_samples.bin".into() }
fn default_out_dir() -> String { "checkpoints/yumon-upscaler-256".into() }
fn default_epochs() -> usize { 10 }
fn default_batch_size() -> usize { 8 }
fn default_lr() -> f64 { 3e-4 }
fn default_min_lr() -> f64 { 3e-5 }
fn default_channels() -> usize { 32 }
fn default_val_fraction() -> f64 { 0.02 }

#[derive(Serialize, Deserialize, Debug)]
struct UpscalerMetadata {
    epoch: usize,
    epochs: usize,
    train_loss: f32,
    val_loss: f32,
    channels: usize,
    timestamp: String,
}

/// Precomputed in-memory dataset of normalized planar f32 tensors.
struct InMemDataset {
    total_samples: usize,
    // [total_samples, INPUT_CHW_LEN]
    inputs: Vec<f32>,
    // [total_samples, TARGET_CHW_LEN]
    targets: Vec<f32>,
}

impl InMemDataset {
    fn load_from_cache(path: &PathBuf) -> Result<Self> {
        let t_load = Instant::now();
        let mut file = File::open(path)
            .with_context(|| format!("Failed to open cache file: {}", path.display()))?;

        let mut header = [0u8; HEADER_SIZE];
        file.read_exact(&mut header)?;

        ensure!(&header[0..8] == UPSCALER_CACHE_MAGIC, "Invalid cache magic header");
        let version = u32::from_le_bytes(header[8..12].try_into().unwrap());
        ensure!(
            version == 2,
            "Cache version {} is outdated. Please regenerate cache using `cargo run --release --bin cache_upscaler`.",
            version
        );

        let total_samples = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
        let record_size = u32::from_le_bytes(header[16..20].try_into().unwrap()) as usize;
        ensure!(
            record_size == RECORD_SIZE,
            "Record size mismatch: {} vs expected {}",
            record_size,
            RECORD_SIZE
        );

        println!("  Loading {} records (1.99 GB) from disk into RAM...", total_samples);
        let mut raw_bytes = vec![0u8; total_samples * RECORD_SIZE];
        file.read_exact(&mut raw_bytes)?;

        println!("  Precomputing planar f32 tensors in parallel across CPU cores...");
        let mut inputs = vec![0.0f32; total_samples * INPUT_CHW_LEN];
        let mut targets = vec![0.0f32; total_samples * TARGET_CHW_LEN];

        // Precompute lookup table: byte [0..255] -> f32 [-1.0, 1.0]
        let mut lut = [0.0f32; 256];
        for i in 0..256 {
            lut[i] = (i as f32 / 255.0) * 2.0 - 1.0;
        }

        // Parallel conversion with Rayon
        inputs
            .par_chunks_exact_mut(INPUT_CHW_LEN)
            .zip(targets.par_chunks_exact_mut(TARGET_CHW_LEN))
            .enumerate()
            .for_each(|(idx, (in_slice, target_slice))| {
                let rec_offset = idx * RECORD_SIZE;
                let in_bytes = &raw_bytes[rec_offset..rec_offset + INPUT_CHW_LEN];
                let target_bytes = &raw_bytes[rec_offset + INPUT_CHW_LEN..rec_offset + RECORD_SIZE];

                for i in 0..INPUT_CHW_LEN {
                    in_slice[i] = lut[in_bytes[i] as usize];
                }
                for i in 0..TARGET_CHW_LEN {
                    target_slice[i] = lut[target_bytes[i] as usize];
                }
            });

        println!("  ✓ Dataset ready in RAM in {:.2?} (Zero disk I/O during training!)", t_load.elapsed());
        Ok(Self {
            total_samples,
            inputs,
            targets,
        })
    }

    fn make_batch_tensors<B: Backend>(
        &self,
        indices: &[usize],
        device: &B::Device,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let b = indices.len();
        let mut flat_in = Vec::with_capacity(b * INPUT_CHW_LEN);
        let mut flat_target = Vec::with_capacity(b * TARGET_CHW_LEN);

        for &idx in indices {
            let in_start = idx * INPUT_CHW_LEN;
            flat_in.extend_from_slice(&self.inputs[in_start..in_start + INPUT_CHW_LEN]);

            let target_start = idx * TARGET_CHW_LEN;
            flat_target.extend_from_slice(&self.targets[target_start..target_start + TARGET_CHW_LEN]);
        }

        let in_tensor = Tensor::from_floats(
            TensorData::new(flat_in, [b, 3, INPUT_GRID_SIZE, INPUT_GRID_SIZE]),
            device,
        );
        let target_tensor = Tensor::from_floats(
            TensorData::new(flat_target, [b, 3, TARGET_SIZE, TARGET_SIZE]),
            device,
        );

        (in_tensor, target_tensor)
    }
}

/// Save visual comparison image: [Input 32x32 Upscaled | Model Predicted 256 | Ground Truth 256]
fn save_preview_image(
    input_slice: &[f32],
    pred_tensor: Tensor<TrainBackend, 4>,
    target_slice: &[f32],
    out_path: &PathBuf,
) -> Result<()> {
    let mut preview = image::RgbImage::new((TARGET_SIZE * 3) as u32, TARGET_SIZE as u32);
    let plane_32 = INPUT_GRID_SIZE * INPUT_GRID_SIZE;
    let plane_256 = TARGET_SIZE * TARGET_SIZE;

    // 1. Input 32x32 upscaled nearest-neighbor to 256x256
    for y in 0..TARGET_SIZE {
        let src_y = y / 8;
        for x in 0..TARGET_SIZE {
            let src_x = x / 8;
            let idx_32 = src_y * INPUT_GRID_SIZE + src_x;
            let r = (((input_slice[0 * plane_32 + idx_32] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
            let g = (((input_slice[1 * plane_32 + idx_32] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
            let b = (((input_slice[2 * plane_32 + idx_32] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
            preview.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }

    // 2. Model Predicted 256x256
    let pred_data: Vec<f32> = pred_tensor.into_data().convert::<f32>().to_vec().unwrap();
    for y in 0..TARGET_SIZE {
        for x in 0..TARGET_SIZE {
            let idx = y * TARGET_SIZE + x;
            let r = (((pred_data[0 * plane_256 + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
            let g = (((pred_data[1 * plane_256 + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
            let b = (((pred_data[2 * plane_256 + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
            preview.put_pixel((TARGET_SIZE + x) as u32, y as u32, image::Rgb([r, g, b]));
        }
    }

    // 3. Ground Truth 256x256
    for y in 0..TARGET_SIZE {
        for x in 0..TARGET_SIZE {
            let idx = y * TARGET_SIZE + x;
            let r = (((target_slice[0 * plane_256 + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
            let g = (((target_slice[1 * plane_256 + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
            let b = (((target_slice[2 * plane_256 + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
            preview.put_pixel((TARGET_SIZE * 2 + x) as u32, y as u32, image::Rgb([r, g, b]));
        }
    }

    preview.save(out_path)?;
    Ok(())
}

fn main() -> Result<()> {
    let mut args = Args::parse();
    if let Some(config_path) = &args.config {
        let content = fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;
        let cfg: UpscalerConfigFile = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {}", config_path.display()))?;

        args.cache = PathBuf::from(cfg.cache);
        args.out_dir = PathBuf::from(cfg.out_dir);
        args.epochs = cfg.epochs;
        args.batch_size = cfg.batch_size;
        args.lr = cfg.lr;
        args.min_lr = cfg.min_lr;
        args.channels = cfg.channels;
        args.val_fraction = cfg.val_fraction;
    }

    let device: <TrainBackend as Backend>::Device = Default::default();

    println!("==================================================");
    println!("     Yumon Pass 4 Upscaler Deep Residual Trainer");
    println!("==================================================");
    println!("  Cache binary    : {}", args.cache.display());
    println!("  Output directory: {}", args.out_dir.display());
    println!("  Epochs          : {}", args.epochs);
    println!("  Batch size      : {}", args.batch_size);
    println!("  Channels        : {}", args.channels);
    println!("  Learning rate   : {:.2e} -> {:.2e}", args.lr, args.min_lr);

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

    // 2. Initialize Model and Loss
    let config = UpscalerModelConfig::new()
        .with_channels(args.channels);

    let mut model: UpscalerModel<TrainBackend> = config.init(&device);
    let loss_fn = HuberLossConfig::new(1.0).init();

    let mut optimizer = AdamWConfig::new()
        .with_epsilon(1e-8)
        .with_weight_decay(1e-4)
        .init();

    let num_batches = train_indices.len() / args.batch_size;
    let t_train_start = Instant::now();

    // 3. Training Loop
    for epoch in 1..=args.epochs {
        let t_epoch = Instant::now();
        let mut epoch_indices = train_indices.clone();
        epoch_indices.shuffle(&mut rng);

        // Cosine learning rate decay
        let progress = (epoch - 1) as f64 / args.epochs.max(1) as f64;
        let lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + (progress * std::f64::consts::PI).cos());

        let pb = ProgressBar::new(num_batches as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [Epoch {msg}] [{bar:30.cyan/blue}] {pos}/{len} ({eta}) loss: {prefix}")
                .unwrap()
                .progress_chars("#>-"),
        );
        pb.set_message(format!("{}/{}", epoch, args.epochs));

        let mut batch_count = 0;
        let mut last_loss_val = 0.0f32;

        for chunk in epoch_indices.chunks(args.batch_size) {
            if chunk.len() < args.batch_size {
                continue;
            }

            // High-speed RAM batch slicing (Zero Disk I/O, Zero CPU pixel loops)
            let (input_tensor, target_tensor) = dataset.make_batch_tensors(chunk, &device);

            // Forward pass
            let pred = model.forward(input_tensor);

            let pixel_loss = loss_fn.forward(
                pred.clone(),
                target_tensor.clone(),
                Reduction::Mean,
            );

            // Gradient / edge reconstruction loss
            let edge_loss = gradient_loss(
                pred,
                target_tensor,
            );

            // Combined loss
            let loss = pixel_loss + edge_loss * 0.1;

            // Periodically inspect loss (every 25 batches) to avoid halting GPU pipeline on every step
            if batch_count % 25 == 0 {
                let loss_val: f32 = loss.clone().into_data().convert::<f32>().to_vec().unwrap()[0];
                last_loss_val = loss_val;
                pb.set_prefix(format!("{:.4}", loss_val));
            }
            batch_count += 1;

            // Backward pass & optimizer step
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(lr, model, grads);

            pb.inc(1);
        }

        pb.finish_with_message(format!("{}/{} [Loss: {:.4}]", epoch, args.epochs, last_loss_val));

        // 4. Fast Validation Loop
        let eval_count = val_indices.len().min(128);
        let mut val_loss_sum = 0.0f32;
        let mut val_batches = 0;

        for chunk in val_indices[..eval_count].chunks(args.batch_size) {
            let (input_tensor, target_tensor) = dataset.make_batch_tensors(chunk, &device);
            let pred = model.forward(input_tensor);
            let loss = loss_fn.forward(pred, target_tensor, Reduction::Mean);
            let loss_val: f32 = loss.into_data().convert::<f32>().to_vec().unwrap()[0];
            val_loss_sum += loss_val;
            val_batches += 1;
        }

        let avg_val_loss = if val_batches > 0 { val_loss_sum / val_batches as f32 } else { 0.0 };
        println!(
            "  Epoch {:>2}/{} ({:.1?}): Val Loss = {:.4} | LR = {:.2e}",
            epoch, args.epochs, t_epoch.elapsed(), avg_val_loss, lr
        );

        // 5. Save visual preview sample
        let preview_idx = val_indices[0];
        let (preview_in, _) = dataset.make_batch_tensors(&[preview_idx], &device);
        let preview_pred = model.forward(preview_in);

        let preview_path = args.out_dir.join(format!("preview_epoch_{:02}.png", epoch));
        let in_slice = &dataset.inputs[preview_idx * INPUT_CHW_LEN..(preview_idx + 1) * INPUT_CHW_LEN];
        let target_slice = &dataset.targets[preview_idx * TARGET_CHW_LEN..(preview_idx + 1) * TARGET_CHW_LEN];

        if let Err(e) = save_preview_image(in_slice, preview_pred, target_slice, &preview_path) {
            eprintln!("Warning: Failed to save preview image: {}", e);
        } else {
            println!("  [Preview] Saved comparison to {}", preview_path.display());
        }

        // 6. Save checkpoint
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        let model_path = args.out_dir.join("model");
        model.clone().save_file(&model_path, &recorder)?;

        let meta = UpscalerMetadata {
            epoch,
            epochs: args.epochs,
            train_loss: last_loss_val,
            val_loss: avg_val_loss,
            channels: args.channels,
            timestamp: chrono::Local::now().to_rfc3339(),
        };
        let meta_json = serde_json::to_string_pretty(&meta)?;
        fs::write(args.out_dir.join("metadata.json"), meta_json)?;
    }

    println!("\nTraining completed in {:.2?}. Model saved to {}", t_train_start.elapsed(), args.out_dir.display());
    Ok(())
}

fn gradient_loss<B: Backend>(
    pred: Tensor<B, 4>,
    target: Tensor<B, 4>,
) -> Tensor<B, 1> {
    let d = pred.dims();
    let td = target.dims();

    assert_eq!(d, td, "Prediction and target shapes must match");
    assert!(d[2] > 1 && d[3] > 1, "Image dimensions must exceed 1");

    // Horizontal differences: width dimension
    let pred_dx =
        pred.clone().slice([
            0..d[0], 0..d[1], 0..d[2], 1..d[3],
        ])
        - pred.clone().slice([
            0..d[0], 0..d[1], 0..d[2], 0..(d[3] - 1),
        ]);

    let target_dx =
        target.clone().slice([
            0..d[0], 0..d[1], 0..d[2], 1..d[3],
        ])
        - target.clone().slice([
            0..d[0], 0..d[1], 0..d[2], 0..(d[3] - 1),
        ]);

    // Vertical differences: height dimension
    let pred_dy =
        pred.clone().slice([
            0..d[0], 0..d[1], 1..d[2], 0..d[3],
        ])
        - pred.clone().slice([
            0..d[0], 0..d[1], 0..(d[2] - 1), 0..d[3],
        ]);

    let target_dy =
        target.clone().slice([
            0..d[0], 0..d[1], 1..d[2], 0..d[3],
        ])
        - target.clone().slice([
            0..d[0], 0..d[1], 0..(d[2] - 1), 0..d[3],
        ]);

    let loss_x = (pred_dx - target_dx).abs().mean();
    let loss_y = (pred_dy - target_dy).abs().mean();

    loss_x + loss_y
}