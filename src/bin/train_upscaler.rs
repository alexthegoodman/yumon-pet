//! Train Pass 4 Generative Upscaler (32x32 -> 256x256) using Burn WGPU Backend.
//!
//! Features:
//! - Direct high-speed random-seek binary cache loading
//! - Composite loss (Charbonnier + Sobel/Laplacian High-Frequency Gradient Loss)
//! - Generates side-by-side visual previews at each epoch:
//!   [Coarse Input (Nearest 256) | Model Super-Resolved | Ground Truth 256]

use anyhow::{Context, Result, ensure};
use burn::{
    backend::{
        autodiff::checkpoint::strategy::BalancedCheckpointing,
        Autodiff,
    },
    module::Module,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{Tensor, TensorData},
};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rand::{seq::SliceRandom, thread_rng};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
    time::Instant,
};

use yumon_pet::vision::{
    paint_action::palette_to_rgb,
    upscaler::{UpscalerModel, UpscalerModelConfig},
};

pub type TrainBackend = Autodiff<
    burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>,
    BalancedCheckpointing,
>;

pub const UPSCALER_CACHE_MAGIC: &[u8; 8] = b"YUMON_UP";
pub const INPUT_GRID_SIZE: usize = 32;
pub const INPUT_TOKENS_LEN: usize = 1024;
pub const TARGET_SIZE: usize = 256;
pub const TARGET_RGB_LEN: usize = TARGET_SIZE * TARGET_SIZE * 3;
pub const RECORD_SIZE: usize = INPUT_TOKENS_LEN + TARGET_RGB_LEN;
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

    /// Number of residual blocks in the backbone.
    #[arg(long, default_value_t = 6)]
    res_blocks: usize,

    /// Feature channels in the residual backbone.
    #[arg(long, default_value_t = 64)]
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
    #[serde(default = "default_res_blocks")]
    res_blocks: usize,
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
fn default_res_blocks() -> usize { 6 }
fn default_channels() -> usize { 64 }
fn default_val_fraction() -> f64 { 0.02 }

#[derive(Serialize, Deserialize, Debug)]
struct UpscalerMetadata {
    epoch: usize,
    epochs: usize,
    train_loss: f32,
    val_loss: f32,
    channels: usize,
    num_res_blocks: usize,
    timestamp: String,
}

struct DatasetReader {
    file: File,
    total_records: usize,
}

impl DatasetReader {
    fn open(path: &PathBuf) -> Result<Self> {
        let mut file = File::open(path)
            .with_context(|| format!("Failed to open cache file: {}", path.display()))?;

        let mut header = [0u8; HEADER_SIZE];
        file.read_exact(&mut header)?;

        ensure!(&header[0..8] == UPSCALER_CACHE_MAGIC, "Invalid cache magic header");
        let version = u32::from_le_bytes(header[8..12].try_into().unwrap());
        ensure!(version == 1, "Unsupported cache version: {}", version);

        let total_records = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
        let record_size = u32::from_le_bytes(header[16..20].try_into().unwrap()) as usize;
        ensure!(record_size == RECORD_SIZE, "Mismatch in record size: {} vs expected {}", record_size, RECORD_SIZE);

        Ok(Self { file, total_records })
    }

    fn read_record(&mut self, index: usize) -> Result<(Vec<u8>, Vec<u8>)> {
        let offset = HEADER_SIZE as u64 + (index as u64) * (RECORD_SIZE as u64);
        self.file.seek(SeekFrom::Start(offset))?;

        let mut tokens = vec![0u8; INPUT_TOKENS_LEN];
        self.file.read_exact(&mut tokens)?;

        let mut target_rgb = vec![0u8; TARGET_RGB_LEN];
        self.file.read_exact(&mut target_rgb)?;

        Ok((tokens, target_rgb))
    }
}

/// Convert batch of tokens [B, 1024] into normalized float tensor [B, 3, 32, 32] in [-1.0, 1.0].
fn tokens_batch_to_tensor<B: Backend>(tokens_batch: &[Vec<u8>], device: &B::Device) -> Tensor<B, 4> {
    let batch_size = tokens_batch.len();
    let plane = INPUT_GRID_SIZE * INPUT_GRID_SIZE;
    let mut flat = vec![0.0f32; batch_size * 3 * plane];

    for (b, tokens) in tokens_batch.iter().enumerate() {
        let b_offset = b * 3 * plane;
        for i in 0..tokens.len() {
            let [r, g, b_val] = palette_to_rgb(tokens[i]);
            // Planar CHW
            flat[b_offset + 0 * plane + i] = (r as f32 / 255.0) * 2.0 - 1.0;
            flat[b_offset + 1 * plane + i] = (g as f32 / 255.0) * 2.0 - 1.0;
            flat[b_offset + 2 * plane + i] = (b_val as f32 / 255.0) * 2.0 - 1.0;
        }
    }

    Tensor::from_floats(
        TensorData::new(flat, [batch_size, 3, INPUT_GRID_SIZE, INPUT_GRID_SIZE]),
        device,
    )
}

/// Convert batch of target raw RGB [B, 196608] into normalized float tensor [B, 3, 256, 256] in [-1.0, 1.0].
fn target_batch_to_tensor<B: Backend>(target_batch: &[Vec<u8>], device: &B::Device) -> Tensor<B, 4> {
    let batch_size = target_batch.len();
    let plane = TARGET_SIZE * TARGET_SIZE;
    let mut flat = vec![0.0f32; batch_size * 3 * plane];

    for (b, rgb) in target_batch.iter().enumerate() {
        let b_offset = b * 3 * plane;
        for y in 0..TARGET_SIZE {
            for x in 0..TARGET_SIZE {
                let hwc_idx = (y * TARGET_SIZE + x) * 3;
                let pix_idx = y * TARGET_SIZE + x;
                flat[b_offset + 0 * plane + pix_idx] = (rgb[hwc_idx] as f32 / 255.0) * 2.0 - 1.0;
                flat[b_offset + 1 * plane + pix_idx] = (rgb[hwc_idx + 1] as f32 / 255.0) * 2.0 - 1.0;
                flat[b_offset + 2 * plane + pix_idx] = (rgb[hwc_idx + 2] as f32 / 255.0) * 2.0 - 1.0;
            }
        }
    }

    Tensor::from_floats(
        TensorData::new(flat, [batch_size, 3, TARGET_SIZE, TARGET_SIZE]),
        device,
    )
}

/// Save visual comparison image: [Input 32x32 Upscaled | Model Predicted 256 | Ground Truth 256]
fn save_preview_image(
    input_tokens: &[u8],
    pred_tensor: Tensor<TrainBackend, 4>,
    target_rgb: &[u8],
    out_path: &PathBuf,
) -> Result<()> {
    // 1. Convert input tokens to nearest-neighbor 256x256 image
    let mut preview = image::RgbImage::new((TARGET_SIZE * 3) as u32, TARGET_SIZE as u32);

    for y in 0..TARGET_SIZE {
        let src_y = y / 8;
        for x in 0..TARGET_SIZE {
            let src_x = x / 8;
            let token = input_tokens[src_y * 32 + src_x];
            let rgb = palette_to_rgb(token);
            preview.put_pixel(x as u32, y as u32, image::Rgb(rgb));
        }
    }

    // 2. Convert predicted tensor [1, 3, 256, 256] to RGB
    let pred_data: Vec<f32> = pred_tensor.into_data().convert::<f32>().to_vec().unwrap();
    let plane = TARGET_SIZE * TARGET_SIZE;
    for y in 0..TARGET_SIZE {
        for x in 0..TARGET_SIZE {
            let idx = y * TARGET_SIZE + x;
            let r = (((pred_data[0 * plane + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
            let g = (((pred_data[1 * plane + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
            let b = (((pred_data[2 * plane + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
            preview.put_pixel((TARGET_SIZE + x) as u32, y as u32, image::Rgb([r, g, b]));
        }
    }

    // 3. Put ground truth on the right
    for y in 0..TARGET_SIZE {
        for x in 0..TARGET_SIZE {
            let idx = (y * TARGET_SIZE + x) * 3;
            let r = target_rgb[idx];
            let g = target_rgb[idx + 1];
            let b = target_rgb[idx + 2];
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
        
        // Use values from config unless explicitly overridden
        args.cache = PathBuf::from(cfg.cache);
        args.out_dir = PathBuf::from(cfg.out_dir);
        args.epochs = cfg.epochs;
        args.batch_size = cfg.batch_size;
        args.lr = cfg.lr;
        args.min_lr = cfg.min_lr;
        args.res_blocks = cfg.res_blocks;
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
    println!("  Backbone        : {} ResBlocks ({} channels)", args.res_blocks, args.channels);
    println!("  Learning rate   : {:.2e} -> {:.2e}", args.lr, args.min_lr);

    fs::create_dir_all(&args.out_dir)?;

    // 1. Open dataset
    let mut reader = DatasetReader::open(&args.cache)?;
    let total_samples = reader.total_records;
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

    // 2. Initialize Model
    let config = UpscalerModelConfig::new()
        .with_channels(args.channels)
        .with_num_res_blocks(args.res_blocks);

    let mut model: UpscalerModel<TrainBackend> = config.init(&device);

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

        let mut epoch_loss = 0.0f32;
        let mut batch_count = 0;

        for chunk in epoch_indices.chunks(args.batch_size) {
            if chunk.len() < args.batch_size {
                continue;
            }

            // Load batch records
            let mut tokens_batch = Vec::with_capacity(chunk.len());
            let mut target_batch = Vec::with_capacity(chunk.len());

            for &idx in chunk {
                let (tokens, target) = reader.read_record(idx)?;
                tokens_batch.push(tokens);
                target_batch.push(target);
            }

            let input_tensor = tokens_batch_to_tensor(&tokens_batch, &device);
            let target_tensor = target_batch_to_tensor(&target_batch, &device);

            // Forward pass
            let pred = model.forward(input_tensor);
            let loss = model.loss(pred, target_tensor);

            let loss_val: f32 = loss.clone().into_data().convert::<f32>().to_vec().unwrap()[0];
            epoch_loss += loss_val;
            batch_count += 1;

            pb.set_prefix(format!("{:.4}", loss_val));

            // Backward pass & optimizer step
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(lr, model, grads);

            pb.inc(1);
        }

        let avg_train_loss = if batch_count > 0 { epoch_loss / batch_count as f32 } else { 0.0 };
        pb.finish_with_message(format!("{}/{} [Avg Loss: {:.4}]", epoch, args.epochs, avg_train_loss));

        // 4. Validation Loop (evaluate on subset of val_indices)
        let eval_count = val_indices.len().min(128);
        let mut val_loss_sum = 0.0f32;
        let mut val_batches = 0;

        for chunk in val_indices[..eval_count].chunks(args.batch_size) {
            let mut tokens_batch = Vec::with_capacity(chunk.len());
            let mut target_batch = Vec::with_capacity(chunk.len());

            for &idx in chunk {
                let (tokens, target) = reader.read_record(idx)?;
                tokens_batch.push(tokens);
                target_batch.push(target);
            }

            let input_tensor = tokens_batch_to_tensor(&tokens_batch, &device);
            let target_tensor = target_batch_to_tensor(&target_batch, &device);

            let pred = model.forward(input_tensor);
            let loss = model.loss(pred, target_tensor);
            let loss_val: f32 = loss.into_data().convert::<f32>().to_vec().unwrap()[0];
            val_loss_sum += loss_val;
            val_batches += 1;
        }

        let avg_val_loss = if val_batches > 0 { val_loss_sum / val_batches as f32 } else { 0.0 };
        println!(
            "  Epoch {:>2}/{} ({:.1?}): Train Loss = {:.4} | Val Loss = {:.4} | LR = {:.2e}",
            epoch, args.epochs, t_epoch.elapsed(), avg_train_loss, avg_val_loss, lr
        );

        // 5. Save visual preview sample
        let (preview_tokens, preview_target) = reader.read_record(val_indices[0])?;
        let preview_input = tokens_batch_to_tensor(&[preview_tokens.clone()], &device);
        let preview_pred = model.forward(preview_input);
        let preview_path = args.out_dir.join(format!("preview_epoch_{:02}.png", epoch));
        if let Err(e) = save_preview_image(&preview_tokens, preview_pred, &preview_target, &preview_path) {
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
            train_loss: avg_train_loss,
            val_loss: avg_val_loss,
            channels: args.channels,
            num_res_blocks: args.res_blocks,
            timestamp: chrono::Local::now().to_rfc3339(),
        };
        let meta_json = serde_json::to_string_pretty(&meta)?;
        fs::write(args.out_dir.join("metadata.json"), meta_json)?;
    }

    println!("\nTraining completed in {:.2?}. Model saved to {}", t_train_start.elapsed(), args.out_dir.display());
    Ok(())
}
