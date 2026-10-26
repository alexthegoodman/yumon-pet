//! Train Pass 4 Generative Upscaler (64x64/32x32 -> 256x256) using Burn WGPU Backend.
//!
//! Features:
//! - Multi-scale support (64x64 or 32x32 input).
//! - Joint colorization & super-resolution (1-channel grayscale input -> 3-channel RGB output).
//! - Prompt text conditioning via Cross-Attention at the feature bottleneck.
//! - Dual-objective training: regression loss (Huber + Edge gradients) + optional PatchGAN adversarial loss.
//! - Non-blocking GPU execution with interactive Ratatui dashboard.

use anyhow::{Context, Result, ensure};
use burn::{
    backend::Autodiff,
    module::Module,
    nn::loss::{HuberLossConfig, Reduction},
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
    chart::{render as render_upscaler_chart, UpscalerTrainingState},
    upscaler::{
        UpscalerDiscriminator, UpscalerDiscriminatorConfig, UpscalerModel, UpscalerModelConfig,
    },
};

pub type TrainBackend = Autodiff<
    burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>,
>;

pub const UPSCALER_CACHE_MAGIC: &[u8; 8] = b"YUMON_UP";
pub const TARGET_SIZE: usize = 256;
pub const TARGET_CHW_LEN: usize = 3 * TARGET_SIZE * TARGET_SIZE; // 196,608 bytes

#[derive(Parser, Debug)]
#[command(about = "Train Pass 4 Deep Residual Upscaler with Prompt Conditioning & Colorization")]
struct Args {
    /// Optional path to JSON config file.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Path to binary cache file.
    #[arg(long, default_value = "training-cache/upscaler_guided_256.bin")]
    cache: PathBuf,

    /// Output checkpoint directory.
    #[arg(long, default_value = "checkpoints/yumon-upscaler-guided")]
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

    /// Input resolution (auto-detected from cache if omitted).
    #[arg(long)]
    scales_from: Option<usize>,

    /// Whether input is grayscale to be colorized (auto-detected from cache if omitted).
    #[arg(long)]
    colorizes: Option<bool>,

    /// Enable prompt guidance via Cross-Attention.
    #[arg(long, default_value_t = true)]
    text_guidance: bool,

    /// Tokenizer vocabulary size.
    #[arg(long, default_value_t = 16384)]
    vocab_size: usize,

    /// Text embedding dimension.
    #[arg(long, default_value_t = 64)]
    text_dim: usize,

    /// Fraction of samples used for validation.
    #[arg(long, default_value_t = 0.02)]
    val_fraction: f64,

    /// Enable GAN (PatchGAN discriminator + adversarial loss for sharp textures).
    #[arg(long, default_value_t = false)]
    gan: bool,

    /// Weight of adversarial loss for Generator.
    #[arg(long, default_value_t = 0.02)]
    adv_weight: f64,

    /// Weight of feature matching loss for Generator.
    #[arg(long, default_value_t = 0.05)]
    feat_weight: f64,

    /// Initial learning rate for Discriminator.
    #[arg(long, default_value_t = 3e-5)]
    d_lr: f64,

    /// Base channels for PatchGAN Discriminator.
    #[arg(long, default_value_t = 32)]
    d_channels: usize,

    /// Disable interactive Ratatui TUI dashboard.
    #[arg(long, default_value_t = false)]
    no_tui: bool,

    /// Target label for real images in Discriminator (one-sided label smoothing, e.g. 0.9).
    #[arg(long, default_value_t = 0.9)]
    real_label: f32,
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
    #[serde(default)]
    scales_from: Option<usize>,
    #[serde(default)]
    colorizes: Option<bool>,
    #[serde(default = "default_true")]
    text_guidance: bool,
    #[serde(default = "default_vocab")]
    vocab_size: usize,
    #[serde(default = "default_text_dim")]
    text_dim: usize,
    #[serde(default = "default_val_fraction")]
    val_fraction: f64,
    #[serde(default)]
    gan: bool,
    #[serde(default = "default_adv_weight")]
    adv_weight: f64,
    #[serde(default = "default_feat_weight")]
    feat_weight: f64,
    #[serde(default = "default_d_lr")]
    d_lr: f64,
    #[serde(default = "default_d_channels")]
    d_channels: usize,
    #[serde(default)]
    no_tui: bool,
    #[serde(default = "default_real_label")]
    real_label: f32,
}

fn default_cache() -> String { "training-cache/upscaler_guided_256.bin".into() }
fn default_out_dir() -> String { "checkpoints/yumon-upscaler-guided".into() }
fn default_epochs() -> usize { 10 }
fn default_batch_size() -> usize { 8 }
fn default_lr() -> f64 { 3e-4 }
fn default_min_lr() -> f64 { 3e-5 }
fn default_channels() -> usize { 32 }
fn default_true() -> bool { true }
fn default_vocab() -> usize { 16384 }
fn default_text_dim() -> usize { 64 }
fn default_val_fraction() -> f64 { 0.02 }
fn default_adv_weight() -> f64 { 0.02 }
fn default_feat_weight() -> f64 { 0.05 }
fn default_d_lr() -> f64 { 3e-5 }
fn default_d_channels() -> usize { 32 }
fn default_real_label() -> f32 { 0.9 }

#[derive(Serialize, Deserialize, Debug)]
struct UpscalerMetadata {
    epoch: usize,
    epochs: usize,
    train_loss: f32,
    val_loss: f32,
    #[serde(default)]
    d_loss: Option<f32>,
    #[serde(default)]
    adv_loss: Option<f32>,
    channels: usize,
    scales_from: usize,
    colorizes: bool,
    use_text_conditioning: bool,
    vocab_size: usize,
    text_dim: usize,
    gan: bool,
    timestamp: String,
}

/// Precomputed in-memory dataset of normalized planar f32 tensors and prompt token IDs.
struct InMemDataset {
    total_samples: usize,
    input_grid_size: usize,
    input_channels: usize,
    max_text_len: usize,
    colorizes: bool,
    input_chw_len: usize,
    _text_bytes_len: usize,
    target_chw_len: usize,
    _record_size: usize,
    inputs: Vec<f32>,
    prompt_tokens: Vec<i32>,
    targets: Vec<f32>,
}

impl InMemDataset {
    fn load_from_cache(path: &PathBuf) -> Result<Self> {
        let t_load = Instant::now();
        let mut file = File::open(path)
            .with_context(|| format!("Failed to open cache file: {}", path.display()))?;

        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        ensure!(&magic == UPSCALER_CACHE_MAGIC, "Invalid cache magic header");

        let mut ver_bytes = [0u8; 4];
        file.read_exact(&mut ver_bytes)?;
        let version = u32::from_le_bytes(ver_bytes);

        let (total_samples, record_size, input_grid_size, target_size, input_channels, max_text_len, colorizes) =
            if version == 3 {
                let mut rest = [0u8; 52];
                file.read_exact(&mut rest)?;
                let total_samples = u32::from_le_bytes(rest[0..4].try_into().unwrap()) as usize;
                let record_size = u32::from_le_bytes(rest[4..8].try_into().unwrap()) as usize;
                let input_grid_size = u32::from_le_bytes(rest[8..12].try_into().unwrap()) as usize;
                let target_size = u32::from_le_bytes(rest[12..16].try_into().unwrap()) as usize;
                let input_channels = u32::from_le_bytes(rest[16..20].try_into().unwrap()) as usize;
                let max_text_len = u32::from_le_bytes(rest[20..24].try_into().unwrap()) as usize;
                let colorizes = u32::from_le_bytes(rest[24..28].try_into().unwrap()) == 1;
                (total_samples, record_size, input_grid_size, target_size, input_channels, max_text_len, colorizes)
            } else if version == 2 {
                let mut rest = [0u8; 20];
                file.read_exact(&mut rest)?;
                let total_samples = u32::from_le_bytes(rest[0..4].try_into().unwrap()) as usize;
                let record_size = u32::from_le_bytes(rest[4..8].try_into().unwrap()) as usize;
                let input_grid_size = u32::from_le_bytes(rest[8..12].try_into().unwrap()) as usize;
                let target_size = u32::from_le_bytes(rest[12..16].try_into().unwrap()) as usize;
                (total_samples, record_size, input_grid_size, target_size, 3, 0, false)
            } else {
                anyhow::bail!("Unsupported cache version: {}", version);
            };

        let input_chw_len = input_channels * input_grid_size * input_grid_size;
        let text_bytes_len = max_text_len * 2;
        let target_chw_len = 3 * target_size * target_size;
        ensure!(
            record_size == input_chw_len + text_bytes_len + target_chw_len,
            "Record size mismatch: header says {} vs computed {}",
            record_size,
            input_chw_len + text_bytes_len + target_chw_len
        );

        println!(
            "  Cache metadata: {} samples, {}x{} input ({} ch), {} text tokens, {}x{} target (colorizes={})",
            total_samples, input_grid_size, input_grid_size, input_channels, max_text_len, target_size, target_size, colorizes
        );
        println!("  Loading {} records ({:.2} MB) from disk into RAM...", total_samples, (total_samples * record_size) as f64 / 1e6);

        let mut raw_bytes = vec![0u8; total_samples * record_size];
        file.read_exact(&mut raw_bytes)?;

        println!("  Precomputing planar f32 tensors in parallel across CPU cores...");
        let mut inputs = vec![0.0f32; total_samples * input_chw_len];
        let mut prompt_tokens = vec![0i32; total_samples * max_text_len.max(1)];
        let mut targets = vec![0.0f32; total_samples * target_chw_len];

        // Lookup table: byte [0..255] -> f32 [-1.0, 1.0]
        let mut lut = [0.0f32; 256];
        for i in 0..256 {
            lut[i] = (i as f32 / 255.0) * 2.0 - 1.0;
        }

        let eff_text_len = max_text_len.max(1);
        inputs
            .par_chunks_exact_mut(input_chw_len)
            .zip(prompt_tokens.par_chunks_exact_mut(eff_text_len))
            .zip(targets.par_chunks_exact_mut(target_chw_len))
            .enumerate()
            .for_each(|(idx, ((in_slice, tok_slice), target_slice))| {
                let rec_offset = idx * record_size;
                let in_bytes = &raw_bytes[rec_offset..rec_offset + input_chw_len];
                let text_bytes = &raw_bytes[rec_offset + input_chw_len..rec_offset + input_chw_len + text_bytes_len];
                let target_bytes = &raw_bytes[rec_offset + input_chw_len + text_bytes_len..rec_offset + record_size];

                for i in 0..input_chw_len {
                    in_slice[i] = lut[in_bytes[i] as usize];
                }

                if max_text_len > 0 {
                    for t in 0..max_text_len {
                        let tid = u16::from_le_bytes([text_bytes[t * 2], text_bytes[t * 2 + 1]]);
                        tok_slice[t] = tid as i32;
                    }
                } else {
                    tok_slice[0] = 0;
                }

                for i in 0..target_chw_len {
                    target_slice[i] = lut[target_bytes[i] as usize];
                }
            });

        println!("  ✓ Dataset ready in RAM in {:.2?} (Zero disk I/O during training!)", t_load.elapsed());
        Ok(Self {
            total_samples,
            input_grid_size,
            input_channels,
            max_text_len,
            colorizes,
            input_chw_len,
            _text_bytes_len: text_bytes_len,
            target_chw_len,
            _record_size: record_size,
            inputs,
            prompt_tokens,
            targets,
        })
    }

    fn make_batch_tensors<B: Backend>(
        &self,
        indices: &[usize],
        device: &B::Device,
    ) -> (Tensor<B, 4>, Tensor<B, 2, Int>, Tensor<B, 4>) {
        let b = indices.len();
        let mut flat_in = Vec::with_capacity(b * self.input_chw_len);
        let eff_text = self.max_text_len.max(1);
        let mut flat_tok = Vec::with_capacity(b * eff_text);
        let mut flat_target = Vec::with_capacity(b * self.target_chw_len);

        for &idx in indices {
            let in_start = idx * self.input_chw_len;
            flat_in.extend_from_slice(&self.inputs[in_start..in_start + self.input_chw_len]);

            let tok_start = idx * eff_text;
            flat_tok.extend_from_slice(&self.prompt_tokens[tok_start..tok_start + eff_text]);

            let target_start = idx * self.target_chw_len;
            flat_target.extend_from_slice(&self.targets[target_start..target_start + self.target_chw_len]);
        }

        let in_tensor = Tensor::from_floats(
            TensorData::new(flat_in, [b, self.input_channels, self.input_grid_size, self.input_grid_size]),
            device,
        );
        let tok_tensor = Tensor::from_data(
            TensorData::new(flat_tok, [b, eff_text]),
            device,
        );
        let target_tensor = Tensor::from_floats(
            TensorData::new(flat_target, [b, 3, TARGET_SIZE, TARGET_SIZE]),
            device,
        );

        (in_tensor, tok_tensor, target_tensor)
    }
}

/// Save 16-sample visual comparison grid: 4 rows x 4 columns of [Input NN | Model Predicted 256 | Ground Truth 256]
fn save_preview_grid(
    dataset: &InMemDataset,
    indices: &[usize],
    pred_tensor: Tensor<TrainBackend, 4>,
    out_path: &PathBuf,
) -> Result<()> {
    let b = indices.len();
    let cols = 4u32;
    let rows = ((b as u32) + cols - 1) / cols;

    let triplet_w = (TARGET_SIZE * 3 + 2) as u32; // 770 px
    let triplet_h = TARGET_SIZE as u32; // 256 px
    let cell_margin = 8u32;
    let bg_color = image::Rgb([18, 18, 26]);

    let img_w = cell_margin + cols * (triplet_w + cell_margin);
    let img_h = cell_margin + rows * (triplet_h + cell_margin);

    let mut grid_img = image::RgbImage::from_pixel(img_w, img_h, bg_color);
    let pred_data: Vec<f32> = pred_tensor.into_data().convert::<f32>().to_vec().unwrap();

    let in_grid = dataset.input_grid_size;
    let in_channels = dataset.input_channels;
    let in_chw_len = dataset.input_chw_len;
    let scale = (TARGET_SIZE / in_grid).max(1);
    let plane_256 = TARGET_SIZE * TARGET_SIZE;
    let target_chw_len = dataset.target_chw_len;

    for (i, &idx) in indices.iter().enumerate() {
        let r = (i as u32) / cols;
        let c = (i as u32) % cols;
        let x_offset = cell_margin + c * (triplet_w + cell_margin);
        let y_offset = cell_margin + r * (triplet_h + cell_margin);

        let in_slice = &dataset.inputs[idx * in_chw_len..(idx + 1) * in_chw_len];
        let target_slice = &dataset.targets[idx * target_chw_len..(idx + 1) * target_chw_len];
        let pred_slice = &pred_data[i * target_chw_len..(i + 1) * target_chw_len];

        // 1. Input upscaled nearest-neighbor to 256x256
        for y in 0..TARGET_SIZE {
            let src_y = (y / scale).min(in_grid - 1);
            for x in 0..TARGET_SIZE {
                let src_x = (x / scale).min(in_grid - 1);
                let p_in = src_y * in_grid + src_x;

                let (red, grn, blu) = if in_channels == 1 {
                    let val = (((in_slice[p_in] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                    (val, val, val)
                } else {
                    let plane_in = in_grid * in_grid;
                    let red = (((in_slice[0 * plane_in + p_in] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                    let grn = (((in_slice[1 * plane_in + p_in] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                    let blu = (((in_slice[2 * plane_in + p_in] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                    (red, grn, blu)
                };

                grid_img.put_pixel(x_offset + x as u32, y_offset + y as u32, image::Rgb([red, grn, blu]));
            }
        }

        // Separator 1
        for y in 0..TARGET_SIZE {
            grid_img.put_pixel(x_offset + TARGET_SIZE as u32, y_offset + y as u32, image::Rgb([30, 30, 40]));
        }

        // 2. Model Predicted 256x256
        let x_pred_base = x_offset + TARGET_SIZE as u32 + 1;
        for y in 0..TARGET_SIZE {
            for x in 0..TARGET_SIZE {
                let p_idx = y * TARGET_SIZE + x;
                let red = (((pred_slice[0 * plane_256 + p_idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                let grn = (((pred_slice[1 * plane_256 + p_idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                let blu = (((pred_slice[2 * plane_256 + p_idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                grid_img.put_pixel(x_pred_base + x as u32, y_offset + y as u32, image::Rgb([red, grn, blu]));
            }
        }

        // Separator 2
        for y in 0..TARGET_SIZE {
            grid_img.put_pixel(x_pred_base + TARGET_SIZE as u32, y_offset + y as u32, image::Rgb([30, 30, 40]));
        }

        // 3. Ground Truth 256x256
        let x_target_base = x_pred_base + TARGET_SIZE as u32 + 1;
        for y in 0..TARGET_SIZE {
            for x in 0..TARGET_SIZE {
                let t_idx = y * TARGET_SIZE + x;
                let red = (((target_slice[0 * plane_256 + t_idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                let grn = (((target_slice[1 * plane_256 + t_idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                let blu = (((target_slice[2 * plane_256 + t_idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                grid_img.put_pixel(x_target_base + x as u32, y_offset + y as u32, image::Rgb([red, grn, blu]));
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
        let cfg: UpscalerConfigFile = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {}", config_path.display()))?;

        args.cache = PathBuf::from(cfg.cache);
        args.out_dir = PathBuf::from(cfg.out_dir);
        args.epochs = cfg.epochs;
        args.batch_size = cfg.batch_size;
        args.lr = cfg.lr;
        args.min_lr = cfg.min_lr;
        args.channels = cfg.channels;
        args.scales_from = cfg.scales_from;
        args.colorizes = cfg.colorizes;
        args.text_guidance = cfg.text_guidance;
        args.vocab_size = cfg.vocab_size;
        args.text_dim = cfg.text_dim;
        args.val_fraction = cfg.val_fraction;
        args.gan = cfg.gan;
        args.adv_weight = cfg.adv_weight;
        args.feat_weight = cfg.feat_weight;
        args.d_lr = cfg.d_lr;
        args.d_channels = cfg.d_channels;
        args.no_tui = cfg.no_tui;
        args.real_label = cfg.real_label;
    }

    let device: <TrainBackend as Backend>::Device = Default::default();

    println!("==================================================");
    println!("  Yumon Pass 4 Guided Colorizing Upscaler Trainer");
    println!("==================================================");
    println!("  Cache binary    : {}", args.cache.display());
    println!("  Output directory: {}", args.out_dir.display());
    println!("  Epochs          : {}", args.epochs);
    println!("  Batch size      : {}", args.batch_size);
    println!("  Channels        : {}", args.channels);
    println!("  Text guidance   : {}", args.text_guidance);
    println!("  Learning rate   : {:.2e} -> {:.2e}", args.lr, args.min_lr);
    if args.gan {
        println!("  GAN Training    : ENABLED (PatchGAN LSGAN + Feature-Matching)");
        println!("  D base channels : {}", args.d_channels);
        println!("  D learning rate : {:.2e}", args.d_lr);
        println!("  Adversarial wt  : {:.4}", args.adv_weight);
        println!("  Feature match wt: {:.4}", args.feat_weight);
    } else {
        println!("  GAN Training    : DISABLED (pure regression)");
    }

    fs::create_dir_all(&args.out_dir)?;

    // 1. Load dataset into RAM
    let dataset = InMemDataset::load_from_cache(&args.cache)?;
    let total_samples = dataset.total_samples;
    println!("  Total cache samples: {}", total_samples);

    let scales_from = args.scales_from.unwrap_or(dataset.input_grid_size);
    let colorizes = args.colorizes.unwrap_or(dataset.colorizes);

    let val_count = ((total_samples as f64) * args.val_fraction).max(10.0) as usize;
    let train_count = total_samples.saturating_sub(val_count);

    let mut indices: Vec<usize> = (0..total_samples).collect();
    let mut rng = thread_rng();
    indices.shuffle(&mut rng);

    let val_indices = indices[train_count..].to_vec();
    let train_indices = indices[..train_count].to_vec();

    println!("  Train samples   : {}", train_indices.len());
    println!("  Val samples     : {}", val_indices.len());
    println!("  Model resolution: {}x{} -> 256x256 (colorizes: {})", scales_from, scales_from, colorizes);

    // 2. Initialize Model, Discriminator, and Loss
    let use_text = args.text_guidance && dataset.max_text_len > 0;
    let config = UpscalerModelConfig::new()
        .with_scales_from(scales_from)
        .with_colorizes(colorizes)
        .with_channels(args.channels)
        .with_use_text_conditioning(use_text)
        .with_vocab_size(args.vocab_size)
        .with_text_dim(args.text_dim);

    let mut model: UpscalerModel<TrainBackend> = config.init(&device);
    let loss_fn = HuberLossConfig::new(1.0).init();

    let mut optimizer = AdamWConfig::new()
        .with_epsilon(1e-8)
        .with_weight_decay(1e-4)
        .init();

    let (mut discriminator, mut optimizer_d) = if args.gan {
        let d_config = UpscalerDiscriminatorConfig::new()
            .with_base_channels(args.d_channels);
        let disc: UpscalerDiscriminator<TrainBackend> = d_config.init(&device);
        let opt_d = AdamWConfig::new()
            .with_epsilon(1e-8)
            .with_weight_decay(1e-4)
            .init();
        (Some(disc), Some(opt_d))
    } else {
        (None, None)
    };

    let num_batches = train_indices.len() / args.batch_size;
    let t_train_start = Instant::now();

    let mut tui_state = UpscalerTrainingState::new(args.epochs, num_batches, args.gan);

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
                eprintln!("Warning: Failed to initialize inline TUI terminal: {e}. Falling back to standard output.");
                None
            }
        }
    } else {
        None
    };

    // 3. Training Loop
    for epoch in 1..=args.epochs {
        let t_epoch = Instant::now();
        let mut epoch_indices = train_indices.clone();
        epoch_indices.shuffle(&mut rng);

        // Cosine learning rate decay
        let progress = (epoch - 1) as f64 / args.epochs.max(1) as f64;
        let lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + (progress * std::f64::consts::PI).cos());
        let d_lr = args.d_lr * (0.5 * (1.0 + (progress * std::f64::consts::PI).cos())).max(0.1);

        let pb = if terminal.is_none() {
            let bar = ProgressBar::new(num_batches as u64);
            if args.gan {
                bar.set_style(
                    ProgressStyle::default_bar()
                        .template("{spinner:.green} [Epoch {msg}] [{bar:25.cyan/blue}] {pos}/{len} ({eta}) {prefix}")
                        .unwrap()
                        .progress_chars("#>-"),
                );
            } else {
                bar.set_style(
                    ProgressStyle::default_bar()
                        .template("{spinner:.green} [Epoch {msg}] [{bar:30.cyan/blue}] {pos}/{len} ({eta}) loss: {prefix}")
                        .unwrap()
                        .progress_chars("#>-"),
                );
            }
            bar.set_message(format!("{}/{}", epoch, args.epochs));
            Some(bar)
        } else {
            None
        };

        let mut batch_count = 0;
        let mut last_loss_val = 0.0f32;
        let mut last_d_loss_val = 0.0f32;
        let mut last_adv_loss_val = 0.0f32;

        for chunk in epoch_indices.chunks(args.batch_size) {
            if chunk.len() < args.batch_size {
                continue;
            }

            let should_inspect = (batch_count + 1) % 10 == 0 || (batch_count + 1) == num_batches;

            // High-speed RAM batch slicing (Zero Disk I/O)
            let (input_tensor, prompt_tensor, target_tensor) = dataset.make_batch_tensors(chunk, &device);

            // Forward pass: Generator produces 256x256 image with prompt conditioning
            let pred = model.forward_with_text(input_tensor, Some(prompt_tensor));

            let (loss, d_loss_item, adv_loss_item) = if discriminator.is_some() {
                let mut disc = discriminator.take().unwrap();
                let opt_d = optimizer_d.as_mut().unwrap();

                // 1. Train PatchGAN Discriminator (LSGAN objective with one-sided label smoothing)
                let (d_real, d_real_feats) = disc.forward_with_features(target_tensor.clone());
                let err_real = d_real - args.real_label;
                let loss_d_real = (err_real.clone() * err_real).mean();

                let pred_detached = pred.clone().detach();
                let (d_fake_detached, _) = disc.forward_with_features(pred_detached);
                let loss_d_fake = (d_fake_detached.clone() * d_fake_detached).mean();

                let loss_d = (loss_d_real + loss_d_fake) * 0.5;

                let grads_d = loss_d.clone().backward();
                let grads_d = GradientsParams::from_grads(grads_d, &disc);
                disc = opt_d.step(d_lr, disc, grads_d);

                let d_val = if should_inspect {
                    loss_d.into_data().convert::<f32>().to_vec().unwrap()[0]
                } else {
                    0.0
                };

                // 2. Train Generator: Pixel + Edge + Feature Matching + Adversarial
                let pixel_loss = loss_fn.forward(
                    pred.clone(),
                    target_tensor.clone(),
                    Reduction::Mean,
                );
                let edge_loss = gradient_loss(
                    pred.clone(),
                    target_tensor,
                );

                let (d_fake_for_g, d_fake_feats) = disc.forward_with_features(pred);
                let adv_err = d_fake_for_g - 1.0;
                let adv_loss = (adv_err.clone() * adv_err).mean();

                let mut feat_loss_sum: Option<Tensor<TrainBackend, 1>> = None;
                for (real_feat, fake_feat) in d_real_feats.into_iter().zip(d_fake_feats.into_iter()) {
                    let diff = (fake_feat - real_feat.detach()).abs().mean();
                    feat_loss_sum = Some(match feat_loss_sum {
                        Some(acc) => acc + diff,
                        None => diff,
                    });
                }
                let feat_loss = feat_loss_sum.unwrap();

                let adv_val = if should_inspect {
                    adv_loss.clone().into_data().convert::<f32>().to_vec().unwrap()[0]
                } else {
                    0.0
                };

                let g_loss = pixel_loss + edge_loss * 0.1
                    + adv_loss * (args.adv_weight as f32)
                    + feat_loss * (args.feat_weight as f32);

                discriminator = Some(disc);

                (g_loss, d_val, adv_val)
            } else {
                let pixel_loss = loss_fn.forward(
                    pred.clone(),
                    target_tensor.clone(),
                    Reduction::Mean,
                );
                let edge_loss = gradient_loss(
                    pred,
                    target_tensor,
                );
                (pixel_loss + edge_loss * 0.1, 0.0, 0.0)
            };

            batch_count += 1;

            if should_inspect {
                let loss_val: f32 = loss.clone().into_data().convert::<f32>().to_vec().unwrap()[0];
                last_loss_val = loss_val;
                if args.gan {
                    last_d_loss_val = d_loss_item;
                    last_adv_loss_val = adv_loss_item;
                }

                let global_step = (epoch - 1) * num_batches + batch_count;
                tui_state.epoch = epoch;
                tui_state.batch = batch_count;
                tui_state.global_step = global_step;
                tui_state.current_lr = lr;
                tui_state.current_g_loss = loss_val;
                tui_state.current_d_loss = d_loss_item;
                tui_state.current_adv_loss = adv_loss_item;

                tui_state.g_loss_history.push((global_step as f64, loss_val as f64));
                tui_state.lr_history.push((global_step as f64, lr));
                if args.gan {
                    tui_state.d_loss_history.push((global_step as f64, d_loss_item as f64));
                    tui_state.adv_loss_history.push((global_step as f64, adv_loss_item as f64));
                }

                if let Some(term) = terminal.as_mut() {
                    let _ = term.draw(|f| render_upscaler_chart(f, &tui_state));
                } else if let Some(ref p) = pb {
                    if args.gan {
                        p.set_prefix(format!("G:{:.4} D:{:.4} Adv:{:.4}", loss_val, d_loss_item, adv_loss_item));
                    } else {
                        p.set_prefix(format!("{:.4}", loss_val));
                    }
                }
            }

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(lr, model, grads);

            if let Some(ref p) = pb {
                p.inc(1);
            }
        }

        if let Some(p) = pb {
            if args.gan {
                p.finish_with_message(format!(
                    "{}/{} [G: {:.4} | D: {:.4} | Adv: {:.4}]",
                    epoch, args.epochs, last_loss_val, last_d_loss_val, last_adv_loss_val
                ));
            } else {
                p.finish_with_message(format!("{}/{} [Loss: {:.4}]", epoch, args.epochs, last_loss_val));
            }
        }

        // 4. Validation Loop
        let eval_count = val_indices.len().min(128);
        let mut val_loss_sum = 0.0f32;
        let mut val_batches = 0;

        for chunk in val_indices[..eval_count].chunks(args.batch_size) {
            let (input_tensor, prompt_tensor, target_tensor) = dataset.make_batch_tensors(chunk, &device);
            let pred = model.forward_with_text(input_tensor, Some(prompt_tensor));
            let loss = loss_fn.forward(pred, target_tensor, Reduction::Mean);
            let loss_val: f32 = loss.into_data().convert::<f32>().to_vec().unwrap()[0];
            val_loss_sum += loss_val;
            val_batches += 1;
        }

        let avg_val_loss = if val_batches > 0 { val_loss_sum / val_batches as f32 } else { 0.0 };

        let global_step = epoch * num_batches;
        tui_state.current_val_loss = Some(avg_val_loss);
        tui_state.val_loss_history.push((global_step as f64, avg_val_loss as f64));
        let _ = tui_state.save_csv(&args.out_dir.join("loss_history.csv"));

        if let Some(term) = terminal.as_mut() {
            let _ = term.draw(|f| render_upscaler_chart(f, &tui_state));
        } else {
            println!(
                "  Epoch {:>2}/{} ({:.1?}): Val Loss = {:.4} | LR = {:.2e}",
                epoch, args.epochs, t_epoch.elapsed(), avg_val_loss, lr
            );
        }

        // 5. Save visual preview sample grid (16 validation samples)
        let preview_count = 16.min(val_indices.len());
        let preview_indices = &val_indices[..preview_count];
        let (preview_in, preview_prompt, _) = dataset.make_batch_tensors(preview_indices, &device);
        let preview_pred = model.forward_with_text(preview_in, Some(preview_prompt));

        let preview_path = args.out_dir.join(format!("preview_epoch_{:02}.png", epoch));

        if let Err(e) = save_preview_grid(&dataset, preview_indices, preview_pred, &preview_path) {
            eprintln!("Warning: Failed to save preview image grid: {}", e);
        } else {
            tui_state.last_preview_path = Some(preview_path.to_string_lossy().to_string());
            if terminal.is_none() {
                println!("  [Preview] Saved 16-sample comparison grid to {}", preview_path.display());
            }
        }

        // 6. Save checkpoint
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        let model_path = args.out_dir.join("model");
        model.clone().save_file(&model_path, &recorder)?;

        if let Some(disc) = &discriminator {
            let disc_path = args.out_dir.join("discriminator");
            disc.clone().save_file(&disc_path, &recorder)?;
        }

        let meta = UpscalerMetadata {
            epoch,
            epochs: args.epochs,
            train_loss: last_loss_val,
            val_loss: avg_val_loss,
            d_loss: if args.gan { Some(last_d_loss_val) } else { None },
            adv_loss: if args.gan { Some(last_adv_loss_val) } else { None },
            channels: args.channels,
            scales_from,
            colorizes,
            use_text_conditioning: use_text,
            vocab_size: args.vocab_size,
            text_dim: args.text_dim,
            gan: args.gan,
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

    let pred_dx = pred.clone().slice([0..d[0], 0..d[1], 0..d[2], 1..d[3]])
        - pred.clone().slice([0..d[0], 0..d[1], 0..d[2], 0..(d[3] - 1)]);

    let target_dx = target.clone().slice([0..td[0], 0..td[1], 0..td[2], 1..td[3]])
        - target.clone().slice([0..td[0], 0..td[1], 0..td[2], 0..(td[3] - 1)]);

    let pred_dy = pred.clone().slice([0..d[0], 0..d[1], 1..d[2], 0..d[3]])
        - pred.clone().slice([0..d[0], 0..d[1], 0..(d[2] - 1), 0..d[3]]);

    let target_dy = target.clone().slice([0..td[0], 0..td[1], 1..td[2], 0..td[3]])
        - target.clone().slice([0..td[0], 0..td[1], 0..(td[2] - 1), 0..td[3]]);

    let loss_x = (pred_dx - target_dx).abs().mean();
    let loss_y = (pred_dy - target_dy).abs().mean();

    loss_x + loss_y
}
