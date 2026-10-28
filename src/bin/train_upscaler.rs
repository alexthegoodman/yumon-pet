//! Train Pass 4/5 Upscaler (64x64/32x32 -> 256x256 Grayscale to Grayscale) using Burn WGPU Backend.
//!
//! Features:
//! - Pure 1-channel grayscale input -> 1-channel grayscale super-resolved output.
//! - Multi-scale support (64x64 default, supports 32x32).
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
    tensor::{Tensor, TensorData},
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
        default_fsq_levels, CodebookUpscalerModel, CodebookUpscalerModelConfig,
        UpscalerDiscriminator, UpscalerDiscriminatorConfig, UpscalerModel, UpscalerModelConfig,
    },
};

pub type TrainBackend = Autodiff<
    burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>,
>;
pub type InnerBackend = <TrainBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend;

use burn::module::AutodiffModule;

pub trait TrainableUpscaler: AutodiffModule<TrainBackend> + Clone + Send + 'static {
    type ValidModel: Module<InnerBackend> + Send + 'static;

    fn forward_train(&self, x: Tensor<TrainBackend, 4>) -> Tensor<TrainBackend, 4>;
    fn to_valid(&self) -> Self::ValidModel;
    fn forward_valid(valid_model: &Self::ValidModel, x: Tensor<InnerBackend, 4>) -> Tensor<InnerBackend, 4>;
}

impl TrainableUpscaler for UpscalerModel<TrainBackend> {
    type ValidModel = UpscalerModel<InnerBackend>;

    fn forward_train(&self, x: Tensor<TrainBackend, 4>) -> Tensor<TrainBackend, 4> {
        self.forward(x)
    }

    fn to_valid(&self) -> Self::ValidModel {
        self.valid()
    }

    fn forward_valid(valid_model: &Self::ValidModel, x: Tensor<InnerBackend, 4>) -> Tensor<InnerBackend, 4> {
        valid_model.forward(x)
    }
}

impl TrainableUpscaler for CodebookUpscalerModel<TrainBackend> {
    type ValidModel = CodebookUpscalerModel<InnerBackend>;

    fn forward_train(&self, x: Tensor<TrainBackend, 4>) -> Tensor<TrainBackend, 4> {
        self.forward(x)
    }

    fn to_valid(&self) -> Self::ValidModel {
        self.valid()
    }

    fn forward_valid(valid_model: &Self::ValidModel, x: Tensor<InnerBackend, 4>) -> Tensor<InnerBackend, 4> {
        valid_model.forward(x)
    }
}

pub const UPSCALER_CACHE_MAGIC: &[u8; 8] = b"YUMON_UP";
pub const TARGET_SIZE: usize = 256;
pub const TARGET_LEN: usize = TARGET_SIZE * TARGET_SIZE; // 65,536 bytes

#[derive(Parser, Debug)]
#[command(about = "Train Pass 4/5 Deep Residual Upscaler (Grayscale to Grayscale)")]
struct Args {
    /// Optional path to JSON config file.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Path to binary cache file.
    #[arg(long, default_value = "training-cache/upscaler_64_gray.bin")]
    cache: PathBuf,

    /// Output checkpoint directory.
    #[arg(long, default_value = "checkpoints/yumon-upscaler-64")]
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

    /// Number of residual blocks in the backbone (default 4).
    #[arg(long, default_value_t = 4)]
    res_blocks: usize,

    /// Enable discrete FSQ codebook quantization bottleneck to hallucinate sharp textures.
    #[arg(long, default_value_t = true)]
    codebook: bool,

    /// Input resolution (auto-detected from cache if omitted).
    #[arg(long)]
    scales_from: Option<usize>,

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
    #[serde(default = "default_res_blocks")]
    res_blocks: usize,
    #[serde(default = "default_codebook")]
    codebook: bool,
    #[serde(default)]
    scales_from: Option<usize>,
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

fn default_cache() -> String { "training-cache/upscaler_64_gray.bin".into() }
fn default_out_dir() -> String { "checkpoints/yumon-upscaler-64".into() }
fn default_epochs() -> usize { 10 }
fn default_batch_size() -> usize { 8 }
fn default_lr() -> f64 { 3e-4 }
fn default_min_lr() -> f64 { 3e-5 }
fn default_channels() -> usize { 64 }
fn default_res_blocks() -> usize { 4 }
fn default_codebook() -> bool { true }
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
    #[serde(default)]
    in_channels: usize,
    #[serde(default)]
    out_channels: usize,
    scales_from: usize,
    gan: bool,
    #[serde(default)]
    codebook: bool,
    #[serde(default)]
    fsq_levels: Option<Vec<usize>>,
    timestamp: String,
}

/// Precomputed in-memory dataset of normalized planar f32 tensors.
struct InMemDataset {
    total_samples: usize,
    input_grid_size: usize,
    channels: usize,
    input_len: usize,
    target_len: usize,
    inputs: Vec<f32>,
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

        let (total_samples, record_size, input_grid_size, target_size, channels) = if version == 4 {
            let mut rest = [0u8; 20];
            file.read_exact(&mut rest)?;
            let total_samples = u32::from_le_bytes(rest[0..4].try_into().unwrap()) as usize;
            let record_size = u32::from_le_bytes(rest[4..8].try_into().unwrap()) as usize;
            let input_grid_size = u32::from_le_bytes(rest[8..12].try_into().unwrap()) as usize;
            let target_size = u32::from_le_bytes(rest[12..16].try_into().unwrap()) as usize;
            let raw_channels = u32::from_le_bytes(rest[16..20].try_into().unwrap()) as usize;
            let channels = if raw_channels == 0 { 1 } else { raw_channels };
            (total_samples, record_size, input_grid_size, target_size, channels)
        } else {
            anyhow::bail!("Unsupported cache version: {}. Expected version 4. Please regenerate with cache_upscaler.", version);
        };

        let plane_in = input_grid_size * input_grid_size;
        let plane_target = target_size * target_size;
        let input_len = channels * plane_in;
        let target_len = channels * plane_target;
        ensure!(
            record_size == input_len + target_len,
            "Record size mismatch: header says {} vs expected {}",
            record_size,
            input_len + target_len
        );

        println!(
            "  Cache metadata: {} samples, {}x{}x{} input -> {}x{}x{} target ({})",
            total_samples, channels, input_grid_size, input_grid_size, channels, target_size, target_size,
            if channels == 3 { "RGB Color" } else { "Grayscale" }
        );
        println!("  Loading {} records ({:.2} MB) from disk into RAM...", total_samples, (total_samples * record_size) as f64 / 1e6);

        let mut raw_bytes = vec![0u8; total_samples * record_size];
        file.read_exact(&mut raw_bytes)?;

        println!("  Precomputing planar f32 tensors in parallel across CPU cores...");
        let mut inputs = vec![0.0f32; total_samples * input_len];
        let mut targets = vec![0.0f32; total_samples * target_len];

        // Lookup table: byte [0..255] -> f32 [-1.0, 1.0]
        let mut lut = [0.0f32; 256];
        for i in 0..256 {
            lut[i] = (i as f32 / 255.0) * 2.0 - 1.0;
        }

        inputs
            .par_chunks_exact_mut(input_len)
            .zip(targets.par_chunks_exact_mut(target_len))
            .enumerate()
            .for_each(|(idx, (in_slice, target_slice))| {
                let rec_offset = idx * record_size;
                let in_bytes = &raw_bytes[rec_offset..rec_offset + input_len];
                let target_bytes = &raw_bytes[rec_offset + input_len..rec_offset + record_size];

                for i in 0..input_len {
                    in_slice[i] = lut[in_bytes[i] as usize];
                }
                for i in 0..target_len {
                    target_slice[i] = lut[target_bytes[i] as usize];
                }
            });

        println!("  ✓ Dataset ready in RAM in {:.2?} (Zero disk I/O during training!)", t_load.elapsed());
        Ok(Self {
            total_samples,
            input_grid_size,
            channels,
            input_len,
            target_len,
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
        let mut flat_in = Vec::with_capacity(b * self.input_len);
        let mut flat_target = Vec::with_capacity(b * self.target_len);

        for &idx in indices {
            let in_start = idx * self.input_len;
            flat_in.extend_from_slice(&self.inputs[in_start..in_start + self.input_len]);

            let target_start = idx * self.target_len;
            flat_target.extend_from_slice(&self.targets[target_start..target_start + self.target_len]);
        }

        let in_tensor = Tensor::from_floats(
            TensorData::new(flat_in, [b, self.channels, self.input_grid_size, self.input_grid_size]),
            device,
        );
        let target_tensor = Tensor::from_floats(
            TensorData::new(flat_target, [b, self.channels, TARGET_SIZE, TARGET_SIZE]),
            device,
        );

        (in_tensor, target_tensor)
    }
}

/// Save 16-sample visual comparison grid: 4 rows x 4 columns of [Input NN | Model Predicted 256 | Ground Truth 256]
fn save_preview_grid<B: Backend>(
    dataset: &InMemDataset,
    indices: &[usize],
    pred_tensor: Tensor<B, 4>,
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
    let in_len = dataset.input_len;
    let scale = (TARGET_SIZE / in_grid).max(1);
    let target_len = dataset.target_len;

    for (i, &idx) in indices.iter().enumerate() {
        let r = (i as u32) / cols;
        let c = (i as u32) % cols;
        let x_offset = cell_margin + c * (triplet_w + cell_margin);
        let y_offset = cell_margin + r * (triplet_h + cell_margin);

        let in_slice = &dataset.inputs[idx * in_len..(idx + 1) * in_len];
        let target_slice = &dataset.targets[idx * target_len..(idx + 1) * target_len];
        let pred_slice = &pred_data[i * target_len..(i + 1) * target_len];

        if dataset.channels == 3 {
            let plane_in = in_grid * in_grid;
            let plane_target = TARGET_SIZE * TARGET_SIZE;

            // 1. Input upscaled nearest-neighbor to 256x256
            for y in 0..TARGET_SIZE {
                let src_y = (y / scale).min(in_grid - 1);
                for x in 0..TARGET_SIZE {
                    let src_x = (x / scale).min(in_grid - 1);
                    let p_in = src_y * in_grid + src_x;
                    let r = (((in_slice[0 * plane_in + p_in] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                    let g = (((in_slice[1 * plane_in + p_in] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                    let b = (((in_slice[2 * plane_in + p_in] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                    grid_img.put_pixel(x_offset + x as u32, y_offset + y as u32, image::Rgb([r, g, b]));
                }
            }

            // Separator 1
            for y in 0..TARGET_SIZE {
                grid_img.put_pixel(x_offset + TARGET_SIZE as u32, y_offset + y as u32, image::Rgb([30, 30, 40]));
            }

            // 2. Model Predicted 256x256 RGB
            let x_pred_base = x_offset + TARGET_SIZE as u32 + 1;
            for y in 0..TARGET_SIZE {
                for x in 0..TARGET_SIZE {
                    let p_idx = y * TARGET_SIZE + x;
                    let r = (((pred_slice[0 * plane_target + p_idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                    let g = (((pred_slice[1 * plane_target + p_idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                    let b = (((pred_slice[2 * plane_target + p_idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                    grid_img.put_pixel(x_pred_base + x as u32, y_offset + y as u32, image::Rgb([r, g, b]));
                }
            }

            // Separator 2
            for y in 0..TARGET_SIZE {
                grid_img.put_pixel(x_pred_base + TARGET_SIZE as u32, y_offset + y as u32, image::Rgb([30, 30, 40]));
            }

            // 3. Ground Truth 256x256 RGB
            let x_target_base = x_pred_base + TARGET_SIZE as u32 + 1;
            for y in 0..TARGET_SIZE {
                for x in 0..TARGET_SIZE {
                    let t_idx = y * TARGET_SIZE + x;
                    let r = (((target_slice[0 * plane_target + t_idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                    let g = (((target_slice[1 * plane_target + t_idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                    let b = (((target_slice[2 * plane_target + t_idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                    grid_img.put_pixel(x_target_base + x as u32, y_offset + y as u32, image::Rgb([r, g, b]));
                }
            }
        } else {
            // 1. Input upscaled nearest-neighbor to 256x256
            for y in 0..TARGET_SIZE {
                let src_y = (y / scale).min(in_grid - 1);
                for x in 0..TARGET_SIZE {
                    let src_x = (x / scale).min(in_grid - 1);
                    let p_in = src_y * in_grid + src_x;
                    let val = (((in_slice[p_in] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                    grid_img.put_pixel(x_offset + x as u32, y_offset + y as u32, image::Rgb([val, val, val]));
                }
            }

            // Separator 1
            for y in 0..TARGET_SIZE {
                grid_img.put_pixel(x_offset + TARGET_SIZE as u32, y_offset + y as u32, image::Rgb([30, 30, 40]));
            }

            // 2. Model Predicted 256x256 Grayscale
            let x_pred_base = x_offset + TARGET_SIZE as u32 + 1;
            for y in 0..TARGET_SIZE {
                for x in 0..TARGET_SIZE {
                    let p_idx = y * TARGET_SIZE + x;
                    let val = (((pred_slice[p_idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                    grid_img.put_pixel(x_pred_base + x as u32, y_offset + y as u32, image::Rgb([val, val, val]));
                }
            }

            // Separator 2
            for y in 0..TARGET_SIZE {
                grid_img.put_pixel(x_pred_base + TARGET_SIZE as u32, y_offset + y as u32, image::Rgb([30, 30, 40]));
            }

            // 3. Ground Truth 256x256 Grayscale
            let x_target_base = x_pred_base + TARGET_SIZE as u32 + 1;
            for y in 0..TARGET_SIZE {
                for x in 0..TARGET_SIZE {
                    let t_idx = y * TARGET_SIZE + x;
                    let val = (((target_slice[t_idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                    grid_img.put_pixel(x_target_base + x as u32, y_offset + y as u32, image::Rgb([val, val, val]));
                }
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
        args.res_blocks = cfg.res_blocks;
        args.codebook = cfg.codebook;
        args.scales_from = cfg.scales_from;
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
    println!("     Yumon Pass 4/5 Upscaler Trainer (256x256)");
    println!("==================================================");
    println!("  Cache binary    : {}", args.cache.display());
    println!("  Output directory: {}", args.out_dir.display());
    println!("  Epochs          : {}", args.epochs);
    println!("  Batch size      : {}", args.batch_size);
    println!("  Channels        : {}", args.channels);
    println!("  Res blocks      : {}", args.res_blocks);
    println!("  Architecture    : {}", if args.codebook { "FSQ Codebook Quantized (Zero-Blur)" } else { "Continuous ResNet" });
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

    let val_count = ((total_samples as f64) * args.val_fraction).max(10.0) as usize;
    let train_count = total_samples.saturating_sub(val_count);

    let mut indices: Vec<usize> = (0..total_samples).collect();
    let mut rng = thread_rng();
    indices.shuffle(&mut rng);

    let val_indices = indices[train_count..].to_vec();
    let train_indices = indices[..train_count].to_vec();

    println!("  Train samples   : {}", train_indices.len());
    println!("  Val samples     : {}", val_indices.len());
    println!("  Model resolution: {}x{}x{} -> {}x256x256 ({})", dataset.channels, scales_from, scales_from, dataset.channels, if dataset.channels == 3 { "RGB Color" } else { "Grayscale" });

    if args.codebook {
        let config = CodebookUpscalerModelConfig::new()
            .with_scales_from(scales_from)
            .with_channels(args.channels)
            .with_in_channels(dataset.channels)
            .with_out_channels(dataset.channels)
            .with_num_res_blocks(args.res_blocks);
        let model = config.init(&device);
        run_training(model, &args, &dataset, scales_from, train_indices, val_indices, &device, true)?;
    } else {
        let config = UpscalerModelConfig::new()
            .with_scales_from(scales_from)
            .with_channels(args.channels)
            .with_in_channels(dataset.channels)
            .with_out_channels(dataset.channels)
            .with_num_res_blocks(args.res_blocks);
        let model = config.init(&device);
        run_training(model, &args, &dataset, scales_from, train_indices, val_indices, &device, false)?;
    }

    Ok(())
}

fn run_training<M: TrainableUpscaler>(
    mut model: M,
    args: &Args,
    dataset: &InMemDataset,
    scales_from: usize,
    train_indices: Vec<usize>,
    val_indices: Vec<usize>,
    device: &<TrainBackend as Backend>::Device,
    is_codebook: bool,
) -> Result<()> {
    let mut rng = thread_rng();
    let loss_fn = HuberLossConfig::new(1.0).init();
    let val_loss_fn = HuberLossConfig::new(1.0).init();

    let mut optimizer = AdamWConfig::new()
        .with_epsilon(1e-8)
        .with_weight_decay(1e-4)
        .init();

    let (mut discriminator, mut optimizer_d) = if args.gan {
        let d_config = UpscalerDiscriminatorConfig::new()
            .with_base_channels(args.d_channels)
            .with_in_channels(dataset.channels);
        let disc: UpscalerDiscriminator<TrainBackend> = d_config.init(device);
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

            // let should_inspect = (batch_count + 1) % 10 == 0 || (batch_count + 1) == num_batches;
            let should_inspect = true;

            // High-speed RAM batch slicing (Zero Disk I/O)
            let (input_tensor, target_tensor) = dataset.make_batch_tensors(chunk, device);

            // Forward pass: Generator produces 1x256x256 grayscale image
            let pred = model.forward_train(input_tensor);

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

        // 4. Validation Loop (using model.valid() to prevent autodiff VRAM buildup)
        let eval_count = val_indices.len().min(128);
        let mut val_loss_sum = 0.0f32;
        let mut val_batches = 0;
        let inference_model = model.to_valid();

        for chunk in val_indices[..eval_count].chunks(args.batch_size) {
            let (input_tensor, target_tensor) = dataset.make_batch_tensors(chunk, device);
            let pred = M::forward_valid(&inference_model, input_tensor);
            let loss = val_loss_fn.forward(pred, target_tensor, Reduction::Mean);
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
        let (preview_in, _) = dataset.make_batch_tensors(preview_indices, device);
        let preview_pred = M::forward_valid(&inference_model, preview_in);

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
            in_channels: dataset.channels,
            out_channels: dataset.channels,
            scales_from,
            gan: args.gan,
            codebook: is_codebook,
            fsq_levels: if is_codebook { Some(default_fsq_levels()) } else { None },
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
