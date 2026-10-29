//! Train a vanilla Deep Convolutional GAN (DCGAN) at 64x64.
//!
//! This is a *single-pass*, unconditional generator: it samples latent noise
//! `z ~ N(0, 1)` and maps it directly to a 64x64 RGB image, following the
//! original Radford et al. (2015) DCGAN architecture and recipe.
//!
//! It reuses the same 64x64 art-image directory as the other vision pipelines
//! (`data/animals_images/images-64` by default), but unlike `cache_diffusion_kilo`
//! it does *not* tokenize images: raw RGB pixels are fed straight into the
//! discriminator as the "real" distribution.

use anyhow::{ensure, Context, Result};
use burn::{
    backend::Autodiff,
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{Tensor, TensorData},
};
use clap::Parser;
use image::imageops::FilterType;
use indicatif::{ProgressBar, ProgressStyle};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use ratatui::{backend::CrosstermBackend, Terminal, TerminalOptions, Viewport};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{stdout, IsTerminal},
    path::{Path, PathBuf},
    time::Instant,
};

use yumon_pet::vision::chart::{render_dcgan_chart, DcganTrainingState};
use yumon_pet::vision::dcgan_model::{
    discriminator_loss, generator_loss, sample_noise, sample_noise_vec,
    DcganDiscriminator, DcganDiscriminatorConfig, DcganGenerator, DcganGeneratorConfig,
    DCGAN_CHANNELS, DCGAN_IMAGE_SIZE, DCGAN_NOISE_DIM,
};

pub type InnerBackend = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;
pub type TrainBackend = Autodiff<InnerBackend>;

#[derive(Parser, Debug)]
#[command(about = "Train a vanilla DCGAN (64x64 RGB)")]
struct Args {
    /// Directory containing 64x64 training images (PNG/JPG/JPEG/WebP).
    #[arg(long, default_value = "data/animals_images/images-64")]
    images_dir: PathBuf,

    /// Output checkpoint directory.
    #[arg(long, default_value = "checkpoints/yumon-dcgan")]
    out_dir: PathBuf,

    /// Number of training epochs.
    #[arg(long, default_value_t = 100)]
    epochs: usize,

    /// Batch size.
    #[arg(long, default_value_t = 64)]
    batch_size: usize,

    /// Base learning rate for both generator and discriminator (Adam).
    #[arg(long, default_value_t = 2e-4)]
    lr: f64,

    /// Generator initial learning rate (defaults to --lr).
    #[arg(long)]
    lr_g: Option<f64>,

    /// Discriminator initial learning rate (defaults to --lr).
    #[arg(long)]
    lr_d: Option<f64>,

    /// Minimum learning rate floor for schedules (defaults to 1e-5).
    #[arg(long, default_value_t = 1e-5)]
    min_lr: f64,

    /// Generator minimum learning rate floor (defaults to --min-lr).
    #[arg(long)]
    min_lr_g: Option<f64>,

    /// Discriminator minimum learning rate floor (defaults to --min-lr).
    #[arg(long)]
    min_lr_d: Option<f64>,

    /// Learning rate schedule: cosine, linear, half-decay, constant.
    #[arg(long, default_value = "cosine")]
    lr_schedule: String,

    /// Epoch to start decay (for half-decay or custom schedule).
    #[arg(long)]
    decay_start_epoch: Option<usize>,

    /// Adam beta1 (DCGAN uses 0.5 instead of the usual 0.9).
    #[arg(long, default_value_t = 0.5)]
    beta1: f32,

    /// Latent noise dimension (nz).
    #[arg(long, default_value_t = DCGAN_NOISE_DIM)]
    latent_dim: usize,

    /// Generator feature-map width (ngf).
    #[arg(long, default_value_t = 64)]
    ngf: usize,

    /// Discriminator feature-map width (ndf).
    #[arg(long, default_value_t = 64)]
    ndf: usize,

    /// Output image resolution (square). 64 is the DCGAN standard.
    #[arg(long, default_value_t = DCGAN_IMAGE_SIZE)]
    image_size: usize,

    /// Maximum images to load (0 = all).
    #[arg(long, default_value_t = 0)]
    limit: usize,

    /// Random seed for shuffling and noise sampling.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Disable inline interactive terminal TUI dashboard.
    #[arg(long, default_value_t = false)]
    no_tui: bool,

    /// Resume training from checkpoint in out-dir if present (or explicitly pass --resume).
    #[arg(long, default_value_t = false)]
    resume: bool,

    /// Explicit checkpoint directory to resume from (defaults to out-dir).
    #[arg(long)]
    resume_from: Option<PathBuf>,

    /// Start fresh even if existing checkpoints exist in out-dir.
    #[arg(long, default_value_t = false)]
    fresh: bool,
}

#[derive(Serialize, Deserialize, Debug)]
struct DcganMetadata {
    epoch: usize,
    epochs: usize,
    g_loss: f32,
    d_loss: f32,
    lr: f64,
    #[serde(default)]
    lr_g: Option<f64>,
    #[serde(default)]
    lr_d: Option<f64>,
    beta1: f32,
    latent_dim: usize,
    ngf: usize,
    ndf: usize,
    image_size: usize,
    num_images: usize,
    timestamp: String,
}

/// Compute scheduled learning rate based on current epoch and policy.
fn compute_scheduled_lr(
    base_lr: f64,
    min_lr: f64,
    epoch: usize,
    total_epochs: usize,
    schedule: &str,
    decay_start: usize,
) -> f64 {
    if total_epochs <= 1 || base_lr <= min_lr {
        return base_lr;
    }
    match schedule.to_ascii_lowercase().as_str() {
        "cosine" => {
            let progress = (epoch.saturating_sub(1)) as f64 / (total_epochs - 1) as f64;
            let factor = 0.5 * (1.0 + (progress * std::f64::consts::PI).cos());
            min_lr + (base_lr - min_lr) * factor
        }
        "linear" => {
            let progress = (epoch.saturating_sub(1)) as f64 / (total_epochs - 1) as f64;
            (base_lr + (min_lr - base_lr) * progress).max(min_lr)
        }
        "half-decay" | "half_decay" | "half" => {
            let start = if decay_start > 0 {
                decay_start
            } else {
                (total_epochs / 2).max(1)
            };
            if epoch <= start {
                base_lr
            } else {
                let remaining = total_epochs.saturating_sub(start).max(1);
                let progress = (epoch - start) as f64 / remaining as f64;
                (base_lr + (min_lr - base_lr) * progress.clamp(0.0, 1.0)).max(min_lr)
            }
        }
        "constant" | "none" => base_lr,
        _ => {
            let progress = (epoch.saturating_sub(1)) as f64 / (total_epochs - 1) as f64;
            let factor = 0.5 * (1.0 + (progress * std::f64::consts::PI).cos());
            min_lr + (base_lr - min_lr) * factor
        }
    }
}

/// In-memory dataset of normalized `[C, H, W]` planar `[-1, 1]` pixels.
struct ImageDataset {
    count: usize,
    channels: usize,
    image_size: usize,
    plane_len: usize,
    /// Flat `count * channels * plane_len` buffer.
    data: Vec<f32>,
}

impl ImageDataset {
    fn load(args: &Args) -> Result<Self> {
        ensure!(
            args.images_dir.exists(),
            "Images directory not found: {}",
            args.images_dir.display()
        );

        let mut files: Vec<PathBuf> = fs::read_dir(&args.images_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|s| s.to_str())
                    .map(|e| matches!(e.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg" | "webp"))
                    .unwrap_or(false)
            })
            .collect();
        files.sort();

        if args.limit > 0 && args.limit < files.len() {
            files.truncate(args.limit);
        }
        ensure!(!files.is_empty(), "No images found in {}", args.images_dir.display());

        let size = args.image_size;
        let plane_len = size * size;
        let channels = DCGAN_CHANNELS;
        let img_len = channels * plane_len;

        let pb = ProgressBar::new(files.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );

        let t0 = Instant::now();
        let mut data = vec![0.0f32; files.len() * img_len];

        data.par_chunks_exact_mut(img_len)
            .zip(files.par_iter())
            .for_each(|(slot, path)| {
                pb.inc(1);
                let Ok(img) = image::open(path) else {
                    slot.fill(0.0);
                    return;
                };
                let rgb = img.resize_exact(size as u32, size as u32, FilterType::Triangle).to_rgb8();
                let raw = rgb.into_raw(); // HWC, length 3 * plane_len
                for p in 0..plane_len {
                    for c in 0..channels {
                        let v = raw[p * channels + c] as f32;
                        slot[c * plane_len + p] = (v / 255.0) * 2.0 - 1.0;
                    }
                }
            });

        pb.finish_with_message("done");

        println!(
            "  ✓ Loaded {} images ({}x{}x{}) into RAM in {:.2?} ({:.1} MB)",
            files.len(),
            channels,
            size,
            size,
            t0.elapsed(),
            (files.len() * img_len * 4) as f64 / 1e6
        );

        Ok(Self {
            count: files.len(),
            channels,
            image_size: size,
            plane_len,
            data,
        })
    }

    fn batch_tensor<B: Backend>(&self, indices: &[usize], device: &B::Device) -> Tensor<B, 4> {
        let img_len = self.channels * self.plane_len;
        let mut flat = Vec::with_capacity(indices.len() * img_len);
        for &idx in indices {
            let start = idx * img_len;
            flat.extend_from_slice(&self.data[start..start + img_len]);
        }
        Tensor::from_floats(
            TensorData::new(flat, [indices.len(), self.channels, self.image_size, self.image_size]),
            device,
        )
    }
}


/// Save a square grid of generated samples (default 8x8 = 64 images).
fn save_sample_grid(
    fake: Tensor<InnerBackend, 4>,
    image_size: usize,
    out_path: &Path,
) -> Result<()> {
    let [n, c, h, w] = fake.dims();
    let data: Vec<f32> = fake.into_data().convert::<f32>().to_vec().unwrap();
    let cols = (n as f32).sqrt().ceil() as u32;
    let rows = ((n as u32) + cols - 1) / cols;

    let margin = 2u32;
    let tile = image_size as u32;
    let img_w = margin + cols * (tile + margin);
    let img_h = margin + rows * (tile + margin);
    let mut grid = image::RgbImage::from_pixel(img_w, img_h, image::Rgb([18, 18, 26]));

    let plane = h * w;
    for i in 0..n {
        let row = (i as u32) / cols;
        let col = (i as u32) % cols;
        let x0 = margin + col * (tile + margin);
        let y0 = margin + row * (tile + margin);
        let base = i * c * plane;
        for y in 0..h {
            for x in 0..w {
                let p = y * w + x;
                let r = ((data[base + 0 * plane + p] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0) as u8;
                let (g, b) = if c >= 3 {
                    (
                        ((data[base + 1 * plane + p] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0) as u8,
                        ((data[base + 2 * plane + p] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0) as u8,
                    )
                } else {
                    (r, r)
                };
                grid.put_pixel(x0 + x as u32, y0 + y as u32, image::Rgb([r, g, b]));
            }
        }
    }

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }
    grid.save(out_path)?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device: <TrainBackend as Backend>::Device = Default::default();

    println!("==================================================");
    println!("     Yumon Vanilla DCGAN Trainer (64x64 RGB)");
    println!("==================================================");
    println!("  Images dir  : {}", args.images_dir.display());
    println!("  Output dir  : {}", args.out_dir.display());
    println!("  Epochs      : {}", args.epochs);
    println!("  Batch size  : {}", args.batch_size);

    let base_lr_g = args.lr_g.unwrap_or(args.lr);
    let base_lr_d = args.lr_d.unwrap_or(args.lr);
    let min_lr_g = args.min_lr_g.unwrap_or(args.min_lr);
    let min_lr_d = args.min_lr_d.unwrap_or(args.min_lr);
    let decay_start = args.decay_start_epoch.unwrap_or(args.epochs / 2);

    println!("  LR (G)      : {:.2e} -> {:.2e} (schedule: {})", base_lr_g, min_lr_g, args.lr_schedule);
    println!("  LR (D)      : {:.2e} -> {:.2e} (schedule: {})", base_lr_d, min_lr_d, args.lr_schedule);
    println!("  Adam beta1  : {:.2}", args.beta1);
    println!("  Latent dim  : {}", args.latent_dim);
    println!("  ngf / ndf   : {} / {}", args.ngf, args.ndf);
    println!("  Image size  : {}x{}", args.image_size, args.image_size);

    let dataset = ImageDataset::load(&args)?;
    let num_images = dataset.count;

    fs::create_dir_all(&args.out_dir)?;

    // --- Models ---
    let gen_cfg = DcganGeneratorConfig::new()
        .with_noise_dim(args.latent_dim)
        .with_ngf(args.ngf)
        .with_out_channels(dataset.channels)
        .with_image_size(args.image_size);
    let disc_cfg = DcganDiscriminatorConfig::new()
        .with_in_channels(dataset.channels)
        .with_ndf(args.ndf)
        .with_image_size(args.image_size);

    let mut generator: DcganGenerator<TrainBackend> = gen_cfg.init(&device);
    let mut discriminator: DcganDiscriminator<TrainBackend> = disc_cfg.init(&device);

    let mut opt_g = AdamConfig::new()
        .with_beta_1(args.beta1)
        .with_beta_2(0.999)
        .init();
    let mut opt_d = AdamConfig::new()
        .with_beta_1(args.beta1)
        .with_beta_2(0.999)
        .init();

    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
    let mut indices: Vec<usize> = (0..num_images).collect();

    // Fixed latent vector for consistent visual comparison across epochs.
    let fixed_grid = 64usize;
    let fixed_noise = sample_noise_vec(fixed_grid * args.latent_dim, &mut rng);

    let num_batches = num_images / args.batch_size;
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();

    let mut last_g_loss = 0.0f32;
    let mut last_d_loss = 0.0f32;

    let mut tui_state = DcganTrainingState::with_lrs(args.epochs, num_batches, base_lr_g, base_lr_d);

    // --- Resume checkpoint support ---
    let resume_dir = args.resume_from.as_ref().unwrap_or(&args.out_dir);
    let gen_checkpoint = resume_dir.join("generator.bin");
    let disc_checkpoint = resume_dir.join("discriminator.bin");
    let has_checkpoint = gen_checkpoint.exists() && disc_checkpoint.exists();

    let should_resume = if args.fresh {
        false
    } else if args.resume || args.resume_from.is_some() {
        ensure!(
            has_checkpoint,
            "Cannot resume: checkpoint not found in {} (expected generator.bin and discriminator.bin)",
            resume_dir.display()
        );
        true
    } else {
        has_checkpoint
    };

    let mut start_epoch = 1usize;
    if should_resume {
        let meta_file = resume_dir.join("metadata.json");
        if meta_file.exists() {
            if let Ok(content) = fs::read_to_string(&meta_file) {
                if let Ok(meta) = serde_json::from_str::<DcganMetadata>(&content) {
                    start_epoch = meta.epoch + 1;
                    last_g_loss = meta.g_loss;
                    last_d_loss = meta.d_loss;
                    println!(
                        "  ↻ Resuming from epoch {}/{} (checkpoint: {})",
                        start_epoch,
                        args.epochs,
                        resume_dir.display()
                    );
                }
            }
        } else {
            println!(
                "  ↻ Resuming model weights from checkpoint: {}",
                resume_dir.display()
            );
        }

        if start_epoch > args.epochs {
            println!(
                "  Checkpoint in {} already completed all {} epochs! Set --epochs > {} to train further.",
                resume_dir.display(),
                args.epochs,
                start_epoch - 1
            );
            return Ok(());
        }

        print!("  Loading generator weights... ");
        generator = generator
            .load_file(resume_dir.join("generator"), &recorder, &device)
            .with_context(|| format!("Failed to load generator from {}", resume_dir.display()))?;
        println!("done");

        print!("  Loading discriminator weights... ");
        discriminator = discriminator
            .load_file(resume_dir.join("discriminator"), &recorder, &device)
            .with_context(|| format!("Failed to load discriminator from {}", resume_dir.display()))?;
        println!("done");

        let csv_path = resume_dir.join("loss_history.csv");
        if csv_path.exists() {
            if let Err(e) = tui_state.load_csv(&csv_path) {
                eprintln!("Warning: Failed to load previous loss_history.csv: {e}");
            } else {
                println!(
                    "  Loaded {} historical training steps from {}",
                    tui_state.g_loss_history.len(),
                    csv_path.display()
                );
            }
        }
        tui_state.epoch = start_epoch;
    }

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

    for epoch in start_epoch..=args.epochs {
        indices.shuffle(&mut rng);
        let t_epoch = Instant::now();
        tui_state.epoch = epoch;

        let lr_g = compute_scheduled_lr(
            base_lr_g,
            min_lr_g,
            epoch,
            args.epochs,
            &args.lr_schedule,
            decay_start,
        );
        let lr_d = compute_scheduled_lr(
            base_lr_d,
            min_lr_d,
            epoch,
            args.epochs,
            &args.lr_schedule,
            decay_start,
        );
        tui_state.current_lr = lr_g;
        tui_state.current_lr_d = lr_d;

        let pb = if terminal.is_none() {
            let bar = ProgressBar::new(num_batches as u64);
            bar.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Epoch {msg}] [{bar:30.cyan/blue}] {pos}/{len} ({eta}) G:{prefix}")
                    .unwrap()
                    .progress_chars("#>-"),
            );
            bar.set_message(format!(
                "{}/{} [LR G:{:.1e} D:{:.1e}]",
                epoch, args.epochs, lr_g, lr_d
            ));
            Some(bar)
        } else {
            None
        };

        let mut epoch_g = 0.0f32;
        let mut epoch_d = 0.0f32;
        let mut steps = 0usize;

        for (bi, chunk) in indices.chunks(args.batch_size).enumerate() {
            if chunk.len() < args.batch_size {
                continue;
            }
            let real = dataset.batch_tensor(chunk, &device);

            // --- Discriminator step ---
            let z = sample_noise(args.batch_size, args.latent_dim, &device, &mut rng);
            let fake = generator.forward(z);
            let d_real = discriminator.forward_logits(real);
            let d_fake = discriminator.forward_logits(fake.detach());
            let d_loss = discriminator_loss(d_real, d_fake);
            let d_loss_val: f32 = d_loss.clone().into_data().convert::<f32>().to_vec().unwrap()[0];

            let grads_d = GradientsParams::from_grads(d_loss.backward(), &discriminator);
            discriminator = opt_d.step(lr_d, discriminator, grads_d);

            // --- Generator step (fresh noise) ---
            let z2 = sample_noise(args.batch_size, args.latent_dim, &device, &mut rng);
            let fake2 = generator.forward(z2);
            let d_fake2 = discriminator.forward_logits(fake2);
            let g_loss = generator_loss(d_fake2);
            let g_loss_val: f32 = g_loss.clone().into_data().convert::<f32>().to_vec().unwrap()[0];

            let grads_g = GradientsParams::from_grads(g_loss.backward(), &generator);
            generator = opt_g.step(lr_g, generator, grads_g);

            epoch_g += g_loss_val;
            epoch_d += d_loss_val;
            steps += 1;
            last_g_loss = g_loss_val;
            last_d_loss = d_loss_val;

            let global_step = (epoch - 1) * num_batches + bi + 1;
            tui_state.batch = bi + 1;
            tui_state.global_step = global_step;
            tui_state.current_g_loss = g_loss_val;
            tui_state.current_d_loss = d_loss_val;
            tui_state.g_loss_history.push((global_step as f64, g_loss_val as f64));
            tui_state.d_loss_history.push((global_step as f64, d_loss_val as f64));
            tui_state.lr_history.push((global_step as f64, lr_g));
            tui_state.lr_d_history.push((global_step as f64, lr_d));

            if let Some(term) = terminal.as_mut() {
                let _ = term.draw(|f| render_dcgan_chart(f, &tui_state));
            } else if let Some(p) = pb.as_ref() {
                p.set_prefix(format!("{:.4}", g_loss_val));
                p.inc(1);

                if bi == 0 || bi == num_batches - 1 {
                    println!(
                        "  [{}] step {} G:{:.4} D:{:.4}",
                        epoch, bi, g_loss_val, d_loss_val
                    );
                }
            }
        }

        if let Some(p) = pb {
            p.finish_with_message(format!(
                "{}/{} [G:{:.4} D:{:.4}]",
                epoch, args.epochs, epoch_g / steps.max(1) as f32, epoch_d / steps.max(1) as f32
            ));
        }

        // --- Preview grid from fixed noise (inference mode) ---
        let gen_valid = generator.valid();
        let z_fixed: Tensor<InnerBackend, 2> = Tensor::from_floats(
            TensorData::new(fixed_noise.clone(), [fixed_grid, args.latent_dim]),
            &device,
        );
        let fake_fixed = gen_valid.forward(z_fixed);
        let preview_path = args.out_dir.join(format!("preview_epoch_{:02}.png", epoch));
        if let Err(e) = save_sample_grid(fake_fixed, args.image_size, &preview_path) {
            eprintln!("Warning: failed to save preview grid: {e}");
        } else {
            let _ = fs::copy(&preview_path, args.out_dir.join("preview_latest.png"));
            tui_state.last_preview_path = Some(preview_path.to_string_lossy().to_string());
        }

        // --- Save chart PNG and CSV ---
        let chart_path = args.out_dir.join("chart.png");
        let epoch_chart_path = args.out_dir.join(format!("chart_epoch_{:02}.png", epoch));
        if let Err(e) = tui_state.save_chart_image(&chart_path) {
            eprintln!("Warning: failed to save chart image: {e}");
        } else {
            let _ = fs::copy(&chart_path, &epoch_chart_path);
        }
        let _ = tui_state.save_csv(&args.out_dir.join("loss_history.csv"));

        if let Some(term) = terminal.as_mut() {
            let _ = term.draw(|f| render_dcgan_chart(f, &tui_state));
        }

        // --- Checkpoint ---
        generator.clone().save_file(args.out_dir.join("generator"), &recorder)?;
        discriminator.clone().save_file(args.out_dir.join("discriminator"), &recorder)?;

        let meta = DcganMetadata {
            epoch,
            epochs: args.epochs,
            g_loss: last_g_loss,
            d_loss: last_d_loss,
            lr: lr_g,
            lr_g: Some(lr_g),
            lr_d: Some(lr_d),
            beta1: args.beta1,
            latent_dim: args.latent_dim,
            ngf: args.ngf,
            ndf: args.ndf,
            image_size: args.image_size,
            num_images,
            timestamp: chrono::Local::now().to_rfc3339(),
        };
        fs::write(
            args.out_dir.join("metadata.json"),
            serde_json::to_string_pretty(&meta)?,
        )?;

        if terminal.is_none() {
            println!(
                "  Epoch {:>3}/{} ({:.1?}): avg G:{:.4} avg D:{:.4} | LR G:{:.2e} D:{:.2e} | saved {}",
                epoch,
                args.epochs,
                t_epoch.elapsed(),
                epoch_g / steps.max(1) as f32,
                epoch_d / steps.max(1) as f32,
                lr_g,
                lr_d,
                args.out_dir.display()
            );
        }
    }

    if let Some(mut term) = terminal {
        let _ = term.clear();
    }

    println!("\nTraining complete. Generator saved to {}", args.out_dir.join("generator").display());
    Ok(())
}
