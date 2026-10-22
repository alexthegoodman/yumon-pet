// train_dit.rs - Standalone Vision DiT (64x64) generator CLI: training, overfitting, and 30-step Euler sampling

use std::path::PathBuf;
use std::time::Instant;
use anyhow::{Context, Result};
use clap::Parser;

use burn::backend::autodiff::{Autodiff, checkpoint::strategy::BalancedCheckpointing};
use burn_cubecl::CubeBackend;
use cubecl::wgpu::WgpuRuntime;

use yumon_pet::vision::{
    discover_images, load_image_batch, save_batch_as_pngs,
    DiTTrainer, EulerSampler, VisionDiTConfig,
};

type TrainBackend = Autodiff<CubeBackend<WgpuRuntime, f32, i32, u32>, BalancedCheckpointing>;

#[derive(Parser, Debug)]
#[command(name = "train_dit", about = "Vision DiT 64x64 Generation CLI")]
struct Args {
    /// Action to execute: "overfit", "train", or "sample"
    #[arg(short, long, default_value = "overfit")]
    action: String,

    /// Directory containing 64x64 PNG images
    #[arg(long, default_value = "data/DiffusionDb/images-64")]
    data_dir: PathBuf,

    /// Output directory for sample images
    #[arg(long, default_value = "dist/dit_samples")]
    output_dir: PathBuf,

    /// Checkpoint file path
    #[arg(long, default_value = "checkpoints/dit_ema.bin")]
    checkpoint: PathBuf,

    /// Preset architecture: "micro" or "tiny"
    #[arg(long, default_value = "micro")]
    preset: String,

    /// Batch size for training
    #[arg(long, default_value_t = 8)]
    batch_size: usize,

    /// Number of epochs for dataset training
    #[arg(long, default_value_t = 5)]
    epochs: usize,

    /// Steps for overfit test
    #[arg(long, default_value_t = 60)]
    steps: usize,

    /// Learning rate
    #[arg(long, default_value_t = 2e-4)]
    lr: f64,

    /// Weight decay
    #[arg(long, default_value_t = 0.01)]
    weight_decay: f32,

    /// Gradient clipping norm
    #[arg(long, default_value_t = 1.0)]
    grad_clip: f32,

    /// Number of Euler sampling steps
    #[arg(long, default_value_t = 30)]
    sample_steps: usize,

    /// Number of samples to generate in sample mode
    #[arg(long, default_value_t = 4)]
    num_samples: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("=== Vision DiT 64x64 Generative Model ===");
    println!("Action: {}", args.action);
    println!("Preset: {}", args.preset);

    let config = match args.preset.to_lowercase().as_str() {
        "tiny" => VisionDiTConfig::tiny(),
        _ => VisionDiTConfig::micro(),
    };

    let device = Default::default();

    match args.action.as_str() {
        "overfit" => {
            println!("--- Running Single-Batch Overfit Test ---");
            let all_images = discover_images(&args.data_dir)
                .with_context(|| format!("discovering images in {:?}", args.data_dir))?;
            anyhow::ensure!(!all_images.is_empty(), "No images found in {:?}", args.data_dir);

            let n_overfit = args.batch_size.min(all_images.len()).min(8);
            let selected_paths: Vec<_> = all_images.into_iter().take(n_overfit).collect();
            println!("Loaded {} images for overfitting.", selected_paths.len());

            std::fs::create_dir_all(&args.output_dir)?;
            for (i, p) in selected_paths.iter().enumerate() {
                let target = args.output_dir.join(format!("target_{:03}.png", i));
                std::fs::copy(p, &target)?;
            }
            println!("Saved reference targets to {:?}", args.output_dir);

            let batch_tensor = load_image_batch::<TrainBackend>(&selected_paths, &device)?;

            let mut trainer = DiTTrainer::<TrainBackend>::new(
                &config,
                args.lr,
                0.0, // no weight decay for clean overfit
                args.grad_clip,
                0.999,
                &device,
            );

            let start_time = Instant::now();
            let mut first_loss = 0.0f32;
            let mut final_loss = 0.0f32;

            for step in 1..=args.steps {
                let loss = trainer.step(batch_tensor.clone());
                if step == 1 {
                    first_loss = loss;
                }
                final_loss = loss;

                if step % 10 == 0 || step == 1 || step == args.steps {
                    println!("Step {:03}/{:03} | Flow Matching MSE Loss: {:.6}", step, args.steps, loss);
                }
            }

            println!(
                "Overfitting complete in {:.2}s. First loss: {:.6}, Final loss: {:.6}",
                start_time.elapsed().as_secs_f32(),
                first_loss,
                final_loss,
            );

            // 30-step Euler ODE sampling from EMA shadow weights
            println!("Generating {}-step Euler samples from EMA weights...", args.sample_steps);
            let sampler = EulerSampler::new(args.sample_steps);
            let generated = sampler.sample(&trainer.ema.ema_model, n_overfit, &device);

            let saved = save_batch_as_pngs(generated, &args.output_dir, "reconstructed")?;
            println!("Exported {} samples to {:?}", saved.len(), args.output_dir);

            // Save EMA checkpoint
            trainer.ema.save_checkpoint(&args.checkpoint)?;
            println!("Saved EMA checkpoint to {:?}", args.checkpoint);
        }

        "train" => {
            println!("--- Full Dataset Training ---");
            let all_images = discover_images(&args.data_dir)?;
            anyhow::ensure!(!all_images.is_empty(), "No images found in {:?}", args.data_dir);
            println!("Found {} images in {:?}", all_images.len(), args.data_dir);

            let mut trainer = DiTTrainer::<TrainBackend>::new(
                &config,
                args.lr,
                args.weight_decay,
                args.grad_clip,
                0.999,
                &device,
            );

            let sampler = EulerSampler::new(args.sample_steps);
            std::fs::create_dir_all(&args.output_dir)?;

            for epoch in 1..=args.epochs {
                let mut epoch_loss = 0.0f32;
                let mut n_batches = 0;
                let epoch_start = Instant::now();

                for chunk in all_images.chunks(args.batch_size) {
                    if chunk.len() < args.batch_size {
                        continue;
                    }
                    let batch = load_image_batch::<TrainBackend>(chunk, &device)?;
                    let loss = trainer.step(batch);
                    epoch_loss += loss;
                    n_batches += 1;

                    if n_batches % 20 == 0 {
                        println!("Epoch {} | Batch {} | Loss: {:.6}", epoch, n_batches, loss);
                    }
                }

                let avg_loss = if n_batches > 0 { epoch_loss / n_batches as f32 } else { 0.0 };
                println!(
                    "Epoch {}/{} complete in {:.2}s | Avg Loss: {:.6}",
                    epoch, args.epochs, epoch_start.elapsed().as_secs_f32(), avg_loss,
                );

                // Periodic checkpoint
                trainer.ema.save_checkpoint(&args.checkpoint)?;
                println!("Saved EMA checkpoint to {:?}", args.checkpoint);

                // Sample images
                let generated = sampler.sample(&trainer.ema.ema_model, 4, &device);
                save_batch_as_pngs(generated, &args.output_dir, &format!("epoch_{:03}", epoch))?;
            }
        }

        "sample" => {
            println!("--- Generating Samples via 30-step Euler ODE ---");
            let model = if args.checkpoint.exists() {
                println!("Loading EMA checkpoint from {:?}", args.checkpoint);
                yumon_pet::vision::EmaTracker::<TrainBackend>::load_checkpoint(&config, &args.checkpoint, &device)?
            } else {
                println!("No checkpoint found at {:?}, using initialized model.", args.checkpoint);
                config.init(&device)
            };

            let sampler = EulerSampler::new(args.sample_steps);
            let samples = sampler.sample(&model, args.num_samples, &device);
            std::fs::create_dir_all(&args.output_dir)?;
            let paths = save_batch_as_pngs(samples, &args.output_dir, "sample")?;
            println!("Saved {} samples to {:?}", paths.len(), args.output_dir);
        }

        unknown => {
            anyhow::bail!("Unknown action: '{}'. Valid actions: overfit, train, sample", unknown);
        }
    }

    Ok(())
}
