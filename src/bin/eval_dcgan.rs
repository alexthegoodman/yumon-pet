//! Sample images from a trained DCGAN generator and save them as a grid PNG.

use anyhow::{Context, Result};
use burn::{
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{Tensor, TensorData},
};
use clap::Parser;
use rand::{Rng, SeedableRng};
use std::path::PathBuf;

use yumon_pet::vision::dcgan_model::{
    DcganGenerator, DcganGeneratorConfig, DCGAN_CHANNELS, DCGAN_IMAGE_SIZE, DCGAN_NOISE_DIM,
};

pub type EvalBackend = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;

#[derive(Parser, Debug)]
#[command(about = "Sample images from a trained DCGAN generator")]
struct Args {
    /// Checkpoint directory containing a saved `generator` (and `metadata.json`).
    #[arg(long, default_value = "checkpoints/yumon-dcgan")]
    checkpoint: PathBuf,

    /// Output PNG path.
    #[arg(long, default_value = "dcgan_samples.png")]
    out: PathBuf,

    /// Number of samples to generate (arranged in a square grid).
    #[arg(long, default_value_t = 64)]
    count: usize,

    /// Latent noise dimension (overrides metadata if set).
    #[arg(long)]
    latent_dim: Option<usize>,

    /// Random seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device: <EvalBackend as Backend>::Device = Default::default();

    println!("=== Yumon DCGAN Sampler ===");
    println!("  Checkpoint : {}", args.checkpoint.display());
    println!("  Output     : {}", args.out.display());
    println!("  Samples    : {}", args.count);

    // Read latent dim + image size from metadata if present.
    let (latent_dim, image_size) = load_meta(&args.checkpoint, args.latent_dim);

    let generator: DcganGenerator<EvalBackend> = DcganGeneratorConfig::new()
        .with_noise_dim(latent_dim)
        .with_out_channels(DCGAN_CHANNELS)
        .with_image_size(image_size)
        .init(&device);

    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    let generator = generator
        .load_file(&args.checkpoint.join("generator"), &recorder, &device)
        .with_context(|| format!("failed to load generator from {}", args.checkpoint.display()))?;

    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
    let flat: Vec<f32> = (0..args.count * latent_dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
    let z: Tensor<EvalBackend, 2> = Tensor::from_floats(TensorData::new(flat, [args.count, latent_dim]), &device);

    let fake = generator.forward(z);
    let data: Vec<f32> = fake.into_data().convert::<f32>().to_vec().unwrap();

    let cols = (args.count as f32).sqrt().ceil() as u32;
    let rows = ((args.count as u32) + cols - 1) / cols;
    let margin = 2u32;
    let tile = image_size as u32;
    let plane = image_size * image_size;

    let img_w = margin + cols * (tile + margin);
    let img_h = margin + rows * (tile + margin);
    let mut grid = image::RgbImage::from_pixel(img_w, img_h, image::Rgb([18, 18, 26]));

    for i in 0..args.count {
        let row = (i as u32) / cols;
        let col = (i as u32) % cols;
        let x0 = margin + col * (tile + margin);
        let y0 = margin + row * (tile + margin);
        let base = i * 3 * plane;
        for y in 0..image_size {
            for x in 0..image_size {
                let p = y * image_size + x;
                let r = ((data[base + p] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0) as u8;
                let g = ((data[base + plane + p] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0) as u8;
                let b = ((data[base + 2 * plane + p] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0) as u8;
                grid.put_pixel(x0 + x as u32, y0 + y as u32, image::Rgb([r, g, b]));
            }
        }
    }

    grid.save(&args.out)?;
    println!("  ✓ Saved {} samples to {}", args.count, args.out.display());
    Ok(())
}

fn load_meta(checkpoint: &PathBuf, override_latent: Option<usize>) -> (usize, usize) {
    let mut latent_dim = DCGAN_NOISE_DIM;
    let mut image_size = DCGAN_IMAGE_SIZE;

    if let Some(l) = override_latent {
        latent_dim = l;
    }

    let meta_path = checkpoint.join("metadata.json");
    if let Ok(content) = std::fs::read_to_string(&meta_path) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
            if override_latent.is_none() {
                if let Some(l) = v.get("latent_dim").and_then(|x| x.as_u64()) {
                    latent_dim = l as usize;
                }
            }
            if let Some(s) = v.get("image_size").and_then(|x| x.as_u64()) {
                image_size = s as usize;
            }
        }
    }

    (latent_dim, image_size)
}
