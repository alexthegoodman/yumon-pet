//! Evaluate and Run Pass 6 Colorizer (Full Zhang et al. 2016).
//!
//! Features:
//! - Loads a trained Colorizer checkpoint (`checkpoints/yumon-colorizer-256/model`).
//! - Colorizes arbitrary input grayscale images (e.g. from Pass 5 upscaler or real photos).
//! - Adjustable temperature $T$ for annealed mean softmax ($T = 0.38$ default).
//! - Evaluates validation metrics on binary cache or test images.

use anyhow::{Context, Result};
use burn::{
    module::Module,
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{Tensor, TensorData},
};
use clap::Parser;
use std::{
    fs::File,
    io::Read,
    path::PathBuf,
    time::Instant,
};

use yumon_pet::vision::colorizer::{
    colorize_fused_rgb, ColorizerModel, ColorizerModelConfig, COLOR_BINS_AB,
    DEFAULT_TEMPERATURE, INPUT_SIZE, NUM_COLOR_BINS, TARGET_GRID,
};

pub type EvalBackend = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;

#[derive(Parser, Debug)]
#[command(about = "Evaluate and Run Pass 6 Colorizer (Zhang et al. 2016)")]
struct Args {
    /// Checkpoint directory containing model.
    #[arg(long, default_value = "checkpoints/yumon-colorizer-256")]
    checkpoint: PathBuf,

    /// Optional input grayscale image to colorize.
    #[arg(long)]
    input: Option<PathBuf>,

    /// Output path for colorized image.
    #[arg(long, default_value = "colorized_output.png")]
    out: PathBuf,

    /// Optional path to binary cache file to evaluate metrics & preview grid.
    #[arg(long)]
    cache: Option<PathBuf>,

    /// Number of evaluation samples from cache.
    #[arg(long, default_value_t = 16)]
    samples: usize,

    /// Feature channels in the model.
    #[arg(long, default_value_t = 64)]
    channels: usize,

    /// Number of residual blocks in model.
    #[arg(long, default_value_t = 5)]
    num_res_blocks: usize,

    /// Softmax annealing temperature (default 0.38).
    #[arg(long, default_value_t = DEFAULT_TEMPERATURE)]
    temperature: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device: <EvalBackend as Backend>::Device = Default::default();

    println!("==================================================");
    println!("     Pass 6 Colorizer Evaluator (Zhang 2016)");
    println!("==================================================");
    println!("  Checkpoint      : {}", args.checkpoint.display());
    println!("  Temperature (T) : {:.2}", args.temperature);

    // 1. Initialize and load model
    let config = ColorizerModelConfig::new()
        .with_scales_from(INPUT_SIZE)
        .with_channels(args.channels)
        .with_num_res_blocks(args.num_res_blocks);

    let model: ColorizerModel<EvalBackend> = config.init(&device);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    let model_path = args.checkpoint.join("model");

    let model = model
        .load_file(&model_path, &recorder, &device)
        .with_context(|| format!("Failed to load model from {}", model_path.display()))?;
    println!("  ✓ Loaded model checkpoint from {}", model_path.display());

    // Case A: Colorize single input image
    if let Some(ref in_path) = args.input {
        println!("\nColorizing input image: {}", in_path.display());
        let img = image::open(in_path)
            .with_context(|| format!("Failed to open image: {}", in_path.display()))?
            .to_rgb8();

        let img_256 = if img.width() == (INPUT_SIZE as u32) && img.height() == (INPUT_SIZE as u32) {
            img
        } else {
            image::imageops::resize(&img, INPUT_SIZE as u32, INPUT_SIZE as u32, image::imageops::FilterType::Triangle)
        };

        let mut raw_l = Vec::with_capacity(INPUT_SIZE * INPUT_SIZE);
        let mut flat_in = Vec::with_capacity(INPUT_SIZE * INPUT_SIZE);

        for p in img_256.pixels() {
            let lum = ((0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32).round().clamp(0.0, 255.0)) as u8;
            raw_l.push(lum);
            flat_in.push((lum as f32 / 255.0) * 2.0 - 1.0);
        }

        let in_tensor = Tensor::from_floats(
            TensorData::new(flat_in, [1, 1, INPUT_SIZE, INPUT_SIZE]),
            &device,
        );

        let t0 = Instant::now();
        let (pred_a, pred_b) = model.predict_ab_annealed(in_tensor, args.temperature);
        let colorized = colorize_fused_rgb(&raw_l, &pred_a, &pred_b, INPUT_SIZE);
        println!("  ✓ Colorized in {:.2?}", t0.elapsed());

        colorized.save(&args.out)?;
        println!("  ✓ Saved colorized image to {}", args.out.display());
    }

    // Case B: Evaluate from cache file
    if let Some(ref cache_path) = args.cache {
        println!("\nEvaluating on cache: {}", cache_path.display());
        let mut file = File::open(cache_path)
            .with_context(|| format!("Failed to open cache: {}", cache_path.display()))?;

        let mut header = [0u8; 32 + (NUM_COLOR_BINS * 4)];
        file.read_exact(&mut header)?;

        let total_samples = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
        let record_size = u32::from_le_bytes(header[16..20].try_into().unwrap()) as usize;
        let num_eval = args.samples.min(total_samples);

        println!("  Evaluating {} samples...", num_eval);
        let mut raw_bytes = vec![0u8; num_eval * record_size];
        file.read_exact(&mut raw_bytes)?;

        let triplet_w = (INPUT_SIZE * 3 + 2) as u32;
        let triplet_h = INPUT_SIZE as u32;
        let cell_margin = 8u32;
        let cols = 4u32.min(num_eval as u32);
        let rows = ((num_eval as u32) + cols - 1) / cols;

        let img_w = cell_margin + cols * (triplet_w + cell_margin);
        let img_h = cell_margin + rows * (triplet_h + cell_margin);
        let mut grid_img = image::RgbImage::from_pixel(img_w, img_h, image::Rgb([18, 18, 26]));

        let in_len = INPUT_SIZE * INPUT_SIZE;
        let tgt_len = TARGET_GRID * TARGET_GRID;
        let pts = &COLOR_BINS_AB;

        for i in 0..num_eval {
            let rec_offset = i * record_size;
            let l_slice = &raw_bytes[rec_offset..rec_offset + in_len];
            let tgt_slice = &raw_bytes[rec_offset + in_len..rec_offset + record_size];

            let r = (i as u32) / cols;
            let c = (i as u32) % cols;
            let x_offset = cell_margin + c * (triplet_w + cell_margin);
            let y_offset = cell_margin + r * (triplet_h + cell_margin);

            // 1. Grayscale
            for y in 0..INPUT_SIZE {
                for x in 0..INPUT_SIZE {
                    let val = l_slice[y * INPUT_SIZE + x];
                    grid_img.put_pixel(x_offset + x as u32, y_offset + y as u32, image::Rgb([val, val, val]));
                }
            }

            // 2. Predicted Colorized
            let flat_in: Vec<f32> = l_slice.iter().map(|&b| (b as f32 / 255.0) * 2.0 - 1.0).collect();
            let in_tensor = Tensor::from_floats(
                TensorData::new(flat_in, [1, 1, INPUT_SIZE, INPUT_SIZE]),
                &device,
            );
            let (pred_a, pred_b) = model.predict_ab_annealed(in_tensor, args.temperature);
            let pred_img = colorize_fused_rgb(l_slice, &pred_a, &pred_b, INPUT_SIZE);

            let x_pred = x_offset + INPUT_SIZE as u32 + 1;
            for y in 0..INPUT_SIZE {
                for x in 0..INPUT_SIZE {
                    let p = pred_img.get_pixel(x as u32, y as u32);
                    grid_img.put_pixel(x_pred + x as u32, y_offset + y as u32, *p);
                }
            }

            // 3. Target Reconstructed
            let mut target_a = vec![0.0f32; tgt_len];
            let mut target_b = vec![0.0f32; tgt_len];
            for k in 0..tgt_len {
                let bin = u16::from_le_bytes([tgt_slice[k * 2], tgt_slice[k * 2 + 1]]) as usize;
                target_a[k] = pts[bin][0];
                target_b[k] = pts[bin][1];
            }
            let target_img = colorize_fused_rgb(l_slice, &target_a, &target_b, INPUT_SIZE);

            let x_target = x_pred + INPUT_SIZE as u32 + 1;
            for y in 0..INPUT_SIZE {
                for x in 0..INPUT_SIZE {
                    let p = target_img.get_pixel(x as u32, y as u32);
                    grid_img.put_pixel(x_target + x as u32, y_offset + y as u32, *p);
                }
            }
        }

        grid_img.save(&args.out)?;
        println!("  ✓ Saved comparison grid to {}", args.out.display());
    }

    Ok(())
}
