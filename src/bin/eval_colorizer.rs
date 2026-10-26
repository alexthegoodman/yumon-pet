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
    fs::{self, File},
    io::Read,
    path::PathBuf,
    time::Instant,
};

use yumon_pet::vision::colorizer::{
    colorize_fused_rgb, srgb_to_lab, Colorizer, ColorizerModel, ColorizerModelConfig,
    TransformerColorizerModel, TransformerColorizerConfig, COLOR_BINS_AB,
    DEFAULT_TEMPERATURE, DEFAULT_CHROMA_BOOST, INPUT_SIZE, NUM_COLOR_BINS, TARGET_GRID,
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

    /// Softmax annealing temperature (default 0.20).
    #[arg(long, default_value_t = DEFAULT_TEMPERATURE)]
    temperature: f32,

    /// Saturation chroma boost multiplier to counteract expectation contraction (default 1.8).
    #[arg(long, default_value_t = DEFAULT_CHROMA_BOOST)]
    chroma_boost: f32,

    /// Model architecture type: "transformer" (default, non-autoregressive ViT) or "cnn".
    #[arg(long, default_value = "transformer")]
    model_type: String,

    /// Transformer feature dimension d_model.
    #[arg(long, default_value_t = 256)]
    d_model: usize,

    /// Transformer attention heads.
    #[arg(long, default_value_t = 8)]
    n_heads: usize,

    /// Transformer layers.
    #[arg(long, default_value_t = 6)]
    num_layers: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device: <EvalBackend as Backend>::Device = Default::default();

    println!("==================================================");
    println!("     Pass 6 Colorizer Evaluator (Zhang 2016)");
    println!("==================================================");
    println!("  Checkpoint      : {}", args.checkpoint.display());
    println!("  Temperature (T) : {:.2}", args.temperature);
    println!("  Chroma boost    : {:.2}", args.chroma_boost);

    // 1. Detect architecture from metadata.json or CLI flags
    let meta_path = args.checkpoint.join("metadata.json");
    let (model_type, d_model, n_heads, num_layers, channels, num_res_blocks) = if meta_path.exists() {
        if let Ok(content) = fs::read_to_string(&meta_path) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
                let m_type = v.get("model_type").and_then(|x| x.as_str()).unwrap_or("transformer").to_string();
                let dm = v.get("d_model").and_then(|x| x.as_u64()).unwrap_or(256) as usize;
                let nh = v.get("n_heads").and_then(|x| x.as_u64()).unwrap_or(8) as usize;
                let nl = v.get("num_layers").and_then(|x| x.as_u64()).unwrap_or(6) as usize;
                let ch = v.get("channels").and_then(|x| x.as_u64()).unwrap_or(64) as usize;
                let rb = v.get("num_res_blocks").and_then(|x| x.as_u64()).unwrap_or(5) as usize;
                (m_type, dm, nh, nl, ch, rb)
            } else {
                (args.model_type.clone(), args.d_model, args.n_heads, args.num_layers, args.channels, args.num_res_blocks)
            }
        } else {
            (args.model_type.clone(), args.d_model, args.n_heads, args.num_layers, args.channels, args.num_res_blocks)
        }
    } else {
        (args.model_type.clone(), args.d_model, args.n_heads, args.num_layers, args.channels, args.num_res_blocks)
    };

    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    let model_path = args.checkpoint.join("model");

    let is_cnn = model_type.eq_ignore_ascii_case("cnn") || model_type.eq_ignore_ascii_case("zhang_2016_colorizer");
    if is_cnn {
        println!("  Model Architecture: Legacy CNN (Zhang 2016)");
        let config = ColorizerModelConfig::new()
            .with_scales_from(INPUT_SIZE)
            .with_channels(channels)
            .with_num_res_blocks(num_res_blocks);
        let model: ColorizerModel<EvalBackend> = config.init(&device);
        let model = model.load_file(&model_path, &recorder, &device)
            .with_context(|| format!("Failed to load model from {}", model_path.display()))?;
        println!("  ✓ Loaded model checkpoint from {}", model_path.display());
        evaluate_model(&model, &args, &device)?;
    } else {
        println!(
            "  Model Architecture: Non-Autoregressive ViT (d_model={}, heads={}, layers={})",
            d_model, n_heads, num_layers
        );
        let config = TransformerColorizerConfig::new()
            .with_d_model(d_model)
            .with_n_heads(n_heads)
            .with_num_layers(num_layers);
        let model: TransformerColorizerModel<EvalBackend> = config.init(&device);
        let model = model.load_file(&model_path, &recorder, &device)
            .with_context(|| format!("Failed to load model from {}", model_path.display()))?;
        println!("  ✓ Loaded model checkpoint from {}", model_path.display());
        evaluate_model(&model, &args, &device)?;
    }

    Ok(())
}

fn evaluate_model(
    model: &impl Colorizer<EvalBackend>,
    args: &Args,
    device: &<EvalBackend as Backend>::Device,
) -> Result<()> {
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
            let (l, _, _) = srgb_to_lab(p[0], p[1], p[2]);
            let l_byte = ((l / 100.0) * 255.0).round().clamp(0.0, 255.0) as u8;
            raw_l.push(l_byte);
            flat_in.push((l - 50.0) / 100.0);
        }

        let in_tensor = Tensor::from_floats(
            TensorData::new(flat_in, [1, 1, INPUT_SIZE, INPUT_SIZE]),
            device,
        );

        let t0 = Instant::now();
        let (pred_a, pred_b) = model.predict_ab_annealed(in_tensor, args.temperature, args.chroma_boost);
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
            let flat_in: Vec<f32> = l_slice.iter().map(|&b| {
                let l_star = (b as f32 / 255.0) * 100.0;
                (l_star - 50.0) / 100.0
            }).collect();
            let in_tensor = Tensor::from_floats(
                TensorData::new(flat_in, [1, 1, INPUT_SIZE, INPUT_SIZE]),
                device,
            );
            let (pred_a, pred_b) = model.predict_ab_annealed(in_tensor, args.temperature, args.chroma_boost);
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
