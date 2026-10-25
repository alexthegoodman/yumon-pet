//! Multi-Sample Grid Visualizer for Pass 4 Upscaler.
//!
//! Generates side-by-side comparison grids of 16 (or custom N) samples:
//! [ Input 32x32 (NN) | Model Predicted 256x256 | Ground Truth 256x256 ]
//!
//! Optimizations:
//! - Direct disk seeking (loads only requested records in <1ms without 2GB RAM caching)
//! - Single-pass batched GPU inference
//! - Multiple layout options: 4x4 triplets matrix (default), 3-panel collage, or vertical strip.

use anyhow::{ensure, Context, Result};
use burn::{
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{Tensor, TensorData},
};
use clap::{Parser, ValueEnum};
use rand::{seq::SliceRandom, SeedableRng};
use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
    time::Instant,
};

use yumon_pet::vision::upscaler::{UpscalerModel, UpscalerModelConfig};

pub type EvalBackend = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;

pub const UPSCALER_CACHE_MAGIC: &[u8; 8] = b"YUMON_UP";
pub const INPUT_GRID_SIZE: usize = 32;
pub const INPUT_CHW_LEN: usize = 3 * INPUT_GRID_SIZE * INPUT_GRID_SIZE; // 3,072 bytes
pub const TARGET_SIZE: usize = 256;
pub const TARGET_CHW_LEN: usize = 3 * TARGET_SIZE * TARGET_SIZE; // 196,608 bytes
pub const RECORD_SIZE: usize = INPUT_CHW_LEN + TARGET_CHW_LEN; // 199,680 bytes
pub const HEADER_SIZE: usize = 32;

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridLayout {
    /// 4 rows of 4 triplets (widescreen 4x4 grid of [Before | After | Ground Truth]).
    Triplets,
    /// Three 4x4 sheets: [ 16 Before | 16 After | 16 Ground Truth ].
    Panels,
    /// Single vertical strip of N stacked triplets.
    Vertical,
    /// 8 rows of 2 triplets.
    Wide,
}

#[derive(Parser, Debug)]
#[command(about = "Generate multi-sample visual comparison grids for Pass 4 Upscaler")]
struct Args {
    /// Path to model checkpoint file or checkpoint directory.
    #[arg(short, long, default_value = "checkpoints/yumon-upscaler-gan-2")]
    checkpoint: PathBuf,

    /// Path to training cache binary file.
    #[arg(long, default_value = "training-cache/upscaler_256_samples.bin")]
    cache: PathBuf,

    /// Output path for the generated comparison grid PNG.
    #[arg(short, long, default_value = "eval_grid_16.png")]
    out: PathBuf,

    /// Number of image samples to compare (default 16).
    #[arg(short, long, default_value_t = 16)]
    count: usize,

    /// Layout style for arranging the before/after/ground-truth images.
    #[arg(long, value_enum, default_value_t = GridLayout::Triplets)]
    layout: GridLayout,

    /// Override model feature channels (auto-detected from metadata.json if omitted).
    #[arg(long)]
    channels: Option<usize>,

    /// Index offset in the cache file (used when shuffle is disabled).
    #[arg(long, default_value_t = 0)]
    offset: usize,

    /// Random seed for sample selection (omit for random selection).
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Shuffle sample selection.
    #[arg(long, default_value_t = true)]
    shuffle: bool,
}

#[derive(serde::Deserialize)]
struct CheckpointMetadata {
    #[serde(default)]
    channels: usize,
}

struct SamplePair {
    input_f32: Vec<f32>,
    target_f32: Vec<f32>,
}

fn load_samples_from_cache(
    cache_path: &PathBuf,
    indices: &[usize],
) -> Result<Vec<SamplePair>> {
    let t0 = Instant::now();
    let mut file = File::open(cache_path)
        .with_context(|| format!("Failed to open cache file: {}", cache_path.display()))?;

    let mut header = [0u8; HEADER_SIZE];
    file.read_exact(&mut header)?;
    ensure!(&header[0..8] == UPSCALER_CACHE_MAGIC, "Invalid cache magic header");

    let total_samples = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
    for &idx in indices {
        ensure!(
            idx < total_samples,
            "Index {} exceeds cache total samples {}",
            idx,
            total_samples
        );
    }

    // Lookup table: byte [0..255] -> f32 [-1.0, 1.0]
    let mut lut = [0.0f32; 256];
    for i in 0..256 {
        lut[i] = (i as f32 / 255.0) * 2.0 - 1.0;
    }

    let mut record_buf = vec![0u8; RECORD_SIZE];
    let mut samples = Vec::with_capacity(indices.len());

    for &idx in indices {
        let pos = (HEADER_SIZE + idx * RECORD_SIZE) as u64;
        file.seek(SeekFrom::Start(pos))?;
        file.read_exact(&mut record_buf)?;

        let in_bytes = &record_buf[..INPUT_CHW_LEN];
        let target_bytes = &record_buf[INPUT_CHW_LEN..RECORD_SIZE];

        let mut in_f32 = vec![0.0f32; INPUT_CHW_LEN];
        let mut target_f32 = vec![0.0f32; TARGET_CHW_LEN];

        for i in 0..INPUT_CHW_LEN {
            in_f32[i] = lut[in_bytes[i] as usize];
        }
        for i in 0..TARGET_CHW_LEN {
            target_f32[i] = lut[target_bytes[i] as usize];
        }

        samples.push(SamplePair {
            input_f32: in_f32,
            target_f32,
        });
    }

    println!("  ✓ Loaded {} sample records from cache in {:.2?}", samples.len(), t0.elapsed());
    Ok(samples)
}

fn resolve_model_path(checkpoint_arg: &PathBuf) -> (PathBuf, Option<PathBuf>) {
    if checkpoint_arg.is_file() {
        let meta = checkpoint_arg.parent().map(|p| p.join("metadata.json"));
        (checkpoint_arg.clone(), meta)
    } else {
        let meta = checkpoint_arg.join("metadata.json");
        let direct = checkpoint_arg.join("model");
        if direct.exists() || checkpoint_arg.join("model.bin").exists() {
            (direct, Some(meta))
        } else {
            (checkpoint_arg.clone(), Some(meta))
        }
    }
}

fn render_triplet_cell(
    img: &mut image::RgbImage,
    x_offset: u32,
    y_offset: u32,
    in_slice: &[f32],
    pred_slice: &[f32],
    target_slice: &[f32],
) {
    let plane_32 = INPUT_GRID_SIZE * INPUT_GRID_SIZE;
    let plane_256 = TARGET_SIZE * TARGET_SIZE;

    // 1. Input 32x32 (Nearest-neighbor upscaled to 256x256)
    for y in 0..TARGET_SIZE {
        let src_y = y / 8;
        for x in 0..TARGET_SIZE {
            let src_x = x / 8;
            let idx_32 = src_y * INPUT_GRID_SIZE + src_x;
            let r = (((in_slice[0 * plane_32 + idx_32] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
            let g = (((in_slice[1 * plane_32 + idx_32] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
            let b = (((in_slice[2 * plane_32 + idx_32] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
            img.put_pixel(x_offset + x as u32, y_offset + y as u32, image::Rgb([r, g, b]));
        }
    }

    // Border line 1
    for y in 0..TARGET_SIZE {
        img.put_pixel(x_offset + TARGET_SIZE as u32, y_offset + y as u32, image::Rgb([30, 30, 40]));
    }

    // 2. Model Predicted 256x256
    let x_pred_base = x_offset + TARGET_SIZE as u32 + 1;
    for y in 0..TARGET_SIZE {
        for x in 0..TARGET_SIZE {
            let idx = y * TARGET_SIZE + x;
            let r = (((pred_slice[0 * plane_256 + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
            let g = (((pred_slice[1 * plane_256 + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
            let b = (((pred_slice[2 * plane_256 + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
            img.put_pixel(x_pred_base + x as u32, y_offset + y as u32, image::Rgb([r, g, b]));
        }
    }

    // Border line 2
    for y in 0..TARGET_SIZE {
        img.put_pixel(x_pred_base + TARGET_SIZE as u32, y_offset + y as u32, image::Rgb([30, 30, 40]));
    }

    // 3. Ground Truth 256x256
    let x_target_base = x_pred_base + TARGET_SIZE as u32 + 1;
    for y in 0..TARGET_SIZE {
        for x in 0..TARGET_SIZE {
            let idx = y * TARGET_SIZE + x;
            let r = (((target_slice[0 * plane_256 + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
            let g = (((target_slice[1 * plane_256 + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
            let b = (((target_slice[2 * plane_256 + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
            img.put_pixel(x_target_base + x as u32, y_offset + y as u32, image::Rgb([r, g, b]));
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t_start = Instant::now();

    println!("==================================================");
    println!("     Pass 4 Upscaler Multi-Sample Grid Visualizer");
    println!("==================================================");
    println!("  Checkpoint : {}", args.checkpoint.display());
    println!("  Cache file : {}", args.cache.display());
    println!("  Output PNG : {}", args.out.display());
    println!("  Samples    : {}", args.count);
    println!("  Layout     : {:?}", args.layout);

    // 1. Resolve model path and channels
    let (model_path, meta_path) = resolve_model_path(&args.checkpoint);
    let channels = if let Some(c) = args.channels {
        c
    } else if let Some(ref m) = meta_path {
        if m.exists() {
            let content = fs::read_to_string(m)?;
            if let Ok(meta) = serde_json::from_str::<CheckpointMetadata>(&content) {
                if meta.channels > 0 {
                    println!("  Auto-detected channels: {}", meta.channels);
                    meta.channels
                } else {
                    32
                }
            } else {
                32
            }
        } else {
            32
        }
    } else {
        32
    };

    let device: <EvalBackend as Backend>::Device = Default::default();

    // 2. Load model
    print!("  Initializing model on GPU (channels={})... ", channels);
    let config = UpscalerModelConfig::new().with_channels(channels);
    let model: UpscalerModel<EvalBackend> = config.init(&device);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    let model = model
        .load_file(&model_path, &recorder, &device)
        .with_context(|| format!("Failed to load upscaler model from: {}", model_path.display()))?;
    println!("✓ Loaded.");

    // 3. Determine sample indices
    let mut indices: Vec<usize> = if args.shuffle {
        let mut file = File::open(&args.cache)?;
        let mut header = [0u8; HEADER_SIZE];
        file.read_exact(&mut header)?;
        let total_samples = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
        let mut all: Vec<usize> = (0..total_samples).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
        all.shuffle(&mut rng);
        all.into_iter().take(args.count).collect()
    } else {
        (args.offset..args.offset + args.count).collect()
    };
    indices.truncate(args.count);

    // 4. Fast direct disk read of selected sample records
    println!("  Fetching {} samples directly from binary cache...", indices.len());
    let samples = load_samples_from_cache(&args.cache, &indices)?;

    // 5. Batched GPU inference (in chunks of 4 to stay well within WGPU buffer allocations)
    let b = samples.len();
    println!("  Running GPU inference on {} samples in chunks of 4...", b);
    let mut pred_data = Vec::with_capacity(b * TARGET_CHW_LEN);

    let num_chunks = (b + 3) / 4;
    for (chunk_idx, chunk) in samples.chunks(4).enumerate() {
        let chunk_len = chunk.len();
        let mut flat_in = Vec::with_capacity(chunk_len * INPUT_CHW_LEN);
        for s in chunk {
            flat_in.extend_from_slice(&s.input_f32);
        }
        let in_tensor = Tensor::from_floats(
            TensorData::new(flat_in, [chunk_len, 3, INPUT_GRID_SIZE, INPUT_GRID_SIZE]),
            &device,
        );
        let t_chunk = Instant::now();
        let pred_tensor = model.forward(in_tensor);
        let chunk_out: Vec<f32> = pred_tensor.into_data().convert::<f32>().to_vec().unwrap();
        pred_data.extend(chunk_out);
        println!(
            "    Chunk {}/{} ({} samples) completed in {:.2?}",
            chunk_idx + 1,
            num_chunks,
            chunk_len,
            t_chunk.elapsed()
        );
    }
    println!("  ✓ All inferences completed.");

    // 6. Assemble output comparison image
    println!("  Assembling grid image ({:?} layout)...", args.layout);

    let triplet_w = (TARGET_SIZE * 3 + 2) as u32; // 770 px
    let triplet_h = TARGET_SIZE as u32; // 256 px
    let cell_margin = 8u32;
    let bg_color = image::Rgb([18, 18, 26]);

    let (img_w, img_h, render_positions): (u32, u32, Vec<(u32, u32)>) = match args.layout {
        GridLayout::Triplets => {
            // 4 rows x 4 columns of triplets
            let cols = 4u32;
            let rows = 4u32;
            let w = cell_margin + cols * (triplet_w + cell_margin);
            let h = cell_margin + rows * (triplet_h + cell_margin);

            let mut pos = Vec::with_capacity(b);
            for i in 0..b {
                let r = (i as u32) / cols;
                let c = (i as u32) % cols;
                let x = cell_margin + c * (triplet_w + cell_margin);
                let y = cell_margin + r * (triplet_h + cell_margin);
                pos.push((x, y));
            }
            (w, h, pos)
        }
        GridLayout::Wide => {
            // 8 rows x 2 columns of triplets
            let cols = 2u32;
            let rows = 8u32;
            let w = cell_margin + cols * (triplet_w + cell_margin);
            let h = cell_margin + rows * (triplet_h + cell_margin);

            let mut pos = Vec::with_capacity(b);
            for i in 0..b {
                let r = (i as u32) / cols;
                let c = (i as u32) % cols;
                let x = cell_margin + c * (triplet_w + cell_margin);
                let y = cell_margin + r * (triplet_h + cell_margin);
                pos.push((x, y));
            }
            (w, h, pos)
        }
        GridLayout::Vertical => {
            // Single vertical strip of N stacked triplets
            let w = cell_margin * 2 + triplet_w;
            let h = cell_margin + (b as u32) * (triplet_h + cell_margin);

            let mut pos = Vec::with_capacity(b);
            for i in 0..b {
                let y = cell_margin + (i as u32) * (triplet_h + cell_margin);
                pos.push((cell_margin, y));
            }
            (w, h, pos)
        }
        GridLayout::Panels => {
            // 3 side-by-side 4x4 sheets: [ 16 Before | 16 After | 16 Ground Truth ]
            let panel_dim = (TARGET_SIZE * 4) as u32; // 1024 px
            let panel_margin = 16u32;
            let w = panel_margin + 3 * (panel_dim + panel_margin);
            let h = panel_margin * 2 + panel_dim;

            let mut grid_img = image::RgbImage::from_pixel(w, h, bg_color);
            let plane_32 = INPUT_GRID_SIZE * INPUT_GRID_SIZE;
            let plane_256 = TARGET_SIZE * TARGET_SIZE;

            for i in 0..b.min(16) {
                let r = (i as u32) / 4;
                let c = (i as u32) % 4;

                let in_slice = &samples[i].input_f32;
                let pred_slice = &pred_data[i * TARGET_CHW_LEN..(i + 1) * TARGET_CHW_LEN];
                let target_slice = &samples[i].target_f32;

                let sub_x = c * TARGET_SIZE as u32;
                let sub_y = r * TARGET_SIZE as u32;

                // Panel 1: Before (Left)
                let p1_x = panel_margin + sub_x;
                let p1_y = panel_margin + sub_y;
                for y in 0..TARGET_SIZE {
                    let src_y = y / 8;
                    for x in 0..TARGET_SIZE {
                        let src_x = x / 8;
                        let idx_32 = src_y * INPUT_GRID_SIZE + src_x;
                        let red = (((in_slice[0 * plane_32 + idx_32] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                        let grn = (((in_slice[1 * plane_32 + idx_32] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                        let blu = (((in_slice[2 * plane_32 + idx_32] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                        grid_img.put_pixel(p1_x + x as u32, p1_y + y as u32, image::Rgb([red, grn, blu]));
                    }
                }

                // Panel 2: Predicted (Middle)
                let p2_x = panel_margin + panel_dim + panel_margin + sub_x;
                let p2_y = panel_margin + sub_y;
                for y in 0..TARGET_SIZE {
                    for x in 0..TARGET_SIZE {
                        let idx = y * TARGET_SIZE + x;
                        let red = (((pred_slice[0 * plane_256 + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                        let grn = (((pred_slice[1 * plane_256 + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                        let blu = (((pred_slice[2 * plane_256 + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                        grid_img.put_pixel(p2_x + x as u32, p2_y + y as u32, image::Rgb([red, grn, blu]));
                    }
                }

                // Panel 3: Ground Truth (Right)
                let p3_x = panel_margin + (panel_dim + panel_margin) * 2 + sub_x;
                let p3_y = panel_margin + sub_y;
                for y in 0..TARGET_SIZE {
                    for x in 0..TARGET_SIZE {
                        let idx = y * TARGET_SIZE + x;
                        let red = (((target_slice[0 * plane_256 + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                        let grn = (((target_slice[1 * plane_256 + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                        let blu = (((target_slice[2 * plane_256 + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                        grid_img.put_pixel(p3_x + x as u32, p3_y + y as u32, image::Rgb([red, grn, blu]));
                    }
                }
            }

            if let Some(parent) = args.out.parent() {
                fs::create_dir_all(parent)?;
            }
            grid_img.save(&args.out)?;
            println!("\n  ✓ Saved 16-sample panels grid to: {}", args.out.display());
            println!("  Dimensions: {} x {} px | Total elapsed: {:.2?}", w, h, t_start.elapsed());
            return Ok(());
        }
    };

    let mut grid_img = image::RgbImage::from_pixel(img_w, img_h, bg_color);

    for (i, &(x, y)) in render_positions.iter().enumerate() {
        if i >= b {
            break;
        }
        let in_slice = &samples[i].input_f32;
        let pred_slice = &pred_data[i * TARGET_CHW_LEN..(i + 1) * TARGET_CHW_LEN];
        let target_slice = &samples[i].target_f32;

        render_triplet_cell(&mut grid_img, x, y, in_slice, pred_slice, target_slice);
    }

    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    grid_img.save(&args.out)?;

    println!("\n  ✓ Successfully generated 16-sample comparison grid!");
    println!("  Output path : {}", args.out.display());
    println!("  Dimensions  : {} x {} px", img_w, img_h);
    println!("  Total time  : {:.2?}", t_start.elapsed());
    Ok(())
}
