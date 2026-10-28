//! Multi-Sample Grid Visualizer for Pass 4/5 Upscaler (Color and Grayscale).
//!
//! Generates side-by-side comparison grids of 16 (or custom N) samples:
//! [ Input (NN) | Model Predicted 256x256 | Ground Truth 256x256 ]
//!
//! Features:
//! - Direct disk seeking (<1ms load time)
//! - Single-pass chunked GPU inference
//! - Multiple layouts: triplets, panels, vertical, wide.

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

use yumon_pet::vision::upscaler::{
    AnyUpscaler, CodebookUpscalerModel, CodebookUpscalerModelConfig,
    UpscalerModel, UpscalerModelConfig,
};

pub type EvalBackend = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;

pub const UPSCALER_CACHE_MAGIC: &[u8; 8] = b"YUMON_UP";
pub const TARGET_SIZE: usize = 256;
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
#[command(about = "Generate multi-sample visual comparison grids for Upscaler")]
struct Args {
    /// Path to model checkpoint file or checkpoint directory.
    #[arg(short, long, default_value = "checkpoints/yumon-upscaler-64")]
    checkpoint: PathBuf,

    /// Path to training cache binary file.
    #[arg(long, default_value = "training-cache/upscaler_64_color.bin")]
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

    /// Override input resolution (auto-detected from metadata.json or cache if omitted).
    #[arg(long)]
    scales_from: Option<usize>,

    /// Index offset in the cache file (used when shuffle is disabled).
    #[arg(long, default_value_t = 0)]
    offset: usize,

    /// Random seed for sample selection.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Shuffle sample selection.
    #[arg(long, default_value_t = true)]
    shuffle: bool,
}

#[derive(serde::Deserialize, Debug)]
struct CheckpointMetadata {
    #[serde(default = "default_channels")]
    channels: usize,
    #[serde(default)]
    in_channels: usize,
    #[serde(default)]
    out_channels: usize,
    #[serde(default)]
    scales_from: Option<usize>,
    #[serde(default)]
    codebook: bool,
}

fn default_channels() -> usize { 32 }

#[derive(Clone, Debug)]
struct CacheHeaderInfo {
    total_samples: usize,
    record_size: usize,
    input_grid_size: usize,
    channels: usize,
    input_len: usize,
    target_len: usize,
}

fn read_cache_header(cache_path: &PathBuf) -> Result<CacheHeaderInfo> {
    let mut file = File::open(cache_path)
        .with_context(|| format!("Failed to open cache file: {}", cache_path.display()))?;

    let mut header = [0u8; HEADER_SIZE];
    file.read_exact(&mut header)?;
    ensure!(&header[0..8] == UPSCALER_CACHE_MAGIC, "Invalid cache magic header");

    let version = u32::from_le_bytes(header[8..12].try_into().unwrap());
    ensure!(version == 4, "Expected cache version 4, got version {}", version);

    let total_samples = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
    let record_size = u32::from_le_bytes(header[16..20].try_into().unwrap()) as usize;
    let input_grid_size = u32::from_le_bytes(header[20..24].try_into().unwrap()) as usize;
    let target_size = u32::from_le_bytes(header[24..28].try_into().unwrap()) as usize;
    let channels = u32::from_le_bytes(header[28..32].try_into().unwrap()) as usize;
    let channels = if channels == 0 { 1 } else { channels };

    let input_len = channels * input_grid_size * input_grid_size;
    let target_len = channels * target_size * target_size;

    Ok(CacheHeaderInfo {
        total_samples,
        record_size,
        input_grid_size,
        channels,
        input_len,
        target_len,
    })
}

struct SamplePair {
    input_f32: Vec<f32>,
    target_f32: Vec<f32>,
}

fn load_samples_from_cache(
    cache_path: &PathBuf,
    info: &CacheHeaderInfo,
    indices: &[usize],
) -> Result<Vec<SamplePair>> {
    let t0 = Instant::now();
    let mut file = File::open(cache_path)
        .with_context(|| format!("Failed to open cache file: {}", cache_path.display()))?;

    for &idx in indices {
        ensure!(
            idx < info.total_samples,
            "Index {} exceeds cache total samples {}",
            idx,
            info.total_samples
        );
    }

    let mut lut = [0.0f32; 256];
    for i in 0..256 {
        lut[i] = (i as f32 / 255.0) * 2.0 - 1.0;
    }

    let mut record_buf = vec![0u8; info.record_size];
    let mut samples = Vec::with_capacity(indices.len());

    for &idx in indices {
        let pos = (HEADER_SIZE + idx * info.record_size) as u64;
        file.seek(SeekFrom::Start(pos))?;
        file.read_exact(&mut record_buf)?;

        let in_bytes = &record_buf[..info.input_len];
        let target_bytes = &record_buf[info.input_len..info.record_size];

        let mut in_f32 = vec![0.0f32; info.input_len];
        let mut target_f32 = vec![0.0f32; info.target_len];

        for i in 0..info.input_len {
            in_f32[i] = lut[in_bytes[i] as usize];
        }
        for i in 0..info.target_len {
            target_f32[i] = lut[target_bytes[i] as usize];
        }

        samples.push(SamplePair {
            input_f32: in_f32,
            target_f32,
        });
    }

    println!("  ✓ Loaded {} sample records ({} channels) in {:.2?}", samples.len(), info.channels, t0.elapsed());
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
    in_grid: usize,
    channels: usize,
    in_slice: &[f32],
    pred_slice: &[f32],
    target_slice: &[f32],
) {
    let scale = (TARGET_SIZE / in_grid).max(1);

    if channels == 3 {
        let plane_in = in_grid * in_grid;
        let plane_target = TARGET_SIZE * TARGET_SIZE;

        // 1. Input (Nearest-neighbor upscaled to 256x256 RGB)
        for y in 0..TARGET_SIZE {
            let src_y = (y / scale).min(in_grid - 1);
            for x in 0..TARGET_SIZE {
                let src_x = (x / scale).min(in_grid - 1);
                let p_in = src_y * in_grid + src_x;
                let r = (((in_slice[0 * plane_in + p_in] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                let g = (((in_slice[1 * plane_in + p_in] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                let b = (((in_slice[2 * plane_in + p_in] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                img.put_pixel(x_offset + x as u32, y_offset + y as u32, image::Rgb([r, g, b]));
            }
        }

        // Border line 1
        for y in 0..TARGET_SIZE {
            img.put_pixel(x_offset + TARGET_SIZE as u32, y_offset + y as u32, image::Rgb([30, 30, 40]));
        }

        // 2. Model Predicted 256x256 RGB
        let x_pred_base = x_offset + TARGET_SIZE as u32 + 1;
        for y in 0..TARGET_SIZE {
            for x in 0..TARGET_SIZE {
                let idx = y * TARGET_SIZE + x;
                let r = (((pred_slice[0 * plane_target + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                let g = (((pred_slice[1 * plane_target + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                let b = (((pred_slice[2 * plane_target + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                img.put_pixel(x_pred_base + x as u32, y_offset + y as u32, image::Rgb([r, g, b]));
            }
        }

        // Border line 2
        for y in 0..TARGET_SIZE {
            img.put_pixel(x_pred_base + TARGET_SIZE as u32, y_offset + y as u32, image::Rgb([30, 30, 40]));
        }

        // 3. Ground Truth 256x256 RGB
        let x_target_base = x_pred_base + TARGET_SIZE as u32 + 1;
        for y in 0..TARGET_SIZE {
            for x in 0..TARGET_SIZE {
                let idx = y * TARGET_SIZE + x;
                let r = (((target_slice[0 * plane_target + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                let g = (((target_slice[1 * plane_target + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                let b = (((target_slice[2 * plane_target + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                img.put_pixel(x_target_base + x as u32, y_offset + y as u32, image::Rgb([r, g, b]));
            }
        }
    } else {
        // 1. Input (Nearest-neighbor upscaled to 256x256 Grayscale)
        for y in 0..TARGET_SIZE {
            let src_y = (y / scale).min(in_grid - 1);
            for x in 0..TARGET_SIZE {
                let src_x = (x / scale).min(in_grid - 1);
                let p_in = src_y * in_grid + src_x;
                let val = (((in_slice[p_in] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                img.put_pixel(x_offset + x as u32, y_offset + y as u32, image::Rgb([val, val, val]));
            }
        }

        // Border line 1
        for y in 0..TARGET_SIZE {
            img.put_pixel(x_offset + TARGET_SIZE as u32, y_offset + y as u32, image::Rgb([30, 30, 40]));
        }

        // 2. Model Predicted 256x256 Grayscale
        let x_pred_base = x_offset + TARGET_SIZE as u32 + 1;
        for y in 0..TARGET_SIZE {
            for x in 0..TARGET_SIZE {
                let idx = y * TARGET_SIZE + x;
                let val = (((pred_slice[idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                img.put_pixel(x_pred_base + x as u32, y_offset + y as u32, image::Rgb([val, val, val]));
            }
        }

        // Border line 2
        for y in 0..TARGET_SIZE {
            img.put_pixel(x_pred_base + TARGET_SIZE as u32, y_offset + y as u32, image::Rgb([30, 30, 40]));
        }

        // 3. Ground Truth 256x256 Grayscale
        let x_target_base = x_pred_base + TARGET_SIZE as u32 + 1;
        for y in 0..TARGET_SIZE {
            for x in 0..TARGET_SIZE {
                let idx = y * TARGET_SIZE + x;
                let val = (((target_slice[idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                img.put_pixel(x_target_base + x as u32, y_offset + y as u32, image::Rgb([val, val, val]));
            }
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t_start = Instant::now();

    let cache_path = if args.cache.exists() {
        args.cache.clone()
    } else if PathBuf::from("training-cache/upscaler_64_gray.bin").exists() {
        PathBuf::from("training-cache/upscaler_64_gray.bin")
    } else {
        args.cache.clone()
    };

    println!("==================================================");
    println!("     Pass 4/5 Upscaler Multi-Sample Visualizer");
    println!("==================================================");
    println!("  Checkpoint : {}", args.checkpoint.display());
    println!("  Cache file : {}", cache_path.display());
    println!("  Output PNG : {}", args.out.display());
    println!("  Samples    : {}", args.count);
    println!("  Layout     : {:?}", args.layout);

    // 1. Read cache header info
    let cache_info = read_cache_header(&cache_path)?;
    println!(
        "  Cache Info : {} samples, {}x{}x{} input -> {}x{}x{} target",
        cache_info.total_samples,
        cache_info.channels,
        cache_info.input_grid_size,
        cache_info.input_grid_size,
        cache_info.channels,
        TARGET_SIZE,
        TARGET_SIZE
    );

    // 2. Resolve model path and configuration
    let (model_path, meta_path) = resolve_model_path(&args.checkpoint);

    let mut channels = 32;
    let mut in_channels = cache_info.channels;
    let mut out_channels = cache_info.channels;
    let mut scales_from = cache_info.input_grid_size;

    if let Some(ref m) = meta_path {
        if m.exists() {
            if let Ok(content) = fs::read_to_string(m) {
                if let Ok(meta) = serde_json::from_str::<CheckpointMetadata>(&content) {
                    if meta.channels > 0 { channels = meta.channels; }
                    if meta.in_channels > 0 { in_channels = meta.in_channels; }
                    if meta.out_channels > 0 { out_channels = meta.out_channels; }
                    if let Some(s) = meta.scales_from { scales_from = s; }
                    println!(
                        "  Auto-detected metadata: channels={}, in_channels={}, out_channels={}, scales_from={}",
                        channels, in_channels, out_channels, scales_from
                    );
                }
            }
        }
    }

    if let Some(c) = args.channels { channels = c; }
    if let Some(s) = args.scales_from { scales_from = s; }

    let device: <EvalBackend as Backend>::Device = Default::default();

    // 3. Initialize & load model
    let is_codebook = meta_path.as_ref().map_or(false, |m| {
        if let Ok(content) = fs::read_to_string(m) {
            if let Ok(meta) = serde_json::from_str::<CheckpointMetadata>(&content) {
                return meta.codebook;
            }
        }
        false
    });

    print!(
        "  Initializing {} model on GPU (channels={}, in_channels={}, out_channels={}, scales_from={})... ",
        if is_codebook { "FSQ Codebook" } else { "Standard" },
        channels, in_channels, out_channels, scales_from
    );

    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    let model = if is_codebook {
        let config = CodebookUpscalerModelConfig::new()
            .with_channels(channels)
            .with_in_channels(in_channels)
            .with_out_channels(out_channels)
            .with_scales_from(scales_from);
        let m: CodebookUpscalerModel<EvalBackend> = config.init(&device);
        let m = m
            .load_file(&model_path, &recorder, &device)
            .with_context(|| format!("Failed to load codebook upscaler model from: {}", model_path.display()))?;
        AnyUpscaler::Codebook(m)
    } else {
        let config = UpscalerModelConfig::new()
            .with_channels(channels)
            .with_in_channels(in_channels)
            .with_out_channels(out_channels)
            .with_scales_from(scales_from);
        let m: UpscalerModel<EvalBackend> = config.init(&device);
        let m = m
            .load_file(&model_path, &recorder, &device)
            .with_context(|| format!("Failed to load standard upscaler model from: {}", model_path.display()))?;
        AnyUpscaler::Standard(m)
    };
    println!("✓ Loaded.");

    // 4. Determine sample indices
    let mut indices: Vec<usize> = if args.shuffle {
        let mut all: Vec<usize> = (0..cache_info.total_samples).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
        all.shuffle(&mut rng);
        all.into_iter().take(args.count).collect()
    } else {
        (args.offset..args.offset + args.count).collect()
    };
    indices.truncate(args.count);

    // 5. Fast direct disk read of selected sample records
    println!("  Fetching {} samples directly from binary cache...", indices.len());
    let samples = load_samples_from_cache(&cache_path, &cache_info, &indices)?;

    // 6. Batched GPU inference (in chunks of 4)
    let b = samples.len();
    println!("  Running GPU inference on {} samples in chunks of 4...", b);
    let mut pred_data = Vec::with_capacity(b * cache_info.target_len);

    let num_chunks = (b + 3) / 4;
    for (chunk_idx, chunk) in samples.chunks(4).enumerate() {
        let chunk_len = chunk.len();
        let mut flat_in = Vec::with_capacity(chunk_len * cache_info.input_len);

        for s in chunk {
            flat_in.extend_from_slice(&s.input_f32);
        }

        let in_tensor = Tensor::from_floats(
            TensorData::new(flat_in, [chunk_len, cache_info.channels, cache_info.input_grid_size, cache_info.input_grid_size]),
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

    // 7. Assemble output comparison image
    println!("  Assembling grid image ({:?} layout)...", args.layout);

    let triplet_w = (TARGET_SIZE * 3 + 2) as u32; // 770 px
    let triplet_h = TARGET_SIZE as u32; // 256 px
    let cell_margin = 8u32;
    let bg_color = image::Rgb([18, 18, 26]);

    let (img_w, img_h, render_positions): (u32, u32, Vec<(u32, u32)>) = match args.layout {
        GridLayout::Triplets => {
            let cols = 4u32;
            let rows = ((b as u32) + cols - 1) / cols;
            let w = cell_margin + cols * (triplet_w + cell_margin);
            let h = cell_margin + rows * (triplet_h + cell_margin);
            let mut positions = Vec::with_capacity(b);
            for i in 0..b {
                let r = (i as u32) / cols;
                let c = (i as u32) % cols;
                let x = cell_margin + c * (triplet_w + cell_margin);
                let y = cell_margin + r * (triplet_h + cell_margin);
                positions.push((x, y));
            }
            (w, h, positions)
        }
        GridLayout::Wide => {
            let cols = 2u32;
            let rows = ((b as u32) + cols - 1) / cols;
            let w = cell_margin + cols * (triplet_w + cell_margin);
            let h = cell_margin + rows * (triplet_h + cell_margin);
            let mut positions = Vec::with_capacity(b);
            for i in 0..b {
                let r = (i as u32) / cols;
                let c = (i as u32) % cols;
                let x = cell_margin + c * (triplet_w + cell_margin);
                let y = cell_margin + r * (triplet_h + cell_margin);
                positions.push((x, y));
            }
            (w, h, positions)
        }
        GridLayout::Vertical => {
            let w = cell_margin * 2 + triplet_w;
            let h = cell_margin + (b as u32) * (triplet_h + cell_margin);
            let mut positions = Vec::with_capacity(b);
            for i in 0..b {
                let x = cell_margin;
                let y = cell_margin + (i as u32) * (triplet_h + cell_margin);
                positions.push((x, y));
            }
            (w, h, positions)
        }
        GridLayout::Panels => {
            let cols = 4u32;
            let rows = ((b as u32) + cols - 1) / cols;
            let panel_w = cols * (TARGET_SIZE as u32) + (cols + 1) * cell_margin;
            let panel_h = rows * (TARGET_SIZE as u32) + (rows + 1) * cell_margin;
            let total_w = panel_w * 3 + cell_margin * 2;
            let total_h = panel_h;
            let mut grid_img = image::RgbImage::from_pixel(total_w, total_h, bg_color);

            let is_color = cache_info.channels == 3;
            let plane_in = cache_info.input_grid_size * cache_info.input_grid_size;
            let plane_target = TARGET_SIZE * TARGET_SIZE;
            let scale = (TARGET_SIZE / cache_info.input_grid_size).max(1);

            for (i, s) in samples.iter().enumerate() {
                let r = (i as u32) / cols;
                let c = (i as u32) % cols;
                let x_local = cell_margin + c * (TARGET_SIZE as u32 + cell_margin);
                let y_local = cell_margin + r * (TARGET_SIZE as u32 + cell_margin);

                // Panel 1: Input (x_local)
                for y in 0..TARGET_SIZE {
                    let src_y = (y / scale).min(cache_info.input_grid_size - 1);
                    for x in 0..TARGET_SIZE {
                        let src_x = (x / scale).min(cache_info.input_grid_size - 1);
                        let p_in = src_y * cache_info.input_grid_size + src_x;
                        if is_color {
                            let r_val = (((s.input_f32[0 * plane_in + p_in] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                            let g_val = (((s.input_f32[1 * plane_in + p_in] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                            let b_val = (((s.input_f32[2 * plane_in + p_in] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                            grid_img.put_pixel(x_local + x as u32, y_local + y as u32, image::Rgb([r_val, g_val, b_val]));
                        } else {
                            let val = (((s.input_f32[p_in] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                            grid_img.put_pixel(x_local + x as u32, y_local + y as u32, image::Rgb([val, val, val]));
                        }
                    }
                }

                // Panel 2: Predicted (panel_w + cell_margin + x_local)
                let x_pred = panel_w + cell_margin + x_local;
                let pred_slice = &pred_data[i * cache_info.target_len..(i + 1) * cache_info.target_len];
                for y in 0..TARGET_SIZE {
                    for x in 0..TARGET_SIZE {
                        let idx = y * TARGET_SIZE + x;
                        if is_color {
                            let r_val = (((pred_slice[0 * plane_target + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                            let g_val = (((pred_slice[1 * plane_target + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                            let b_val = (((pred_slice[2 * plane_target + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                            grid_img.put_pixel(x_pred + x as u32, y_local + y as u32, image::Rgb([r_val, g_val, b_val]));
                        } else {
                            let val = (((pred_slice[idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                            grid_img.put_pixel(x_pred + x as u32, y_local + y as u32, image::Rgb([val, val, val]));
                        }
                    }
                }

                // Panel 3: Ground Truth (panel_w * 2 + cell_margin * 2 + x_local)
                let x_gt = panel_w * 2 + cell_margin * 2 + x_local;
                for y in 0..TARGET_SIZE {
                    for x in 0..TARGET_SIZE {
                        let idx = y * TARGET_SIZE + x;
                        if is_color {
                            let r_val = (((s.target_f32[0 * plane_target + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                            let g_val = (((s.target_f32[1 * plane_target + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                            let b_val = (((s.target_f32[2 * plane_target + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                            grid_img.put_pixel(x_gt + x as u32, y_local + y as u32, image::Rgb([r_val, g_val, b_val]));
                        } else {
                            let val = (((s.target_f32[idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                            grid_img.put_pixel(x_gt + x as u32, y_local + y as u32, image::Rgb([val, val, val]));
                        }
                    }
                }
            }

            grid_img.save(&args.out)?;
            println!("  ✓ Grid image saved to: {}", args.out.display());
            println!("  Completed in {:.2?}", t_start.elapsed());
            return Ok(());
        }
    };

    let mut grid_img = image::RgbImage::from_pixel(img_w, img_h, bg_color);
    for (i, (x_pos, y_pos)) in render_positions.into_iter().enumerate() {
        let s = &samples[i];
        let pred_slice = &pred_data[i * cache_info.target_len..(i + 1) * cache_info.target_len];
        render_triplet_cell(
            &mut grid_img,
            x_pos,
            y_pos,
            cache_info.input_grid_size,
            cache_info.channels,
            &s.input_f32,
            pred_slice,
            &s.target_f32,
        );
    }

    grid_img.save(&args.out)?;
    println!("  ✓ Grid image saved to: {}", args.out.display());
    println!("  Total runtime: {:.2?}", t_start.elapsed());

    Ok(())
}
