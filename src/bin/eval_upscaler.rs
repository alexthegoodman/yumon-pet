//! Multi-Sample Grid Visualizer for Pass 4 Upscaler.
//!
//! Generates side-by-side comparison grids of 16 (or custom N) samples:
//! [ Input (NN) | Model Predicted 256x256 (RGB) | Ground Truth 256x256 (RGB) ]
//!
//! Supports:
//! - Multi-scale inputs (64x64 or 32x32)
//! - Joint colorization & super-resolution (1-channel grayscale input -> 3-channel RGB output)
//! - Cross-attention prompt conditioning
//! - Direct disk seeking (loads only requested records in <1ms without 2GB RAM caching)
//! - Single-pass batched GPU inference with layout options (triplets, panels, vertical, wide).

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

use yumon_pet::{
    brain::bpe::BpeTokenizer,
    vision::upscaler::{UpscalerModel, UpscalerModelConfig},
};

pub type EvalBackend = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;

pub const UPSCALER_CACHE_MAGIC: &[u8; 8] = b"YUMON_UP";
pub const TARGET_SIZE: usize = 256;
pub const TARGET_CHW_LEN: usize = 3 * TARGET_SIZE * TARGET_SIZE; // 196,608 bytes

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
    #[arg(short, long, default_value = "checkpoints/yumon-upscaler-guided")]
    checkpoint: PathBuf,

    /// Path to training cache binary file.
    #[arg(long, default_value = "training-cache/upscaler_guided_256.bin")]
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

    /// Override colorizes flag (auto-detected from metadata.json or cache if omitted).
    #[arg(long)]
    colorizes: Option<bool>,

    /// Optional custom text prompt to test conditional synthesis on all samples.
    #[arg(long)]
    prompt: Option<String>,

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
    scales_from: Option<usize>,
    #[serde(default)]
    colorizes: Option<bool>,
    #[serde(default = "default_true")]
    use_text_conditioning: bool,
    #[serde(default = "default_vocab")]
    vocab_size: usize,
    #[serde(default = "default_text_dim")]
    text_dim: usize,
}

fn default_channels() -> usize { 32 }
fn default_true() -> bool { true }
fn default_vocab() -> usize { 16384 }
fn default_text_dim() -> usize { 64 }

#[derive(Clone, Debug)]
struct CacheHeaderInfo {
    total_samples: usize,
    record_size: usize,
    input_grid_size: usize,
    input_channels: usize,
    max_text_len: usize,
    colorizes: bool,
    header_size: usize,
}

fn read_cache_header(cache_path: &PathBuf) -> Result<CacheHeaderInfo> {
    let mut file = File::open(cache_path)
        .with_context(|| format!("Failed to open cache file: {}", cache_path.display()))?;

    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)?;
    ensure!(&magic == UPSCALER_CACHE_MAGIC, "Invalid cache magic header");

    let mut ver_bytes = [0u8; 4];
    file.read_exact(&mut ver_bytes)?;
    let version = u32::from_le_bytes(ver_bytes);

    if version == 3 {
        let mut rest = [0u8; 52];
        file.read_exact(&mut rest)?;
        let total_samples = u32::from_le_bytes(rest[0..4].try_into().unwrap()) as usize;
        let record_size = u32::from_le_bytes(rest[4..8].try_into().unwrap()) as usize;
        let input_grid_size = u32::from_le_bytes(rest[8..12].try_into().unwrap()) as usize;
        let _target_size = u32::from_le_bytes(rest[12..16].try_into().unwrap()) as usize;
        let input_channels = u32::from_le_bytes(rest[16..20].try_into().unwrap()) as usize;
        let max_text_len = u32::from_le_bytes(rest[20..24].try_into().unwrap()) as usize;
        let colorizes = u32::from_le_bytes(rest[24..28].try_into().unwrap()) == 1;

        Ok(CacheHeaderInfo {
            total_samples,
            record_size,
            input_grid_size,
            input_channels,
            max_text_len,
            colorizes,
            header_size: 64,
        })
    } else if version == 2 {
        let mut rest = [0u8; 20];
        file.read_exact(&mut rest)?;
        let total_samples = u32::from_le_bytes(rest[0..4].try_into().unwrap()) as usize;
        let record_size = u32::from_le_bytes(rest[4..8].try_into().unwrap()) as usize;
        let input_grid_size = u32::from_le_bytes(rest[8..12].try_into().unwrap()) as usize;

        Ok(CacheHeaderInfo {
            total_samples,
            record_size,
            input_grid_size,
            input_channels: 3,
            max_text_len: 0,
            colorizes: false,
            header_size: 32,
        })
    } else {
        anyhow::bail!("Unsupported cache version: {}", version);
    }
}

struct SamplePair {
    input_f32: Vec<f32>,
    prompt_tokens: Vec<i32>,
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

    let input_chw_len = info.input_channels * info.input_grid_size * info.input_grid_size;
    let text_bytes_len = info.max_text_len * 2;
    let mut record_buf = vec![0u8; info.record_size];
    let mut samples = Vec::with_capacity(indices.len());

    for &idx in indices {
        let pos = (info.header_size + idx * info.record_size) as u64;
        file.seek(SeekFrom::Start(pos))?;
        file.read_exact(&mut record_buf)?;

        let in_bytes = &record_buf[..input_chw_len];
        let text_bytes = &record_buf[input_chw_len..input_chw_len + text_bytes_len];
        let target_bytes = &record_buf[input_chw_len + text_bytes_len..info.record_size];

        let mut in_f32 = vec![0.0f32; input_chw_len];
        let mut target_f32 = vec![0.0f32; TARGET_CHW_LEN];
        let mut prompt_tokens = vec![0i32; info.max_text_len.max(1)];

        for i in 0..input_chw_len {
            in_f32[i] = lut[in_bytes[i] as usize];
        }

        if info.max_text_len > 0 {
            for t in 0..info.max_text_len {
                let tid = u16::from_le_bytes([text_bytes[t * 2], text_bytes[t * 2 + 1]]);
                prompt_tokens[t] = tid as i32;
            }
        }

        for i in 0..TARGET_CHW_LEN {
            target_f32[i] = lut[target_bytes[i] as usize];
        }

        samples.push(SamplePair {
            input_f32: in_f32,
            prompt_tokens,
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
    in_grid: usize,
    in_channels: usize,
    in_slice: &[f32],
    pred_slice: &[f32],
    target_slice: &[f32],
) {
    let scale = (TARGET_SIZE / in_grid).max(1);
    let plane_256 = TARGET_SIZE * TARGET_SIZE;
    let plane_in = in_grid * in_grid;

    // 1. Input (Nearest-neighbor upscaled to 256x256)
    for y in 0..TARGET_SIZE {
        let src_y = (y / scale).min(in_grid - 1);
        for x in 0..TARGET_SIZE {
            let src_x = (x / scale).min(in_grid - 1);
            let p_in = src_y * in_grid + src_x;

            let (r, g, b) = if in_channels == 1 {
                let val = (((in_slice[p_in] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                (val, val, val)
            } else {
                let r = (((in_slice[0 * plane_in + p_in] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                let g = (((in_slice[1 * plane_in + p_in] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                let b = (((in_slice[2 * plane_in + p_in] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                (r, g, b)
            };
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

    // 3. Ground Truth 256x256 RGB
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

    // 1. Read cache header info
    let cache_info = read_cache_header(&args.cache)?;
    println!(
        "  Cache Info : {} samples, {}x{} input ({} ch, colorizes={}), {} text tokens",
        cache_info.total_samples,
        cache_info.input_grid_size,
        cache_info.input_grid_size,
        cache_info.input_channels,
        cache_info.colorizes,
        cache_info.max_text_len
    );

    // 2. Resolve model path and configuration
    let (model_path, meta_path) = resolve_model_path(&args.checkpoint);

    let mut channels = 32;
    let mut scales_from = cache_info.input_grid_size;
    let mut colorizes = cache_info.colorizes;
    let mut use_text_conditioning = cache_info.max_text_len > 0;
    let mut vocab_size = 16384;
    let mut text_dim = 64;

    if let Some(ref m) = meta_path {
        if m.exists() {
            if let Ok(content) = fs::read_to_string(m) {
                if let Ok(meta) = serde_json::from_str::<CheckpointMetadata>(&content) {
                    if meta.channels > 0 { channels = meta.channels; }
                    if let Some(s) = meta.scales_from { scales_from = s; }
                    if let Some(c) = meta.colorizes { colorizes = c; }
                    use_text_conditioning = meta.use_text_conditioning;
                    vocab_size = meta.vocab_size;
                    text_dim = meta.text_dim;
                    println!("  Auto-detected metadata: channels={}, scales_from={}, colorizes={}, text_guidance={}",
                        channels, scales_from, colorizes, use_text_conditioning);
                }
            }
        }
    }

    if let Some(c) = args.channels { channels = c; }
    if let Some(s) = args.scales_from { scales_from = s; }
    if let Some(c) = args.colorizes { colorizes = c; }

    let device: <EvalBackend as Backend>::Device = Default::default();

    // 3. Initialize & load model
    print!("  Initializing model on GPU (channels={}, scales_from={}, colorizes={})... ", channels, scales_from, colorizes);
    let config = UpscalerModelConfig::new()
        .with_channels(channels)
        .with_scales_from(scales_from)
        .with_colorizes(colorizes)
        .with_use_text_conditioning(use_text_conditioning)
        .with_vocab_size(vocab_size)
        .with_text_dim(text_dim);

    let model: UpscalerModel<EvalBackend> = config.init(&device);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    let model = model
        .load_file(&model_path, &recorder, &device)
        .with_context(|| format!("Failed to load upscaler model from: {}", model_path.display()))?;
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
    let mut samples = load_samples_from_cache(&args.cache, &cache_info, &indices)?;

    // 6. Handle prompt conditioning & display prompt text
    let tokenizer = BpeTokenizer::load("yumon_bpe").ok();
    if let Some(ref custom_prompt) = args.prompt {
        println!("\n  Applying custom override prompt to all samples: \"{}\"", custom_prompt);
        if let Some(ref tok) = tokenizer {
            if let Ok(encoded) = tok.encode(custom_prompt) {
                let max_len = cache_info.max_text_len.max(1);
                let mut tokens = vec![0i32; max_len];
                for (i, &t) in encoded.iter().take(max_len).enumerate() {
                    tokens[i] = t as i32;
                }
                for s in &mut samples {
                    s.prompt_tokens = tokens.clone();
                }
            }
        }
    } else if let Some(ref tok) = tokenizer {
        println!("\n  Sample ground truth prompts:");
        for (i, s) in samples.iter().take(8).enumerate() {
            let u32_tokens: Vec<u32> = s.prompt_tokens.iter().filter(|&&t| t > 0).map(|&t| t as u32).collect();
            let text = tok.decode(&u32_tokens).unwrap_or_else(|_| "<decode error>".into());
            println!("    [{:02}] \"{}\"", i + 1, text);
        }
        if samples.len() > 8 {
            println!("    ... and {} more samples", samples.len() - 8);
        }
        println!();
    }

    // 7. Batched GPU inference (in chunks of 4)
    let b = samples.len();
    println!("  Running GPU inference on {} samples in chunks of 4...", b);
    let mut pred_data = Vec::with_capacity(b * TARGET_CHW_LEN);

    let num_chunks = (b + 3) / 4;
    let in_chw_len = cache_info.input_channels * cache_info.input_grid_size * cache_info.input_grid_size;
    let eff_text = cache_info.max_text_len.max(1);

    for (chunk_idx, chunk) in samples.chunks(4).enumerate() {
        let chunk_len = chunk.len();
        let mut flat_in = Vec::with_capacity(chunk_len * in_chw_len);
        let mut flat_tok = Vec::with_capacity(chunk_len * eff_text);

        for s in chunk {
            flat_in.extend_from_slice(&s.input_f32);
            flat_tok.extend_from_slice(&s.prompt_tokens);
        }

        let in_tensor = Tensor::from_floats(
            TensorData::new(flat_in, [chunk_len, cache_info.input_channels, cache_info.input_grid_size, cache_info.input_grid_size]),
            &device,
        );
        let tok_tensor = Tensor::from_data(
            TensorData::new(flat_tok, [chunk_len, eff_text]),
            &device,
        );

        let t_chunk = Instant::now();
        let pred_tensor = model.forward_with_text(in_tensor, Some(tok_tensor));
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

    // 8. Assemble output comparison image
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

            for (i, s) in samples.iter().enumerate() {
                let r = (i as u32) / cols;
                let c = (i as u32) % cols;
                let x_local = cell_margin + c * (TARGET_SIZE as u32 + cell_margin);
                let y_local = cell_margin + r * (TARGET_SIZE as u32 + cell_margin);

                // Panel 1: Input (x_local)
                let scale = (TARGET_SIZE / cache_info.input_grid_size).max(1);
                let plane_in = cache_info.input_grid_size * cache_info.input_grid_size;
                for y in 0..TARGET_SIZE {
                    let src_y = (y / scale).min(cache_info.input_grid_size - 1);
                    for x in 0..TARGET_SIZE {
                        let src_x = (x / scale).min(cache_info.input_grid_size - 1);
                        let p_in = src_y * cache_info.input_grid_size + src_x;
                        let (red, grn, blu) = if cache_info.input_channels == 1 {
                            let val = (((s.input_f32[p_in] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                            (val, val, val)
                        } else {
                            let red = (((s.input_f32[0 * plane_in + p_in] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                            let grn = (((s.input_f32[1 * plane_in + p_in] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                            let blu = (((s.input_f32[2 * plane_in + p_in] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                            (red, grn, blu)
                        };
                        grid_img.put_pixel(x_local + x as u32, y_local + y as u32, image::Rgb([red, grn, blu]));
                    }
                }

                // Panel 2: Predicted (panel_w + cell_margin + x_local)
                let x_pred = panel_w + cell_margin + x_local;
                let pred_slice = &pred_data[i * TARGET_CHW_LEN..(i + 1) * TARGET_CHW_LEN];
                let plane_256 = TARGET_SIZE * TARGET_SIZE;
                for y in 0..TARGET_SIZE {
                    for x in 0..TARGET_SIZE {
                        let idx = y * TARGET_SIZE + x;
                        let red = (((pred_slice[0 * plane_256 + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                        let grn = (((pred_slice[1 * plane_256 + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                        let blu = (((pred_slice[2 * plane_256 + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                        grid_img.put_pixel(x_pred + x as u32, y_local + y as u32, image::Rgb([red, grn, blu]));
                    }
                }

                // Panel 3: Ground Truth (panel_w * 2 + cell_margin * 2 + x_local)
                let x_gt = panel_w * 2 + cell_margin * 2 + x_local;
                for y in 0..TARGET_SIZE {
                    for x in 0..TARGET_SIZE {
                        let idx = y * TARGET_SIZE + x;
                        let red = (((s.target_f32[0 * plane_256 + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                        let grn = (((s.target_f32[1 * plane_256 + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                        let blu = (((s.target_f32[2 * plane_256 + idx] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)) as u8;
                        grid_img.put_pixel(x_gt + x as u32, y_local + y as u32, image::Rgb([red, grn, blu]));
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
        let pred_slice = &pred_data[i * TARGET_CHW_LEN..(i + 1) * TARGET_CHW_LEN];
        render_triplet_cell(
            &mut grid_img,
            x_pos,
            y_pos,
            cache_info.input_grid_size,
            cache_info.input_channels,
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
