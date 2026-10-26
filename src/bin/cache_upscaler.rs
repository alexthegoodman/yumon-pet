//! Build domain-matched training cache for Pass 4 Upscaler.
//!
//! Input: High-res images in `D:/DiffusionDb/images-256` and prompt metadata in `D:/DiffusionDb/images`.
//! Output: `training-cache/upscaler_guided_256.bin` (or custom path).
//!
//! Supports:
//! - Configurable resolution (`--scales-from`: 64 or 32, default 64).
//! - Joint colorization & super-resolution (`--colorizes`: default true: 1-channel grayscale input -> 3-channel RGB target).
//! - Prompt text conditioning: strips prompt after the first comma, tokenizes with `BpeTokenizer` (16 u16 tokens).
//!
//! Record layout (Version 3):
//! - `input_chw_len` bytes: 1x64x64 grayscale (4,096 bytes) or 3x64x64 / 3x32x32 RGB
//! - 32 bytes             : 16 x u16 BPE prompt token IDs
//! - 196,608 bytes        : 3x256x256 planar RGB target
//! Total record size (64x64 grayscale): 200,736 bytes (~822 MB for 4,096 samples).

use anyhow::{Context, Result};
use clap::Parser;
use image::imageops::FilterType;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use yumon_pet::brain::bpe::BpeTokenizer;

pub const UPSCALER_CACHE_MAGIC: &[u8; 8] = b"YUMON_UP";
pub const UPSCALER_CACHE_VERSION: u32 = 3;
pub const HEADER_SIZE: usize = 64;
pub const TARGET_SIZE: usize = 256;
pub const TARGET_CHW_LEN: usize = 3 * TARGET_SIZE * TARGET_SIZE; // 196,608
pub const MAX_TEXT_LEN: usize = 16;
pub const TEXT_BYTES_LEN: usize = MAX_TEXT_LEN * 2; // 32 bytes

#[derive(Parser, Debug)]
#[command(about = "Generate prompt-guided domain-matched training cache for Pass 4 Upscaler")]
struct Args {
    /// Directory containing 256x256 target images.
    #[arg(long, default_value = "D:/DiffusionDb/images-256")]
    images_dir: PathBuf,

    /// Directory containing metadata JSON parts (part-*.json).
    #[arg(long, default_value = "D:/DiffusionDb/images")]
    prompts_dir: PathBuf,

    /// Output binary cache file.
    #[arg(long, default_value = "training-cache/upscaler_guided_256.bin")]
    out_file: PathBuf,

    /// Input resolution to downsample from (64 or 32).
    #[arg(long, default_value_t = 64)]
    scales_from: usize,

    /// Whether the upscaler learns colorization from grayscale input (1-channel -> 3-channel).
    #[arg(long, default_value_t = true)]
    colorizes: bool,

    /// Path to BPE tokenizer directory.
    #[arg(long, default_value = "yumon_bpe")]
    tokenizer: PathBuf,

    /// Maximum images to process (0 = all).
    #[arg(long, default_value_t = 0)]
    limit: usize,
}

#[derive(serde::Deserialize)]
struct ImageEntry {
    p: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("==================================================");
    println!("     Pass 4 Upscaler Guided Cache Generator");
    println!("==================================================");
    println!("  Images directory : {}", args.images_dir.display());
    println!("  Prompts directory: {}", args.prompts_dir.display());
    println!("  Output cache file: {}", args.out_file.display());
    println!("  Scales from      : {}x{}", args.scales_from, args.scales_from);
    println!("  Colorizes        : {}", args.colorizes);
    println!("  Max text tokens  : {}", MAX_TEXT_LEN);

    if !args.images_dir.exists() {
        anyhow::bail!("Images directory not found: {}", args.images_dir.display());
    }

    if let Some(parent) = args.out_file.parent() {
        fs::create_dir_all(parent)?;
    }

    // 1. Load prompt metadata from all JSON parts
    println!("\n[1/4] Loading prompt metadata from JSON parts...");
    let t_meta = Instant::now();
    let mut prompts_map: HashMap<String, String> = HashMap::new();

    if args.prompts_dir.exists() {
        let mut json_files: Vec<PathBuf> = fs::read_dir(&args.prompts_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
            .collect();
        json_files.sort();

        for json_path in &json_files {
            if let Ok(data) = fs::read(json_path) {
                if let Ok(entries) = serde_json::from_slice::<HashMap<String, ImageEntry>>(&data) {
                    for (fname, entry) in entries {
                        prompts_map.insert(fname, entry.p);
                    }
                }
            }
        }
        println!("  ✓ Loaded {} prompts from {} JSON files ({:.2?})", prompts_map.len(), json_files.len(), t_meta.elapsed());
    } else {
        println!("  ! Warning: prompts directory not found: {}. Continuing without text.", args.prompts_dir.display());
    }

    // 2. Load BPE Tokenizer
    println!("\n[2/4] Initializing BPE tokenizer...");
    let tokenizer = BpeTokenizer::load(args.tokenizer.to_str().unwrap_or("yumon_bpe"))
        .context("Failed to load BPE tokenizer from yumon_bpe")?;
    println!("  ✓ BPE tokenizer loaded (vocab size: {})", tokenizer.vocab_size);

    // 3. Collect target images
    println!("\n[3/4] Scanning target images...");
    let mut image_files: Vec<PathBuf> = fs::read_dir(&args.images_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
                let ext_lower = ext.to_lowercase();
                ext_lower == "png" || ext_lower == "jpg" || ext_lower == "jpeg" || ext_lower == "webp"
            } else {
                false
            }
        })
        .collect();

    image_files.sort();

    if args.limit > 0 && args.limit < image_files.len() {
        image_files.truncate(args.limit);
    }

    let total = image_files.len();
    println!("  Found {} images to process.\n", total);
    if total == 0 {
        println!("No images found. Exiting.");
        return Ok(());
    }

    // Calculate dimensions
    let in_channels = if args.colorizes { 1 } else { 3 };
    let in_grid = args.scales_from;
    let input_chw_len = in_channels * in_grid * in_grid;
    let record_size = input_chw_len + TEXT_BYTES_LEN + TARGET_CHW_LEN;

    println!("  Record layout:");
    println!("    Input tensor : {} channels x {}x{} = {} bytes", in_channels, in_grid, in_grid, input_chw_len);
    println!("    Text tokens  : {} x u16 = {} bytes", MAX_TEXT_LEN, TEXT_BYTES_LEN);
    println!("    Target RGB   : 3 channels x 256x256 = {} bytes", TARGET_CHW_LEN);
    println!("    Total/record : {} bytes ({:.2} MB for {} records)", record_size, (total * record_size) as f64 / 1e6, total);

    // 4. Pre-tokenize all prompts for selected images
    println!("\n[4/4] Pre-tokenizing prompts and generating binary cache...");
    let mut precomputed_tokens: Vec<[u16; MAX_TEXT_LEN]> = Vec::with_capacity(total);
    let mut sample_preview_prompts: Vec<String> = Vec::new();

    for (idx, path) in image_files.iter().enumerate() {
        let filename = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let raw = prompts_map.get(filename).map(|s| s.as_str()).unwrap_or("");

        // Strip prompt after the first comma to focus on core subject
        let clean = if let Some(comma_pos) = raw.find(',') {
            &raw[..comma_pos]
        } else {
            raw
        }.trim();

        if idx < 5 {
            sample_preview_prompts.push(format!("'{}' -> '{}'", filename, clean));
        }

        let mut token_ids = [0u16; MAX_TEXT_LEN];
        if !clean.is_empty() {
            if let Ok(encoded) = tokenizer.encode(clean) {
                for (t_idx, &tid) in encoded.iter().take(MAX_TEXT_LEN).enumerate() {
                    token_ids[t_idx] = tid as u16;
                }
            }
        }
        precomputed_tokens.push(token_ids);
    }

    println!("  Sample stripped prompts:");
    for s in &sample_preview_prompts {
        println!("    • {}", s);
    }
    println!();

    // Prepare output file and allocate space
    let mut out_file = File::create(&args.out_file)
        .with_context(|| format!("Failed to create output file: {}", args.out_file.display()))?;

    // Write header (64 bytes)
    let mut header = [0u8; HEADER_SIZE];
    header[0..8].copy_from_slice(UPSCALER_CACHE_MAGIC);
    header[8..12].copy_from_slice(&UPSCALER_CACHE_VERSION.to_le_bytes());
    header[12..16].copy_from_slice(&(total as u32).to_le_bytes());
    header[16..20].copy_from_slice(&(record_size as u32).to_le_bytes());
    header[20..24].copy_from_slice(&(in_grid as u32).to_le_bytes());
    header[24..28].copy_from_slice(&(TARGET_SIZE as u32).to_le_bytes());
    header[28..32].copy_from_slice(&(in_channels as u32).to_le_bytes());
    header[32..36].copy_from_slice(&(MAX_TEXT_LEN as u32).to_le_bytes());
    header[36..40].copy_from_slice(&(if args.colorizes { 1u32 } else { 0u32 }).to_le_bytes());
    out_file.write_all(&header)?;

    let expected_len = HEADER_SIZE as u64 + (total as u64) * (record_size as u64);
    out_file.set_len(expected_len)?;
    drop(out_file);

    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    let processed = AtomicUsize::new(0);
    let errors = AtomicUsize::new(0);
    let t_start = Instant::now();

    let file_mutex = std::sync::Mutex::new(
        fs::OpenOptions::new()
            .write(true)
            .open(&args.out_file)?,
    );

    image_files
        .par_iter()
        .enumerate()
        .for_each(|(idx, path)| {
            let res: Result<()> = (|| {
                let img = image::open(path)?;
                let rgb_img = img.to_rgb8();

                // 1. Target 256x256 converted to contiguous planar CHW [3, 256, 256]
                let target_raw = rgb_img.into_raw();
                if target_raw.len() != TARGET_CHW_LEN {
                    anyhow::bail!("Invalid target image size: {} bytes", target_raw.len());
                }
                let plane_256 = TARGET_SIZE * TARGET_SIZE;
                let mut target_chw = vec![0u8; TARGET_CHW_LEN];
                for i in 0..plane_256 {
                    let r = target_raw[i * 3];
                    let g = target_raw[i * 3 + 1];
                    let b = target_raw[i * 3 + 2];
                    target_chw[0 * plane_256 + i] = r;
                    target_chw[1 * plane_256 + i] = g;
                    target_chw[2 * plane_256 + i] = b;
                }

                // 2. Downscaled input tensor
                let downsampled = img.resize_exact(
                    in_grid as u32,
                    in_grid as u32,
                    FilterType::Triangle,
                );

                let input_chw = if args.colorizes {
                    // Grayscale 1-channel [1, in_grid, in_grid]
                    let luma = downsampled.to_luma8();
                    luma.into_raw()
                } else {
                    // RGB 3-channel planar [3, in_grid, in_grid]
                    let rgb_down = downsampled.to_rgb8();
                    let raw = rgb_down.into_raw();
                    let plane = in_grid * in_grid;
                    let mut chw = vec![0u8; input_chw_len];
                    for i in 0..plane {
                        chw[0 * plane + i] = raw[i * 3];
                        chw[1 * plane + i] = raw[i * 3 + 1];
                        chw[2 * plane + i] = raw[i * 3 + 2];
                    }
                    chw
                };

                // 3. Prompt token bytes (16 x u16 little-endian = 32 bytes)
                let tokens = &precomputed_tokens[idx];
                let mut prompt_bytes = [0u8; TEXT_BYTES_LEN];
                for (t_idx, &tok) in tokens.iter().enumerate() {
                    let le = tok.to_le_bytes();
                    prompt_bytes[t_idx * 2] = le[0];
                    prompt_bytes[t_idx * 2 + 1] = le[1];
                }

                // 4. Write record at fixed offset: HEADER_SIZE + idx * RECORD_SIZE
                let offset = HEADER_SIZE as u64 + (idx as u64) * (record_size as u64);
                let mut guard = file_mutex.lock().unwrap();
                guard.seek(SeekFrom::Start(offset))?;
                guard.write_all(&input_chw)?;
                guard.write_all(&prompt_bytes)?;
                guard.write_all(&target_chw)?;

                Ok(())
            })();

            match res {
                Ok(_) => {
                    processed.fetch_add(1, Ordering::Relaxed);
                }
                Err(e) => {
                    eprintln!("Error processing {:?}: {}", path, e);
                    errors.fetch_add(1, Ordering::Relaxed);
                }
            }
            pb.inc(1);
        });

    pb.finish_with_message("Done!");

    let valid_count = processed.load(Ordering::SeqCst);
    println!("\nCache generation completed in {:.2?}:", t_start.elapsed());
    println!("  Total records written : {}", valid_count);
    println!("  Errors                : {}", errors.load(Ordering::SeqCst));
    println!("  Cache file size       : {:.2} MB", (expected_len as f64) / 1e6);
    println!("  Destination           : {}", args.out_file.display());

    Ok(())
}
