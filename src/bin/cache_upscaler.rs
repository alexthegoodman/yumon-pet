//! Build domain-matched training cache for Pass 4 Upscaler.
//!
//! Input: High-res images in `D:/DiffusionDb/images-256`
//! Output: `training-cache/upscaler_256_samples.bin`
//!
//! Each fixed-size record contains:
//! - 1,024 bytes: 32x32 palette tokens (quantized with 256-color palette to match Pass 3)
//! - 196,608 bytes: 256x256x3 raw RGB target bytes

use anyhow::{Context, Result};
use clap::Parser;
use image::imageops::FilterType;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use yumon_pet::vision::paint_action::rgb_to_palette;

pub const UPSCALER_CACHE_MAGIC: &[u8; 8] = b"YUMON_UP";
pub const UPSCALER_CACHE_VERSION: u32 = 1;
pub const INPUT_GRID_SIZE: usize = 32;
pub const INPUT_TOKENS_LEN: usize = INPUT_GRID_SIZE * INPUT_GRID_SIZE; // 1,024
pub const TARGET_SIZE: usize = 256;
pub const TARGET_RGB_LEN: usize = TARGET_SIZE * TARGET_SIZE * 3; // 196,608
pub const RECORD_SIZE: usize = INPUT_TOKENS_LEN + TARGET_RGB_LEN; // 197,632 bytes
pub const HEADER_SIZE: usize = 32;

#[derive(Parser, Debug)]
#[command(about = "Generate domain-matched training cache for Pass 4 Upscaler")]
struct Args {
    /// Directory containing 256x256 target images.
    #[arg(long, default_value = "D:/DiffusionDb/images-256")]
    images_dir: PathBuf,

    /// Output binary cache file.
    #[arg(long, default_value = "training-cache/upscaler_256_samples.bin")]
    out_file: PathBuf,

    /// Maximum images to process (0 = all).
    #[arg(long, default_value_t = 0)]
    limit: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("==================================================");
    println!("     Pass 4 Upscaler Cache Generator (32x32 -> 256x256)");
    println!("==================================================");
    println!("  Images directory : {}", args.images_dir.display());
    println!("  Output cache file: {}", args.out_file.display());

    if !args.images_dir.exists() {
        anyhow::bail!("Images directory not found: {}", args.images_dir.display());
    }

    if let Some(parent) = args.out_file.parent() {
        fs::create_dir_all(parent)?;
    }

    // Collect all valid 256x256 images
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

    // Prepare output file and allocate space
    let mut out_file = File::create(&args.out_file)
        .with_context(|| format!("Failed to create output file: {}", args.out_file.display()))?;

    // Write placeholder header (32 bytes)
    let mut header = [0u8; HEADER_SIZE];
    header[0..8].copy_from_slice(UPSCALER_CACHE_MAGIC);
    header[8..12].copy_from_slice(&UPSCALER_CACHE_VERSION.to_le_bytes());
    header[12..16].copy_from_slice(&(total as u32).to_le_bytes());
    header[16..20].copy_from_slice(&(RECORD_SIZE as u32).to_le_bytes());
    header[20..24].copy_from_slice(&(INPUT_GRID_SIZE as u32).to_le_bytes());
    header[24..28].copy_from_slice(&(TARGET_SIZE as u32).to_le_bytes());
    out_file.write_all(&header)?;

    // Set file size up-front for sequential/random writes
    let expected_len = HEADER_SIZE as u64 + (total as u64) * (RECORD_SIZE as u64);
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

    // Process in parallel batches and write to binary file
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

                // 1. Target 256x256 RGB bytes
                let target_raw = rgb_img.into_raw();
                if target_raw.len() != TARGET_RGB_LEN {
                    anyhow::bail!("Invalid target image size: {} bytes", target_raw.len());
                }

                // 2. Downsample to 32x32 and quantize to 256-color palette
                let downsampled = img.resize_exact(
                    INPUT_GRID_SIZE as u32,
                    INPUT_GRID_SIZE as u32,
                    FilterType::Triangle,
                ).to_rgb8();

                let mut tokens = vec![0u8; INPUT_TOKENS_LEN];
                for py in 0..INPUT_GRID_SIZE {
                    for px in 0..INPUT_GRID_SIZE {
                        let p = downsampled.get_pixel(px as u32, py as u32);
                        let pal_idx = rgb_to_palette(p[0], p[1], p[2]);
                        tokens[py * INPUT_GRID_SIZE + px] = pal_idx;
                    }
                }

                // 3. Write record at fixed offset: HEADER_SIZE + idx * RECORD_SIZE
                let offset = HEADER_SIZE as u64 + (idx as u64) * (RECORD_SIZE as u64);
                let mut guard = file_mutex.lock().unwrap();
                guard.seek(SeekFrom::Start(offset))?;
                guard.write_all(&tokens)?;
                guard.write_all(&target_raw)?;

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
    println!("  Cache file size       : {:.2} GB", (expected_len as f64) / 1e9);
    println!("  Destination           : {}", args.out_file.display());

    Ok(())
}
