//! Build domain-matched training cache for Pass 4/5 Upscaler (Grayscale to Grayscale).
//!
//! Input: High-res images in `D:/DiffusionDb/images-256`.
//! Output: `training-cache/upscaler_64_gray.bin`.
//!
//! Pure grayscale-to-grayscale super-resolution:
//! - 4,096 bytes  : 1x64x64 input grayscale planar bytes
//! - 65,536 bytes : 1x256x256 target grayscale planar bytes
//! Total record size: 69,632 bytes (~285 MB for 4,096 records).

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

pub const UPSCALER_CACHE_MAGIC: &[u8; 8] = b"YUMON_UP";
pub const UPSCALER_CACHE_VERSION: u32 = 4;
pub const HEADER_SIZE: usize = 32;
pub const TARGET_SIZE: usize = 256;
pub const TARGET_LEN: usize = TARGET_SIZE * TARGET_SIZE; // 65,536 bytes

#[derive(Parser, Debug)]
#[command(about = "Generate grayscale-to-grayscale training cache for Upscaler")]
struct Args {
    /// Directory containing 256x256 target images.
    #[arg(long, default_value = "D:/DiffusionDb/images-256")]
    images_dir: PathBuf,

    /// Output binary cache file.
    #[arg(long, default_value = "training-cache/upscaler_64_gray.bin")]
    out_file: PathBuf,

    /// Input resolution to downsample from (default 64).
    #[arg(long, default_value_t = 64)]
    scales_from: usize,

    /// Maximum images to process (0 = all).
    #[arg(long, default_value_t = 4096)]
    limit: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("==================================================");
    println!("     Pass 4/5 Upscaler Grayscale Cache Generator");
    println!("==================================================");
    println!("  Images directory : {}", args.images_dir.display());
    println!("  Output cache file: {}", args.out_file.display());
    println!("  Scales from      : {}x{}", args.scales_from, args.scales_from);
    println!("  Target size      : {}x{}", TARGET_SIZE, TARGET_SIZE);

    if !args.images_dir.exists() {
        anyhow::bail!("Images directory not found: {}", args.images_dir.display());
    }

    if let Some(parent) = args.out_file.parent() {
        fs::create_dir_all(parent)?;
    }

    // 1. Collect target images
    println!("\n[1/2] Scanning target images...");
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

    let in_grid = args.scales_from;
    let input_len = in_grid * in_grid; // 4,096 for 64x64
    let record_size = input_len + TARGET_LEN; // 69,632

    println!("  Record layout (Grayscale):");
    println!("    Input tensor : 1 channel x {}x{} = {} bytes", in_grid, in_grid, input_len);
    println!("    Target tensor: 1 channel x 256x256 = {} bytes", TARGET_LEN);
    println!("    Total/record : {} bytes ({:.2} MB for {} records)", record_size, (total * record_size) as f64 / 1e6, total);

    // Prepare output file and allocate space
    let mut out_file = File::create(&args.out_file)
        .with_context(|| format!("Failed to create output file: {}", args.out_file.display()))?;

    // Write header (32 bytes)
    let mut header = [0u8; HEADER_SIZE];
    header[0..8].copy_from_slice(UPSCALER_CACHE_MAGIC);
    header[8..12].copy_from_slice(&UPSCALER_CACHE_VERSION.to_le_bytes());
    header[12..16].copy_from_slice(&(total as u32).to_le_bytes());
    header[16..20].copy_from_slice(&(record_size as u32).to_le_bytes());
    header[20..24].copy_from_slice(&(in_grid as u32).to_le_bytes());
    header[24..28].copy_from_slice(&(TARGET_SIZE as u32).to_le_bytes());
    header[28..32].copy_from_slice(&1u32.to_le_bytes()); // 1 channel
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

    println!("\n[2/2] Generating grayscale cache in parallel across CPU cores...");

    image_files
        .par_iter()
        .enumerate()
        .for_each(|(idx, path)| {
            let res: Result<()> = (|| {
                let img = image::open(path)?;

                // 1. Target 256x256 grayscale (65,536 bytes)
                let target_luma = img.to_luma8();
                let target_raw = target_luma.into_raw();
                if target_raw.len() != TARGET_LEN {
                    anyhow::bail!("Invalid target image size: {} bytes", target_raw.len());
                }

                // 2. Downscaled input 64x64 grayscale (4,096 bytes)
                let downsampled = img.resize_exact(
                    in_grid as u32,
                    in_grid as u32,
                    FilterType::Triangle,
                ).to_luma8();
                let input_raw = downsampled.into_raw();

                // 3. Write record at fixed offset: HEADER_SIZE + idx * record_size
                let offset = HEADER_SIZE as u64 + (idx as u64) * (record_size as u64);
                let mut guard = file_mutex.lock().unwrap();
                guard.seek(SeekFrom::Start(offset))?;
                guard.write_all(&input_raw)?;
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
    println!("  Cache file size       : {:.2} MB", (expected_len as f64) / 1e6);
    println!("  Destination           : {}", args.out_file.display());

    Ok(())
}
