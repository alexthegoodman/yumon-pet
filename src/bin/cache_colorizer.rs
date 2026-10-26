//! Generate training cache for Pass 6 Colorizer (Full Zhang et al. 2016).
//!
//! Input: High-res RGB images in `D:/DiffusionDb/images-256`.
//! Output: `training-cache/colorizer_256.bin` & `training-cache/colorizer_weights.json`.
//!
//! Format:
//! - 65,536 bytes: 1x256x256 input grayscale (L) bytes
//! - 8,192 bytes : 64x64 target color bin indices (u16 little-endian, range 0..312)
//! Total record size: 73,728 bytes (~302 MB for 4,096 records).
//!
//! Header includes 313 precomputed class-rebalancing weights w(q).

use anyhow::{Context, Result};
use clap::Parser;
use image::imageops::FilterType;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use yumon_pet::vision::colorizer::{
    ColorizerLut, NUM_COLOR_BINS, srgb_to_lab, official_class_weights,
};

pub const COLORIZER_CACHE_MAGIC: &[u8; 8] = b"YUMON_CL";
pub const COLORIZER_CACHE_VERSION: u32 = 1;
pub const HEADER_SIZE: usize = 32 + (NUM_COLOR_BINS * 4); // 32 bytes + 313 * 4 = 1,284 bytes
pub const INPUT_SIZE: usize = 256;
pub const INPUT_LEN: usize = INPUT_SIZE * INPUT_SIZE;     // 65,536 bytes
pub const TARGET_GRID: usize = 64;
pub const TARGET_LEN: usize = TARGET_GRID * TARGET_GRID;  // 4,096 indices
pub const RECORD_SIZE: usize = INPUT_LEN + (TARGET_LEN * 2); // 73,728 bytes

#[derive(Parser, Debug)]
#[command(about = "Generate training cache for Pass 6 Colorizer (Zhang et al. 2016)")]
struct Args {
    /// Directory containing 256x256 target images.
    #[arg(long, default_value = "D:/DiffusionDb/images-256")]
    images_dir: PathBuf,

    /// Output binary cache file.
    #[arg(long, default_value = "training-cache/colorizer_256.bin")]
    out_file: PathBuf,

    /// Maximum images to process (0 = all).
    #[arg(long, default_value_t = 4096)]
    limit: usize,

    /// Minimum average chroma (sqrt(a^2 + b^2)) across pixels required to accept an image.
    #[arg(long, default_value_t = 8.0)]
    min_chroma: f32,

    /// Maximum fraction of pixels in neutral bin 120 (a=0, b=0) before image is rejected as monochromatic.
    #[arg(long, default_value_t = 0.35)]
    max_neutral_fraction: f32,

    /// Class rebalancing smoothing parameter lambda (default 0.5 from Zhang et al.).
    #[arg(long, default_value_t = 0.5)]
    lambda: f64,
}

struct SampleRecord {
    path: PathBuf,
    input_l: Vec<u8>,
    target_bins: Vec<u16>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("==================================================");
    println!("     Pass 6 Colorizer Cache Generator (Zhang 2016)");
    println!("==================================================");
    println!("  Images directory : {}", args.images_dir.display());
    println!("  Output cache file: {}", args.out_file.display());
    println!("  Input resolution : {}x{} Grayscale (L)", INPUT_SIZE, INPUT_SIZE);
    println!("  Target resolution: {}x{} (313-class bins)", TARGET_GRID, TARGET_GRID);
    println!("  Record size      : {} bytes", RECORD_SIZE);
    println!("  Min mean chroma  : {:.1}", args.min_chroma);
    println!("  Max neutral frac : {:.0}%", args.max_neutral_fraction * 100.0);

    if !args.images_dir.exists() {
        anyhow::bail!("Images directory not found: {}", args.images_dir.display());
    }

    if let Some(parent) = args.out_file.parent() {
        fs::create_dir_all(parent)?;
    }

    // 1. Scan images
    println!("\n[1/3] Scanning source images...");
    let mut image_files: Vec<PathBuf> = fs::read_dir(&args.images_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|ext| ext.to_str())
                .map_or(false, |ext| {
                    ext.eq_ignore_ascii_case("png")
                        || ext.eq_ignore_ascii_case("jpg")
                        || ext.eq_ignore_ascii_case("jpeg")
                })
        })
        .collect();

    image_files.sort();
    println!("  Found {} source images in directory", image_files.len());

    // 2. Build fast O(1) LUT
    println!("\n[2/3] Filtering monochromatic images & quantizing in parallel...");
    let lut = ColorizerLut::new();
    let bin_counters: Vec<AtomicU64> = (0..NUM_COLOR_BINS).map(|_| AtomicU64::new(0)).collect();
    let rejected_counter = AtomicU64::new(0);

    let pb = ProgressBar::new(image_files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:30.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    let t_start = Instant::now();
    let mut records: Vec<SampleRecord> = image_files
        .par_iter()
        .filter_map(|path| {
            let img = image::open(path).ok()?.to_rgb8();
            let img_256 = if img.width() == (INPUT_SIZE as u32) && img.height() == (INPUT_SIZE as u32) {
                img
            } else {
                image::imageops::resize(&img, INPUT_SIZE as u32, INPUT_SIZE as u32, FilterType::Triangle)
            };

            // Input L: 256x256 true CIE Lab L* channel in [0..100] stored as byte [0..255]
            let mut input_l = Vec::with_capacity(INPUT_LEN);
            for p in img_256.pixels() {
                let (l, _, _) = srgb_to_lab(p[0], p[1], p[2]);
                let l_byte = ((l / 100.0) * 255.0).round().clamp(0.0, 255.0) as u8;
                input_l.push(l_byte);
            }

            // Target bins: downsampled 64x64
            let img_64 = image::imageops::resize(&img_256, TARGET_GRID as u32, TARGET_GRID as u32, FilterType::Triangle);
            let mut target_bins = Vec::with_capacity(TARGET_LEN);
            let mut total_chroma = 0.0f32;
            let mut neutral_count = 0usize;

            for p in img_64.pixels() {
                let (_, a, b) = srgb_to_lab(p[0], p[1], p[2]);
                let chroma = (a * a + b * b).sqrt();
                total_chroma += chroma;
                let q = lut.lookup(a, b);
                if q == 120 {
                    neutral_count += 1;
                }
                target_bins.push(q as u16);
            }

            let mean_chroma = total_chroma / (TARGET_LEN as f32);
            let neutral_fraction = (neutral_count as f32) / (TARGET_LEN as f32);

            // Filter out monochromatic or desaturated images
            if mean_chroma < args.min_chroma || neutral_fraction > args.max_neutral_fraction {
                rejected_counter.fetch_add(1, Ordering::Relaxed);
                pb.inc(1);
                return None;
            }

            for &q in &target_bins {
                bin_counters[q as usize].fetch_add(1, Ordering::Relaxed);
            }

            pb.inc(1);
            Some(SampleRecord { path: path.clone(), input_l, target_bins })
        })
        .collect();

    pb.finish_with_message("Images processed");
    let total_scanned = image_files.len();
    let total_rejected = rejected_counter.load(Ordering::Relaxed);
    let total_accepted = records.len();
    println!(
        "  Scanned {} images: {} vibrant accepted, {} rejected as monochromatic/low-chroma ({:.1}% rejection)",
        total_scanned, total_accepted, total_rejected, (total_rejected as f64 / total_scanned as f64) * 100.0
    );

    if args.limit > 0 && records.len() > args.limit {
        records.truncate(args.limit);
        println!("  Truncated dataset to limit of {} samples", args.limit);
    }
    let total_samples = records.len();
    println!("  Final dataset size: {} samples in {:.2?}", total_samples, t_start.elapsed());

    // 3. Official Zhang et al. (2016) class rebalancing weights w(q) from 1.3M ImageNet prior
    println!("\n[3/3] Applying official Zhang et al. class rebalancing weights w(q)...");
    let final_weights = official_class_weights(args.lambda as f32);

    let min_w = final_weights.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_w = final_weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("  Class weights range: [{:.3}, {:.3}] (Mean = 1.0)", min_w, max_w);

    // Save weights JSON for inspection
    let weights_json_path = args.out_file.with_file_name("colorizer_weights.json");
    let json_content = serde_json::to_string_pretty(&final_weights.to_vec())?;
    fs::write(&weights_json_path, json_content)?;
    println!("  Saved class weights to {}", weights_json_path.display());

    // Save manifest of filenames
    let manifest_path = args.out_file.with_file_name("colorizer_files.txt");
    let manifest_content: String = records
        .iter()
        .map(|r| r.path.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(&manifest_path, manifest_content)?;
    println!("  Saved sample manifest to {}", manifest_path.display());

    // 4. Write binary cache
    println!("  Writing binary cache to {}...", args.out_file.display());
    let file = File::create(&args.out_file)
        .with_context(|| format!("Failed to create cache file: {}", args.out_file.display()))?;
    let mut writer = BufWriter::with_capacity(1024 * 1024 * 8, file);

    // Header:
    // [0..8]   : Magic "YUMON_CL"
    // [8..12]  : Version
    // [12..16] : Total samples
    // [16..20] : Record size
    // [20..24] : Input size (256)
    // [24..28] : Target grid (64)
    // [28..32] : Reserved
    // [32..1284] : 313 f32 weights
    let mut header = vec![0u8; HEADER_SIZE];
    header[0..8].copy_from_slice(COLORIZER_CACHE_MAGIC);
    header[8..12].copy_from_slice(&COLORIZER_CACHE_VERSION.to_le_bytes());
    header[12..16].copy_from_slice(&(total_samples as u32).to_le_bytes());
    header[16..20].copy_from_slice(&(RECORD_SIZE as u32).to_le_bytes());
    header[20..24].copy_from_slice(&(INPUT_SIZE as u32).to_le_bytes());
    header[24..28].copy_from_slice(&(TARGET_GRID as u32).to_le_bytes());

    for (i, &w) in final_weights.iter().enumerate() {
        let offset = 32 + (i * 4);
        header[offset..offset + 4].copy_from_slice(&w.to_le_bytes());
    }

    writer.write_all(&header)?;

    // Write all records sequentially
    for rec in &records {
        writer.write_all(&rec.input_l)?;
        for &bin_id in &rec.target_bins {
            writer.write_all(&bin_id.to_le_bytes())?;
        }
    }

    writer.flush()?;
    let mut file = writer.into_inner()?;
    let total_bytes = file.seek(SeekFrom::Current(0))?;
    println!(
        "\n✓ Cache successfully generated: {:.2} MB for {} records in {:.2?}",
        total_bytes as f64 / 1e6,
        total_samples,
        t_start.elapsed()
    );

    Ok(())
}
