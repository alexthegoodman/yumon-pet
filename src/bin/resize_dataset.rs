//! Resize dataset images to target dimensions (e.g. 256x256) in parallel with Rayon.

use anyhow::{Context, Result};
use clap::Parser;
use image::imageops::FilterType;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "Resize dataset images in parallel")]
struct Args {
    /// Input directory containing full-sized images.
    #[arg(long, default_value = "D:/DiffusionDb/images")]
    input_dir: PathBuf,

    /// Output directory for resized images.
    #[arg(long, default_value = "D:/DiffusionDb/images-256")]
    output_dir: PathBuf,

    /// Target width and height in pixels.
    #[arg(long, default_value_t = 256)]
    size: u32,

    /// Maximum images to process (0 = all).
    #[arg(long, default_value_t = 0)]
    limit: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("==================================================");
    println!("     Parallel Image Resizer (Lanczos3 -> {}x{})", args.size, args.size);
    println!("==================================================");
    println!("  Input directory : {}", args.input_dir.display());
    println!("  Output directory: {}", args.output_dir.display());

    if !args.input_dir.exists() {
        anyhow::bail!("Input directory does not exist: {}", args.input_dir.display());
    }

    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("Failed to create output directory: {}", args.output_dir.display()))?;

    // Collect all valid image files
    let mut image_files: Vec<PathBuf> = fs::read_dir(&args.input_dir)?
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
    println!("  Found {} candidate images to process.\n", total);
    if total == 0 {
        println!("No images to resize. Exiting.");
        return Ok(());
    }

    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    let processed = AtomicUsize::new(0);
    let skipped = AtomicUsize::new(0);
    let errors = AtomicUsize::new(0);
    let t_start = Instant::now();

    image_files.par_iter().for_each(|src_path| {
        let file_name = match src_path.file_name() {
            Some(name) => name,
            None => return,
        };
        let out_path = args.output_dir.join(file_name);

        // Skip if already exists and is non-empty
        if out_path.exists() {
            if let Ok(meta) = fs::metadata(&out_path) {
                if meta.len() > 0 {
                    skipped.fetch_add(1, Ordering::Relaxed);
                    pb.inc(1);
                    return;
                }
            }
        }

        match image::open(src_path) {
            Ok(img) => {
                // High-quality Lanczos3 filter
                let resized = img.resize_exact(args.size, args.size, FilterType::Lanczos3);
                if let Err(e) = resized.save(&out_path) {
                    eprintln!("Failed to save {:?}: {}", out_path, e);
                    errors.fetch_add(1, Ordering::Relaxed);
                } else {
                    processed.fetch_add(1, Ordering::Relaxed);
                }
            }
            Err(e) => {
                eprintln!("Failed to open {:?}: {}", src_path, e);
                errors.fetch_add(1, Ordering::Relaxed);
            }
        }

        pb.inc(1);
    });

    pb.finish_with_message("Done!");

    println!("\nResize completed in {:.2?}:", t_start.elapsed());
    println!("  Processed : {}", processed.load(Ordering::SeqCst));
    println!("  Skipped   : {}", skipped.load(Ordering::SeqCst));
    println!("  Errors    : {}", errors.load(Ordering::SeqCst));
    println!("  Destination: {}", args.output_dir.display());

    Ok(())
}
