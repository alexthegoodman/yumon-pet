//! Export and visualize diagnostic samples from the Pass 6 Colorizer binary cache (`colorizer_256.bin`).
//!
//! Features:
//! - Exports cached grayscale L channel (as grayscale PNG).
//! - Reconstructs TrueColor RGB from target color bin IDs using official CIE Lab centroids.
//! - Pairs each sample with the original source image (from `D:/DiffusionDb/images-256`) into a crisp side-by-side comparison:
//!   `[Original Source RGB]` │ `[Cached Input Grayscale L]` │ `[Reconstructed Target Bins RGB]`
//! - Calculates color statistics (unique bins, dominant color, MAE, PSNR).
//! - Generates an interactive HTML preview gallery (`cache_previews/gallery.html`).

use anyhow::{Context, Result};
use clap::Parser;
use image::{imageops::FilterType, Rgb, RgbImage};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

use yumon_pet::vision::colorizer::{
    colorize_fused_rgb, COLOR_BINS_AB, NUM_COLOR_BINS, INPUT_SIZE, TARGET_GRID,
};

#[derive(Parser, Debug)]
#[command(about = "Export and visualize samples from the Colorizer training cache")]
struct Args {
    /// Path to binary cache file.
    #[arg(long, default_value = "training-cache/colorizer_256.bin")]
    cache: PathBuf,

    /// Optional directory containing original source images.
    #[arg(long, default_value = "D:/DiffusionDb/images-256")]
    source_dir: Option<PathBuf>,

    /// Directory where exported images and HTML gallery will be saved.
    #[arg(long, default_value = "training-cache/colorizer_previews")]
    out_dir: PathBuf,

    /// Number of samples to export.
    #[arg(long, default_value_t = 32)]
    samples: usize,

    /// Starting index of samples to export.
    #[arg(long, default_value_t = 0)]
    start: usize,

    /// Also save individual files (orig.png, l.png, target.png) in addition to side-by-side triplets.
    #[arg(long, default_value_t = false)]
    individual: bool,
}

struct SampleCard {
    idx: usize,
    triplet_rel_path: String,
    unique_bins: usize,
    dominant_bin: usize,
    dom_a: f32,
    dom_b: f32,
    dom_pct: f64,
    mae: Option<f64>,
    psnr: Option<f64>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("==================================================");
    println!("     Yumon Colorizer Cache Exporter & Inspector   ");
    println!("==================================================");
    println!("  Cache binary    : {}", args.cache.display());
    println!("  Output directory: {}", args.out_dir.display());
    println!("  Samples to dump : {} (starting from index {})", args.samples, args.start);

    fs::create_dir_all(&args.out_dir)?;

    let mut file = File::open(&args.cache)
        .with_context(|| format!("Failed to open cache file: {}", args.cache.display()))?;

    // Read header (1,284 bytes)
    let mut header = [0u8; 32 + (NUM_COLOR_BINS * 4)];
    file.read_exact(&mut header)?;

    let magic = &header[0..8];
    let version = u32::from_le_bytes(header[8..12].try_into().unwrap());
    let total_samples = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
    let record_size = u32::from_le_bytes(header[16..20].try_into().unwrap()) as usize;

    println!("  Header Magic    : {:?}", std::str::from_utf8(magic).unwrap_or("???"));
    println!("  Cache Version   : {}", version);
    println!("  Total in cache  : {} records", total_samples);
    println!("  Record size     : {} bytes", record_size);

    if args.start >= total_samples {
        anyhow::bail!("Start index {} is beyond total samples ({})", args.start, total_samples);
    }

    // Scan source directory if provided
    let image_files: Vec<PathBuf> = if let Some(ref sdir) = args.source_dir {
        if sdir.exists() {
            println!("  Source images   : {}", sdir.display());
            let mut files: Vec<PathBuf> = fs::read_dir(sdir)?
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
            files.sort();
            files
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    let in_len = INPUT_SIZE * INPUT_SIZE;
    let tgt_len = TARGET_GRID * TARGET_GRID;
    let pts = &COLOR_BINS_AB;

    let num_to_export = args.samples.min(total_samples.saturating_sub(args.start));
    let mut total_mae = 0.0f64;
    let mut total_psnr = 0.0f64;
    let mut paired_count = 0;
    let mut cards: Vec<SampleCard> = Vec::with_capacity(num_to_export);

    println!("\nExporting samples...\n");

    for i in 0..num_to_export {
        let sample_idx = args.start + i;
        let rec_pos = (32 + (NUM_COLOR_BINS * 4) + sample_idx * record_size) as u64;

        file.seek(SeekFrom::Start(rec_pos))?;

        let mut rec_bytes = vec![0u8; record_size];
        file.read_exact(&mut rec_bytes)?;

        let l_slice = &rec_bytes[..in_len];
        let tgt_slice = &rec_bytes[in_len..record_size];

        // 1. Grayscale L image
        let mut gray_img = RgbImage::new(INPUT_SIZE as u32, INPUT_SIZE as u32);
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                let v = l_slice[y * INPUT_SIZE + x];
                gray_img.put_pixel(x as u32, y as u32, Rgb([v, v, v]));
            }
        }

        // 2. Reconstructed Target from 313 bins
        let mut target_a = vec![0.0f32; tgt_len];
        let mut target_b = vec![0.0f32; tgt_len];
        let mut bin_counts = [0usize; NUM_COLOR_BINS];

        for k in 0..tgt_len {
            let bin = u16::from_le_bytes([tgt_slice[k * 2], tgt_slice[k * 2 + 1]]) as usize;
            bin_counts[bin] += 1;
            target_a[k] = pts[bin][0];
            target_b[k] = pts[bin][1];
        }

        let recon_img = colorize_fused_rgb(l_slice, &target_a, &target_b, INPUT_SIZE);

        // Stats
        let mut unique_bins = 0;
        let mut max_c = 0;
        let mut dom_bin = 0;
        for (b, &c) in bin_counts.iter().enumerate() {
            if c > 0 {
                unique_bins += 1;
            }
            if c > max_c {
                max_c = c;
                dom_bin = b;
            }
        }
        let dom_pct = (max_c as f64 / tgt_len as f64) * 100.0;
        let dom_a = pts[dom_bin][0];
        let dom_b = pts[dom_bin][1];

        // 3. Optional Source Image comparison
        let orig_img = if sample_idx < image_files.len() {
            if let Ok(img) = image::open(&image_files[sample_idx]) {
                let rgb = img.to_rgb8();
                if rgb.width() == (INPUT_SIZE as u32) && rgb.height() == (INPUT_SIZE as u32) {
                    Some(rgb)
                } else {
                    Some(image::imageops::resize(
                        &rgb,
                        INPUT_SIZE as u32,
                        INPUT_SIZE as u32,
                        FilterType::Triangle,
                    ))
                }
            } else {
                None
            }
        } else {
            None
        };

        let (mae_opt, psnr_opt, diff_str) = if let Some(ref orig) = orig_img {
            let mut mae = 0.0f64;
            let mut mse = 0.0f64;
            for y in 0..INPUT_SIZE as u32 {
                for x in 0..INPUT_SIZE as u32 {
                    let po = orig.get_pixel(x, y);
                    let pr = recon_img.get_pixel(x, y);

                    let dr = (po[0] as f64) - (pr[0] as f64);
                    let dg = (po[1] as f64) - (pr[1] as f64);
                    let db = (po[2] as f64) - (pr[2] as f64);

                    mae += (dr.abs() + dg.abs() + db.abs()) / 3.0;
                    mse += (dr * dr + dg * dg + db * db) / 3.0;
                }
            }
            let denom = (INPUT_SIZE * INPUT_SIZE) as f64;
            mae /= denom;
            mse /= denom;
            let psnr = if mse > 0.0 { 10.0 * (255.0 * 255.0 / mse).log10() } else { 99.0 };

            total_mae += mae;
            total_psnr += psnr;
            paired_count += 1;

            (Some(mae), Some(psnr), format!("│ MAE: {:4.1}/255 │ PSNR: {:4.1} dB", mae, psnr))
        } else {
            (None, None, String::new())
        };

        println!(
            "  [{:04}] Unique bins: {:3} │ Dominant: Bin {:3} (a={:3.0}, b={:3.0}) {:4.1}% {}",
            sample_idx, unique_bins, dom_bin, dom_a, dom_b, dom_pct, diff_str
        );

        // 4. Save side-by-side Triplet
        let has_orig = orig_img.is_some();
        let num_panels = if has_orig { 3 } else { 2 };
        let triplet_w = (INPUT_SIZE as u32) * num_panels + (num_panels - 1) * 2;
        let triplet_h = INPUT_SIZE as u32;

        let mut comp = RgbImage::from_pixel(triplet_w, triplet_h, Rgb([18, 18, 24]));
        let mut cur_x = 0u32;

        if let Some(ref orig) = orig_img {
            for y in 0..INPUT_SIZE as u32 {
                for x in 0..INPUT_SIZE as u32 {
                    comp.put_pixel(cur_x + x, y, *orig.get_pixel(x, y));
                }
            }
            cur_x += INPUT_SIZE as u32;
            for y in 0..INPUT_SIZE as u32 {
                comp.put_pixel(cur_x, y, Rgb([50, 50, 60]));
                comp.put_pixel(cur_x + 1, y, Rgb([50, 50, 60]));
            }
            cur_x += 2;
        }

        // Grayscale L
        for y in 0..INPUT_SIZE as u32 {
            for x in 0..INPUT_SIZE as u32 {
                comp.put_pixel(cur_x + x, y, *gray_img.get_pixel(x, y));
            }
        }
        cur_x += INPUT_SIZE as u32;
        for y in 0..INPUT_SIZE as u32 {
            comp.put_pixel(cur_x, y, Rgb([50, 50, 60]));
            comp.put_pixel(cur_x + 1, y, Rgb([50, 50, 60]));
        }
        cur_x += 2;

        // Reconstructed Target
        for y in 0..INPUT_SIZE as u32 {
            for x in 0..INPUT_SIZE as u32 {
                comp.put_pixel(cur_x + x, y, *recon_img.get_pixel(x, y));
            }
        }

        let triplet_filename = format!("sample_{:04}_triplet.png", sample_idx);
        let out_triplet = args.out_dir.join(&triplet_filename);
        comp.save(&out_triplet)?;

        cards.push(SampleCard {
            idx: sample_idx,
            triplet_rel_path: triplet_filename,
            unique_bins,
            dominant_bin: dom_bin,
            dom_a,
            dom_b,
            dom_pct,
            mae: mae_opt,
            psnr: psnr_opt,
        });

        // Optional individual files
        if args.individual {
            let sample_sub = args.out_dir.join(format!("sample_{:04}", sample_idx));
            fs::create_dir_all(&sample_sub)?;
            gray_img.save(sample_sub.join("grayscale_l.png"))?;
            recon_img.save(sample_sub.join("target_reconstructed.png"))?;
            if let Some(ref orig) = orig_img {
                orig.save(sample_sub.join("original.png"))?;
            }
        }
    }

    // 5. Generate interactive HTML gallery
    let html_path = args.out_dir.join("gallery.html");
    let mut html = File::create(&html_path)?;

    writeln!(html, "<!DOCTYPE html>")?;
    writeln!(html, "<html lang='en'><head><meta charset='UTF-8'><title>Yumon Colorizer Cache Inspector</title>")?;
    writeln!(html, "<style>")?;
    writeln!(html, "body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f111a; color: #e6edf3; margin: 0; padding: 24px; }}")?;
    writeln!(html, "h1 {{ color: #d2a8ff; margin-bottom: 8px; }}")?;
    writeln!(html, ".meta {{ color: #8b949e; margin-bottom: 24px; font-size: 14px; }}")?;
    writeln!(html, ".grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(780px, 1fr)); gap: 24px; }}")?;
    writeln!(html, ".card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; padding: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }}")?;
    writeln!(html, ".card img {{ width: 100%; height: auto; display: block; border-radius: 4px; border: 1px solid #21262d; }}")?;
    writeln!(html, ".card-title {{ font-size: 15px; font-weight: 600; color: #58a6ff; margin-bottom: 8px; display: flex; justify-content: space-between; }}")?;
    writeln!(html, ".card-stats {{ font-size: 12px; color: #8b949e; margin-top: 8px; line-height: 1.6; font-family: monospace; }}")?;
    writeln!(html, ".legend {{ background: #21262d; padding: 12px 16px; border-radius: 6px; margin-bottom: 24px; font-size: 13px; }}")?;
    writeln!(html, ".tag {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 600; margin-left: 6px; }}")?;
    writeln!(html, ".tag-mono {{ background: #388bfd33; color: #58a6ff; }}")?;
    writeln!(html, ".tag-color {{ background: #23863633; color: #3fb950; }}")?;
    writeln!(html, "</style></head><body>")?;

    writeln!(html, "<h1>🎨 Yumon Colorizer Cache Inspector</h1>")?;
    writeln!(html, "<div class='meta'>Cache: <code>{}</code> │ Total Records: {} │ Inspected: {}</div>", args.cache.display(), total_samples, num_to_export)?;
    writeln!(html, "<div class='legend'><strong>Panel Layout:</strong> Left: <code>Original DiffusionDB Image</code> │ Middle: <code>Cached Grayscale Lightness (L*)</code> │ Right: <code>Ground-Truth Reconstructed from 313 Color Bins</code></div>")?;

    writeln!(html, "<div class='grid'>")?;
    for card in &cards {
        let is_mono = card.unique_bins <= 3 && (card.dom_a.abs() < 10.0 && card.dom_b.abs() < 10.0);
        let tag_html = if is_mono {
            "<span class='tag tag-mono'>MONOCHROME / GRAYSCALE</span>"
        } else {
            "<span class='tag tag-color'>COLOR VIBRANT</span>"
        };

        let diff_html = match (card.mae, card.psnr) {
            (Some(m), Some(p)) => format!(" │ MAE: {:.2}/255 │ PSNR: {:.2} dB", m, p),
            _ => String::new(),
        };

        writeln!(html, "<div class='card'>")?;
        writeln!(html, "  <div class='card-title'><span>Sample #{:04} {}</span><span>Unique Bins: {}</span></div>", card.idx, tag_html, card.unique_bins)?;
        writeln!(html, "  <img src='{}' alt='Sample {:04}'>", card.triplet_rel_path, card.idx)?;
        writeln!(html, "  <div class='card-stats'>Dominant: Bin {} (a={:.0}, b={:.0}, {:.1}% of pixels){}</div>", card.dominant_bin, card.dom_a, card.dom_b, card.dom_pct, diff_html)?;
        writeln!(html, "</div>")?;
    }
    writeln!(html, "</div></body></html>")?;

    println!("\n--------------------------------------------------");
    if paired_count > 0 {
        println!("  Average Source vs Cache MAE : {:.2} / 255", total_mae / paired_count as f64);
        println!("  Average Source vs Cache PSNR: {:.2} dB", total_psnr / paired_count as f64);
    }
    println!("  ✓ Saved HTML Gallery to file:///{}", html_path.canonicalize().unwrap_or(html_path).display().to_string().replace('\\', "/").trim_start_matches("//?/"));
    println!("==================================================");

    Ok(())
}
