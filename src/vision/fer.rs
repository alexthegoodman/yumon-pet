/// FER2013 emotion dataset loader.
///
/// Directory layout (as provided):
///   data/fer2013-archive/
///     train/
///       angry/   *.jpg
///       disgust/ *.jpg
///       fear/    *.jpg
///       happy/   *.jpg
///       neutral/ *.jpg
///       sad/     *.jpg
///       surprise/*.jpg
///     test/
///       (same structure)
///
/// Images are 48×48 grayscale JPEGs.
/// We resize to IMG_SIZE (32), convert to RGB (duplicate channels),
/// and normalize with μ=0.5, σ=0.5.

use anyhow::{Result, Context};
#[cfg(target_os = "windows")]
use image::{imageops::FilterType, DynamicImage};
use std::path::{Path, PathBuf};

use crate::vision::{EMOTE_CLASSES, EMOTE_NAMES, IMG_SIZE};

#[derive(Debug, Clone)]
pub struct FerRecord {
    pub emote_idx: usize,
    /// Shape: [3, IMG_SIZE, IMG_SIZE] — CHW, normalized to [-1, 1]
    pub pixels:    Vec<f32>,
}

pub struct FerDataset {
    pub records: Vec<FerRecord>,
}

#[cfg(target_os = "windows")]
impl FerDataset {
    /// Load a split directory (e.g. "data/fer2013-archive/train").
    pub fn load(split_dir: &str) -> Result<Self> {
        let base = Path::new(split_dir);
        let mut records = Vec::new();

        for (idx, name) in EMOTE_NAMES.iter().enumerate() {
            let class_dir = base.join(name);
            if !class_dir.exists() {
                eprintln!("⚠️  FER dir missing: {:?} — skipping", class_dir);
                continue;
            }

            let entries = std::fs::read_dir(&class_dir)
                .with_context(|| format!("reading {:?}", class_dir))?;

            for entry in entries.flatten() {
                let path = entry.path();
                if !is_image(&path) { continue; }

                match load_fer_image(&path) {
                    Ok(pixels) => records.push(FerRecord { emote_idx: idx, pixels }),
                    Err(e)     => eprintln!("⚠️  skip {:?}: {e}", path),
                }
            }
        }

        println!("📦 FER2013 loaded: {} records from {}", records.len(), split_dir);
        Ok(Self { records })
    }

    pub fn len(&self) -> usize { self.records.len() }
    pub fn is_empty(&self) -> bool { self.records.is_empty() }
}

fn is_image(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("jpg" | "jpeg" | "png" | "bmp")
    )
}

#[cfg(target_os = "windows")]
fn load_fer_image(path: &Path) -> Result<Vec<f32>> {
    let img = image::open(path)
        .with_context(|| format!("open {:?}", path))?;

    // Resize to model input size, convert to RGB
    let img = img.resize_exact(IMG_SIZE as u32, IMG_SIZE as u32, FilterType::Lanczos3)
                 .to_rgb8();

    let mut pixels = vec![0.0f32; 3 * IMG_SIZE * IMG_SIZE];
    for (i, pixel) in img.pixels().enumerate() {
        let [r, g, b] = pixel.0;
        let hw = i; // row-major
        pixels[0 * IMG_SIZE * IMG_SIZE + hw] = r as f32 / 255.0 * 2.0 - 1.0;
        pixels[1 * IMG_SIZE * IMG_SIZE + hw] = g as f32 / 255.0 * 2.0 - 1.0;
        pixels[2 * IMG_SIZE * IMG_SIZE + hw] = b as f32 / 255.0 * 2.0 - 1.0;
    }

    Ok(pixels)
}

/// Collate a batch of FerRecords into flat tensors.
pub fn collate_batch(records: &[&FerRecord]) -> (Vec<f32>, Vec<usize>) {
    let mut pixels = Vec::with_capacity(records.len() * 3 * IMG_SIZE * IMG_SIZE);
    let mut labels = Vec::with_capacity(records.len());
    for r in records {
        pixels.extend_from_slice(&r.pixels);
        labels.push(r.emote_idx);
    }
    (pixels, labels)
}
