/// CIFAR-100 binary format loader.
///
/// File format (cifar-100-binary/train.bin, test.bin):
///   Each record = 1 coarse_label byte + 1 fine_label byte + 3072 pixel bytes (RGB, row-major)
///   Total records: 50 000 train / 10 000 test
///
/// Pixel values are normalized to [0, 1] and mean/std normalized with
/// CIFAR-100 channel statistics:
///   mean = [0.5071, 0.4867, 0.4408]
///   std  = [0.2675, 0.2565, 0.2761]

use anyhow::Result;
use std::fs::File;
use std::io::Read;

pub const RECORD_BYTES: usize = 2 + 3 * 32 * 32; // coarse + fine + pixels

// CIFAR-100 ImageNet-style channel normalization constants
const MEAN: [f32; 3] = [0.5071, 0.4867, 0.4408];
const STD:  [f32; 3] = [0.2675, 0.2565, 0.2761];

#[derive(Debug, Clone)]
pub struct CifarRecord {
    pub coarse_label: u8,
    pub fine_label:   u8,
    /// Shape: [3, 32, 32] — CHW order, normalized
    pub pixels:       Vec<f32>,
}

pub struct CifarDataset {
    pub records: Vec<CifarRecord>,
}

impl CifarDataset {
    /// Load a CIFAR-100 binary file (train.bin or test.bin).
    pub fn load(path: &str) -> Result<Self> {
        let mut f = File::open(path)?;
        let mut raw = Vec::new();
        f.read_to_end(&mut raw)?;

        let n = raw.len() / RECORD_BYTES;
        let mut records = Vec::with_capacity(n);

        for i in 0..n {
            let base = i * RECORD_BYTES;
            let coarse_label = raw[base];
            let fine_label   = raw[base + 1];

            let pixel_bytes = &raw[base + 2..base + RECORD_BYTES];

            // CIFAR stores pixels as [R plane 1024 bytes][G plane 1024][B plane 1024]
            // We want CHW float normalized
            let mut pixels = vec![0.0f32; 3 * 32 * 32];
            for c in 0..3 {
                for hw in 0..1024 {
                    let raw_val = pixel_bytes[c * 1024 + hw] as f32 / 255.0;
                    pixels[c * 1024 + hw] = (raw_val - MEAN[c]) / STD[c];
                }
            }

            records.push(CifarRecord { coarse_label, fine_label, pixels });
        }

        println!("📦 CIFAR-100 loaded: {} records from {}", n, path);
        Ok(Self { records })
    }

    pub fn len(&self) -> usize { self.records.len() }
    pub fn is_empty(&self) -> bool { self.records.is_empty() }
}

/// Returns flat [batch * 3 * 32 * 32] pixel data and [batch] label indices.
pub fn collate_batch(records: &[&CifarRecord]) -> (Vec<f32>, Vec<usize>) {
    let mut pixels = Vec::with_capacity(records.len() * 3 * 32 * 32);
    let mut labels = Vec::with_capacity(records.len());

    for r in records {
        pixels.extend_from_slice(&r.pixels);
        labels.push(r.fine_label as usize);
    }

    (pixels, labels)
}
