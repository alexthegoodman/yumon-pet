use std::fs::File;
use std::io::BufReader;
use yumon_pet::vision::paint_action::PaintAction;

#[derive(serde::Deserialize)]
pub struct ArtCacheMetadata {
    pub version: u32,
    pub total_samples: usize,
    pub images_processed: usize,
    pub canvas_size: usize,
    pub vocab_size: usize,
    pub trajectory_steps: usize,
}

#[derive(serde::Deserialize)]
pub struct ArtSample {
    pub prompt_text: String,
    pub prompt_tokens: Vec<u32>,
    pub canvas_hwc: Vec<u8>,
    pub action_tokens: Vec<u32>,
    pub is_done: bool,
    #[serde(default)]
    pub target_edge: f32,
}

fn main() -> anyhow::Result<()> {
    let cache_path = std::env::args().nth(1).unwrap_or_else(|| "training-cache/diffusion_kilo_samples.bin".to_string());
    let file = File::open(&cache_path)?;
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 8];
    std::io::Read::read_exact(&mut reader, &mut magic)?;
    let mut meta_len_bytes = [0u8; 4];
    std::io::Read::read_exact(&mut reader, &mut meta_len_bytes)?;
    let meta_len = u32::from_le_bytes(meta_len_bytes) as usize;
    let mut meta_bytes = vec![0u8; meta_len];
    std::io::Read::read_exact(&mut reader, &mut meta_bytes)?;
    let meta: ArtCacheMetadata = bincode::deserialize(&meta_bytes)?;

    println!("Total samples: {}", meta.total_samples);

    let mut color_counts = [0usize; 64];
    let mut x_counts = [0usize; 64];
    let mut y_counts = [0usize; 64];

    for _ in 0..meta.total_samples {
        let mut sample_len_bytes = [0u8; 4];
        if std::io::Read::read_exact(&mut reader, &mut sample_len_bytes).is_err() {
            break;
        }
        let sample_len = u32::from_le_bytes(sample_len_bytes) as usize;
        let mut sample_bytes = vec![0u8; sample_len];
        std::io::Read::read_exact(&mut reader, &mut sample_bytes)?;
        let sample: ArtSample = bincode::deserialize(&sample_bytes)?;

        let toks: Vec<usize> = sample.action_tokens.iter().map(|&t| t as usize).collect();
        if let Some(PaintAction::Paint { x, y, color, .. }) = PaintAction::from_tokens(&toks) {
            color_counts[color as usize] += 1;
            x_counts[x as usize] += 1;
            y_counts[y as usize] += 1;
        }
    }

    println!("\nTop 5 colors in dataset:");
    let mut colors_sorted: Vec<(usize, usize)> = color_counts.iter().copied().enumerate().collect();
    colors_sorted.sort_by(|a, b| b.1.cmp(&a.1));
    for (c, cnt) in &colors_sorted[..5] {
        println!("  Color idx {}: count {} ({:.1}%)", c, cnt, *cnt as f32 / meta.total_samples as f32 * 100.0);
    }

    println!("\nTop 5 X coords in dataset:");
    let mut x_sorted: Vec<(usize, usize)> = x_counts.iter().copied().enumerate().collect();
    x_sorted.sort_by(|a, b| b.1.cmp(&a.1));
    for (x, cnt) in &x_sorted[..5] {
        println!("  X coord {}: count {} ({:.1}%)", x, cnt, *cnt as f32 / meta.total_samples as f32 * 100.0);
    }

    println!("\nTop 5 Y coords in dataset:");
    let mut y_sorted: Vec<(usize, usize)> = y_counts.iter().copied().enumerate().collect();
    y_sorted.sort_by(|a, b| b.1.cmp(&a.1));
    for (y, cnt) in &y_sorted[..5] {
        println!("  Y coord {}: count {} ({:.1}%)", y, cnt, *cnt as f32 / meta.total_samples as f32 * 100.0);
    }

    Ok(())
}
