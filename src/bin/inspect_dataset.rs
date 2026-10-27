use std::fs::File;
use std::io::BufReader;
use std::collections::HashMap;

#[derive(serde::Deserialize, Debug)]
pub struct ArtCacheMetadata {
    pub version: u32,
    pub total_samples: usize,
    pub images_processed: usize,
    pub canvas_size: usize,
    pub vocab_size: usize,
    pub trajectory_steps: usize,
}

#[derive(serde::Deserialize, Clone, Debug)]
pub struct ArtSample {
    pub prompt_text: String,
    pub pass: u8,
    pub tokens: Vec<u32>,
    pub target_start: usize,
    #[serde(default)]
    pub spatial_weights: Option<Vec<f32>>,
}

#[derive(serde::Deserialize, Clone, Debug)]
pub struct ArtSampleV3 {
    pub prompt_text: String,
    pub pass: u8,
    pub tokens: Vec<u32>,
    pub target_start: usize,
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

    println!("Metadata: {:?}", meta);

    let mut pass_counts: HashMap<u8, usize> = HashMap::new();
    let mut total = 0;
    let mut sample_len_bytes = [0u8; 4];

    while std::io::Read::read_exact(&mut reader, &mut sample_len_bytes).is_ok() {
        let sample_len = u32::from_le_bytes(sample_len_bytes) as usize;
        let mut sample_bytes = vec![0u8; sample_len];
        std::io::Read::read_exact(&mut reader, &mut sample_bytes)?;
        let sample: ArtSample = if meta.version == 3 {
            let v3: ArtSampleV3 = bincode::deserialize(&sample_bytes)?;
            ArtSample {
                prompt_text: v3.prompt_text,
                pass: v3.pass,
                tokens: v3.tokens,
                target_start: v3.target_start,
                spatial_weights: None,
            }
        } else {
            bincode::deserialize(&sample_bytes)?
        };
        *pass_counts.entry(sample.pass).or_insert(0) += 1;
        total += 1;
        if total <= 3 {
            let sw_info = if let Some(ref sw) = sample.spatial_weights {
                let eyes = sw.iter().filter(|&&w| w >= 9.0).count();
                format!(", spatial_weights: Some(eye_toks={})", eyes)
            } else {
                "".to_string()
            };
            println!("Sample {}: pass={}, prompt=\"{}\", target_start={}, total_tokens={}{}",
                total, sample.pass, sample.prompt_text, sample.target_start, sample.tokens.len(), sw_info);
            if sample.pass == 3 {
                println!("Sample 3 tokens: total={}", sample.tokens.len());
                println!("Prefix before conditioning: {:?}", &sample.tokens[..sample.target_start.min(20)]);
                println!("Marker before conditioning (at prompt end): {:?}", sample.tokens[sample.target_start - 257]);
                println!("First 10 cond tokens: {:?}", &sample.tokens[sample.target_start - 256 .. sample.target_start - 246]);
                println!("Last 10 cond tokens: {:?}", &sample.tokens[sample.target_start - 10 .. sample.target_start]);
                println!("Marker at target_start - 1: {:?}", sample.tokens[sample.target_start - 1]);
                println!("First 10 target tokens: {:?}", &sample.tokens[sample.target_start .. sample.target_start + 10]);
                let min_c = sample.tokens[sample.target_start - 256 .. sample.target_start - 1].iter().min();
                let max_c = sample.tokens[sample.target_start - 256 .. sample.target_start - 1].iter().max();
                println!("Cond token range: min={:?}, max={:?}", min_c, max_c);
                let min_t = sample.tokens[sample.target_start .. sample.tokens.len() - 1].iter().min();
                let max_t = sample.tokens[sample.target_start .. sample.tokens.len() - 1].iter().max();
                println!("Target token range: min={:?}, max={:?}", min_t, max_t);
            }
        }
    }

    println!("Read {} total samples", total);
    for (p, count) in &pass_counts {
        println!("  Pass {}: {} samples", p, count);
    }

    Ok(())
}
