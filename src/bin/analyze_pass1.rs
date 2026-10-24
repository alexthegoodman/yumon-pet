use anyhow::{Context, Result};
use burn::{
    prelude::*,
    tensor::{Int, TensorData},
};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufReader, Read},
    path::Path,
};
use yumon_pet::brain::{
    kilo_vocab::KiloVocab,
    moe_model::YumonMoeBrain,
    tokenizer::BOS_TOKEN,
    yumon_talk::TranslationContext,
};
use yumon_pet::vision::paint_action::{
    ART_VOCAB_SIZE, COLOR_START_ID, PALETTE_SIZE, PASS_1_ID,
    build_cascaded_inference_coords, palette_to_rgb,
};

type BackendType = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;

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
}

fn load_pass1_samples(path: &Path, max_samples: usize) -> Result<Vec<ArtSample>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    let mut meta_len_buf = [0u8; 4];
    reader.read_exact(&mut meta_len_buf)?;
    let meta_len = u32::from_le_bytes(meta_len_buf) as usize;
    let mut meta_bytes = vec![0u8; meta_len];
    reader.read_exact(&mut meta_bytes)?;

    let mut samples = Vec::new();
    let mut len_buf = [0u8; 4];
    while reader.read_exact(&mut len_buf).is_ok() {
        let sample_len = u32::from_le_bytes(len_buf) as usize;
        let mut sample_bytes = vec![0u8; sample_len];
        reader.read_exact(&mut sample_bytes)?;
        let sample: ArtSample = bincode::deserialize(&sample_bytes)?;
        if sample.pass == 1 {
            samples.push(sample);
            if samples.len() >= max_samples {
                break;
            }
        }
    }
    Ok(samples)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&x| x / sum).collect()
}

fn main() -> Result<()> {
    println!("=== Pass 1 Diagnostic & Failure Mode Analysis ===\n");

    let cache_path = Path::new("training-cache/diffusion_kilo_samples.bin");
    if cache_path.exists() {
        println!("Analyzing ground-truth Pass 1 dataset (up to 2000 samples)...");
        let samples = load_pass1_samples(cache_path, 2000)?;
        println!("Loaded {} Pass 1 samples.", samples.len());

        let mut total_color_counts: HashMap<u32, usize> = HashMap::new();
        let mut unique_colors_per_img = Vec::new();
        let mut same_as_prev_count = 0usize;
        let mut same_as_above_count = 0usize;
        let mut total_spatial_transitions = 0usize;
        let mut total_tokens = 0usize;

        for s in &samples {
            let colors: Vec<u32> = s.tokens[s.target_start..]
                .iter()
                .filter(|&&t| t >= COLOR_START_ID as u32 && t < (COLOR_START_ID + PALETTE_SIZE) as u32)
                .cloned()
                .collect();
            if colors.len() == 64 {
                let uniq: HashSet<u32> = colors.iter().cloned().collect();
                unique_colors_per_img.push(uniq.len());

                for (idx, &c) in colors.iter().enumerate() {
                    *total_color_counts.entry(c).or_insert(0) += 1;
                    total_tokens += 1;
                    if idx > 0 && idx % 8 != 0 {
                        total_spatial_transitions += 1;
                        if c == colors[idx - 1] {
                            same_as_prev_count += 1;
                        }
                    }
                    if idx >= 8 {
                        if c == colors[idx - 8] {
                            same_as_above_count += 1;
                        }
                    }
                }
            }
        }

        let avg_unique = unique_colors_per_img.iter().sum::<usize>() as f32 / unique_colors_per_img.len().max(1) as f32;
        println!("Dataset Pass 1 Stats:");
        println!("  - Average unique colors per 8x8 canvas: {:.2} (min: {}, max: {})",
            avg_unique,
            unique_colors_per_img.iter().min().unwrap_or(&0),
            unique_colors_per_img.iter().max().unwrap_or(&0)
        );
        println!("  - P(color_t == color_{{t-1}} | same row): {:.2}%",
            (same_as_prev_count as f32 / total_spatial_transitions.max(1) as f32) * 100.0
        );
        println!("  - P(color_t == color_{{t-8}} | row above): {:.2}%",
            (same_as_above_count as f32 / (unique_colors_per_img.len() * 56).max(1) as f32) * 100.0
        );

        // Sort top colors
        let mut top_colors: Vec<(u32, usize)> = total_color_counts.into_iter().collect();
        top_colors.sort_by(|a, b| b.1.cmp(&a.1));
        println!("  - Top 10 most frequent colors in dataset:");
        for (i, (tok, count)) in top_colors.iter().take(10).enumerate() {
            let p_idx = (tok - COLOR_START_ID as u32) as u8;
            let [r, g, b] = palette_to_rgb(p_idx);
            let pct = (*count as f32 / total_tokens as f32) * 100.0;
            println!("      #{}: token {} (palette {}) RGB({},{},{}) -> {:.2}% ({} times)",
                i + 1, tok, p_idx, r, g, b, pct, count
            );
        }
    }

    // Now test the Pass 1 model with different prompts
    let ckpt_path = Path::new("checkpoints/yumon-art-pass1-db");
    if !ckpt_path.exists() {
        println!("Checkpoint {} not found, skipping model inference analysis.", ckpt_path.display());
        return Ok(());
    }

    println!("\nLoading model from {}...", ckpt_path.display());
    let device: <BackendType as Backend>::Device = Default::default();
    let (model, _, _) = YumonMoeBrain::<BackendType>::load(ckpt_path.to_str().unwrap(), &device)?;

    let vocab_path = Path::new("training-cache/diffusion_talk_vocab.json");
    let vocab = KiloVocab::load_from_json(vocab_path)?;

    let test_prompts = [
        "a bright red sports car on asphalt",
        "a lush green dense forest canopy with moss",
        "deep blue ocean waves under clear sky",
    ];

    for prompt in &test_prompts {
        println!("\n------------------------------------------------------------");
        println!("Analyzing Prompt: \"{}\"", prompt);
        let mut ctx = TranslationContext::new();
        let prompt_tokens = vocab.encode(prompt, &mut ctx)?;

        let mut p1_prefix = vec![BOS_TOKEN as i32];
        for &tok in &prompt_tokens {
            p1_prefix.push(tok as i32);
        }
        p1_prefix.push(PASS_1_ID as i32);

        let p1_coords = build_cascaded_inference_coords(&p1_prefix, 1, 64);
        let mut cache_p1 = model.new_kv_cache_mrope(&[p1_coords.clone()], &device);
        let prefill_tensor = Tensor::<BackendType, 2, Int>::from_ints(
            TensorData::new(p1_prefix.clone(), [1, p1_prefix.len()]),
            &device,
        );
        let logits = model.forward_cached(prefill_tensor, &mut cache_p1);
        let mut last_logit: Vec<f32> = logits
            .slice([0..1, p1_prefix.len() - 1..p1_prefix.len(), 0..ART_VOCAB_SIZE])
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();

        let mut greedy_tokens = Vec::new();
        let mut step_top1_probs = Vec::new();
        let mut step_top1_tokens = Vec::new();

        for step in 0..64 {
            let color_logits = &last_logit[COLOR_START_ID..COLOR_START_ID + PALETTE_SIZE];
            let probs = softmax(color_logits);

            // Find top 3 candidates
            let mut indexed_probs: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
            indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let top1_idx = indexed_probs[0].0;
            let top1_prob = indexed_probs[0].1;
            let top1_rgb = palette_to_rgb(top1_idx as u8);

            greedy_tokens.push(top1_idx as u8);
            step_top1_probs.push(top1_prob);
            step_top1_tokens.push(top1_idx);

            if step == 0 || step == 1 || step == 8 || step == 32 || step == 63 {
                println!("  Step {:>2} (row {}, col {}):", step, step / 8, step % 8);
                for k in 0..3 {
                    let (p_idx, p_val) = indexed_probs[k];
                    let [r, g, b] = palette_to_rgb(p_idx as u8);
                    println!("     Rank {}: Palette {:>3} RGB({:>3},{:>3},{:>3}) prob: {:.3}% (logit: {:.3})",
                        k + 1, p_idx, r, g, b, p_val * 100.0, color_logits[p_idx]
                    );
                }
            }

            let next_token = (COLOR_START_ID + top1_idx) as u32;
            let step_tensor = Tensor::<BackendType, 2, Int>::from_ints(
                TensorData::new(vec![next_token as i32], [1, 1]),
                &device,
            );
            let step_logits = model.forward_cached(step_tensor, &mut cache_p1);
            last_logit = step_logits.reshape([ART_VOCAB_SIZE]).into_data().convert::<f32>().to_vec().unwrap();
        }

        let test_strategies: Vec<(&str, f32, usize, f32, f32, f32)> = vec![
            ("Baseline (current yumon_paint)", 0.7, 12, 0.90, 0.0, 1.0),
            ("Higher Temp (1.0, top_k 24, top_p 0.95)", 1.0, 24, 0.95, 0.0, 1.0),
            ("Frequency Penalty (0.8) + Presence Penalty (1.5)", 0.85, 20, 0.95, 0.8, 1.5),
            ("Aggressive Frequency Penalty (1.5) + Temp 1.0", 1.0, 24, 0.95, 1.5, 2.0),
        ];

        for (strat_name, temp, top_k, top_p, freq_pen, pres_pen) in test_strategies {
            let mut cache_strat = model.new_kv_cache_mrope(&[p1_coords.clone()], &device);
            let prefill_tensor = Tensor::<BackendType, 2, Int>::from_ints(
                TensorData::new(p1_prefix.clone(), [1, p1_prefix.len()]),
                &device,
            );
            let logits = model.forward_cached(prefill_tensor, &mut cache_strat);
            let mut current_logit: Vec<f32> = logits
                .slice([0..1, p1_prefix.len() - 1..p1_prefix.len(), 0..ART_VOCAB_SIZE])
                .into_data()
                .convert::<f32>()
                .to_vec()
                .unwrap();

            let mut sampled_colors = Vec::new();
            let mut color_counts: HashMap<usize, usize> = HashMap::new();

            for _ in 0..64 {
                let mut working_logits: Vec<f32> = current_logit[COLOR_START_ID..COLOR_START_ID + PALETTE_SIZE].to_vec();

                // Apply frequency and presence penalties across sequence
                for (&c, &cnt) in &color_counts {
                    if c < working_logits.len() {
                        working_logits[c] -= freq_pen * (cnt as f32) + pres_pen;
                    }
                }

                // Top-K
                let mut candidates: Vec<(usize, f32)> = working_logits.iter().cloned().enumerate().collect();
                candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                if top_k > 0 && top_k < candidates.len() {
                    candidates.truncate(top_k);
                }

                // Softmax
                let max_l = candidates[0].1;
                let mut exp_p: Vec<(usize, f32)> = candidates.into_iter().map(|(idx, l)| (idx, ((l - max_l) / temp).exp())).collect();
                let sum_p: f32 = exp_p.iter().map(|(_, p)| p).sum();
                for (_, p) in &mut exp_p { *p /= sum_p; }

                // Top-P
                if top_p > 0.0 && top_p < 1.0 {
                    let mut cum_p = 0.0;
                    let mut cut = exp_p.len();
                    for (i, &(_, p)) in exp_p.iter().enumerate() {
                        cum_p += p;
                        if cum_p >= top_p { cut = (i + 1).max(1); break; }
                    }
                    exp_p.truncate(cut);
                    let renorm: f32 = exp_p.iter().map(|(_, p)| p).sum();
                    if renorm > 0.0 { for (_, p) in &mut exp_p { *p /= renorm; } }
                }

                // Sample
                let r = rand::random::<f32>();
                let mut acc = 0.0;
                let mut chosen = exp_p.last().map(|&(idx, _)| idx).unwrap_or(0);
                for &(idx, p) in &exp_p {
                    acc += p;
                    if acc >= r { chosen = idx; break; }
                }

                *color_counts.entry(chosen).or_insert(0) += 1;
                sampled_colors.push(chosen as u8);

                let next_token = (COLOR_START_ID + chosen) as u32;
                let step_tensor = Tensor::<BackendType, 2, Int>::from_ints(
                    TensorData::new(vec![next_token as i32], [1, 1]),
                    &device,
                );
                let step_logits = model.forward_cached(step_tensor, &mut cache_strat);
                current_logit = step_logits.reshape([ART_VOCAB_SIZE]).into_data().convert::<f32>().to_vec().unwrap();
            }

            let uniq: HashSet<u8> = sampled_colors.iter().cloned().collect();
            let mut top_counts: Vec<(u8, usize)> = color_counts.into_iter().map(|(k, v)| (k as u8, v)).collect();
            top_counts.sort_by(|a, b| b.1.cmp(&a.1));
            let dominant_pct = (top_counts[0].1 as f32 / 64.0) * 100.0;

            println!("    Strategy: {:<45} -> Unique colors: {:>2}, Top color: {:>2}% (palette {:>3})",
                strat_name, uniq.len(), dominant_pct as usize, top_counts[0].0
            );

            // Print 8x8 terminal preview
            for row in 0..8 {
                print!("      ");
                for col in 0..8 {
                    let p_idx = sampled_colors[row * 8 + col];
                    let [r, g, b] = palette_to_rgb(p_idx);
                    print!("\x1b[38;2;{};{};{}m██\x1b[0m", r, g, b);
                }
                println!();
            }
        }

    }

    Ok(())
}
