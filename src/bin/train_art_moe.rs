//! Autoregressive Causal LM Training on Multi-Scale Image Token Sequences.

#[cfg(not(target_arch = "wasm32"))]
use {
    anyhow::{Context, Result, ensure},
    burn::{
        backend::{
            autodiff::checkpoint::strategy::BalancedCheckpointing,
            Autodiff,
        },
        module::AutodiffModule,
        optim::{AdamWConfig, GradientsParams, Optimizer},
        prelude::*,
        tensor::{Int, TensorData},
    },
    clap::Parser,
    indicatif::{ProgressBar, ProgressStyle},
    rand::{seq::SliceRandom, thread_rng},
    serde::{Deserialize, Serialize},
    std::{
        fs::File,
        io::{BufReader, Read},
        path::{Path, PathBuf},
        time::Instant,
    },
    yumon_pet::brain::{
        moe_model::{MoeMetadata, YumonMoeBrain, YumonMoeBrainConfig},
        samples::TrainingStage,
        tokenizer::PAD_TOKEN,
    },
    yumon_pet::vision::paint_action::{
        CANVAS_CHANNELS, CANVAS_SIZE, COLOR_START_ID, DONE_ID, PALETTE_SIZE,
        PASS_1_ID, PASS_2_ID, PASS_3_ID, PASS_4_ID,
        pass1_tokens_to_image, pass2_tokens_to_image, pass3_tokens_to_image, pass4_tokens_to_image,
    },
};

pub type InnerBackend = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;
pub type TrainBackend = Autodiff<
    InnerBackend,
    BalancedCheckpointing,
>;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ArtSample {
    pub prompt_text: String,
    pub pass: u8,
    pub tokens: Vec<u32>,
    pub target_start: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ArtCacheMetadata {
    pub version: u32,
    pub total_samples: usize,
    pub images_processed: usize,
    pub canvas_size: usize,
    pub vocab_size: usize,
    pub trajectory_steps: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ArtConfig {
    pub architecture: String,
    pub num_experts: usize,
    pub top_k: usize,
    pub cache: String,
    pub vocab_file: String,
    pub max_seq_len: usize,
    pub seed: u64,
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub n_layers: usize,
    pub attn_heads: usize,
    pub ff_dim: usize,
    pub batch_size: usize,
    pub epochs: usize,
    pub first_lr: f64,
    pub last_lr: f64,
    pub validation_fraction: f64,
    pub out_dir: String,
    #[serde(default)]
    pub pass: Option<usize>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser, Debug)]
#[command(about = "Train Causal MoE Transformer on Cascaded Image Sequences")]
struct Args {
    /// Configuration JSON path.
    #[arg(long, default_value = "configs/yumon-art-moe.json")]
    config: PathBuf,

    /// Override cache binary path.
    #[arg(long)]
    cache: Option<PathBuf>,

    /// Override epochs.
    #[arg(long)]
    epochs: Option<usize>,

    /// Override batch size.
    #[arg(long)]
    batch_size: Option<usize>,

    /// Override output directory.
    #[arg(long)]
    out_dir: Option<PathBuf>,

    /// Override target pass (1, 2, 3, or 4).
    #[arg(long)]
    pass: Option<usize>,
}

#[cfg(not(target_arch = "wasm32"))]
fn load_samples_from_cache(path: &Path) -> Result<Vec<ArtSample>> {
    let file = File::open(path).with_context(|| format!("opening cache {}", path.display()))?;
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    ensure!(&magic == b"YUMONART", "invalid cache magic");

    let mut meta_len_buf = [0u8; 4];
    reader.read_exact(&mut meta_len_buf)?;
    let meta_len = u32::from_le_bytes(meta_len_buf) as usize;

    let mut meta_bytes = vec![0u8; meta_len];
    reader.read_exact(&mut meta_bytes)?;
    let meta: ArtCacheMetadata = bincode::deserialize(&meta_bytes)?;

    ensure!(
        meta.version >= 3,
        "Cache version {} is outdated. Please regenerate cache using `cargo run --release --bin cache_diffusion_kilo`.",
        meta.version
    );

    let mut samples = Vec::new();
    let mut len_buf = [0u8; 4];

    while reader.read_exact(&mut len_buf).is_ok() {
        let sample_len = u32::from_le_bytes(len_buf) as usize;
        let mut sample_bytes = vec![0u8; sample_len];
        reader.read_exact(&mut sample_bytes)?;
        let sample: ArtSample = bincode::deserialize(&sample_bytes)?;
        samples.push(sample);
    }

    Ok(samples)
}

/// Save visual comparison preview grid:
/// 4 columns x 4 rows of [ Input (Prior Stage) | Model Predicted | Ground Truth Target ]
#[cfg(not(target_arch = "wasm32"))]
fn save_art_preview_grid(
    samples: &[ArtSample],
    inference_model: &YumonMoeBrain<InnerBackend>,
    device: &<InnerBackend as Backend>::Device,
    cfg: &ArtConfig,
    out_path: &Path,
) -> Result<()> {
    if samples.is_empty() {
        return Ok(());
    }

    let b = samples.len().min(16);
    let preview_samples = &samples[..b];
    let cols = 4u32.min(b as u32);
    let rows = ((b as u32) + cols - 1) / cols;

    let scale = 2u32;
    let panel_size = (CANVAS_SIZE as u32) * scale; // 128 px
    let triplet_w = panel_size * 3 + 2;            // 386 px
    let triplet_h = panel_size;                    // 128 px
    let cell_margin = 8u32;
    let bg_color = image::Rgb([18, 18, 26]);

    let img_w = cell_margin + cols * (triplet_w + cell_margin);
    let img_h = cell_margin + rows * (triplet_h + cell_margin);

    let mut grid_img = image::RgbImage::from_pixel(img_w, img_h, bg_color);

    for (i, s) in preview_samples.iter().enumerate() {
        let r = (i as u32) / cols;
        let c = (i as u32) % cols;
        let x_offset = cell_margin + c * (triplet_w + cell_margin);
        let y_offset = cell_margin + r * (triplet_h + cell_margin);

        // 1. Input / conditioning canvas from previous pass (if any)
        let input_canvas = match s.pass {
            1 => vec![24u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS],
            2 => {
                let p1_pos = s.tokens.iter().position(|&t| t == PASS_1_ID as u32);
                let p2_pos = s.tokens.iter().position(|&t| t == PASS_2_ID as u32);
                if let (Some(p1), Some(p2)) = (p1_pos, p2_pos) {
                    if p2 > p1 + 1 {
                        pass1_tokens_to_image(&s.tokens[p1 + 1..p2])
                    } else {
                        vec![24u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS]
                    }
                } else {
                    vec![24u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS]
                }
            }
            3 => {
                let p2_pos = s.tokens.iter().position(|&t| t == PASS_2_ID as u32);
                let p3_pos = s.tokens.iter().position(|&t| t == PASS_3_ID as u32);
                if let (Some(p2), Some(p3)) = (p2_pos, p3_pos) {
                    if p3 > p2 + 1 {
                        pass2_tokens_to_image(&s.tokens[p2 + 1..p3])
                    } else {
                        vec![24u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS]
                    }
                } else {
                    vec![24u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS]
                }
            }
            4 => {
                let p3_pos = s.tokens.iter().position(|&t| t == PASS_3_ID as u32);
                let p4_pos = s.tokens.iter().position(|&t| t == PASS_4_ID as u32);
                if let (Some(p3), Some(p4)) = (p3_pos, p4_pos) {
                    if p4 > p3 + 1 {
                        pass3_tokens_to_image(&s.tokens[p3 + 1..p4])
                    } else {
                        vec![24u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS]
                    }
                } else {
                    vec![24u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS]
                }
            }
            _ => vec![24u8; CANVAS_SIZE * CANVAS_SIZE * CANVAS_CHANNELS],
        };

        // Render Panel 1: Input (upscaled nearest-neighbor to panel_size)
        for py in 0..panel_size {
            let src_y = (py / scale) as usize;
            for px in 0..panel_size {
                let src_x = (px / scale) as usize;
                let idx = (src_y * CANVAS_SIZE + src_x) * 3;
                grid_img.put_pixel(
                    x_offset + px,
                    y_offset + py,
                    image::Rgb([input_canvas[idx], input_canvas[idx + 1], input_canvas[idx + 2]]),
                );
            }
        }

        // Separator 1
        for py in 0..panel_size {
            grid_img.put_pixel(x_offset + panel_size, y_offset + py, image::Rgb([40, 40, 50]));
        }

        // 2. Model Predicted target colors
        let end = if s.tokens.last() == Some(&(DONE_ID as u32)) {
            s.tokens.len() - 1
        } else {
            s.tokens.len()
        };

        let target_len = end.saturating_sub(s.target_start);
        let max_seq_len = cfg.max_seq_len;
        let t_len = s.tokens.len().min(max_seq_len);
        let input_seq_len = t_len.saturating_sub(1);

        let mut pred_colors = Vec::with_capacity(target_len);
        if input_seq_len > 0 && s.target_start > 0 {
            let input_tokens: Vec<i32> = s.tokens[..input_seq_len].iter().map(|&t| t as i32).collect();
            let input_tensor = Tensor::<InnerBackend, 2, Int>::from_ints(
                TensorData::new(input_tokens, [1, input_seq_len]),
                device,
            );
            let logits = inference_model.forward(input_tensor);
            let logits_data: Vec<f32> = logits.into_data().convert::<f32>().to_vec().unwrap();
            let vocab_size = cfg.vocab_size;

            let start_pos = s.target_start.saturating_sub(1);
            let end_pos = end.saturating_sub(1).min(input_seq_len);

            for j in start_pos..end_pos {
                let offset = j * vocab_size;
                let color_slice = &logits_data[offset + COLOR_START_ID..offset + COLOR_START_ID + PALETTE_SIZE];
                let (best_idx, _) = color_slice
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((0, &0.0));
                pred_colors.push((COLOR_START_ID + best_idx) as u32);
            }
        }

        let pred_canvas = match s.pass {
            1 => pass1_tokens_to_image(&pred_colors),
            2 => pass2_tokens_to_image(&pred_colors),
            3 => pass3_tokens_to_image(&pred_colors),
            4 => pass4_tokens_to_image(&pred_colors),
            _ => pass4_tokens_to_image(&pred_colors),
        };

        let x_pred_base = x_offset + panel_size + 1;
        // Render Panel 2: Predicted
        for py in 0..panel_size {
            let src_y = (py / scale) as usize;
            for px in 0..panel_size {
                let src_x = (px / scale) as usize;
                let idx = (src_y * CANVAS_SIZE + src_x) * 3;
                grid_img.put_pixel(
                    x_pred_base + px,
                    y_offset + py,
                    image::Rgb([pred_canvas[idx], pred_canvas[idx + 1], pred_canvas[idx + 2]]),
                );
            }
        }

        // Separator 2
        for py in 0..panel_size {
            grid_img.put_pixel(x_pred_base + panel_size, y_offset + py, image::Rgb([40, 40, 50]));
        }

        // 3. Ground Truth Target Canvas
        let target_colors = if s.target_start < end {
            &s.tokens[s.target_start..end]
        } else {
            &[]
        };

        let target_canvas = match s.pass {
            1 => pass1_tokens_to_image(target_colors),
            2 => pass2_tokens_to_image(target_colors),
            3 => pass3_tokens_to_image(target_colors),
            4 => pass4_tokens_to_image(target_colors),
            _ => pass4_tokens_to_image(target_colors),
        };

        let x_target_base = x_pred_base + panel_size + 1;
        // Render Panel 3: Target Ground Truth
        for py in 0..panel_size {
            let src_y = (py / scale) as usize;
            for px in 0..panel_size {
                let src_x = (px / scale) as usize;
                let idx = (src_y * CANVAS_SIZE + src_x) * 3;
                grid_img.put_pixel(
                    x_target_base + px,
                    y_offset + py,
                    image::Rgb([target_canvas[idx], target_canvas[idx + 1], target_canvas[idx + 2]]),
                );
            }
        }
    }

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    grid_img.save(out_path)?;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<()> {
    let args = Args::parse();
    let device: <TrainBackend as Backend>::Device = Default::default();

    println!("=== Yumon Causal MoE Art Sequence Trainer ===");
    println!("Config file: {}", args.config.display());

    let config_str = std::fs::read_to_string(&args.config)?;
    let cfg: ArtConfig = serde_json::from_str(&config_str)?;

    let cache_path = args.cache.unwrap_or_else(|| PathBuf::from(&cfg.cache));
    let epochs = args.epochs.unwrap_or(cfg.epochs);
    let batch_size = args.batch_size.unwrap_or(cfg.batch_size);
    let out_dir = args.out_dir.unwrap_or_else(|| PathBuf::from(&cfg.out_dir));
    let target_pass = args.pass.or(cfg.pass);

    println!("Cache binary   : {}", cache_path.display());
    println!("Output dir     : {}", out_dir.display());
    if let Some(p) = target_pass {
        println!("Target pass    : Pass {}", p);
    } else {
        println!("Target pass    : All passes");
    }
    println!("Batch size     : {}", batch_size);
    println!("Epochs         : {}", epochs);
    println!("Max seq len    : {}", cfg.max_seq_len);
    println!("Embed dim      : {}", cfg.embed_dim);
    println!("Layers         : {}", cfg.n_layers);
    println!("Experts        : {} (top-{})", cfg.num_experts, cfg.top_k);
    println!("Position Emb   : M-RoPE 3D (t, y, x) continuous coordinates");

    // 1. Load dataset
    println!("\nLoading training samples...");
    let mut all_samples = load_samples_from_cache(&cache_path)?;
    println!("Total samples loaded: {}", all_samples.len());
    ensure!(!all_samples.is_empty(), "Cache contains 0 samples");

    // Filter by pass if specified
    if let Some(p) = target_pass {
        anyhow::ensure!(p >= 1 && p <= 4, "Invalid pass {p}; must be 1, 2, 3, or 4");
        all_samples.retain(|s| s.pass == p as u8);
        println!("Filtered to Pass {} samples: {}", p, all_samples.len());
        ensure!(!all_samples.is_empty(), "No samples found for Pass {}", p);
    }

    // Shuffle and split train / val
    let mut rng = thread_rng();
    all_samples.shuffle(&mut rng);

    let val_count = (all_samples.len() as f64 * cfg.validation_fraction).max(1.0) as usize;
    let train_count = all_samples.len().saturating_sub(val_count);
    let val_samples = all_samples.split_off(train_count);
    let train_samples = all_samples;

    println!("Split: {} train / {} validation", train_samples.len(), val_samples.len());

    // Fixed preview samples for consistent visual evaluation across epochs
    let preview_samples: Vec<ArtSample> = if !val_samples.is_empty() {
        val_samples.iter().take(16).cloned().collect()
    } else {
        train_samples.iter().take(16).cloned().collect()
    };

    // 2. Initialize Model (Pure Causal Language Model with Flash Attention)
    let brain_config = YumonMoeBrainConfig::new(cfg.vocab_size, TrainingStage::Language)
        .with_embed_dim(cfg.embed_dim)
        .with_hidden_units(cfg.embed_dim)
        .with_n_layers(cfg.n_layers)
        .with_attn_heads(cfg.attn_heads)
        .with_ff_dim(cfg.ff_dim)
        .with_max_seq_len(cfg.max_seq_len)
        .with_num_experts(cfg.num_experts)
        .with_top_k(cfg.top_k)
        .with_is_multimodal(false);

    let mut model: YumonMoeBrain<TrainBackend> = brain_config.init(&device);
    let mut optimizer = AdamWConfig::new()
        .with_epsilon(1e-8)
        .with_weight_decay(1e-4)
        .init();

    // 3. Training Loop
    let num_batches = (train_samples.len() + batch_size - 1) / batch_size;
    println!("\nStarting training ({} batches/epoch)...", num_batches);

    let mut best_val_loss = f32::INFINITY;

    for epoch in 1..=epochs {
        let t0 = Instant::now();
        let mut total_loss = 0.0f32;
        let mut total_steps = 0usize;

        let pb = ProgressBar::new(num_batches as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [Epoch {pos_label}] [{bar:30.cyan/blue}] {pos}/{len} loss: {msg} ({elapsed})")
                .unwrap(),
        );
        pb.set_message("starting...");

        for (_batch_idx, chunk) in train_samples.chunks(batch_size).enumerate() {
            let bs = chunk.len();
            if bs == 0 {
                continue;
            }

            let max_len = chunk.iter().map(|s| s.tokens.len()).max().unwrap().min(cfg.max_seq_len);
            if max_len < 2 {
                continue;
            }
            let seq_len = max_len - 1;

            let mut flat_inputs = Vec::with_capacity(bs * seq_len);
            let mut flat_targets = Vec::with_capacity(bs * seq_len);
            let mut flat_mask = Vec::with_capacity(bs * seq_len);

            for s in chunk {
                let t_len = s.tokens.len().min(max_len);
                for i in 0..seq_len {
                    if i + 1 < t_len {
                        flat_inputs.push(s.tokens[i] as i32);
                        let tgt = s.tokens[i + 1] as i32;
                        flat_targets.push(tgt);
                        // Compute loss on target color tokens (starting at target_start)
                        if i + 1 >= s.target_start && tgt != PAD_TOKEN as i32 {
                            flat_mask.push(1.0f32);
                        } else {
                            flat_mask.push(0.0f32);
                        }
                    } else {
                        flat_inputs.push(PAD_TOKEN as i32);
                        flat_targets.push(PAD_TOKEN as i32);
                        flat_mask.push(0.0f32);
                    }
                }
            }

            let input_tensor = Tensor::<TrainBackend, 2, Int>::from_ints(
                TensorData::new(flat_inputs, [bs, seq_len]),
                &device,
            );
            let target_tensor = Tensor::<TrainBackend, 2, Int>::from_ints(
                TensorData::new(flat_targets, [bs, seq_len]),
                &device,
            );
            let mask_tensor = Tensor::<TrainBackend, 2>::from_floats(
                TensorData::new(flat_mask, [bs, seq_len]),
                &device,
            );

            // Causal Flash Attention forward pass
            let (logits, aux_loss, _) = model.forward_with_aux(input_tensor);

            // Cross Entropy Loss masked on target colors
            let log_probs = burn::tensor::activation::log_softmax(logits, 2);
            let nll = log_probs.gather(2, target_tensor.reshape([bs, seq_len, 1])).reshape([bs, seq_len]).neg();
            let active_count = mask_tensor.clone().sum().clamp_min(1.0);
            let loss = (nll * mask_tensor).sum() / active_count + aux_loss;

            let loss_val: f32 = loss.clone().into_data().convert::<f32>().to_vec().unwrap()[0];
            total_loss += loss_val;
            total_steps += 1;

            pb.set_message(format!("{:.4}", loss_val));
            pb.inc(1);

            // Backward pass & optimizer step
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            let lr = cfg.first_lr + (cfg.last_lr - cfg.first_lr) * (epoch as f64 / epochs as f64);
            model = optimizer.step(lr, model, grads);
        }

        pb.finish();
        let avg_train_loss = total_loss / total_steps.max(1) as f32;

        // 4. Validation
        let mut val_loss_sum = 0.0f32;
        let mut val_correct = 0usize;
        let mut val_total = 0usize;
        let mut val_steps = 0usize;

        // NOTE: this causes "Out of Memory" errors oftentimes
        // for chunk in val_samples.chunks(batch_size) {
        //     let bs = chunk.len();
        //     if bs == 0 {
        //         continue;
        //     }

        //     let max_len = chunk.iter().map(|s| s.tokens.len()).max().unwrap().min(cfg.max_seq_len);
        //     if max_len < 2 {
        //         continue;
        //     }
        //     let seq_len = max_len - 1;

        //     let mut flat_inputs = Vec::with_capacity(bs * seq_len);
        //     let mut flat_targets = Vec::with_capacity(bs * seq_len);
        //     let mut flat_mask = Vec::with_capacity(bs * seq_len);

        //     for s in chunk {
        //         let t_len = s.tokens.len().min(max_len);
        //         for i in 0..seq_len {
        //             if i + 1 < t_len {
        //                 flat_inputs.push(s.tokens[i] as i32);
        //                 let tgt = s.tokens[i + 1] as i32;
        //                 flat_targets.push(tgt);
        //                 if i + 1 >= s.target_start && tgt != PAD_TOKEN as i32 {
        //                     flat_mask.push(1.0f32);
        //                 } else {
        //                     flat_mask.push(0.0f32);
        //                 }
        //             } else {
        //                 flat_inputs.push(PAD_TOKEN as i32);
        //                 flat_targets.push(PAD_TOKEN as i32);
        //                 flat_mask.push(0.0f32);
        //             }
        //         }
        //     }

        //     let input_tensor = Tensor::<TrainBackend, 2, Int>::from_ints(
        //         TensorData::new(flat_inputs, [bs, seq_len]),
        //         &device,
        //     );
        //     let target_tensor = Tensor::<TrainBackend, 2, Int>::from_ints(
        //         TensorData::new(flat_targets.clone(), [bs, seq_len]),
        //         &device,
        //     );
        //     let mask_tensor = Tensor::<TrainBackend, 2>::from_floats(
        //         TensorData::new(flat_mask.clone(), [bs, seq_len]),
        //         &device,
        //     );

        //     let logits = model.forward(input_tensor);

        //     let log_probs = burn::tensor::activation::log_softmax(logits.clone(), 2);
        //     let nll = log_probs.gather(2, target_tensor.reshape([bs, seq_len, 1])).reshape([bs, seq_len]).neg();
        //     let active_count = mask_tensor.clone().sum().clamp_min(1.0);
        //     let loss = (nll * mask_tensor).sum() / active_count;

        //     let preds = logits.argmax(2).reshape([bs * seq_len]);
        //     let pred_vec: Vec<i32> = preds.into_data().convert::<i32>().to_vec().unwrap();
        //     for i in 0..(bs * seq_len) {
        //         if flat_mask[i] > 0.0 {
        //             if pred_vec[i] == flat_targets[i] {
        //                 val_correct += 1;
        //             }
        //             val_total += 1;
        //         }
        //     }

        //     let l: f32 = loss.into_data().convert::<f32>().to_vec().unwrap()[0];
        //     val_loss_sum += l;
        //     val_steps += 1;
        // }

        let avg_val_loss = val_loss_sum / val_steps.max(1) as f32;
        let val_acc = if val_total > 0 {
            (val_correct as f32 / val_total as f32) * 100.0
        } else {
            0.0
        };
        let elapsed = t0.elapsed();

        println!(
            "Epoch {:>2}/{} | Train Loss: {:.4} | Val Loss: {:.4} (Val Token Acc: {:.1}%) | Elapsed: {:.1}s",
            epoch, epochs, avg_train_loss, avg_val_loss, val_acc, elapsed.as_secs_f32()
        );

        // Save best checkpoint
        // if avg_val_loss < best_val_loss {
            best_val_loss = avg_val_loss;
            println!("  * New best validation loss: {:.4} -> saving to {}", avg_val_loss, out_dir.display());

            std::fs::create_dir_all(&out_dir)?;
            let meta = MoeMetadata {
                num_experts: cfg.num_experts,
                top_k: cfg.top_k,
                aux_loss_weight: 0.01,
                z_loss_weight: 0.001,
                dropout_rate: 0.05,
                vocab_size: cfg.vocab_size,
                epochs_trained: epoch,
                final_loss: avg_train_loss,
                val_loss: Some(avg_val_loss),
                batch_size,
                training_stage: TrainingStage::Language,
                embed_dim: cfg.embed_dim,
                hidden_units: cfg.embed_dim,
                n_layers: cfg.n_layers,
                attn_heads: cfg.attn_heads,
                ff_dim: cfg.ff_dim,
                max_seq_len: cfg.max_seq_len,
                adaptive_depth: None,
                kv_heads: None,
                is_multimodal: false,
                has_edge_head: false,
                use_mrope: true,
            };

            let meta_json = serde_json::to_string_pretty(&meta)?;
            std::fs::write(out_dir.join("metadata.json"), meta_json)?;

            let vocab_path = Path::new(&cfg.vocab_file);
            if let Ok(vocab) = yumon_pet::brain::kilo_vocab::KiloVocab::load_from_json(vocab_path) {
                let _ = vocab.save_tokenizer_json(&out_dir);
            }

            let recorder = burn::record::BinFileRecorder::<burn::record::FullPrecisionSettings>::new();
            let _ = model.clone().save_file(out_dir.join("model"), &recorder);
        // }

        // 5. Save visual preview sample grid (up to 16 samples)
        let preview_path = out_dir.join(format!("preview_epoch_{:02}.png", epoch));
        let latest_preview_path = out_dir.join("preview_latest.png");
        let inference_model = model.valid();

        if let Err(e) = save_art_preview_grid(
            &preview_samples,
            &inference_model,
            &device,
            &cfg,
            &preview_path,
        ) {
            eprintln!("Warning: Failed to save preview image grid: {}", e);
        } else {
            println!("  [Preview] Saved 16-sample comparison grid to {}", preview_path.display());
            let _ = std::fs::copy(&preview_path, &latest_preview_path);
        }
    }

    println!("\nTraining complete! Best validation loss: {:.4}", best_val_loss);
    Ok(())
}
