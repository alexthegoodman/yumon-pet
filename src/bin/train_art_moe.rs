//! Autoregressive Multimodal MoE Training on Bounding Box Art Sequences.

#[cfg(not(target_arch = "wasm32"))]
use {
    anyhow::{Context, Result, ensure},
    burn::{
        backend::{
            autodiff::checkpoint::strategy::BalancedCheckpointing,
            Autodiff,
        },
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
        tokenizer::{BOS_TOKEN, EOS_TOKEN, PAD_TOKEN},
    },
    yumon_pet::vision::COLOR_START_ID,
};

pub type TrainBackend = Autodiff<
    burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>,
    BalancedCheckpointing,
>;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ArtSample {
    pub prompt_text: String,
    pub prompt_tokens: Vec<u32>,
    pub canvas_hwc: Vec<u8>,
    pub action_tokens: Vec<u32>,
    pub is_done: bool,
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
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser, Debug)]
#[command(about = "Train the Multimodal Vision MoE on Bounding Box Art Sequences")]
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

/// Compute cross-entropy loss over non-PAD supervised tokens only.
fn masked_token_ce<B: Backend>(
    logits: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
    supervised: usize,
) -> Tensor<B, 1> {
    let [rows, _] = logits.dims();
    let mask = targets.clone().not_equal_elem(PAD_TOKEN as i32).float();
    let log_probs = burn::tensor::activation::log_softmax(logits, 1);
    let nll = log_probs.gather(1, targets.reshape([rows, 1])).reshape([rows]).neg();
    (nll * mask).sum() / (supervised.max(1) as f64)
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<()> {
    let args = Args::parse();
    let device: <TrainBackend as Backend>::Device = Default::default();

    println!("=== Yumon Multimodal MoE Art Trainer: Bounding Box Studio ===");
    println!("Config file: {}", args.config.display());

    let config_str = std::fs::read_to_string(&args.config)?;
    let cfg: ArtConfig = serde_json::from_str(&config_str)?;

    let cache_path = args.cache.unwrap_or_else(|| PathBuf::from(&cfg.cache));
    let epochs = args.epochs.unwrap_or(cfg.epochs);
    let batch_size = args.batch_size.unwrap_or(cfg.batch_size);
    let out_dir = args.out_dir.unwrap_or_else(|| PathBuf::from(&cfg.out_dir));

    println!("Cache binary   : {}", cache_path.display());
    println!("Output dir     : {}", out_dir.display());
    println!("Batch size     : {}", batch_size);
    println!("Epochs         : {}", epochs);
    println!("Embed dim      : {}", cfg.embed_dim);
    println!("Layers         : {}", cfg.n_layers);
    println!("Experts        : {} (top-{})", cfg.num_experts, cfg.top_k);

    // 1. Load dataset
    println!("\nLoading training samples...");
    let mut all_samples = load_samples_from_cache(&cache_path)?;
    println!("Total samples loaded: {}", all_samples.len());
    ensure!(!all_samples.is_empty(), "Cache contains 0 samples");

    // Shuffle and split train / val
    let mut rng = thread_rng();
    all_samples.shuffle(&mut rng);

    let val_count = (all_samples.len() as f64 * cfg.validation_fraction).max(1.0) as usize;
    let train_count = all_samples.len().saturating_sub(val_count);
    let val_samples = all_samples.split_off(train_count);
    let train_samples = all_samples;

    println!("Split: {} train / {} validation", train_samples.len(), val_samples.len());

    // 2. Initialize Model
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

            // Prepare batch input & target sequences
            let mut batch_inputs = Vec::with_capacity(bs);
            let mut batch_targets = Vec::with_capacity(bs);
            let mut max_len = 0;

            for s in chunk {
                let mut full_seq = vec![BOS_TOKEN as i32];
                for &t in &s.prompt_tokens {
                    full_seq.push(t as i32);
                }
                let prompt_end = full_seq.len();
                for &t in &s.action_tokens {
                    full_seq.push(t as i32);
                }
                full_seq.push(EOS_TOKEN as i32);

                if full_seq.len() > cfg.max_seq_len {
                    full_seq.truncate(cfg.max_seq_len);
                }

                let seq_len = full_seq.len();
                let inp = full_seq[..seq_len - 1].to_vec();
                let mut tgt = full_seq[1..seq_len].to_vec();

                // Mask prompt tokens in target (indices 0..prompt_end - 1)
                let mask_end = (prompt_end - 1).min(tgt.len());
                for i in 0..mask_end {
                    tgt[i] = PAD_TOKEN as i32;
                }

                if inp.len() > max_len {
                    max_len = inp.len();
                }
                batch_inputs.push(inp);
                batch_targets.push(tgt);
            }

            max_len = max_len.min(cfg.max_seq_len);
            let mut flat_inputs = Vec::with_capacity(bs * max_len);
            let mut flat_targets = Vec::with_capacity(bs * max_len);
            let mut supervised_count = 0usize;

            for (mut inp, mut tgt) in batch_inputs.into_iter().zip(batch_targets.into_iter()) {
                inp.truncate(max_len);
                tgt.truncate(max_len);
                while inp.len() < max_len {
                    inp.push(PAD_TOKEN as i32);
                    tgt.push(PAD_TOKEN as i32);
                }
                for &t in &tgt {
                    if t != PAD_TOKEN as i32 {
                        supervised_count += 1;
                    }
                }
                flat_inputs.extend(inp);
                flat_targets.extend(tgt);
            }

            let token_tensor = Tensor::<TrainBackend, 2, Int>::from_ints(
                TensorData::new(flat_inputs, [bs, max_len]),
                &device,
            );
            let target_tensor = Tensor::<TrainBackend, 1, Int>::from_ints(
                TensorData::new(flat_targets, [bs * max_len]),
                &device,
            );

            // Forward pass (pure autoregressive language mode)
            let (logits, aux_loss, _, _) = model.forward_with_depth(token_tensor);
            let flat_logits = logits.reshape([bs * max_len, cfg.vocab_size]);

            let ce = masked_token_ce(flat_logits, target_tensor, supervised_count);
            let loss = ce + aux_loss;

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
        let mut val_steps = 0usize;
        let mut val_tokens_correct = 0usize;
        let mut val_tokens_total = 0usize;
        let mut val_colors_correct = 0usize;
        let mut val_colors_total = 0usize;

        for chunk in val_samples.chunks(batch_size) {
            let bs = chunk.len();
            if bs == 0 {
                continue;
            }

            let mut batch_inputs = Vec::with_capacity(bs);
            let mut batch_targets = Vec::with_capacity(bs);
            let mut max_len = 0;

            for s in chunk {
                let mut full_seq = vec![BOS_TOKEN as i32];
                for &t in &s.prompt_tokens {
                    full_seq.push(t as i32);
                }
                let prompt_end = full_seq.len();
                for &t in &s.action_tokens {
                    full_seq.push(t as i32);
                }
                full_seq.push(EOS_TOKEN as i32);

                if full_seq.len() > cfg.max_seq_len {
                    full_seq.truncate(cfg.max_seq_len);
                }

                let seq_len = full_seq.len();
                let inp = full_seq[..seq_len - 1].to_vec();
                let mut tgt = full_seq[1..seq_len].to_vec();

                let mask_end = (prompt_end - 1).min(tgt.len());
                for i in 0..mask_end {
                    tgt[i] = PAD_TOKEN as i32;
                }

                if inp.len() > max_len {
                    max_len = inp.len();
                }
                batch_inputs.push(inp);
                batch_targets.push(tgt);
            }

            max_len = max_len.min(cfg.max_seq_len);
            let mut flat_inputs = Vec::with_capacity(bs * max_len);
            let mut flat_targets = Vec::with_capacity(bs * max_len);
            let mut supervised_count = 0usize;

            for (mut inp, mut tgt) in batch_inputs.into_iter().zip(batch_targets.into_iter()) {
                inp.truncate(max_len);
                tgt.truncate(max_len);
                while inp.len() < max_len {
                    inp.push(PAD_TOKEN as i32);
                    tgt.push(PAD_TOKEN as i32);
                }
                for &t in &tgt {
                    if t != PAD_TOKEN as i32 {
                        supervised_count += 1;
                    }
                }
                flat_inputs.extend(inp);
                flat_targets.extend(tgt);
            }

            let token_tensor = Tensor::<TrainBackend, 2, Int>::from_ints(
                TensorData::new(flat_inputs, [bs, max_len]),
                &device,
            );
            let target_tensor = Tensor::<TrainBackend, 1, Int>::from_ints(
                TensorData::new(flat_targets.clone(), [bs * max_len]),
                &device,
            );

            let (logits, _, _, _) = model.forward_with_depth(token_tensor);
            let flat_logits = logits.clone().reshape([bs * max_len, cfg.vocab_size]);
            let ce = masked_token_ce(flat_logits.clone(), target_tensor, supervised_count);

            let l: f32 = ce.into_data().convert::<f32>().to_vec().unwrap()[0];
            val_loss_sum += l;
            val_steps += 1;

            let preds = flat_logits.argmax(1).reshape([bs * max_len]);
            let pred_vec: Vec<i32> = preds.into_data().convert::<i32>().to_vec().unwrap();

            for (&pred, &tgt) in pred_vec.iter().zip(flat_targets.iter()) {
                if tgt != PAD_TOKEN as i32 {
                    val_tokens_total += 1;
                    if pred == tgt {
                        val_tokens_correct += 1;
                    }
                    if tgt >= COLOR_START_ID as i32 && tgt < (COLOR_START_ID + 256) as i32 {
                        val_colors_total += 1;
                        if pred == tgt {
                            val_colors_correct += 1;
                        }
                    }
                }
            }
        }

        let avg_val_loss = val_loss_sum / val_steps.max(1) as f32;
        let token_acc = if val_tokens_total > 0 {
            (val_tokens_correct as f32 / val_tokens_total as f32) * 100.0
        } else {
            0.0
        };
        let color_acc = if val_colors_total > 0 {
            (val_colors_correct as f32 / val_colors_total as f32) * 100.0
        } else {
            0.0
        };
        let elapsed = t0.elapsed();

        println!(
            "Epoch {:>2}/{} | Train Loss: {:.4} | Val Loss: {:.4} | Box Acc: {:.1}% | Color Acc: {:.1}% | Elapsed: {:.1}s",
            epoch, epochs, avg_train_loss, avg_val_loss, token_acc, color_acc, elapsed.as_secs_f32()
        );

        // Save best checkpoint
        if avg_val_loss < best_val_loss {
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
            };

            let meta_json = serde_json::to_string_pretty(&meta)?;
            std::fs::write(out_dir.join("metadata.json"), meta_json)?;

            let tok_src = Path::new("training-cache/kilo_talk_tokenizer/tokenizer.json");
            if tok_src.exists() {
                let _ = std::fs::copy(tok_src, out_dir.join("tokenizer.json"));
            }

            let recorder = burn::record::BinFileRecorder::<burn::record::FullPrecisionSettings>::new();
            let _ = model.clone().save_file(out_dir.join("model"), &recorder);
        }
    }

    println!("\nTraining complete! Best validation loss: {:.4}", best_val_loss);
    Ok(())
}
