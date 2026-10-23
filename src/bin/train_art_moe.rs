//! Multimodal MoE Training on Canvas Progression & Kilo Talk Trajectories.

#[cfg(not(target_arch = "wasm32"))]
use {
    anyhow::{Context, Result, ensure},
    burn::{
        backend::{
            autodiff::checkpoint::strategy::BalancedCheckpointing,
            Autodiff,
        },
        nn::loss::CrossEntropyLossConfig,
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
    yumon_pet::vision::paint_action::{
        CANVAS_SIZE, CANVAS_START_ID, CANVAS_END_ID, COLOR_START_ID, PALETTE_SIZE,
        hwc_to_chw_floats,
    },
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
#[command(about = "Train the Multimodal Vision MoE on Canvas Progression")]
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

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<()> {
    let args = Args::parse();
    let device: <TrainBackend as Backend>::Device = Default::default();

    println!("=== Yumon Multimodal MoE Art Trainer ===");
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
        .with_is_multimodal(true);

    let mut model: YumonMoeBrain<TrainBackend> = brain_config.init(&device);
    let mut optimizer = AdamWConfig::new()
        .with_epsilon(1e-8)
        .with_weight_decay(1e-4)
        .init();

    let ce_loss = CrossEntropyLossConfig::new()
        .with_pad_tokens(Some(vec![PAD_TOKEN]))
        .init(&device);

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

            // A. Pack Canvas Tensors [bs, 3, 64, 64]
            let mut flat_canvas = Vec::with_capacity(bs * 3 * CANVAS_SIZE * CANVAS_SIZE);
            for s in chunk {
                let chw = hwc_to_chw_floats(&s.canvas_hwc);
                flat_canvas.extend(chw);
            }
            let canvas_tensor = Tensor::<TrainBackend, 4>::from_floats(
                TensorData::new(flat_canvas, [bs, 3, CANVAS_SIZE, CANVAS_SIZE]),
                &device,
            );

            // B. Pack Tokens: [BOS, prompt..., CANVAS_START, CANVAS_END, action_tokens...]
            // We find max token length in this batch for padding
            let mut batch_token_rows = Vec::with_capacity(bs);
            let mut max_len = 0;

            for s in chunk {
                let mut row = vec![BOS_TOKEN as i32];
                for &tok in &s.prompt_tokens {
                    row.push(tok as i32);
                }
                row.push(CANVAS_START_ID as i32);
                row.push(CANVAS_END_ID as i32);
                for &tok in &s.action_tokens {
                    row.push(tok as i32);
                }
                row.push(EOS_TOKEN as i32);

                if row.len() > max_len {
                    max_len = row.len();
                }
                batch_token_rows.push(row);
            }

            // Cap at max_seq_len - 256
            let max_text = max_len.min(cfg.max_seq_len - 256);
            let mut flat_tokens = Vec::with_capacity(bs * max_text);
            let mut flat_targets = Vec::with_capacity(bs * (max_text - 1));

            for mut row in batch_token_rows {
                row.truncate(max_text);
                while row.len() < max_text {
                    row.push(PAD_TOKEN as i32);
                }
                flat_tokens.extend_from_slice(&row);

                // Loss masking: only predict next tokens that are palette colors
                for i in 0..max_text - 1 {
                    let next_tok = row[i + 1];
                    if next_tok >= COLOR_START_ID as i32
                        && next_tok < (COLOR_START_ID + PALETTE_SIZE) as i32
                    {
                        flat_targets.push(next_tok);
                    } else {
                        flat_targets.push(PAD_TOKEN as i32);
                    }
                }
            }

            let token_tensor = Tensor::<TrainBackend, 2, Int>::from_ints(
                TensorData::new(flat_tokens, [bs, max_text]),
                &device,
            );
            let target_labels = Tensor::<TrainBackend, 2, Int>::from_ints(
                TensorData::new(flat_targets, [bs, max_text - 1]),
                &device,
            );

            // C. Forward pass
            let (logits, aux_loss, _, _) = model.forward_multimodal_with_depth(
                token_tensor.clone(),
                Some(canvas_tensor),
            );

            // Logits have shape [bs, 256 + max_text, vocab_size]
            // We train on predicting target tokens (shifted by 1)
            let text_logits = logits.slice([0..bs, 256..256 + max_text - 1, 0..cfg.vocab_size]);

            // Reshape for cross entropy
            let [_, seq_m1, vocab_sz] = text_logits.dims();
            let flat_logits = text_logits.reshape([bs * seq_m1, vocab_sz]);
            let flat_targets = target_labels.reshape([bs * seq_m1]);

            let ce = ce_loss.forward(flat_logits, flat_targets);
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

        for chunk in val_samples.chunks(batch_size) {
            let bs = chunk.len();
            let mut flat_canvas = Vec::with_capacity(bs * 3 * CANVAS_SIZE * CANVAS_SIZE);
            for s in chunk {
                flat_canvas.extend(hwc_to_chw_floats(&s.canvas_hwc));
            }
            let canvas_tensor = Tensor::<TrainBackend, 4>::from_floats(
                TensorData::new(flat_canvas, [bs, 3, CANVAS_SIZE, CANVAS_SIZE]),
                &device,
            );

            let mut batch_token_rows = Vec::with_capacity(bs);
            let mut max_len = 0;
            for s in chunk {
                let mut row = vec![BOS_TOKEN as i32];
                for &tok in &s.prompt_tokens {
                    row.push(tok as i32);
                }
                row.push(CANVAS_START_ID as i32);
                row.push(CANVAS_END_ID as i32);
                for &tok in &s.action_tokens {
                    row.push(tok as i32);
                }
                row.push(EOS_TOKEN as i32);
                if row.len() > max_len {
                    max_len = row.len();
                }
                batch_token_rows.push(row);
            }
            let max_text = max_len.min(cfg.max_seq_len - 256);
            let mut flat_tokens = Vec::with_capacity(bs * max_text);
            let mut flat_targets = Vec::with_capacity(bs * (max_text - 1));

            for mut row in batch_token_rows {
                row.truncate(max_text);
                while row.len() < max_text {
                    row.push(PAD_TOKEN as i32);
                }
                flat_tokens.extend_from_slice(&row);

                for i in 0..max_text - 1 {
                    let next_tok = row[i + 1];
                    if next_tok >= COLOR_START_ID as i32
                        && next_tok < (COLOR_START_ID + PALETTE_SIZE) as i32
                    {
                        flat_targets.push(next_tok);
                    } else {
                        flat_targets.push(PAD_TOKEN as i32);
                    }
                }
            }

            let token_tensor = Tensor::<TrainBackend, 2, Int>::from_ints(
                TensorData::new(flat_tokens, [bs, max_text]),
                &device,
            );
            let target_labels = Tensor::<TrainBackend, 2, Int>::from_ints(
                TensorData::new(flat_targets, [bs, max_text - 1]),
                &device,
            );

            let (logits, _, _, _) = model.forward_multimodal_with_depth(
                token_tensor.clone(),
                Some(canvas_tensor),
            );

            let text_logits = logits.slice([0..bs, 256..256 + max_text - 1, 0..cfg.vocab_size]);
            let [_, seq_m1, vocab_sz] = text_logits.dims();
            let loss = ce_loss.forward(
                text_logits.reshape([bs * seq_m1, vocab_sz]),
                target_labels.reshape([bs * seq_m1]),
            );

            let l: f32 = loss.into_data().convert::<f32>().to_vec().unwrap()[0];
            val_loss_sum += l;
            val_steps += 1;
        }

        let avg_val_loss = val_loss_sum / val_steps.max(1) as f32;
        let elapsed = t0.elapsed();

        println!(
            "Epoch {:>2}/{} | Train Loss: {:.4} | Val Loss: {:.4} | Elapsed: {:.1}s",
            epoch, epochs, avg_train_loss, avg_val_loss, elapsed.as_secs_f32()
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
                is_multimodal: true,
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
