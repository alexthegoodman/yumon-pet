//! Multimodal MoE Training on Canvas Progression & Kilo Talk Trajectories.

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
        tokenizer::{BOS_TOKEN, PAD_TOKEN},
    },
    yumon_pet::vision::paint_action::{
        CANVAS_SIZE, CANVAS_START_ID,
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
    #[serde(default)]
    pub target_edge: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct LegacyArtSampleV1 {
    pub prompt_text: String,
    pub prompt_tokens: Vec<u32>,
    pub canvas_hwc: Vec<u8>,
    pub action_tokens: Vec<u32>,
    pub is_done: bool,
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
    let meta: ArtCacheMetadata = bincode::deserialize(&meta_bytes)?;

    let mut samples = Vec::new();
    let mut len_buf = [0u8; 4];

    if meta.version >= 2 {
        while reader.read_exact(&mut len_buf).is_ok() {
            let sample_len = u32::from_le_bytes(len_buf) as usize;
            let mut sample_bytes = vec![0u8; sample_len];
            reader.read_exact(&mut sample_bytes)?;
            let sample: ArtSample = bincode::deserialize(&sample_bytes)?;
            samples.push(sample);
        }
    } else {
        while reader.read_exact(&mut len_buf).is_ok() {
            let sample_len = u32::from_le_bytes(len_buf) as usize;
            let mut sample_bytes = vec![0u8; sample_len];
            reader.read_exact(&mut sample_bytes)?;
            let v1: LegacyArtSampleV1 = bincode::deserialize(&sample_bytes)?;
            samples.push(ArtSample {
                prompt_text: v1.prompt_text,
                prompt_tokens: v1.prompt_tokens,
                canvas_hwc: v1.canvas_hwc,
                action_tokens: v1.action_tokens,
                is_done: v1.is_done,
                target_edge: 0.0,
            });
        }
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

            // B. Pack Tokens & Target Edges
            let mut batch_token_rows = Vec::with_capacity(bs);
            let mut targets = Vec::with_capacity(bs);
            let mut batch_edges = Vec::with_capacity(bs);
            let mut edge_weights_vec = Vec::with_capacity(bs);
            let mut max_len = 0;

            for s in chunk {
                let mut row = vec![BOS_TOKEN as i32];
                for &tok in &s.prompt_tokens {
                    row.push(tok as i32);
                }
                row.push(CANVAS_START_ID as i32);
                row.push(s.action_tokens[0] as i32); // PASS_TAG
                row.push(s.action_tokens[1] as i32); // ROW_y
                row.push(s.action_tokens[2] as i32); // COL_x
                targets.push(s.action_tokens[3] as i32); // COLOR_c
                batch_edges.push(s.target_edge);
                edge_weights_vec.push(1.0 + 2.0 * s.target_edge);

                if row.len() > max_len {
                    max_len = row.len();
                }
                batch_token_rows.push(row);
            }

            // Cap at max_seq_len - 256
            let max_text = max_len.min(cfg.max_seq_len - 256);
            let mut flat_tokens = Vec::with_capacity(bs * max_text);

            for mut row in batch_token_rows {
                row.truncate(max_text);
                while row.len() < max_text {
                    row.push(PAD_TOKEN as i32);
                }
                flat_tokens.extend_from_slice(&row);
            }

            let token_tensor = Tensor::<TrainBackend, 2, Int>::from_ints(
                TensorData::new(flat_tokens, [bs, max_text]),
                &device,
            );
            let target_labels = Tensor::<TrainBackend, 1, Int>::from_ints(
                TensorData::new(targets, [bs]),
                &device,
            );
            let target_edges = Tensor::<TrainBackend, 2>::from_floats(
                TensorData::new(batch_edges, [bs, 1]),
                &device,
            );
            let edge_weights = Tensor::<TrainBackend, 1>::from_floats(
                TensorData::new(edge_weights_vec, [bs]),
                &device,
            );

            // C. Forward pass with Conv2D edge detection head
            let (logits, edge_preds, aux_loss, _, _) = model.forward_multimodal_with_edges(
                token_tensor,
                Some(canvas_tensor),
            );

            // Extract the active logit vector and edge prediction at the (x, y) query coordinate for each sample
            let mut logit_slices = Vec::with_capacity(bs);
            let mut edge_slices = Vec::with_capacity(bs);
            for (b, s) in chunk.iter().enumerate() {
                let pos = 256 + 4 + s.prompt_tokens.len();
                logit_slices.push(logits.clone().slice([b..b + 1, pos..pos + 1, 0..cfg.vocab_size]));
                edge_slices.push(edge_preds.clone().slice([b..b + 1, pos..pos + 1, 0..1]));
            }
            let active_logits = Tensor::cat(logit_slices, 0).reshape([bs, cfg.vocab_size]);
            let active_edges = Tensor::cat(edge_slices, 0).reshape([bs, 1]);

            // 1. Target Conv2D Edge-weighted Cross Entropy Loss
            let log_probs = burn::tensor::activation::log_softmax(active_logits, 1);
            let nll = log_probs.gather(1, target_labels.reshape([bs, 1])).reshape([bs]).neg();
            let weighted_ce = (nll * edge_weights).mean();

            // 2. Auxiliary Conv2D Edge Detection Loss (MSE against target image edges)
            let edge_mse = (active_edges - target_edges).powf_scalar(2.0).mean();

            let loss = weighted_ce + edge_mse * 0.5 + aux_loss;

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
        let mut val_edge_sum = 0.0f32;
        let mut val_correct = 0usize;
        let mut val_total = 0usize;
        let mut val_steps = 0usize;

        // NOTE: disable validation check to avoid Our of Memory error at end of epoch
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
            let mut targets = Vec::with_capacity(bs);
            let mut batch_edges = Vec::with_capacity(bs);
            let mut edge_weights_vec = Vec::with_capacity(bs);
            let mut max_len = 0;
            for s in chunk {
                let mut row = vec![BOS_TOKEN as i32];
                for &tok in &s.prompt_tokens {
                    row.push(tok as i32);
                }
                row.push(CANVAS_START_ID as i32);
                row.push(s.action_tokens[0] as i32); // PASS_TAG
                row.push(s.action_tokens[1] as i32); // ROW_y
                row.push(s.action_tokens[2] as i32); // COL_x
                targets.push(s.action_tokens[3] as i32); // COLOR_c
                batch_edges.push(s.target_edge);
                edge_weights_vec.push(1.0 + 2.0 * s.target_edge);

                if row.len() > max_len {
                    max_len = row.len();
                }
                batch_token_rows.push(row);
            }
            let max_text = max_len.min(cfg.max_seq_len - 256);
            let mut flat_tokens = Vec::with_capacity(bs * max_text);

            for mut row in batch_token_rows {
                row.truncate(max_text);
                while row.len() < max_text {
                    row.push(PAD_TOKEN as i32);
                }
                flat_tokens.extend_from_slice(&row);
            }

            let token_tensor = Tensor::<TrainBackend, 2, Int>::from_ints(
                TensorData::new(flat_tokens, [bs, max_text]),
                &device,
            );
            let target_labels = Tensor::<TrainBackend, 1, Int>::from_ints(
                TensorData::new(targets.clone(), [bs]),
                &device,
            );
            let target_edges = Tensor::<TrainBackend, 2>::from_floats(
                TensorData::new(batch_edges, [bs, 1]),
                &device,
            );
            let edge_weights = Tensor::<TrainBackend, 1>::from_floats(
                TensorData::new(edge_weights_vec, [bs]),
                &device,
            );

            let (logits, edge_preds, _, _, _) = model.forward_multimodal_with_edges(
                token_tensor,
                Some(canvas_tensor),
            );

            let mut logit_slices = Vec::with_capacity(bs);
            let mut edge_slices = Vec::with_capacity(bs);
            for (b, s) in chunk.iter().enumerate() {
                let pos = 256 + 4 + s.prompt_tokens.len();
                logit_slices.push(logits.clone().slice([b..b + 1, pos..pos + 1, 0..cfg.vocab_size]));
                edge_slices.push(edge_preds.clone().slice([b..b + 1, pos..pos + 1, 0..1]));
            }
            let active_logits = Tensor::cat(logit_slices, 0).reshape([bs, cfg.vocab_size]);
            let active_edges = Tensor::cat(edge_slices, 0).reshape([bs, 1]);

            let log_probs = burn::tensor::activation::log_softmax(active_logits.clone(), 1);
            let nll = log_probs.gather(1, target_labels.reshape([bs, 1])).reshape([bs]).neg();
            let loss = (nll * edge_weights).mean();

            let edge_mse = (active_edges - target_edges).powf_scalar(2.0).mean();
            let e_mse_val: f32 = edge_mse.into_data().convert::<f32>().to_vec().unwrap()[0];
            val_edge_sum += e_mse_val;

            let preds = active_logits.argmax(1).reshape([bs]);
            let pred_vec: Vec<i32> = preds.into_data().convert::<i32>().to_vec().unwrap();
            for (pred, target) in pred_vec.iter().zip(targets.iter()) {
                if *pred == *target {
                    val_correct += 1;
                }
                val_total += 1;
            }

            let l: f32 = loss.into_data().convert::<f32>().to_vec().unwrap()[0];
            val_loss_sum += l;
            val_steps += 1;
        }

        let avg_val_loss = val_loss_sum / val_steps.max(1) as f32;
        let avg_val_edge_mse = val_edge_sum / val_steps.max(1) as f32;
        let val_acc = if val_total > 0 {
            (val_correct as f32 / val_total as f32) * 100.0
        } else {
            0.0
        };
        let elapsed = t0.elapsed();

        println!(
            "Epoch {:>2}/{} | Train Loss: {:.4} | Val Loss: {:.4} (Val Color Acc: {:.1}% | Edge MSE: {:.4}) | Elapsed: {:.1}s",
            epoch, epochs, avg_train_loss, avg_val_loss, val_acc, avg_val_edge_mse, elapsed.as_secs_f32()
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
                has_edge_head: true,
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
