#![recursion_limit = "256"]
#![allow(warnings)]

//! Training binary for the DAW action prediction model.
//!
//! Loads trajectories produced by gen_daw_data, packs them into
//! fixed-length context/target windows, and trains the MoE prediction
//! model with teacher-forced next-action prediction loss.
//!
//! Usage:
//!   cargo run --release --bin train_prediction -- \
//!       --data data/daw_sequences.bin \
//!       --out checkpoints/prediction \
//!       --epochs 20 \
//!       --batch-size 32

use anyhow::Result;
use burn::{
    backend::Wgpu,
    module::AutodiffModule,
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::{Int, TensorData},
};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

use yumon_pet::brain::{
    daw_actions::*,
    prediction_model::{PredictionMetadata, PredictionModel, PredictionModelConfig},
};

type B = burn::backend::Autodiff<Wgpu>;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "train_prediction", about = "Train the DAW action prediction model")]
struct Cli {
    /// Path to the trajectory data file (from gen_daw_data).
    #[arg(long, default_value = "data/daw_sequences.bin")]
    data: String,

    /// Output checkpoint directory.
    #[arg(long, default_value = "checkpoints/prediction")]
    out: String,

    /// Number of training epochs.
    #[arg(long, default_value_t = 20)]
    epochs: usize,

    /// Batch size.
    #[arg(long, default_value_t = 32)]
    batch_size: usize,

    /// Context window size (action history the model sees).
    #[arg(long, default_value_t = 8)]
    context_len: usize,

    /// Model embedding dimension.
    #[arg(long, default_value_t = 128)]
    embed_dim: usize,

    /// Number of transformer layers.
    #[arg(long, default_value_t = 4)]
    n_layers: usize,

    /// Number of attention heads.
    #[arg(long, default_value_t = 4)]
    attn_heads: usize,

    /// FFN hidden dimension (per expert).
    #[arg(long, default_value_t = 256)]
    ff_dim: usize,

    /// Number of MoE experts per layer.
    #[arg(long, default_value_t = 4)]
    num_experts: usize,

    /// Top-k expert routing.
    #[arg(long, default_value_t = 1)]
    top_k: usize,

    /// Learning rate start.
    #[arg(long, default_value_t = 3e-4)]
    lr_start: f64,

    /// Learning rate end.
    #[arg(long, default_value_t = 3e-5)]
    lr_end: f64,

    /// RNG seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

// ── Trajectory type (must match gen_daw_data) ─────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Trajectory {
    task_id: u32,
    steps: Vec<ActionStep>,
}

// ── Training sample ───────────────────────────────────────────────────────────

struct TrainSample {
    /// Action IDs for the input context [context_len]
    input_ids: Vec<i32>,
    /// Action parameters [context_len * MAX_ACTION_PARAMS]
    input_params: Vec<f32>,
    /// Target action IDs (shifted by 1) [context_len]
    target_ids: Vec<i32>,
}

/// Extract sliding-window training samples from trajectories.
fn prepare_samples(trajectories: &[Trajectory], context_len: usize) -> Vec<TrainSample> {
    let mut samples = Vec::new();

    for traj in trajectories {
        if traj.steps.len() < 2 { continue; }

        // Sliding window over the trajectory
        let max_start = traj.steps.len().saturating_sub(2);
        for start in 0..=max_start {
            let end = (start + context_len + 1).min(traj.steps.len());
            let window = &traj.steps[start..end];
            if window.len() < 2 { continue; }

            let input_len = window.len() - 1;
            let mut input_ids = Vec::with_capacity(context_len);
            let mut input_params = Vec::with_capacity(context_len * MAX_ACTION_PARAMS);
            let mut target_ids = Vec::with_capacity(context_len);

            for i in 0..input_len {
                input_ids.push(window[i].action_id as i32);
                input_params.extend_from_slice(&window[i].params);
                target_ids.push(window[i + 1].action_id as i32);
            }

            // Pad to context_len
            while input_ids.len() < context_len {
                input_ids.push(PAD_ACTION as i32);
                input_params.extend_from_slice(&[0.0; MAX_ACTION_PARAMS]);
                target_ids.push(PAD_ACTION as i32);
            }

            samples.push(TrainSample {
                input_ids,
                input_params,
                target_ids,
            });
        }
    }

    samples
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();
    let device = burn::backend::wgpu::WgpuDevice::default();

    // Load trajectories
    println!("Loading trajectories from {}", cli.data);
    let data = std::fs::read(&cli.data)?;
    let trajectories: Vec<Trajectory> = serde_json::from_slice(&data)?;
    println!("Loaded {} trajectories", trajectories.len());

    // Prepare training samples
    println!("Preparing training samples (context_len={})...", cli.context_len);
    let samples = prepare_samples(&trajectories, cli.context_len);
    println!("Created {} training samples", samples.len());

    if samples.is_empty() {
        anyhow::bail!("No training samples - check your data file");
    }

    // Build model
    let config = PredictionModelConfig::new()
        .with_vocab_size(ACTION_VOCAB_SIZE)
        .with_embed_dim(cli.embed_dim)
        .with_n_layers(cli.n_layers)
        .with_attn_heads(cli.attn_heads)
        .with_ff_dim(cli.ff_dim)
        .with_max_seq_len(cli.context_len)
        .with_num_experts(cli.num_experts)
        .with_top_k(cli.top_k)
        .with_dropout_rate(0.05)
        .with_prediction_depth(5);

    println!("\nModel config: {:?}", config);

    let mut model: PredictionModel<B> = config.init(&device);
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.01)
        .init();

    let ce_loss = CrossEntropyLossConfig::new()
        .with_pad_tokens(Some(vec![PAD_ACTION as usize]))
        .init(&device);

    let mut rng = StdRng::seed_from_u64(cli.seed);
    let num_batches = samples.len() / cli.batch_size;
    let mut best_loss = f32::INFINITY;

    println!("\nTraining for {} epochs, {} batches/epoch, batch_size={}",
        cli.epochs, num_batches, cli.batch_size);

    for epoch in 0..cli.epochs {
        let mut indices: Vec<usize> = (0..samples.len()).collect();
        indices.shuffle(&mut rng);

        let mut epoch_loss = 0.0f32;
        let progress = ProgressBar::new(num_batches as u64);
        progress.set_style(
            ProgressStyle::default_bar()
                .template(&format!(
                    "Epoch {}/{} [{{elapsed_precise}}] {{bar:40.cyan/blue}} {{pos}}/{{len}} loss={{msg}}",
                    epoch + 1, cli.epochs
                ))
                .unwrap(),
        );

        for batch_num in 0..num_batches {
            // Linear LR schedule
            let total_steps = cli.epochs * num_batches;
            let step = epoch * num_batches + batch_num;
            let t = step as f64 / total_steps as f64;
            let lr = cli.lr_start * (1.0 - t) + cli.lr_end * t;

            let batch_start = batch_num * cli.batch_size;
            let batch_end = (batch_start + cli.batch_size).min(samples.len());
            let batch_indices = &indices[batch_start..batch_end];
            let current_batch_size = batch_indices.len();
            if current_batch_size == 0 { continue; }

            // Build batch tensors
            let mut all_ids: Vec<i32> = Vec::with_capacity(current_batch_size * cli.context_len);
            let mut all_params: Vec<f32> = Vec::with_capacity(current_batch_size * cli.context_len * MAX_ACTION_PARAMS);
            let mut all_targets: Vec<i32> = Vec::with_capacity(current_batch_size * cli.context_len);

            for &i in batch_indices {
                let s = &samples[i];
                all_ids.extend_from_slice(&s.input_ids);
                all_params.extend_from_slice(&s.input_params);
                all_targets.extend_from_slice(&s.target_ids);
            }

            let ids_tensor = Tensor::<B, 2, Int>::from_ints(
                TensorData::new(all_ids, [current_batch_size, cli.context_len]),
                &device,
            );
            let params_tensor = Tensor::<B, 3>::from_data(
                TensorData::new(all_params, [current_batch_size, cli.context_len, MAX_ACTION_PARAMS]),
                &device,
            );
            let target_tensor = Tensor::<B, 1, Int>::from_ints(
                TensorData::new(all_targets, [current_batch_size * cli.context_len]),
                &device,
            );

            let (logits, aux) = model.forward(ids_tensor, params_tensor);
            let logits_2d = logits.reshape([current_batch_size * cli.context_len, ACTION_VOCAB_SIZE]);
            let lang_loss = ce_loss.forward(logits_2d, target_tensor);
            let loss = lang_loss + aux;

            let loss_val: f32 = loss.clone().inner().to_data().to_vec::<f32>().unwrap()[0];
            epoch_loss += loss_val;

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(lr, model, grads);

            let avg = epoch_loss / (batch_num + 1) as f32;
            progress.set_message(format!("{:.4}", avg));
            progress.inc(1);
        }

        let avg_loss = epoch_loss / num_batches.max(1) as f32;
        progress.finish_with_message(format!("{:.4}", avg_loss));

        println!("  Epoch {} avg loss: {:.4}", epoch + 1, avg_loss);

        // Save checkpoint if best
        if avg_loss < best_loss {
            best_loss = avg_loss;
            let meta = PredictionMetadata {
                vocab_size: config.vocab_size,
                embed_dim: config.embed_dim,
                n_layers: config.n_layers,
                attn_heads: config.attn_heads,
                ff_dim: config.ff_dim,
                max_seq_len: config.max_seq_len,
                num_experts: config.num_experts,
                top_k: config.top_k,
                dropout_rate: config.dropout_rate,
                prediction_depth: config.prediction_depth,
                aux_loss_weight: config.aux_loss_weight,
                z_loss_weight: config.z_loss_weight,
                epochs_trained: epoch + 1,
                final_loss: avg_loss,
                batch_size: cli.batch_size,
                num_sequences: trajectories.len(),
            };

            let save_model = model.valid();
            save_model.save(&cli.out, &meta)?;
            println!("  Saved best checkpoint (loss={:.4})", avg_loss);
        }

        // Quick inference check
        if (epoch + 1) % 5 == 0 || epoch + 1 == cli.epochs {
            let eval_model = model.valid();
            let context = [BOS_ACTION, DawAction::SetBpm.id(), DawAction::AddTrack.id(), DawAction::SelectTrack.id()];
            let context_params = [[0.0f32; MAX_ACTION_PARAMS]; 4];
            let preds = eval_model.predict(&context, &context_params, 5, &device);
            let pred_names: Vec<&str> = preds.iter()
                .filter_map(|&id| DawAction::from_id(id).map(|a| a.name()))
                .collect();
            println!("  Eval: SetBpm->AddTrack->SelectTrack => {:?}", pred_names);
        }
    }

    println!("\nTraining complete. Best loss: {:.4}", best_loss);
    println!("Checkpoint saved to: {}", cli.out);

    Ok(())
}
