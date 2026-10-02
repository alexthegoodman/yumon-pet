//! MoE prediction model for DAW next-action prediction.
//!
//! Reuses the existing SparseMoe router and MoeBlock components from
//! moe_model.rs, adapted for action-sequence prediction rather than
//! natural language generation. The model takes a context window of
//! recent semantic actions and predicts a short plan of likely next
//! actions.

use super::{
    daw_actions::{ACTION_VOCAB_SIZE, BOS_ACTION, DawAction, EOS_ACTION, MAX_ACTION_PARAMS, NUM_DAW_ACTIONS, PAD_ACTION},
    model::{MLPConfig, RMSNorm, RMSNormConfig},
    moe_model::SparseMoe,
};
use anyhow::Result;
use burn::{
    backend::Wgpu,
    module::Ignored,
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig,
        RotaryEncoding, RotaryEncodingConfig,
    },
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::activation::softmax,
};
use serde::{Deserialize, Serialize};

// ── Prediction block ──────────────────────────────────────────────────────────

/// A single transformer block with causal self-attention and sparse MoE FFN,
/// structurally identical to MoeBlock in moe_model.rs but defined here to
/// keep the prediction model self-contained and avoid coupling to the
/// language model's padding/active-token logic.
#[derive(Module, Debug)]
struct PredictionBlock<B: Backend> {
    attn_norm: RMSNorm<B>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    o: Linear<B>,
    ffn_norm: RMSNorm<B>,
    moe: SparseMoe<B>,
    heads: usize,
}

impl<B: Backend> PredictionBlock<B> {
    fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEncoding<B>,
    ) -> (Tensor<B, 3>, Tensor<B, 1>, Tensor<B, 1>) {
        let [batch, seq, width] = x.dims();
        let device = x.device();
        let hd = width / self.heads;

        // Self-attention with RoPE
        let norm = self.attn_norm.forward(x.clone());
        let split =
            |t: Tensor<B, 3>| t.reshape([batch, seq, self.heads, hd]).swap_dims(1, 2);
        let q = rope.forward(split(self.q.forward(norm.clone()))) / (hd as f64).sqrt();
        let k = rope.forward(split(self.k.forward(norm.clone())));
        let v = split(self.v.forward(norm));

        // Causal mask
        let future = Tensor::<B, 2>::ones([seq, seq], &device).triu(1).bool();
        let mask = future.unsqueeze::<4>();
        let weights = softmax(
            q.matmul(k.transpose()).mask_fill(mask, -1e9),
            3,
        );
        let attn = weights
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, seq, width]);
        let x = x + self.o.forward(attn);

        // Sparse MoE FFN over all positions (no padding mask needed -
        // the data generator does not pad within a context window;
        // short sequences are padded at the end and excluded from loss).
        let flat = self.ffn_norm.forward(x.clone()).reshape([batch * seq, width]);
        let (values, balance, z_loss, _counts) = self.moe.forward(flat);
        let ffn_out = values.reshape([batch, seq, width]);

        (x + ffn_out, balance, z_loss)
    }
}

// ── Prediction model ──────────────────────────────────────────────────────────

/// Configuration for the action prediction MoE model.
#[derive(Config, Debug)]
pub struct PredictionModelConfig {
    /// Size of the action vocabulary (including sentinels).
    #[config(default = 35)] // ACTION_VOCAB_SIZE
    pub vocab_size: usize,
    /// Embedding dimension.
    #[config(default = 128)]
    pub embed_dim: usize,
    /// Number of transformer layers.
    #[config(default = 4)]
    pub n_layers: usize,
    /// Number of attention heads.
    #[config(default = 4)]
    pub attn_heads: usize,
    /// FFN hidden dimension (per expert).
    #[config(default = 256)]
    pub ff_dim: usize,
    /// Maximum context window (action history length).
    #[config(default = 64)]
    pub max_seq_len: usize,
    /// Number of MoE experts per layer.
    #[config(default = 4)]
    pub num_experts: usize,
    /// Top-k expert routing.
    #[config(default = 1)]
    pub top_k: usize,
    /// Dropout rate.
    #[config(default = 0.05)]
    pub dropout_rate: f64,
    /// Number of future actions to predict simultaneously.
    #[config(default = 5)]
    pub prediction_depth: usize,
    /// Auxiliary load-balancing loss weight.
    #[config(default = 0.01)]
    pub aux_loss_weight: f64,
    /// Router z-loss weight.
    #[config(default = 0.001)]
    pub z_loss_weight: f64,
}

impl PredictionModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PredictionModel<B> {
        assert!(self.embed_dim % self.attn_heads == 0);
        assert!((self.embed_dim / self.attn_heads) % 2 == 0, "RoPE needs even head width");

        let linear = || {
            LinearConfig::new(self.embed_dim, self.embed_dim)
                .with_bias(false)
                .init(device)
        };

        PredictionModel {
            config: Ignored(self.clone()),
            action_embed: EmbeddingConfig::new(self.vocab_size, self.embed_dim).init(device),
            param_proj: LinearConfig::new(MAX_ACTION_PARAMS, self.embed_dim)
                .with_bias(true)
                .init(device),
            rope: RotaryEncodingConfig::new(
                self.max_seq_len,
                self.embed_dim / self.attn_heads,
            )
            .init(device),
            blocks: (0..self.n_layers)
                .map(|_| PredictionBlock {
                    attn_norm: RMSNormConfig::new(self.embed_dim).init(device),
                    q: linear(),
                    k: linear(),
                    v: linear(),
                    o: linear(),
                    ffn_norm: RMSNormConfig::new(self.embed_dim).init(device),
                    moe: SparseMoe::new(
                        self.embed_dim,
                        self.ff_dim,
                        self.num_experts,
                        self.top_k,
                        device,
                    ),
                    heads: self.attn_heads,
                })
                .collect(),
            norm: RMSNormConfig::new(self.embed_dim).init(device),
            dropout: DropoutConfig::new(self.dropout_rate).init(),
            action_head: LinearConfig::new(self.embed_dim, self.vocab_size).init(device),
        }
    }
}

/// Sparse MoE transformer for predicting the next N DAW actions from a
/// context window of recent semantic action history.
///
/// Input representation: each timestep is the sum of an action embedding
/// (from the action ID) and a parameter projection (the action's float
/// parameters projected into embed_dim). This avoids needing a separate
/// tokenizer - the action vocabulary is small and fixed.
#[derive(Module, Debug)]
pub struct PredictionModel<B: Backend> {
    pub config: Ignored<PredictionModelConfig>,
    action_embed: Embedding<B>,
    param_proj: Linear<B>,
    rope: RotaryEncoding<B>,
    blocks: Vec<PredictionBlock<B>>,
    norm: RMSNorm<B>,
    dropout: Dropout,
    pub action_head: Linear<B>,
}

impl<B: Backend> PredictionModel<B> {
    /// Forward pass. Returns (logits, auxiliary_loss).
    ///
    /// - `action_ids`: [batch, seq] integer action IDs
    /// - `action_params`: [batch, seq, MAX_ACTION_PARAMS] float parameters
    ///
    /// Returns logits of shape [batch, seq, vocab_size] and a scalar
    /// auxiliary loss for load balancing.
    pub fn forward(
        &self,
        action_ids: Tensor<B, 2, Int>,
        action_params: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let device = action_ids.device();

        // Embed action IDs and project parameters, then sum
        let act_emb = self.action_embed.forward(action_ids);
        let param_emb = self.param_proj.forward(action_params);
        let mut x = self.dropout.forward(act_emb + param_emb);

        let mut auxiliary = Tensor::zeros([1], &device);
        for block in &self.blocks {
            let (next, balance, z_loss) = block.forward(x, &self.rope);
            x = next;
            auxiliary = auxiliary
                + balance * self.config.aux_loss_weight
                + z_loss * self.config.z_loss_weight;
        }

        let logits = self.action_head.forward(self.norm.forward(x));
        (logits, auxiliary / self.blocks.len() as f64)
    }

    /// Greedy multi-step prediction from a context window.
    ///
    /// Given a partial action history, autoregressively predict the next
    /// `steps` actions. Returns a vec of predicted action IDs.
    pub fn predict(
        &self,
        context_ids: &[u32],
        context_params: &[[f32; MAX_ACTION_PARAMS]],
        steps: usize,
        device: &B::Device,
    ) -> Vec<u32> {
        let mut ids: Vec<i32> = context_ids.iter().map(|&id| id as i32).collect();
        let mut params: Vec<[f32; MAX_ACTION_PARAMS]> = context_params.to_vec();
        let mut predictions = Vec::with_capacity(steps);

        for _ in 0..steps {
            let current_len = ids.len().min(self.config.max_seq_len);
            let start = ids.len().saturating_sub(self.config.max_seq_len);
            let window_ids = &ids[start..];
            let window_params = &params[start..];

            let ids_tensor = Tensor::<B, 2, Int>::from_ints(
                burn::tensor::TensorData::new(
                    window_ids.to_vec(),
                    [1, current_len],
                ),
                device,
            );

            let params_flat: Vec<f32> = window_params.iter().flat_map(|p| p.iter().copied()).collect();
            let params_tensor = Tensor::<B, 3>::from_data(
                burn::tensor::TensorData::new(
                    params_flat,
                    [1, current_len, MAX_ACTION_PARAMS],
                ),
                device,
            );

            let (logits, _aux) = self.forward(ids_tensor, params_tensor);
            let last_logits = logits
                .slice([0..1, current_len - 1..current_len, 0..self.config.vocab_size])
                .reshape([self.config.vocab_size]);

            let logits_vec: Vec<f32> = last_logits.to_data().to_vec().unwrap();
            let next = logits_vec
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(PAD_ACTION);

            if next == EOS_ACTION || next == PAD_ACTION {
                break;
            }

            predictions.push(next);
            ids.push(next as i32);
            params.push([0.0; MAX_ACTION_PARAMS]); // predicted actions have no params yet
        }

        predictions
    }

    // ── Checkpoint I/O ────────────────────────────────────────────────────

    pub fn save(&self, directory: &str, metadata: &PredictionMetadata) -> Result<()> {
        let dir = std::path::Path::new(directory);
        std::fs::create_dir_all(dir)?;

        let meta_json = serde_json::to_string_pretty(metadata)?;
        std::fs::write(dir.join("metadata.json"), meta_json)?;

        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        self.clone()
            .save_file(dir.join("model"), &recorder)
            .map_err(|e| anyhow::anyhow!("save_file: {e:?}"))?;

        Ok(())
    }

    pub fn load(directory: &str, device: &B::Device) -> Result<(Self, PredictionModelConfig)> {
        let dir = std::path::Path::new(directory);

        let meta_json = std::fs::read_to_string(dir.join("metadata.json"))?;
        let metadata: PredictionMetadata = serde_json::from_str(&meta_json)?;

        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let record = recorder
            .load(dir.join("model").into(), device)
            .map_err(|e| anyhow::anyhow!("load: {e:?}"))?;

        let config = PredictionModelConfig {
            vocab_size: metadata.vocab_size,
            embed_dim: metadata.embed_dim,
            n_layers: metadata.n_layers,
            attn_heads: metadata.attn_heads,
            ff_dim: metadata.ff_dim,
            max_seq_len: metadata.max_seq_len,
            num_experts: metadata.num_experts,
            top_k: metadata.top_k,
            dropout_rate: metadata.dropout_rate,
            prediction_depth: metadata.prediction_depth,
            aux_loss_weight: metadata.aux_loss_weight,
            z_loss_weight: metadata.z_loss_weight,
        };

        let model = config.init::<B>(device).load_record(record);
        Ok((model, config))
    }
}

// ── Metadata ──────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct PredictionMetadata {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub n_layers: usize,
    pub attn_heads: usize,
    pub ff_dim: usize,
    pub max_seq_len: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub dropout_rate: f64,
    pub prediction_depth: usize,
    pub aux_loss_weight: f64,
    pub z_loss_weight: f64,
    pub epochs_trained: usize,
    pub final_loss: f32,
    pub batch_size: usize,
    pub num_sequences: usize,
}

// ── High-Level Inference API ──────────────────────────────────────────────────

/// A predicted action with metadata, suggested defaults, and confidence score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedAction {
    pub action_id: u32,
    pub name: String,
    pub display_name: String,
    pub category: String,
    pub icon: String,
    pub params: Vec<f32>,
    pub confidence: f32,
}

/// Action predictor instance holding the loaded model on WGPU device.
pub struct ActionPredictor {
    model: PredictionModel<Wgpu>,
    pub config: PredictionModelConfig,
    device: burn::backend::wgpu::WgpuDevice,
}

impl ActionPredictor {
    pub fn new(directory: &str) -> Result<Self> {
        let device = burn::backend::wgpu::WgpuDevice::default();
        let (model, config) = PredictionModel::<Wgpu>::load(directory, &device)?;
        Ok(Self { model, config, device })
    }

    pub fn predict_actions(
        &self,
        context_ids: &[u32],
        context_params: Option<&[[f32; MAX_ACTION_PARAMS]]>,
        steps: usize,
    ) -> Vec<PredictedAction> {
        let mut ids: Vec<u32> = if context_ids.is_empty() {
            vec![BOS_ACTION]
        } else {
            let mut v = Vec::with_capacity(context_ids.len() + 1);
            if context_ids[0] != BOS_ACTION {
                v.push(BOS_ACTION);
            }
            v.extend_from_slice(context_ids);
            v
        };

        let mut params: Vec<[f32; MAX_ACTION_PARAMS]> = Vec::new();
        if let Some(cp) = context_params {
            if context_ids.is_empty() || context_ids[0] != BOS_ACTION {
                params.push([0.0; MAX_ACTION_PARAMS]);
            }
            params.extend_from_slice(cp);
        }
        while params.len() < ids.len() {
            params.push([0.0; MAX_ACTION_PARAMS]);
        }

        let mut predicted_actions = Vec::with_capacity(steps);
        let max_seq = self.config.max_seq_len;

        for _ in 0..steps {
            let cur_len = ids.len().min(max_seq);
            let start = ids.len().saturating_sub(max_seq);
            let win_ids = &ids[start..];
            let win_params = &params[start..];

            let ids_tensor = Tensor::<Wgpu, 2, Int>::from_ints(
                burn::tensor::TensorData::new(
                    win_ids.iter().map(|&x| x as i32).collect::<Vec<_>>(),
                    [1, cur_len],
                ),
                &self.device,
            );

            let params_flat: Vec<f32> = win_params.iter().flat_map(|p| p.iter().copied()).collect();
            let params_tensor = Tensor::<Wgpu, 3>::from_data(
                burn::tensor::TensorData::new(
                    params_flat,
                    [1, cur_len, MAX_ACTION_PARAMS],
                ),
                &self.device,
            );

            let (logits, _aux) = self.model.forward(ids_tensor, params_tensor);
            let last_logits = logits
                .slice([0..1, cur_len - 1..cur_len, 0..self.config.vocab_size])
                .reshape([self.config.vocab_size]);

            let logits_vec: Vec<f32> = last_logits.to_data().to_vec().unwrap();

            let max_logit = logits_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits_vec.iter().map(|&x| (x - max_logit).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();

            let mut best_id = PAD_ACTION;
            let mut best_prob = 0.0f32;

            for (i, &p) in exps.iter().enumerate() {
                let prob = if sum_exp > 0.0 { p / sum_exp } else { 0.0 };
                if (i as u32) < NUM_DAW_ACTIONS as u32 {
                    if prob > best_prob {
                        best_prob = prob;
                        best_id = i as u32;
                    }
                }
            }

            if best_id == PAD_ACTION || best_id == EOS_ACTION {
                break;
            }

            if let Some(action) = DawAction::from_id(best_id) {
                predicted_actions.push(PredictedAction {
                    action_id: best_id,
                    name: action.name().to_string(),
                    display_name: action.display_name().to_string(),
                    category: action.category().to_string(),
                    icon: action.icon().to_string(),
                    params: action.default_params().to_vec(),
                    confidence: (best_prob * 100.0).round() / 100.0,
                });

                ids.push(best_id);
                params.push(action.default_params());
            } else {
                break;
            }
        }

        predicted_actions
    }
}

static PREDICTOR: std::sync::Mutex<Option<ActionPredictor>> = std::sync::Mutex::new(None);

/// Resolve the prediction checkpoint directory by checking candidate locations.
pub fn resolve_prediction_checkpoint_dir() -> Result<std::path::PathBuf> {
    let candidates = [
        "checkpoints/prediction",
        "../yumon-pet/checkpoints/prediction",
        "yumon-pet/checkpoints/prediction",
        "../../yumon-pet/checkpoints/prediction",
        "D:/projects/common/yumon-pet/checkpoints/prediction",
    ];

    for candidate in &candidates {
        let p = std::path::Path::new(candidate);
        if p.join("metadata.json").exists() && (p.join("model.bin").exists() || p.join("model").exists()) {
            return Ok(p.to_path_buf());
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let p = parent.join("checkpoints/prediction");
            if p.join("metadata.json").exists() {
                return Ok(p);
            }
        }
    }

    anyhow::bail!("Could not locate prediction checkpoint directory")
}

/// High-level function to predict next DAW actions from action history.
pub fn predict_next_actions(
    checkpoint_dir: Option<&str>,
    context_ids: &[u32],
    context_params: Option<&[[f32; MAX_ACTION_PARAMS]]>,
    steps: usize,
) -> Result<Vec<PredictedAction>> {
    let mut guard = PREDICTOR.lock().map_err(|e| anyhow::anyhow!("prediction mutex poisoned: {e}"))?;
    if guard.is_none() {
        let dir = if let Some(d) = checkpoint_dir {
            std::path::PathBuf::from(d)
        } else {
            resolve_prediction_checkpoint_dir()?
        };
        let predictor = ActionPredictor::new(dir.to_str().unwrap())?;
        *guard = Some(predictor);
    }

    let predictor = guard.as_ref().unwrap();
    Ok(predictor.predict_actions(context_ids, context_params, steps))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, Wgpu};
    use burn::optim::{AdamWConfig, GradientsParams, Optimizer};

    type B = Autodiff<Wgpu>;

    fn tiny_config() -> PredictionModelConfig {
        PredictionModelConfig::new()
            .with_vocab_size(ACTION_VOCAB_SIZE)
            .with_embed_dim(16)
            .with_n_layers(2)
            .with_attn_heads(2)
            .with_ff_dim(32)
            .with_max_seq_len(16)
            .with_num_experts(4)
            .with_top_k(1)
            .with_dropout_rate(0.0)
            .with_prediction_depth(3)
    }

    #[test]
    fn prediction_model_forward_shapes() {
        let device = Default::default();
        let model: PredictionModel<B> = tiny_config().init(&device);

        let ids = Tensor::from_ints([[1, 4, 5, 6, 0, 0]], &device);
        let params = Tensor::<B, 3>::zeros([1, 6, MAX_ACTION_PARAMS], &device);

        let (logits, aux) = model.forward(ids, params);
        assert_eq!(logits.dims(), [1, 6, ACTION_VOCAB_SIZE]);
        assert!(aux.to_data().to_vec::<f32>().unwrap()[0].is_finite());
    }

    #[test]
    fn prediction_model_optimizer_step() {
        let device = Default::default();
        let model: PredictionModel<B> = tiny_config().init(&device);

        let ids = Tensor::from_ints([[1, 4, 5, 6]], &device);
        let params = Tensor::<B, 3>::zeros([1, 4, MAX_ACTION_PARAMS], &device);

        let (logits, aux) = model.forward(ids, params);
        let loss = logits.powf_scalar(2.0).mean() + aux;
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let mut optimizer = AdamWConfig::new().init();
        let _updated = optimizer.step(1e-3, model, grads);
    }

    #[test]
    fn prediction_model_greedy_predict() {
        let device = Default::default();
        let model: PredictionModel<Wgpu> = tiny_config().init(&device);

        let context_ids = [BOS_ACTION, 0, 4, 8]; // BOS, AddTrack, Play, AddNote
        let context_params = [[0.0; MAX_ACTION_PARAMS]; 4];

        let preds = model.predict(&context_ids, &context_params, 3, &device);
        // Should return up to 3 predictions (may be fewer if EOS/PAD hit)
        assert!(preds.len() <= 3);
        for &p in &preds {
            assert!(p < ACTION_VOCAB_SIZE as u32);
        }
    }

    #[test]
    fn prediction_model_checkpoint_roundtrip() {
        let device = Default::default();
        let config = tiny_config();
        let model: PredictionModel<Wgpu> = config.init(&device);

        let path = std::env::temp_dir().join(format!("yumon-pred-{}", uuid::Uuid::new_v4()));
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
            epochs_trained: 1,
            final_loss: 1.0,
            batch_size: 8,
            num_sequences: 100,
        };
        model.save(path.to_str().unwrap(), &meta).unwrap();

        let (restored, restored_config) =
            PredictionModel::<Wgpu>::load(path.to_str().unwrap(), &device).unwrap();
        assert_eq!(restored_config.num_experts, config.num_experts);
        assert_eq!(restored_config.prediction_depth, config.prediction_depth);

        // Verify weights match
        let ids = Tensor::from_ints([[1, 4, 5]], &device);
        let params = Tensor::<Wgpu, 3>::zeros([1, 3, MAX_ACTION_PARAMS], &device);
        let (orig_logits, _) = model.forward(ids.clone(), params.clone());
        let (rest_logits, _) = restored.forward(ids, params);

        let orig: Vec<f32> = orig_logits.to_data().to_vec().unwrap();
        let rest: Vec<f32> = rest_logits.to_data().to_vec().unwrap();
        for (a, b) in orig.iter().zip(rest.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }

        assert_eq!(path.parent(), Some(std::env::temp_dir().as_path()));
        std::fs::remove_dir_all(path).unwrap();
    }
}
