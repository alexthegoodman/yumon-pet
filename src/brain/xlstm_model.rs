// ═════════════════════════════════════════════════════════════════════════════
// YumonXLstmBrain — an xLSTM (Beck et al., 2024) variant of the brain.
//
// Decoder-only, like the (currently dead) YumonDecBrain in decoder_model.rs:
// prompt + separator + reply packed into one sequence, loss only on the
// separator+reply tokens. No RoPE, no attention — position is carried purely
// by the recurrent state, which is xLSTM's whole premise.
//
// This implements the mLSTM ("matrix LSTM") block only, not sLSTM. The paper
// stacks both; production-scale xLSTM models (e.g. the NX-AI 7B release) lean
// almost entirely on mLSTM because it parallelizes across time via a chunked
// scan, while sLSTM's per-timestep scalar memory-mixing does not. We don't
// even get that parallel win here — this is a plain sequential scan over
// `seq_len` in Rust, one small tensor op per timestep — but the recurrence
// itself (exponential input/forget gating, a per-head [head_dim, head_dim]
// matrix memory, a stabilizer, and a normalizer) is the real thing, not a
// renamed GRU.
//
// Deliberately NOT implemented (documented simplification, not an oversight):
// the paper's causal Conv1d over the q/k branch before gating. That conv is
// there to help local n-gram-ish recall; the exponential-gated matrix memory
// below is xLSTM's actual defining mechanism and works without it.
// ═════════════════════════════════════════════════════════════════════════════

use burn::{
    module::Ignored, nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig,
    }, prelude::*, record::{BinFileRecorder, FullPrecisionSettings, Recorder}, tensor::activation::sigmoid,
};
use anyhow::Result;
use serde::{Serialize, Deserialize};

use super::tokenizer::{Tokenizer, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN};
use crate::brain::{
    bpe::{BpeTokenizer, TokenizerKind},
    fixer::fix_json_syntax,
    model::{GenerationResult, MLP, MLPConfig, RMSNorm, RMSNormConfig, TEMPERATURE, TOP_K},
    samples::{Action, CardinalDir, TrainingStage},
};

/// Forget-gate bias offset, added to the raw linear pre-activation before the
/// log-sigmoid. Classic LSTM trick (Jozefowicz et al., 2015): biasing the
/// forget gate open at initialization keeps early gradients from vanishing
/// before the model has learned anything about what to forget.
const FORGET_GATE_BIAS: f64 = 3.0;

// ═════════════════════════════════════════════════════════════════════════════
// Numerically-stable helpers
// ═════════════════════════════════════════════════════════════════════════════

/// log(sigmoid(x)), computed as -softplus(-x) via the overflow-safe identity
/// softplus(z) = relu(z) + log1p(exp(-|z|)) — exp(-|z|) never overflows.
fn log_sigmoid<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let neg_x = -x.clone();
    let abs_x = x.abs();
    (neg_x.clamp_min(0.0) + ((-abs_x).exp().add_scalar(1.0)).log()).neg()
}

/// Elementwise max(a, b) = relu(a - b) + b.
fn max_pair<B: Backend, const D: usize>(a: Tensor<B, D>, b: Tensor<B, D>) -> Tensor<B, D> {
    (a - b.clone()).clamp_min(0.0) + b
}

// ═════════════════════════════════════════════════════════════════════════════
// mLSTM block — exponential-gated matrix memory
// ═════════════════════════════════════════════════════════════════════════════

#[derive(Config, Debug)]
pub struct MLstmBlockConfig {
    d_model: usize,
    n_heads: usize,
}

impl MLstmBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MLstmBlock<B> {
        assert!(self.d_model % self.n_heads == 0, "xLSTM: d_model must be divisible by n_heads");
        let head_dim = self.d_model / self.n_heads;
        MLstmBlock {
            q: LinearConfig::new(self.d_model, self.d_model).with_bias(false).init(device),
            k: LinearConfig::new(self.d_model, self.d_model).with_bias(false).init(device),
            v: LinearConfig::new(self.d_model, self.d_model).with_bias(false).init(device),
            input_gate: LinearConfig::new(self.d_model, self.n_heads).init(device),
            forget_gate: LinearConfig::new(self.d_model, self.n_heads).init(device),
            out_gate: LinearConfig::new(self.d_model, self.d_model).init(device),
            o: LinearConfig::new(self.d_model, self.d_model).with_bias(false).init(device),
            n_heads: self.n_heads,
            head_dim,
        }
    }
}

#[derive(Module, Debug)]
pub struct MLstmBlock<B: Backend> {
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    input_gate:  Linear<B>,
    forget_gate: Linear<B>,
    out_gate:    Linear<B>,
    o: Linear<B>,
    n_heads:  usize,
    head_dim: usize,
}

impl<B: Backend> MLstmBlock<B> {
    /// x: [batch, seq, d_model] (already normed by the caller)
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, _] = x.dims();
        let nh = self.n_heads;
        let hd = self.head_dim;
        let device = x.device();
        let scale = (hd as f64).powf(-0.5);

        let reshape = |t: Tensor<B, 3>| {
            t.reshape([batch, seq, nh, hd]).swap_dims(1, 2) // [batch, nh, seq, hd]
        };

        let q = reshape(self.q.forward(x.clone())) * scale;
        let k = reshape(self.k.forward(x.clone()));
        let v = reshape(self.v.forward(x.clone()));

        // [batch, seq, nh] -> [batch, nh, seq]
        let i_tilde = self.input_gate.forward(x.clone()).swap_dims(1, 2);
        let f_tilde = self.forget_gate.forward(x.clone()).add_scalar(FORGET_GATE_BIAS).swap_dims(1, 2);

        let mut c_state = Tensor::<B, 4>::zeros([batch, nh, hd, hd], &device); // [b, nh, d_v, d_k]
        let mut n_state = Tensor::<B, 3>::zeros([batch, nh, hd], &device);     // [b, nh, d_k]
        let mut m_state = Tensor::<B, 2>::zeros([batch, nh], &device);         // [b, nh]

        let mut outputs: Vec<Tensor<B, 3>> = Vec::with_capacity(seq);

        for t in 0..seq {
            let q_t = q.clone().slice([0..batch, 0..nh, t..t + 1, 0..hd]).reshape([batch, nh, hd]);
            let k_t = k.clone().slice([0..batch, 0..nh, t..t + 1, 0..hd]).reshape([batch, nh, hd]);
            let v_t = v.clone().slice([0..batch, 0..nh, t..t + 1, 0..hd]).reshape([batch, nh, hd]);
            let i_raw = i_tilde.clone().slice([0..batch, 0..nh, t..t + 1]).reshape([batch, nh]);
            let f_raw = f_tilde.clone().slice([0..batch, 0..nh, t..t + 1]).reshape([batch, nh]);

            // Stabilized exponential gating (xLSTM paper, section 2.2/A.2):
            // m_t = max(log_f_t + m_{t-1}, i_tilde_t); gates re-based off m_t
            // so neither i_t nor f_t ever needs to exponentiate an unbounded value.
            let log_f = log_sigmoid(f_raw);
            let m_new = max_pair(log_f.clone() + m_state.clone(), i_raw.clone());
            let i_gate = (i_raw - m_new.clone()).exp();                    // [b, nh]
            let f_gate = (log_f + m_state - m_new.clone()).exp();          // [b, nh]

            // Rank-1 update to the matrix memory: C_t = f_t*C_{t-1} + i_t*(v_t k_t^T)
            let outer = v_t.clone().unsqueeze_dim::<4>(3) * k_t.clone().unsqueeze_dim::<4>(2); // [b,nh,hd(v),hd(k)]
            let f_c = f_gate.clone().unsqueeze_dim::<3>(2).unsqueeze_dim::<4>(3); // [b,nh,1,1]
            let i_c = i_gate.clone().unsqueeze_dim::<3>(2).unsqueeze_dim::<4>(3);
            c_state = c_state * f_c + outer * i_c;

            // Normalizer: n_t = f_t*n_{t-1} + i_t*k_t
            let f_n = f_gate.clone().unsqueeze_dim::<3>(2); // [b,nh,1]
            let i_n = i_gate.clone().unsqueeze_dim::<3>(2);
            n_state = n_state * f_n + k_t.clone() * i_n;

            // Readout: h_t = (C_t q_t) / max(|n_t . q_t|, exp(-m_t))
            //
            // Must be squeeze_dim (not the squeeze-all-size-1-dims form): at
            // inference/generation time batch=1, and a bare `.squeeze::<D2>()`
            // would also collapse the batch dim, panicking with a rank
            // mismatch ("Resulting dimensions 2 do not match the required D2
            // size 3") the moment a periodic mid-training inference check ran.
            let h_tilde = (c_state.clone() * q_t.clone().unsqueeze_dim::<4>(2)).sum_dim(3).squeeze_dim::<3>(3); // [b,nh,hd]
            let dot = (n_state.clone() * q_t.clone()).sum_dim(2).squeeze_dim::<2>(2).abs(); // [b,nh]
            let floor = (-m_new.clone()).exp();
            let denom = max_pair(dot, floor).clamp_min(1e-6);
            let h_t = h_tilde / denom.unsqueeze_dim::<3>(2);

            outputs.push(h_t.reshape([batch, 1, nh * hd]));
            m_state = m_new;
        }

        let h_all = Tensor::cat(outputs, 1); // [batch, seq, d_model]

        // Output gate, computed from the same (pre-mLSTM) input — same shape
        // as CrossAttentionBlock's unused `gate` field in classic_attn.rs.
        let og = sigmoid(self.out_gate.forward(x));
        self.o.forward(h_all * og)
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// XLstmBlock — pre-norm mLSTM + pre-norm MLP, residual around both
// ═════════════════════════════════════════════════════════════════════════════

#[derive(Config, Debug)]
pub struct XLstmBlockConfig {
    d_model:  usize,
    d_hidden: usize,
    n_heads:  usize,
}

impl XLstmBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> XLstmBlock<B> {
        XLstmBlock {
            mlstm_norm: RMSNormConfig::new(self.d_model).init(device),
            mlstm:      MLstmBlockConfig::new(self.d_model, self.n_heads).init(device),
            mlp_norm:   RMSNormConfig::new(self.d_model).init(device),
            mlp:        MLPConfig::new(self.d_model, self.d_hidden).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct XLstmBlock<B: Backend> {
    mlstm_norm: RMSNorm<B>,
    mlstm:      MLstmBlock<B>,
    mlp_norm:   RMSNorm<B>,
    mlp:        MLP<B>,
}

impl<B: Backend> XLstmBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x.clone() + self.mlstm.forward(self.mlstm_norm.forward(x));
        x.clone() + self.mlp.forward(self.mlp_norm.forward(x))
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// YumonXLstmBrain
// ═════════════════════════════════════════════════════════════════════════════

#[derive(Module, Debug)]
pub struct YumonXLstmBrain<B: Backend> {
    pub config: Ignored<YumonXLstmBrainConfig>,

    embedding: Embedding<B>,
    blocks:    Vec<XLstmBlock<B>>,
    norm:      RMSNorm<B>,

    dropout:        Dropout,
    pub token_head: Linear<B>,
}

#[derive(Config, Debug)]
pub struct YumonXLstmBrainConfig {
    pub vocab_size:   usize,
    #[config(default = 0.05)]
    pub dropout_rate: f64,
    #[config(default = 256)]
    pub embed_dim:    usize,
    #[config(default = 256)]
    pub hidden_units: usize,
    #[config(default = 2)]
    pub n_layers:     usize,
    /// Number of independent per-head matrix-memory states, reusing the same
    /// RunConfig field name attention architectures call `attn_heads`.
    #[config(default = 4)]
    pub attn_heads:   usize,
    #[config(default = 1024)]
    pub ff_dim:       usize,
    #[config(default = 320)]
    pub max_seq_len:  usize,
    pub training_stage: TrainingStage,
}

impl YumonXLstmBrainConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> YumonXLstmBrain<B> {
        let blocks = (0..self.n_layers)
            .map(|_| XLstmBlockConfig::new(self.embed_dim, self.ff_dim, self.attn_heads).init(device))
            .collect();

        YumonXLstmBrain {
            config: Ignored(self.clone()),

            embedding: EmbeddingConfig::new(self.vocab_size, self.embed_dim).init(device),
            blocks,
            norm:      RMSNormConfig::new(self.embed_dim).init(device),

            dropout:    DropoutConfig::new(self.dropout_rate).init(),
            token_head: LinearConfig::new(self.embed_dim, self.vocab_size).init(device),
        }
    }
}

impl<B: Backend> YumonXLstmBrain<B> {
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut x = self.dropout.forward(self.embedding.forward(tokens));
        for block in &self.blocks {
            x = block.forward(x);
        }
        let x = self.norm.forward(x);
        self.token_head.forward(x)
    }

    pub fn generate_unmasked_parsed(
        &self,
        tokenizer:  &TokenizerKind,
        seed_text:  &str,
        max_tokens: usize,
        device:     &B::Device,
    ) -> GenerationResult {
        let mut dec_ids: Vec<usize> = vec![BOS_TOKEN];
        if !seed_text.is_empty() {
            dec_ids.extend(tokenizer.encode(seed_text).iter().map(|&t| t as usize));
        }

        let sep_text = if self.config.training_stage == TrainingStage::Structured { "\n---\n" } else { " " };
        let sep_tokens = tokenizer.encode(sep_text);
        dec_ids.extend(sep_tokens.iter().map(|&t| t as usize));

        let prompt_len = dec_ids.len();
        let mut rng = rand::thread_rng();

        for _ in 0..max_tokens {
            let current_len = dec_ids.len();
            if current_len >= self.config.max_seq_len { break; }

            let dec_tokens_t = Tensor::<B, 2, Int>::from_ints(
                TensorData::new(
                    dec_ids.iter().map(|&t| t as i32).collect::<Vec<_>>(),
                    [1, current_len],
                ),
                device,
            );

            let token_logits = self.forward(dec_tokens_t);

            let vocab_size  = tokenizer.vocab_size();
            let last_logits = token_logits
                .slice([0..1, current_len - 1..current_len, 0..vocab_size])
                .reshape([vocab_size]);

            let logits_vec: Vec<f32> = last_logits.to_data().to_vec().unwrap();
            let next_token = sample_top_k(&logits_vec, TOP_K, TEMPERATURE, &mut rng);

            if next_token == EOS_TOKEN || next_token == PAD_TOKEN { break; }
            dec_ids.push(next_token);
        }

        let raw_output = tokenizer.decode(&dec_ids[prompt_len..]);
        let fixed = fix_json_syntax(&raw_output).fixed;

        let extract = |key: &str| -> String {
            fancy_regex::Regex::new(&format!(r#"(?<=\s*"{key}"\s*:\s*)"([^"]*)""#))
                .ok()
                .and_then(|re| re.captures(&fixed).ok().flatten())
                .and_then(|caps| caps.get(1))
                .map(|m| m.as_str().to_string())
                .unwrap_or_default()
        };

        let mut parsed_action   = extract("action");
        let mut parsed_reply    = extract("reply");
        let mut parsed_emotion  = extract("emotion");

        if parsed_action.is_empty() || parsed_action.len() < 3 {
            parsed_action  = extract(" action");
            parsed_reply   = extract(" reply");
            parsed_emotion = extract(" emotion");
        }

        if parsed_reply.is_empty() || parsed_reply.len() < 4 {
            let parsed: serde_json::Value = serde_json::from_str(&fixed)
                .unwrap_or_else(|_| {
                    let extract = |key: &str| -> String {
                        regex::Regex::new(&format!(r#""{key}"\s*:\s*"([^"]*)"#))
                            .ok()
                            .and_then(|re| re.captures(&fixed))
                            .and_then(|caps| caps.get(1))
                            .map(|m| m.as_str().to_string())
                            .unwrap_or_default()
                    };

                    serde_json::json!({
                        "action":  extract("action"),
                        "emotion": extract("emotion"),
                        "reply":   extract("reply"),
                    })
                });

            parsed_action  = parsed["action"].to_string().trim().to_string();
            parsed_reply   = parsed["reply"].to_string().trim().to_string();
            parsed_emotion = parsed["emotion"].to_string().trim().to_string();
        }

        parsed_action  = parsed_action.replace("\"", "").trim().to_string();
        parsed_reply   = parsed_reply.replace("\"", "").trim().to_string();
        parsed_emotion = parsed_emotion.replace("\"", "").trim().to_string();

        let action = match parsed_action.as_str().trim() {
            "go to destination" => Action::GoToDestination,
            "go home"           => Action::GoHome,
            "follow"            => Action::Follow,
            "get help"          => Action::GetHelp,
            "survey area"       => Action::Survey,
            "collect items"     => Action::Collect,
            "stack items"       => Action::Stack,
            _                   => Action::Sit,
        };

        GenerationResult {
            reply: parsed_reply,
            action,
            motion_dir: CardinalDir::None,
            parsed_emotion,
            raw_output,
            fsm_state: 0,
            allowed_count: None,
        }
    }

    // ── Checkpoint I/O ────────────────────────────────────────────────────────

    pub fn save(&self, directory: &str, tokenizer: &TokenizerKind, metadata: &XLstmMetadata) -> Result<()> {
        let dir = std::path::Path::new(directory);
        std::fs::create_dir_all(dir)?;

        let meta_json = serde_json::to_string_pretty(metadata)?;
        std::fs::write(dir.join("metadata.json"), meta_json)?;

        tokenizer.save(dir.join("tokenizer.json").to_str().unwrap())?;

        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        self.clone().save_file(dir.join("model"), &recorder)
            .map_err(|e| anyhow::anyhow!("save_file: {e:?}"))?;

        Ok(())
    }

    pub fn load(directory: &str, device: &B::Device) -> Result<(Self, TokenizerKind, YumonXLstmBrainConfig)> {
        let dir = std::path::Path::new(directory);

        let meta_json = std::fs::read_to_string(dir.join("metadata.json"))?;
        let metadata: XLstmMetadata = serde_json::from_str(&meta_json)?;

        let tokenizer = TokenizerKind::Bpe(BpeTokenizer::load("yumon_bpe")?);

        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let record = recorder.load(dir.join("model").into(), device)
            .map_err(|e| anyhow::anyhow!("load: {e:?}"))?;

        let config = YumonXLstmBrainConfig {
            vocab_size:   metadata.vocab_size,
            embed_dim:    metadata.embed_dim,
            hidden_units: metadata.hidden_units,
            n_layers:     metadata.n_layers,
            attn_heads:   metadata.attn_heads,
            ff_dim:       metadata.ff_dim,
            max_seq_len:  metadata.max_seq_len,
            training_stage: metadata.training_stage,
            dropout_rate: 0.05,
        };

        let model = config.init::<B>(device).load_record(record);

        Ok((model, tokenizer, config))
    }
}

// ─── Metadata ─────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct XLstmMetadata {
    pub vocab_size:     usize,
    pub epochs_trained: usize,
    pub final_loss:     f32,
    pub batch_size:     usize,
    pub training_stage: TrainingStage,
    pub embed_dim:      usize,
    pub hidden_units:   usize,
    pub n_layers:       usize,
    pub attn_heads:     usize,
    pub ff_dim:         usize,
    pub max_seq_len:    usize,
}

// ─── Sampling helper (same shape as model.rs's private copy) ─────────────────

fn sample_top_k(logits: &[f32], k: usize, temperature: f32, rng: &mut impl rand::Rng) -> usize {
    use rand::distributions::WeightedIndex;
    use rand::prelude::*;

    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate()
        .map(|(i, &l)| (i, l / temperature))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(k);

    let max = indexed[0].1;
    let weights: Vec<f32> = indexed.iter().map(|(_, l)| (l - max).exp()).collect();
    let sum: f32 = weights.iter().sum();
    let probs: Vec<f32> = weights.iter().map(|w| w / sum).collect();

    let dist = WeightedIndex::new(&probs).unwrap();
    indexed[dist.sample(rng)].0
}

// ─── Smoke test — real Wgpu/Autodiff backend (same one training uses), tiny
// dims, one forward+backward pass. Catches tensor-shape/rank mistakes in the
// mLSTM recurrence without running an actual training loop. ───────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::train::TrainBackend;
    use burn::optim::GradientsParams;

    #[test]
    fn xlstm_forward_backward_smoke() {
        let device = burn::backend::wgpu::WgpuDevice::default();

        let config = YumonXLstmBrainConfig {
            vocab_size:   50,
            embed_dim:    16,
            hidden_units: 16,
            n_layers:     1,
            attn_heads:   2,
            ff_dim:       32,
            max_seq_len:  8,
            training_stage: TrainingStage::Structured,
            dropout_rate: 0.0,
        };
        let model: YumonXLstmBrain<TrainBackend> = config.init(&device);

        let (batch, seq, vocab) = (2usize, 8usize, config.vocab_size);
        let ids: Vec<i32> = (0..(batch * seq) as i32).map(|i| i % vocab as i32).collect();
        let tokens = Tensor::<TrainBackend, 2, Int>::from_ints(TensorData::new(ids, [batch, seq]), &device);

        let logits = model.forward(tokens);
        assert_eq!(logits.dims(), [batch, seq, vocab]);

        // Confirms gradients actually flow back through the recurrence (the
        // per-timestep scan, the stabilizer, the matrix-memory update) without
        // panicking — the failure mode a shape/rank bug here would produce.
        let loss = logits.sum();
        let _grads = GradientsParams::from_grads(loss.backward(), &model);

        // Real regression caught in an actual training run: at batch=1 (the
        // shape every periodic mid-training inference/generation call uses),
        // a squeeze that collapses ALL size-1 dims also eats the batch dim,
        // not just the intended one, and panics with a rank mismatch. Cover
        // batch=1 here so this class of bug fails a test, not a live run.
        let ids_b1: Vec<i32> = (0..seq as i32).map(|i| i % vocab as i32).collect();
        let tokens_b1 = Tensor::<TrainBackend, 2, Int>::from_ints(TensorData::new(ids_b1, [1, seq]), &device);
        let logits_b1 = model.forward(tokens_b1);
        assert_eq!(logits_b1.dims(), [1, seq, vocab]);
    }

    /// Not a correctness check — measures steady-state (post-warmup) wall clock
    /// for one real grid-search-sized config (128h/2l/32a/64len/b2, matching
    /// generate_run_configs' smallest cell) so the sequential per-timestep scan's
    /// real cost is known before a real training run, not guessed. Run with
    /// `cargo test --release --lib xlstm_timing_probe -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn xlstm_timing_probe() {
        let device = burn::backend::wgpu::WgpuDevice::default();

        let config = YumonXLstmBrainConfig {
            vocab_size:   4000,
            embed_dim:    128,
            hidden_units: 128,
            n_layers:     2,
            attn_heads:   32,
            ff_dim:       512,
            max_seq_len:  64,
            training_stage: TrainingStage::Structured,
            dropout_rate: 0.05,
        };
        let model: YumonXLstmBrain<TrainBackend> = config.init(&device);

        let (batch, seq, vocab) = (2usize, config.max_seq_len, config.vocab_size);

        // Warmup — first call pays shader-compilation cost, not representative.
        for _ in 0..2 {
            let ids: Vec<i32> = (0..(batch * seq) as i32).map(|i| i % vocab as i32).collect();
            let tokens = Tensor::<TrainBackend, 2, Int>::from_ints(TensorData::new(ids, [batch, seq]), &device);
            let logits = model.forward(tokens);
            let _grads = GradientsParams::from_grads(logits.sum().backward(), &model);
        }

        let n_steps = 10;
        let start = std::time::Instant::now();
        for _ in 0..n_steps {
            let ids: Vec<i32> = (0..(batch * seq) as i32).map(|i| i % vocab as i32).collect();
            let tokens = Tensor::<TrainBackend, 2, Int>::from_ints(TensorData::new(ids, [batch, seq]), &device);
            let logits = model.forward(tokens);
            let _grads = GradientsParams::from_grads(logits.sum().backward(), &model);
        }
        let elapsed = start.elapsed();
        println!(
            "xlstm_timing_probe: {} steps (batch={}, seq={}, embed={}, layers={}, heads={}) in {:?} -> {:?}/step",
            n_steps, batch, seq, config.embed_dim, config.n_layers, config.attn_heads,
            elapsed, elapsed / n_steps as u32,
        );
    }
}
