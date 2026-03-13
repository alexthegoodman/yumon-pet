/// Yumon Brain — LSTM language model with vision context injection.
///
/// Architecture:
///   Input per timestep: token_embedding (EMBED_DIM) ++ context_vector (CONTEXT_DIMS)
///   → LSTM(512) → Dropout(0.3) → Dense(256, ReLU)
///   ├─ token_head:       Dense(vocab_size)   — next character prediction
///   └─ yumon_emote_head: Dense(EMOTE_CLASSES) — Yumon's emotional response
///
/// Context vector (114 dims):
///   [class_probs: 100] ++ [user_emote_probs: 7] ++ [user_emote_onehot: 7]
///
/// The emote head only fires on the EOS timestep during training (conditioned on
/// the full context of the generated reply). At inference we read it from the
/// last generated step.
///
/// Token generation: temperature-sampled autoregressive decoding with top-k filtering.

use burn::{
    nn::{
        Dropout, DropoutConfig,
        Embedding, EmbeddingConfig,
        Linear, LinearConfig,
        Lstm, LstmConfig,
    },
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::TensorData,
};
use anyhow::Result;
use serde::{Serialize, Deserialize};

use crate::{brain::bpe::{BpeTokenizer, TokenizerKind}, vision::EMOTE_CLASSES};
use super::{
    tokenizer::{Tokenizer, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN},
    CONTEXT_DIMS,
};

// ─── Hyperparameters ─────────────────────────────────────────────────────────

// pub const EMBED_DIM:   usize = 64;
pub const EMBED_DIM:   usize = 256; //  we have ~173 Ints total per input
pub const LSTM_UNITS:  usize = 512; // medium (fish)
pub const HIDDEN_UNITS: usize = 256; // medium
// pub const LSTM_UNITS:  usize = 1024; // large
// pub const HIDDEN_UNITS: usize = 512; // large
// pub const EMBED_DIM:   usize = 16; // too small
// pub const LSTM_UNITS:  usize = 128; // too small
// pub const HIDDEN_UNITS: usize = 32; // too small
pub const TEMPERATURE: f32   = 0.7;
pub const TOP_K:       usize = 10;

// ─── Model ───────────────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct YumonBrain<B: Backend> {
    embedding:       Embedding<B>,
    // We project (embed + context) → lstm_input_dim before the LSTM
    // because Burn's LSTM input size must be fixed at init.
    input_proj:      Linear<B>,
    lstm:            Lstm<B>,
    dropout:         Dropout,
    dense:           Linear<B>,
    token_head:      Linear<B>,
    yumon_emote_head: Linear<B>,
}

#[derive(Config, Debug)]
pub struct YumonBrainConfig {
    pub vocab_size:  usize,
    #[config(default = 0.3)]
    pub dropout_rate: f64,
}

impl YumonBrainConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> YumonBrain<B> {
        let lstm_in = EMBED_DIM + CONTEXT_DIMS;
        YumonBrain {
            embedding:        EmbeddingConfig::new(self.vocab_size, EMBED_DIM).init(device),
            input_proj:       LinearConfig::new(lstm_in, lstm_in).init(device),
            lstm:             LstmConfig::new(lstm_in, LSTM_UNITS, false).init(device),
            dropout:          DropoutConfig::new(self.dropout_rate).init(),
            dense:            LinearConfig::new(LSTM_UNITS, HIDDEN_UNITS).init(device),
            token_head:       LinearConfig::new(HIDDEN_UNITS, self.vocab_size).init(device),
            yumon_emote_head: LinearConfig::new(HIDDEN_UNITS, EMOTE_CLASSES).init(device),
        }
    }
}

impl<B: Backend> YumonBrain<B> {
    /// Forward pass for training — processes a full sequence.
    ///
    /// token_ids:    [batch, seq_len]       — input token indices
    /// context:      [batch, CONTEXT_DIMS]  — vision + emote context (same for all timesteps)
    ///
    /// Returns:
    ///   token_logits:  [batch, seq_len, vocab_size]
    ///   emote_logits:  [batch, EMOTE_CLASSES]  — from last timestep only
    pub fn forward(
        &self,
        token_ids: Tensor<B, 2, burn::tensor::Int>,
        context:   Tensor<B, 2>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        use burn::tensor::activation::relu;

        let [batch, seq_len] = token_ids.dims();

        // Embed tokens → [batch, seq_len, EMBED_DIM]
        let embeds = self.embedding.forward(token_ids);

        // Expand context to [batch, seq_len, CONTEXT_DIMS] and concat
        let ctx_expanded = context
            .unsqueeze_dim::<3>(1)                          // [batch, 1, ctx]
            .expand([batch, seq_len, CONTEXT_DIMS]);         // [batch, seq, ctx]

        let lstm_in = Tensor::cat(vec![embeds, ctx_expanded], 2); // [batch, seq, embed+ctx]

        // Project (no-op if already right size, but helps with gradient flow)
        let lstm_in = self.input_proj.forward(lstm_in);

        // LSTM
        let (out_seq, _) = self.lstm.forward(lstm_in, None);
        // out_seq: [batch, seq_len, LSTM_UNITS]

        let dropped = self.dropout.forward(out_seq.clone());
        let shared  = relu(self.dense.forward(dropped)); // [batch, seq, HIDDEN]

        // Token logits — every timestep
        let token_logits = self.token_head.forward(shared.clone()); // [batch, seq, vocab]

        // Emote logits — last timestep only
        let last = shared
            .slice([0..batch, seq_len - 1..seq_len, 0..HIDDEN_UNITS])
            .reshape([batch, HIDDEN_UNITS]);
        let emote_logits = self.yumon_emote_head.forward(last); // [batch, EMOTE]

        (token_logits, emote_logits)
    }

    /// Autoregressive generation.
    /// Returns a GenerationResult with the reply text and Yumon's emote index.
    pub fn generate(
        &self,
        tokenizer:       &TokenizerKind,
        class_probs:     &[f32],
        emote_probs:     &[f32],
        user_emote_idx:  usize,
        seed_text:       &str,
        max_tokens:      usize,
        device:          &B::Device,
    ) -> GenerationResult {
        use burn::tensor::activation::log_softmax;

        // Build context vector [1, CONTEXT_DIMS]
        let mut ctx_flat = Vec::with_capacity(CONTEXT_DIMS);
        ctx_flat.extend_from_slice(class_probs);
        ctx_flat.extend_from_slice(emote_probs);
        // user_emote_onehot
        let mut onehot = vec![0.0f32; EMOTE_CLASSES];
        onehot[user_emote_idx.min(EMOTE_CLASSES - 1)] = 1.0;
        ctx_flat.extend_from_slice(&onehot);

        let context_t = Tensor::<B, 2>::from_floats(
            TensorData::new(ctx_flat, [1, CONTEXT_DIMS]),
            device,
        );

        // Seed tokens
        let mut token_ids: Vec<usize> = vec![BOS_TOKEN];
        if !seed_text.is_empty() {
            token_ids.extend(tokenizer.encode(seed_text));
        }

        let mut rng = rand::thread_rng();
        let mut last_emote_logits: Option<Vec<f32>> = None;

        for _ in 0..max_tokens {
            let seq_len = token_ids.len();
            let ids_flat: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();

            let ids_t = Tensor::<B, 2, burn::tensor::Int>::from_ints(
                TensorData::new(ids_flat, [1, seq_len]),
                device,
            );

            let (token_logits, emote_logits) = self.forward(ids_t, context_t.clone());

            // Last timestep token logits → [vocab_size]
            let vocab = tokenizer.vocab_size();
            let last_logits = token_logits
                .slice([0..1, seq_len - 1..seq_len, 0..vocab])
                .reshape([vocab]);

            let logits_vec: Vec<f32> = last_logits.to_data().to_vec().unwrap();

            // Save emote logits
            last_emote_logits = Some(emote_logits.to_data().to_vec::<f32>().unwrap());

            // Temperature + top-k sampling
            let next_token = sample_top_k(&logits_vec, TOP_K, TEMPERATURE, &mut rng);
            if next_token == EOS_TOKEN || next_token == PAD_TOKEN { break; }
            token_ids.push(next_token);
        }

        // Decode reply (skip BOS)
        let reply = tokenizer.decode(&token_ids[1..]);

        // Yumon emote = argmax of last emote_logits
        let yumon_emote_idx = last_emote_logits
            .as_deref()
            .map(argmax)
            .unwrap_or(4); // default neutral

        GenerationResult { reply, yumon_emote_idx }
    }

    // ── Checkpoint I/O ────────────────────────────────────────────────────────

    pub fn save(&self, directory: &str, tokenizer: &Tokenizer, metadata: &BrainMetadata) -> Result<()> {
        let dir = std::path::Path::new(directory);
        std::fs::create_dir_all(dir)?;

        let meta_json = serde_json::to_string_pretty(metadata)?;
        std::fs::write(dir.join("metadata.json"), meta_json)?;

        tokenizer.save(dir.join("tokenizer.json").to_str().unwrap())?;

        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        self.clone().save_file(dir.join("model"), &recorder)
            .map_err(|e| anyhow::anyhow!("save_file: {e:?}"))?;

        println!("✅ Brain checkpoint saved → {directory}");
        Ok(())
    }

    pub fn load(directory: &str, device: &B::Device) -> Result<(Self, TokenizerKind)> {
        let dir = std::path::Path::new(directory);

        let meta_json = std::fs::read_to_string(dir.join("metadata.json"))?;
        let metadata: BrainMetadata = serde_json::from_str(&meta_json)?;

        // let tokenizer = Tokenizer::load(dir.join("tokenizer.json").to_str().unwrap())?;

        let use_bpe = true;

        let tokenizer = if use_bpe {
            TokenizerKind::Bpe(BpeTokenizer::load("yumon_bpe")?)
        } else {
            TokenizerKind::Char(Tokenizer::load(dir.join("tokenizer.json").to_str().unwrap())?)
        };

        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let record   = recorder.load(dir.join("model").into(), device)
            .map_err(|e| anyhow::anyhow!("load: {e:?}"))?;
        let model    = YumonBrainConfig::new(metadata.vocab_size).init::<B>(device)
                          .load_record(record);

        Ok((model, tokenizer))
    }
}

// ─── Generation Result ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub reply:           String,
    pub yumon_emote_idx: usize,
}

// ─── Metadata ─────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct BrainMetadata {
    pub vocab_size:     usize,
    pub epochs_trained: usize,
    pub final_loss:     f32,
}

// ─── Sampling helpers ─────────────────────────────────────────────────────────

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

fn argmax(arr: &[f32]) -> usize {
    arr.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}
