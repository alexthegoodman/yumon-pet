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
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, Lstm, LstmConfig, LstmState, attention::{CrossAttention, CrossAttentionConfig, MhaInput, MultiHeadAttention, MultiHeadAttentionConfig}
    },
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{TensorData, activation::sigmoid},
};
use anyhow::Result;
use serde::{Serialize, Deserialize};

use crate::{brain::{bpe::{BpeTokenizer, TokenizerKind}, train::MAX_SEQ_LEN}, vision::EMOTE_CLASSES};
use super::{
    tokenizer::{Tokenizer, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN},
    CONTEXT_DIMS,
};

use burn::nn::transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput};

// pub const EMBED_DIM:    usize = 768;
// pub const HIDDEN_UNITS: usize = 768;
// pub const EMBED_DIM:    usize = 512;
// pub const HIDDEN_UNITS: usize = 512;
// pub const EMBED_DIM:    usize = 256;
// pub const HIDDEN_UNITS: usize = 256;
pub const EMBED_DIM:    usize = 128;
pub const HIDDEN_UNITS: usize = 128;
pub const ATTN_HEADS:   usize = 2;
// pub const N_LAYERS:     usize = 4;
// pub const N_LAYERS:     usize = 3;
pub const N_LAYERS:     usize = 2;
// pub const FF_DIM:       usize = 512;
// pub const FF_DIM:       usize = 256;
pub const FF_DIM:       usize = 256;
// pub const FF_DIM:       usize = 2048;

pub const TEMPERATURE: f32  = 0.7;
pub const TOP_K:       usize = 10;

#[derive(Module, Debug)]
pub struct YumonBrain<B: Backend> {
    embedding:        Embedding<B>,
    pos_embedding:    Embedding<B>,   // learned positional
    transformer:      TransformerEncoder<B>,
    norm:             LayerNorm<B>,
    dropout:          Dropout,
    token_head:       Linear<B>,
    yumon_emote_head: Linear<B>,
}

#[derive(Config, Debug)]
pub struct YumonBrainConfig {
    pub vocab_size:   usize,
    #[config(default = 0.05)]
    pub dropout_rate: f64,
}

impl YumonBrainConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> YumonBrain<B> {
        let transformer_config = TransformerEncoderConfig::new(
            EMBED_DIM,
            FF_DIM,
            ATTN_HEADS,
            N_LAYERS,
        )
        .with_dropout(self.dropout_rate);

        YumonBrain {
            embedding:        EmbeddingConfig::new(self.vocab_size, EMBED_DIM).init(device),
            pos_embedding:    EmbeddingConfig::new(MAX_SEQ_LEN, EMBED_DIM).init(device),
            transformer:      transformer_config.init(device),
            norm:             LayerNormConfig::new(EMBED_DIM).init(device),
            dropout:          DropoutConfig::new(self.dropout_rate).init(),
            token_head:       LinearConfig::new(EMBED_DIM, self.vocab_size).init(device),
            yumon_emote_head: LinearConfig::new(EMBED_DIM, EMOTE_CLASSES).init(device),
        }
    }
}

impl<B: Backend> YumonBrain<B> {
    pub fn forward(
        &self,
        tokens:  Tensor<B, 2, Int>,
        context: Tensor<B, 2>,
        _prev_state: Option<()>,   // no recurrent state needed, kept for API compat
    ) -> (Tensor<B, 3>, Tensor<B, 2>, ()) {
        let [batch, seq_len] = tokens.dims();

        // 1. Token + positional embeddings
        let tok_emb = self.embedding.forward(tokens.clone());   // [batch, seq, embed]

        let positions = Tensor::<B, 2, Int>::from_ints(
            TensorData::new(
                (0..seq_len as i32).collect::<Vec<_>>()
                    .into_iter()
                    .cycle()
                    .take(batch * seq_len)
                    .collect::<Vec<_>>(),
                [batch, seq_len],
            ),
            &tok_emb.device(),
        );
        let pos_emb = self.pos_embedding.forward(positions);    // [batch, seq, embed]

        let x = self.dropout.forward(tok_emb + pos_emb);       // [batch, seq, embed]

        // 2. Causal mask — prevents attending to future tokens
        let mask_attn = Tensor::<B, 3, Bool>::tril_mask(
            [batch, seq_len, seq_len], 0, &x.device()
        );

        // 3. Padding mask from PAD tokens
        let mask_pad = tokens.equal_elem(PAD_TOKEN as u32);     // [batch, seq], true = pad

        // 4. Transformer encoder (GPT-style causal)
        let input = TransformerEncoderInput::new(x)
            .mask_attn(mask_attn)
            .mask_pad(mask_pad);

        let x = self.transformer.forward(input);                // [batch, seq, embed]
        let x = self.norm.forward(x);

        // 5. Heads
        let token_logits = self.token_head.forward(x.clone());  // [batch, seq, vocab]

        let last = x
            .slice([0..batch, seq_len - 1..seq_len])
            .reshape([batch, EMBED_DIM]);
        let emote_logits = self.yumon_emote_head.forward(last);

        (token_logits, emote_logits, ())
    }

    pub fn generate(
        &self,
        tokenizer:      &TokenizerKind,
        class_probs:    &[f32],
        emote_probs:    &[f32],
        user_emote_idx: usize,
        seed_text:      &str,   // completion prefix — Yumon continues from here
        max_tokens:     usize,
        device:         &B::Device,
    ) -> GenerationResult {
        // println!("Generate {:?}", seed_text);

        // Build static context tensor
        let mut ctx_flat = Vec::with_capacity(CONTEXT_DIMS);
        ctx_flat.extend_from_slice(class_probs);
        ctx_flat.extend_from_slice(emote_probs);
        let mut onehot = vec![0.0f32; EMOTE_CLASSES];
        onehot[user_emote_idx.min(EMOTE_CLASSES - 1)] = 1.0;
        ctx_flat.extend_from_slice(&onehot);

        let context_t = Tensor::<B, 2>::from_floats(
            TensorData::new(ctx_flat, [1, CONTEXT_DIMS]),
            device,
        );

        // Seed token sequence — the prefix Yumon completes
        let mut token_ids: Vec<usize> = if seed_text.is_empty() {
            vec![BOS_TOKEN]
        } else {
            std::iter::once(BOS_TOKEN)
                .chain(tokenizer.encode(seed_text).into_iter())
                .collect()
        };

        let mut rng = rand::thread_rng();
        let mut last_emote_logits: Option<Vec<f32>> = None;
        // let mut last_state = None;

        for _ in 0..max_tokens {
            // Clamp + pad to MAX_SEQ_LEN
            let clamped_len = token_ids.len().min(MAX_SEQ_LEN);
            let mut padded = token_ids[token_ids.len() - clamped_len..].to_vec();
            padded.resize(MAX_SEQ_LEN, PAD_TOKEN);

            let tokens_t = Tensor::<B, 2, Int>::from_ints(
                TensorData::new(
                    padded.iter().map(|&t| t as i32).collect::<Vec<_>>(),
                    [1, MAX_SEQ_LEN],
                ),
                device,
            );

            let (token_logits, emote_logits, lstm_state) =
                self.forward(tokens_t, context_t.clone(), None);

            // last_state = Some(lstm_state);

            // Sample from the last real position
            let last_idx  = clamped_len - 1;
            let vocab     = tokenizer.vocab_size();
            let last_logits = token_logits
                .slice([0..1, last_idx..last_idx + 1, 0..vocab])
                .reshape([vocab]);

            let logits_vec: Vec<f32> = last_logits.to_data().to_vec().unwrap();
            last_emote_logits = Some(emote_logits.to_data().to_vec::<f32>().unwrap());

            let next_token = sample_top_k(&logits_vec, TOP_K, TEMPERATURE, &mut rng);
            // println!("Check token {:?}", next_token);
            if next_token == EOS_TOKEN || next_token == PAD_TOKEN { break; }
            token_ids.push(next_token);
        }

        // Decode only the generated portion (after seed)
        let seed_len = if seed_text.is_empty() { 1 } 
                       else { 1 + tokenizer.encode(seed_text).len() };
        let reply = tokenizer.decode(&token_ids[seed_len..]);
        let yumon_emote_idx = last_emote_logits.as_deref().map(argmax).unwrap_or(4);

        GenerationResult { reply, yumon_emote_idx }
    }

    // ── Checkpoint I/O ────────────────────────────────────────────────────────

    pub fn save(&self, directory: &str, tokenizer: &TokenizerKind, metadata: &BrainMetadata) -> Result<()> {
        let dir = std::path::Path::new(directory);
        std::fs::create_dir_all(dir)?;

        let meta_json = serde_json::to_string_pretty(metadata)?;
        std::fs::write(dir.join("metadata.json"), meta_json)?;

        tokenizer.save(dir.join("tokenizer.json").to_str().unwrap())?;

        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        self.clone().save_file(dir.join("model"), &recorder)
            .map_err(|e| anyhow::anyhow!("save_file: {e:?}"))?;

        // println!("✅ Brain checkpoint saved → {directory}");
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

    pub fn load_app(directory: &str, device: &B::Device) -> Result<(Self, TokenizerKind)> {
        let dir = std::path::Path::new(directory);

        let meta_json = std::fs::read_to_string(dir.join("metadata.json"))?;
        let metadata: BrainMetadata = serde_json::from_str(&meta_json)?;

        // let tokenizer = Tokenizer::load(dir.join("tokenizer.json").to_str().unwrap())?;

        let use_bpe = true;

        let tokenizer = if use_bpe {
            TokenizerKind::Bpe(BpeTokenizer::load("../../yumon-pet/yumon_bpe")?)
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
