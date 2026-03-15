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
        Lstm, LstmConfig, attention::{CrossAttention, CrossAttentionConfig, MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
    },
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::TensorData,
};
use anyhow::Result;
use serde::{Serialize, Deserialize};

use crate::{brain::{bpe::{BpeTokenizer, TokenizerKind}, train::MAX_SEQ_LEN}, vision::EMOTE_CLASSES};
use super::{
    tokenizer::{Tokenizer, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN},
    CONTEXT_DIMS,
};

// ─── Hyperparameters ─────────────────────────────────────────────────────────

// pub const EMBED_DIM:   usize = 64;
pub const EMBED_DIM:   usize = 128; //  we have ~173 Ints total per input
// pub const LSTM_UNITS:  usize = 512; // medium (fish)
// pub const HIDDEN_UNITS: usize = 256; // medium

// pub const EMBED_DIM:   usize = 128; //  we have ~173 Ints total per input
// pub const LSTM_UNITS:  usize = 256; // medium-small with 2 layers
// pub const HIDDEN_UNITS: usize = 128; // medium

pub const LSTM_UNITS:  usize = 512;
pub const HIDDEN_UNITS: usize = 512;
pub const ATTN_HEADS: usize = 4;

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
    // lstm2:            Lstm<B>,
    dropout:         Dropout,
    // dense:           Linear<B>,
    // encoder_attention: MultiHeadAttention<B>,
    encoder_attention: CrossAttention<B>,
    attn_proj:  Linear<B>,
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
        println!("lstm_in usize {:?}", lstm_in);
        YumonBrain {
            embedding:        EmbeddingConfig::new(self.vocab_size, EMBED_DIM).init(device),
            input_proj:       LinearConfig::new(lstm_in, lstm_in).init(device),
            lstm:             LstmConfig::new(lstm_in, LSTM_UNITS, false).init(device),
            // lstm2:             LstmConfig::new(LSTM_UNITS, LSTM_UNITS, false).init(device),
            dropout:          DropoutConfig::new(self.dropout_rate).init(),
            // encoder_attention: MultiHeadAttentionConfig::new(LSTM_UNITS, ATTN_HEADS).init(device),
            encoder_attention: CrossAttentionConfig::new(LSTM_UNITS, LSTM_UNITS, ATTN_HEADS, 1, LSTM_UNITS / ATTN_HEADS).with_quiet_softmax(true).init(device),
            // dense:            LinearConfig::new(LSTM_UNITS, HIDDEN_UNITS).init(device),
            // token_head:       LinearConfig::new(HIDDEN_UNITS, self.vocab_size).init(device),
            // yumon_emote_head: LinearConfig::new(HIDDEN_UNITS, EMOTE_CLASSES).init(device),
            attn_proj: LinearConfig::new(LSTM_UNITS, HIDDEN_UNITS).init(device),
            token_head:       LinearConfig::new(LSTM_UNITS, self.vocab_size).init(device),
            yumon_emote_head: LinearConfig::new(LSTM_UNITS, EMOTE_CLASSES).init(device),
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
    // pub fn forward(
    //     &self,
    //     token_ids: Tensor<B, 2, burn::tensor::Int>,
    //     context:   Tensor<B, 2>,
    // ) -> (Tensor<B, 3>, Tensor<B, 2>) {
    //     use burn::tensor::activation::relu;

    //     let [batch, seq_len] = token_ids.dims();

    //     // Embed tokens → [batch, seq_len, EMBED_DIM]
    //     let embeds = self.embedding.forward(token_ids);

    //     // Expand context to [batch, seq_len, CONTEXT_DIMS] and concat
    //     let ctx_expanded = context
    //         .unsqueeze_dim::<3>(1)                          // [batch, 1, ctx]
    //         .expand([batch, seq_len, CONTEXT_DIMS]);         // [batch, seq, ctx]

    //     let lstm_in = Tensor::cat(vec![embeds, ctx_expanded], 2); // [batch, seq, embed+ctx]

    //     // Project (no-op if already right size, but helps with gradient flow)
    //     let lstm_in = self.input_proj.forward(lstm_in);

    //     // LSTM
    //     let (out_seq, out_state) = self.lstm.forward(lstm_in, None);
    //     // let (out_seq, _) = self.lstm2.forward(out_seq, None); // slows things down, but seems to contribute little (with or without out_state)
    //     // out_seq: [batch, seq_len, LSTM_UNITS]

    //     let dropped = self.dropout.forward(out_seq);

    //     let attn_input = MhaInput::self_attn(dropped.clone());
    //     let attended = self.encoder_attention.forward(attn_input);

    //     let combined = attended.context + dropped;

    //     // let shared  = relu(self.dense.forward(dropped)); // [batch, seq, HIDDEN]

    //     // Token logits — every timestep
    //     // let token_logits = self.token_head.forward(shared.clone()); // [batch, seq, vocab]
    //     let token_logits = self.token_head.forward(combined.clone()); // [batch, seq, vocab]

    //     // Emote logits — last timestep only
    //     let last = 
    //         //shared
    //         combined
    //         .slice([0..batch, seq_len - 1..seq_len, 0..HIDDEN_UNITS])
    //         // .slice([0..batch, MAX_SEQ_LEN - 1..MAX_SEQ_LEN, 0..HIDDEN_UNITS])
    //         .reshape([batch, HIDDEN_UNITS]);
    //     let emote_logits = self.yumon_emote_head.forward(last); // [batch, EMOTE]

    //     (token_logits, emote_logits)
    // }

    pub fn forward(
        &self,
        source_tokens: Tensor<B, 2, Int>, // The message received (Input)
        target_tokens: Tensor<B, 2, Int>, // The response to learn (Output)
        context:       Tensor<B, 2>,      // Optional static context
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let [batch, src_len] = source_tokens.dims();
        let [_, trg_len] = target_tokens.dims();

        // println!("far a");

        // // --- 1. ENCODER PHASE ---
        // // Turn the source message into a "thought vector" (State)
        // let src_embeds = self.embedding.forward(source_tokens);
        // // We pass None as initial state, let it process the whole input
        // let (src_seq, src_state) = self.lstm.forward(src_embeds, None);

        // --- 1. ENCODER PHASE ---
        let src_embeds = self.embedding.forward(source_tokens); // [batch, src_len, 128]
        let ctx_expanded = context.clone()
            .unsqueeze_dim::<3>(1)
            .expand([batch, src_len, CONTEXT_DIMS]);             // [batch, src_len, 114]
        let src_input = Tensor::cat(vec![src_embeds, ctx_expanded], 2); // [batch, src_len, 242]
        let src_input = self.input_proj.forward(src_input);
        let (src_seq, src_state) = self.lstm.forward(src_input, None);

        // println!("far b");

        // // --- 2. DECODER PHASE ---
        // // Start the response using the final state of the encoder as the "initial memory"
        // let trg_embeds = self.embedding.forward(target_tokens);
        // // Crucial: Use src_state here to bridge the "similarity" gap
        // let (trg_seq, _trg_state) = self.lstm.forward(trg_embeds, Some(src_state));

        // --- 2. DECODER PHASE ---
        let trg_embeds = self.embedding.forward(target_tokens); // [batch, trg_len, 128]
        let ctx_expanded_trg = context
            .unsqueeze_dim::<3>(1)
            .expand([batch, trg_len, CONTEXT_DIMS]);             // [batch, trg_len, 114]
        let trg_input = Tensor::cat(vec![trg_embeds, ctx_expanded_trg], 2); // [batch, trg_len, 242]
        let trg_input = self.input_proj.forward(trg_input);
        let (trg_seq, _trg_state) = self.lstm.forward(trg_input, Some(src_state));

        // println!("far 1");

        // --- 3. CROSS-ATTENTION ---
        // Replace the MhaInput logic with this direct call:
        let attended = self.encoder_attention.forward(
            trg_seq.clone(), // Query
            src_seq.clone(), // Context (used for both Key and Value internally)
            None             // Mask (we'll start with None to get it running)
        );

        // The rest remains the same
        // let combined = attended + trg_seq;

        // println!("far 2");

        let attended_proj = self.attn_proj.forward(attended); // [batch, seq, LSTM_UNITS]
        let combined = attended_proj + trg_seq;               // both [batch, seq, 512]

        // --- 4. HEADS ---
        let token_logits = self.token_head.forward(combined.clone());

        // For the emote head, take the last timestep of the combined sequence
        let last = combined
            .slice([0..batch, trg_len - 1..trg_len])
            .reshape([batch, LSTM_UNITS]);
            
        let emote_logits = self.yumon_emote_head.forward(last);

        (token_logits, emote_logits)
    }

    /// Autoregressive generation.
    /// Returns a GenerationResult with the reply text and Yumon's emote index.
    // pub fn generate(
    //     &self,
    //     tokenizer:       &TokenizerKind,
    //     class_probs:     &[f32], // for which object detected from cifar-100
    //     emote_probs:     &[f32], // which emotion detected fer2013
    //     user_emote_idx:  usize,
    //     seed_text:       &str,
    //     max_tokens:      usize,
    //     device:          &B::Device,
    // ) -> GenerationResult {
    //     use burn::tensor::activation::log_softmax;

    //     // Build context vector [1, CONTEXT_DIMS]
    //     let mut ctx_flat = Vec::with_capacity(CONTEXT_DIMS);
    //     ctx_flat.extend_from_slice(class_probs);
    //     ctx_flat.extend_from_slice(emote_probs);
    //     // user_emote_onehot
    //     let mut onehot = vec![0.0f32; EMOTE_CLASSES];
    //     onehot[user_emote_idx.min(EMOTE_CLASSES - 1)] = 1.0;
    //     ctx_flat.extend_from_slice(&onehot);

    //     let context_t = Tensor::<B, 2>::from_floats(
    //         TensorData::new(ctx_flat, [1, CONTEXT_DIMS]),
    //         device,
    //     );

    //     // Seed tokens
    //     let mut token_ids: Vec<usize> = vec![BOS_TOKEN];
    //     if !seed_text.is_empty() {
    //         token_ids.extend(tokenizer.encode(seed_text));
    //     }

    //     let mut rng = rand::thread_rng();
    //     let mut last_emote_logits: Option<Vec<f32>> = None;

    //     for _ in 0..max_tokens {
    //         let seq_len = token_ids.len();
    //         let ids_flat: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();

    //         let ids_t = Tensor::<B, 2, burn::tensor::Int>::from_ints(
    //             TensorData::new(ids_flat, [1, seq_len]),
    //             device,
    //         );

    //         let (token_logits, emote_logits) = self.forward(ids_t, context_t.clone());

    //         // Last timestep token logits → [vocab_size]
    //         let vocab = tokenizer.vocab_size();
    //         let last_logits = token_logits
    //             .slice([0..1, seq_len - 1..seq_len, 0..vocab])
    //             .reshape([vocab]);

    //         let logits_vec: Vec<f32> = last_logits.to_data().to_vec().unwrap();

    //         // Save emote logits
    //         last_emote_logits = Some(emote_logits.to_data().to_vec::<f32>().unwrap());

    //         // Temperature + top-k sampling
    //         let next_token = sample_top_k(&logits_vec, TOP_K, TEMPERATURE, &mut rng);
    //         if next_token == EOS_TOKEN || next_token == PAD_TOKEN { break; }
    //         token_ids.push(next_token);

    //         // println!("token_ids after each step: {:?}", token_ids);
    //         // println!("next_token: {}", next_token);
    //         // println!("EOS: {} PAD: {}", EOS_TOKEN, PAD_TOKEN);
    //     }

    //     // Decode reply (skip BOS)
    //     let reply = tokenizer.decode(&token_ids[1..]);

    //     // Yumon emote = argmax of last emote_logits
    //     let yumon_emote_idx = last_emote_logits
    //         .as_deref()
    //         .map(argmax)
    //         .unwrap_or(4); // default neutral

    //     GenerationResult { reply, yumon_emote_idx }
    // }

    pub fn generate(
        &self,
        tokenizer: &TokenizerKind,
        class_probs: &[f32],
        emote_probs: &[f32],
        user_emote_idx: usize,
        seed_text: &str, // This is now our "Encoder Input"
        max_tokens: usize,
        device: &B::Device,
    ) -> GenerationResult {
        // 1. Prepare static context (same as before)
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

        // 2. Encode the Source (User Message)
        // Even if seed_text is empty, we need at least a PAD or BOS to keep the tensor dims valid
        let source_ids = if seed_text.is_empty() { vec![PAD_TOKEN as i32] } 
                        else { tokenizer.encode(seed_text).into_iter().map(|t| t as i32).collect() };
        
        let source_len = source_ids.len();
        let source_t = Tensor::<B, 2, Int>::from_ints(
            TensorData::new(source_ids, [1, source_len]),
            device,
        );

        // 3. Setup Target Generation
        let mut token_ids: Vec<usize> = vec![BOS_TOKEN]; // The response starts here
        let mut rng = rand::thread_rng();
        let mut last_emote_logits: Option<Vec<f32>> = None;

        for _ in 0..max_tokens {
            let current_trg_len = token_ids.len();
            // let trg_flat: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();

            // let trg_t = Tensor::<B, 2, Int>::from_ints(
            //     TensorData::new(trg_flat, [1, current_trg_len]),
            //     device,
            // );

            let clamped_len = current_trg_len.min(MAX_SEQ_LEN);
            let pad_len = MAX_SEQ_LEN - clamped_len;

            let mut padded = token_ids[..clamped_len].to_vec();
            padded.extend(vec![PAD_TOKEN; pad_len]);

            let trg_t = Tensor::<B, 2, Int>::from_ints(
                TensorData::new(
                    padded.iter().map(|&t| t as i32).collect(),
                    [1, MAX_SEQ_LEN]
                ),
                device,
            );

            // --- NEW FORWARD CALL ---
            // Pass source (input) and current target (response so far)
            let (token_logits, emote_logits) = self.forward(
                source_t.clone(), 
                trg_t, 
                context_t.clone()
            );

            // Get logits for the LAST token of the generated sequence
            let vocab = tokenizer.vocab_size();
            // let last_logits = token_logits
            //     .slice([0..1, current_trg_len - 1..current_trg_len, 0..vocab])
            //     .reshape([vocab]);

            let last_idx = clamped_len - 1;
            let last_logits = token_logits
                .slice([0..1, last_idx..last_idx + 1, 0..vocab])
                .reshape([vocab]);

            let logits_vec: Vec<f32> = last_logits.to_data().to_vec().unwrap();
            last_emote_logits = Some(emote_logits.to_data().to_vec::<f32>().unwrap());

            // Sampling
            let next_token = sample_top_k(&logits_vec, TOP_K, TEMPERATURE, &mut rng);
            
            if next_token == EOS_TOKEN || next_token == PAD_TOKEN { break; }
            token_ids.push(next_token);
        }

        // Decode reply (everything after the initial BOS)
        let reply = tokenizer.decode(&token_ids[1..]);
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

        println!("✅ Brain checkpoint saved → {directory}");
        Ok(())
    }

    pub fn load(directory: &str, device: &B::Device) -> Result<(Self, TokenizerKind)> {
        let dir = std::path::Path::new(directory);

        let meta_json = std::fs::read_to_string(dir.join("metadata.json"))?;
        let metadata: BrainMetadata = serde_json::from_str(&meta_json)?;

        // let tokenizer = Tokenizer::load(dir.join("tokenizer.json").to_str().unwrap())?;

        let use_bpe = false;

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
