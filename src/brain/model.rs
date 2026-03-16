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

// //  have 4096 vocab size during bpe training currently
// // pub const EMBED_DIM:   usize = 64;
// pub const EMBED_DIM:   usize = 128; // smaller
// // pub const EMBED_DIM:  usize = 512;
// // pub const EMBED_DIM: usize = 512; // bigger

// // pub const LSTM_UNITS:  usize = 2048;
// // pub const HIDDEN_UNITS: usize = 2048;
// // pub const LSTM_UNITS:  usize = 1024; // medium-large
// // pub const HIDDEN_UNITS: usize = 1024; 
// pub const LSTM_UNITS:  usize = 512;
// pub const HIDDEN_UNITS: usize = 512;
// // pub const LSTM_UNITS:  usize = 256;
// // pub const HIDDEN_UNITS: usize = 256;

// // pub const ATTN_HEADS: usize = 4;
// pub const ATTN_HEADS:  usize = 64;       // medium
// // pub const ATTN_HEADS:  usize = 16;       // was 4
// // pub const ATTN_HEAD_DIM: usize = 64;    // was LSTM_UNITS/ATTN_HEADS = 64 — same, but now explicit
// pub const ATTN_HEAD_DIM: usize = 128;  // 8 heads × 128 = 1024 total attn capacity (total should be close to LSTM_UNITS)
// // pub const ATTN_HEAD_DIM: usize = 256;  // 8 heads × 256 = 2048 total attn capacity
// // pub const ATTN_HEAD_DIM: usize = 512;

// pub const TEMPERATURE: f32   = 0.7;
// pub const TOP_K:       usize = 10;

// // ─── Model ───────────────────────────────────────────────────────────────────

// #[derive(Module, Debug)]
// pub struct YumonBrain<B: Backend> {
//     embedding:       Embedding<B>,
//     // We project (embed + context) → lstm_input_dim before the LSTM
//     // because Burn's LSTM input size must be fixed at init.
//     // input_proj:      Linear<B>,
//     // lstm:            Lstm<B>,

//     enc_input_proj:     Linear<B>,   // was: input_proj
//     dec_input_proj:     Linear<B>,   // new
//     enc_lstm:           Lstm<B>,     // was: lstm
//     dec_lstm:           Lstm<B>,     // new
//     enc_norm:           LayerNorm<B>,  // new
//     dec_norm:           LayerNorm<B>,  // new

//     // lstm2:            Lstm<B>,
//     dropout:         Dropout,
//     // dense:           Linear<B>,
//     // encoder_attention: MultiHeadAttention<B>,
//     encoder_attention: CrossAttention<B>,
//     attn_proj:  Linear<B>,
//     token_head:      Linear<B>,
//     yumon_emote_head: Linear<B>,
// }

// #[derive(Config, Debug)]
// pub struct YumonBrainConfig {
//     pub vocab_size:  usize,
//     #[config(default = 0.3)]
//     pub dropout_rate: f64,
// }

// impl YumonBrainConfig {
//     pub fn init<B: Backend>(&self, device: &B::Device) -> YumonBrain<B> {
//         let lstm_in = EMBED_DIM + CONTEXT_DIMS;
//         println!("lstm_in usize {:?}", lstm_in);
//         YumonBrain {
//             embedding:        EmbeddingConfig::new(self.vocab_size, EMBED_DIM).init(device),
//             // input_proj:       LinearConfig::new(lstm_in, lstm_in).init(device),
//             // lstm:             LstmConfig::new(lstm_in, LSTM_UNITS, false).init(device),

//             enc_input_proj: LinearConfig::new(lstm_in, lstm_in).init(device),
//             dec_input_proj: LinearConfig::new(lstm_in, lstm_in).init(device),
//             enc_lstm:       LstmConfig::new(lstm_in, LSTM_UNITS, false).init(device),
//             dec_lstm:       LstmConfig::new(lstm_in, LSTM_UNITS, false).init(device),

//             enc_norm: LayerNormConfig::new(LSTM_UNITS).init(device),
//             dec_norm: LayerNormConfig::new(LSTM_UNITS).init(device),

//             // lstm2:             LstmConfig::new(LSTM_UNITS, LSTM_UNITS, false).init(device),
//             dropout:          DropoutConfig::new(self.dropout_rate).init(),
//             // encoder_attention: MultiHeadAttentionConfig::new(LSTM_UNITS, ATTN_HEADS).init(device),
//             // encoder_attention: CrossAttentionConfig::new(LSTM_UNITS, LSTM_UNITS, ATTN_HEADS, 1, LSTM_UNITS / ATTN_HEADS).with_quiet_softmax(true).init(device),
//             encoder_attention: CrossAttentionConfig::new(
//                 LSTM_UNITS,       // query dim  (from LSTM)
//                 LSTM_UNITS,       // key/value dim (context)
//                 ATTN_HEADS,       // 8 heads
//                 1,
//                 ATTN_HEAD_DIM,    // now independently tunable
//             )
//             .with_quiet_softmax(false)
//             .init(device),
//             // dense:            LinearConfig::new(LSTM_UNITS, HIDDEN_UNITS).init(device),
//             // token_head:       LinearConfig::new(HIDDEN_UNITS, self.vocab_size).init(device),
//             // yumon_emote_head: LinearConfig::new(HIDDEN_UNITS, EMOTE_CLASSES).init(device),
//             attn_proj: LinearConfig::new(LSTM_UNITS, HIDDEN_UNITS).init(device),
//             token_head:       LinearConfig::new(LSTM_UNITS, self.vocab_size).init(device),
//             yumon_emote_head: LinearConfig::new(LSTM_UNITS, EMOTE_CLASSES).init(device),
//         }
//     }
// }

pub const EMBED_DIM:     usize = 128;
pub const LSTM_UNITS:    usize = 512;
pub const HIDDEN_UNITS:  usize = 512;
pub const ATTN_HEADS:    usize = 8;
pub const ATTN_HEAD_DIM: usize = 64;   // LSTM_UNITS / ATTN_HEADS

pub const TEMPERATURE: f32  = 0.7;
pub const TOP_K:       usize = 10;

#[derive(Module, Debug)]
pub struct YumonBrain<B: Backend> {
    embedding:  Embedding<B>,
    input_proj: Linear<B>,
    lstm:       Lstm<B>,
    norm:       LayerNorm<B>,
    dropout:    Dropout,
    attention:  MultiHeadAttention<B>,  // used as self-attention (q=k=v)
    attn_proj:  Linear<B>,
    token_head: Linear<B>,
    yumon_emote_head: Linear<B>,
}

#[derive(Config, Debug)]
pub struct YumonBrainConfig {
    pub vocab_size:   usize,
    #[config(default = 0.3)]
    pub dropout_rate: f64,
}

impl YumonBrainConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> YumonBrain<B> {
        let lstm_in = EMBED_DIM + CONTEXT_DIMS;
        YumonBrain {
            embedding:  EmbeddingConfig::new(self.vocab_size, EMBED_DIM).init(device),
            input_proj: LinearConfig::new(lstm_in, lstm_in).init(device),
            lstm:       LstmConfig::new(lstm_in, LSTM_UNITS, false).init(device),
            norm:       LayerNormConfig::new(LSTM_UNITS).init(device),
            dropout:    DropoutConfig::new(self.dropout_rate).init(),
            // attention:  CrossAttentionConfig::new(
            //                 LSTM_UNITS,
            //                 LSTM_UNITS,
            //                 ATTN_HEADS,
            //                 1,
            //                 ATTN_HEAD_DIM,
            //             )
            //             .with_quiet_softmax(false)
            //             .init(device),
            attention:  MultiHeadAttentionConfig::new(
                            LSTM_UNITS,
                            ATTN_HEADS,
                        )
                        .with_quiet_softmax(false)
                        .init(device),
            attn_proj:  LinearConfig::new(LSTM_UNITS, LSTM_UNITS).init(device),
            token_head: LinearConfig::new(LSTM_UNITS, self.vocab_size).init(device),
            yumon_emote_head: LinearConfig::new(LSTM_UNITS, EMOTE_CLASSES).init(device),
        }
    }
}

// impl<B: Backend> YumonBrain<B> {
//     pub fn forward(
//         &self,
//         source_tokens: Tensor<B, 2, Int>, // The message received (Input)
//         target_tokens: Tensor<B, 2, Int>, // The response to learn (Output)
//         context:       Tensor<B, 2>,      // Optional static context
//     ) -> (Tensor<B, 3>, Tensor<B, 2>) {
//         let [batch, src_len] = source_tokens.dims();
//         let [_, trg_len] = target_tokens.dims();

//         // --- 1. ENCODER PHASE ---
//         let src_embeds = self.embedding.forward(source_tokens); // [batch, src_len, 128]
//         let ctx_expanded = context.clone()
//             .unsqueeze_dim::<3>(1)
//             .expand([batch, src_len, CONTEXT_DIMS]);             // [batch, src_len, 114]
//         let src_input = Tensor::cat(vec![src_embeds, ctx_expanded], 2); // [batch, src_len, 242]
//         // let src_input = self.input_proj.forward(src_input);
//         // let (src_seq, src_state) = self.lstm.forward(src_input, None);
//         let src_input = self.enc_input_proj.forward(src_input);
//         let (src_seq, src_state) = self.enc_lstm.forward(src_input, None);
//         let src_seq = self.enc_norm.forward(src_seq);
        
//         // println!("src_seq mean={:.5}", src_seq.clone().mean().into_scalar());

//         // --- 2. DECODER PHASE ---
//         let trg_embeds = self.embedding.forward(target_tokens); // [batch, trg_len, 128]
//         let ctx_expanded_trg = context
//             .unsqueeze_dim::<3>(1)
//             .expand([batch, trg_len, CONTEXT_DIMS]);             // [batch, trg_len, 114]
//         let trg_input = Tensor::cat(vec![trg_embeds, ctx_expanded_trg], 2); // [batch, trg_len, 242]
//         // let trg_input = self.input_proj.forward(trg_input);
//         // let (trg_seq, _trg_state) = self.lstm.forward(trg_input, Some(src_state));
//         let trg_input = self.dec_input_proj.forward(trg_input);
//         let (trg_seq, _) = self.dec_lstm.forward(trg_input, Some(src_state));
//         let trg_seq = self.dec_norm.forward(trg_seq);

//         // println!("trg_seq mean={:.5}", trg_seq.clone().mean().into_scalar());

//         // println!("trg_seq min={:.5} max={:.5}", 
//         //     trg_seq.clone().min().into_scalar(),
//         //     trg_seq.clone().max().into_scalar());
//         // println!("src_seq min={:.5} max={:.5}", 
//         //     src_seq.clone().min().into_scalar(),
//         //     src_seq.clone().max().into_scalar());

//         // println!("trg_seq std={:.5}", trg_seq.clone().var(2).mean().into_scalar());
//         // println!("src_seq std={:.5}", src_seq.clone().var(2).mean().into_scalar());

//         // --- 3. CROSS-ATTENTION ---
//         // Replace the MhaInput logic with this direct call:
//         let attended = self.encoder_attention.forward(
//             trg_seq.clone(), // Query
//             src_seq.clone(), // Context (used for both Key and Value internally)
//             None             // Mask (we'll start with None to get it running)
//         );

//         // --- DIAGNOSTICS ---
//         // let src_data = src_seq.clone().mean().into_scalar();
//         // let trg_data = trg_seq.clone().mean().into_scalar();
//         // let att_data = attended.clone().mean().into_scalar();
//         // let att_std  = attended.clone().var(0).mean().into_scalar();
        
//         // println!("src_seq   mean: {:.5}", src_data);
//         // println!("trg_seq   mean: {:.5}", trg_data);
//         // println!("attended  mean: {:.5}", att_data);
//         // println!("attended  std:  {:.5}", att_std);

//         // Check if attended is just copying trg_seq
//         // let diff = (attended.clone() - trg_seq.clone()).abs().mean().into_scalar();
//         // println!("attended vs trg_seq diff: {:.5}", diff);

//         // // Check if attended is just copying src_seq (wrong)
//         // let diff2 = (attended.clone() - src_seq.clone().slice([0..batch, 0..trg_len])).abs().mean().into_scalar();
//         // println!("attended vs src_seq diff: {:.5}", diff2);
//         // -------------------

//         let attended_proj = self.attn_proj.forward(attended); // [batch, seq, LSTM_UNITS]

//         // println!("attended_proj mean={:.5}", attended_proj.clone().mean().into_scalar());

//         // let combined = attended_proj + trg_seq;               // both [batch, seq, 512]
//         let combined = attended_proj + trg_seq;               // both [batch, seq, 512]

//         // --- 4. HEADS ---
//         let token_logits = self.token_head.forward(combined.clone());

//         // For the emote head, take the last timestep of the combined sequence
//         let last = combined
//             .slice([0..batch, trg_len - 1..trg_len])
//             .reshape([batch, LSTM_UNITS]);
            
//         let emote_logits = self.yumon_emote_head.forward(last);

//         (token_logits, emote_logits)
//     }

//     pub fn generate(
//         &self,
//         tokenizer: &TokenizerKind,
//         class_probs: &[f32],
//         emote_probs: &[f32],
//         user_emote_idx: usize,
//         seed_text: &str, // This is now our "Encoder Input"
//         max_tokens: usize,
//         device: &B::Device,
//     ) -> GenerationResult {
//         // 1. Prepare static context (same as before)
//         let mut ctx_flat = Vec::with_capacity(CONTEXT_DIMS);
//         ctx_flat.extend_from_slice(class_probs);
//         ctx_flat.extend_from_slice(emote_probs);
//         let mut onehot = vec![0.0f32; EMOTE_CLASSES];
//         onehot[user_emote_idx.min(EMOTE_CLASSES - 1)] = 1.0;
//         ctx_flat.extend_from_slice(&onehot);

//         let context_t = Tensor::<B, 2>::from_floats(
//             TensorData::new(ctx_flat, [1, CONTEXT_DIMS]),
//             device,
//         );

//         // 2. Encode the Source (User Message)
//         // Even if seed_text is empty, we need at least a PAD or BOS to keep the tensor dims valid
//         let source_ids = if seed_text.is_empty() { vec![PAD_TOKEN as i32] } 
//                         else { tokenizer.encode(seed_text).into_iter().map(|t| t as i32).collect() };
        
//         let source_len = source_ids.len();
//         let source_t = Tensor::<B, 2, Int>::from_ints(
//             TensorData::new(source_ids, [1, source_len]),
//             device,
//         );

//         // 3. Setup Target Generation
//         let mut token_ids: Vec<usize> = vec![BOS_TOKEN]; // The response starts here
//         let mut rng = rand::thread_rng();
//         let mut last_emote_logits: Option<Vec<f32>> = None;

//         for _ in 0..max_tokens {
//             let current_trg_len = token_ids.len();
//             // let trg_flat: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();

//             // let trg_t = Tensor::<B, 2, Int>::from_ints(
//             //     TensorData::new(trg_flat, [1, current_trg_len]),
//             //     device,
//             // );

//             let clamped_len = current_trg_len.min(MAX_SEQ_LEN);
//             let pad_len = MAX_SEQ_LEN - clamped_len;

//             let mut padded = token_ids[..clamped_len].to_vec();
//             padded.extend(vec![PAD_TOKEN; pad_len]);

//             let trg_t = Tensor::<B, 2, Int>::from_ints(
//                 TensorData::new(
//                     padded.iter().map(|&t| t as i32).collect(),
//                     [1, MAX_SEQ_LEN]
//                 ),
//                 device,
//             );

//             // --- NEW FORWARD CALL ---
//             // Pass source (input) and current target (response so far)
//             let (token_logits, emote_logits) = self.forward(
//                 source_t.clone(), 
//                 trg_t, 
//                 context_t.clone()
//             );

//             // Get logits for the LAST token of the generated sequence
//             let vocab = tokenizer.vocab_size();
//             // let last_logits = token_logits
//             //     .slice([0..1, current_trg_len - 1..current_trg_len, 0..vocab])
//             //     .reshape([vocab]);

//             let last_idx = clamped_len - 1;
//             let last_logits = token_logits
//                 .slice([0..1, last_idx..last_idx + 1, 0..vocab])
//                 .reshape([vocab]);

//             let logits_vec: Vec<f32> = last_logits.to_data().to_vec().unwrap();
//             last_emote_logits = Some(emote_logits.to_data().to_vec::<f32>().unwrap());

//             // Sampling
//             let next_token = sample_top_k(&logits_vec, TOP_K, TEMPERATURE, &mut rng);
            
//             if next_token == EOS_TOKEN || next_token == PAD_TOKEN { break; }
//             token_ids.push(next_token);
//         }

//         // Decode reply (everything after the initial BOS)
//         let reply = tokenizer.decode(&token_ids[1..]);
//         let yumon_emote_idx = last_emote_logits.as_deref().map(argmax).unwrap_or(4);

//         GenerationResult { reply, yumon_emote_idx }
//     }

// Build a causal mask [seq_len, seq_len] — upper triangle = -inf
// fn causal_mask<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 2, Bool> {
//     // 1. Create a range tensor [0, 1, 2, ..., seq_len-1]
//     let index = Tensor::<B, 1, Int>::arange(0..seq_len as i64, device);
    
//     // 2. Reshape to allow broadcasting: 
//     // row_indices: [seq_len, 1]
//     // col_indices: [1, seq_len]
//     let row_indices = index.clone().reshape([seq_len, 1]);
//     let col_indices = index.reshape([1, seq_len]);

//     // 3. Create the mask: True where col <= row
//     // This gives you the lower triangular matrix (including diagonal)
//     row_indices.greater_equal(col_indices)
// }

impl<B: Backend> YumonBrain<B> {
    /// Completion-style forward pass.
    /// tokens:  [batch, seq_len]        — the input token sequence
    /// context: [batch, CONTEXT_DIMS]   — static vision/emote context
    /// returns: token_logits [batch, seq_len, vocab], emote_logits [batch, EMOTE_CLASSES]
    pub fn forward(
        &self,
        tokens:  Tensor<B, 2, Int>,
        context: Tensor<B, 2>,
        prev_state: Option<LstmState<B, 2>>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>, LstmState<B, 2>) {
        let [batch, seq_len] = tokens.dims();

        // 1. Embed + concat context
        let embeds = self.embedding.forward(tokens);             // [batch, seq, 128]
        let ctx = context
            .unsqueeze_dim::<3>(1)
            .expand([batch, seq_len, CONTEXT_DIMS]);             // [batch, seq, 114]
        let x = Tensor::cat(vec![embeds, ctx], 2);              // [batch, seq, 242]

        // 2. Input projection + LSTM
        // println!("see 1");
        let x = self.input_proj.forward(x);
        let (x, x_state) = self.lstm.forward(x, prev_state);                // [batch, seq, 512]
        let x = self.norm.forward(x);
        let x = self.dropout.forward(x);

        // 3. Self-attention (query = key = value = same sequence)
        // println!("see 2");
        // let base_mask = causal_mask::<B>(seq_len, &x.device());
        let mask_attn = Tensor::<B, 3, Bool>::tril_mask([1, seq_len, seq_len], 0, &x.device());

        // let mask_attn = base_mask
        //                                         .unsqueeze_dim::<3>(0)
        //                                         .expand([batch, seq_len, seq_len]); // [batch, seq, seq]

        // println!("see 3");
        // let mha_input = MhaInput::new(query, key, value) // each a Tensor<B, 3>
        //                                         .mask_attn(mask_attn) // Tensor<B, 3, Bool>
        //                                         .mask_pad(mask_pad); // Tensor<B, 2, Bool>

        let mha_input = MhaInput::new(x.clone(), x.clone(), x.clone())
                                                    .mask_attn(mask_attn);

        let attended = self.attention.forward(mha_input);
        // println!("see 4");
        let x = self.attn_proj.forward(attended.context) + x;           // residual

        // 4. Heads
        let token_logits = self.token_head.forward(x.clone());  // [batch, seq, vocab]

        let last = x
            .slice([0..batch, seq_len - 1..seq_len])
            .reshape([batch, LSTM_UNITS]);
        let emote_logits = self.yumon_emote_head.forward(last);

        (token_logits, emote_logits, x_state)
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
        println!("Generate");

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
        let mut last_state = None;

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
                self.forward(tokens_t, context_t.clone(), last_state);

            last_state = Some(lstm_state);

            // Sample from the last real position
            let last_idx  = clamped_len - 1;
            let vocab     = tokenizer.vocab_size();
            let last_logits = token_logits
                .slice([0..1, last_idx..last_idx + 1, 0..vocab])
                .reshape([vocab]);

            let logits_vec: Vec<f32> = last_logits.to_data().to_vec().unwrap();
            last_emote_logits = Some(emote_logits.to_data().to_vec::<f32>().unwrap());

            let next_token = sample_top_k(&logits_vec, TOP_K, TEMPERATURE, &mut rng);
            println!("Check token {:?}", next_token);
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
