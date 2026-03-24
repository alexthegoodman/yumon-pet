// use burn::{
//     nn::{
//         Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, Lstm, LstmConfig, LstmState, attention::{CrossAttention, CrossAttentionConfig, MhaInput, MultiHeadAttention, MultiHeadAttentionConfig}, transformer::{TransformerDecoder, TransformerDecoderConfig, TransformerDecoderInput}
//     },
//     prelude::*,
//     record::{BinFileRecorder, FullPrecisionSettings, Recorder},
//     tensor::{TensorData, activation::sigmoid},
// };
// use anyhow::Result;
// use outlines_core::{index::Index, prelude::Vocabulary};
// use serde::{Serialize, Deserialize};

// use crate::{brain::{bpe::{BpeTokenizer, EOS_ID, TokenizerKind}, fixer::fix_json_syntax, samples::{Action, CardinalDir, WorldContext}, train::MAX_SEQ_LEN}, vision::EMOTE_CLASSES};
// use super::{
//     tokenizer::{Tokenizer, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN},
//     // CONTEXT_DIMS,
// };

// use burn::nn::transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput};

// // pub const EMBED_DIM:    usize = 768;
// // pub const HIDDEN_UNITS: usize = 768;
// // pub const EMBED_DIM:    usize = 512;
// // pub const HIDDEN_UNITS: usize = 512;
// // pub const EMBED_DIM:    usize = 256;
// // pub const HIDDEN_UNITS: usize = 256;
// pub const EMBED_DIM:    usize = 128;
// pub const HIDDEN_UNITS: usize = 128;
// // pub const ATTN_HEADS:   usize = 4;
// pub const ATTN_HEADS:   usize = 2;
// // pub const N_LAYERS:     usize = 1;
// // pub const N_LAYERS:     usize = 4;
// // pub const N_LAYERS:     usize = 3;
// pub const N_LAYERS:     usize = 2;
// // pub const FF_DIM:       usize = 512;
// // pub const FF_DIM:       usize = 256;
// pub const FF_DIM:       usize = 256;
// // pub const FF_DIM:       usize = 1024;
// // pub const FF_DIM:       usize = 2048;

// // pub const TEMPERATURE: f32  = 0.7;
// pub const TEMPERATURE: f32  = 0.95;
// pub const TOP_K:       usize = 10;

// // ── CONTEXT_DIMS update ────────────────────────────────────────────────────────
// // Change this constant wherever it's defined:
// pub const CONTEXT_DIMS: usize = 132;  // was 114, +18 for world spatial context

// pub const YUMON_SCHEMA: &str = r#"{
//     "type": "object",
//     "properties": {
//         "action":     { "type": "string", "enum": ["speak","build","travel","idle","use"] },
//         "motion_dir": { "type": "string", "enum": ["north","south","east","west","none"] },
//         "reply":      { "type": "string" }
//     },
//     "required": ["action", "motion_dir", "reply"]
// }"#;

// #[derive(Module, Debug)]
// pub struct YumonBrain<B: Backend> {
//     // Encoder side
//     enc_embedding:    Embedding<B>,
//     enc_pos_embedding: Embedding<B>,
//     encoder:          TransformerEncoder<B>,
//     enc_norm:         LayerNorm<B>,

//     // Context projection — injects context_vec into encoder memory
//     context_proj:     Linear<B>,

//     // Decoder side
//     dec_embedding:    Embedding<B>,
//     dec_pos_embedding: Embedding<B>,
//     decoder:          TransformerDecoder<B>,
//     dec_norm:         LayerNorm<B>,

//     dropout:          Dropout,
//     token_head:       Linear<B>,
//     yumon_emote_head: Linear<B>,
// }

// #[derive(Config, Debug)]
// pub struct YumonBrainConfig {
//     pub vocab_size:   usize,
//     #[config(default = 0.05)]
//     pub dropout_rate: f64,
// }

// impl YumonBrainConfig {
//     pub fn init<B: Backend>(&self, device: &B::Device) -> YumonBrain<B> {
//         let encoder_config = TransformerEncoderConfig::new(
//             EMBED_DIM,
//             FF_DIM,
//             ATTN_HEADS,
//             N_LAYERS,
//         )
//         .with_norm_first(true)
//         .with_quiet_softmax(true)
//         .with_dropout(self.dropout_rate);

//         let decoder_config = TransformerDecoderConfig::new(  // ← changed
//             EMBED_DIM,
//             FF_DIM,
//             ATTN_HEADS,
//             N_LAYERS,
//         )
//         .with_norm_first(true)
//         .with_quiet_softmax(true)
//         .with_dropout(self.dropout_rate);

//         YumonBrain {
//             enc_embedding:        EmbeddingConfig::new(self.vocab_size, EMBED_DIM).init(device),
//             enc_pos_embedding:    EmbeddingConfig::new(MAX_SEQ_LEN, EMBED_DIM).init(device),
//             encoder:                encoder_config.init(device),
//             enc_norm:             LayerNormConfig::new(EMBED_DIM).init(device),

//             context_proj:       LinearConfig::new(CONTEXT_DIMS, EMBED_DIM).init(device),

//             dec_embedding:        EmbeddingConfig::new(self.vocab_size, EMBED_DIM).init(device),
//             dec_pos_embedding:    EmbeddingConfig::new(MAX_SEQ_LEN, EMBED_DIM).init(device),
//             decoder:                decoder_config.init(device),
//             dec_norm:             LayerNormConfig::new(EMBED_DIM).init(device),

//             dropout:          DropoutConfig::new(self.dropout_rate).init(),
//             token_head:       LinearConfig::new(EMBED_DIM, self.vocab_size).init(device),
//             yumon_emote_head: LinearConfig::new(EMBED_DIM, EMOTE_CLASSES).init(device),
//         }
//     }
// }

// impl<B: Backend> YumonBrain<B> {
//     pub fn encode(
//         &self,
//         enc_tokens:  Tensor<B, 2, Int>,
//         context_vec: Tensor<B, 2>,
//     ) -> Tensor<B, 3> {
//         let [batch, enc_len] = enc_tokens.dims();

//         let tok_emb = self.enc_embedding.forward(enc_tokens.clone());
//         let positions = make_positions(batch, enc_len, &tok_emb.device());
//         let pos_emb = self.enc_pos_embedding.forward(positions);
//         let x = self.dropout.forward(tok_emb + pos_emb);

//         let enc_pad_mask = enc_tokens.equal_elem(PAD_TOKEN as u32);
//         let enc_input = TransformerEncoderInput::new(x)
//             .mask_pad(enc_pad_mask);
//         let memory = self.encoder.forward(enc_input);
//         let memory = self.enc_norm.forward(memory);

//         // Inject context
//         let ctx_projected: Tensor<B, 3> = self.context_proj.forward(context_vec)
//             .unsqueeze_dim(1);
//         let ctx_projected = ctx_projected
//             .expand([batch, enc_len, EMBED_DIM]);

//         memory + ctx_projected
//     }

//     pub fn decode(
//         &self,
//         dec_tokens: Tensor<B, 2, Int>,
//         memory:     Tensor<B, 3>,
//     ) -> (Tensor<B, 3>, Tensor<B, 2>) {
//         let [batch, dec_len] = dec_tokens.dims();

//         let tok_emb = self.dec_embedding.forward(dec_tokens.clone());
//         let positions = make_positions(batch, dec_len, &tok_emb.device());
//         let pos_emb = self.dec_pos_embedding.forward(positions);
//         let x = self.dropout.forward(tok_emb + pos_emb);

//         let dec_causal_mask = Tensor::<B, 3, Bool>::tril_mask(
//             [batch, dec_len, dec_len], 0, &x.device()
//         );
//         let dec_pad_mask = dec_tokens.equal_elem(PAD_TOKEN as u32);

//         let dec_input = TransformerDecoderInput::new(x, memory)
//             .target_mask_attn(dec_causal_mask)
//             .target_mask_pad(dec_pad_mask);
//         let x = self.decoder.forward(dec_input);
//         let x = self.dec_norm.forward(x);

//         let token_logits = self.token_head.forward(x.clone());

//         let last = x
//             .slice([0..batch, dec_len - 1..dec_len])
//             .reshape([batch, EMBED_DIM]);
//         let emote_logits = self.yumon_emote_head.forward(last);

//         (token_logits, emote_logits)
//     }

//     pub fn forward(
//         &self,
//         enc_tokens:  Tensor<B, 2, Int>,
//         dec_tokens:  Tensor<B, 2, Int>,
//         context_vec: Tensor<B, 2>,
//     ) -> (Tensor<B, 3>, Tensor<B, 2>) {
//         let memory = self.encode(enc_tokens, context_vec);
//         self.decode(dec_tokens, memory)
//     }

use burn::{
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig, RotaryEncoding, RotaryEncodingConfig,
    },
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::activation::sigmoid,
};
use anyhow::Result;
use outlines_core::{index::Index, prelude::Vocabulary};
use serde::{Serialize, Deserialize};

use crate::{
    brain::{
        bpe::{BpeTokenizer, EOS_ID, TokenizerKind},
        fixer::fix_json_syntax,
        samples::{Action, CardinalDir, WorldContext},
        train::MAX_SEQ_LEN,
    },
    vision::EMOTE_CLASSES,
};
use super::tokenizer::{Tokenizer, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN};

// ── Model dimensions ──────────────────────────────────────────────────────────
pub const EMBED_DIM:    usize = 128;
pub const HIDDEN_UNITS: usize = 128;
// pub const EMBED_DIM:    usize = 256;
// pub const HIDDEN_UNITS: usize = 256;
pub const ATTN_HEADS:   usize = 2;
pub const N_LAYERS:     usize = 2;
pub const FF_DIM:       usize = 256;
// pub const FF_DIM:       usize = 512;

pub const TEMPERATURE:  f32   = 0.95;
pub const TOP_K:        usize = 10;

pub const CONTEXT_DIMS: usize = 132;

pub const YUMON_SCHEMA: &str = r#"{
    "type": "object",
    "properties": {
        "action":     { "type": "string", "enum": ["speak","build","travel","idle","use"] },
        "motion_dir": { "type": "string", "enum": ["north","south","east","west","none"] },
        "reply":      { "type": "string" }
    },
    "required": ["action", "motion_dir", "reply"]
}"#;


// ═════════════════════════════════════════════════════════════════════════════
// RMSNorm
// ═════════════════════════════════════════════════════════════════════════════

#[derive(Config, Debug)]
pub struct RMSNormConfig {
    size: usize,
    #[config(default = 1e-6)]
    eps: f64,
}

impl RMSNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RMSNorm<B> {
        let weight = burn::module::Param::from_tensor(Tensor::ones([self.size], device));
        RMSNorm { weight, eps: self.eps }
    }
}

#[derive(Module, Debug)]
pub struct RMSNorm<B: Backend> {
    weight: burn::module::Param<Tensor<B, 1>>,
    eps: f64,
}

impl<B: Backend> RMSNorm<B> {
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let rms = (x.clone().powf_scalar(2.0).mean_dim(D - 1) + self.eps).sqrt();
        (x / rms) * self.weight.val().unsqueeze()
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// SiLU activation
// ═════════════════════════════════════════════════════════════════════════════

#[derive(Module, Clone, Debug)]
pub struct SiLU {}

impl SiLU {
    pub fn new() -> Self { Self {} }
    pub fn forward<B: Backend, const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        x.clone() * sigmoid(x)
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// SwiGLU MLP  (same gated structure as Llama)
// ═════════════════════════════════════════════════════════════════════════════

#[derive(Config, Debug)]
pub struct MLPConfig {
    d_model:    usize,
    d_hidden:   usize,
}

impl MLPConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MLP<B> {
        MLP {
            w1:   LinearConfig::new(self.d_model, self.d_hidden).with_bias(false).init(device),
            w2:   LinearConfig::new(self.d_hidden, self.d_model).with_bias(false).init(device),
            w3:   LinearConfig::new(self.d_model, self.d_hidden).with_bias(false).init(device),
            silu: SiLU::new(),
        }
    }
}

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    w1:   Linear<B>,
    w2:   Linear<B>,
    w3:   Linear<B>,
    silu: SiLU,
}

impl<B: Backend> MLP<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // SwiGLU: w2( silu(w1(x)) * w3(x) )
        self.w2.forward(self.silu.forward(self.w1.forward(x.clone())) * self.w3.forward(x))
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Causal mask helper
// ═════════════════════════════════════════════════════════════════════════════

pub fn causal_mask<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 2> {
    use std::f32::NEG_INFINITY;
    Tensor::full([seq_len, seq_len], NEG_INFINITY, device).triu(1)
}


// ═════════════════════════════════════════════════════════════════════════════
// Multi-Head Self-Attention  (with RoPE, optional causal mask, optional pad mask)
// ═════════════════════════════════════════════════════════════════════════════

#[derive(Config, Debug)]
pub struct SelfAttentionConfig {
    d_model:  usize,
    n_heads:  usize,
}

impl SelfAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SelfAttention<B> {
        assert!(self.d_model % self.n_heads == 0);
        let head_dim = self.d_model / self.n_heads;
        SelfAttention {
            q: LinearConfig::new(self.d_model, self.d_model).with_bias(false).init(device),
            k: LinearConfig::new(self.d_model, self.d_model).with_bias(false).init(device),
            v: LinearConfig::new(self.d_model, self.d_model).with_bias(false).init(device),
            o: LinearConfig::new(self.d_model, self.d_model).with_bias(false).init(device),
            n_heads: self.n_heads,
            head_dim,
        }
    }
}

#[derive(Module, Debug)]
pub struct SelfAttention<B: Backend> {
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    o: Linear<B>,
    n_heads:  usize,
    head_dim: usize,
}

impl<B: Backend> SelfAttention<B> {
    /// x:          [batch, seq, d_model]
    /// rope:       shared RotaryEncoding
    /// causal_mask: optional [seq, seq] additive mask (NEG_INFINITY upper triangle)
    /// pad_mask:   optional [batch, seq] bool — true where padded
    pub fn forward(
        &self,
        x:           Tensor<B, 3>,
        rope:        &RotaryEncoding<B>,
        causal_mask: Option<Tensor<B, 2>>,
        pad_mask:    Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let [batch, seq, _] = x.dims();
        let scale = (self.head_dim as f64).powf(-0.25);

        let reshape = |t: Tensor<B, 3>| {
            t.reshape([batch, seq, self.n_heads, self.head_dim])
             .swap_dims(1, 2)          // [batch, heads, seq, head_dim]
        };

        let q = rope.forward(reshape(self.q.forward(x.clone()))) * scale;
        let k = rope.forward(reshape(self.k.forward(x.clone()))) * scale;
        let v =              reshape(self.v.forward(x));

        let mut qk = q.matmul(k.transpose()); // [batch, heads, seq, seq]

        // additive causal mask
        if let Some(mask) = causal_mask {
            qk = qk + mask.slice([0..seq, 0..seq]).unsqueeze::<4>();
        }

        // padding mask: set attention to -inf where key is a pad token
        if let Some(pmask) = pad_mask {
            // pmask: [batch, seq] → [batch, 1, 1, seq]
            let pmask_f = pmask
                .unsqueeze_dim::<3>(1)
                .unsqueeze_dim::<4>(2)
                .float()
                .mul_scalar(-1e9);
            qk = qk + pmask_f;
        }

        let w = burn::tensor::activation::softmax(qk, 3);
        let out = w.matmul(v)              // [batch, heads, seq, head_dim]
            .swap_dims(1, 2)               // [batch, seq, heads, head_dim]
            .flatten(2, 3);                // [batch, seq, d_model]

        self.o.forward(out)
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// Cross-Attention  (Q from decoder, K/V from encoder memory)
// ═════════════════════════════════════════════════════════════════════════════

#[derive(Config, Debug)]
pub struct CrossAttentionConfig {
    d_model: usize,
    n_heads: usize,
}

impl CrossAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CrossAttentionBlock<B> {
        assert!(self.d_model % self.n_heads == 0);
        let head_dim = self.d_model / self.n_heads;
        CrossAttentionBlock {
            q: LinearConfig::new(self.d_model, self.d_model).with_bias(false).init(device),
            k: LinearConfig::new(self.d_model, self.d_model).with_bias(false).init(device),
            v: LinearConfig::new(self.d_model, self.d_model).with_bias(false).init(device),
            o: LinearConfig::new(self.d_model, self.d_model).with_bias(false).init(device),
            n_heads: self.n_heads,
            head_dim,
        }
    }
}

#[derive(Module, Debug)]
pub struct CrossAttentionBlock<B: Backend> {
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    o: Linear<B>,
    n_heads:  usize,
    head_dim: usize,
}

impl<B: Backend> CrossAttentionBlock<B> {
    /// x:      [batch, dec_len, d_model]  — decoder states
    /// memory: [batch, enc_len, d_model]  — encoder output
    /// rope:   shared RotaryEncoding (applied to both Q and K)
    /// enc_pad_mask: optional [batch, enc_len] bool — true where encoder input was padded
    pub fn forward(
        &self,
        x:            Tensor<B, 3>,
        memory:       Tensor<B, 3>,
        rope:         &RotaryEncoding<B>,
        enc_pad_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let [batch, dec_len, _] = x.dims();
        let [_, enc_len, _]     = memory.dims();
        let scale = (self.head_dim as f64).powf(-0.25);

        let reshape_dec = |t: Tensor<B, 3>| {
            t.reshape([batch, dec_len, self.n_heads, self.head_dim])
             .swap_dims(1, 2)
        };
        let reshape_enc = |t: Tensor<B, 3>| {
            t.reshape([batch, enc_len, self.n_heads, self.head_dim])
             .swap_dims(1, 2)
        };

        // Q from decoder, K/V from memory; RoPE on Q and K
        let q = rope.forward(reshape_dec(self.q.forward(x)))      * scale;
        let k = rope.forward(reshape_enc(self.k.forward(memory.clone()))) * scale;
        let v =              reshape_enc(self.v.forward(memory));

        let mut qk = q.matmul(k.transpose()); // [batch, heads, dec_len, enc_len]

        if let Some(pmask) = enc_pad_mask {
            let pmask_f = pmask
                .unsqueeze_dim::<3>(1)
                .unsqueeze_dim::<4>(2)
                .float()
                .mul_scalar(-1e9);
            qk = qk + pmask_f;
        }

        let w = burn::tensor::activation::softmax(qk, 3);
        let out = w.matmul(v)
            .swap_dims(1, 2)
            .flatten(2, 3);

        self.o.forward(out)
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// Encoder Block
//   pre-norm → self-attn → residual
//   pre-norm → MLP       → residual
// ═════════════════════════════════════════════════════════════════════════════

#[derive(Config, Debug)]
pub struct EncoderBlockConfig {
    d_model:  usize,
    d_hidden: usize,
    n_heads:  usize,
}

impl EncoderBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> EncoderBlock<B> {
        EncoderBlock {
            attn_norm: RMSNormConfig::new(self.d_model).init(device),
            attn:      SelfAttentionConfig::new(self.d_model, self.n_heads).init(device),
            mlp_norm:  RMSNormConfig::new(self.d_model).init(device),
            mlp:       MLPConfig::new(self.d_model, self.d_hidden).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct EncoderBlock<B: Backend> {
    attn_norm: RMSNorm<B>,
    attn:      SelfAttention<B>,
    mlp_norm:  RMSNorm<B>,
    mlp:       MLP<B>,
}

impl<B: Backend> EncoderBlock<B> {
    pub fn forward(
        &self,
        x:        Tensor<B, 3>,
        rope:     &RotaryEncoding<B>,
        pad_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        // self-attention with residual
        let x = x.clone()
            + self.attn.forward(self.attn_norm.forward(x), rope, None, pad_mask);
        // MLP with residual
        x.clone() + self.mlp.forward(self.mlp_norm.forward(x))
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// Decoder Block
//   pre-norm → masked self-attn  → residual
//   pre-norm → cross-attn        → residual
//   pre-norm → MLP               → residual
// ═════════════════════════════════════════════════════════════════════════════

#[derive(Config, Debug)]
pub struct DecoderBlockConfig {
    d_model:  usize,
    d_hidden: usize,
    n_heads:  usize,
}

impl DecoderBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DecoderBlock<B> {
        DecoderBlock {
            self_attn_norm:  RMSNormConfig::new(self.d_model).init(device),
            self_attn:       SelfAttentionConfig::new(self.d_model, self.n_heads).init(device),
            cross_attn_norm: RMSNormConfig::new(self.d_model).init(device),
            cross_attn:      CrossAttentionConfig::new(self.d_model, self.n_heads).init(device),
            mlp_norm:        RMSNormConfig::new(self.d_model).init(device),
            mlp:             MLPConfig::new(self.d_model, self.d_hidden).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct DecoderBlock<B: Backend> {
    self_attn_norm:  RMSNorm<B>,
    self_attn:       SelfAttention<B>,
    cross_attn_norm: RMSNorm<B>,
    cross_attn:      CrossAttentionBlock<B>,
    mlp_norm:        RMSNorm<B>,
    mlp:             MLP<B>,
}

impl<B: Backend> DecoderBlock<B> {
    pub fn forward(
        &self,
        x:            Tensor<B, 3>,
        memory:       Tensor<B, 3>,
        rope:         &RotaryEncoding<B>,
        causal_mask:  Tensor<B, 2>,
        dec_pad_mask: Option<Tensor<B, 2, Bool>>,
        enc_pad_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        // masked self-attention
        let x = x.clone()
            + self.self_attn.forward(
                self.self_attn_norm.forward(x),
                rope,
                Some(causal_mask),
                dec_pad_mask,
            );
        // cross-attention
        let x = x.clone()
            + self.cross_attn.forward(
                self.cross_attn_norm.forward(x.clone()),
                memory,
                rope,
                enc_pad_mask,
            );
        // MLP
        x.clone() + self.mlp.forward(self.mlp_norm.forward(x))
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// YumonBrain
// ═════════════════════════════════════════════════════════════════════════════

#[derive(Module, Debug)]
pub struct YumonBrain<B: Backend> {
    // Shared RoPE (one instance, passed by reference to all blocks)
    rope: RotaryEncoding<B>,

    // Encoder
    enc_embedding: Embedding<B>,
    enc_blocks:    Vec<EncoderBlock<B>>,
    enc_norm:      RMSNorm<B>,

    // Context injection
    context_proj:  Linear<B>,

    // Decoder
    dec_embedding: Embedding<B>,
    dec_blocks:    Vec<DecoderBlock<B>>,
    dec_norm:      RMSNorm<B>,

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
        let head_dim = EMBED_DIM / ATTN_HEADS;

        // let rope = RotaryEncodingConfig::new(MAX_SEQ_LEN, head_dim)
        //     .init(device);

        let rope = RotaryEncodingConfig::new(MAX_SEQ_LEN, EMBED_DIM / ATTN_HEADS).init(device);

        let enc_blocks = (0..N_LAYERS)
            .map(|_| EncoderBlockConfig::new(EMBED_DIM, FF_DIM, ATTN_HEADS).init(device))
            .collect();

        let dec_blocks = (0..N_LAYERS)
            .map(|_| DecoderBlockConfig::new(EMBED_DIM, FF_DIM, ATTN_HEADS).init(device))
            .collect();

        YumonBrain {
            rope,

            enc_embedding: EmbeddingConfig::new(self.vocab_size, EMBED_DIM).init(device),
            enc_blocks,
            enc_norm:      RMSNormConfig::new(EMBED_DIM).init(device),

            context_proj:  LinearConfig::new(CONTEXT_DIMS, EMBED_DIM).init(device),

            dec_embedding: EmbeddingConfig::new(self.vocab_size, EMBED_DIM).init(device),
            dec_blocks,
            dec_norm:      RMSNormConfig::new(EMBED_DIM).init(device),

            dropout:          DropoutConfig::new(self.dropout_rate).init(),
            token_head:       LinearConfig::new(EMBED_DIM, self.vocab_size).init(device),
            yumon_emote_head: LinearConfig::new(EMBED_DIM, EMOTE_CLASSES).init(device),
        }
    }
}

impl<B: Backend> YumonBrain<B> {
    // ── Encoder ──────────────────────────────────────────────────────────────
    pub fn encode(
        &self,
        enc_tokens:  Tensor<B, 2, Int>,
        context_vec: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let enc_tokens_cl = enc_tokens.clone();
        
        let [batch, enc_len] = enc_tokens.dims();

        let pad_mask = enc_tokens.equal_elem(PAD_TOKEN as u32); // [batch, enc_len]

        let mut x = self.dropout.forward(
            self.enc_embedding.forward(enc_tokens_cl)
        );
        // NOTE: No positional embedding — RoPE handles position inside attention

        // Context injection: project context_vec and broadcast across sequence
        // let ctx: Tensor<B, 3> = self.context_proj.forward(context_vec)
        //     .unsqueeze_dim(1)
        //     .expand([batch, enc_len, EMBED_DIM]);
        let ctx: Tensor<B, 3> = self.context_proj.forward(context_vec)
            .unsqueeze_dim(1);
        let ctx = ctx
            .expand([batch, enc_len, EMBED_DIM]);

        x = x + ctx;

        for block in &self.enc_blocks {
            x = block.forward(x, &self.rope, Some(pad_mask.clone()));
        }

        self.enc_norm.forward(x)
    }

    // ── Decoder ──────────────────────────────────────────────────────────────
    pub fn decode(
        &self,
        dec_tokens:  Tensor<B, 2, Int>,
        memory:      Tensor<B, 3>,
        enc_pad_mask: Option<Tensor<B, 2, Bool>>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let dec_tokens_cl = dec_tokens.clone();

        let [batch, dec_len] = dec_tokens.dims();

        let dec_pad_mask = dec_tokens.equal_elem(PAD_TOKEN as u32);

        let mut x = self.dropout.forward(
            self.dec_embedding.forward(dec_tokens_cl)
        );

        let cmask = causal_mask::<B>(dec_len, &x.device());

        for block in &self.dec_blocks {
            x = block.forward(
                x,
                memory.clone(),
                &self.rope,
                cmask.clone(),
                Some(dec_pad_mask.clone()),
                enc_pad_mask.clone(),
            );
        }

        let x = self.dec_norm.forward(x);

        let token_logits = self.token_head.forward(x.clone());

        // Emote head — use last non-pad decoder position
        let last = x
            .slice([0..batch, dec_len - 1..dec_len])
            .reshape([batch, EMBED_DIM]);
        let emote_logits = self.yumon_emote_head.forward(last);

        (token_logits, emote_logits)
    }

    // ── Forward ───────────────────────────────────────────────────────────────
    pub fn forward(
        &self,
        enc_tokens:  Tensor<B, 2, Int>,
        dec_tokens:  Tensor<B, 2, Int>,
        context_vec: Tensor<B, 2>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let enc_tokens_cl = enc_tokens.clone();
        let enc_pad_mask = enc_tokens.equal_elem(PAD_TOKEN as u32);
        let memory = self.encode(enc_tokens_cl, context_vec);
        self.decode(dec_tokens, memory, Some(enc_pad_mask))
    }

    // ── Startup: build outlines Vocabulary from BPE tokenizer ─────────────────

    pub fn build_outlines_vocabulary(tokenizer: &BpeTokenizer) -> Vocabulary {
        let hf_vocab = tokenizer.inner.get_vocab(true);  // String → u32
        let mut vocab = Vocabulary::new(EOS_ID);
        for (token_str, &token_id) in &hf_vocab {
            // println!("token {:?} {:?}", token_id, token_str);
            vocab.try_insert(token_str.as_str(), token_id).ok();
        }
        // println!("vocab size for outlines {:?}", vocab.len());
        vocab
    }

    pub fn build_outlines_index(
        tokenizer: &BpeTokenizer,
        schema: &str,
    ) -> anyhow::Result<Index> {
        let regex = outlines_core::json_schema::regex_from_str(schema, None, None)
            .map_err(|e| anyhow::anyhow!("schema → regex: {e}"))?;
        let vocab = Self::build_outlines_vocabulary(tokenizer);
        let index = Index::new(&regex, &vocab)
            .map_err(|e| anyhow::anyhow!("Index::new: {e}"))?;

        // println!("index: {:?}", index.vocab_size());

        Ok(index)
    }

    // pub fn generate_structured(
    //     &self,
    //     tokenizer:      &TokenizerKind,
    //     index:          &Index,
    //     world:          &WorldContext,
    //     class_probs:    &[f32],
    //     emote_probs:    &[f32],
    //     user_emote_idx: usize,
    //     seed_text:      &str,
    //     max_tokens:     usize,
    //     device:         &B::Device,
    // ) -> GenerationResult {
    //     // ── Build context tensor ───────────────────────────────────────────────
    //     let mut ctx_flat = Vec::with_capacity(CONTEXT_DIMS);
    //     ctx_flat.extend_from_slice(class_probs);
    //     ctx_flat.extend_from_slice(emote_probs);
    //     let mut onehot = vec![0.0f32; EMOTE_CLASSES];
    //     onehot[user_emote_idx.min(EMOTE_CLASSES - 1)] = 1.0;
    //     ctx_flat.extend_from_slice(&onehot);
    //     ctx_flat.extend_from_slice(&world.to_context_slice());

    //     let context_t = Tensor::<B, 2>::from_floats(
    //         TensorData::new(ctx_flat, [1, CONTEXT_DIMS]),
    //         device,
    //     );

    //     // ── Encode input once ──────────────────────────────────────────────────
    //     let enc_ids: Vec<i32> = {
    //         let mut ids = vec![BOS_TOKEN as i32];
    //         if !seed_text.is_empty() {
    //             ids.extend(tokenizer.encode(seed_text).iter().map(|&t| t as i32));
    //         }
    //         ids.resize(MAX_SEQ_LEN, PAD_TOKEN as i32);
    //         ids
    //     };

    //     let enc_tokens_t = Tensor::<B, 2, Int>::from_ints(
    //         TensorData::new(enc_ids, [1, MAX_SEQ_LEN]),
    //         device,
    //     );

    //     let memory = self.encode(enc_tokens_t, context_t);  // [1, enc_len, embed] — run once

    //     // ── Decoder state ──────────────────────────────────────────────────────
    //     let mut dec_ids: Vec<usize> = vec![BOS_TOKEN];  // decoder starts with just BOS
    //     let mut fsm_state = index.initial_state();

    //     let mut rng = rand::thread_rng();
    //     let mut last_emote_logits: Option<Vec<f32>> = None;
    //     let mut debug_allowed_count: Option<usize> = None;
    //     let mut debug_fsm_state = fsm_state;

    //     for _ in 0..max_tokens {
    //         // ── Build decoder input tensor from current dec_ids ────────────────
    //         let clamped_len = dec_ids.len().min(MAX_SEQ_LEN);
    //         let mut padded = dec_ids[dec_ids.len() - clamped_len..].to_vec();
    //         padded.resize(MAX_SEQ_LEN, PAD_TOKEN);

    //         let dec_tokens_t = Tensor::<B, 2, Int>::from_ints(
    //             TensorData::new(
    //                 padded.iter().map(|&t| t as i32).collect::<Vec<_>>(),
    //                 [1, MAX_SEQ_LEN],
    //             ),
    //             device,
    //         );

    //         // ── Decode ────────────────────────────────────────────────────────
    //         let (token_logits, emote_logits) = self.decode(dec_tokens_t, memory.clone());

    //         last_emote_logits = Some(emote_logits.to_data().to_vec::<f32>().unwrap());

    //         let vocab_size  = tokenizer.vocab_size();
    //         let last_logits = token_logits
    //             .slice([0..1, clamped_len - 1..clamped_len, 0..vocab_size])
    //             .reshape([vocab_size]);

    //         let mut logits_vec: Vec<f32> = last_logits.to_data().to_vec().unwrap();

    //         // ── Apply FSM mask ─────────────────────────────────────────────────
    //         let allowed = index.allowed_tokens(&fsm_state);
    //         if debug_allowed_count.is_none() {
    //             debug_allowed_count = Some(allowed.as_ref().map(|a| a.len()).unwrap_or(0));
    //         }
    //         debug_fsm_state = fsm_state;

    //         if let Some(allowed) = allowed {
    //             let mut masked = vec![f32::NEG_INFINITY; logits_vec.len()];
    //             for &token_id in &allowed {
    //                 let idx = token_id as usize;
    //                 if idx < masked.len() {
    //                     masked[idx] = logits_vec[idx];
    //                 }
    //             }
    //             logits_vec = masked;
    //         }

    //         // ── Sample ────────────────────────────────────────────────────────
    //         let next_token = sample_top_k(&logits_vec, TOP_K, TEMPERATURE, &mut rng);

    //         // ── Advance FSM ───────────────────────────────────────────────────
    //         fsm_state = index
    //             .next_state(&fsm_state, &(next_token as u32))
    //             .unwrap_or(fsm_state);

    //         dec_ids.push(next_token);

    //         if index.final_states().contains(&fsm_state) { break; }
    //         if next_token == EOS_TOKEN || next_token == PAD_TOKEN { break; }
    //     }

    //     // ── Decode tokens → string (skip BOS) ─────────────────────────────────
    //     let raw_output = tokenizer.decode(&dec_ids[1..]);

    //     // let parsed: serde_json::Value =
    //     //     serde_json::from_str(&raw_output).unwrap_or(serde_json::json!({
    //     //         "action":     "idle",
    //     //         "motion_dir": "none",
    //     //         "reply":      ""
    //     //     }));

    //     let parsed: serde_json::Value = serde_json::from_str(&raw_output)
    //         .or_else(|_| serde_json::from_str(&fix_json_syntax(&raw_output).fixed))
    //         .unwrap_or_else(|_| {
    //             let extract = |key: &str| -> String {
    //                 regex::Regex::new(&format!(r#""{key}"\s*:\s*"([^"]*)"#))
    //                     .ok()
    //                     .and_then(|re| re.captures(&raw_output))
    //                     .and_then(|caps| caps.get(1))
    //                     .map(|m| m.as_str().to_string())
    //                     .unwrap_or_default()
    //             };

    //             serde_json::json!({
    //                 "action":     extract("action"),
    //                 "motion_dir": extract("motion_dir"),
    //                 "reply":      extract("reply"),
    //             })
    //         });

    //     let action = match parsed["action"].as_str().unwrap_or("idle") {
    //         "speak"  => Action::Speak,
    //         "build"  => Action::Build,
    //         "travel" => Action::Travel,
    //         "use"    => Action::Use,
    //         _        => Action::Idle,
    //     };

    //     let motion_dir = match parsed["motion_dir"].as_str().unwrap_or("none") {
    //         "north" => CardinalDir::North,
    //         "south" => CardinalDir::South,
    //         "east"  => CardinalDir::East,
    //         "west"  => CardinalDir::West,
    //         _       => CardinalDir::None,
    //     };

    //     let reply = parsed["reply"]
    //         .as_str()
    //         .unwrap_or("")
    //         .to_string();

    //     let yumon_emote_idx = last_emote_logits
    //         .as_deref()
    //         .map(argmax)
    //         .unwrap_or(4);

    //     GenerationResult {
    //         reply,
    //         action,
    //         motion_dir,
    //         yumon_emote_idx,
    //         raw_output,
    //         fsm_state: debug_fsm_state,
    //         allowed_count: debug_allowed_count,
    //     }
    // }

    pub fn generate_unmasked_parsed(
        &self,
        tokenizer:      &TokenizerKind,
        world:          &WorldContext,
        class_probs:    &[f32],
        emote_probs:    &[f32],
        user_emote_idx: usize,
        seed_text:      &str,
        max_tokens:     usize,
        device:         &B::Device,
    ) -> GenerationResult {
        // ── Build context tensor ───────────────────────────────────────────────
        let mut ctx_flat = Vec::with_capacity(CONTEXT_DIMS);
        ctx_flat.extend_from_slice(class_probs);
        ctx_flat.extend_from_slice(emote_probs);
        let mut onehot = vec![0.0f32; EMOTE_CLASSES];
        onehot[user_emote_idx.min(EMOTE_CLASSES - 1)] = 1.0;
        ctx_flat.extend_from_slice(&onehot);
        ctx_flat.extend_from_slice(&world.to_context_slice());

        let context_t = Tensor::<B, 2>::from_floats(
            TensorData::new(ctx_flat, [1, CONTEXT_DIMS]),
            device,
        );

        // ── Encode input once ──────────────────────────────────────────────────
        let enc_ids: Vec<i32> = {
            let mut ids = vec![BOS_TOKEN as i32];
            if !seed_text.is_empty() {
                ids.extend(tokenizer.encode(seed_text).iter().map(|&t| t as i32));
            }
            ids.resize(MAX_SEQ_LEN, PAD_TOKEN as i32);
            ids
        };

        let enc_tokens_t = Tensor::<B, 2, Int>::from_ints(
            TensorData::new(enc_ids, [1, MAX_SEQ_LEN]),
            device,
        );

        let enc_tokens_t_cl = enc_tokens_t.clone();

        let enc_pad_mask = enc_tokens_t.equal_elem(PAD_TOKEN as u32);

        let memory = self.encode(enc_tokens_t_cl, context_t);  // run once

        // ── Decode autoregressively — no FSM masking ───────────────────────────
        let mut dec_ids: Vec<usize> = vec![BOS_TOKEN];
        let mut rng = rand::thread_rng();
        let mut last_emote_logits: Option<Vec<f32>> = None;

        for _ in 0..max_tokens {
            let clamped_len = dec_ids.len().min(MAX_SEQ_LEN);
            let mut padded = dec_ids[dec_ids.len() - clamped_len..].to_vec();
            padded.resize(MAX_SEQ_LEN, PAD_TOKEN);

            let dec_tokens_t = Tensor::<B, 2, Int>::from_ints(
                TensorData::new(
                    padded.iter().map(|&t| t as i32).collect::<Vec<_>>(),
                    [1, MAX_SEQ_LEN],
                ),
                device,
            );

            let (token_logits, emote_logits) = self.decode(dec_tokens_t, memory.clone(), Some(enc_pad_mask.clone()));
            last_emote_logits = Some(emote_logits.to_data().to_vec::<f32>().unwrap());

            let vocab_size  = tokenizer.vocab_size();
            let last_logits = token_logits
                .slice([0..1, clamped_len - 1..clamped_len, 0..vocab_size])
                .reshape([vocab_size]);

            // no masking — pure model output
            let logits_vec: Vec<f32> = last_logits.to_data().to_vec().unwrap();
            let next_token = sample_top_k(&logits_vec, TOP_K, TEMPERATURE, &mut rng);

            if next_token == EOS_TOKEN || next_token == PAD_TOKEN { break; }
            dec_ids.push(next_token);
        }

        // ── Decode tokens → string (skip BOS) ─────────────────────────────────
        let raw_output = tokenizer.decode(&dec_ids[1..]);

        let fixed = fix_json_syntax(&raw_output).fixed;

        let extract = |key: &str| -> String {
            regex::Regex::new(&format!(r#"\s*{key}"?\s*:?\s*"([^"]*)"#))
                .ok()
                .and_then(|re| re.captures(&fixed))
                .and_then(|caps| caps.get(1))
                .map(|m| m.as_str().to_string())
                .unwrap_or_default()
        };

        let parsed_action     = extract("action");
        let parsed_motion_dir = extract("motion_dir");
        let mut parsed_reply  = extract("reply");

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
                        "action":     extract("action"),
                        "motion_dir": extract("motion_dir"),
                        "reply":      extract("reply"),
                    })
                });

            parsed_reply = parsed["reply"].to_string();
        }

        let action = match parsed_action.as_str() {
            "speak"  => Action::Speak,
            "build"  => Action::Build,
            "travel" => Action::Travel,
            "use"    => Action::Use,
            _        => Action::Idle,
        };

        let motion_dir = match parsed_motion_dir.as_str() {
            "north" => CardinalDir::North,
            "south" => CardinalDir::South,
            "east"  => CardinalDir::East,
            "west"  => CardinalDir::West,
            _       => CardinalDir::None,
        };

        let reply = parsed_reply
            .as_str()
            .to_string();

        let yumon_emote_idx = last_emote_logits
            .as_deref()
            .map(argmax)
            .unwrap_or(4);

        GenerationResult {
            reply,
            action,
            motion_dir,
            yumon_emote_idx,
            raw_output,
            fsm_state: 0,
            allowed_count: None,
        }
    }

    // good for debugging raw json output as well as TrainingStage::Language
    pub fn generate_unmasked(
        &self,
        tokenizer:      &TokenizerKind,
        world:          &WorldContext,
        class_probs:    &[f32],
        emote_probs:    &[f32],
        user_emote_idx: usize,
        seed_text:      &str,
        max_tokens:     usize,
        device:         &B::Device,
    ) -> GenerationResult {
        // ── Build context tensor ───────────────────────────────────────────────
        let mut ctx_flat = Vec::with_capacity(CONTEXT_DIMS);
        ctx_flat.extend_from_slice(class_probs);
        ctx_flat.extend_from_slice(emote_probs);
        let mut onehot = vec![0.0f32; EMOTE_CLASSES];
        onehot[user_emote_idx.min(EMOTE_CLASSES - 1)] = 1.0;
        ctx_flat.extend_from_slice(&onehot);
        ctx_flat.extend_from_slice(&world.to_context_slice());

        let context_t = Tensor::<B, 2>::from_floats(
            TensorData::new(ctx_flat, [1, CONTEXT_DIMS]),
            device,
        );

        // ── Encode input once ──────────────────────────────────────────────────
        let enc_ids: Vec<i32> = {
            let mut ids = vec![BOS_TOKEN as i32];
            if !seed_text.is_empty() {
                ids.extend(tokenizer.encode(seed_text).iter().map(|&t| t as i32));
            }
            ids.resize(MAX_SEQ_LEN, PAD_TOKEN as i32);
            ids
        };

        let enc_tokens_t = Tensor::<B, 2, Int>::from_ints(
            TensorData::new(enc_ids, [1, MAX_SEQ_LEN]),
            device,
        );

        let enc_tokens_t_cl = enc_tokens_t.clone();

        let enc_pad_mask = enc_tokens_t.equal_elem(PAD_TOKEN as u32);

        let memory = self.encode(enc_tokens_t_cl, context_t);  // run once

        // ── Decode autoregressively — no FSM masking ───────────────────────────
        let mut dec_ids: Vec<usize> = vec![BOS_TOKEN];
        let mut rng = rand::thread_rng();
        let mut last_emote_logits: Option<Vec<f32>> = None;

        for _ in 0..max_tokens {
            let clamped_len = dec_ids.len().min(MAX_SEQ_LEN);
            let mut padded = dec_ids[dec_ids.len() - clamped_len..].to_vec();
            padded.resize(MAX_SEQ_LEN, PAD_TOKEN);

            let dec_tokens_t = Tensor::<B, 2, Int>::from_ints(
                TensorData::new(
                    padded.iter().map(|&t| t as i32).collect::<Vec<_>>(),
                    [1, MAX_SEQ_LEN],
                ),
                device,
            );

            let (token_logits, emote_logits) = self.decode(dec_tokens_t, memory.clone(), Some(enc_pad_mask.clone()));
            last_emote_logits = Some(emote_logits.to_data().to_vec::<f32>().unwrap());

            let vocab_size  = tokenizer.vocab_size();
            let last_logits = token_logits
                .slice([0..1, clamped_len - 1..clamped_len, 0..vocab_size])
                .reshape([vocab_size]);

            // no masking — pure model output
            let logits_vec: Vec<f32> = last_logits.to_data().to_vec().unwrap();
            let next_token = sample_top_k(&logits_vec, TOP_K, TEMPERATURE, &mut rng);

            if next_token == EOS_TOKEN || next_token == PAD_TOKEN { break; }
            dec_ids.push(next_token);
        }

        // ── Decode tokens → string (skip BOS) ─────────────────────────────────
        let raw_output = tokenizer.decode(&dec_ids[1..]);

        let parsed: serde_json::Value = serde_json::from_str(&raw_output)
            .unwrap_or(serde_json::json!({
                "action": "idle", "motion_dir": "none", "reply": ""
            }));

        GenerationResult {
            reply:           parsed["reply"].as_str().unwrap_or("").to_string(),
            action:          Action::Speak,
            motion_dir:      CardinalDir::None,
            yumon_emote_idx: last_emote_logits.as_deref().map(argmax).unwrap_or(4),
            raw_output,
            fsm_state:       0,
            allowed_count:   None,
        }
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
}

fn make_positions<B: Backend>(
    batch: usize,
    seq_len: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    Tensor::from_ints(
        TensorData::new(
            (0..seq_len as i32).cycle().take(batch * seq_len).collect::<Vec<_>>(),
            [batch, seq_len],
        ),
        device,
    )
}

// ─── Generation Result ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub reply:           String,
    pub action:          Action,
    pub motion_dir:      CardinalDir,
    pub yumon_emote_idx: usize,
    pub raw_output: String,
    pub fsm_state:       u32,      // ← final FSM state
    pub allowed_count:   Option<usize>, // ← None if masking never fired
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
