use burn::{
    backend::Wgpu, module::Ignored, nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig, RotaryEncoding, RotaryEncodingConfig
    }, prelude::*, record::{BinFileRecorder, FullPrecisionSettings, Recorder}, tensor::activation::sigmoid
};
use anyhow::Result;
use cubecl::wgpu::{AutoGraphicsApi, GraphicsApi, WebGpu, WgpuDevice, init_setup_async};
use log::Level;
use log::info;

#[cfg(target_os = "windows")]
use outlines_core::{index::Index, prelude::Vocabulary};

use serde::{Serialize, Deserialize};
use burn::record::{BinBytesRecorder};

use super::tokenizer::{Tokenizer, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN};
use crate::{
    brain::{
        bpe::{BpeTokenizer, EOS_ID, TokenizerKind}, 
        fixer::fix_json_syntax, samples::{Action, CardinalDir, TrainingStage, WorldContext}, train::MAX_SEQ_LEN
    },
    vision::EMOTE_CLASSES,
};

// use crate::brain::classic_attn::{DecoderBlock, DecoderBlockConfig, EncoderBlock, EncoderBlockConfig, causal_mask};
use crate::brain::flash_attn::attention::{DecoderBlock, DecoderBlockConfig, EncoderBlock, EncoderBlockConfig, causal_mask};

pub const TEMPERATURE:  f32   = 0.9;
// pub const TEMPERATURE:  f32   = 0.75;
// pub const TEMPERATURE:  f32   = 0.25;
pub const TOP_K:        usize = 10;

// pub const CONTEXT_DIMS: usize = 132;

pub const YUMON_SCHEMA_INPUT: &str = r#"{
    "type": "object",
    "properties": {
        "obstacle_dir": { "type": "string", "enum": ["north","south","east","west","none"] },
        "building_dir": { "type": "string", "enum": ["north","south","east","west","none"] },
        "resource_dir": { "type": "string", "enum": ["north","south","east","west","none"] },
        "message":      { "type": "string" },
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "yumon":   { "type": "string" },
                    "human": { "type": "string" }
                },
                "required": ["yumon", "human"]
            }
        }
    },
    "required": ["obstacle_dir", "building_dir", "resource_dir", "message", "memories"]
}"#;

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
// "Soft" Mixture of Experts
// ═════════════════════════════════════════════════════════════════════════════

#[derive(Config, Debug)]
pub struct MoEConfig {
    d_model:    usize,
    d_hidden:   usize,
    n_experts:  usize,
    #[config(default = 2)]
    top_k:      usize,
}

impl MoEConfig {
    pub fn init<B: Backend<Device = WgpuDevice>>(&self, device: &B::Device) -> MoE<B> {
        let experts = (0..self.n_experts)
            .map(|_| MLPConfig::new(self.d_model, self.d_hidden).init(device))
            .collect();
        MoE {
            experts,
            router: LinearConfig::new(self.d_model, self.n_experts)
                .with_bias(false)
                .init(device),
            n_experts: self.n_experts,
            top_k: self.top_k,
        }
    }
}

#[derive(Module, Debug)]
pub struct MoE<B: Backend<Device = WgpuDevice>> {
    experts:   Vec<MLP<B>>,
    router:    Linear<B>,
    n_experts: usize,
    top_k:     usize,
}

impl<B: Backend<Device = WgpuDevice>> MoE<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, d_model] = x.dims();

        // Router logits → softmax weights
        // [batch, seq, n_experts]
        let logits = self.router.forward(x.clone());
        let weights = burn::tensor::activation::softmax(logits, 2);

        // Run ALL experts, then mask — avoids dynamic indexing
        // [batch, seq, d_model] for each expert
        let expert_outputs: Vec<Tensor<B, 3>> = self.experts
            .iter()
            .map(|e| e.forward(x.clone()))
            .collect();

        // Stack → [batch, seq, n_experts, d_model]
        let stacked = Tensor::stack(expert_outputs, 2);

        // Weighted sum over experts
        // weights: [batch, seq, n_experts] → [batch, seq, n_experts, 1]
        let w = weights.unsqueeze_dim::<4>(3);

        // [batch, seq, n_experts, d_model] * [batch, seq, n_experts, 1]
        // → sum over expert dim → [batch, seq, d_model]
        // (stacked * w).sum_dim(2).squeeze(2)
        (stacked * w).sum_dim(2).squeeze::<3>()
    }
}

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
    pub fn init<B: Backend<Device = WgpuDevice>>(&self, device: &B::Device) -> RMSNorm<B> {
        let weight = burn::module::Param::from_tensor(Tensor::ones([self.size], device));
        RMSNorm { weight, eps: self.eps }
    }
}

#[derive(Module, Debug)]
pub struct RMSNorm<B: Backend<Device = WgpuDevice>> {
    weight: burn::module::Param<Tensor<B, 1>>,
    eps: f64,
}

impl<B: Backend<Device = WgpuDevice>> RMSNorm<B> {
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
// Linear Force
// ═════════════════════════════════════════════════════════════════════════════

#[derive(Config, Debug)]
pub struct LinearForceConfig {
    d_model:    usize,
    d_hidden:   usize,
}

impl LinearForceConfig {
    pub fn init<B: Backend<Device = WgpuDevice>>(&self, device: &B::Device) -> MLP<B> {
        MLP {
            w1:   LinearConfig::new(self.d_model, self.d_hidden).with_bias(false).init(device),
            w2:   LinearConfig::new(self.d_hidden, self.d_model).with_bias(false).init(device),
            w3:   LinearConfig::new(self.d_model, self.d_hidden).with_bias(false).init(device),
            silu: SiLU::new(),
        }
    }
}

#[derive(Module, Debug)]
pub struct LinearForce<B: Backend<Device = WgpuDevice>> {
    w1:   Linear<B>,
    w2:   Linear<B>,
    w3:   Linear<B>,
    silu: SiLU,
}

impl<B: Backend<Device = WgpuDevice>> LinearForce<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let one = self.silu.forward(self.w1.forward(x));
        let two = self.silu.forward(self.w2.forward(one));
        let three = self.silu.forward(self.w3.forward(two));

        three
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
    pub fn init<B: Backend<Device = WgpuDevice>>(&self, device: &B::Device) -> MLP<B> {
        MLP {
            w1:   LinearConfig::new(self.d_model, self.d_hidden)
                .with_initializer(Initializer::KaimingUniform {
                    gain: 1.0 / f64::sqrt(3.0),
                    fan_out_only: false,
                })
                .with_bias(false).init(device),
            w2:   LinearConfig::new(self.d_hidden, self.d_model)
                .with_initializer(Initializer::KaimingUniform {
                    gain: 1.0 / f64::sqrt(3.0),
                    fan_out_only: false,
                })
                .with_bias(false).init(device),
            w3:   LinearConfig::new(self.d_model, self.d_hidden)
                .with_initializer(Initializer::KaimingUniform {
                    gain: 1.0 / f64::sqrt(3.0),
                    fan_out_only: false,
                })
                .with_bias(false).init(device),
            silu: SiLU::new(),
        }
    }
}

#[derive(Module, Debug)]
pub struct MLP<B: Backend<Device = WgpuDevice>> {
    w1:   Linear<B>,
    w2:   Linear<B>,
    w3:   Linear<B>,
    silu: SiLU,
}

impl<B: Backend<Device = WgpuDevice>> MLP<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // SwiGLU: w2( silu(w1(x)) * w3(x) )
        self.w2.forward(self.silu.forward(self.w1.forward(x.clone())) * self.w3.forward(x))
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// YumonBrain
// ═════════════════════════════════════════════════════════════════════════════

#[derive(Module, Debug)]
pub struct YumonBrain<B: Backend<Device = WgpuDevice>> {
    pub config: Ignored<YumonBrainConfig>,
    // Shared RoPE (one instance, passed by reference to all blocks)
    rope: RotaryEncoding<B>,

    // // Encoder
    enc_embedding: Embedding<B>,
    enc_blocks:    Vec<EncoderBlock<B>>,
    enc_norm:      RMSNorm<B>,

    // Decoder
    dec_embedding: Embedding<B>,
    dec_blocks:    Vec<DecoderBlock<B>>,
    dec_norm:      RMSNorm<B>,

    dropout:          Dropout,
    pub token_head:       Linear<B>,
}

#[derive(Config, Debug)]
pub struct YumonBrainConfig {
    pub vocab_size:   usize,
    #[config(default = 0.05)]
    pub dropout_rate: f64,
    #[config(default = 256)]
    pub embed_dim:    usize,
    #[config(default = 256)]
    pub hidden_units: usize,
    #[config(default = 2)]
    pub n_layers:     usize,
    #[config(default = 4)]
    pub attn_heads:   usize,
    #[config(default = 1024)]
    pub ff_dim:       usize,
    #[config(default = 320)]
    pub max_seq_len:  usize,
    pub training_stage: TrainingStage
}

impl YumonBrainConfig {
    pub fn init<B: Backend<Device = WgpuDevice>>(&self, device: &B::Device) -> YumonBrain<B> {
        let rope = RotaryEncodingConfig::new(self.max_seq_len, self.embed_dim / self.attn_heads).init(device);

        let enc_blocks = (0..self.n_layers)
            .map(|_| EncoderBlockConfig::new(self.embed_dim, self.ff_dim, self.attn_heads).init(device))
            .collect();

        let dec_blocks = (0..self.n_layers)
            .map(|_| DecoderBlockConfig::new(self.embed_dim, self.ff_dim, self.attn_heads).init(device))
            .collect();

        YumonBrain {
            config: Ignored(self.clone()),
            rope,

            enc_embedding: EmbeddingConfig::new(self.vocab_size, self.embed_dim).init(device),
            enc_blocks,
            enc_norm:      RMSNormConfig::new(self.embed_dim).init(device),

            dec_embedding: EmbeddingConfig::new(self.vocab_size, self.embed_dim).init(device),
            dec_blocks,
            dec_norm:      RMSNormConfig::new(self.embed_dim).init(device),

            dropout:          DropoutConfig::new(self.dropout_rate).init(),
            token_head:       LinearConfig::new(self.embed_dim, self.vocab_size).init(device),
        }
    }
}

impl<B: Backend<Device = WgpuDevice>> YumonBrain<B> {
    // ── Encoder ──────────────────────────────────────────────────────────────
    pub fn encode(
        &self,
        enc_tokens:  Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let enc_tokens_cl = enc_tokens.clone();
        
        let [batch, enc_len] = enc_tokens.dims();

        let pad_mask = enc_tokens.equal_elem(PAD_TOKEN as u32); // [batch, enc_len]

        let mut x = self.dropout.forward(
            self.enc_embedding.forward(enc_tokens_cl)
        );

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
    ) -> Tensor<B, 3> {
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

        token_logits
    }

    // ── Forward ───────────────────────────────────────────────────────────────
    pub fn forward(
        &self,
        enc_tokens:  Tensor<B, 2, Int>,
        dec_tokens:  Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let enc_tokens_cl = enc_tokens.clone();
        let enc_pad_mask = enc_tokens.equal_elem(PAD_TOKEN as u32);
        let memory = self.encode(enc_tokens_cl);
        self.decode(dec_tokens, memory, Some(enc_pad_mask))
    }

    // ── Encoder ──────────────────────────────────────────────────────────────
    pub async fn encode_async(
        &self,
        enc_tokens:  Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let enc_tokens_cl = enc_tokens.clone();
        
        let [batch, enc_len] = enc_tokens.dims();

        let pad_mask = enc_tokens.equal_elem(PAD_TOKEN as u32); // [batch, enc_len]

        let mut x = self.dropout.forward(
            self.enc_embedding.forward(enc_tokens_cl)
        );

        for block in &self.enc_blocks {
            x = block.forward_async(x, &self.rope, Some(pad_mask.clone())).await;
        }

        self.enc_norm.forward(x)
    }

    // ── Decoder ──────────────────────────────────────────────────────────────
    pub async fn decode_async(
        &self,
        dec_tokens:  Tensor<B, 2, Int>,
        memory:      Tensor<B, 3>,
        enc_pad_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let dec_tokens_cl = dec_tokens.clone();

        let [batch, dec_len] = dec_tokens.dims();

        let dec_pad_mask = dec_tokens.equal_elem(PAD_TOKEN as u32);

        let mut x = self.dropout.forward(
            self.dec_embedding.forward(dec_tokens_cl)
        );

        let cmask = causal_mask::<B>(dec_len, &x.device());

        for block in &self.dec_blocks {
            x = block.forward_async(
                x,
                memory.clone(),
                &self.rope,
                cmask.clone(),
                Some(dec_pad_mask.clone()),
                enc_pad_mask.clone(),
            ).await;
        }

        let x = self.dec_norm.forward(x);

        let token_logits = self.token_head.forward(x.clone());

        token_logits
    }

    // ── Forward ───────────────────────────────────────────────────────────────
    pub async fn forward_async(
        &self,
        enc_tokens:  Tensor<B, 2, Int>,
        dec_tokens:  Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let enc_tokens_cl = enc_tokens.clone();
        let enc_pad_mask = enc_tokens.equal_elem(PAD_TOKEN as u32);
        let memory = self.encode_async(enc_tokens_cl).await;
        self.decode_async(dec_tokens, memory, Some(enc_pad_mask)).await
    }

    pub fn generate_unmasked_parsed(
        &self,
        tokenizer:      &TokenizerKind,
        seed_text:      &str,
        max_tokens:     usize,
        device:         &B::Device,
    ) -> GenerationResult {
        // ── Encode input once ──────────────────────────────────────────────────
        let enc_ids: Vec<i32> = {
            let mut ids = vec![BOS_TOKEN as i32];
            if !seed_text.is_empty() {
                ids.extend(tokenizer.encode(seed_text).iter().map(|&t| t as i32));
            }
            ids.resize(self.config.max_seq_len, PAD_TOKEN as i32);
            ids
        };

        let enc_tokens_t = Tensor::<B, 2, Int>::from_ints(
            TensorData::new(enc_ids, [1, self.config.max_seq_len]),
            device,
        );

        let enc_tokens_t_cl = enc_tokens_t.clone();

        let enc_pad_mask = enc_tokens_t.equal_elem(PAD_TOKEN as u32);

        let memory = self.encode(enc_tokens_t_cl);  // run once

        // ── Decode autoregressively — no FSM masking ───────────────────────────
        let mut dec_ids: Vec<usize> = vec![BOS_TOKEN];
        let mut rng = rand::thread_rng();
        let mut last_emote_logits: Option<Vec<f32>> = None;

        for _ in 0..max_tokens {
            let clamped_len = dec_ids.len().min(self.config.max_seq_len);
            let mut padded = dec_ids[dec_ids.len() - clamped_len..].to_vec();
            padded.resize(self.config.max_seq_len, PAD_TOKEN);

            let dec_tokens_t = Tensor::<B, 2, Int>::from_ints(
                TensorData::new(
                    padded.iter().map(|&t| t as i32).collect::<Vec<_>>(),
                    [1, self.config.max_seq_len],
                ),
                device,
            );

            let token_logits = self.decode(dec_tokens_t, memory.clone(), Some(enc_pad_mask.clone()));

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
            fancy_regex::Regex::new(&format!(r#"(?<=\s*"{key}"\s*:\s*)"([^"]*)""#))
                .ok()
                .and_then(|re| re.captures(&fixed).ok().flatten())
                .and_then(|caps| caps.get(1))
                .map(|m| m.as_str().to_string())
                .unwrap_or_default()
        };

        let mut parsed_action     = extract("action");
        let mut parsed_motion_dir = extract("motion_dir");
        let mut parsed_reply  = extract("reply");
        let mut parsed_emotion  = extract("emotion");

        if parsed_action.is_empty() || parsed_action.len() < 3 {
            parsed_action     = extract(" action");
            parsed_motion_dir = extract(" motion_dir");

            if parsed_motion_dir.is_empty() {
                parsed_motion_dir = extract("motiondir");
            }

            if parsed_motion_dir.is_empty() {
                parsed_motion_dir = extract(" motiondir");
            }

            parsed_reply  = extract(" reply");
            parsed_emotion  = extract(" emotion");
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

                    let mut motion_dir = extract("motion_dir");

                    if motion_dir.is_empty() {
                        motion_dir = extract("motiondir");
                    }

                    serde_json::json!({
                        "action":     extract("action"),
                        "motion_dir": motion_dir,
                        "emotion":      extract("emotion"),
                        "reply":      extract("reply"),
                    })
                });

            parsed_action = parsed["action"].to_string().trim().to_string();
            parsed_motion_dir = parsed["motion_dir"].to_string().trim().to_string();
            parsed_reply = parsed["reply"].to_string().trim().to_string();
            parsed_emotion = parsed["emotion"].to_string().trim().to_string();
        }

        parsed_action = parsed_action.replace("\"", "").trim().to_string();
        parsed_motion_dir = parsed_motion_dir.replace("\"", "").trim().to_string();
        parsed_reply = parsed_reply.replace("\"", "").trim().to_string();
        parsed_emotion = parsed_emotion.replace("\"", "").trim().to_string();

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

        GenerationResult {
            reply,
            action,
            motion_dir,
            parsed_emotion,
            raw_output,
            fsm_state: 0,
            allowed_count: None,
        }
    }

    pub async fn generate_unmasked_parsed_async(
        &self,
        tokenizer:      &TokenizerKind,
        seed_text:      &str,
        max_tokens:     usize,
        device:         &B::Device,
    ) -> GenerationResult {
        // ── Encode input once ──────────────────────────────────────────────────
        let enc_ids: Vec<i32> = {
            let mut ids = vec![BOS_TOKEN as i32];
            if !seed_text.is_empty() {
                ids.extend(tokenizer.encode(seed_text).iter().map(|&t| t as i32));
            }
            ids.resize(self.config.max_seq_len, PAD_TOKEN as i32);
            ids
        };

        let enc_tokens_t = Tensor::<B, 2, Int>::from_ints(
            TensorData::new(enc_ids, [1, self.config.max_seq_len]),
            device,
        );

        let enc_tokens_t_cl = enc_tokens_t.clone();

        let enc_pad_mask = enc_tokens_t.equal_elem(PAD_TOKEN as u32);

        let memory = self.encode_async(enc_tokens_t_cl).await;  // run once

        // ── Decode autoregressively — no FSM masking ───────────────────────────
        let mut dec_ids: Vec<usize> = vec![BOS_TOKEN];
        let mut rng = rand::thread_rng();
        let mut last_emote_logits: Option<Vec<f32>> = None;

        for _ in 0..max_tokens {
            let clamped_len = dec_ids.len().min(self.config.max_seq_len);
            let mut padded = dec_ids[dec_ids.len() - clamped_len..].to_vec();
            padded.resize(self.config.max_seq_len, PAD_TOKEN);

            let dec_tokens_t = Tensor::<B, 2, Int>::from_ints(
                TensorData::new(
                    padded.iter().map(|&t| t as i32).collect::<Vec<_>>(),
                    [1, self.config.max_seq_len],
                ),
                device,
            );

            let token_logits = self.decode_async(dec_tokens_t, memory.clone(), Some(enc_pad_mask.clone()));

            let vocab_size  = tokenizer.vocab_size();
            let last_logits = token_logits.await
                .slice([0..1, clamped_len - 1..clamped_len, 0..vocab_size])
                .reshape([vocab_size]);

            // no masking — pure model output
            let logits_vec: Vec<f32> = last_logits.to_data_async().await.expect("Need logits").to_vec().unwrap();
            let next_token = sample_top_k(&logits_vec, TOP_K, TEMPERATURE, &mut rng);

            if next_token == EOS_TOKEN || next_token == PAD_TOKEN { break; }
            dec_ids.push(next_token);
        }

        // ── Decode tokens → string (skip BOS) ─────────────────────────────────
        let raw_output = tokenizer.decode(&dec_ids[1..]);

        let fixed = fix_json_syntax(&raw_output).fixed;

        let extract = |key: &str| -> String {
            fancy_regex::Regex::new(&format!(r#"(?<=\s*"{key}"\s*:\s*)"([^"]*)""#))
                .ok()
                .and_then(|re| re.captures(&fixed).ok().flatten())
                .and_then(|caps| caps.get(1))
                .map(|m| m.as_str().to_string())
                .unwrap_or_default()
        };

        let mut parsed_action     = extract("action");
        let mut parsed_motion_dir = extract("motion_dir");
        let mut parsed_reply  = extract("reply");
        let mut parsed_emotion  = extract("emotion");

        if parsed_action.is_empty() || parsed_action.len() < 3 {
            parsed_action     = extract(" action");
            parsed_motion_dir = extract(" motion_dir");
            parsed_reply  = extract(" reply");
            parsed_emotion  = extract(" emotion");
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
                        "action":     extract("action"),
                        "motion_dir": extract("motion_dir"),
                        "emotion":      extract("emotion"),
                        "reply":      extract("reply"),
                    })
                });

            parsed_action = parsed["action"].to_string().trim().to_string();
            parsed_motion_dir = parsed["motion_dir"].to_string().trim().to_string();
            parsed_reply = parsed["reply"].to_string().trim().to_string();
            parsed_emotion = parsed["emotion"].to_string().trim().to_string();
        }

        parsed_action = parsed_action.replace("\"", "").trim().to_string();
        parsed_motion_dir = parsed_motion_dir.replace("\"", "").trim().to_string();
        parsed_reply = parsed_reply.replace("\"", "").trim().to_string();
        parsed_emotion = parsed_emotion.replace("\"", "").trim().to_string();

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

        

        GenerationResult {
            reply,
            action,
            motion_dir,
            parsed_emotion,
            raw_output,
            fsm_state: 0,
            allowed_count: None,
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

        Ok(())
    }

    pub fn load(directory: &str, device: &B::Device) -> Result<(Self, TokenizerKind, YumonBrainConfig)> {
        let dir = std::path::Path::new(directory);

        let meta_json = std::fs::read_to_string(dir.join("metadata.json"))?;
        let metadata: BrainMetadata = serde_json::from_str(&meta_json)?;

        let tokenizer = TokenizerKind::Bpe(BpeTokenizer::load("yumon_bpe")?);

        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let record   = recorder.load(dir.join("model").into(), device)
            .map_err(|e| anyhow::anyhow!("load: {e:?}"))?;
        
        let config = YumonBrainConfig {
            vocab_size: metadata.vocab_size,
            embed_dim: metadata.embed_dim,
            hidden_units: metadata.hidden_units,
            n_layers: metadata.n_layers,
            attn_heads: metadata.attn_heads,
            ff_dim: metadata.ff_dim,
            max_seq_len: metadata.max_seq_len,
            training_stage: metadata.training_stage,
            dropout_rate: 0.05, // default or stored in metadata
        };

        let model = config.init::<B>(device).load_record(record);

        Ok((model, tokenizer, config))
    }

    // We change the signature to take byte slices instead of a path
    pub async fn load_from_bytes(
        model_bytes: &[u8],
        metadata_json: &str,
        tokenizer_bytes: &[u8],
        device: &B::Device,
    ) -> Result<(Self, TokenizerKind, YumonBrainConfig)> {

        info!("load_from_bytes 1!");

        // // init_setup_async::<AutoGraphicsApi>(&WgpuDevice::default(), Default::default()).await;
        init_setup_async::<WebGpu>(&WgpuDevice::default(), Default::default()).await;
        
        // 1. Parse metadata from the string (instead of fs::read_to_string)
        let metadata: BrainMetadata = serde_json::from_str(metadata_json)?;

        info!("load_from_bytes 2!");

        // 2. Load the Tokenizer 
        let tokenizer = TokenizerKind::Bpe(BpeTokenizer::load_from_bytes(tokenizer_bytes)?);

        info!("load_from_bytes 3!");

        // 3. Use BinBytesRecorder instead of BinFileRecorder
        // let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        // let record = recorder.load(model_bytes.to_vec(), device)
        //     .map_err(|e| anyhow::anyhow!("load_record: {e:?}"))?;
        // let model: YumonBrain<Backend> = YumonBrain::new(&Default::default());
        let record = BinBytesRecorder::<FullPrecisionSettings, &'static [u8]>::default()
            .load(model_bytes, &Default::default())
            .expect("Failed to decode state");

        info!("load_from_bytes 4!");
        
        let config = YumonBrainConfig {
            vocab_size: metadata.vocab_size,
            embed_dim: metadata.embed_dim,
            hidden_units: metadata.hidden_units,
            n_layers: metadata.n_layers,
            attn_heads: metadata.attn_heads,
            ff_dim: metadata.ff_dim,
            max_seq_len: metadata.max_seq_len,
            training_stage: metadata.training_stage,
            dropout_rate: 0.05,
        };

        let model = config.init::<B>(device).load_record(record);
        // let model = model.load_record(record);

        info!("load_from_bytes 5!");

        Ok((model, tokenizer, config))
    }
}

fn make_positions<B: Backend<Device = WgpuDevice>>(
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
    pub parsed_emotion: String,
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
    pub batch_size:     usize,
    pub training_stage: TrainingStage,
    pub embed_dim:      usize,
    pub hidden_units:   usize,
    pub n_layers:       usize,
    pub attn_heads:     usize,
    pub ff_dim:         usize,
    pub max_seq_len:    usize,
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
