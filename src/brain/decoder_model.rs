//! Dense decoder-only transformer: [prompt][separator][reply] in one causal
//! sequence, pre-norm blocks of RoPE self-attention and a SwiGLU MLP.
//! Attention runs through the causal FlashAttention op (flash_attn/backend.rs),
//! so no [seq, seq] score matrix is stored for backward and the whole graph
//! works with `Autodiff<_, BalancedCheckpointing>`.
//!
//! Padding: batches are right-padded (see `build_moe_batch` in train.rs), so
//! with a causal mask no real token ever attends to a PAD key. PAD rows get
//! valid but meaningless outputs and are excluded from the loss. A batch with
//! PAD before real tokens would need a key mask this op doesn't take.
//!
//! RoPE is applied with elementwise ops (`apply_rope`) rather than
//! burn::nn::RotaryEncoding, which rotates via a [seq, dim/2, 2] x [2, 4]
//! matmul batched over batch*heads. On the non-fusion backend that matmul's
//! autotune key ignores the batch size, so a choice cached at a small batch
//! (matmul_naive) panics at batch*heads*seq >= 65536 ("Cube count too big").
//! Elementwise ops are also memory-bound, so BalancedCheckpointing recomputes
//! them instead of storing the rotated copies.
use super::{
    bpe::{BpeTokenizer, TokenizerKind},
    fixer::fix_json_syntax,
    flash_attn::backend::{FlashAttention, causal_flash_attention},
    model::{GenerationResult, MLP, MLPConfig, RMSNorm, RMSNormConfig, TEMPERATURE, TOP_K},
    samples::{Action, CardinalDir, TrainingStage},
    tokenizer::{BOS_TOKEN, EOS_TOKEN, PAD_TOKEN},
};
use anyhow::Result;
use burn::{
    module::Ignored,
    nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig},
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
};
use serde::{Deserialize, Serialize};

#[derive(Module, Debug)]
struct DecBlock<B: Backend> {
    attn_norm: RMSNorm<B>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    o: Linear<B>,
    ffn_norm: RMSNorm<B>,
    mlp: MLP<B>,
    heads: usize,
}
/// RoPE base, as in burn::nn::RotaryEncodingConfig's default.
const ROPE_THETA: f64 = 10_000.0;

/// cos/sin of position * theta^(-2i/dim) for pair i, shaped [1, 1, seq, dim/2, 1].
struct RopeTables<B: Backend> {
    cos: Tensor<B, 5>,
    sin: Tensor<B, 5>,
}
impl<B: Backend> RopeTables<B> {
    fn new(seq: usize, dim: usize, device: &B::Device) -> Self {
        let half = dim / 2;
        let (mut cos, mut sin) = (Vec::with_capacity(seq * half), Vec::with_capacity(seq * half));
        for pos in 0..seq {
            for i in 0..half {
                let angle = pos as f64 * ROPE_THETA.powf(-((2 * i) as f64) / dim as f64);
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        let table = |v: Vec<f32>| Tensor::from_data(TensorData::new(v, [1, 1, seq, half, 1]), device);
        Self { cos: table(cos), sin: table(sin) }
    }
}

/// Rotates interleaved pairs (x[2i], x[2i+1]) of x: [batch, heads, seq, dim],
/// the same convention as burn::nn::RotaryEncoding.
fn apply_rope<B: Backend>(x: Tensor<B, 4>, rope: &RopeTables<B>) -> Tensor<B, 4> {
    let [batch, heads, seq, dim] = x.dims();
    let pairs = x.reshape([batch, heads, seq, dim / 2, 2]);
    let x0 = pairs.clone().narrow(4, 0, 1);
    let x1 = pairs.narrow(4, 1, 1);
    let r0 = x0.clone() * rope.cos.clone() - x1.clone() * rope.sin.clone();
    let r1 = x0 * rope.sin.clone() + x1 * rope.cos.clone();
    Tensor::cat(vec![r0, r1], 4).reshape([batch, heads, seq, dim])
}

impl<B: FlashAttention> DecBlock<B> {
    fn forward(&self, x: Tensor<B, 3>, rope: &RopeTables<B>) -> Tensor<B, 3> {
        let [batch, seq, width] = x.dims();
        let hd = width / self.heads;
        let norm = self.attn_norm.forward(x.clone());
        let split = |t: Tensor<B, 3>| t.reshape([batch, seq, self.heads, hd]).swap_dims(1, 2);
        let q = apply_rope(split(self.q.forward(norm.clone())), rope);
        let k = apply_rope(split(self.k.forward(norm.clone())), rope);
        let v = split(self.v.forward(norm));
        let attn = causal_flash_attention(q, k, v)
            .swap_dims(1, 2)
            .reshape([batch, seq, width]);
        let x = x + self.o.forward(attn);
        x.clone() + self.mlp.forward(self.ffn_norm.forward(x))
    }
}

#[derive(Module, Debug)]
pub struct YumonDecBrain<B: Backend> {
    pub config: Ignored<YumonDecBrainConfig>,
    embedding: Embedding<B>,
    blocks: Vec<DecBlock<B>>,
    norm: RMSNorm<B>,
    dropout: Dropout,
    pub token_head: Linear<B>,
}
#[derive(Config, Debug)]
pub struct YumonDecBrainConfig {
    pub vocab_size: usize,
    pub training_stage: TrainingStage,
    #[config(default = 0.05)]
    pub dropout_rate: f64,
    #[config(default = 256)]
    pub embed_dim: usize,
    #[config(default = 256)]
    pub hidden_units: usize,
    #[config(default = 2)]
    pub n_layers: usize,
    #[config(default = 4)]
    pub attn_heads: usize,
    #[config(default = 1024)]
    pub ff_dim: usize,
    #[config(default = 320)]
    pub max_seq_len: usize,
}
impl YumonDecBrainConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> YumonDecBrain<B> {
        assert!(self.n_layers > 0 && self.max_seq_len > 0 && self.vocab_size > 0);
        assert!(self.attn_heads > 0 && self.embed_dim > 0 && self.embed_dim % self.attn_heads == 0);
        assert!(
            (self.embed_dim / self.attn_heads) % 2 == 0,
            "RoPE requires even head width"
        );
        let linear = || {
            LinearConfig::new(self.embed_dim, self.embed_dim)
                .with_bias(false)
                .init(device)
        };
        YumonDecBrain {
            config: Ignored(self.clone()),
            embedding: EmbeddingConfig::new(self.vocab_size, self.embed_dim).init(device),
            blocks: (0..self.n_layers)
                .map(|_| DecBlock {
                    attn_norm: RMSNormConfig::new(self.embed_dim).init(device),
                    q: linear(),
                    k: linear(),
                    v: linear(),
                    o: linear(),
                    ffn_norm: RMSNormConfig::new(self.embed_dim).init(device),
                    mlp: MLPConfig::new(self.embed_dim, self.ff_dim).init(device),
                    heads: self.attn_heads,
                })
                .collect(),
            norm: RMSNormConfig::new(self.embed_dim).init(device),
            dropout: DropoutConfig::new(self.dropout_rate).init(),
            token_head: LinearConfig::new(self.embed_dim, self.vocab_size).init(device),
        }
    }
    /// Weights in the model, embeddings and head included.
    pub fn param_count(&self) -> usize {
        let (d, f, v) = (self.embed_dim, self.ff_dim, self.vocab_size);
        let block = 4 * d * d + 3 * d * f + 2 * d;
        v * d + self.n_layers * block + d + d * v + v
    }
}
impl<B: FlashAttention> YumonDecBrain<B> {
    /// Logits [batch, seq, vocab]. Right-padded input only (see module docs).
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_, seq] = tokens.dims();
        assert!(seq > 0 && seq <= self.config.max_seq_len);
        let rope = RopeTables::new(seq, self.config.embed_dim / self.config.attn_heads, &tokens.device());
        let mut x = self.dropout.forward(self.embedding.forward(tokens));
        for block in &self.blocks {
            x = block.forward(x, &rope);
        }
        self.token_head.forward(self.norm.forward(x))
    }

    pub fn generate_unmasked_parsed(
        &self,
        tokenizer: &TokenizerKind,
        seed_text: &str,
        max_tokens: usize,
        device: &B::Device,
    ) -> GenerationResult {
        generate_causal(
            |tokens| self.forward(tokens),
            self.config.training_stage,
            self.config.max_seq_len,
            tokenizer,
            seed_text,
            max_tokens,
            device,
        )
    }

    // ── Checkpoint I/O ────────────────────────────────────────────────────────

    pub fn save(&self, directory: &str, tokenizer: &TokenizerKind, metadata: &DecMetadata) -> Result<()> {
        let dir = std::path::Path::new(directory);
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("metadata.json"), serde_json::to_string_pretty(metadata)?)?;
        match tokenizer {
            TokenizerKind::Bpe(t) => t.save(directory)?,
            TokenizerKind::Char(_) => anyhow::bail!("decoder checkpoints require a BPE tokenizer"),
        }
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        self.clone()
            .save_file(dir.join("model"), &recorder)
            .map_err(|e| anyhow::anyhow!("save_file: {e:?}"))?;
        Ok(())
    }

    pub fn load(directory: &str, device: &B::Device) -> Result<(Self, TokenizerKind, YumonDecBrainConfig)> {
        let dir = std::path::Path::new(directory);
        let metadata: DecMetadata =
            serde_json::from_str(&std::fs::read_to_string(dir.join("metadata.json"))?)?;
        let tokenizer = TokenizerKind::Bpe(BpeTokenizer::load(directory)?);
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let record = recorder
            .load(dir.join("model").into(), device)
            .map_err(|e| anyhow::anyhow!("load: {e:?}"))?;
        let config = YumonDecBrainConfig {
            vocab_size: metadata.vocab_size,
            training_stage: metadata.training_stage,
            dropout_rate: metadata.dropout_rate,
            embed_dim: metadata.embed_dim,
            hidden_units: metadata.hidden_units,
            n_layers: metadata.n_layers,
            attn_heads: metadata.attn_heads,
            ff_dim: metadata.ff_dim,
            max_seq_len: metadata.max_seq_len,
        };
        let model = config.init::<B>(device).load_record(record);
        Ok((model, tokenizer, config))
    }
}

// ─── Metadata ─────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct DecMetadata {
    pub dropout_rate: f64,
    pub vocab_size: usize,
    pub epochs_trained: usize,
    /// Per-token training loss.
    pub final_loss: f32,
    /// Per-token loss on the held-out split, when one was evaluated.
    #[serde(default)]
    pub val_loss: Option<f32>,
    pub batch_size: usize,
    pub training_stage: TrainingStage,
    pub embed_dim: usize,
    pub hidden_units: usize,
    pub n_layers: usize,
    pub attn_heads: usize,
    pub ff_dim: usize,
    pub max_seq_len: usize,
}

// ─── Generation (shared with the MoE decoder) ────────────────────────────────

/// Top-k sampling from BOS + prompt + separator until EOS/PAD, the token
/// budget, or `max_seq_len`, then JSON-repair and field extraction.
/// No KV cache: each step reruns the full prefix.
pub(crate) fn generate_causal<B: Backend>(
    forward: impl Fn(Tensor<B, 2, Int>) -> Tensor<B, 3>,
    training_stage: TrainingStage,
    max_seq_len: usize,
    tokenizer: &TokenizerKind,
    seed_text: &str,
    max_tokens: usize,
    device: &B::Device,
) -> GenerationResult {
    let mut dec_ids: Vec<usize> = vec![BOS_TOKEN];
    if !seed_text.is_empty() {
        dec_ids.extend(tokenizer.encode(seed_text).iter().map(|&t| t as usize));
    }

    let sep_text = if training_stage == TrainingStage::Structured {
        "\n---\n"
    } else {
        " "
    };
    let sep_tokens = tokenizer.encode(sep_text);
    dec_ids.extend(sep_tokens.iter().map(|&t| t as usize));

    let prompt_len = dec_ids.len();
    let mut rng = rand::thread_rng();

    for _ in 0..max_tokens {
        let current_len = dec_ids.len();
        if current_len >= max_seq_len {
            break;
        }

        let dec_tokens_t = Tensor::<B, 2, Int>::from_ints(
            TensorData::new(
                dec_ids.iter().map(|&t| t as i32).collect::<Vec<_>>(),
                [1, current_len],
            ),
            device,
        );

        let token_logits = forward(dec_tokens_t);

        let vocab_size = tokenizer.vocab_size();
        let last_logits = token_logits
            .slice([0..1, current_len - 1..current_len, 0..vocab_size])
            .reshape([vocab_size]);

        let logits_vec: Vec<f32> = last_logits.to_data().convert::<f32>().to_vec().unwrap();
        let next_token = sample_top_k(&logits_vec, TOP_K, TEMPERATURE, &mut rng);

        if next_token == EOS_TOKEN || next_token == PAD_TOKEN {
            break;
        }
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

    let mut parsed_action = extract("action");
    let mut parsed_reply = extract("reply");
    let mut parsed_emotion = extract("emotion");

    if parsed_action.is_empty() || parsed_action.len() < 3 {
        parsed_action = extract(" action");
        parsed_reply = extract(" reply");
        parsed_emotion = extract(" emotion");
    }

    if parsed_reply.is_empty() || parsed_reply.len() < 4 {
        let parsed: serde_json::Value = serde_json::from_str(&fixed).unwrap_or_else(|_| {
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

        parsed_action = parsed["action"].to_string().trim().to_string();
        parsed_reply = parsed["reply"].to_string().trim().to_string();
        parsed_emotion = parsed["emotion"].to_string().trim().to_string();
    }

    parsed_action = parsed_action.replace("\"", "").trim().to_string();
    parsed_reply = parsed_reply.replace("\"", "").trim().to_string();
    parsed_emotion = parsed_emotion.replace("\"", "").trim().to_string();

    let action = match parsed_action.as_str().trim() {
        "go to destination" => Action::GoToDestination,
        "go home" => Action::GoHome,
        "follow" => Action::Follow,
        "get help" => Action::GetHelp,
        "survey area" => Action::Survey,
        "collect items" => Action::Collect,
        "stack items" => Action::Stack,
        _ => Action::Sit,
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

fn sample_top_k(logits: &[f32], k: usize, temperature: f32, rng: &mut impl rand::Rng) -> usize {
    use rand::distributions::WeightedIndex;
    use rand::prelude::*;

    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::autodiff::{
            Autodiff,
            checkpoint::strategy::{BalancedCheckpointing, CheckpointStrategy, NoCheckpointing},
        },
        module::AutodiffModule,
        optim::{AdamWConfig, GradientsParams, Optimizer},
        record::BinBytesRecorder,
        tensor::{Distribution, activation::softmax, backend::AutodiffBackend},
    };
    use burn_cubecl::CubeBackend;
    use cubecl::wgpu::WgpuRuntime;

    type Inner = CubeBackend<WgpuRuntime, f32, i32, u32>;
    type Balanced = Autodiff<Inner, BalancedCheckpointing>;
    type Plain = Autodiff<Inner, NoCheckpointing>;

    fn assert_close(actual: Vec<f32>, expected: Vec<f32>, tolerance: f32, what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}: length");
        let mut worst = 0.0f32;
        for (a, b) in actual.iter().zip(&expected) {
            assert!(a.is_finite() && b.is_finite(), "{what}: non-finite {a} vs {b}");
            worst = worst.max((a - b).abs() / (1.0 + b.abs()));
        }
        assert!(worst <= tolerance, "{what}: max relative error {worst} > {tolerance}");
    }

    fn naive_causal<B: Backend>(q: Tensor<B, 4>, k: Tensor<B, 4>, v: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, _, seq, dim] = q.dims();
        // tril_mask marks what to fill: everything above the diagonal, i.e. future keys.
        let future = Tensor::<B, 2, Bool>::tril_mask([seq, seq], 0, &q.device()).unsqueeze::<4>();
        let scores = q.matmul(k.transpose()) / (dim as f64).sqrt();
        // Finite fill: -inf through Burn's softmax gives NaN on this backend.
        softmax(scores.mask_fill(future, -1e9), 3).matmul(v)
    }

    fn values<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
        t.into_data().convert::<f32>().to_vec().unwrap()
    }

    #[test]
    #[ignore = "requires a CUDA GPU with native BF16 support; no profiling"]
    fn cuda_bf16_flash_forward_backward_matches_fp32() {
        use burn::tensor::{DType, FloatDType};
        type AD = crate::brain::train::CudaTrainBackend;
        let device = Default::default();
        for seq in [1, 37, 70] {
            let random = || Tensor::<AD, 4>::random(
                [1, 2, seq, 16], Distribution::Normal(0.0, 0.5), &device,
            );
            let (q, k, v, w) = (random(), random(), random(), random());
            assert_eq!(q.dtype(), DType::BF16);
            let run = |flash: bool| {
                // The FP32 reference uses the same BF16-rounded inputs.
                let dtype = if flash { FloatDType::BF16 } else { FloatDType::F32 };
                let (q, k, v) = (
                    q.clone().cast(dtype).require_grad(),
                    k.clone().cast(dtype).require_grad(),
                    v.clone().cast(dtype).require_grad(),
                );
                let out = if flash {
                    causal_flash_attention(q.clone(), k.clone(), v.clone())
                } else {
                    naive_causal(q.clone(), k.clone(), v.clone())
                };
                let grads = (out.clone() * w.clone().cast(dtype))
                    .cast(FloatDType::F32).sum().backward();
                let dq = q.grad(&grads).unwrap();
                let dk = k.grad(&grads).unwrap();
                let dv = v.grad(&grads).unwrap();
                if flash {
                    for tensor in [&out.clone().inner(), &dq, &dk, &dv] {
                        assert_eq!(tensor.dtype(), DType::BF16);
                    }
                }
                (values(out), values(dq), values(dk), values(dv))
            };
            let (out, dq, dk, dv) = run(true);
            let (reference, rq, rk, rv) = run(false);
            assert_close(out, reference, 0.02, "BF16 output");
            assert_close(dq, rq, 0.03, "BF16 dQ");
            assert_close(dk, rk, 0.03, "BF16 dK");
            assert_close(dv, rv, 0.03, "BF16 dV");
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU with native BF16 support; no profiling"]
    fn cuda_bf16_decoder_optimizer_and_checkpoint() {
        use burn::tensor::{DType, FloatDType};
        type AD = crate::brain::train::CudaTrainBackend;
        let device = Default::default();
        let config = tiny_config();
        let model: YumonDecBrain<AD> = config.init(&device);
        let before = values(model.token_head.weight.val());
        let logits = model.forward(Tensor::from_ints([[1, 4, 5, 6]], &device));
        assert_eq!(logits.dtype(), DType::BF16);
        let loss = logits.cast(FloatDType::F32).square().mean();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let updated = AdamWConfig::new().init().step(0.01, model, grads);
        assert_eq!(updated.token_head.weight.val().dtype(), DType::BF16);
        assert_ne!(before, values(updated.token_head.weight.val()));
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let bytes = recorder.record(updated.clone().into_record(), ()).unwrap();
        let record = recorder.load(bytes, &device).unwrap();
        let loaded: YumonDecBrain<AD> = config.init(&device).load_record(record);
        assert_eq!(loaded.token_head.weight.val().dtype(), DType::BF16);
        let tokens = || Tensor::from_ints([[1, 4, 5]], &device);
        assert_close(values(updated.valid().forward(tokens())),
            values(loaded.valid().forward(tokens())), 0.0, "BF16 checkpoint");
    }

    /// Kernel vs matmul+softmax reference: output and dQ/dK/dV, including
    /// partial last tiles, multi-tile sequences, single-token sequences and
    /// several head widths (tile rows shrink as dim grows, see block_for_dim).
    fn flash_matches_reference<AD: AutodiffBackend + FlashAttention>() {
        let device = Default::default();
        for [batch, heads, seq, dim] in [
            [2, 2, 1, 32],
            [1, 3, 37, 32],
            [1, 1, 200, 32],
            [2, 2, 130, 64],
            [1, 2, 70, 16],
            [1, 1, 50, 128],
        ] {
            let what = format!("b{batch} h{heads} s{seq} d{dim}");
            let rand = || Tensor::<AD, 4>::random([batch, heads, seq, dim], Distribution::Normal(0.0, 1.0), &device);
            let (q, k, v, w) = (rand(), rand(), rand(), rand());
            let run = |flash: bool| {
                let (q, k, v) = (q.clone().require_grad(), k.clone().require_grad(), v.clone().require_grad());
                let out = if flash {
                    causal_flash_attention(q.clone(), k.clone(), v.clone())
                } else {
                    naive_causal(q.clone(), k.clone(), v.clone())
                };
                // Weighted sum so every output element gets a distinct gradient.
                let grads = (out.clone() * w.clone()).sum().backward();
                (
                    values(out.inner()),
                    values(q.grad(&grads).unwrap()),
                    values(k.grad(&grads).unwrap()),
                    values(v.grad(&grads).unwrap()),
                )
            };
            let (o, dq, dk, dv) = run(true);
            let (o_ref, dq_ref, dk_ref, dv_ref) = run(false);
            assert_close(o, o_ref, 1e-4, &format!("{what} output"));
            assert_close(dq, dq_ref, 1e-3, &format!("{what} dQ"));
            assert_close(dk, dk_ref, 1e-3, &format!("{what} dK"));
            assert_close(dv, dv_ref, 1e-3, &format!("{what} dV"));
        }
    }

    #[test]
    fn flash_matches_reference_balanced_checkpointing() {
        flash_matches_reference::<Balanced>();
    }

    #[test]
    fn flash_matches_reference_no_checkpointing() {
        flash_matches_reference::<Plain>();
    }

    #[test]
    fn flash_inference_backend_matches_reference() {
        let device = Default::default();
        let rand = || Tensor::<Inner, 4>::random([2, 4, 45, 32], Distribution::Normal(0.0, 1.0), &device);
        let (q, k, v) = (rand(), rand(), rand());
        assert_close(
            values(causal_flash_attention(q.clone(), k.clone(), v.clone())),
            values(naive_causal(q, k, v)),
            1e-4,
            "inner backend output",
        );
    }

    #[test]
    fn rope_matches_burn_rotary_encoding() {
        let device = Default::default();
        let (seq, dim) = (40, 32);
        let x = Tensor::<Inner, 4>::random([2, 3, seq, dim], Distribution::Normal(0.0, 1.0), &device);
        let burn_rope = burn::nn::RotaryEncodingConfig::new(seq, dim).init::<Inner>(&device);
        assert_close(
            values(apply_rope(x.clone(), &RopeTables::new(seq, dim, &device))),
            values(burn_rope.forward(x)),
            1e-4,
            "rope",
        );
    }

    fn tiny_config() -> YumonDecBrainConfig {
        YumonDecBrainConfig::new(32, TrainingStage::Language)
            .with_embed_dim(16)
            .with_n_layers(2)
            .with_attn_heads(2)
            .with_ff_dim(32)
            .with_max_seq_len(8)
            .with_dropout_rate(0.0)
    }

    #[test]
    fn decoder_causal_padding_and_optimizer_step() {
        let device = Default::default();
        let model: YumonDecBrain<Balanced> = tiny_config().init(&device);
        let tokens = Tensor::from_ints([[1, 4, 5, 6, 0, 0]], &device);
        let logits = model.forward(tokens);
        assert_eq!(logits.dims(), [1, 6, 32]);
        // Changing later tokens leaves earlier positions alone.
        let changed = model.forward(Tensor::from_ints([[1, 4, 9, 8, 0, 0]], &device));
        assert_close(
            values(logits.clone().slice([0..1, 0..2, 0..32])),
            values(changed.slice([0..1, 0..2, 0..32])),
            1e-5,
            "causal prefix",
        );
        // Right padding doesn't change the real positions.
        let short = model.forward(Tensor::from_ints([[1, 4, 5, 6]], &device));
        assert_close(
            values(logits.clone().slice([0..1, 0..4, 0..32])),
            values(short),
            1e-5,
            "right padding",
        );
        let grads = GradientsParams::from_grads(logits.powf_scalar(2.0).mean().backward(), &model);
        let updated = AdamWConfig::new().init().step(1e-3, model, grads);
        let out = updated.valid().forward(Tensor::from_ints([[1, 2, 3]], &device));
        assert!(values(out).iter().all(|x| x.is_finite()));
    }

    fn grads_of<C: CheckpointStrategy>(
        config: &YumonDecBrainConfig,
        bytes: &[u8],
        tokens: &[i32],
    ) -> (f32, Vec<Vec<f32>>) {
        let device = Default::default();
        let record = BinBytesRecorder::<FullPrecisionSettings>::default()
            .load(bytes.to_vec(), &device)
            .unwrap();
        let model: YumonDecBrain<Autodiff<Inner, C>> = config.init(&device).load_record(record);
        let batch = tokens.len() / config.max_seq_len;
        let tokens = Tensor::<Autodiff<Inner, C>, 2, Int>::from_ints(
            TensorData::new(tokens.to_vec(), [batch, config.max_seq_len]),
            &device,
        );
        let loss = model.forward(tokens).powf_scalar(2.0).mean();
        let grads = loss.backward();
        struct Collect<'a, C: CheckpointStrategy> {
            grads: &'a <Autodiff<Inner, C> as AutodiffBackend>::Gradients,
            out: Vec<Vec<f32>>,
        }
        impl<C: CheckpointStrategy> burn::module::ModuleVisitor<Autodiff<Inner, C>> for Collect<'_, C> {
            fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<Tensor<Autodiff<Inner, C>, D>>) {
                self.out.push(values(param.val().grad(self.grads).expect("every weight trains")));
            }
        }
        let mut collect = Collect { grads: &grads, out: vec![] };
        model.visit(&mut collect);
        (values(loss.inner())[0], collect.out)
    }

    /// Same weights and batch: BalancedCheckpointing recomputes instead of
    /// storing, so loss and every parameter gradient must match NoCheckpointing.
    #[test]
    fn decoder_balanced_checkpointing_matches_plain_gradients() {
        let device = Default::default();
        let config = tiny_config().with_max_seq_len(40).with_embed_dim(64);
        let bytes = BinBytesRecorder::<FullPrecisionSettings>::default()
            .record(config.init::<Inner>(&device).into_record(), ())
            .unwrap();
        let tokens: Vec<i32> = (0..2 * 40).map(|i| 1 + (i * 7 % 31)).collect();

        let (loss_b, grads_b) = grads_of::<BalancedCheckpointing>(&config, &bytes, &tokens);
        let (loss_p, grads_p) = grads_of::<NoCheckpointing>(&config, &bytes, &tokens);
        assert!((loss_b - loss_p).abs() <= 1e-5 * loss_p.abs().max(1.0), "{loss_b} vs {loss_p}");
        assert_eq!(grads_b.len(), grads_p.len());
        for (i, (b, p)) in grads_b.into_iter().zip(grads_p).enumerate() {
            assert_close(b, p, 1e-4, &format!("param {i}"));
        }
    }

    #[test]
    fn decoder_checkpoint_roundtrip() {
        let device = Default::default();
        let tokenizer = TokenizerKind::Bpe(BpeTokenizer::load("yumon_bpe").unwrap());
        let mut config = tiny_config();
        config.vocab_size = tokenizer.vocab_size();
        let model: YumonDecBrain<Inner> = config.init(&device);
        let path = std::env::temp_dir().join(format!("yumon-dec-{}", uuid::Uuid::new_v4()));
        let meta = DecMetadata {
            dropout_rate: config.dropout_rate,
            vocab_size: config.vocab_size,
            epochs_trained: 1,
            final_loss: 1.0,
            val_loss: None,
            batch_size: 1,
            training_stage: config.training_stage,
            embed_dim: config.embed_dim,
            hidden_units: config.hidden_units,
            n_layers: config.n_layers,
            attn_heads: config.attn_heads,
            ff_dim: config.ff_dim,
            max_seq_len: config.max_seq_len,
        };
        model.save(path.to_str().unwrap(), &tokenizer, &meta).unwrap();
        let (restored, tok, restored_config) =
            YumonDecBrain::<Inner>::load(path.to_str().unwrap(), &device).unwrap();
        assert_eq!(tok.encode("hello"), tokenizer.encode("hello"));
        assert_eq!(restored_config.n_layers, config.n_layers);
        let input = || Tensor::<Inner, 2, Int>::from_ints([[1, 4, 5]], &device);
        assert_close(values(model.forward(input())), values(restored.forward(input())), 1e-6, "roundtrip");
        // Only remove the uniquely created test directory after verifying its parent.
        assert_eq!(path.parent(), Some(std::env::temp_dir().as_path()));
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn param_count_matches_module() {
        let device = Default::default();
        let config = tiny_config();
        let model: YumonDecBrain<Inner> = config.init(&device);
        assert_eq!(model.num_params(), config.param_count());
    }
}
