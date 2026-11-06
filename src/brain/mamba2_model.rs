//! Mamba-2 decoder (Dao & Gu, 2024, "Transformers are SSMs"): pre-norm blocks of
//! the SSD mixer, optionally followed by the same SwiGLU MLP the dense decoder
//! uses (mamba_ssm's `d_intermediate`), so the token mixer is the only thing
//! that differs from `YumonDecBrain` when `mlp` is on. No attention, no RoPE:
//! position lives in the causal conv and the recurrent state.
//!
//! Mixer, per token t and head h (scalar decay per head, the SSD restriction):
//!   dt   = softplus(W_dt x + dt_bias)                 A = -exp(A_log)
//!   S_t  = exp(dt A) S_{t-1} + dt * B_t^T x_t          S: [d_state, head_dim]
//!   y_t  = C_t S_t + D x_t
//! with x, B, C coming out of a short causal depthwise conv + SiLU, and the
//! output gated by SiLU(z) before an RMSNorm and the output projection.
//!
//! Training runs the chunked SSD algorithm (the paper's `ssd_minimal`): within a
//! chunk the quadratic "attention-like" form (C B^T masked by the decay matrix),
//! across chunks a small state-passing matmul. Everything is batched matmuls and
//! elementwise ops, so there is no per-timestep loop like xlstm_model.rs has.
//! Generation feeds new tokens through the same function with the previous
//! state (`Mamba2State`), so decoding costs O(1) per token, not O(context).
//!
//! Simplifications vs. mamba_ssm: one B/C group (ngroups = 1), no fused
//! kernels, and A_log / D / dt_bias get the trainer's weight decay (mamba_ssm
//! exempts them; the shared AdamW here has a single decay value).
//!
//! Right padding is safe for the same reason as in decoder_model.rs: the model
//! is causal, so PAD after the real tokens never reaches them.
use super::{
    bpe::{BpeTokenizer, TokenizerKind},
    decoder_model::generate_causal_cached,
    model::{GenerationResult, MLP, MLPConfig, RMSNorm, RMSNormConfig},
    samples::TrainingStage,
};
use anyhow::Result;
use burn::{
    module::{Ignored, Param},
    nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig},
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{Distribution, activation::silu},
};
use serde::{Deserialize, Serialize};

/// `mamba` section of a code config. Heads come from `attn_heads`, so
/// head_dim = expand * embed_dim / attn_heads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Mamba2Settings {
    /// SSM state size N per head (B and C width).
    pub d_state: usize,
    /// Inner width = expand * embed_dim.
    pub expand: usize,
    /// Causal depthwise conv kernel width.
    pub d_conv: usize,
    /// SSD chunk length; only changes speed and rounding, not the math.
    pub chunk_size: usize,
    /// SwiGLU MLP (ff_dim) after each mixer, as in the dense decoder.
    pub mlp: bool,
}

impl Default for Mamba2Settings {
    fn default() -> Self {
        Self { d_state: 64, expand: 2, d_conv: 4, chunk_size: 64, mlp: true }
    }
}

impl Mamba2Settings {
    pub fn validate(&self, embed_dim: usize, heads: usize) -> Result<()> {
        anyhow::ensure!(self.d_state > 0 && self.expand > 0 && self.d_conv > 0 && self.chunk_size > 0,
            "mamba d_state, expand, d_conv and chunk_size must be positive");
        anyhow::ensure!((self.expand * embed_dim) % heads == 0,
            "mamba: expand * embed_dim must be divisible by attn_heads (the SSM heads)");
        Ok(())
    }

    /// Run-name tag; chunk_size is left out because it does not change the model.
    pub fn tag(&self) -> String {
        format!("Mamba2_n{}_x{}{}{}", self.d_state, self.expand,
            if self.d_conv != 4 { format!("_conv{}", self.d_conv) } else { String::new() },
            if self.mlp { "" } else { "_nomlp" })
    }
}

const DT_MIN: f64 = 1e-3;
const DT_MAX: f64 = 1e-1;
const DT_FLOOR: f64 = 1e-4;
/// Stands in for -inf above the diagonal of the decay matrix; exp() of it is 0.
const MASKED: f32 = -1e30;

/// softplus(x) = relu(x) + log(1 + exp(-|x|)); burn's version overflows for large x.
fn softplus<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.clone().clamp_min(0.0) + x.abs().neg().exp().add_scalar(1.0).log()
}

/// exp(segsum): [.., L] cumulative sums -> [.., L, L] with out[i, j] =
/// exp(cs[i] - cs[j]) for j <= i and 0 above the diagonal.
fn decay_matrix<B: Backend>(cs: Tensor<B, 3>) -> Tensor<B, 4> {
    let [a, b, l] = cs.dims();
    let diff = cs.clone().unsqueeze_dim::<4>(3) - cs.unsqueeze_dim::<4>(2);
    let upper = Tensor::<B, 2, Bool>::tril_mask([l, l], 0, &diff.device())
        .reshape([1, 1, l, l]).expand([a, b, l, l]);
    diff.mask_fill(upper, MASKED).exp()
}

/// Chunked SSD scan. x: [b, t, h, p] (already times dt), a: [b, t, h] (dt * A),
/// bm/cm: [b, t, n], initial state [b, h, n, p]. Returns y [b, t, h, p] and the
/// state after the last token.
fn ssd<B: Backend>(x: Tensor<B, 4>, a: Tensor<B, 3>, bm: Tensor<B, 3>, cm: Tensor<B, 3>,
    initial: Tensor<B, 4>, chunk: usize) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [b, t, h, p] = x.dims();
    let n = bm.dims()[2];
    let q = chunk.min(t);
    let c = t.div_ceil(q);
    let padded = c * q;
    // Zero padding at the end has dt = 0: no input and no decay, so the final
    // state is the one after the last real token.
    let pad_time = |len: usize| padded - len;
    let (x, a, bm, cm) = if padded > t {
        let device = x.device();
        (Tensor::cat(vec![x, Tensor::zeros([b, pad_time(t), h, p], &device)], 1),
         Tensor::cat(vec![a, Tensor::zeros([b, pad_time(t), h], &device)], 1),
         Tensor::cat(vec![bm, Tensor::zeros([b, pad_time(t), n], &device)], 1),
         Tensor::cat(vec![cm, Tensor::zeros([b, pad_time(t), n], &device)], 1))
    } else { (x, a, bm, cm) };
    let bh = b * h;
    // Heads fold into the batch: [b*h, chunks, chunk_len, ..].
    let x = x.reshape([b, c, q, h, p]).permute([0, 3, 1, 2, 4]).reshape([bh, c, q, p]);
    let a = a.reshape([b, c, q, h]).permute([0, 3, 1, 2]).reshape([bh, c, q]);
    let per_head = |m: Tensor<B, 3>| m.reshape([b, 1, c, q, n]).expand([b, h, c, q, n]).reshape([bh, c, q, n]);
    let (bm, cm) = (per_head(bm), per_head(cm));

    let cs = a.cumsum(2); // [bh, c, q]
    // 1. Within each chunk: (C B^T * decay) X.
    let y_diag = (cm.clone().matmul(bm.clone().swap_dims(2, 3)) * decay_matrix(cs.clone())).matmul(x.clone());
    // 2. State at the end of each chunk from its own tokens.
    let last = cs.clone().slice([0..bh, 0..c, q - 1..q]); // [bh, c, 1]
    let to_end = (last.clone() - cs.clone()).exp();
    let states = (bm * to_end.unsqueeze_dim::<4>(3)).swap_dims(2, 3).matmul(x); // [bh, c, n, p]
    // 3. Pass states across chunks, starting from `initial`.
    let initial = initial.reshape([bh, 1, n, p]);
    let states = Tensor::cat(vec![initial, states], 1).reshape([bh, 1, c + 1, n * p]);
    let chunk_cs = Tensor::cat(vec![Tensor::zeros([bh, 1], &cs.device()), last.reshape([bh, c])], 1)
        .cumsum(1).reshape([bh, 1, c + 1]);
    let passed = decay_matrix(chunk_cs).matmul(states).reshape([bh, c + 1, n, p]);
    let entering = passed.clone().slice([0..bh, 0..c, 0..n, 0..p]);
    let last_state = passed.slice([0..bh, c..c + 1, 0..n, 0..p]).reshape([b, h, n, p]);
    // 4. Contribution of the state entering each chunk.
    let y_off = cm.matmul(entering) * cs.exp().unsqueeze_dim::<4>(3);
    let y = (y_diag + y_off).reshape([b, h, c, q, p]).permute([0, 2, 3, 1, 4]).reshape([b, padded, h, p]);
    (y.slice([0..b, 0..t, 0..h, 0..p]), last_state)
}

/// Per-layer recurrent state: the last d_conv - 1 conv inputs and the SSM state.
#[derive(Clone, Debug)]
pub struct LayerState<B: Backend> {
    conv: Tensor<B, 3>, // [b, d_conv - 1, conv_dim]
    ssm: Tensor<B, 4>,  // [b, h, n, p]
}

/// Generation state for every layer; empty until the first `forward_state`.
#[derive(Clone, Debug, Default)]
pub struct Mamba2State<B: Backend> {
    layers: Vec<LayerState<B>>,
}

#[derive(Module, Debug)]
struct Mamba2Mixer<B: Backend> {
    /// d_model -> [z | x B C | dt]
    in_proj: Linear<B>,
    conv_weight: Param<Tensor<B, 2>>, // [d_conv, conv_dim]
    conv_bias: Param<Tensor<B, 1>>,
    dt_bias: Param<Tensor<B, 1>>, // [h]
    a_log: Param<Tensor<B, 1>>,   // [h]
    d_skip: Param<Tensor<B, 1>>,  // [h]
    norm: RMSNorm<B>,
    out_proj: Linear<B>,
    heads: usize,
    head_dim: usize,
    d_state: usize,
    d_conv: usize,
    chunk: usize,
}

impl<B: Backend> Mamba2Mixer<B> {
    fn new(d_model: usize, heads: usize, s: &Mamba2Settings, residuals: usize, device: &B::Device) -> Self {
        let inner = s.expand * d_model;
        let conv_dim = inner + 2 * s.d_state;
        let bound = 1.0 / (s.d_conv as f64).sqrt(); // PyTorch Conv1d default
        // dt log-uniform in [DT_MIN, DT_MAX]; dt_bias is its inverse softplus.
        let dt =Tensor::<B, 1>::random([heads], Distribution::Uniform(0.0, 1.0), device)
            .mul_scalar(DT_MAX.ln() - DT_MIN.ln()).add_scalar(DT_MIN.ln()).exp().clamp_min(DT_FLOOR);
        // softplus^-1(dt) = dt + log(1 - exp(-dt))
        let dt_bias = dt.clone() + dt.neg().exp().neg().add_scalar(1.0).log();
        let a = Tensor::<B, 1>::random([heads], Distribution::Uniform(1.0, 16.0), device);
        // mamba_ssm rescales each residual output projection by 1/sqrt(residuals per layer * layers).
        let out_gain = 1.0 / 3f64.sqrt() / (residuals as f64).sqrt();
        Self {
            in_proj: LinearConfig::new(d_model, 2 * inner + 2 * s.d_state + heads).with_bias(false).init(device),
            conv_weight: Param::from_tensor(Tensor::random([s.d_conv, conv_dim], Distribution::Uniform(-bound, bound), device)),
            conv_bias: Param::from_tensor(Tensor::random([conv_dim], Distribution::Uniform(-bound, bound), device)),
            dt_bias: Param::from_tensor(dt_bias),
            a_log: Param::from_tensor(a.log()),
            d_skip: Param::from_tensor(Tensor::ones([heads], device)),
            norm: RMSNormConfig::new(inner).init(device),
            out_proj: LinearConfig::new(inner, d_model).with_bias(false)
                .with_initializer(Initializer::KaimingUniform { gain: out_gain, fan_out_only: false })
                .init(device),
            heads,
            head_dim: inner / heads,
            d_state: s.d_state,
            d_conv: s.d_conv,
            chunk: s.chunk_size,
        }
    }

    /// x: [b, t, d_model] -> output and the state after the last token.
    fn forward(&self, x: Tensor<B, 3>, state: Option<&LayerState<B>>) -> (Tensor<B, 3>, LayerState<B>) {
        let [b, t, _] = x.dims();
        let (h, p, n, k) = (self.heads, self.head_dim, self.d_state, self.d_conv);
        let inner = h * p;
        let conv_dim = inner + 2 * n;
        let device = x.device();
        let proj = self.in_proj.forward(x);
        let z = proj.clone().slice([0..b, 0..t, 0..inner]);
        let xbc = proj.clone().slice([0..b, 0..t, inner..inner + conv_dim]);
        let dt = proj.slice([0..b, 0..t, inner + conv_dim..inner + conv_dim + h]);

        // Causal depthwise conv over time, continuing from the saved inputs.
        let previous = state.map(|s| s.conv.clone())
            .unwrap_or_else(|| Tensor::zeros([b, k - 1, conv_dim], &device));
        let window = Tensor::cat(vec![previous, xbc], 1); // [b, t + k - 1, conv_dim]
        let conv_state = window.clone().slice([0..b, t..t + k - 1, 0..conv_dim]);
        let weight = self.conv_weight.val();
        let mut conv = self.conv_bias.val().reshape([1, 1, conv_dim]).expand([b, t, conv_dim]);
        for j in 0..k {
            conv = conv + window.clone().slice([0..b, j..j + t, 0..conv_dim])
                * weight.clone().slice([j..j + 1, 0..conv_dim]).reshape([1, 1, conv_dim]);
        }
        let xbc = silu(conv);
        let xs = xbc.clone().slice([0..b, 0..t, 0..inner]).reshape([b, t, h, p]);
        let bm = xbc.clone().slice([0..b, 0..t, inner..inner + n]);
        let cm = xbc.slice([0..b, 0..t, inner + n..inner + 2 * n]);

        let dt = softplus(dt + self.dt_bias.val().reshape([1, 1, h])); // [b, t, h]
        let a = self.a_log.val().exp().neg().reshape([1, 1, h]);
        let initial = state.map(|s| s.ssm.clone()).unwrap_or_else(|| Tensor::zeros([b, h, n, p], &device));
        let (y, ssm) = ssd(xs.clone() * dt.clone().unsqueeze_dim::<4>(3), dt * a, bm, cm, initial, self.chunk);
        let y = (y + xs * self.d_skip.val().reshape([1, 1, h, 1])).reshape([b, t, inner]);
        let y = self.norm.forward(y * silu(z));
        (self.out_proj.forward(y), LayerState { conv: conv_state, ssm })
    }
}

#[derive(Module, Debug)]
struct Mamba2Block<B: Backend> {
    mixer_norm: RMSNorm<B>,
    mixer: Mamba2Mixer<B>,
    mlp_norm: Option<RMSNorm<B>>,
    mlp: Option<MLP<B>>,
}

impl<B: Backend> Mamba2Block<B> {
    fn forward(&self, x: Tensor<B, 3>, state: Option<&LayerState<B>>) -> (Tensor<B, 3>, LayerState<B>) {
        let (mixed, state) = self.mixer.forward(self.mixer_norm.forward(x.clone()), state);
        let x = x + mixed;
        let x = match (&self.mlp_norm, &self.mlp) {
            (Some(norm), Some(mlp)) => x.clone() + mlp.forward(norm.forward(x)),
            _ => x,
        };
        (x, state)
    }
}

#[derive(Module, Debug)]
pub struct YumonMamba2Brain<B: Backend> {
    pub config: Ignored<YumonMamba2BrainConfig>,
    embedding: Embedding<B>,
    blocks: Vec<Mamba2Block<B>>,
    norm: RMSNorm<B>,
    dropout: Dropout,
    /// None with tied embeddings: logits use the embedding table.
    token_head: Option<Linear<B>>,
}

#[derive(Config, Debug)]
pub struct YumonMamba2BrainConfig {
    pub vocab_size: usize,
    pub training_stage: TrainingStage,
    #[config(default = 0.05)]
    pub dropout_rate: f64,
    #[config(default = 256)]
    pub embed_dim: usize,
    #[config(default = 2)]
    pub n_layers: usize,
    /// SSM heads.
    #[config(default = 4)]
    pub attn_heads: usize,
    #[config(default = 1024)]
    pub ff_dim: usize,
    /// Generation stops here; the model itself has no context limit.
    #[config(default = 320)]
    pub max_seq_len: usize,
    #[config(default = false)]
    pub tie_embeddings: bool,
    pub settings: Mamba2Settings,
}

impl YumonMamba2BrainConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> YumonMamba2Brain<B> {
        let s = &self.settings;
        s.validate(self.embed_dim, self.attn_heads).expect("invalid mamba settings");
        assert!(self.n_layers > 0 && self.vocab_size > 0);
        let residuals = self.n_layers * if s.mlp { 2 } else { 1 };
        YumonMamba2Brain {
            config: Ignored(self.clone()),
            // mamba_ssm's N(0, 0.02); also keeps tied logits small at init.
            embedding: EmbeddingConfig::new(self.vocab_size, self.embed_dim)
                .with_initializer(Initializer::Normal { mean: 0.0, std: 0.02 }).init(device),
            blocks: (0..self.n_layers).map(|_| Mamba2Block {
                mixer_norm: RMSNormConfig::new(self.embed_dim).init(device),
                mixer: Mamba2Mixer::new(self.embed_dim, self.attn_heads, s, residuals, device),
                mlp_norm: s.mlp.then(|| RMSNormConfig::new(self.embed_dim).init(device)),
                mlp: s.mlp.then(|| MLPConfig::new(self.embed_dim, self.ff_dim).init(device)),
            }).collect(),
            norm: RMSNormConfig::new(self.embed_dim).init(device),
            dropout: DropoutConfig::new(self.dropout_rate).init(),
            token_head: (!self.tie_embeddings)
                .then(|| LinearConfig::new(self.embed_dim, self.vocab_size).init(device)),
        }
    }

    /// Weights in the model, embeddings and head included.
    pub fn param_count(&self) -> usize {
        let (d, v, s) = (self.embed_dim, self.vocab_size, &self.settings);
        let (inner, h, n) = (s.expand * d, self.attn_heads, s.d_state);
        let conv_dim = inner + 2 * n;
        let mixer = d * (2 * inner + 2 * n + h) + (s.d_conv + 1) * conv_dim + 3 * h + inner + inner * d;
        let mlp = if s.mlp { d + 3 * d * self.ff_dim } else { 0 };
        let head = if self.tie_embeddings { 0 } else { d * v + v };
        v * d + self.n_layers * (d + mixer + mlp) + d + head
    }
}

impl<B: Backend> YumonMamba2Brain<B> {
    /// Logits [batch, seq, vocab] from a zero state.
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.forward_state(tokens, &mut Mamba2State { layers: Vec::new() })
    }

    /// Logits for `tokens` continuing from `state`, which is advanced past them.
    pub fn forward_state(&self, tokens: Tensor<B, 2, Int>, state: &mut Mamba2State<B>) -> Tensor<B, 3> {
        let mut x = self.dropout.forward(self.embedding.forward(tokens));
        let mut next = Vec::with_capacity(self.blocks.len());
        for (i, block) in self.blocks.iter().enumerate() {
            let (y, layer) = block.forward(x, state.layers.get(i));
            x = y;
            next.push(layer);
        }
        state.layers = next;
        let x = self.norm.forward(x);
        match &self.token_head {
            Some(head) => head.forward(x),
            None => {
                let [b, t, d] = x.dims();
                let weight = self.embedding.weight.val(); // [vocab, d]
                let v = weight.dims()[0];
                x.reshape([b * t, d]).matmul(weight.transpose()).reshape([b, t, v])
            }
        }
    }

    pub fn generate_unmasked_parsed(&self, tokenizer: &TokenizerKind, seed_text: &str,
        max_tokens: usize, device: &B::Device) -> GenerationResult {
        let mut state = Mamba2State { layers: Vec::new() };
        generate_causal_cached(|tokens| self.forward_state(tokens, &mut state),
            self.config.training_stage, self.config.max_seq_len, tokenizer, seed_text, max_tokens, device)
    }

    pub fn save(&self, directory: &str, tokenizer: &TokenizerKind, metadata: &Mamba2Metadata) -> Result<()> {
        let dir = std::path::Path::new(directory);
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("metadata.json"), serde_json::to_string_pretty(metadata)?)?;
        match tokenizer {
            TokenizerKind::Bpe(t) => t.save(directory)?,
            TokenizerKind::Char(_) => anyhow::bail!("mamba2 checkpoints require a BPE tokenizer"),
        }
        self.clone().save_file(dir.join("model"), &BinFileRecorder::<FullPrecisionSettings>::new())
            .map_err(|e| anyhow::anyhow!("save_file: {e:?}"))?;
        Ok(())
    }

    pub fn load(directory: &str, device: &B::Device) -> Result<(Self, TokenizerKind, YumonMamba2BrainConfig)> {
        let dir = std::path::Path::new(directory);
        let m: Mamba2Metadata = serde_json::from_str(&std::fs::read_to_string(dir.join("metadata.json"))?)?;
        let tokenizer = TokenizerKind::Bpe(BpeTokenizer::load(directory)?);
        let record = BinFileRecorder::<FullPrecisionSettings>::new()
            .load(dir.join("model").into(), device)
            .map_err(|e| anyhow::anyhow!("load: {e:?}"))?;
        let config = YumonMamba2BrainConfig {
            vocab_size: m.vocab_size, training_stage: m.training_stage, dropout_rate: m.dropout_rate,
            embed_dim: m.embed_dim, n_layers: m.n_layers, attn_heads: m.attn_heads, ff_dim: m.ff_dim,
            max_seq_len: m.max_seq_len, tie_embeddings: m.tie_embeddings, settings: m.mamba,
        };
        let model = config.init::<B>(device).load_record(record);
        Ok((model, tokenizer, config))
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Mamba2Metadata {
    pub dropout_rate: f64,
    pub vocab_size: usize,
    pub epochs_trained: usize,
    pub final_loss: f32,
    #[serde(default)]
    pub val_loss: Option<f32>,
    pub batch_size: usize,
    pub training_stage: TrainingStage,
    pub embed_dim: usize,
    pub n_layers: usize,
    pub attn_heads: usize,
    pub ff_dim: usize,
    pub max_seq_len: usize,
    pub tie_embeddings: bool,
    pub mamba: Mamba2Settings,
}

#[cfg(test)]
mod tests {
    use super::*;
    type Inner = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;
    type Train = burn::backend::Autodiff<Inner, burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing>;

    fn config(chunk: usize, mlp: bool, tie: bool) -> YumonMamba2BrainConfig {
        YumonMamba2BrainConfig::new(40, TrainingStage::Language, Mamba2Settings {
            d_state: 8, expand: 2, d_conv: 4, chunk_size: chunk, mlp })
            .with_embed_dim(16).with_n_layers(2).with_attn_heads(4).with_ff_dim(32)
            .with_max_seq_len(32).with_dropout_rate(0.0).with_tie_embeddings(tie)
    }
    fn values<B: Backend>(t: Tensor<B, 3>) -> Vec<f32> { t.into_data().convert::<f32>().to_vec().unwrap() }
    fn ids<B: Backend>(ids: &[i32], device: &B::Device) -> Tensor<B, 2, Int> {
        Tensor::from_ints(TensorData::new(ids.to_vec(), [1, ids.len()]), device)
    }
    fn close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        let worst = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(worst <= tol, "max difference {worst}");
    }

    /// Plain per-token recurrence on the CPU, the definition the chunked scan must match.
    #[test]
    fn chunked_ssd_matches_sequential_recurrence() {
        let device = Default::default();
        let (b, t, h, p, n) = (2, 11, 3, 4, 5);
        let rand = |shape: Vec<usize>| Tensor::<Inner, 1>::random([shape.iter().product::<usize>()],
            Distribution::Uniform(-1.0, 1.0), &device);
        let x = rand(vec![b, t, h, p]);
        let a = rand(vec![b, t, h]).abs().neg().mul_scalar(0.5);
        let (bm, cm, init) = (rand(vec![b, t, n]), rand(vec![b, t, n]), rand(vec![b, h, n, p]));
        let v = |t: &Tensor<Inner, 1>| t.clone().into_data().convert::<f32>().to_vec::<f32>().unwrap();
        let (xv, av, bv, cv, iv) = (v(&x), v(&a), v(&bm), v(&cm), v(&init));
        let mut expected = vec![0.0f32; b * t * h * p];
        let mut final_state = iv.clone();
        for bi in 0..b { for hi in 0..h {
            let mut s: Vec<f32> = (0..n * p).map(|i| iv[(bi * h + hi) * n * p + i]).collect();
            for ti in 0..t {
                let decay = av[(bi * t + ti) * h + hi].exp();
                for ni in 0..n { for pi in 0..p {
                    s[ni * p + pi] = decay * s[ni * p + pi]
                        + bv[(bi * t + ti) * n + ni] * xv[((bi * t + ti) * h + hi) * p + pi];
                }}
                for pi in 0..p {
                    expected[((bi * t + ti) * h + hi) * p + pi] =
                        (0..n).map(|ni| cv[(bi * t + ti) * n + ni] * s[ni * p + pi]).sum();
                }
            }
            final_state[(bi * h + hi) * n * p..(bi * h + hi + 1) * n * p].copy_from_slice(&s);
        }}
        for chunk in [1, 4, 11, 64] {
            let (y, last) = ssd(x.clone().reshape([b, t, h, p]), a.clone().reshape([b, t, h]),
                bm.clone().reshape([b, t, n]), cm.clone().reshape([b, t, n]),
                init.clone().reshape([b, h, n, p]), chunk);
            close(&y.into_data().convert::<f32>().to_vec::<f32>().unwrap(), &expected, 1e-4);
            close(&last.into_data().convert::<f32>().to_vec::<f32>().unwrap(), &final_state, 1e-4);
        }
    }

    #[test]
    fn param_count_matches_module() {
        let device = Default::default();
        for (mlp, tie) in [(true, false), (false, true)] {
            let config = config(4, mlp, tie);
            let model: YumonMamba2Brain<Inner> = config.init(&device);
            assert_eq!(model.num_params(), config.param_count());
        }
    }

    /// Prompt then single tokens through the carried state must equal one full pass,
    /// for chunk sizes that split the sequence unevenly.
    #[test]
    fn recurrent_state_matches_full_forward() {
        let device = Default::default();
        let tokens: Vec<i32> = (0..13).map(|i| 4 + (i * 7) % 36).collect();
        for chunk in [4, 64] {
            let model: YumonMamba2Brain<Inner> = config(chunk, true, true).init(&device);
            let full = values(model.forward(ids::<Inner>(&tokens, &device)));
            let mut state = Mamba2State { layers: Vec::new() };
            let mut stepped = Vec::new();
            let mut start = 0;
            for len in [5, 1, 1, 3, 1, 2] {
                stepped.extend(values(model.forward_state(ids::<Inner>(&tokens[start..start + len], &device), &mut state)));
                start += len;
            }
            close(&full, &stepped, 1e-4);
        }
    }

    #[test]
    fn causal_and_trains_at_batch_one_and_two() {
        let device = Default::default();
        let model: YumonMamba2Brain<Train> = config(4, true, false).init(&device);
        let a = values(model.forward(ids::<Train>(&[1, 4, 5, 6, 7, 8, 9, 2], &device)).slice([0..1, 0..3, 0..40]));
        let b = values(model.forward(ids::<Train>(&[1, 4, 5, 9, 8, 7, 6, 2], &device)).slice([0..1, 0..3, 0..40]));
        close(&a, &b, 1e-5);
        let batch = Tensor::<Train, 2, Int>::from_ints(TensorData::new((0..16).map(|i| i % 40).collect::<Vec<i32>>(), [2, 8]), &device);
        let logits = model.forward(batch);
        assert_eq!(logits.dims(), [2, 8, 40]);
        let grads = burn::optim::GradientsParams::from_grads(logits.sum().backward(), &model);
        assert!(grads.len() > 0);
    }
}
