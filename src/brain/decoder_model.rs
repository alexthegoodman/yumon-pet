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
    jl::{JlProjection, JlSet, JlSite},
    model::{GenerationResult, MLP, MLPConfig, RMSNorm, RMSNormConfig, TEMPERATURE, TOP_K},
    samples::{Action, CardinalDir, TrainingStage},
    tokenizer::{BOS_TOKEN, EOS_TOKEN, PAD_TOKEN},
};
use anyhow::Result;
use burn::{
    module::Ignored,
    nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig},
    prelude::*,
    record::{BinBytesRecorder, BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{FloatDType, activation::softmax},
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

/// The JL maps a block applies (see jl.rs). Residual maps sit between blocks
/// and are applied by the brain.
struct BlockJl<'a, B: Backend> {
    /// head_dim -> dim for q, k (after RoPE) and v.
    attention: Option<&'a JlProjection<B>>,
    /// Normed MLP input, width -> dim.
    ffn: Option<&'a JlProjection<B>>,
}
// Manual impls: a derive would require B: Copy.
impl<B: Backend> Clone for BlockJl<'_, B> {
    fn clone(&self) -> Self { *self }
}
impl<B: Backend> Copy for BlockJl<'_, B> {}
/// RoPE base, as in burn::nn::RotaryEncodingConfig's default.
const ROPE_THETA: f64 = 10_000.0;

/// cos/sin of position * theta^(-2i/dim) for pair i, shaped [1, 1, seq, dim/2, 1].
pub(super) struct RopeTables<B: Backend> {
    cos: Tensor<B, 5>,
    sin: Tensor<B, 5>,
}
impl<B: Backend> RopeTables<B> {
    pub(super) fn new(seq: usize, dim: usize, device: &B::Device) -> Self {
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

    /// Multimodal Rotary Position Embedding (M-RoPE) for (t, y, x) 3D coordinate trajectories.
    /// coords: [batch][seq] of (t, y, x) coordinates.
    pub fn new_mrope(
        coords: &[Vec<(f32, f32, f32)>],
        dim: usize,
        device: &B::Device,
    ) -> Self {
        let batch = coords.len();
        assert!(batch > 0, "mrope requires at least 1 batch sequence");
        let seq = coords[0].len();
        let half = dim / 2;
        assert!(half > 0, "mrope head dim must be at least 2");

        let (dt_pairs, dy_pairs, dx_pairs) = if half >= 3 {
            let dt = (half + 1) / 2;
            let rem = half - dt;
            let dy = rem / 2;
            let dx = rem - dy;
            if dy == 0 {
                (dt.saturating_sub(1), 1, 1)
            } else {
                (dt, dy, dx)
            }
        } else {
            (half, 0, 0)
        };

        let total = batch * seq * half;
        let mut cos_vec = Vec::with_capacity(total);
        let mut sin_vec = Vec::with_capacity(total);

        for b in 0..batch {
            assert_eq!(coords[b].len(), seq, "all batch sequences must have identical seq len");
            for pos in 0..seq {
                let (t, y, x) = coords[b][pos];

                // 1. T axis (temporal / sequence)
                for i in 0..dt_pairs {
                    let d = (2 * dt_pairs) as f64;
                    let angle = t as f64 * ROPE_THETA.powf(-((2 * i) as f64) / d);
                    cos_vec.push(angle.cos() as f32);
                    sin_vec.push(angle.sin() as f32);
                }

                // 2. Y axis (spatial height / row)
                for i in 0..dy_pairs {
                    let d = (2 * dy_pairs) as f64;
                    let angle = y as f64 * ROPE_THETA.powf(-((2 * i) as f64) / d);
                    cos_vec.push(angle.cos() as f32);
                    sin_vec.push(angle.sin() as f32);
                }

                // 3. X axis (spatial width / col)
                for i in 0..dx_pairs {
                    let d = (2 * dx_pairs) as f64;
                    let angle = x as f64 * ROPE_THETA.powf(-((2 * i) as f64) / d);
                    cos_vec.push(angle.cos() as f32);
                    sin_vec.push(angle.sin() as f32);
                }
            }
        }

        let table = |v: Vec<f32>| {
            Tensor::from_data(TensorData::new(v, [batch, 1, seq, half, 1]), device)
        };
        Self {
            cos: table(cos_vec),
            sin: table(sin_vec),
        }
    }
}

/// Rotates interleaved pairs (x[2i], x[2i+1]) of x: [batch, heads, seq, dim],
/// the same convention as burn::nn::RotaryEncoding.
pub(super) fn apply_rope<B: Backend>(x: Tensor<B, 4>, rope: &RopeTables<B>) -> Tensor<B, 4> {
    let [batch, heads, seq, dim] = x.dims();
    let pairs = x.reshape([batch, heads, seq, dim / 2, 2]);
    let x0 = pairs.clone().narrow(4, 0, 1);
    let x1 = pairs.narrow(4, 1, 1);
    let r0 = x0.clone() * rope.cos.clone() - x1.clone() * rope.sin.clone();
    let r1 = x0 * rope.sin.clone() + x1 * rope.cos.clone();
    Tensor::cat(vec![r0, r1], 4).reshape([batch, heads, seq, dim])
}

impl<B: Backend> RopeTables<B> {
    /// Positions `start..start + len` of this table.
    pub(super) fn slice(&self, start: usize, len: usize) -> Self {
        Self { cos: self.cos.clone().narrow(2, start, len), sin: self.sin.clone().narrow(2, start, len) }
    }
}

/// One layer's cached keys (RoPE applied) and values, each [1, kv_heads, len, head_dim].
pub(super) type KvSlot<B> = Option<(Tensor<B, 4>, Tensor<B, 4>)>;

/// Keys and values of every token decoded so far, per layer. Generation feeds
/// the prompt once, then one token per step, instead of rerunning the prefix.
pub struct KvCache<B: Backend> {
    layers: Vec<KvSlot<B>>,
    len: usize,
    rope: RopeTables<B>,
    /// RoPE at the narrower head width of blocks after a JL residual projection.
    narrow_rope: Option<RopeTables<B>>,
}
impl<B: Backend> KvCache<B> {
    pub fn new(layers: usize, max_seq_len: usize, head_dim: usize, device: &B::Device) -> Self {
        Self { layers: (0..layers).map(|_| None).collect(), len: 0, rope: RopeTables::new(max_seq_len, head_dim, device), narrow_rope: None }
    }
    pub fn new_mrope(layers: usize, coords: &[Vec<(f32, f32, f32)>], head_dim: usize, device: &B::Device) -> Self {
        Self { layers: (0..layers).map(|_| None).collect(), len: 0, rope: RopeTables::new_mrope(coords, head_dim, device), narrow_rope: None }
    }
    pub(super) fn with_narrow_rope(mut self, rope: RopeTables<B>) -> Self {
        self.narrow_rope = Some(rope);
        self
    }
    /// RoPE for the next `len` positions.
    pub(super) fn next_rope(&self, len: usize) -> RopeTables<B> { self.rope.slice(self.len, len) }
    pub(super) fn next_narrow_rope(&self, len: usize) -> Option<RopeTables<B>> {
        self.narrow_rope.as_ref().map(|r| r.slice(self.len, len))
    }
    pub(super) fn layer(&mut self, layer: usize) -> &mut KvSlot<B> { &mut self.layers[layer] }
    pub(super) fn advance(&mut self, len: usize) { self.len += len; }
    pub fn len(&self) -> usize { self.len }
}

/// Grouped-query attention: each KV head serves `groups` consecutive query heads,
/// [batch, kv_heads, seq, dim] -> [batch, kv_heads * groups, seq, dim].
pub(super) fn repeat_kv<B: Backend>(x: Tensor<B, 4>, groups: usize) -> Tensor<B, 4> {
    if groups == 1 { return x; }
    let [batch, kv_heads, seq, dim] = x.dims();
    x.unsqueeze_dim::<5>(2).expand([batch, kv_heads, groups, seq, dim]).reshape([batch, kv_heads * groups, seq, dim])
}

/// Appends new keys/values to a layer's cache; returns all of them and how many came before.
pub(super) fn append_kv<B: Backend>(slot: &mut KvSlot<B>, k: Tensor<B, 4>, v: Tensor<B, 4>)
    -> (Tensor<B, 4>, Tensor<B, 4>, usize) {
    let (k, v, start) = match slot.take() {
        None => (k, v, 0),
        Some((past_k, past_v)) => {
            let start = past_k.dims()[2];
            (Tensor::cat(vec![past_k, k], 2), Tensor::cat(vec![past_v, v], 2), start)
        }
    };
    *slot = Some((k.clone(), v.clone()));
    (k, v, start)
}

/// Causal attention of new queries q [1, heads, t, dim] over the cached and new keys
/// (k, v: [1, kv_heads, t, dim] for the new positions, appended to `slot`). The prompt
/// runs through flash attention; later steps use grouped matmuls against the cache,
/// so the cache stays at kv_heads.
pub(super) fn cached_attention<B: FlashAttention>(
    q: Tensor<B, 4>, k: Tensor<B, 4>, v: Tensor<B, 4>, slot: &mut KvSlot<B>,
) -> Tensor<B, 4> {
    let [batch, heads, t, dim] = q.dims();
    assert_eq!(batch, 1, "KV-cached attention decodes one sequence");
    let kv_heads = k.dims()[1];
    let groups = heads / kv_heads;
    let (k, v, start) = append_kv(slot, k, v);
    if start == 0 {
        return causal_flash_attention(q, repeat_kv(k, groups), repeat_kv(v, groups));
    }
    let len = start + t;
    let dtype = q.dtype();
    let device = q.device();
    // Row g * t + i is query head g of the group at position start + i.
    let q = q.reshape([1, kv_heads, groups * t, dim]).cast(FloatDType::F32);
    let mut scores = q.matmul(k.cast(FloatDType::F32).swap_dims(2, 3)) / (dim as f64).sqrt();
    if t > 1 {
        let future: Vec<bool> = (0..groups * t).flat_map(|r| (0..len).map(move |c| c > start + r % t)).collect();
        let future = Tensor::<B, 2, Bool>::from_data(TensorData::new(future, [groups * t, len]), &device);
        scores = scores.mask_fill(future.unsqueeze::<4>(), -1e9);
    }
    softmax(scores, 3).matmul(v.cast(FloatDType::F32)).cast(dtype).reshape([1, heads, t, dim])
}

impl<B: FlashAttention> DecBlock<B> {
    /// Per-head q, k (RoPE applied) and v, [batch, heads, seq, dim], through the
    /// block's attention JL map if it has one.
    fn qkv(&self, x: &Tensor<B, 3>, rope: &RopeTables<B>, jl: BlockJl<B>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, seq, width] = x.dims();
        let hd = width / self.heads;
        let norm = self.attn_norm.forward(x.clone());
        let split = |t: Tensor<B, 3>| t.reshape([batch, seq, self.heads, hd]).swap_dims(1, 2);
        let q = apply_rope(split(self.q.forward(norm.clone())), rope);
        let k = apply_rope(split(self.k.forward(norm.clone())), rope);
        let v = split(self.v.forward(norm));
        match jl.attention {
            Some(p) => {
                // Attention scales scores by 1/sqrt(dim); this restores 1/sqrt(head_dim),
                // so (qR).(kR) is scored like the unprojected q.k it estimates.
                let rescale = (p.out_dim() as f64 / hd as f64).sqrt();
                (p.forward(q) * rescale, p.forward(k), p.forward(v))
            }
            None => (q, k, v),
        }
    }
    /// Residual add of the attention output [batch, heads, seq, dim], then the MLP
    /// (or `mlp` in its place). Returns the block output, MLP input and MLP output.
    fn finish(&self, x: Tensor<B, 3>, attn: Tensor<B, 4>, jl: BlockJl<B>, mlp: Option<&WirMlp<B>>)
        -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, heads, seq, dim] = attn.dims();
        let x = x + self.o.forward(attn.swap_dims(1, 2).reshape([batch, seq, heads * dim]));
        let mut mlp_in = self.ffn_norm.forward(x.clone());
        if let Some(p) = jl.ffn {
            mlp_in = p.forward(mlp_in);
        }
        let mlp_out = match mlp {
            Some(m) => m.forward(mlp_in.clone()),
            None => self.mlp.forward(mlp_in.clone()),
        };
        (x + mlp_out.clone(), mlp_in, mlp_out)
    }
    fn forward_cached(&self, x: Tensor<B, 3>, rope: &RopeTables<B>, jl: BlockJl<B>, slot: &mut KvSlot<B>) -> Tensor<B, 3> {
        let (q, k, v) = self.qkv(&x, rope, jl);
        self.finish(x, cached_attention(q, k, v, slot), jl, None).0
    }
    fn forward(&self, x: Tensor<B, 3>, rope: &RopeTables<B>, jl: BlockJl<B>) -> Tensor<B, 3> {
        let (q, k, v) = self.qkv(&x, rope, jl);
        self.finish(x, causal_flash_attention(q, k, v), jl, None).0
    }
}

/// A one-hidden-layer network standing in for a block's MLP, in nn.Linear layout:
/// w1 [h, d], b1 [h], w2 [d, h], b2 [d]. Hidden units are relu(w1 x + b1), or with
/// `up` = (w3 [h, d], b3 [h]) SwiGLU units silu(w1 x + b1) * (w3 x + b3). Used to run
/// WIR-compiled weights inside the real model (`wir_decoder`).
pub struct WirMlp<B: Backend> {
    pub w1: Tensor<B, 2>,
    pub b1: Tensor<B, 1>,
    pub w2: Tensor<B, 2>,
    pub b2: Tensor<B, 1>,
    pub up: Option<(Tensor<B, 2>, Tensor<B, 1>)>,
}
impl<B: Backend> WirMlp<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, width] = x.dims();
        let x = x.reshape([batch * seq, width]);
        let z = x.clone().matmul(self.w1.clone().transpose()) + self.b1.clone().unsqueeze();
        let h = match &self.up {
            None => burn::tensor::activation::relu(z),
            Some((w3, b3)) => {
                burn::tensor::activation::silu(z) * (x.matmul(w3.clone().transpose()) + b3.clone().unsqueeze())
            }
        };
        let y = h.matmul(self.w2.clone().transpose()) + self.b2.clone().unsqueeze();
        y.reshape([batch, seq, self.w2.dims()[0]])
    }
}

impl<B: FlashAttention> DecBlock<B> {
    /// `forward` with the MLP optionally replaced; also returns the MLP's input and output.
    /// With an FFN JL map, the MLP input is the projected one.
    fn forward_probe(&self, x: Tensor<B, 3>, rope: &RopeTables<B>, jl: BlockJl<B>, mlp: Option<&WirMlp<B>>)
        -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let (q, k, v) = self.qkv(&x, rope, jl);
        self.finish(x, causal_flash_attention(q, k, v), jl, mlp)
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
    /// JL maps (config.jl), in this order: one for `residual`, then one per
    /// affected block for `attention`, then for `ffn`. Older records load as
    /// `LegacyDecBrain`.
    jl: Vec<JlProjection<B>>,
    /// Factorized embedding (config.token_embed_dim): table width -> model width.
    embed_in: Option<Linear<B>>,
    /// Tied head when the head width differs from the table width: head -> table.
    /// The projections are the last fields so older records load as `JlDecBrain`.
    head_down: Option<Linear<B>>,
}

/// Tied-head logits `x E^T`: x [batch, seq, d], table E [vocab, d].
/// Burn 0.20 / CubeCL 0.9 matmul against the strided `E.transpose()` is wrong on
/// WGPU when d is not a multiple of 16 (d = 8 or 24 with vocab >= 4096: off by
/// ~0.6; 16 and 32 exact), so the flat reshape round trip copies it contiguous first.
pub(crate) fn tied_logits<B: Backend>(x: Tensor<B, 3>, table: Tensor<B, 2>) -> Tensor<B, 3> {
    let [batch, seq, d] = x.dims();
    let [vocab, _] = table.dims();
    let table_t = table.transpose().reshape([d * vocab]).reshape([d, vocab]);
    x.reshape([batch * seq, d]).matmul(table_t).reshape([batch, seq, vocab])
}

/// Record layout before the embedding projections.
#[derive(Module, Debug)]
struct JlDecBrain<B: Backend> {
    config: Ignored<YumonDecBrainConfig>,
    embedding: Embedding<B>,
    blocks: Vec<DecBlock<B>>,
    norm: RMSNorm<B>,
    dropout: Dropout,
    token_head: Linear<B>,
    jl: Vec<JlProjection<B>>,
}

/// Record layout before the JL experiment.
#[derive(Module, Debug)]
struct LegacyDecBrain<B: Backend> {
    config: Ignored<YumonDecBrainConfig>,
    embedding: Embedding<B>,
    blocks: Vec<DecBlock<B>>,
    norm: RMSNorm<B>,
    dropout: Dropout,
    token_head: Linear<B>,
}

/// Loads a record; the config picks the layout. A shorter layout also reads newer
/// records whose trailing fields are empty (bincode ignores the trailing bytes), so
/// the full layout is only needed with embedding projections, `JlDecBrain` with JL
/// maps, and `LegacyDecBrain` otherwise. (Try-then-fall-back is not an option:
/// BinBytesRecorder panics on a short record instead of returning an error.)
fn load_dec_record<B: Backend, R: Recorder<B>>(recorder: &R, args: R::LoadArgs,
    config: &YumonDecBrainConfig, device: &B::Device) -> Result<YumonDecBrainRecord<B>> {
    let err = |e| anyhow::anyhow!("load: {e:?}");
    if config.embed_in_dims().is_some() || config.head_down_dims().is_some() {
        return recorder.load(args, device).map_err(err);
    }
    if config.active_jl().is_some() {
        let old = recorder.load::<JlDecBrainRecord<B>>(args, device).map_err(err)?;
        return Ok(YumonDecBrainRecord {
            config: old.config, embedding: old.embedding, blocks: old.blocks, norm: old.norm,
            dropout: old.dropout, token_head: old.token_head, jl: old.jl, embed_in: None, head_down: None,
        });
    }
    let old = recorder.load::<LegacyDecBrainRecord<B>>(args, device).map_err(err)?;
    Ok(YumonDecBrainRecord {
        config: old.config, embedding: old.embedding, blocks: old.blocks, norm: old.norm,
        dropout: old.dropout, token_head: old.token_head, jl: Vec::new(), embed_in: None, head_down: None,
    })
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
    /// Johnson-Lindenstrauss projection experiment (jl.rs, docs/JL.md).
    #[config(default = "None")]
    pub jl: Option<JlSet>,
    /// Logits from the token embedding table (h E^T) instead of a separate head.
    #[config(default = false)]
    pub tie_embeddings: bool,
    /// Token embedding table width when it differs from `embed_dim` (the model
    /// width); a learned projection maps it to the model width.
    #[config(default = "None")]
    pub token_embed_dim: Option<usize>,
}

/// Widths of one block: residual, MLP hidden, per-head attention, MLP input.
struct BlockWidths { width: usize, ff: usize, head: usize, mlp_in: usize }

impl YumonDecBrainConfig {
    /// The JL config when it adds a projection.
    fn active_jl(&self) -> Option<&JlSet> { self.jl.as_ref().filter(|j| j.active()) }
    /// The active map at `site`.
    pub fn jl_site(&self, site: JlSite) -> Option<&crate::brain::jl::JlConfig> {
        self.active_jl().and_then(|j| j.site(site))
    }
    fn jl_from(&self, site: JlSite) -> Option<usize> {
        self.jl_site(site).map(|j| j.from_layer)
    }
    /// `site`'s target width when its map covers block `layer`.
    fn jl_covers(&self, site: JlSite, layer: usize) -> Option<usize> {
        self.jl_site(site).filter(|j| layer >= j.from_layer && layer < self.n_layers).map(|j| j.dim)
    }
    /// Residual width before block `layer` (`n_layers` = at the head).
    pub fn width_at(&self, layer: usize) -> usize {
        self.active_jl().map_or(self.embed_dim, |j| j.width_at(self.embed_dim, layer))
    }
    fn block_widths(&self, layer: usize) -> BlockWidths {
        let width = self.width_at(layer);
        let ff = match self.jl_site(JlSite::Residual) {
            Some(r) if width != self.embed_dim => r.narrow_ff_dim(self.embed_dim, self.ff_dim),
            _ => self.ff_dim,
        };
        let head = self.jl_covers(JlSite::Attention, layer).unwrap_or(width / self.attn_heads);
        let mlp_in = self.jl_covers(JlSite::Ffn, layer).unwrap_or(width);
        BlockWidths { width, ff, head, mlp_in }
    }
    /// Width of the token embedding table.
    pub fn effective_token_embed_dim(&self) -> usize {
        self.token_embed_dim.unwrap_or(self.embed_dim)
    }
    /// (from, to) of the table -> model width projection, when factorized.
    fn embed_in_dims(&self) -> Option<(usize, usize)> {
        let table = self.effective_token_embed_dim();
        (table != self.embed_dim).then_some((table, self.embed_dim))
    }
    /// (from, to) of the head -> table projection: tied, and the head width
    /// (embed_dim, or a residual JL map's width) differs from the table's.
    fn head_down_dims(&self) -> Option<(usize, usize)> {
        let (head, table) = (self.width_at(self.n_layers), self.effective_token_embed_dim());
        (self.tie_embeddings && head != table).then_some((head, table))
    }
    /// Maps per site in `YumonDecBrain.jl`: residual, attention, ffn.
    fn jl_counts(&self) -> [usize; 3] {
        let per_block = |site| self.jl_from(site).map_or(0, |from| self.n_layers - from);
        [self.jl_from(JlSite::Residual).is_some() as usize, per_block(JlSite::Attention), per_block(JlSite::Ffn)]
    }
    pub fn init<B: Backend>(&self, device: &B::Device) -> YumonDecBrain<B> {
        assert!(self.n_layers > 0 && self.max_seq_len > 0 && self.vocab_size > 0);
        assert!(self.attn_heads > 0 && self.embed_dim > 0 && self.embed_dim % self.attn_heads == 0);
        assert!(
            (self.embed_dim / self.attn_heads) % 2 == 0,
            "RoPE requires even head width"
        );
        if let Some(jl) = &self.jl {
            jl.validate(self.embed_dim, self.attn_heads, self.n_layers).expect("invalid jl config");
        }
        let linear = |from: usize, to: usize| LinearConfig::new(from, to).with_bias(false).init(device);
        let mut jl = Vec::new();
        if let Some(set) = self.active_jl() {
            if let Some(from) = self.jl_from(JlSite::Residual) {
                jl.push(set.projection(JlSite::Residual, self.embed_dim, from, device));
            }
            if let Some(from) = self.jl_from(JlSite::Attention) {
                jl.extend((from..self.n_layers)
                    .map(|l| set.projection(JlSite::Attention, self.width_at(l) / self.attn_heads, l, device)));
            }
            if let Some(from) = self.jl_from(JlSite::Ffn) {
                jl.extend((from..self.n_layers).map(|l| set.projection(JlSite::Ffn, self.width_at(l), l, device)));
            }
        }
        let out_width = self.width_at(self.n_layers);
        // N(0, 1/fan_in) keeps the activation scale through each projection.
        let projection = |(from, to): (usize, usize)| LinearConfig::new(from, to)
            .with_initializer(burn::nn::Initializer::Normal { mean: 0.0, std: 1.0 / (from as f64).sqrt() })
            .with_bias(false)
            .init(device);
        let embedding = EmbeddingConfig::new(self.vocab_size, self.effective_token_embed_dim());
        // A tied table is also the head: N(0, 0.02) keeps the first logits small.
        let embedding = if self.tie_embeddings {
            embedding.with_initializer(burn::nn::Initializer::Normal { mean: 0.0, std: 0.02 })
        } else {
            embedding
        };
        YumonDecBrain {
            config: Ignored(self.clone()),
            embedding: embedding.init(device),
            blocks: (0..self.n_layers)
                .map(|layer| {
                    let w = self.block_widths(layer);
                    DecBlock {
                        attn_norm: RMSNormConfig::new(w.width).init(device),
                        q: linear(w.width, w.width),
                        k: linear(w.width, w.width),
                        v: linear(w.width, w.width),
                        o: linear(self.attn_heads * w.head, w.width),
                        ffn_norm: RMSNormConfig::new(w.width).init(device),
                        mlp: MLPConfig::new(w.width, w.ff).with_d_input(Some(w.mlp_in)).init(device),
                        heads: self.attn_heads,
                    }
                })
                .collect(),
            norm: RMSNormConfig::new(out_width).init(device),
            dropout: DropoutConfig::new(self.dropout_rate).init(),
            // Unused 1 x 1 placeholder when tied.
            token_head: if self.tie_embeddings {
                LinearConfig::new(1, 1).init(device)
            } else {
                LinearConfig::new(out_width, self.vocab_size).init(device)
            },
            jl,
            embed_in: self.embed_in_dims().map(projection),
            head_down: self.head_down_dims().map(projection),
        }
    }
    /// Weights in the model, embeddings, head and JL maps included.
    pub fn param_count(&self) -> usize {
        let (v, h) = (self.vocab_size, self.attn_heads);
        let blocks: usize = (0..self.n_layers).map(|layer| {
            let w = self.block_widths(layer);
            3 * w.width * w.width + h * w.head * w.width + 2 * w.width + 2 * w.mlp_in * w.ff + w.ff * w.width
        }).sum();
        let out = self.width_at(self.n_layers);
        // Tied: only the 1 x 1 placeholder (weight and bias).
        let head = if self.tie_embeddings { 2 } else { out * v + v };
        let projections: usize = [self.embed_in_dims(), self.head_down_dims()].into_iter().flatten()
            .map(|(from, to)| from * to).sum();
        v * self.effective_token_embed_dim() + projections + blocks + out + head + self.jl_param_count()
    }
    /// Weights in the JL maps (frozen unless jl.trainable).
    pub fn jl_param_count(&self) -> usize {
        let residual = self.jl_site(JlSite::Residual).map_or(0, |r| self.embed_dim * r.dim);
        let blocks: usize = (0..self.n_layers).map(|l| {
            let width = self.width_at(l);
            self.jl_covers(JlSite::Attention, l).map_or(0, |k| width / self.attn_heads * k)
                + self.jl_covers(JlSite::Ffn, l).map_or(0, |k| width * k)
        }).sum();
        residual + blocks
    }
    /// Per-head width of blocks after a residual JL map, if any.
    fn narrow_head_dim(&self) -> Option<usize> {
        self.jl_site(JlSite::Residual).filter(|r| r.from_layer < self.n_layers).map(|r| r.dim / self.attn_heads)
    }
}
impl<B: FlashAttention> YumonDecBrain<B> {
    /// Logits with any block's MLP replaced by a ReLU network, plus every block's
    /// (MLP input, MLP output) as run. No dropout. `mlps` is indexed by layer.
    pub fn forward_probe(&self, tokens: Tensor<B, 2, Int>, mlps: &[Option<WirMlp<B>>])
        -> (Tensor<B, 3>, Vec<(Tensor<B, 3>, Tensor<B, 3>)>) {
        let [_, seq] = tokens.dims();
        let ropes = self.ropes(seq, &tokens.device());
        let mut x = self.embed(tokens);
        let mut trace = Vec::with_capacity(self.blocks.len());
        for (layer, block) in self.blocks.iter().enumerate() {
            x = self.residual_jl(layer, x);
            let (next, mlp_in, mlp_out) = block.forward_probe(x, ropes.at(layer), self.block_jl(layer),
                mlps.get(layer).and_then(Option::as_ref));
            trace.push((mlp_in, mlp_out));
            x = next;
        }
        x = self.residual_jl(self.blocks.len(), x);
        (self.head(x), trace)
    }

    /// Token embeddings at model width (through `embed_in` when factorized).
    fn embed(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let x = self.embedding.forward(tokens);
        match &self.embed_in {
            Some(up) => up.forward(x),
            None => x,
        }
    }
    /// Final norm, then the head (or the embedding table when tied).
    fn head(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.norm.forward(x);
        if !self.config.tie_embeddings {
            return self.token_head.forward(x);
        }
        tied_logits(self.tied_features(x), self.embedding.weight.val())
    }
    /// Normed hidden states at table width (through `head_down` when factorized).
    fn tied_features(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match &self.head_down {
            Some(down) => down.forward(x),
            None => x,
        }
    }

    /// Grows a tied table to `vocab` rows. New rows start at the mean of the old rows
    /// plus N(0, `noise`) (a fixed-seed generator), so new tokens begin as an
    /// average token instead of a random direction. Used to add special tokens to a
    /// pretrained decoder: every old id keeps its row.
    pub fn grow_vocab(mut self, vocab: usize, noise: f32, device: &B::Device) -> Self {
        assert!(self.config.tie_embeddings, "grow_vocab needs a tied table (the head is the table)");
        let [old, width] = self.embedding.weight.dims();
        assert!(vocab >= old, "cannot shrink the table");
        if vocab == old { return self; }
        // Detached: on an autodiff backend the new Param must start from untracked tensors.
        let table = self.embedding.weight.val().detach();
        let mean = table.clone().mean_dim(0);
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut unit = || {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let extra: Vec<f32> = (0..(vocab - old) * width).map(|_| {
            // Box-Muller
            let (a, b) = (unit().max(1e-12), unit());
            ((-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()) as f32 * noise
        }).collect();
        let noise = Tensor::<B, 2>::from_data(TensorData::new(extra, [vocab - old, width]), device);
        let rows = mean.expand([vocab - old, width]) + noise;
        self.embedding.weight = burn::module::Param::from_tensor(Tensor::cat(vec![table, rows], 0).detach());
        self.config.0.vocab_size = vocab;
        self
    }

    /// Hidden states -> logits over only `rows` of the tied table: [batch, seq, rows.len()].
    /// Column j is token `rows[j]`. Lets a specialist write from a small slice of the
    /// vocabulary while still reading all of it.
    pub fn logits_rows(&self, x_tokens: Tensor<B, 2, Int>, rows: Tensor<B, 1, Int>) -> Tensor<B, 3> {
        assert!(self.config.tie_embeddings, "logits_rows needs a tied table");
        let [_, seq] = x_tokens.dims();
        assert!(seq > 0 && seq <= self.config.max_seq_len);
        let ropes = self.ropes(seq, &x_tokens.device());
        let mut x = self.dropout.forward(self.embed(x_tokens));
        for (layer, block) in self.blocks.iter().enumerate() {
            x = self.residual_jl(layer, x);
            x = block.forward(x, ropes.at(layer), self.block_jl(layer));
        }
        x = self.residual_jl(self.blocks.len(), x);
        self.rows_head(x, rows)
    }
    fn rows_head(&self, x: Tensor<B, 3>, rows: Tensor<B, 1, Int>) -> Tensor<B, 3> {
        let x = self.tied_features(self.norm.forward(x));
        tied_logits(x, self.embedding.weight.val().select(0, rows))
    }

    /// Greedy decoding restricted to `rows`: feeds `prompt` once, then one token per
    /// step through the KV cache, until `stop` or `max_new` tokens. Returns the token ids
    /// written (without `stop`). `rows` must contain `stop`.
    pub fn greedy_rows(&self, prompt: &[u32], rows: &[u32], stop: u32, max_new: usize,
        device: &B::Device) -> Vec<u32> {
        assert!(!prompt.is_empty() && prompt.len() < self.config.max_seq_len);
        let row_ids = Tensor::<B, 1, Int>::from_ints(
            rows.iter().map(|&r| r as i32).collect::<Vec<_>>().as_slice(), device);
        let mut cache = self.new_kv_cache(device);
        let mut input: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
        let mut out = Vec::new();
        while out.len() < max_new && cache.len() + input.len() <= self.config.max_seq_len {
            let t = input.len();
            let tokens = Tensor::<B, 2, Int>::from_ints(TensorData::new(input.clone(), [1, t]), device);
            let hidden = self.hidden_cached(tokens, &mut cache);
            let last = hidden.narrow(1, t - 1, 1);
            let logits = self.rows_head(last, row_ids.clone());
            let best = logits.argmax(2).into_data().convert::<i64>().to_vec::<i64>().unwrap()[0] as usize;
            let token = rows[best];
            if token == stop { break; }
            out.push(token);
            input = vec![token as i32];
        }
        out
    }
    /// `forward_cached` up to the final residual (before the norm and head).
    fn hidden_cached(&self, tokens: Tensor<B, 2, Int>, cache: &mut KvCache<B>) -> Tensor<B, 3> {
        let [_, t] = tokens.dims();
        assert!(t > 0 && cache.len() + t <= self.config.max_seq_len);
        let ropes = Ropes {
            wide: cache.next_rope(t),
            narrow: cache.next_narrow_rope(t),
            from: self.config.jl_from(JlSite::Residual).unwrap_or(usize::MAX),
        };
        let mut x = self.embed(tokens);
        for (layer, block) in self.blocks.iter().enumerate() {
            x = self.residual_jl(layer, x);
            x = block.forward_cached(x, ropes.at(layer), self.block_jl(layer), cache.layer(layer));
        }
        x = self.residual_jl(self.blocks.len(), x);
        cache.advance(t);
        x
    }

    /// The JL map block `layer` applies itself.
    fn block_jl(&self, layer: usize) -> BlockJl<'_, B> {
        let [residual, attention, _] = self.config.jl_counts();
        let map = |site, offset| self.config.jl_from(site).filter(|&from| layer >= from)
            .map(|from| &self.jl[offset + layer - from]);
        BlockJl { attention: map(JlSite::Attention, residual), ffn: map(JlSite::Ffn, residual + attention) }
    }
    /// Applies the residual JL map when it sits before block `layer`
    /// (`n_layers` = before the final norm).
    fn residual_jl(&self, layer: usize, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self.config.jl_from(JlSite::Residual) {
            Some(from) if from == layer => self.jl[0].forward(x),
            _ => x,
        }
    }
    fn ropes(&self, seq: usize, device: &B::Device) -> Ropes<B> {
        Ropes {
            wide: RopeTables::new(seq, self.config.embed_dim / self.config.attn_heads, device),
            narrow: self.config.narrow_head_dim().map(|hd| RopeTables::new(seq, hd, device)),
            from: self.config.jl_from(JlSite::Residual).unwrap_or(usize::MAX),
        }
    }

    /// Block `layer`'s SwiGLU weights (w1, w3, w2), Burn `[in, out]` layout.
    pub fn mlp_weights(&self, layer: usize) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        self.blocks[layer].mlp.swiglu_weights()
    }

    /// Logits [batch, seq, vocab]. Right-padded input only (see module docs).
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_, seq] = tokens.dims();
        assert!(seq > 0 && seq <= self.config.max_seq_len);
        let ropes = self.ropes(seq, &tokens.device());
        let mut x = self.dropout.forward(self.embed(tokens));
        for (layer, block) in self.blocks.iter().enumerate() {
            x = self.residual_jl(layer, x);
            x = block.forward(x, ropes.at(layer), self.block_jl(layer));
        }
        x = self.residual_jl(self.blocks.len(), x);
        self.head(x)
    }

    /// Logits [1, t, vocab] for `t` new tokens after those already in `cache`.
    pub(crate) fn forward_cached(&self, tokens: Tensor<B, 2, Int>, cache: &mut KvCache<B>) -> Tensor<B, 3> {
        let [_, t] = tokens.dims();
        assert!(t > 0 && cache.len() + t <= self.config.max_seq_len);
        let ropes = Ropes {
            wide: cache.next_rope(t),
            narrow: cache.next_narrow_rope(t),
            from: self.config.jl_from(JlSite::Residual).unwrap_or(usize::MAX),
        };
        let mut x = self.embed(tokens);
        for (layer, block) in self.blocks.iter().enumerate() {
            x = self.residual_jl(layer, x);
            x = block.forward_cached(x, ropes.at(layer), self.block_jl(layer), cache.layer(layer));
        }
        x = self.residual_jl(self.blocks.len(), x);
        cache.advance(t);
        self.head(x)
    }

    pub fn new_kv_cache(&self, device: &B::Device) -> KvCache<B> {
        let cache = KvCache::new(self.blocks.len(), self.config.max_seq_len,
            self.config.embed_dim / self.config.attn_heads, device);
        match self.config.narrow_head_dim() {
            Some(hd) => cache.with_narrow_rope(RopeTables::new(self.config.max_seq_len, hd, device)),
            None => cache,
        }
    }

    pub fn new_kv_cache_mrope(&self, coords: &[Vec<(f32, f32, f32)>], device: &B::Device) -> KvCache<B> {
        let cache = KvCache::new_mrope(self.blocks.len(), coords,
            self.config.embed_dim / self.config.attn_heads, device);
        match self.config.narrow_head_dim() {
            Some(hd) => cache.with_narrow_rope(RopeTables::new_mrope(coords, hd, device)),
            None => cache,
        }
    }

    pub fn generate_unmasked_parsed(
        &self,
        tokenizer: &TokenizerKind,
        seed_text: &str,
        max_tokens: usize,
        device: &B::Device,
    ) -> GenerationResult {
        let mut cache = self.new_kv_cache(device);
        generate_causal_cached(
            |tokens| self.forward_cached(tokens, &mut cache),
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
        let config = metadata.config();
        let record = load_dec_record(&recorder, dir.join("model"), &config, device)?;
        let model = config.init::<B>(device).load_record(record);
        Ok((model, tokenizer, config))
    }

    pub fn load_from_bytes(
        model_bytes: &[u8],
        tokenizer_bytes: &[u8],
        metadata_json: &str,
        device: &B::Device,
    ) -> Result<(Self, TokenizerKind, YumonDecBrainConfig)> {
        let metadata: DecMetadata = serde_json::from_str(metadata_json)?;
        let tokenizer = TokenizerKind::Bpe(BpeTokenizer::load_from_bytes(tokenizer_bytes)?);
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let config = metadata.config();
        let record = load_dec_record(&recorder, model_bytes.to_vec(), &config, device)?;
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jl: Option<JlSet>,
    /// Absent in checkpoints from before decoder tying (all untied).
    #[serde(default)]
    pub tie_embeddings: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_embed_dim: Option<usize>,
}
impl DecMetadata {
    pub fn config(&self) -> YumonDecBrainConfig {
        YumonDecBrainConfig {
            vocab_size: self.vocab_size,
            training_stage: self.training_stage,
            dropout_rate: self.dropout_rate,
            embed_dim: self.embed_dim,
            hidden_units: self.hidden_units,
            n_layers: self.n_layers,
            attn_heads: self.attn_heads,
            ff_dim: self.ff_dim,
            max_seq_len: self.max_seq_len,
            jl: self.jl.clone(),
            tie_embeddings: self.tie_embeddings,
            token_embed_dim: self.token_embed_dim,
        }
    }
}

/// RoPE tables for the wide blocks and, after a residual JL map, the narrow ones.
struct Ropes<B: Backend> {
    wide: RopeTables<B>,
    narrow: Option<RopeTables<B>>,
    /// First narrow block.
    from: usize,
}
impl<B: Backend> Ropes<B> {
    fn at(&self, layer: usize) -> &RopeTables<B> {
        if layer >= self.from { self.narrow.as_ref().unwrap_or(&self.wide) } else { &self.wide }
    }
}

// ─── Generation (shared with the MoE decoder) ────────────────────────────────

/// Top-k sampling from BOS + prompt + separator until EOS/PAD, the token
/// budget, or `max_seq_len`, then JSON-repair and field extraction.
/// No KV cache: each step reruns the full prefix (models with learned positions).
pub(crate) fn generate_causal<B: Backend>(
    forward: impl Fn(Tensor<B, 2, Int>) -> Tensor<B, 3>,
    training_stage: TrainingStage,
    max_seq_len: usize,
    tokenizer: &TokenizerKind,
    seed_text: &str,
    max_tokens: usize,
    device: &B::Device,
) -> GenerationResult {
    generate_with(|ids, _| forward(ids_tensor(ids, device)), training_stage, max_seq_len, tokenizer, seed_text, max_tokens)
}

/// `generate_causal` for KV-cached models: `step` gets only the tokens it has
/// not seen (the whole prompt first, then one token) and returns their logits.
pub(crate) fn generate_causal_cached<B: Backend>(
    mut step: impl FnMut(Tensor<B, 2, Int>) -> Tensor<B, 3>,
    training_stage: TrainingStage,
    max_seq_len: usize,
    tokenizer: &TokenizerKind,
    seed_text: &str,
    max_tokens: usize,
    device: &B::Device,
) -> GenerationResult {
    generate_with(|ids, done| step(ids_tensor(&ids[done..], device)), training_stage, max_seq_len, tokenizer, seed_text, max_tokens)
}

fn ids_tensor<B: Backend>(ids: &[usize], device: &B::Device) -> Tensor<B, 2, Int> {
    Tensor::from_ints(TensorData::new(ids.iter().map(|&t| t as i32).collect::<Vec<_>>(), [1, ids.len()]), device)
}

/// Sampling loop shared by both: `step(ids, done)` returns logits whose last row
/// predicts the token after `ids`; the first `done` ids went to earlier steps.
fn generate_with<B: Backend>(
    mut step: impl FnMut(&[usize], usize) -> Tensor<B, 3>,
    training_stage: TrainingStage,
    max_seq_len: usize,
    tokenizer: &TokenizerKind,
    seed_text: &str,
    max_tokens: usize,
) -> GenerationResult {
    let mut dec_ids: Vec<usize> = vec![BOS_TOKEN];
    if !seed_text.is_empty() {
        dec_ids.extend(tokenizer.encode(seed_text).iter().map(|&t| t as usize));
    }

    let sep_text = if training_stage == TrainingStage::Structured {
        "\n---\n"
    } else {
        crate::brain::samples::language_separator_for(tokenizer)
    };
    let sep_tokens = tokenizer.encode(sep_text);
    dec_ids.extend(sep_tokens.iter().map(|&t| t as usize));

    let prompt_len = dec_ids.len();
    let mut rng = rand::thread_rng();
    let mut done = 0;

    for _ in 0..max_tokens {
        let current_len = dec_ids.len();
        if current_len >= max_seq_len {
            break;
        }

        let token_logits = step(&dec_ids, done);
        done = current_len;

        let rows = token_logits.dims()[1];
        let vocab_size = tokenizer.vocab_size();
        let last_logits = token_logits
            .slice([0..1, rows - 1..rows, 0..vocab_size])
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
        type AD = crate::brain::train::CudaBf16TrainBackend;
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
        type AD = crate::brain::train::CudaBf16TrainBackend;
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
            jl: None,
            tie_embeddings: false,
            token_embed_dim: None,
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

    #[test]
    fn decoder_kv_cache_matches_full_forward() {
        let device = Default::default();
        let model: YumonDecBrain<Inner> = tiny_config().with_max_seq_len(16).init(&device);
        let ids: Vec<i32> = (0..12).map(|i| 4 + (i * 5) % 28).collect();
        let tensor = |ids: &[i32]| Tensor::<Inner, 2, Int>::from_ints(TensorData::new(ids.to_vec(), [1, ids.len()]), &device);
        let full = values(model.forward(tensor(&ids)));
        let config = &model.config;
        let mut cache = KvCache::new(config.n_layers, config.max_seq_len, config.embed_dim / config.attn_heads, &device);
        let mut cached = Vec::new();
        let mut start = 0;
        for n in [4, 1, 1, 3, 1, 2] {
            cached.extend(values(model.forward_cached(tensor(&ids[start..start + n]), &mut cache)));
            start += n;
        }
        assert_close(cached, full, 1e-4, "cached logits");
    }
    fn jl_configs() -> Vec<YumonDecBrainConfig> {
        // tiny_config: width 16, 2 heads (head_dim 8), 2 layers.
        // Stacked: after residual:8 the heads are 4 wide and the MLP input 8.
        ["residual:8@0", "residual:8@1", "residual:8@2", "attention:4", "attention:4@1", "ffn:8", "ffn:8@1+learned", "none",
            "residual:8@0,ffn:4", "residual:8@1,attention:2,ffn:4@1", "attention:4+learned,ffn:8+learned"]
            .iter()
            .map(|spec| tiny_config().with_max_seq_len(16).with_jl(Some(JlSet::parse(spec).unwrap())))
            .collect()
    }

    #[test]
    fn jl_param_count_and_cache() {
        let device = Default::default();
        let ids: Vec<i32> = (0..12).map(|i| 4 + (i * 5) % 28).collect();
        let tensor = |ids: &[i32]| Tensor::<Inner, 2, Int>::from_ints(TensorData::new(ids.to_vec(), [1, ids.len()]), &device);
        for config in jl_configs() {
            let what = config.jl.as_ref().unwrap().tag();
            let model: YumonDecBrain<Inner> = config.init(&device);
            assert_eq!(model.num_params(), config.param_count(), "{what}");
            let full = values(model.forward(tensor(&ids)));
            assert_eq!(full.len(), 12 * 32, "{what}");
            let mut cache = model.new_kv_cache(&device);
            let mut cached = Vec::new();
            let mut start = 0;
            for n in [4, 1, 1, 3, 1, 2] {
                cached.extend(values(model.forward_cached(tensor(&ids[start..start + n]), &mut cache)));
                start += n;
            }
            assert_close(cached, full, 1e-4, &format!("cached logits {what}"));
        }
    }

    /// Frozen maps get no gradient; `+learned` ones and every other weight do.
    #[test]
    fn jl_frozen_unless_trainable() {
        let device = Default::default();
        for config in jl_configs() {
            let jl = config.jl.clone().unwrap();
            let trainable = jl.0.iter().any(|j| j.trainable);
            let model: YumonDecBrain<Balanced> = config.init(&device);
            let grads = model.forward(Tensor::from_ints([[1, 4, 5, 6]], &device)).powf_scalar(2.0).mean().backward();
            for map in &model.jl {
                assert_eq!(map.weight.val().grad(&grads).is_some(), trainable, "{}", jl.tag());
            }
            assert!(model.token_head.weight.val().grad(&grads).is_some());
            // An optimizer step leaves frozen maps alone.
            let before: Vec<Vec<f32>> = model.jl.iter().map(|m| values(m.weight.val().inner())).collect();
            let params = GradientsParams::from_grads(grads, &model);
            let updated = AdamWConfig::new().init().step(1e-2, model, params);
            for (map, old) in updated.jl.iter().zip(before) {
                let changed = values(map.weight.val().inner()) != old;
                assert_eq!(changed, trainable, "{}", jl.tag());
            }
        }
    }

    /// Records saved before the JL field existed still load (no maps).
    #[test]
    fn decoder_loads_pre_jl_records() {
        let device = Default::default();
        let tokenizer = TokenizerKind::Bpe(BpeTokenizer::load("yumon_bpe").unwrap());
        let mut config = tiny_config();
        config.vocab_size = tokenizer.vocab_size();
        let model: YumonDecBrain<Inner> = config.init(&device);
        let legacy = LegacyDecBrain {
            config: model.config.clone(), embedding: model.embedding.clone(), blocks: model.blocks.clone(),
            norm: model.norm.clone(), dropout: model.dropout.clone(), token_head: model.token_head.clone(),
        };
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let tokenizer_bytes = match &tokenizer {
            TokenizerKind::Bpe(t) => t.inner.to_string(false).unwrap().into_bytes(),
            _ => unreachable!(),
        };
        let meta = |jl: Option<JlSet>| serde_json::json!({
            "dropout_rate": 0.0, "vocab_size": config.vocab_size, "epochs_trained": 1, "final_loss": 1.0,
            "batch_size": 1, "training_stage": config.training_stage, "embed_dim": config.embed_dim,
            "hidden_units": config.hidden_units, "n_layers": config.n_layers, "attn_heads": config.attn_heads,
            "ff_dim": config.ff_dim, "max_seq_len": config.max_seq_len, "jl": jl,
        }).to_string();
        let input = || Tensor::<Inner, 2, Int>::from_ints([[1, 4, 5]], &device);
        let bytes = recorder.record(legacy.into_record(), ()).unwrap();
        let (restored, _, _) = YumonDecBrain::<Inner>::load_from_bytes(&bytes, &tokenizer_bytes, &meta(None), &device).unwrap();
        assert!(restored.jl.is_empty());
        assert_close(values(model.forward(input())), values(restored.forward(input())), 1e-6, "legacy load");
        // A JL model round-trips with its frozen map.
        let jl = JlSet::parse("residual:8@1,ffn:4@1").unwrap();
        let model: YumonDecBrain<Inner> = config.clone().with_jl(Some(jl.clone())).init(&device);
        let bytes = recorder.record(model.clone().into_record(), ()).unwrap();
        let (restored, _, restored_config) = YumonDecBrain::<Inner>::load_from_bytes(&bytes, &tokenizer_bytes, &meta(Some(jl.clone())), &device).unwrap();
        assert_eq!(restored_config.jl, Some(jl));
        assert_close(values(model.forward(input())), values(restored.forward(input())), 1e-6, "jl roundtrip");
    }

    /// `tied_logits` is exact at widths that are not a multiple of 16 (where a
    /// matmul against the strided transpose is wrong on WGPU).
    #[test]
    fn tied_logits_match_reference() {
        let device = Default::default();
        let (m, v) = (12usize, 4096usize);
        for k in [8usize, 16, 24] {
            let xv: Vec<f32> = (0..m * k).map(|i| ((i * 7919 % 101) as f32 / 101.0) - 0.5).collect();
            let wv: Vec<f32> = (0..v * k).map(|i| ((i * 104729 % 211) as f32 / 211.0) - 0.5).collect();
            let x = Tensor::<Inner, 3>::from_data(TensorData::new(xv.clone(), [1, m, k]), &device);
            let w = Tensor::<Inner, 2>::from_data(TensorData::new(wv.clone(), [v, k]), &device);
            let reference: Vec<f32> = (0..m)
                .flat_map(|i| { let (xv, wv) = (&xv, &wv); (0..v).map(move |j| (0..k).map(|c| xv[i * k + c] * wv[j * k + c]).sum::<f32>()) })
                .collect();
            assert_close(values(tied_logits(x, w)), reference, 1e-5, &format!("tied logits width {k}"));
        }
    }

    /// Tied and factorized embeddings (with and without a residual JL map): param
    /// count, cached decoding, gradients into the table and a checkpoint round trip.
    #[test]
    fn decoder_tied_and_factorized_embeddings() {
        let device = Default::default();
        let tokenizer = TokenizerKind::Bpe(BpeTokenizer::load("yumon_bpe").unwrap());
        let tokenizer_bytes = match &tokenizer {
            TokenizerKind::Bpe(t) => t.inner.to_string(false).unwrap().into_bytes(),
            _ => unreachable!(),
        };
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let ids: Vec<i32> = (0..12).map(|i| 4 + (i * 5) % 28).collect();
        let tensor = |ids: &[i32]| Tensor::<Inner, 2, Int>::from_ints(TensorData::new(ids.to_vec(), [1, ids.len()]), &device);
        // tiny_config: width 16.
        let cases = [(true, None, None), (true, Some(24), None), (false, Some(24), None), (true, Some(8), None),
            (true, None, Some("residual:8@1")), (true, Some(24), Some("residual:8@0"))];
        for (tie, table, jl) in cases {
            let what = format!("tie {tie}, table {table:?}, jl {jl:?}");
            let mut config = tiny_config().with_max_seq_len(16).with_tie_embeddings(tie).with_token_embed_dim(table)
                .with_jl(jl.map(|s| JlSet::parse(s).unwrap()));
            config.vocab_size = tokenizer.vocab_size();
            let model: YumonDecBrain<Inner> = config.init(&device);
            assert_eq!(model.num_params(), config.param_count(), "{what}");
            assert_eq!(model.embedding.weight.dims(), [config.vocab_size, table.unwrap_or(16)], "{what}");
            let full = values(model.forward(tensor(&ids)));
            assert_eq!(full.len(), 12 * config.vocab_size, "{what}");
            let mut cache = model.new_kv_cache(&device);
            let mut cached = Vec::new();
            let mut start = 0;
            for n in [4, 1, 1, 3, 1, 2] {
                cached.extend(values(model.forward_cached(tensor(&ids[start..start + n]), &mut cache)));
                start += n;
            }
            assert_close(cached, full.clone(), 1e-4, &format!("cached logits {what}"));
            // The tied table gets gradient from the head; the placeholder gets none.
            let trained: YumonDecBrain<Balanced> = config.init(&device);
            let grads = trained.forward(Tensor::from_ints([[1, 4, 5, 6]], &device)).powf_scalar(2.0).mean().backward();
            assert!(trained.embedding.weight.val().grad(&grads).is_some(), "{what}");
            assert_eq!(trained.token_head.weight.val().grad(&grads).is_some(), !tie, "{what}");
            for proj in trained.embed_in.iter().chain(&trained.head_down) {
                assert!(proj.weight.val().grad(&grads).is_some(), "{what}");
            }
            let meta = serde_json::json!({
                "dropout_rate": 0.0, "vocab_size": config.vocab_size, "epochs_trained": 1, "final_loss": 1.0,
                "batch_size": 1, "training_stage": config.training_stage, "embed_dim": config.embed_dim,
                "hidden_units": config.hidden_units, "n_layers": config.n_layers, "attn_heads": config.attn_heads,
                "ff_dim": config.ff_dim, "max_seq_len": config.max_seq_len, "jl": config.jl,
                "tie_embeddings": tie, "token_embed_dim": table,
            }).to_string();
            let bytes = recorder.record(model.clone().into_record(), ()).unwrap();
            let (restored, _, restored_config) = YumonDecBrain::<Inner>::load_from_bytes(&bytes, &tokenizer_bytes, &meta, &device).unwrap();
            assert_eq!((restored_config.tie_embeddings, restored_config.token_embed_dim), (tie, table), "{what}");
            assert_close(values(restored.forward(tensor(&ids))), full, 1e-6, &format!("roundtrip {what}"));
        }
    }

    #[test]
    fn test_mrope_spatial_properties() {
        let device = Default::default();
        let dim = 16;
        let coords = vec![vec![
            (0.0, 0.0, 0.0), // Patch (0, 0)
            (0.0, 0.0, 1.0), // Patch (0, 1) -> dx = 1, dy = 0
            (0.0, 1.0, 0.0), // Patch (1, 0) -> dx = 0, dy = 1
            (1.0, 0.0, 0.0), // Query token at (0, 0)
        ]];
        let rope = RopeTables::<Inner>::new_mrope(&coords, dim, &device);
        assert_eq!(rope.cos.dims(), [1, 1, 4, 8, 1]);
        assert_eq!(rope.sin.dims(), [1, 1, 4, 8, 1]);

        let x = Tensor::<Inner, 4>::ones([1, 1, 4, dim], &device);
        let rotated = apply_rope(x, &rope);
        assert_eq!(rotated.dims(), [1, 1, 4, dim]);

        // Dot product between Patch (0, 0) and Patch (0, 1) [dx=1, dy=0]
        let p00 = rotated.clone().slice([0..1, 0..1, 0..1, 0..dim]);
        let p01 = rotated.clone().slice([0..1, 0..1, 1..2, 0..dim]);
        let p10 = rotated.clone().slice([0..1, 0..1, 2..3, 0..dim]);

        let dot_h: f32 = (p00.clone() * p01).sum().into_data().convert::<f32>().to_vec().unwrap()[0];
        let dot_v: f32 = (p00 * p10).sum().into_data().convert::<f32>().to_vec().unwrap()[0];

        // Horizontal and vertical 1-step displacements have exact rotational symmetry
        assert!((dot_h - dot_v).abs() < 1e-5, "M-RoPE horizontal ({dot_h}) and vertical ({dot_v}) dot products should match");
    }
}
