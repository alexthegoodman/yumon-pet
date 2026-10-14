//! Dropless sparse MoE decoder. Only routed rows enter expert matrix multiplies.
//! Router indices are read once per layer for sparse top-k dispatch.
//! Activations/weights stay on device. Benchmark the synchronization overhead.
use super::{
    bpe::{BpeTokenizer, TokenizerKind},
    decoder_model::{KvCache, KvSlot, RopeTables, append_kv, apply_rope, cached_attention, generate_causal_cached, repeat_kv},
    flash_attn::backend::{FlashAttention, causal_flash_attention},
    model::{GenerationResult, MLP, MLPConfig, RMSNorm, RMSNormConfig},
    samples::TrainingStage,
    tokenizer::PAD_TOKEN,
};
use anyhow::Result;
use burn::{
    module::Ignored,
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig, RotaryEncoding,
        RotaryEncodingConfig,
    },
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{
        IndexingUpdateOp, FloatDType,
        activation::{log_softmax, sigmoid, softmax},
    },
};
use super::adaptive_depth::AdaptiveDepthConfig;
use serde::{Deserialize, Serialize};

/// Select experts with the backend's standard top-k operation, then read the
/// indices for host-side sparse dispatch. Tie ordering follows the backend.
fn router_choices<B: Backend>(logits: Tensor<B, 2>, top_k: usize) -> Result<Vec<i32>> {
    let [_, experts] = logits.dims();
    anyhow::ensure!(experts > 0 && experts <= i32::MAX as usize
        && top_k > 0 && top_k <= experts,
        "invalid router dimensions or top-k");
    let (_, indices) = logits.detach().topk_with_indices(top_k, 1);
    let choices = indices.into_data().convert::<i32>().to_vec::<i32>()
        .map_err(|e| anyhow::anyhow!("reading router expert indices: {e:?}"))?;
    anyhow::ensure!(choices.iter().all(|&id| id >= 0 && (id as usize) < experts),
        "router returned an invalid expert index");
    Ok(choices)
}

#[derive(Module, Debug)]
pub struct SparseMoe<B: Backend> {
    router: Linear<B>,
    experts: Vec<MLP<B>>,
    top_k: usize,
}
impl<B: Backend> SparseMoe<B> {
    pub fn new(
        width: usize,
        hidden: usize,
        experts: usize,
        top_k: usize,
        device: &B::Device,
    ) -> Self {
        Self::with_widths(width, &vec![hidden; experts], top_k, device)
    }
    /// Heterogeneous experts: one SwiGLU per entry of `hidden`, one shared router.
    pub fn with_widths(width: usize, hidden: &[usize], top_k: usize, device: &B::Device) -> Self {
        let experts = hidden.len();
        assert!(width > 0 && experts > 0 && hidden.iter().all(|&h| h > 0));
        assert!(top_k > 0 && top_k <= experts);
        Self {
            router: LinearConfig::new(width, experts)
                .with_bias(false)
                .init(device),
            experts: hidden.iter()
                .map(|&h| MLPConfig::new(width, h).init(device))
                .collect(),
            top_k,
        }
    }
    pub fn num_experts(&self) -> usize { self.experts.len() }
    /// Non-padding tokens [tokens, width]; output, balance/z losses, actual row counts.
    pub fn forward(
        &self,
        x: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 1>, Tensor<B, 1>, Vec<usize>) {
        let [tokens, width] = x.dims();
        let device = x.device();
        let n = self.experts.len();
        assert!(tokens > 0);
        // Keep routing probabilities and auxiliary reductions in FP32 under BF16.
        let logits = self.router.forward(x.clone()).cast(FloatDType::F32);
        let probabilities = softmax(logits.clone(), 1);
        // Read selected expert indices; expert activations and gates stay on device.
        let ids = router_choices(logits.clone(), self.top_k)
            .unwrap_or_else(|error| panic!("MoE routing failed: {error}"));
        let mut rows = vec![Vec::<i32>::new(); n];
        for (slot, expert) in ids.into_iter().enumerate() {
            rows[expert as usize].push((slot / self.top_k) as i32);
        }
        let counts: Vec<usize> = rows.iter().map(Vec::len).collect();
        let fractions = Tensor::<B, 2>::from_data_dtype(
            TensorData::new(
                counts
                    .iter()
                    .map(|&c| c as f32 / (tokens * self.top_k) as f32)
                    .collect::<Vec<_>>(),
                [1, n],
            ),
            &device,
            burn::tensor::DType::F32,
        );
        // Switch-style balance: E * sum(f_i * mean(p_i)).
        let balance = (probabilities.clone().mean_dim(0) * fractions).sum() * n as f64;
        let log_z = logits.clone().slice([0..tokens, 0..1])
            - log_softmax(logits, 1).slice([0..tokens, 0..1]);
        let z_loss = log_z.powf_scalar(2.0).mean();
        // Full-softmax gate preserves language-loss router gradients for top-1.
        // Normalizing a single selected gate to 1 would remove that gradient.
        let mut output = Tensor::zeros([tokens, width], &device);
        for (expert_id, row_ids) in rows.into_iter().enumerate() {
            let count = row_ids.len();
            if count == 0 {
                continue;
            }
            let index = Tensor::<B, 1, Int>::from_ints(TensorData::new(row_ids, [count]), &device);
            let input = x
                .clone()
                .select(0, index.clone())
                .reshape([1, count, width]);
            let gate = probabilities
                .clone()
                .select(0, index.clone())
                .slice([0..count, expert_id..expert_id + 1]);
            let value = self.experts[expert_id]
                .forward(input)
                .reshape([count, width])
                * gate.cast(x.dtype());
            output = output.select_assign(0, index, value, IndexingUpdateOp::Add);
        }
        (output, balance, z_loss, counts)
    }
}

#[derive(Module, Debug)]
struct MoeBlock<B: Backend> {
    attn_norm: RMSNorm<B>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    o: Linear<B>,
    ffn_norm: RMSNorm<B>,
    moe: SparseMoe<B>,
    heads: usize,
}
impl<B: FlashAttention> MoeBlock<B> {
    /// Queries [batch, heads, seq, hd] and keys/values [batch, kv_heads, seq, hd];
    /// kv_heads < heads is grouped-query attention (from the k/v weight widths).
    fn qkv(&self, x: &Tensor<B, 3>, rope: &RopeTables<B>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, seq, width] = x.dims();
        let hd = width / self.heads;
        let norm = self.attn_norm.forward(x.clone());
        let split = |t: Tensor<B, 3>| {
            let heads = t.dims()[2] / hd;
            t.reshape([batch, seq, heads, hd]).swap_dims(1, 2)
        };
        let q = apply_rope(split(self.q.forward(norm.clone())), rope);
        let k = apply_rope(split(self.k.forward(norm.clone())), rope);
        (q, k, split(self.v.forward(norm)))
    }
    fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RopeTables<B>,
        active: Tensor<B, 1, Int>,
        pad: Option<Tensor<B, 2, Bool>>,
    ) -> (Tensor<B, 3>, Tensor<B, 1>, Tensor<B, 1>, Vec<usize>) {
        let [_, seq, width] = x.dims();
        let device = x.device();
        let hd = width / self.heads;
        let (q, k, v) = self.qkv(&x, rope);
        let groups = self.heads / k.dims()[1];
        let (k, v) = (repeat_kv(k, groups), repeat_kv(v, groups));
        let attn = if let Some(pad) = pad {
            // Compatibility for left/interior padding. Normal training is right-padded
            // and uses flash; a causal-only kernel cannot mask these PAD keys.
            let future = Tensor::<B, 2>::ones([seq, seq], &device).triu(1).bool();
            let mask = future.unsqueeze::<4>()
                .bool_or(pad.unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(2));
            let dtype = v.dtype();
            let scores = q.cast(FloatDType::F32)
                .matmul(k.cast(FloatDType::F32).transpose()) / (hd as f64).sqrt();
            softmax(scores.mask_fill(mask, -1e9), 3)
                .matmul(v.cast(FloatDType::F32)).cast(dtype)
        } else {
            causal_flash_attention(q, k, v)
        };
        self.feed_forward(x, attn, active)
    }
    /// Output projection and residual for attention [batch, heads, seq, hd], then
    /// the MoE on the `active` rows of [batch * seq].
    fn feed_forward(
        &self,
        x: Tensor<B, 3>,
        attn: Tensor<B, 4>,
        active: Tensor<B, 1, Int>,
    ) -> (Tensor<B, 3>, Tensor<B, 1>, Tensor<B, 1>, Vec<usize>) {
        let [batch, seq, width] = x.dims();
        let device = x.device();
        let attn = attn.swap_dims(1, 2).reshape([batch, seq, width]);
        let x = x + self.o.forward(attn);
        let selected = self
            .ffn_norm
            .forward(x.clone())
            .reshape([batch * seq, width])
            .select(0, active.clone());
        let (values, balance, z_loss, counts) = self.moe.forward(selected);
        let scattered = Tensor::zeros([batch * seq, width], &device)
            .select_assign(0, active, values, IndexingUpdateOp::Add)
            .reshape([batch, seq, width]);
        (x + scattered, balance, z_loss, counts)
    }
    /// Decoding step over one sequence: appends this layer's keys/values for all
    /// new rows to `slot`. Without `active` rows (all exited) only the cache grows.
    fn forward_cached(
        &self,
        x: Tensor<B, 3>,
        rope: &RopeTables<B>,
        active: Option<Tensor<B, 1, Int>>,
        slot: &mut KvSlot<B>,
    ) -> Tensor<B, 3> {
        let (q, k, v) = self.qkv(&x, rope);
        let Some(active) = active else {
            append_kv(slot, k, v);
            return x;
        };
        let attn = cached_attention(q, k, v, slot);
        self.feed_forward(x, attn, active).0
    }
}

/// Per-token continue score before a gated depth tier.
#[derive(Module, Debug)]
struct ExitGate<B: Backend> {
    norm: RMSNorm<B>,
    score: Linear<B>,
}

#[derive(Module, Debug)]
pub struct YumonMoeBrain<B: Backend> {
    pub config: Ignored<YumonMoeBrainConfig>,
    embedding: Embedding<B>,
    // Retained in records for compatibility with existing MoE checkpoints.
    // Forward uses the shared elementwise RoPE, avoiding the batched matmul.
    rope: RotaryEncoding<B>,
    blocks: Vec<MoeBlock<B>>,
    norm: RMSNorm<B>,
    dropout: Dropout,
    pub token_head: Linear<B>,
    /// One per depth tier after the first; empty without `adaptive_depth`.
    exit_gates: Vec<ExitGate<B>>,
}

/// Record layout of checkpoints saved before adaptive depth (no `exit_gates`).
#[derive(Module, Debug)]
struct LegacyMoeBrain<B: Backend> {
    config: Ignored<YumonMoeBrainConfig>,
    embedding: Embedding<B>,
    rope: RotaryEncoding<B>,
    blocks: Vec<MoeBlock<B>>,
    norm: RMSNorm<B>,
    dropout: Dropout,
    token_head: Linear<B>,
}
#[derive(Config, Debug)]
pub struct YumonMoeBrainConfig {
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
    #[config(default = 4)]
    pub num_experts: usize,
    #[config(default = 1)]
    pub top_k: usize,
    #[config(default = 0.01)]
    pub aux_loss_weight: f64,
    #[config(default = 0.001)]
    pub z_loss_weight: f64,
    /// Depth tiers with their own experts; replaces num_experts/top_k/ff_dim.
    #[config(default = "None")]
    pub adaptive_depth: Option<AdaptiveDepthConfig>,
    /// Key/value heads for grouped-query attention; None = attn_heads (MHA).
    #[config(default = "None")]
    pub kv_heads: Option<usize>,
}
impl YumonMoeBrainConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> YumonMoeBrain<B> {
        assert!(self.n_layers > 0 && self.max_seq_len > 0 && self.vocab_size > 0);
        if let Some(depth) = &self.adaptive_depth {
            depth.validate().expect("invalid adaptive_depth");
            assert_eq!(depth.total_layers(), self.n_layers, "adaptive_depth tiers must add up to n_layers");
        }
        assert!(self.attn_heads > 0 && self.embed_dim > 0 && self.embed_dim % self.attn_heads == 0);
        assert!(
            (self.embed_dim / self.attn_heads) % 2 == 0,
            "RoPE requires even head width"
        );
        assert!(self.attn_heads % self.effective_kv_heads() == 0, "attn_heads must be a multiple of kv_heads");
        assert!(self.aux_loss_weight.is_finite() && self.aux_loss_weight >= 0.0);
        assert!(self.z_loss_weight.is_finite() && self.z_loss_weight >= 0.0);
        let linear = || {
            LinearConfig::new(self.embed_dim, self.embed_dim)
                .with_bias(false)
                .init(device)
        };
        let kv_width = self.effective_kv_heads() * (self.embed_dim / self.attn_heads);
        let kv_linear = || LinearConfig::new(self.embed_dim, kv_width).with_bias(false).init(device);
        let tiers = self.adaptive_depth.as_ref().map(|d| d.layer_tiers());
        YumonMoeBrain {
            config: Ignored(self.clone()),
            embedding: EmbeddingConfig::new(self.vocab_size, self.embed_dim).init(device),
            rope: RotaryEncodingConfig::new(self.max_seq_len, self.embed_dim / self.attn_heads)
                .init(device),
            blocks: (0..self.n_layers)
                .map(|layer| MoeBlock {
                    attn_norm: RMSNormConfig::new(self.embed_dim).init(device),
                    q: linear(),
                    k: kv_linear(),
                    v: kv_linear(),
                    o: linear(),
                    ffn_norm: RMSNormConfig::new(self.embed_dim).init(device),
                    moe: match (&self.adaptive_depth, &tiers) {
                        (Some(depth), Some(tiers)) => {
                            let tier = &depth.tiers[tiers[layer]];
                            SparseMoe::with_widths(self.embed_dim, &tier.expert_widths(), tier.top_k, device)
                        }
                        _ => SparseMoe::new(self.embed_dim, self.ff_dim, self.num_experts, self.top_k, device),
                    },
                    heads: self.attn_heads,
                })
                .collect(),
            norm: RMSNormConfig::new(self.embed_dim).init(device),
            dropout: DropoutConfig::new(self.dropout_rate).init(),
            token_head: LinearConfig::new(self.embed_dim, self.vocab_size).init(device),
            // Zero bias: about half the tokens continue at first; the budget loss
            // then pulls each tier toward its capacity.
            exit_gates: self.adaptive_depth.iter().flat_map(|d| d.tiers.iter().skip(1)).map(|_| ExitGate {
                norm: RMSNormConfig::new(self.embed_dim).init(device),
                score: LinearConfig::new(self.embed_dim, 1)
                    .with_initializer(burn::nn::Initializer::Normal { mean: 0.0, std: 0.02 })
                    .init(device),
            }).collect(),
        }
    }
}
impl YumonMoeBrainConfig {
    pub fn effective_kv_heads(&self) -> usize {
        self.kv_heads.unwrap_or(self.attn_heads)
    }
}

impl<B: FlashAttention> YumonMoeBrain<B> {
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.forward_with_aux(tokens).0
    }
    /// Logits, weighted auxiliary loss, and per-layer expert row counts.
    pub fn forward_with_aux(
        &self,
        tokens: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, Tensor<B, 1>, Vec<Vec<usize>>) {
        let (logits, aux, counts, _) = self.forward_with_depth(tokens);
        (logits, aux, counts)
    }
    /// `forward_with_aux` plus the deepest tier index each row (batch * seq,
    /// row-major) reached; all 0 without adaptive depth or at PAD.
    pub fn forward_with_depth(
        &self,
        tokens: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, Tensor<B, 1>, Vec<Vec<usize>>, Vec<u8>) {
        let [batch, seq] = tokens.dims();
        assert!(batch > 0 && seq > 0 && seq <= self.config.max_seq_len);
        let device = tokens.device();
        // Compact once per invocation; reuse non-padding indices in every layer.
        let ids = tokens
            .clone()
            .reshape([batch * seq])
            .to_data()
            .to_vec::<i32>()
            .expect("token ids");
        let active: Vec<i32> = ids
            .iter()
            .enumerate()
            .filter(|(_, id)| **id != PAD_TOKEN as i32)
            .map(|(i, _)| i as i32)
            .collect();
        let active_count = active.len();
        let right_padded = ids.chunks_exact(seq).all(|row| {
            let first_pad = row.iter().position(|&id| id == PAD_TOKEN as i32).unwrap_or(seq);
            row[first_pad..].iter().all(|&id| id == PAD_TOKEN as i32)
        });
        let pad = (!right_padded).then(|| tokens.clone().equal_elem(PAD_TOKEN as i32));
        let rope = RopeTables::new(seq, self.config.embed_dim / self.config.attn_heads, &device);
        let mut x = self.dropout.forward(self.embedding.forward(tokens));
        let mut auxiliary = Tensor::<B, 1>::zeros([1], &device).cast(FloatDType::F32);
        let mut counts = Vec::new();
        let rows = batch * seq;
        let mut deepest = vec![0u8; rows];
        if active_count == 0 {
            return (
                self.token_head.forward(self.norm.forward(x)),
                auxiliary,
                self.blocks.iter().map(|b| vec![0; b.moe.num_experts()]).collect(),
                deepest,
            );
        }
        let layer_tiers = self.config.adaptive_depth.as_ref().map(|d| d.layer_tiers());
        let tier_of = |layer: usize| layer_tiers.as_ref().map_or(0, |t| t[layer]);
        let mut budget = Tensor::<B, 1>::zeros([1], &device).cast(FloatDType::F32);
        // Rows still continuing, the hard keep mask, and (tier input, straight-through mask).
        let mut current = active;
        let mut keep: Option<Tensor<B, 3>> = None;
        let mut tier_start: Option<(Tensor<B, 3>, Tensor<B, 3>)> = None;
        for (layer, block) in self.blocks.iter().enumerate() {
            let tier = tier_of(layer);
            if tier > 0 && tier_of(layer - 1) != tier {
                let depth = self.config.adaptive_depth.as_ref().unwrap();
                let gate = &self.exit_gates[tier - 1];
                let score = sigmoid(gate.score.forward(gate.norm.forward(x.clone()))
                    .reshape([rows]).cast(FloatDType::F32));
                // Causal per-position decision, identical in training and inference.
                // Zero-length selections are not supported by every backend.
                let previous: Vec<f32> = if current.is_empty() { Vec::new() } else {
                    score.clone().detach()
                        .select(0, Tensor::from_ints(TensorData::new(current.clone(), [current.len()]), &device))
                        .into_data().convert::<f32>().to_vec().expect("exit gate scores")
                };
                let mut prev_mask = vec![0f32; rows];
                let mut hard = vec![0f32; rows];
                for (&row, &s) in current.iter().zip(&previous) {
                    prev_mask[row as usize] = 1.0;
                    hard[row as usize] = (s >= 0.5) as u8 as f32;
                }
                current.retain(|&row| hard[row as usize] == 1.0);
                for &row in &current { deepest[row as usize] = tier as u8; }
                let floats = |v: Vec<f32>| Tensor::<B, 1>::from_data_dtype(
                    TensorData::new(v, [rows]), &device, burn::tensor::DType::F32);
                // Straight-through: forward is the hard decision, gradient reaches the gate.
                let soft = score * floats(prev_mask);
                let hard = floats(hard);
                let straight = hard.clone() + soft.clone() - soft.detach();
                let reached = straight.clone().sum() / active_count as f64;
                budget = budget + (reached - depth.tiers[tier].capacity).powf_scalar(2.0);
                keep = Some(hard.reshape([batch, seq, 1]).cast(x.dtype()));
                tier_start = Some((x.clone(), straight.reshape([batch, seq, 1]).cast(x.dtype())));
            }
            if current.is_empty() {
                counts.push(vec![0; block.moe.num_experts()]);
            } else {
                let index = Tensor::<B, 1, Int>::from_ints(
                    TensorData::new(current.clone(), [current.len()]), &device);
                let (next, balance, z_loss, dispatched) = block.forward(x.clone(), &rope, index, pad.clone());
                // Exited positions keep their state; it still serves as keys and values.
                x = match &keep {
                    Some(keep) if tier > 0 => x.clone() + keep.clone() * (next - x),
                    _ => next,
                };
                auxiliary = auxiliary
                    + balance * self.config.aux_loss_weight
                    + z_loss * self.config.z_loss_weight;
                counts.push(dispatched);
            }
            let tier_ends = layer + 1 == self.blocks.len() || tier_of(layer + 1) != tier;
            if tier > 0 && tier_ends {
                let (input, straight) = tier_start.take().unwrap();
                x = input.clone() + straight * (x - input);
            }
        }
        let budget_weight = self.config.adaptive_depth.as_ref().map_or(0.0, |d| d.budget_loss_weight);
        (
            self.token_head.forward(self.norm.forward(x)),
            auxiliary / self.blocks.len() as f64 + budget * budget_weight,
            counts,
            deepest,
        )
    }

    /// Fraction of non-padding tokens that reached each depth tier, from the
    /// per-layer dispatch counts of `forward_with_aux`. Empty without tiers.
    pub fn depth_reach(&self, counts: &[Vec<usize>]) -> Vec<f64> {
        let Some(depth) = self.config.adaptive_depth.as_ref() else { return Vec::new() };
        let mut first = 0;
        let tokens: Vec<f64> = depth.tiers.iter().map(|tier| {
            let routed: usize = counts.get(first).map_or(0, |c| c.iter().sum());
            first += tier.layers;
            routed as f64 / tier.top_k as f64
        }).collect();
        tokens.iter().map(|&t| if tokens[0] > 0.0 { t / tokens[0] } else { 0.0 }).collect()
    }
    /// Logits [1, t, vocab] for `t` new tokens after those already in `cache`;
    /// matches `forward` over the whole sequence. Exit gates decide per position
    /// as in training: exited rows skip deeper attention and experts but still
    /// add their frozen keys/values to every deeper layer's cache.
    pub(crate) fn forward_cached(&self, tokens: Tensor<B, 2, Int>, cache: &mut KvCache<B>) -> Tensor<B, 3> {
        let [batch, t] = tokens.dims();
        assert!(batch == 1 && t > 0 && cache.len() + t <= self.config.max_seq_len);
        let device = tokens.device();
        let rope = cache.next_rope(t);
        let mut x = self.embedding.forward(tokens);
        let layer_tiers = self.config.adaptive_depth.as_ref().map(|d| d.layer_tiers());
        let tier_of = |layer: usize| layer_tiers.as_ref().map_or(0, |t| t[layer]);
        let mut continuing = vec![true; t];
        let mut keep: Option<Tensor<B, 3>> = None;
        for (layer, block) in self.blocks.iter().enumerate() {
            let tier = tier_of(layer);
            if tier > 0 && tier_of(layer - 1) != tier {
                let gate = &self.exit_gates[tier - 1];
                let score: Vec<f32> = sigmoid(gate.score.forward(gate.norm.forward(x.clone()))
                    .reshape([t]).cast(FloatDType::F32))
                    .into_data().convert::<f32>().to_vec().expect("exit gate scores");
                continuing.iter_mut().zip(&score).for_each(|(on, &s)| *on &= s >= 0.5);
                let hard = continuing.iter().map(|&on| on as u8 as f32).collect::<Vec<_>>();
                keep = Some(Tensor::<B, 1>::from_data_dtype(TensorData::new(hard, [t]), &device,
                    burn::tensor::DType::F32).reshape([1, t, 1]).cast(x.dtype()));
            }
            let rows: Vec<i32> = (0..t as i32).filter(|&r| continuing[r as usize]).collect();
            let active = (!rows.is_empty())
                .then(|| Tensor::<B, 1, Int>::from_ints(TensorData::new(rows.clone(), [rows.len()]), &device));
            let next = block.forward_cached(x.clone(), &rope, active, cache.layer(layer));
            x = match &keep {
                Some(keep) if tier > 0 => x.clone() + keep.clone() * (next - x),
                _ => next,
            };
        }
        cache.advance(t);
        self.token_head.forward(self.norm.forward(x))
    }

    pub fn generate_unmasked_parsed(
        &self,
        tokenizer: &TokenizerKind,
        seed_text: &str,
        max_tokens: usize,
        device: &B::Device,
    ) -> GenerationResult {
        let mut cache = KvCache::new(self.blocks.len(), self.config.max_seq_len,
            self.config.embed_dim / self.config.attn_heads, device);
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

    pub fn save(
        &self,
        directory: &str,
        tokenizer: &TokenizerKind,
        metadata: &MoeMetadata,
    ) -> Result<()> {
        let dir = std::path::Path::new(directory);
        std::fs::create_dir_all(dir)?;

        let meta_json = serde_json::to_string_pretty(metadata)?;
        std::fs::write(dir.join("metadata.json"), meta_json)?;

        match tokenizer {
            TokenizerKind::Bpe(t) => t.save(directory)?,
            TokenizerKind::Char(_) => anyhow::bail!("MoE checkpoints require a BPE tokenizer"),
        }

        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        self.clone()
            .save_file(dir.join("model"), &recorder)
            .map_err(|e| anyhow::anyhow!("save_file: {e:?}"))?;

        Ok(())
    }

    pub fn load(
        directory: &str,
        device: &B::Device,
    ) -> Result<(Self, TokenizerKind, YumonMoeBrainConfig)> {
        let dir = std::path::Path::new(directory);

        let meta_json = std::fs::read_to_string(dir.join("metadata.json"))?;
        let metadata: MoeMetadata = serde_json::from_str(&meta_json)?;

        let tokenizer = TokenizerKind::Bpe(BpeTokenizer::load(directory)?);

        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let record = match recorder.load(dir.join("model").into(), device) {
            Ok(record) => record,
            // Checkpoints from before adaptive depth have no exit gate field.
            Err(error) if metadata.adaptive_depth.is_none() => {
                let legacy: LegacyMoeBrainRecord<B> = recorder.load(dir.join("model").into(), device)
                    .map_err(|e| anyhow::anyhow!("load: {error:?}; legacy layout: {e:?}"))?;
                YumonMoeBrainRecord {
                    config: legacy.config, embedding: legacy.embedding, rope: legacy.rope,
                    blocks: legacy.blocks, norm: legacy.norm, dropout: legacy.dropout,
                    token_head: legacy.token_head, exit_gates: Vec::new(),
                }
            }
            Err(error) => anyhow::bail!("load: {error:?}"),
        };

        let config = YumonMoeBrainConfig {
            vocab_size: metadata.vocab_size,
            embed_dim: metadata.embed_dim,
            hidden_units: metadata.hidden_units,
            n_layers: metadata.n_layers,
            attn_heads: metadata.attn_heads,
            ff_dim: metadata.ff_dim,
            max_seq_len: metadata.max_seq_len,
            training_stage: metadata.training_stage,
            dropout_rate: metadata.dropout_rate,
            num_experts: metadata.num_experts,
            top_k: metadata.top_k,
            aux_loss_weight: metadata.aux_loss_weight,
            z_loss_weight: metadata.z_loss_weight,
            adaptive_depth: metadata.adaptive_depth,
            kv_heads: metadata.kv_heads,
        };

        let model = config.init::<B>(device).load_record(record);

        Ok((model, tokenizer, config))
    }
}

// ─── Metadata ─────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct MoeMetadata {
    pub num_experts: usize,
    pub top_k: usize,
    pub aux_loss_weight: f64,
    pub z_loss_weight: f64,
    pub dropout_rate: f64,
    pub vocab_size: usize,
    pub epochs_trained: usize,
    /// Per-token training loss (runs before 2026-10-06 logged it diluted by padding).
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
    /// Depth tiers; absent for uniform-depth models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adaptive_depth: Option<AdaptiveDepthConfig>,
    /// Grouped-query KV heads; absent = attn_heads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_heads: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::{Autodiff, autodiff::checkpoint::strategy::BalancedCheckpointing},
        module::{AutodiffModule, Param},
        optim::{AdamWConfig, GradientsParams, Optimizer},
    };
    type Wgpu = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;
    type B = Autodiff<Wgpu, BalancedCheckpointing>;

    fn assert_close(actual: Vec<f32>, expected: Vec<f32>, tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.into_iter().zip(expected) {
            assert!(
                a.is_finite() && b.is_finite() && (a - b).abs() <= tolerance,
                "{a} vs {b}"
            );
        }
    }
    fn dense_reference(m: &SparseMoe<B>, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let [tokens, width] = x.dims();
        let logits = m.router.forward(x.clone());
        let p = softmax(logits.clone(), 1);
        let (_, ids) = logits.detach().topk_with_indices(m.top_k, 1);
        let mut out = Tensor::zeros([tokens, width], &x.device());
        for (i, expert) in m.experts.iter().enumerate() {
            let mask = ids.clone().equal_elem(i as i32).float().sum_dim(1);
            let gate = p.clone().slice([0..tokens, i..i + 1]) * mask;
            out = out
                + expert
                    .forward(x.clone().reshape([1, tokens, width]))
                    .reshape([tokens, width])
                    * gate;
        }
        out
    }
    #[test]
    fn moe_sparse_matches_dense_outputs_and_gradients() {
        let device = Default::default();
        for top_k in [1, 2, 4] {
            let model = SparseMoe::<B>::new(8, 16, 4, top_k, &device);
            let x = Tensor::<B, 2>::random(
                [7, 8],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            )
            .require_grad();
            let (out, balance, z, counts) = model.forward(x.clone());
            assert_eq!(counts.iter().sum::<usize>(), 7 * top_k);
            let reference = dense_reference(&model, x.clone());
            assert_close(
                out.to_data().to_vec().unwrap(),
                reference.to_data().to_vec().unwrap(),
                1e-4,
            );
            let sparse_grads = out.powf_scalar(2.0).sum().backward();
            let dense_grads = reference.powf_scalar(2.0).sum().backward();
            assert_close(
                x.grad(&sparse_grads).unwrap().to_data().to_vec().unwrap(),
                x.grad(&dense_grads).unwrap().to_data().to_vec().unwrap(),
                2e-3,
            );
            let gate_grad = model.router.weight.val().grad(&sparse_grads).unwrap();
            let values: Vec<f32> = gate_grad.to_data().to_vec().unwrap();
            assert!(values.iter().all(|x| x.is_finite()));
            assert!(
                values.iter().any(|x| x.abs() > 1e-7),
                "language loss must train router, including top-1"
            );
            assert_close(
                values,
                model
                    .router
                    .weight
                    .val()
                    .grad(&dense_grads)
                    .unwrap()
                    .to_data()
                    .to_vec()
                    .unwrap(),
                2e-3,
            );
            assert!(balance.into_scalar().is_finite() && z.into_scalar().is_finite());
        }
    }
    #[test]
    fn moe_unused_experts_have_no_gradients() {
        let device = Default::default();
        let mut model = SparseMoe::<B>::new(8, 16, 4, 1, &device);
        // Positive identical inputs select expert 0 for every token.
        let weights: Vec<f32> = (0..8).flat_map(|_| [3.0, 2.0, 1.0, 0.0]).collect();
        model.router.weight =
            Param::from_tensor(Tensor::from_data(TensorData::new(weights, [8, 4]), &device));
        let (out, _, _, counts) = model.forward(Tensor::ones([5, 8], &device));
        assert_eq!(counts, vec![5, 0, 0, 0]);
        let grads = out.sum().backward();
        struct Check<'a> {
            grads: &'a <B as burn::tensor::backend::AutodiffBackend>::Gradients,
            expected: bool,
            count: usize,
        }
        impl burn::module::ModuleVisitor<B> for Check<'_> {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                assert_eq!(param.val().grad(self.grads).is_some(), self.expected);
                self.count += 1;
            }
        }
        for (i, expert) in model.experts.iter().enumerate() {
            let mut check = Check {
                grads: &grads,
                expected: i == 0,
                count: 0,
            };
            expert.visit(&mut check);
            assert!(check.count > 0);
        }
    }
    fn tiny_config() -> YumonMoeBrainConfig {
        YumonMoeBrainConfig::new(32, TrainingStage::Language)
            .with_embed_dim(16)
            .with_n_layers(2)
            .with_attn_heads(2)
            .with_ff_dim(32)
            .with_max_seq_len(8)
            .with_dropout_rate(0.0)
            .with_num_experts(4)
            .with_top_k(2)
    }

    fn values<T: Backend, const D: usize>(x: Tensor<T, D>) -> Vec<f32> {
        x.into_data().convert::<f32>().to_vec().unwrap()
    }

    #[test]
    fn moe_router_extreme_scores_and_ties() {
        let device = Default::default();
        let rows = [vec![0.0; 16], vec![f32::MIN; 16],
            (0..16).map(|i| -100.0 + i as f32).collect::<Vec<_>>()];
        let logits = Tensor::<B, 2>::from_data(TensorData::new(rows.concat(), [3, 16]), &device);
        for k in [1, 2, 16] {
            let actual = router_choices(logits.clone(), k).unwrap();
            assert_eq!(actual.len(), 3 * k);
            // Equal scores may use any order; each expert must be selected once.
            for row in actual.chunks_exact(k) {
                let unique: std::collections::HashSet<_> = row.iter().collect();
                assert_eq!(unique.len(), k);
                assert!(row.iter().all(|&id| (0..16).contains(&id)));
            }
            let expected: Vec<i32> = (0..16).rev().take(k).collect();
            assert_eq!(&actual[2 * k..], expected.as_slice());
        }
    }

    #[test]
    fn moe_router_fp32_logits() {
        let device = Default::default();
        let logits = Tensor::<B, 2>::from_data(
            TensorData::new(vec![1.0f32, 4.0, 2.0, 3.0, -4.0, -1.0, -3.0, -2.0], [2, 4]),
            &device);
        for k in [1, 2, 4] {
            let ids = router_choices(logits.clone(), k).unwrap();
            let expected: Vec<i32> = [vec![1, 3, 2, 0], vec![1, 3, 2, 0]]
                .iter().flat_map(|row| row[..k].iter().copied()).collect();
            assert_eq!(ids, expected);
        }
    }

    #[test]
    fn moe_flash_matches_masked_attention_and_gradients() {
        let device = Default::default();
        let model: YumonMoeBrain<B> = tiny_config().init(&device);
        let block = &model.blocks[0];
        let rope = RopeTables::new(6, 8, &device);
        let input = Tensor::<B, 3>::random([1, 6, 16],
            burn::tensor::Distribution::Normal(0.0, 0.5), &device).require_grad();
        let active = Tensor::from_ints([0, 1, 2, 3], &device);
        let pad = Tensor::<B, 2, Int>::from_ints([[1, 1, 1, 1, 0, 0]], &device).equal_elem(0);
        let run = |mask| {
            let (out, balance, z, counts) = block.forward(input.clone(), &rope, active.clone(), mask);
            // Only real tokens: padded queries may attend differently.
            let out = out.slice([0..1, 0..4, 0..16]);
            let grads = (out.clone().square().mean() + balance * 0.01 + z * 0.001).backward();
            (values(out), values(input.grad(&grads).unwrap()),
                values(block.q.weight.val().grad(&grads).unwrap()), counts)
        };
        let (out, dx, dq, counts) = run(None);
        let (reference, rx, rq, expected_counts) = run(Some(pad));
        assert_eq!(counts, expected_counts);
        assert_close(out, reference, 1e-4);
        assert_close(dx, rx, 1e-4);
        assert_close(dq, rq, 1e-4);
    }

    #[test]
    fn moe_balanced_matches_plain_gradients() {
        use burn::{backend::autodiff::checkpoint::strategy::{CheckpointStrategy, NoCheckpointing},
            record::BinBytesRecorder, tensor::backend::AutodiffBackend};
        let config = tiny_config().with_top_k(4);
        let bytes = BinBytesRecorder::<FullPrecisionSettings>::default()
            .record(config.init::<Wgpu>(&Default::default()).into_record(), ()).unwrap();
        fn run<C: CheckpointStrategy>(config: &YumonMoeBrainConfig, bytes: &[u8]) -> (Vec<f32>, Vec<Vec<f32>>) {
            type AD<C> = Autodiff<Wgpu, C>;
            let device = Default::default();
            let record = BinBytesRecorder::<FullPrecisionSettings>::default().load(bytes.to_vec(), &device).unwrap();
            let model: YumonMoeBrain<AD<C>> = config.init(&device).load_record(record);
            let (out, aux, _) = model.forward_with_aux(Tensor::from_ints([[1, 4, 5, 0]], &device));
            let loss = out.square().mean() + aux;
            let grads = loss.clone().backward();
            struct Collect<'a, C: CheckpointStrategy> {
                grads: &'a <AD<C> as AutodiffBackend>::Gradients,
                values: Vec<Vec<f32>>,
            }
            impl<C: CheckpointStrategy> burn::module::ModuleVisitor<AD<C>> for Collect<'_, C> {
                fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<AD<C>, D>>) {
                    self.values.push(values(p.val().grad(self.grads).expect("all experts routed")));
                }
            }
            let mut collect = Collect { grads: &grads, values: vec![] };
            model.visit(&mut collect);
            (values(loss), collect.values)
        }
        let (loss, grads) = run::<BalancedCheckpointing>(&config, &bytes);
        let (reference, expected) = run::<NoCheckpointing>(&config, &bytes);
        assert_close(loss, reference, 1e-5);
        assert_eq!(grads.len(), expected.len());
        for (a, b) in grads.into_iter().zip(expected) { assert_close(a, b, 1e-4); }
    }

    #[test]
    #[ignore = "requires a CUDA GPU with native BF16 support; no profiling"]
    fn cuda_bf16_moe_optimizer_and_checkpoint() {
        use burn::{record::BinBytesRecorder, tensor::DType};
        type AD = crate::brain::train::CudaBf16TrainBackend;
        let device = Default::default();
        let config = tiny_config();
        let model: YumonMoeBrain<AD> = config.init(&device);
        let before = values(model.token_head.weight.val());
        let (out, aux, counts) = model.forward_with_aux(Tensor::from_ints([[1, 4, 5, 0]], &device));
        assert_eq!(out.dtype(), DType::BF16);
        assert_eq!(aux.dtype(), DType::F32);
        assert!(values(aux.clone())[0].is_finite());
        assert!(counts.iter().all(|c| c.iter().sum::<usize>() == 6));
        let grads = GradientsParams::from_grads((out.cast(FloatDType::F32).square().mean() + aux).backward(), &model);
        let updated = AdamWConfig::new().init().step(0.01, model, grads);
        assert_eq!(updated.token_head.weight.val().dtype(), DType::BF16);
        assert_ne!(before, values(updated.token_head.weight.val()));
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let bytes = recorder.record(updated.clone().into_record(), ()).unwrap();
        let loaded: YumonMoeBrain<AD> = config.init(&device).load_record(recorder.load(bytes, &device).unwrap());
        let input = || Tensor::from_ints([[1, 4, 5]], &device);
        assert_close(values(updated.valid().forward(input())), values(loaded.valid().forward(input())), 0.0);
    }
    #[test]
    fn moe_causal_padding_and_optimizer_step() {
        let device = Default::default();
        let model: YumonMoeBrain<B> = tiny_config().init(&device);
        let tokens = Tensor::from_ints([[1, 4, 5, 6, 0, 0]], &device);
        let (logits, aux, counts) = model.forward_with_aux(tokens);
        assert_eq!(logits.dims(), [1, 6, 32]);
        for counts in counts {
            assert_eq!(counts.iter().sum::<usize>(), 8);
        }
        let changed = model.forward(Tensor::from_ints([[1, 4, 9, 8, 0, 0]], &device));
        assert_close(
            logits
                .clone()
                .slice([0..1, 0..2, 0..32])
                .to_data()
                .to_vec()
                .unwrap(),
            changed
                .slice([0..1, 0..2, 0..32])
                .to_data()
                .to_vec()
                .unwrap(),
            1e-4,
        );
        let short = model.forward(Tensor::from_ints([[1, 4, 5, 6]], &device));
        assert_close(
            logits
                .clone()
                .slice([0..1, 0..4, 0..32])
                .to_data()
                .to_vec()
                .unwrap(),
            short.to_data().to_vec().unwrap(),
            1e-4,
        );
        let loss = logits.powf_scalar(2.0).mean() + aux;
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let mut optimizer = AdamWConfig::new().init();
        let updated = optimizer.step(1e-3, model, grads);
        let (padding_logits, padding_aux, padding_counts) =
            updated.forward_with_aux(Tensor::from_ints([[0, 0]], &device));
        assert!(
            padding_logits
                .to_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|x| x.is_finite())
        );
        assert_eq!(padding_aux.into_scalar(), 0.0);
        assert!(padding_counts.iter().flatten().all(|&n| n == 0));
        assert_eq!(
            updated
                .valid()
                .forward(Tensor::from_ints([[1]], &device))
                .dims(),
            [1, 1, 32]
        );
    }
    #[test]
    fn moe_checkpoint_roundtrip() {
        let device = Default::default();
        let tokenizer = TokenizerKind::Bpe(BpeTokenizer::load("yumon_bpe").unwrap());
        let mut config = tiny_config();
        config.vocab_size = tokenizer.vocab_size();
        let model: YumonMoeBrain<Wgpu> = config.init(&device);
        let path = std::env::temp_dir().join(format!("yumon-moe-{}", uuid::Uuid::new_v4()));
        let meta = MoeMetadata {
            num_experts: config.num_experts,
            top_k: config.top_k,
            aux_loss_weight: config.aux_loss_weight,
            z_loss_weight: config.z_loss_weight,
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
            adaptive_depth: None,
            kv_heads: config.kv_heads,
        };
        model
            .save(path.to_str().unwrap(), &tokenizer, &meta)
            .unwrap();
        let (restored, tok, restored_config) =
            YumonMoeBrain::<Wgpu>::load(path.to_str().unwrap(), &device).unwrap();
        assert_eq!(tok.encode("hello"), tokenizer.encode("hello"));
        assert_eq!(restored_config.top_k, config.top_k);
        assert_close(
            model
                .forward(Tensor::from_ints([[1, 4, 5]], &device))
                .to_data()
                .to_vec()
                .unwrap(),
            restored
                .forward(Tensor::from_ints([[1, 4, 5]], &device))
                .to_data()
                .to_vec()
                .unwrap(),
            1e-5,
        );
        // Only remove the uniquely created test directory after verifying its parent.
        assert_eq!(path.parent(), Some(std::env::temp_dir().as_path()));
        std::fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn moe_heterogeneous_experts_match_dense() {
        let device = Default::default();
        let model = SparseMoe::<B>::with_widths(8, &[4, 16, 16, 40], 2, &device);
        let x = Tensor::<B, 2>::random([9, 8], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let (out, _, _, counts) = model.forward(x.clone());
        assert_eq!(counts.iter().sum::<usize>(), 18);
        assert_close(values(out), values(dense_reference(&model, x)), 1e-4);
    }

    fn depth_config() -> YumonMoeBrainConfig {
        let depth: AdaptiveDepthConfig = serde_json::from_str(r#"{"tiers": [
            {"name": "easy", "layers": 1, "top_k": 2, "experts": [{"count": 4, "ff_dim": 16}]},
            {"name": "medium", "layers": 1, "capacity": 0.5, "top_k": 1,
             "experts": [{"count": 2, "ff_dim": 16}, {"count": 1, "ff_dim": 32}]},
            {"name": "hard", "layers": 2, "capacity": 0.25, "top_k": 1, "experts": [{"count": 2, "ff_dim": 64}]}
        ]}"#).unwrap();
        tiny_config().with_n_layers(4).with_adaptive_depth(Some(depth))
    }

    /// Forces every exit gate to continue (+) or exit (-) regardless of input.
    fn force_gates(model: YumonMoeBrain<B>, bias: f32) -> YumonMoeBrain<B> {
        let mut model = model;
        for gate in &mut model.exit_gates {
            let device = gate.score.weight.val().device();
            gate.score.weight = Param::from_tensor(Tensor::zeros([16, 1], &device));
            gate.score.bias = Some(Param::from_tensor(Tensor::from_floats([bias], &device)));
        }
        model
    }

    #[test]
    fn moe_depth_layout_and_forced_gates() {
        let device = Default::default();
        let model: YumonMoeBrain<B> = depth_config().init(&device);
        assert_eq!(model.exit_gates.len(), 2);
        let experts: Vec<usize> = model.blocks.iter().map(|b| b.moe.num_experts()).collect();
        assert_eq!(experts, vec![4, 3, 2, 2]);
        let tokens = || Tensor::<B, 2, Int>::from_ints([[1, 4, 5, 6, 0, 0], [7, 8, 9, 3, 2, 0]], &device);

        // Everyone continues: every layer routes all 9 tokens, reach is 100% everywhere.
        let all = force_gates(model.clone(), 20.0);
        let (_, _, counts) = all.forward_with_aux(tokens());
        let routed: Vec<usize> = counts.iter().map(|c| c.iter().sum()).collect();
        assert_eq!(routed, vec![18, 9, 9, 9]);
        assert_eq!(all.depth_reach(&counts), vec![1.0, 1.0, 1.0]);
        let (_, _, _, deepest) = all.forward_with_depth(tokens());
        assert_eq!(deepest, vec![2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 0]);

        // Everyone exits after the easy tier: deeper experts get nothing, and the
        // result equals running only the first block.
        let none = force_gates(model, -20.0);
        let (logits, aux, counts) = none.forward_with_aux(tokens());
        let routed: Vec<usize> = counts.iter().map(|c| c.iter().sum()).collect();
        assert_eq!(routed, vec![18, 0, 0, 0]);
        assert_eq!(none.depth_reach(&counts), vec![1.0, 0.0, 0.0]);
        assert!(none.forward_with_depth(tokens()).3.iter().all(|&t| t == 0));
        assert!(values(aux)[0].is_finite());
        let mut shallow = none.clone();
        shallow.blocks.truncate(1);
        shallow.exit_gates.clear();
        shallow.config.0.adaptive_depth = None;
        shallow.config.0.n_layers = 1;
        assert_close(values(logits), values(shallow.forward(tokens())), 1e-5);
    }

    #[test]
    fn moe_depth_is_causal_and_trains_gates() {
        let device = Default::default();
        let mut model: YumonMoeBrain<B> = depth_config().init(&device);
        // Medium gate passes everyone (s = 0.95), so 6/6 vs capacity 0.5 and k/6 vs
        // 0.25 both leave a non-zero budget gradient for the two gates.
        model.exit_gates[0].score.weight = Param::from_tensor(Tensor::zeros([16, 1], &device));
        model.exit_gates[0].score.bias = Some(Param::from_tensor(Tensor::from_floats([3.0], &device)));
        let (logits, aux, counts) = model.forward_with_aux(Tensor::from_ints([[1, 4, 5, 6, 7, 8]], &device));
        assert_eq!(counts[1].iter().sum::<usize>(), 6);
        let reach = model.depth_reach(&counts);
        assert!(reach[0] == 1.0 && reach[1] >= reach[2], "reach must not grow with depth: {reach:?}");
        // Exit decisions are per position, so later tokens cannot change earlier logits.
        let changed = model.forward(Tensor::from_ints([[1, 4, 5, 9, 3, 2]], &device));
        assert_close(values(logits.clone().slice([0..1, 0..3, 0..32])), values(changed.slice([0..1, 0..3, 0..32])), 1e-4);
        let short = model.forward(Tensor::from_ints([[1, 4, 5]], &device));
        assert_close(values(logits.clone().slice([0..1, 0..3, 0..32])), values(short), 1e-4);
        // The budget term alone reaches every gate through the straight-through mask.
        let grads = (logits.square().mean() + aux).backward();
        for gate in &model.exit_gates {
            let grad = values(gate.score.weight.val().grad(&grads).expect("exit gate gradient"));
            assert!(grad.iter().all(|g| g.is_finite()) && grad.iter().any(|g| g.abs() > 0.0));
        }
    }

    #[test]
    fn moe_depth_checkpoint_and_legacy_roundtrip() {
        let device = Default::default();
        let tokenizer = TokenizerKind::Bpe(BpeTokenizer::load("yumon_bpe").unwrap());
        let input = || Tensor::<Wgpu, 2, Int>::from_ints([[1, 4, 5, 6]], &device);
        for depth in [true, false] {
            let mut config = if depth { depth_config() } else { tiny_config() };
            config.vocab_size = tokenizer.vocab_size();
            let model: YumonMoeBrain<Wgpu> = config.init(&device);
            let path = std::env::temp_dir().join(format!("yumon-moe-depth-{}", uuid::Uuid::new_v4()));
            let meta = MoeMetadata { num_experts: config.num_experts, top_k: config.top_k,
                aux_loss_weight: config.aux_loss_weight, z_loss_weight: config.z_loss_weight,
                dropout_rate: config.dropout_rate, vocab_size: config.vocab_size, epochs_trained: 1,
                final_loss: 1.0, val_loss: None, batch_size: 1, training_stage: config.training_stage,
                embed_dim: config.embed_dim, hidden_units: config.hidden_units, n_layers: config.n_layers,
                attn_heads: config.attn_heads, ff_dim: config.ff_dim, max_seq_len: config.max_seq_len,
                adaptive_depth: config.adaptive_depth.clone(), kv_heads: config.kv_heads };
            model.save(path.to_str().unwrap(), &tokenizer, &meta).unwrap();
            if !depth {
                // Rewrite the weights in the pre-adaptive-depth record layout.
                let r = model.clone().into_record();
                let legacy = LegacyMoeBrainRecord::<Wgpu> { config: r.config, embedding: r.embedding,
                    rope: r.rope, blocks: r.blocks, norm: r.norm, dropout: r.dropout, token_head: r.token_head };
                BinFileRecorder::<FullPrecisionSettings>::new().record(legacy, path.join("model")).unwrap();
            }
            let (restored, _, restored_config) = YumonMoeBrain::<Wgpu>::load(path.to_str().unwrap(), &device).unwrap();
            assert_eq!(restored_config.adaptive_depth, config.adaptive_depth);
            assert_close(values(model.forward(input())), values(restored.forward(input())), 1e-5);
            assert_eq!(path.parent(), Some(std::env::temp_dir().as_path()));
            std::fs::remove_dir_all(path).unwrap();
        }
    }

    /// Real GPU forward/backward timing including dispatch/readback overhead.
    /// Compares against the same experts/router executing every expert for all tokens.
    #[test]
    #[ignore]
    fn moe_timing_probe() {
        let device = Default::default();
        for (tokens, width, hidden) in [(1024, 64, 256), (1024, 256, 1024)] {
            let model = SparseMoe::<B>::new(width, hidden, 4, 1, &device);
            let dense_active: MLP<B> = MLPConfig::new(width, hidden).init(&device);
            let dense_total: MLP<B> = MLPConfig::new(width, hidden * 4).init(&device);
            let x = Tensor::<B, 2>::random(
                [tokens, width],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            )
            .require_grad();
            for mode in [
                "all-experts-masked",
                "sparse",
                "dense-active-size",
                "dense-total-size",
            ] {
                let mut elapsed = std::time::Duration::ZERO;
                for step in 0..8 {
                    B::sync(&device).unwrap();
                    let start = std::time::Instant::now();
                    let out = match mode {
                        "all-experts-masked" => dense_reference(&model, x.clone()),
                        "sparse" => model.forward(x.clone()).0,
                        "dense-active-size" => dense_active
                            .forward(x.clone().reshape([1, tokens, width]))
                            .reshape([tokens, width]),
                        _ => dense_total
                            .forward(x.clone().reshape([1, tokens, width]))
                            .reshape([tokens, width]),
                    };
                    let grads = out.powf_scalar(2.0).mean().backward();
                    let _ = x.grad(&grads).unwrap().to_data();
                    B::sync(&device).unwrap();
                    if step >= 3 {
                        elapsed += start.elapsed();
                    }
                }
                println!(
                    "MoE timing tokens={tokens} width={width} hidden={hidden} {mode}: {:?}/forward+backward",
                    elapsed / 5
                );
            }
        }
    }

    /// Logits of a cached run that feeds `ids` in the given chunk sizes, concatenated.
    fn cached_logits(model: &YumonMoeBrain<Wgpu>, ids: &[i32], chunks: &[usize]) -> Vec<f32> {
        let device = Default::default();
        let config = &model.config;
        let mut cache = KvCache::new(model.blocks.len(), config.max_seq_len, config.embed_dim / config.attn_heads, &device);
        let mut out = Vec::new();
        let mut start = 0;
        for &n in chunks {
            let tokens = Tensor::<Wgpu, 2, Int>::from_ints(TensorData::new(ids[start..start + n].to_vec(), [1, n]), &device);
            out.extend(values(model.forward_cached(tokens, &mut cache)));
            start += n;
        }
        assert_eq!(start, ids.len());
        assert_eq!(cache.len(), ids.len());
        out
    }

    #[test]
    fn moe_kv_cache_matches_full_forward() {
        let device = Default::default();
        let ids: Vec<i32> = (0..14).map(|i| 4 + (i * 7) % 28).collect();
        let gqa = tiny_config().with_attn_heads(4).with_kv_heads(Some(2));
        let mqa_depth = depth_config().with_attn_heads(4).with_kv_heads(Some(1));
        for (name, config) in [("mha", tiny_config()), ("gqa", gqa), ("depth", depth_config()), ("depth-mqa", mqa_depth)] {
            let config = config.with_max_seq_len(16);
            let mut model: YumonMoeBrain<Wgpu> = config.init(&device);
            // Decisive random exit gates, so some positions exit and others continue.
            for gate in &mut model.exit_gates {
                gate.score.weight = Param::from_tensor(Tensor::random([16, 1],
                    burn::tensor::Distribution::Normal(0.0, 3.0), &device));
            }
            let tokens = Tensor::<Wgpu, 2, Int>::from_ints(TensorData::new(ids.clone(), [1, ids.len()]), &device);
            let (full, _, counts) = model.forward_with_aux(tokens);
            if model.config.adaptive_depth.is_some() {
                let reach = model.depth_reach(&counts);
                assert!(reach[1] > 0.0 && reach[1] < 1.0, "{name}: gates should split the tokens, reach {reach:?}");
            }
            // Prompt, single steps, then a multi-token chunk on top of the cache.
            let full = values(full);
            assert_close(cached_logits(&model, &ids, &[5, 1, 1, 1, 3, 1, 2]), full.clone(), 1e-4);
            assert_close(cached_logits(&model, &ids, &[1; 14]), full, 1e-4);
        }
    }

    #[test]
    fn moe_gqa_trains_with_fewer_kv_weights() {
        let device = Default::default();
        let config = tiny_config().with_attn_heads(4).with_kv_heads(Some(2));
        let mut model: YumonMoeBrain<B> = config.init(&device);
        assert_eq!(model.blocks[0].k.weight.val().dims(), [16, 8]);
        assert_eq!(model.blocks[0].q.weight.val().dims(), [16, 16]);
        let tokens = Tensor::<B, 2, Int>::from_ints([[1, 4, 5, 6, 7, 8], [9, 10, 11, 2, 0, 0]], &device);
        let (logits, aux, _) = model.forward_with_aux(tokens);
        let loss = logits.powf_scalar(2.0).mean() + aux;
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let k_grad = grads.get::<Wgpu, 2>(model.blocks[0].k.weight.id).expect("k gradient");
        assert_eq!(k_grad.dims(), [16, 8]);
        let mut optimizer = AdamWConfig::new().init();
        model = optimizer.step(1e-3, model, grads);
        assert_eq!(model.blocks[0].v.weight.val().dims(), [16, 8]);
    }

    #[test]
    #[ignore = "timing probe on the local GPU"]
    fn moe_kv_cache_decode_timing() {
        let device = Default::default();
        let (prompt, steps) = (1536, 32);
        let lite = || YumonMoeBrainConfig::new(16384, TrainingStage::Language)
            .with_embed_dim(128).with_n_layers(8).with_attn_heads(4).with_ff_dim(512)
            .with_max_seq_len(2048).with_dropout_rate(0.0).with_num_experts(4).with_top_k(1);
        // configs/yumon-code-moe-mini-depth.json at a 2048 context.
        let depth: AdaptiveDepthConfig = serde_json::from_str(r#"{"budget_loss_weight": 0.1, "tiers": [
            {"name": "easy", "layers": 8, "top_k": 2, "experts": [{"count": 16, "ff_dim": 512}]},
            {"name": "medium", "layers": 8, "capacity": 0.5, "top_k": 2,
             "experts": [{"count": 12, "ff_dim": 1024}, {"count": 4, "ff_dim": 2048}]},
            {"name": "hard", "layers": 8, "capacity": 0.25, "top_k": 2, "experts": [{"count": 8, "ff_dim": 3072}]}
        ]}"#).unwrap();
        let depth = YumonMoeBrainConfig::new(16384, TrainingStage::Language)
            .with_embed_dim(256).with_n_layers(24).with_attn_heads(8)
            .with_max_seq_len(2048).with_dropout_rate(0.0).with_adaptive_depth(Some(depth));
        for (name, config) in [("lite", lite()), ("lite gqa1", lite().with_kv_heads(Some(1))),
            ("mini-depth", depth.clone()), ("mini-depth gqa2", depth.with_kv_heads(Some(2)))] {
            let model: YumonMoeBrain<Wgpu> = config.init(&device);
            let (layers, hd) = (model.blocks.len(), model.config.embed_dim / model.config.attn_heads);
            let ids: Vec<i32> = (0..prompt + steps).map(|i| 4 + (i as i32 * 31) % 16000).collect();
            let tensor = |ids: &[i32]| Tensor::<Wgpu, 2, Int>::from_ints(TensorData::new(ids.to_vec(), [1, ids.len()]), &device);
            let last = |logits: Tensor<Wgpu, 3>| {
                let [_, rows, vocab] = logits.dims();
                let _ = logits.slice([0..1, rows - 1..rows, 0..vocab]).into_data();
            };
            // Each path runs twice; the first run compiles/autotunes every shape it meets.
            let mut timings = Vec::new();
            for _ in 0..2 {
                let start = std::time::Instant::now();
                for n in prompt..prompt + steps { last(model.forward(tensor(&ids[..n + 1]))); }
                let uncached = start.elapsed() / steps as u32;
                let mut cache = KvCache::new(layers, 2048, hd, &device);
                let start = std::time::Instant::now();
                last(model.forward_cached(tensor(&ids[..prompt]), &mut cache));
                let prefill = start.elapsed();
                let start = std::time::Instant::now();
                for n in prompt..prompt + steps { last(model.forward_cached(tensor(&ids[n..n + 1]), &mut cache)); }
                timings.push((uncached, prefill, start.elapsed() / steps as u32));
            }
            let (uncached, prefill, cached) = timings[1];
            println!("{name}: context {prompt}-{}: full prefix {uncached:?}/token, KV cache {cached:?}/token ({:.1}x), prefill {prefill:?}",
                prompt + steps, uncached.as_secs_f64() / cached.as_secs_f64());
        }
    }
}
