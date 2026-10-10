//! Dropless sparse MoE decoder. Only routed rows enter expert matrix multiplies.
//! Router scores are read once per layer for deterministic top-k dispatch.
//! Activations/weights stay on device. Benchmark the synchronization overhead.
use super::{
    bpe::{BpeTokenizer, TokenizerKind},
    decoder_model::{generate_causal, apply_rope, RopeTables},
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
        activation::{log_softmax, softmax},
    },
};
use serde::{Deserialize, Serialize};

/// The dispatcher already synchronizes with the host once per layer. Read the
/// small [tokens, experts] router matrix there and select exact top-k explicitly.
/// This avoids Burn's dtype-dependent host sort and CubeCL's argmax sentinel.
fn router_choices<B: Backend>(logits: Tensor<B, 2>, top_k: usize) -> Result<Vec<i32>> {
    let [_, experts] = logits.dims();
    let scores = logits.detach().into_data().convert::<f32>().to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("reading FP32 router scores: {e:?}"))?;
    select_experts(&scores, experts, top_k)
}

fn select_experts(scores: &[f32], experts: usize, top_k: usize) -> Result<Vec<i32>> {
    anyhow::ensure!(experts > 0 && experts <= i32::MAX as usize
        && top_k > 0 && top_k <= experts && scores.len() % experts == 0,
        "invalid router dimensions or top-k");
    let mut choices = Vec::with_capacity(scores.len() / experts * top_k);
    let mut order: Vec<usize> = (0..experts).collect();
    for (token, row) in scores.chunks_exact(experts).enumerate() {
        for (expert, score) in row.iter().enumerate() {
            anyhow::ensure!(score.is_finite(),
                "non-finite router logit at compact token {token}, expert {expert}: {score}; \
                 training stopped before dispatch. Check model/optimizer numerical stability; \
                 invalid scores must not be clamped into valid expert IDs");
        }
        // Exact descending score order; equal scores prefer the smaller expert ID.
        order.sort_unstable_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
        choices.extend(order[..top_k].iter().map(|&id| id as i32));
    }
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
        assert!(width > 0 && hidden > 0 && experts > 0);
        assert!(top_k > 0 && top_k <= experts);
        Self {
            router: LinearConfig::new(width, experts)
                .with_bias(false)
                .init(device),
            experts: (0..experts)
                .map(|_| MLPConfig::new(width, hidden).init(device))
                .collect(),
            top_k,
        }
    }
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
        // One host read of router scores; expert activations and gates stay on device.
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
    fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RopeTables<B>,
        active: Tensor<B, 1, Int>,
        pad: Option<Tensor<B, 2, Bool>>,
    ) -> (Tensor<B, 3>, Tensor<B, 1>, Tensor<B, 1>, Vec<usize>) {
        let [batch, seq, width] = x.dims();
        let device = x.device();
        let hd = width / self.heads;
        let norm = self.attn_norm.forward(x.clone());
        let split = |t: Tensor<B, 3>| t.reshape([batch, seq, self.heads, hd]).swap_dims(1, 2);
        let q = apply_rope(split(self.q.forward(norm.clone())), rope);
        let k = apply_rope(split(self.k.forward(norm.clone())), rope);
        let v = split(self.v.forward(norm));
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
        }
            .swap_dims(1, 2)
            .reshape([batch, seq, width]);
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
}
impl YumonMoeBrainConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> YumonMoeBrain<B> {
        assert!(self.n_layers > 0 && self.max_seq_len > 0 && self.vocab_size > 0);
        assert!(self.attn_heads > 0 && self.embed_dim > 0 && self.embed_dim % self.attn_heads == 0);
        assert!(
            (self.embed_dim / self.attn_heads) % 2 == 0,
            "RoPE requires even head width"
        );
        assert!(self.aux_loss_weight.is_finite() && self.aux_loss_weight >= 0.0);
        assert!(self.z_loss_weight.is_finite() && self.z_loss_weight >= 0.0);
        let linear = || {
            LinearConfig::new(self.embed_dim, self.embed_dim)
                .with_bias(false)
                .init(device)
        };
        YumonMoeBrain {
            config: Ignored(self.clone()),
            embedding: EmbeddingConfig::new(self.vocab_size, self.embed_dim).init(device),
            rope: RotaryEncodingConfig::new(self.max_seq_len, self.embed_dim / self.attn_heads)
                .init(device),
            blocks: (0..self.n_layers)
                .map(|_| MoeBlock {
                    attn_norm: RMSNormConfig::new(self.embed_dim).init(device),
                    q: linear(),
                    k: linear(),
                    v: linear(),
                    o: linear(),
                    ffn_norm: RMSNormConfig::new(self.embed_dim).init(device),
                    moe: SparseMoe::new(
                        self.embed_dim,
                        self.ff_dim,
                        self.num_experts,
                        self.top_k,
                        device,
                    ),
                    heads: self.attn_heads,
                })
                .collect(),
            norm: RMSNormConfig::new(self.embed_dim).init(device),
            dropout: DropoutConfig::new(self.dropout_rate).init(),
            token_head: LinearConfig::new(self.embed_dim, self.vocab_size).init(device),
        }
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
        if active_count == 0 {
            return (
                self.token_head.forward(self.norm.forward(x)),
                auxiliary,
                vec![vec![0; self.config.num_experts]; self.blocks.len()],
            );
        }
        let index =
            Tensor::<B, 1, Int>::from_ints(TensorData::new(active, [active_count]), &device);
        for block in &self.blocks {
            let (next, balance, z_loss, dispatched) =
                block.forward(x, &rope, index.clone(), pad.clone());
            x = next;
            auxiliary = auxiliary
                + balance * self.config.aux_loss_weight
                + z_loss * self.config.z_loss_weight;
            counts.push(dispatched);
        }
        (
            self.token_head.forward(self.norm.forward(x)),
            auxiliary / self.blocks.len() as f64,
            counts,
        )
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
        let record = recorder
            .load(dir.join("model").into(), device)
            .map_err(|e| anyhow::anyhow!("load: {e:?}"))?;

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
    fn moe_router_extreme_scores_ties_and_invalid_values() {
        let rows = [vec![0.0; 16], vec![f32::MIN; 16],
            (0..16).map(|i| -100.0 + i as f32).collect::<Vec<_>>()];
        for k in [1, 2, 16] {
            let actual = select_experts(&rows.concat(), 16, k).unwrap();
            let expected: Vec<i32> = (0..k as i32).chain(0..k as i32)
                .chain((0..16).rev().take(k)).collect();
            assert_eq!(actual, expected);
            for row in actual.chunks_exact(k) {
                let unique: std::collections::HashSet<_> = row.iter().collect();
                assert_eq!(unique.len(), k);
                assert!(row.iter().all(|&id| (0..16).contains(&id)));
            }
        }
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut scores = vec![0.0; 32];
            scores[19] = bad;
            let error = select_experts(&scores, 16, 2).unwrap_err().to_string();
            assert!(error.contains("compact token 1, expert 3"), "{error}");
            assert!(error.contains("non-finite router logit"));
        }
    }

    #[test]
    fn moe_router_fp32_logits_on_bf16_backend() {
        // Reproduce the CUDA dtype combination locally without requiring BF16
        // arithmetic: backend default BF16, actual router tensor explicitly FP32.
        type Mixed = Autodiff<burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime,
            burn::tensor::bf16, i32, u32>, BalancedCheckpointing>;
        let device = Default::default();
        let logits = Tensor::<Mixed, 2>::from_data_dtype(
            TensorData::new(vec![1.0f32, 4.0, 2.0, 3.0, -4.0, -1.0, -3.0, -2.0], [2, 4]),
            &device, burn::tensor::DType::F32);
        assert_eq!(logits.dtype(), burn::tensor::DType::F32);
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
        type AD = crate::brain::train::CudaTrainBackend;
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
}
