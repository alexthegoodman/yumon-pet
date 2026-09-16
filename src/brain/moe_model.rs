//! Dropless sparse MoE decoder. Only routed rows enter expert matrix multiplies.
//! Dispatch indices are read once per layer for variable-sized GPU batches.
//! Activations/weights stay on device. Benchmark the synchronization overhead.
use super::{
    bpe::{BpeTokenizer, TokenizerKind},
    fixer::fix_json_syntax,
    model::{GenerationResult, MLP, MLPConfig, RMSNorm, RMSNormConfig, TEMPERATURE, TOP_K},
    samples::{Action, CardinalDir, TrainingStage},
    tokenizer::{BOS_TOKEN, EOS_TOKEN, PAD_TOKEN},
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
        IndexingUpdateOp,
        activation::{log_softmax, softmax},
    },
};
use serde::{Deserialize, Serialize};

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
        let logits = self.router.forward(x.clone());
        let probabilities = softmax(logits.clone(), 1);
        let choices = if self.top_k == 1 {
            logits.clone().detach().argmax(1)
        } else {
            logits.clone().detach().topk_with_indices(self.top_k, 1).1
        };
        // One host read of integer decisions; no dense expert output tensors.
        let ids = choices.to_data().to_vec::<i32>().expect("router indices");
        let mut rows = vec![Vec::<i32>::new(); n];
        for (slot, expert) in ids.into_iter().enumerate() {
            rows[expert as usize].push((slot / self.top_k) as i32);
        }
        let counts: Vec<usize> = rows.iter().map(Vec::len).collect();
        let fractions = Tensor::<B, 2>::from_data(
            TensorData::new(
                counts
                    .iter()
                    .map(|&c| c as f32 / (tokens * self.top_k) as f32)
                    .collect::<Vec<_>>(),
                [1, n],
            ),
            &device,
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
                * gate;
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
impl<B: Backend> MoeBlock<B> {
    fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEncoding<B>,
        active: Tensor<B, 1, Int>,
        pad: Tensor<B, 2, Bool>,
    ) -> (Tensor<B, 3>, Tensor<B, 1>, Tensor<B, 1>, Vec<usize>) {
        let [batch, seq, width] = x.dims();
        let device = x.device();
        let hd = width / self.heads;
        let norm = self.attn_norm.forward(x.clone());
        let split = |t: Tensor<B, 3>| t.reshape([batch, seq, self.heads, hd]).swap_dims(1, 2);
        let q = rope.forward(split(self.q.forward(norm.clone()))) / (hd as f64).sqrt();
        let k = rope.forward(split(self.k.forward(norm.clone())));
        let v = split(self.v.forward(norm));
        let future = Tensor::<B, 2>::ones([seq, seq], &device).triu(1).bool();
        let mask = future
            .unsqueeze::<4>()
            .bool_or(pad.unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(2));
        // Finite mask avoids NaNs in fully padded rows (excluded from MoE/loss).
        let weights = softmax(q.matmul(k.transpose()).mask_fill(mask, -1e9), 3);
        let attn = weights
            .matmul(v)
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
impl<B: Backend> YumonMoeBrain<B> {
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
        let pad = tokens.clone().equal_elem(PAD_TOKEN as i32);
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
        let mut x = self.dropout.forward(self.embedding.forward(tokens));
        let mut auxiliary = Tensor::zeros([1], &device);
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
                block.forward(x, &self.rope, index.clone(), pad.clone());
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
        let mut dec_ids: Vec<usize> = vec![BOS_TOKEN];
        if !seed_text.is_empty() {
            dec_ids.extend(tokenizer.encode(seed_text).iter().map(|&t| t as usize));
        }

        let sep_text = if self.config.training_stage == TrainingStage::Structured {
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
            if current_len >= self.config.max_seq_len {
                break;
            }

            let dec_tokens_t = Tensor::<B, 2, Int>::from_ints(
                TensorData::new(
                    dec_ids.iter().map(|&t| t as i32).collect::<Vec<_>>(),
                    [1, current_len],
                ),
                device,
            );

            let token_logits = self.forward(dec_tokens_t);

            let vocab_size = tokenizer.vocab_size();
            let last_logits = token_logits
                .slice([0..1, current_len - 1..current_len, 0..vocab_size])
                .reshape([vocab_size]);

            let logits_vec: Vec<f32> = last_logits.to_data().to_vec().unwrap();
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
    pub final_loss: f32,
    pub batch_size: usize,
    pub training_stage: TrainingStage,
    pub embed_dim: usize,
    pub hidden_units: usize,
    pub n_layers: usize,
    pub attn_heads: usize,
    pub ff_dim: usize,
    pub max_seq_len: usize,
}

// ─── Sampling helper (same shape as model.rs's private copy) ─────────────────

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
        backend::{Autodiff, Wgpu},
        module::{AutodiffModule, Param},
        optim::{AdamWConfig, GradientsParams, Optimizer},
    };
    type B = Autodiff<Wgpu>;

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
