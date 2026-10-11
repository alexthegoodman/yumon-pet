//! Reference decoder-only transformer built only from Burn's stock modules:
//! `Embedding` (tokens and learned positions), `nn::transformer::TransformerEncoder`
//! with an autoregressive mask (Burn's GPT-style pattern: a decoder-only LM is an
//! encoder stack with a causal mask), `LayerNorm` and `Linear`.
//!
//! No FlashAttention, RoPE, RMSNorm or SwiGLU: a deliberately plain baseline to
//! compare the custom models against. If this learns a task the custom decoder
//! does not, suspect the custom attention path. If both stall, suspect the data.
//!
//! Padding: batches are right-padded, so the causal mask alone keeps real tokens
//! from seeing PAD keys, same as the other decoders. Attention scores are
//! materialized ([batch, heads, seq, seq]), so this costs more memory than flash.
use super::{
    bpe::{BpeTokenizer, TokenizerKind},
    decoder_model::generate_causal,
    model::GenerationResult,
    samples::TrainingStage,
};
use anyhow::Result;
use burn::{
    module::Ignored,
    nn::{
        Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
        attention::generate_autoregressive_mask,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
};
use serde::{Deserialize, Serialize};

#[derive(Module, Debug)]
pub struct YumonReferenceBrain<B: Backend> {
    pub config: Ignored<YumonReferenceBrainConfig>,
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    transformer: TransformerEncoder<B>,
    norm: LayerNorm<B>,
    pub token_head: Linear<B>,
}

#[derive(Config, Debug)]
pub struct YumonReferenceBrainConfig {
    pub vocab_size: usize,
    pub training_stage: TrainingStage,
    #[config(default = 0.05)]
    pub dropout_rate: f64,
    #[config(default = 256)]
    pub embed_dim: usize,
    #[config(default = 2)]
    pub n_layers: usize,
    #[config(default = 4)]
    pub attn_heads: usize,
    #[config(default = 1024)]
    pub ff_dim: usize,
    #[config(default = 320)]
    pub max_seq_len: usize,
}

impl YumonReferenceBrainConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> YumonReferenceBrain<B> {
        assert!(self.n_layers > 0 && self.max_seq_len > 0 && self.vocab_size > 0);
        assert!(self.attn_heads > 0 && self.embed_dim > 0 && self.embed_dim % self.attn_heads == 0);
        YumonReferenceBrain {
            config: Ignored(self.clone()),
            token_embedding: EmbeddingConfig::new(self.vocab_size, self.embed_dim).init(device),
            position_embedding: EmbeddingConfig::new(self.max_seq_len, self.embed_dim).init(device),
            transformer: TransformerEncoderConfig::new(self.embed_dim, self.ff_dim, self.attn_heads, self.n_layers)
                .with_dropout(self.dropout_rate)
                .with_norm_first(true)
                .init(device),
            norm: LayerNormConfig::new(self.embed_dim).init(device),
            token_head: LinearConfig::new(self.embed_dim, self.vocab_size).init(device),
        }
    }
}

impl<B: Backend> YumonReferenceBrain<B> {
    /// Logits [batch, seq, vocab]. Right-padded input only.
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, seq] = tokens.dims();
        assert!(seq > 0 && seq <= self.config.max_seq_len);
        let device = tokens.device();
        let positions = Tensor::<B, 1, Int>::arange(0..seq as i64, &device).reshape([1, seq]);
        let x = self.token_embedding.forward(tokens) + self.position_embedding.forward(positions);
        let mask = generate_autoregressive_mask::<B>(batch, seq, &device);
        let x = self.transformer.forward(TransformerEncoderInput::new(x).mask_attn(mask));
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

    pub fn save(&self, directory: &str, tokenizer: &TokenizerKind, metadata: &RefMetadata) -> Result<()> {
        let dir = std::path::Path::new(directory);
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("metadata.json"), serde_json::to_string_pretty(metadata)?)?;
        match tokenizer {
            TokenizerKind::Bpe(t) => t.save(directory)?,
            TokenizerKind::Char(_) => anyhow::bail!("reference checkpoints require a BPE tokenizer"),
        }
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        self.clone()
            .save_file(dir.join("model"), &recorder)
            .map_err(|e| anyhow::anyhow!("save_file: {e:?}"))?;
        Ok(())
    }

    pub fn load(directory: &str, device: &B::Device) -> Result<(Self, TokenizerKind, YumonReferenceBrainConfig)> {
        let dir = std::path::Path::new(directory);
        let metadata: RefMetadata = serde_json::from_str(&std::fs::read_to_string(dir.join("metadata.json"))?)?;
        let tokenizer = TokenizerKind::Bpe(BpeTokenizer::load(directory)?);
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let record = recorder
            .load(dir.join("model").into(), device)
            .map_err(|e| anyhow::anyhow!("load: {e:?}"))?;
        let config = YumonReferenceBrainConfig {
            vocab_size: metadata.vocab_size,
            training_stage: metadata.training_stage,
            dropout_rate: metadata.dropout_rate,
            embed_dim: metadata.embed_dim,
            n_layers: metadata.n_layers,
            attn_heads: metadata.attn_heads,
            ff_dim: metadata.ff_dim,
            max_seq_len: metadata.max_seq_len,
        };
        let model = config.init::<B>(device).load_record(record);
        Ok((model, tokenizer, config))
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RefMetadata {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::autodiff::{Autodiff, checkpoint::strategy::BalancedCheckpointing},
        module::AutodiffModule,
        optim::{AdamWConfig, GradientsParams, Optimizer},
    };
    use burn_cubecl::CubeBackend;
    use cubecl::wgpu::WgpuRuntime;

    type Train = Autodiff<CubeBackend<WgpuRuntime, f32, i32, u32>, BalancedCheckpointing>;

    fn values<const D: usize>(t: Tensor<impl Backend, D>) -> Vec<f32> {
        t.into_data().convert::<f32>().to_vec::<f32>().unwrap()
    }

    #[test]
    fn reference_is_causal_and_trains() {
        let device = Default::default();
        let config = YumonReferenceBrainConfig::new(32, TrainingStage::Language)
            .with_embed_dim(16).with_n_layers(2).with_attn_heads(2)
            .with_ff_dim(32).with_max_seq_len(8).with_dropout_rate(0.0);
        let model: YumonReferenceBrain<Train> = config.init(&device);
        let logits = model.forward(Tensor::from_ints([[1, 4, 5, 6, 0, 0]], &device));
        assert_eq!(logits.dims(), [1, 6, 32]);
        // Changing later tokens leaves earlier positions alone.
        let changed = model.forward(Tensor::from_ints([[1, 4, 9, 8, 0, 0]], &device));
        let (a, b) = (values(logits.clone().slice([0..1, 0..2, 0..32])), values(changed.slice([0..1, 0..2, 0..32])));
        assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-5), "future tokens leaked into earlier positions");
        let grads = GradientsParams::from_grads(logits.powf_scalar(2.0).mean().backward(), &model);
        let updated = AdamWConfig::new().init().step(1e-3, model, grads);
        assert!(values(updated.valid().forward(Tensor::from_ints([[1, 2, 3]], &device))).iter().all(|x| x.is_finite()));
    }
}
