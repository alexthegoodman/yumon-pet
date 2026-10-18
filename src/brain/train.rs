/// Yumon Brain training loop

use anyhow::Result;
use burn::{
    grad_clipping::GradientClippingConfig, module::AutodiffModule, nn::loss::CrossEntropyLossConfig, optim::{AdamConfig, AdamWConfig, GradientsParams, Optimizer}, prelude::*, tensor::{Int, TensorData, backend::AutodiffBackend}
};
#[cfg(not(target_arch = "wasm32"))]
use rand::{Rng, rngs::StdRng, seq::SliceRandom, thread_rng};
#[cfg(not(target_arch = "wasm32"))]
use indicatif::{ProgressBar, ProgressStyle};
#[cfg(not(target_arch = "wasm32"))]
use ratatui::{Terminal, TerminalOptions, Viewport, prelude::CrosstermBackend};
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use rand::SeedableRng;

use crate::{brain::{PAD_TOKEN, bpe::{BpeTokenizer, CL_ID, CR_ID, TokenizerKind}, chart::{TrainingState}, loader::{DataLoader, FileKind}, samples::{TrainingStage, WorldContext, prepare_paired_samples_split, prepare_paired_samples_split_sep}}, vision::{CIFAR_CLASSES, EMOTE_CLASSES, EMOTE_NAMES}};

use crate::brain::{
    decoder_model::{DecMetadata, YumonDecBrain, YumonDecBrainConfig},
    flash_attn::backend::FlashAttention,
};

#[cfg(not(target_arch = "wasm32"))]
use crate::brain::{chart::render};
#[cfg(not(target_arch = "wasm32"))]
use crate::brain::loss_stats::{TokenLossStats, TokenView, save_batch_heatmap, write_token_html};

#[cfg(not(target_arch = "wasm32"))]
use crate::brain::mdx::{load_dictionary_sentences, load_handcrafted_sentences, load_qa_pairs, load_qa_singles, load_txt_sentences};
#[cfg(not(target_arch = "wasm32"))]
use crate::brain::pdf::{load_pdf_ebook_sentences, load_pdfs};
#[cfg(not(target_arch = "wasm32"))]
use crate::brain::wiki::{save_sentence_pairs_to_file};

use crate::brain::{
    // CONTEXT_DIMS,
    tokenizer::{Tokenizer, BOS_TOKEN, EOS_TOKEN},
    model::{YumonBrain, YumonBrainConfig, BrainMetadata, GenerationResult},
    moe_model::{YumonMoeBrain, YumonMoeBrainConfig, MoeMetadata},
    xlstm_model::{YumonXLstmBrain, YumonXLstmBrainConfig, XLstmMetadata},
    reference_model::{RefMetadata, YumonReferenceBrain, YumonReferenceBrainConfig},
};

// Used by the local desktop training UI (src/bin/train_ui.rs), which runs on
// wgpu since dev machines typically have no CUDA GPU.
//
// Both training backends are plain CubeBackend (no burn-fusion) so the custom
// flash attention op in flash_attn/backend.rs applies, and use Burn's
// BalancedCheckpointing: memory-bound ops (elementwise, norms, activations,
// reshapes) are recomputed during backward instead of stored.
#[cfg(not(any(feature = "cuda-training", all(target_os = "linux", not(feature = "desktop")))))]
pub type TrainBackend = burn::backend::Autodiff<
    burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>,
    burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing,
>;
#[cfg(not(any(feature = "cuda-training", all(target_os = "linux", not(feature = "desktop")))))]
pub type TrainRuntime = cubecl::wgpu::WgpuRuntime;
#[cfg(any(feature = "cuda-training", all(target_os = "linux", not(feature = "desktop"))))]
pub type TrainBackend = CudaTrainBackend;
#[cfg(any(feature = "cuda-training", all(target_os = "linux", not(feature = "desktop"))))]
pub type TrainRuntime = CudaTrainRuntime;
// pub type TrainBackend = burn::backend::Autodiff<burn::backend::NdArray<f32>>;

// Used by the headless `train-brain` CLI path (`run`, below) — this is what
// RunPod/Docker actually runs. CUDA talks to the driver directly and needs no
// Vulkan/GL adapter, unlike wgpu, which RunPod's driver stack doesn't expose.
// FP32 training on CUDA matches local WGPU precision, including AdamW moments.
// Keep the explicit BF16 alias for optional kernel regression tests.
pub type CudaTrainBackend = burn::backend::Autodiff<
    burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime, f32, i32, u8>,
    burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing,
>;
pub type CudaBf16TrainBackend = burn::backend::Autodiff<
    burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime, burn::tensor::bf16, i32, u8>,
    burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing,
>;
pub type CudaTrainRuntime = cubecl::cuda::CudaRuntime;

// Max sequence length during training (tokens)
// pub const MAX_SEQ_LEN:  usize = 120;
// pub const MAX_SEQ_LEN:  usize = 25;
// pub const MAX_SEQ_LEN:  usize = 512;
// pub const MAX_SEQ_LEN:  usize = 1024;
// pub const MAX_SEQ_LEN:  usize = 256;
pub const MAX_SEQ_LEN:  usize = 200;
pub const MAX_SEQ_LEN_CHARS:  usize = 200;
// pub const MAX_SEQ_LEN:  usize = 180;
// pub const MAX_SEQ_LEN:  usize = 90;
// pub const MAX_SEQ_LEN:  usize = 100;
// pub const MAX_SEQ_LEN:  usize = 80; // better for outlines structured output?
// pub const MAX_SEQ_LEN:  usize = 60; // lighter to train on iGPU
// pub const MAX_SEQ_LEN:  usize = 40; // even lower with bpe

// ─── CIFAR-100 fine label table (index 0..99, canonical order) ───────────────
//
// Multi-word labels (e.g. "lawn_mower") are split into constituent keywords so
// that wiki sentences mentioning "lawn" or "mower" both match.
// Single-word labels are also lowercased and stripped of underscores.

const CIFAR_FINE_LABELS: [&str; CIFAR_CLASSES] = [
    "apple", "aquarium_fish", "baby", "bear", "beaver",
    "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly",
    "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach",
    "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox",
    "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid",
    "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe",
    "whale", "willow_tree", "wolf", "woman", "worm",
];

// ─── Inverted index ───────────────────────────────────────────────────────────

/// For each CIFAR class index, the set of keywords that map to it.
/// Multi-word labels contribute all their parts.
pub fn build_label_keywords() -> Vec<Vec<String>> {
    CIFAR_FINE_LABELS.iter().map(|label| {
        label.split('_')
             .map(|w| w.to_lowercase())
             .filter(|w| w.len() >= 3) // skip tiny words like "a", "of"
             .collect()
    }).collect()
}

/// Maps keyword → list of CIFAR class indices that contain it.
pub fn build_keyword_index(label_keywords: &[Vec<String>]) -> HashMap<String, Vec<usize>> {
    let mut idx: HashMap<String, Vec<usize>> = HashMap::new();
    for (class_i, keywords) in label_keywords.iter().enumerate() {
        for kw in keywords {
            idx.entry(kw.clone()).or_default().push(class_i);
        }
    }
    idx
}

/// Given a sentence, return all CIFAR class indices whose keywords appear in it.
/// Uses whole-word matching: "plain" should not match "explanation".
pub fn matched_classes(sentence: &str, keyword_index: &HashMap<String, Vec<usize>>) -> Vec<usize> {
    let lower = sentence.to_lowercase();
    let mut matched = std::collections::HashSet::new();

    for (kw, class_indices) in keyword_index {
        // Whole-word check: the keyword must be surrounded by non-alpha characters
        if whole_word_match(&lower, kw) {
            for &ci in class_indices {
                matched.insert(ci);
            }
        }
    }

    let mut v: Vec<usize> = matched.into_iter().collect();
    v.sort();
    v
}

/// True if `kw` appears in `text` as a whole word (not a substring of another word).
pub fn whole_word_match(text: &str, kw: &str) -> bool {
    let kw_bytes = kw.as_bytes();
    let text_bytes = text.as_bytes();
    let klen = kw_bytes.len();

    if klen > text_bytes.len() { return false; }

    for start in 0..=(text_bytes.len() - klen) {
        if &text_bytes[start..start + klen] == kw_bytes {
            let before_ok = start == 0 || !text_bytes[start - 1].is_ascii_alphabetic();
            let after_ok  = start + klen == text_bytes.len()
                          || !text_bytes[start + klen].is_ascii_alphabetic();
            if before_ok && after_ok { return true; }
        }
    }
    false
}

pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum = exps.iter().sum::<f32>();
    exps.iter().map(|e| e / sum).collect()
}

// ─── Main training entry point ────────────────────────────────────────────────

pub struct StageConfig {
    pub stage: TrainingStage,
    pub loss_threshold: f32,
    pub epochs: usize,
    pub batch_size: usize,
    pub first_lr: f64,
    pub last_lr: f64,
    pub weight_decay: f32,
    pub epsilon: f32,
    pub smoothing: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// YumonBrain (model.rs) — separate encoder/decoder stacks with cross-attention.
    EncoderDecoder,
    /// YumonDecBrain (decoder_model.rs) — single causal stack, prompt+reply as one sequence.
    DecoderOnly,
    /// YumonXLstmBrain (xlstm_model.rs) — stacked mLSTM blocks (exponential-gated
    /// matrix memory), prompt+reply packed as one sequence like DecoderOnly, but
    /// recurrent rather than attention-based.
    XLstm,
    /// Causal transformer with dropless sparse SwiGLU experts.
    Moe { num_experts: usize, top_k: usize },
    /// Stock `burn::nn::transformer` baseline (reference_model.rs).
    Reference,
}

pub struct RunConfig {
    pub name: String,
    pub embed_dim: usize,
    pub hidden_units: usize,
    pub n_layers: usize,
    pub attn_heads: usize,
    pub ff_dim: usize,
    pub max_seq_len: usize,
    pub architecture: Architecture,
    pub stages: Vec<StageConfig>,
    pub dropout_rate: f64,
    /// Sanity-check run: fewer log lines, generation only every 25 epochs and at the end.
    pub overfit: bool,
    /// Steps between loss heatmap/plot snapshots; 0 disables.
    pub heatmap_every: usize,
    /// Adaptive learning-value token weighting; raw-code runs only.
    #[cfg(not(target_arch = "wasm32"))]
    pub adaptive_loss: Option<crate::brain::adaptive_loss::AdaptiveLossConfig>,
    /// MoE depth tiers with per-tier experts and exit gates.
    pub adaptive_depth: Option<crate::brain::adaptive_depth::AdaptiveDepthConfig>,
    /// MoE grouped-query attention KV heads; None = attn_heads.
    pub kv_heads: Option<usize>,
}

/// Number of held-out prompts run through the model every time we do a
/// qualitative inference check, for a well-rounded read on output quality
/// (a single prompt can look fine or awful by chance).
fn code_eval_prompts() -> Vec<String> {
    vec![
        "pub struct RunConfig {".to_string(),
        "/// A video encoder accepts pixel data".to_string(),
        "/// Process the data".to_string(),
        "pub fn softmax(logits: &[f32]) -> Vec<f32> {".to_string(),
        "let reverb_params = |mix: f64| {".to_string(),
    ]
}

fn kilo_eval_prompts() -> Vec<String> {
    vec![
        "# [ derive ( Debug , Clone , Default ) ] pub struct T0 {".to_string(),
        "impl T0 { pub fn fn0 ( & mut self , arg0 : & T1 ) -> bool {".to_string(),
        "pub async fn fn0 ( arg0 : & str ) -> Result < T0 , T1 > { let mut v0 =".to_string(),
        "match v0 { Ok ( v1 ) =>".to_string(),
        "v0 . iter ( ) . map ( | v1 | v1 . fn0 ( ) ) . filter ( | v1 |".to_string(),
        "fn fn0 ( mut arg0 : Commands , arg1 : Query < ( Entity , & T0 ) > ) {".to_string(),
    ]
}

/// Yumon Code uses the same trainer locally (WGPU) and on RunPod (CUDA).
#[cfg(not(target_arch = "wasm32"))]
pub fn run_code(config: &crate::brain::code_corpus::CodeConfig, check_only: bool) -> Result<()> {
    let prompts = if config.cache.to_string_lossy().contains("talk") {
        eval_prompts()
    } else if config.vocab_size == 1024 || config.cache.to_string_lossy().contains("kilo") {
        kilo_eval_prompts()
    } else {
        code_eval_prompts()
    };
    use crate::brain::{code_corpus, code_curriculum};
    config.validate()?;
    let tokenizer = code_corpus::load_tokenizer(&config.tokenizer)?;
    let mut prepared = code_curriculum::prepare(config, &tokenizer)?;
    let overfit = config.overfit_samples.is_some();
    if let Some(count) = config.overfit_samples {
        // Fixed subset, no pretraining stage or tier schedule; validate on the same sequences.
        prepared.training.drain(..prepared.pretrain_len);
        prepared.pretrain_len = 0;
        prepared.tier_ends = None;
        anyhow::ensure!(prepared.training.len() >= count, "overfit_samples {count} exceeds {} training samples", prepared.training.len());
        prepared.training.truncate(count);
        prepared.validation = prepared.training.clone();
        if prepared.objective == crate::brain::sample_cache::CacheObjective::RawCode {
            let (mut matching, mut total) = (0usize, 0usize);
            for s in &prepared.training {
                for i in 0..s.target_labels.len().saturating_sub(1) {
                    // The final target (e.g. EOS) has no following input token.
                    if s.target_labels[i] == PAD_TOKEN || s.input_ids[i + 1] == PAD_TOKEN { continue; }
                    total += 1;
                    matching += (s.target_labels[i] == s.input_ids[i + 1]) as usize;
                }
            }
            println!("Overfit run: {count} fixed sequences, dropout 0, constant LR {}. Label shift check: {matching}/{total} targets equal the next input token.", config.first_lr);
            anyhow::ensure!(matching == total, "targets are not input shifted by one position");
        } else {
            println!("Overfit run: {count} fixed sequences, dropout 0, constant LR {}.", config.first_lr);
        }
    }
    prepared.print_plan(config.epochs, config.batch_size);
    if check_only { return Ok(()); }

    {
        let architecture = match config.architecture {
            code_corpus::CodeArchitecture::DecoderOnly => Architecture::DecoderOnly,
            code_corpus::CodeArchitecture::Moe => Architecture::Moe {
                num_experts: config.num_experts, top_k: config.top_k,
            },
            code_corpus::CodeArchitecture::XLstm => Architecture::XLstm,
            code_corpus::CodeArchitecture::EncoderDecoder => Architecture::EncoderDecoder,
            code_corpus::CodeArchitecture::Reference => Architecture::Reference,
        };
        let arch_tag = match architecture {
            Architecture::Moe { .. } if config.adaptive_depth.is_some() => config.adaptive_depth.as_ref().unwrap().tag(),
            Architecture::Moe { num_experts, top_k } => format!("Moe_e{num_experts}_k{top_k}"),
            Architecture::DecoderOnly => "DecoderOnly".to_string(),
            Architecture::XLstm => "XLstm".to_string(),
            Architecture::EncoderDecoder => "EncoderDecoder".to_string(),
            Architecture::Reference => "Reference".to_string(),
        };
        let run = RunConfig {
            // Depth tiers carry their own expert widths in arch_tag.
            name: format!("code_{}h_{}l_{}a_{}{}len_b{}_{arch_tag}_f32{}{}", config.embed_dim,
                config.n_layers, config.attn_heads,
                if config.adaptive_depth.is_some() { String::new() } else { format!("{}ff_", config.ff_dim) },
                config.max_seq_len, config.batch_size,
                if config.curriculum.is_some() { "_curriculum_documented_tiers_v3" } else { "" },
                if config.docs.is_some() { "_rust_docs_v1".to_owned() } else if config.pretrain_code {
                    format!("_code_pretrain_v1{}", if config.pretrain_min_tokens > 0 { format!("_min{}", config.pretrain_min_tokens) } else { String::new() })
                } else if config.cache.to_string_lossy().contains("talk") {
                    "_yumon_talk_v1".to_owned()
                } else { String::new() })
                + &config.overfit_samples.map_or(String::new(), |n| format!("_overfit{n}"))
                // Separate directories so A/B/C/D experiments never resume each other.
                + &config.adaptive_loss.as_ref().map_or(String::new(), |a| format!("_alw_{}", a.effective_mode().tag()))
                + &config.kv_heads.filter(|&kv| kv != config.attn_heads).map_or(String::new(), |kv| format!("_gqa{kv}")),
            embed_dim: config.embed_dim, hidden_units: config.embed_dim,
            n_layers: config.n_layers, attn_heads: config.attn_heads, ff_dim: config.ff_dim,
            max_seq_len: config.max_seq_len, architecture,
            stages: vec![StageConfig { stage: TrainingStage::Language, loss_threshold: 0.0,
                epochs: config.epochs, batch_size: config.batch_size, first_lr: config.first_lr,
                last_lr: if overfit { config.first_lr } else { config.last_lr },
                weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.0 }],
            dropout_rate: if overfit { 0.0 } else { 0.05 },
            overfit, heatmap_every: config.heatmap_every,
            adaptive_loss: config.adaptive_loss.clone(),
            adaptive_depth: config.adaptive_depth.clone(),
            kv_heads: config.kv_heads.filter(|&kv| kv != config.attn_heads),
        };
        let run_dir = config.out_dir.join(&run.name);
        std::fs::create_dir_all(&run_dir)?;
        // Preserve the effective config and cache identity alongside checkpoints.
        let identity = serde_json::json!({"config": config, "cache": prepared.identity});
        let identity_path = run_dir.join("code-config.json");
        if identity_path.exists() {
            let previous: serde_json::Value = serde_json::from_slice(&std::fs::read(&identity_path)?)?;
            anyhow::ensure!(previous["cache"] == identity["cache"], "checkpoint cache identity differs; use a new out_dir");
        }
        std::fs::write(identity_path, serde_json::to_vec_pretty(&identity)?)?;
        let device = <TrainBackend as Backend>::Device::default();
        if config.adaptive_loss.is_some() {
            // Adaptive-loss experiment arms must share initialization and dropout streams.
            <TrainBackend as Backend>::seed(&device, config.seed);
        }
        println!("Training precision: FP32; Flash Attention + BalancedCheckpointing.");
        match architecture {
            Architecture::Moe { .. } => train_causal_lm::<YumonMoeBrain<TrainBackend>>(
                &run, &run_dir, &tokenizer, &HashMap::new(), &prompts, &device, Some(prepared))?,
            Architecture::DecoderOnly => train_causal_lm::<YumonDecBrain<TrainBackend>>(
                &run, &run_dir, &tokenizer, &HashMap::new(), &prompts, &device, Some(prepared))?,
            Architecture::XLstm => train_causal_lm::<YumonXLstmBrain<TrainBackend>>(
                &run, &run_dir, &tokenizer, &HashMap::new(), &prompts, &device, Some(prepared))?,
            Architecture::EncoderDecoder => train_causal_lm::<YumonBrain<TrainBackend>>(
                &run, &run_dir, &tokenizer, &HashMap::new(), &prompts, &device, Some(prepared))?,
            Architecture::Reference => train_causal_lm::<YumonReferenceBrain<TrainBackend>>(
                &run, &run_dir, &tokenizer, &HashMap::new(), &prompts, &device, Some(prepared))?,
        }
    }
    Ok(())
}

/// Tiny CUDA verification is kept independent of corpus preparation so the
/// deployed train_code binary can check the real RunPod backend before training.
pub fn cuda_moe_smoke_test() -> Result<()> {
    println!("CUDA/FP32 MoE smoke test: Flash Attention + BalancedCheckpointing");
    tiny_moe_smoke::<CudaTrainBackend>(&Default::default(), burn::tensor::DType::F32)
}

fn tiny_moe_smoke<B>(device: &B::Device, expected_dtype: burn::tensor::DType) -> Result<()>
where
    B: AutodiffBackend + FlashAttention,
    B::InnerBackend: FlashAttention,
{
    use burn::{record::{BinBytesRecorder, FullPrecisionSettings, Recorder}, tensor::FloatDType};
    let inputs = || Tensor::<B, 2, Int>::from_ints(
        [[1, 4, 5, 6, 0, 0, 0, 0], [1, 7, 8, 0, 0, 0, 0, 0]], device);
    let targets = || Tensor::<B, 2, Int>::from_ints(
        [[4, 5, 6, 2, 0, 0, 0, 0], [7, 8, 2, 0, 0, 0, 0, 0]], device).reshape([16]);
    // Match the current 16-expert/top-2 config; also cover top-1 and k=E.
    for top_k in [1, 2, 16] {
        let config = YumonMoeBrainConfig::new(32, TrainingStage::Language)
            .with_embed_dim(16).with_n_layers(2).with_attn_heads(2)
            .with_ff_dim(32).with_max_seq_len(8).with_dropout_rate(0.0)
            .with_num_experts(16).with_top_k(top_k);
        let mut model: YumonMoeBrain<B> = config.init(device);
        let before = model.token_head.weight.val().into_data().convert::<f32>().to_vec::<f32>().unwrap();
        let mut optimizer = AdamWConfig::new().with_epsilon(1e-7)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
            .with_weight_decay(0.01).init();
        // Exercise repeated optimizer updates beyond the reported ~50 batches.
        let steps = if top_k == 2 { 64 } else { 3 };
        for step in 1..=steps {
            let (out, aux, counts) = model.forward_with_aux(inputs());
            anyhow::ensure!(out.dtype() == expected_dtype, "unexpected model dtype");
            anyhow::ensure!(aux.dtype() == burn::tensor::DType::F32, "auxiliary loss must be FP32");
            anyhow::ensure!(counts.iter().all(|c| c.iter().sum::<usize>() == 7 * top_k), "incorrect dispatch counts");
            let logits = out.reshape([16, 32]);
            let correct = top3_correct(logits.clone().inner(), targets().inner());
            let loss = masked_token_ce(logits, targets(), 7, 0.0) + aux;
            let value = loss.clone().into_data().convert::<f32>().to_vec::<f32>()?[0];
            anyhow::ensure!(value.is_finite(), "non-finite loss at top-{top_k}, step {step}");
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(0.01, model, grads);
            println!("top-{top_k} step {step}/{steps}: loss={value:.6}, top3={correct}/7");
        }
        let after = model.token_head.weight.val().into_data().convert::<f32>().to_vec::<f32>()?;
        anyhow::ensure!(before != after && after.iter().all(|x| x.is_finite()), "weights did not update correctly");
        anyhow::ensure!(model.token_head.weight.val().dtype() == expected_dtype, "optimizer changed dtype");
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let bytes = recorder.record(model.clone().into_record(), ())
            .map_err(|e| anyhow::anyhow!("checkpoint save: {e:?}"))?;
        let record = recorder.load(bytes, device).map_err(|e| anyhow::anyhow!("checkpoint load: {e:?}"))?;
        let restored: YumonMoeBrain<B> = config.init(device).load_record(record);
        let reference = model.valid().forward(inputs().inner()).cast(FloatDType::F32)
            .into_data().to_vec::<f32>()?;
        let result = restored.valid().forward(inputs().inner()).cast(FloatDType::F32)
            .into_data().to_vec::<f32>()?;
        anyhow::ensure!(result.iter().zip(reference).all(|(a, b)| a.is_finite() && (a - b).abs() <= 1e-5 * (1.0 + b.abs())),
            "checkpoint predictions differ");
    }
    println!("MoE smoke test passed: 2 samples, 64 top-2 steps plus 3 each for top-1/top-16, checkpoint round trips.");
    Ok(())
}

/// A run is abandoned (rather than run to its full epoch count) if one epoch
/// fails to drop the average loss by at least this much versus the previous
/// epoch. Most grid-searched configs never converge at all, so this is what
/// makes covering the full grid in generate_run_configs() tractable.
// const MIN_EPOCH_LOSS_DROP: f32 = 0.2;
// const MIN_EPOCH_LOSS_DROP: f32 = 0.05;
const MIN_EPOCH_LOSS_DROP: f32 = 0.025;

/// Number of held-out prompts run through the model every time we do a
/// qualitative inference check, for a well-rounded read on output quality
/// (a single prompt can look fine or awful by chance).
fn eval_prompts() -> Vec<String> {
    vec![
        "Should I start a business?".to_string(),
        "What is the universe?".to_string(),
        "How do plants grow?".to_string(),
        "Tell me about friendship.".to_string(),
        "What should I do today?".to_string(),
        "The key was in the box. I moved it to the shelf. Where is the key?".to_string(),
        "Mia finished before Jo. Jo finished before Lee. Who finished last?".to_string(),
        "Copy these words in reverse order, separated by commas: pear, apple, plum.".to_string(),
        "If the door is closed, wait; otherwise enter. The door is closed. What will you do?".to_string(),
        "Bring me the cup. There are two cups, one blue and one red.".to_string(),
        "Human: My favorite color is blue.\nYumon: I will remember that.\nHuman: My favorite color is now green.\nYumon: I will remember green.\nHuman: What is my favorite color?".to_string(),
    ]
}

fn build_inference_prompt(stage: TrainingStage, message: &str) -> String {
    if stage == TrainingStage::Structured {
        serde_json::to_string_pretty(&serde_json::json!({
            "memories": Vec::<String>::new(),
            "message":  message,
        })).unwrap()
    } else {
        message.to_string()
    }
}

#[derive(Debug, Clone)]
pub struct EvalLogEntry {
    pub prompt: String,
    pub reply: String,
    pub yumon_prompt: Option<String>,
    pub yumon_reply: Option<String>,
}

pub fn is_yumon_talk_run(run_name: &str, tokenizer: &TokenizerKind, raw_code: bool) -> bool {
    let name_lower = run_name.to_lowercase();
    if name_lower.contains("kilo") {
        return false;
    }
    if name_lower.contains("talk") {
        return true;
    }
    #[cfg(not(target_arch = "wasm32"))]
    if std::env::var_os("YUMON_SAMPLE_CACHE").map_or(false, |c| c.to_string_lossy().to_lowercase().contains("talk")) {
        return true;
    }
    #[cfg(not(target_arch = "wasm32"))]
    if std::env::var("YUMON_TALK").map_or(false, |v| v == "1" || v.eq_ignore_ascii_case("true")) {
        return true;
    }
    if !raw_code && tokenizer.vocab_size() == crate::brain::yumon_talk::VOCAB_SIZE {
        return true;
    }
    false
}

fn evaluate_prompt<F>(
    p: &str,
    stage: TrainingStage,
    is_yumon_talk: bool,
    mut generate_fn: F,
) -> EvalLogEntry
where
    F: FnMut(&str) -> String,
{
    if is_yumon_talk {
        let (prompt_to_model, yumon_prompt_str, ctx) = match crate::brain::yumon_talk::english_to_yumon_with_map(p) {
            Ok((yt_prompt, ctx)) => (yt_prompt.clone(), Some(yt_prompt), Some(ctx)),
            Err(e) => {
                // eprintln!("[Yumon Talk] Eval prompt translation fallback for \"{p}\": {e}");
                (p.to_string(), None, None)
            }
        };
        let inference_prompt = build_inference_prompt(stage, &prompt_to_model);
        let raw_reply = generate_fn(&inference_prompt);
        let raw_reply_trimmed = raw_reply.trim().to_string();

        let mut rng = rand::thread_rng();
        let english_reply = if let Some(ref c) = ctx {
            match crate::brain::yumon_talk::yumon_to_english_with_context(&raw_reply_trimmed, c, &mut rng) {
                Ok(eng) => eng,
                Err(e) => {
                    // eprintln!("[Yumon Talk] Eval reply translation fallback: {e}");
                    crate::brain::yumon_talk::yumon_to_english(&raw_reply_trimmed).unwrap_or_else(|_| raw_reply_trimmed.clone())
                }
            }
        } else {
            crate::brain::yumon_talk::yumon_to_english(&raw_reply_trimmed).unwrap_or_else(|_| raw_reply_trimmed.clone())
        };

        EvalLogEntry {
            prompt: p.to_string(),
            reply: english_reply,
            yumon_prompt: yumon_prompt_str,
            yumon_reply: Some(raw_reply_trimmed),
        }
    } else {
        let prompt = build_inference_prompt(stage, p);
        let reply = generate_fn(&prompt);
        EvalLogEntry {
            prompt: p.to_string(),
            reply,
            yumon_prompt: None,
            yumon_reply: None,
        }
    }
}

/// Appends one qualitative-eval snapshot (all `entries` prompt/reply pairs)
/// to the run's inference log. Appends rather than overwrites so the log
/// reads as a history of how replies evolved over the run.
fn append_inference_log(
    log_path:      &str,
    run_name:      &str,
    stage_idx:     usize,
    stage:         TrainingStage,
    epoch:         usize,
    total_epochs:  usize,
    batch:         usize,
    total_batches: usize,
    avg_loss:      f32,
    entries:       &[EvalLogEntry],
) -> Result<()> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;

    writeln!(
        file,
        "\n=== {} | stage {} ({:?}) | epoch {}/{} batch {}/{} | avg_loss {:.4} ===",
        run_name, stage_idx + 1, stage, epoch, total_epochs, batch, total_batches, avg_loss,
    )?;
    for (i, entry) in entries.iter().enumerate() {
        match (&entry.yumon_prompt, &entry.yumon_reply) {
            (Some(yp), Some(yr)) => {
                writeln!(file, "[{}] PROMPT (English):    {}", i + 1, entry.prompt)?;
                writeln!(file, "[{}] PROMPT (Yumon Talk): {}", i + 1, yp)?;
                writeln!(file, "[{}] REPLY  (Yumon Talk): {}", i + 1, yr)?;
                writeln!(file, "[{}] REPLY  (English):    {}", i + 1, entry.reply)?;
            }
            (Some(yp), None) => {
                writeln!(file, "[{}] PROMPT (English):    {}", i + 1, entry.prompt)?;
                writeln!(file, "[{}] PROMPT (Yumon Talk): {}", i + 1, yp)?;
                writeln!(file, "[{}] REPLY:               {}", i + 1, entry.reply)?;
            }
            (None, Some(yr)) => {
                writeln!(file, "[{}] PROMPT:              {}", i + 1, entry.prompt)?;
                writeln!(file, "[{}] REPLY  (Yumon Talk): {}", i + 1, yr)?;
                writeln!(file, "[{}] REPLY  (English):    {}", i + 1, entry.reply)?;
            }
            (None, None) => {
                writeln!(file, "[{}] PROMPT: {}", i + 1, entry.prompt)?;
                writeln!(file, "[{}] REPLY:  {}", i + 1, entry.reply)?;
            }
        }
    }
    Ok(())
}

fn append_log_line(log_path: &str, line: &str) -> Result<()> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(log_path)?;
    writeln!(file, "{line}")?;
    Ok(())
}

/// Held-out samples for MoE validation loss: 1% of the data, capped so each
/// evaluation stays a small fraction of 500 training batches.
const MAX_VAL_SAMPLES: usize = 2048;

/// Decoder-only layout: [prompt][separator][reply] packed into one causal
/// sequence and padded. Targets are next tokens over the separator and reply
/// only (PAD elsewhere). Plain-text samples are [BOS][text][EOS] with every
/// token trained. Returns token ids, targets and the supervised count.
fn build_moe_batch(
    samples:     &[crate::brain::samples::Sample],
    batch_idx:   &[usize],
    sep_tokens:  &[usize],
    max_seq_len: usize,
    raw_code: bool,
) -> (Vec<i32>, Vec<i32>, usize) {
    let mut all_seq_ids: Vec<i32> = Vec::with_capacity(batch_idx.len() * max_seq_len);
    let mut all_targets: Vec<i32> = Vec::with_capacity(batch_idx.len() * max_seq_len);
    let mut supervised = 0usize;

    for &i in batch_idx {
        let sample = &samples[i];
        // Raw-code caches already contain shifted next-token targets.
        if raw_code {
            assert_eq!(sample.input_ids.len(), max_seq_len);
            assert_eq!(sample.target_labels.len(), max_seq_len);
            supervised += sample.target_labels.iter().filter(|&&t| t != PAD_TOKEN).count();
            all_seq_ids.extend(sample.input_ids.iter().map(|&t| t as i32));
            all_targets.extend(sample.target_labels.iter().map(|&t| t as i32));
            continue;
        }
        let input_ids = &sample.input_ids;
        let target_labels = &sample.target_labels;
        let sep_tokens = if crate::brain::samples::is_plain_text(sample) { &[][..] } else { sep_tokens };
        let sep_len = sep_tokens.len();

        let input_len = input_ids.iter().position(|&t| t == PAD_TOKEN).unwrap_or(input_ids.len())
            .min(max_seq_len - sep_len - 1);
        let target_len = target_labels.iter().position(|&t| t == PAD_TOKEN).unwrap_or(target_labels.len());

        let mut full_seq = Vec::with_capacity(max_seq_len);
        full_seq.extend(&input_ids[..input_len]);
        full_seq.extend(sep_tokens);

        let remaining_space = max_seq_len.saturating_sub(full_seq.len());
        let actual_target_len = target_len.min(remaining_space);
        full_seq.extend(&target_labels[..actual_target_len]);
        full_seq.resize(max_seq_len, PAD_TOKEN);

        // Loss only on the separator + target tokens, not the prompt.
        let mut loss_targets = vec![PAD_TOKEN as i32; max_seq_len];
        let start_predict_idx = input_len.saturating_sub(1);
        let end_predict_idx = (input_len + sep_len + actual_target_len).saturating_sub(1).min(max_seq_len - 1);
        for idx in start_predict_idx..end_predict_idx {
            loss_targets[idx] = full_seq[idx + 1] as i32;
        }
        supervised += loss_targets.iter().filter(|&&t| t != PAD_TOKEN as i32).count();

        all_seq_ids.extend(full_seq.iter().map(|&t| t as i32));
        all_targets.extend(loss_targets);
    }

    (all_seq_ids, all_targets, supervised)
}

/// Drops the all-PAD columns after the batch's longest row. Attention is causal,
/// PAD targets carry no loss and MoE routing skips PAD rows, so the result matches
/// the full context while short batches skip the unused positions (and logits).
fn trim_batch(ids: Vec<i32>, targets: Vec<i32>, rows: usize, max_seq_len: usize) -> (Vec<i32>, Vec<i32>, usize) {
    let pad = PAD_TOKEN as i32;
    let used = |row: &[i32]| row.iter().rposition(|&t| t != pad).map_or(0, |p| p + 1);
    let seq_len = ids.chunks(max_seq_len).zip(targets.chunks(max_seq_len))
        .map(|(i, t)| used(i).max(used(t))).max().unwrap_or(0).max(1);
    if seq_len == max_seq_len { return (ids, targets, seq_len); }
    let crop = |flat: Vec<i32>| -> Vec<i32> {
        flat.chunks(max_seq_len).take(rows).flat_map(|row| row[..seq_len].iter().copied()).collect()
    };
    (crop(ids), crop(targets), seq_len)
}

/// Number of supervised targets among the three highest logits. Softmax preserves
/// their order, so no probability tensor or full vocabulary sort is needed.
/// Ties prefer the lower token ID. Integer reductions avoid BF16 count rounding.
fn top3_correct<B: Backend>(logits: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> usize {
    let [rows, vocab] = logits.dims();
    let logits = logits.detach();
    let targets = targets.reshape([rows, 1]);
    let target_scores = logits.clone().gather(1, targets.clone());
    let ids = Tensor::<B, 1, Int>::arange(0..vocab as i64, &logits.device()).reshape([1, vocab]);
    let precedes = logits.clone().greater(target_scores.clone()).bool_or(
        logits.equal(target_scores).bool_and(ids.lower(targets.clone())),
    );
    precedes.int().sum_dim(1).lower_elem(3)
        .bool_and(targets.not_equal_elem(PAD_TOKEN as i32))
        .int().sum().into_scalar().elem::<i64>() as usize
}

/// Mean cross-entropy per supervised (non-PAD) target. Burn 0.20's
/// CrossEntropyLoss zeroes PAD rows but still divides by every row, which
/// scales the loss by the fraction of positions that are supervised.
fn masked_token_ce<B: Backend>(
    logits:     Tensor<B, 2>,
    targets:    Tensor<B, 1, Int>,
    supervised: usize,
    smoothing:  f32,
) -> Tensor<B, 1> {
    weighted_token_ce(logits, targets, None, supervised, smoothing)
}

/// `masked_token_ce` with optional per-row weights (adaptive loss). Still divided
/// by the supervised count, not the weight sum, so gradient-scale drift shows up
/// in the logged mean effective weight instead of being normalized away.
fn weighted_token_ce<B: Backend>(
    logits:      Tensor<B, 2>,
    targets:     Tensor<B, 1, Int>,
    row_weights: Option<Tensor<B, 1>>,
    supervised:  usize,
    smoothing:   f32,
) -> Tensor<B, 1> {
    let [rows, _] = logits.dims();
    let mask = targets.clone().equal_elem(PAD_TOKEN as i32).bool_not().float();
    let mask = match row_weights { Some(weights) => mask * weights.detach(), None => mask };
    let log_probs = burn::tensor::activation::log_softmax(logits, 1);
    let nll = log_probs.clone().gather(1, targets.reshape([rows, 1])).reshape([rows]).neg();
    let per_token = if smoothing > 0.0 {
        let uniform = log_probs.mean_dim(1).reshape([rows]).neg();
        nll * (1.0 - smoothing as f64) + uniform * smoothing as f64
    } else {
        nll
    };
    (per_token * mask).cast(burn::tensor::FloatDType::F32).sum() / supervised.max(1) as f64
}

/// Global L2 norm of all parameter gradients before clipping, with one readback.
#[cfg(not(target_arch = "wasm32"))]
fn global_grad_norm<M: AutodiffModule<TrainBackend>>(model: &M, grads: &GradientsParams) -> f32 {
    struct SquaredSum<'a> { grads: &'a GradientsParams, total: Option<Tensor<InnerBackend, 1>> }
    impl burn::module::ModuleVisitor<TrainBackend> for SquaredSum<'_> {
        fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<Tensor<TrainBackend, D>>) {
            if let Some(grad) = self.grads.get::<InnerBackend, D>(param.id) {
                let sum = grad.square().sum();
                self.total = Some(match self.total.take() { Some(total) => total + sum, None => sum });
            }
        }
    }
    let mut visitor = SquaredSum { grads, total: None };
    model.visit(&mut visitor);
    visitor.total.map_or(0.0, |t| t.into_scalar().elem::<f32>().sqrt())
}

/// Negative log-likelihood of each target as a flat [batch * seq] vector. Rows
/// whose target is PAD hold a meaningless value; callers mask them by target.
#[cfg(not(target_arch = "wasm32"))]
fn per_token_nll<B: Backend>(log_probs: Tensor<B, 3>, targets: Tensor<B, 1, Int>) -> Vec<f32> {
    let [batch, seq, _] = log_probs.dims();
    log_probs.gather(2, targets.reshape([batch, seq, 1])).reshape([batch * seq]).neg()
        .cast(burn::tensor::FloatDType::F32).into_data().convert::<f32>().to_vec::<f32>().unwrap()
}

/// Writes up to three held-out sequences as HTML with every token colored by its loss.
#[cfg(not(target_arch = "wasm32"))]
fn write_validation_token_view<B: Backend, M: CausalLm<B>>(
    model: &M, tokenizer: &TokenizerKind, samples: &[crate::brain::samples::Sample],
    max_seq_len: usize, vocab: usize, device: &B::Device, path: &std::path::Path, ceiling: f32,
    raw_code: bool, sep_tokens: &[usize], is_yumon_talk: bool,
) -> Result<()> {
    let idx: Vec<usize> = (0..samples.len().min(3)).collect();
    let (ids, targets, supervised) = build_moe_batch(samples, &idx, sep_tokens, max_seq_len, raw_code);
    if supervised == 0 { return Ok(()); }
    let n = idx.len();
    let (ids, targets, cur_seq_len) = trim_batch(ids, targets, n, max_seq_len);
    let tokens = Tensor::<B, 2, Int>::from_ints(TensorData::new(ids, [n, cur_seq_len]), device);
    let target_t = Tensor::<B, 1, Int>::from_ints(TensorData::new(targets.clone(), [n * cur_seq_len]), device);
    let (logits, depth) = model.logits_with_depth(tokens);
    let log_probs = burn::tensor::activation::log_softmax(logits.cast(burn::tensor::FloatDType::F32), 2);
    debug_assert_eq!(log_probs.dims()[2], vocab);
    let nll = per_token_nll(log_probs, target_t);
    // Row i predicts targets[i], so its tier is the depth spent on that prediction.
    let rows: Vec<Vec<TokenView>> = (0..n).map(|r| {
        let mut prev_token: Option<String> = None;
        (0..cur_seq_len).map(|p| r * cur_seq_len + p).filter(|&i| targets[i] != PAD_TOKEN as i32)
            .map(|i| {
                let raw = tokenizer.decode(&[targets[i] as usize]);
                let needs_space = is_yumon_talk && prev_token.as_deref().is_some_and(|prev| {
                    !matches!(prev, "(" | "[") && !matches!(raw.as_str(), "." | "," | "!" | "?" | ";" | ":" | ")" | "]" | "\n")
                });
                let text = if needs_space { format!(" {raw}") } else { raw.clone() };
                prev_token = Some(raw);
                TokenView { text, loss: nll[i],
                    tier: depth.as_ref().map(|(_, deepest)| deepest[i]) }
            })
            .collect()
    }).collect();
    let tier_names = depth.map(|(names, _)| names);
    write_token_html(path, &format!("Held-out token loss: {}", path.display()), &rows, tier_names.as_deref(), ceiling)
}

/// Per-token loss over the held-out samples, without dropout or gradients.
/// Returns loss, top-3 accuracy, and the number of supervised tokens covered.
#[cfg(not(target_arch = "wasm32"))]
fn causal_validation_loss<B: Backend, M: CausalLm<B>>(
    model:       &M,
    samples:     &[crate::brain::samples::Sample],
    sep_tokens:  &[usize],
    max_seq_len: usize,
    batch_size:  usize,
    vocab:       usize,
    device:      &B::Device,
    raw_code: bool,
    // Receives each batch's flat targets and per-token NLL (adaptive-loss diagnostics).
    mut per_token: Option<&mut dyn FnMut(&[i32], &[f32])>,
) -> Option<(f32, f64, usize)> {
    let mut total = 0.0f64;
    let mut correct = 0usize;
    let mut tokens = 0usize;
    let idx: Vec<usize> = (0..samples.len()).collect();
    for batch_idx in idx.chunks(batch_size) {
        let (seq_ids, targets, supervised) = build_moe_batch(samples, batch_idx, sep_tokens, max_seq_len, raw_code);
        if supervised == 0 { continue; }
        let n = batch_idx.len();
        let (seq_ids, targets, max_seq_len) = trim_batch(seq_ids, targets, n, max_seq_len);
        let tokens_t = Tensor::<B, 2, Int>::from_ints(TensorData::new(seq_ids, [n, max_seq_len]), device);
        let targets_cpu = per_token.is_some().then(|| targets.clone());
        let targets_t = Tensor::<B, 1, Int>::from_ints(TensorData::new(targets, [n * max_seq_len]), device);
        let logits = model.logits(tokens_t).reshape([n * max_seq_len, vocab]);
        if let (Some(callback), Some(targets_cpu)) = (per_token.as_mut(), targets_cpu) {
            let log_probs = burn::tensor::activation::log_softmax(logits.clone().cast(burn::tensor::FloatDType::F32), 1);
            callback(&targets_cpu, &per_token_nll(log_probs.reshape([n, max_seq_len, vocab]), targets_t.clone()));
        }
        correct += top3_correct(logits.clone(), targets_t.clone());
        let loss = masked_token_ce(logits, targets_t, supervised, 0.0)
            .into_data().convert::<f32>().to_vec::<f32>().ok()?[0];
        total += loss as f64 * supervised as f64;
        tokens += supervised;
    }
    (tokens > 0).then(|| ((total / tokens as f64) as f32, correct as f64 / tokens as f64, tokens))
}

// ─── Shared causal-LM training loop (MoE and dense decoder-only) ─────────────

type InnerBackend = <TrainBackend as AutodiffBackend>::InnerBackend;

/// What validation and qualitative eval need from a decoder-only model.
trait CausalLm<B: Backend> {
    fn logits(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3>;
    /// `logits` plus, with adaptive depth, the tier names and the deepest tier
    /// index per row (batch * seq, row-major).
    fn logits_with_depth(&self, tokens: Tensor<B, 2, Int>) -> (Tensor<B, 3>, Option<(Vec<String>, Vec<u8>)>) {
        (self.logits(tokens), None)
    }
    fn generate(&self, tokenizer: &TokenizerKind, prompt: &str, max_tokens: usize, device: &B::Device) -> GenerationResult;
}

/// Progress stamped into a checkpoint's metadata.json.
struct RunProgress<'a> {
    run_cfg:        &'a RunConfig,
    epochs_trained: usize,
    final_loss:     f32,
    val_loss:       Option<f32>,
    batch_size:     usize,
    stage:          TrainingStage,
}

/// Architecture-specific hooks for `train_causal_lm`.
trait CausalLmTrain: AutodiffModule<TrainBackend> + Sized {
    /// Fresh model, or the run directory's checkpoint plus its epochs trained.
    fn init_or_resume(run_cfg: &RunConfig, tokenizer: &TokenizerKind, run_dir: &std::path::Path, device: &<TrainBackend as Backend>::Device) -> Result<(Self, usize)>;
    fn set_stage(&mut self, stage: TrainingStage);
    /// Logits and the weighted auxiliary loss (zero for dense models).
    fn forward_train(&self, tokens: Tensor<TrainBackend, 2, Int>) -> (Tensor<TrainBackend, 3>, Tensor<TrainBackend, 1>);
    /// `forward_train` plus the fraction of tokens reaching each depth tier
    /// (empty for models without adaptive depth).
    fn forward_train_depth(&self, tokens: Tensor<TrainBackend, 2, Int>) -> (Tensor<TrainBackend, 3>, Tensor<TrainBackend, 1>, Vec<f64>) {
        let (logits, aux) = self.forward_train(tokens);
        (logits, aux, Vec::new())
    }
    fn depth_tier_names(&self) -> Vec<String> { Vec::new() }
    fn save_run(&self, dir: &str, tokenizer: &TokenizerKind, progress: &RunProgress) -> Result<()>;
}

fn ensure_same_tokenizer(current: &TokenizerKind, saved: &TokenizerKind) -> Result<()> {
    let json = |t: &TokenizerKind| -> Result<serde_json::Value> {
        match t {
            TokenizerKind::Bpe(t) => Ok(serde_json::from_str(&t.inner.to_string(false).map_err(|e| anyhow::anyhow!("{e}"))?)?),
            _ => anyhow::bail!("causal LM checkpoints require a BPE tokenizer"),
        }
    };
    anyhow::ensure!(json(current)? == json(saved)?, "checkpoint tokenizer differs from training tokenizer");
    Ok(())
}

impl<B: FlashAttention> CausalLm<B> for YumonMoeBrain<B> {
    fn logits(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> { self.forward(tokens) }
    fn logits_with_depth(&self, tokens: Tensor<B, 2, Int>) -> (Tensor<B, 3>, Option<(Vec<String>, Vec<u8>)>) {
        let Some(depth) = self.config.adaptive_depth.as_ref() else { return (self.forward(tokens), None) };
        let names = depth.tiers.iter().map(|t| t.name.clone()).collect();
        let (logits, _, _, deepest) = self.forward_with_depth(tokens);
        (logits, Some((names, deepest)))
    }
    fn generate(&self, tokenizer: &TokenizerKind, prompt: &str, max_tokens: usize, device: &B::Device) -> GenerationResult {
        self.generate_unmasked_parsed(tokenizer, prompt, max_tokens, device)
    }
}

impl CausalLmTrain for YumonMoeBrain<TrainBackend> {
    fn init_or_resume(run_cfg: &RunConfig, tokenizer: &TokenizerKind, run_dir: &std::path::Path, device: &<TrainBackend as Backend>::Device) -> Result<(Self, usize)> {
        let Architecture::Moe { num_experts, top_k } = run_cfg.architecture else { unreachable!() };
        let config = YumonMoeBrainConfig::new(tokenizer.vocab_size(), run_cfg.stages[0].stage)
            .with_embed_dim(run_cfg.embed_dim).with_hidden_units(run_cfg.hidden_units)
            .with_n_layers(run_cfg.n_layers).with_attn_heads(run_cfg.attn_heads)
            .with_ff_dim(run_cfg.ff_dim).with_max_seq_len(run_cfg.max_seq_len)
            .with_dropout_rate(run_cfg.dropout_rate)
            .with_num_experts(num_experts).with_top_k(top_k)
            .with_adaptive_depth(run_cfg.adaptive_depth.clone())
            .with_kv_heads(run_cfg.kv_heads);
        if let Some(kv) = run_cfg.kv_heads {
            println!("Grouped-query attention: {kv} KV heads for {} query heads.", run_cfg.attn_heads);
        }
        if let Some(depth) = &run_cfg.adaptive_depth {
            println!("MoE adaptive depth: {} tiers, budget loss weight {}; Flash Attention, BalancedCheckpointing.",
                depth.tiers.len(), depth.budget_loss_weight);
            for (tier, (total, active)) in depth.tiers.iter().zip(depth.ffn_params(run_cfg.embed_dim)) {
                println!("  {}: {} layers, {} experts {:?}, top-{}, target reach {:.0}%; FFN parameters/layer: {total} total, ~{active} active/token.",
                    tier.name, tier.layers, tier.num_experts(), tier.expert_widths(), tier.top_k, 100.0 * tier.capacity);
            }
        } else {
            println!("MoE: {} experts, top-{}; Flash Attention, BalancedCheckpointing; FFN parameters/layer: {} total, {} active/token (excluding router).",
                num_experts, top_k, 3 * run_cfg.embed_dim * run_cfg.ff_dim * num_experts,
                3 * run_cfg.embed_dim * run_cfg.ff_dim * top_k);
        }
        if !run_dir.join("model.bin").exists() {
            return Ok((config.init(device), 0));
        }
        let (m, checkpoint_tokenizer, saved) = Self::load(run_dir.to_str().unwrap(), device)?;
        anyhow::ensure!(saved.vocab_size == config.vocab_size && saved.embed_dim == config.embed_dim
            && saved.ff_dim == config.ff_dim && saved.n_layers == config.n_layers
            && saved.attn_heads == config.attn_heads && saved.max_seq_len == config.max_seq_len
            && saved.num_experts == num_experts && saved.top_k == top_k
            && saved.adaptive_depth == config.adaptive_depth
            && saved.effective_kv_heads() == config.effective_kv_heads(),
            "MoE checkpoint configuration differs from requested run");
        ensure_same_tokenizer(tokenizer, &checkpoint_tokenizer)?;
        let meta: MoeMetadata = serde_json::from_str(&std::fs::read_to_string(run_dir.join("metadata.json"))?)?;
        Ok((m, meta.epochs_trained))
    }
    fn set_stage(&mut self, stage: TrainingStage) { self.config.0.training_stage = stage; }
    fn forward_train(&self, tokens: Tensor<TrainBackend, 2, Int>) -> (Tensor<TrainBackend, 3>, Tensor<TrainBackend, 1>) {
        let (logits, aux, _expert_counts) = self.forward_with_aux(tokens);
        (logits, aux)
    }
    fn forward_train_depth(&self, tokens: Tensor<TrainBackend, 2, Int>) -> (Tensor<TrainBackend, 3>, Tensor<TrainBackend, 1>, Vec<f64>) {
        let (logits, aux, counts) = self.forward_with_aux(tokens);
        let reach = self.depth_reach(&counts);
        (logits, aux, reach)
    }
    fn depth_tier_names(&self) -> Vec<String> {
        self.config.adaptive_depth.iter().flat_map(|d| d.tiers.iter().map(|t| t.name.clone())).collect()
    }
    fn save_run(&self, dir: &str, tokenizer: &TokenizerKind, p: &RunProgress) -> Result<()> {
        let meta = MoeMetadata {
            num_experts:     self.config.num_experts,
            top_k:           self.config.top_k,
            aux_loss_weight: self.config.aux_loss_weight,
            z_loss_weight:   self.config.z_loss_weight,
            dropout_rate:    self.config.dropout_rate,
            vocab_size:      tokenizer.vocab_size(),
            epochs_trained:  p.epochs_trained,
            final_loss:      p.final_loss,
            val_loss:        p.val_loss,
            batch_size:      p.batch_size,
            training_stage:  p.stage,
            embed_dim:       p.run_cfg.embed_dim,
            hidden_units:    p.run_cfg.hidden_units,
            n_layers:        p.run_cfg.n_layers,
            attn_heads:      p.run_cfg.attn_heads,
            ff_dim:          p.run_cfg.ff_dim,
            max_seq_len:     p.run_cfg.max_seq_len,
            adaptive_depth:  self.config.adaptive_depth.clone(),
            kv_heads:        self.config.kv_heads,
        };
        self.save(dir, tokenizer, &meta)
    }
}

impl<B: FlashAttention> CausalLm<B> for YumonDecBrain<B> {
    fn logits(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> { self.forward(tokens) }
    fn generate(&self, tokenizer: &TokenizerKind, prompt: &str, max_tokens: usize, device: &B::Device) -> GenerationResult {
        self.generate_unmasked_parsed(tokenizer, prompt, max_tokens, device)
    }
}

impl CausalLmTrain for YumonDecBrain<TrainBackend> {
    fn init_or_resume(run_cfg: &RunConfig, tokenizer: &TokenizerKind, run_dir: &std::path::Path, device: &<TrainBackend as Backend>::Device) -> Result<(Self, usize)> {
        let config = YumonDecBrainConfig::new(tokenizer.vocab_size(), run_cfg.stages[0].stage)
            .with_embed_dim(run_cfg.embed_dim).with_hidden_units(run_cfg.hidden_units)
            .with_n_layers(run_cfg.n_layers).with_attn_heads(run_cfg.attn_heads)
            .with_ff_dim(run_cfg.ff_dim).with_max_seq_len(run_cfg.max_seq_len)
            .with_dropout_rate(run_cfg.dropout_rate);
        println!("Decoder-only: {:.1}M parameters, flash attention, BalancedCheckpointing.",
            config.param_count() as f64 / 1e6);
        if !run_dir.join("model.bin").exists() {
            return Ok((config.init(device), 0));
        }
        let (m, checkpoint_tokenizer, saved) = Self::load(run_dir.to_str().unwrap(), device)?;
        anyhow::ensure!(saved.vocab_size == config.vocab_size && saved.embed_dim == config.embed_dim
            && saved.ff_dim == config.ff_dim && saved.n_layers == config.n_layers
            && saved.attn_heads == config.attn_heads && saved.max_seq_len == config.max_seq_len,
            "decoder checkpoint configuration differs from requested run");
        ensure_same_tokenizer(tokenizer, &checkpoint_tokenizer)?;
        let meta: DecMetadata = serde_json::from_str(&std::fs::read_to_string(run_dir.join("metadata.json"))?)?;
        Ok((m, meta.epochs_trained))
    }
    fn set_stage(&mut self, stage: TrainingStage) { self.config.0.training_stage = stage; }
    fn forward_train(&self, tokens: Tensor<TrainBackend, 2, Int>) -> (Tensor<TrainBackend, 3>, Tensor<TrainBackend, 1>) {
        let device = tokens.device();
        (self.forward(tokens), Tensor::zeros([1], &device))
    }
    fn save_run(&self, dir: &str, tokenizer: &TokenizerKind, p: &RunProgress) -> Result<()> {
        let meta = DecMetadata {
            dropout_rate:   self.config.dropout_rate,
            vocab_size:     tokenizer.vocab_size(),
            epochs_trained: p.epochs_trained,
            final_loss:     p.final_loss,
            val_loss:       p.val_loss,
            batch_size:     p.batch_size,
            training_stage: p.stage,
            embed_dim:      p.run_cfg.embed_dim,
            hidden_units:   p.run_cfg.hidden_units,
            n_layers:       p.run_cfg.n_layers,
            attn_heads:     p.run_cfg.attn_heads,
            ff_dim:         p.run_cfg.ff_dim,
            max_seq_len:    p.run_cfg.max_seq_len,
        };
        self.save(dir, tokenizer, &meta)
    }
}

impl<B: Backend> CausalLm<B> for YumonReferenceBrain<B> {
    fn logits(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> { self.forward(tokens) }
    fn generate(&self, tokenizer: &TokenizerKind, prompt: &str, max_tokens: usize, device: &B::Device) -> GenerationResult {
        self.generate_unmasked_parsed(tokenizer, prompt, max_tokens, device)
    }
}

impl CausalLmTrain for YumonReferenceBrain<TrainBackend> {
    fn init_or_resume(run_cfg: &RunConfig, tokenizer: &TokenizerKind, run_dir: &std::path::Path, device: &<TrainBackend as Backend>::Device) -> Result<(Self, usize)> {
        let config = YumonReferenceBrainConfig::new(tokenizer.vocab_size(), run_cfg.stages[0].stage)
            .with_embed_dim(run_cfg.embed_dim).with_n_layers(run_cfg.n_layers)
            .with_attn_heads(run_cfg.attn_heads).with_ff_dim(run_cfg.ff_dim)
            .with_max_seq_len(run_cfg.max_seq_len).with_dropout_rate(run_cfg.dropout_rate);
        println!("Reference: stock burn::nn::transformer, learned positions, LayerNorm; no flash attention or RoPE.");
        if !run_dir.join("model.bin").exists() {
            return Ok((config.init(device), 0));
        }
        let (m, checkpoint_tokenizer, saved) = Self::load(run_dir.to_str().unwrap(), device)?;
        anyhow::ensure!(saved.vocab_size == config.vocab_size && saved.embed_dim == config.embed_dim
            && saved.ff_dim == config.ff_dim && saved.n_layers == config.n_layers
            && saved.attn_heads == config.attn_heads && saved.max_seq_len == config.max_seq_len,
            "reference checkpoint configuration differs from requested run");
        ensure_same_tokenizer(tokenizer, &checkpoint_tokenizer)?;
        let meta: RefMetadata = serde_json::from_str(&std::fs::read_to_string(run_dir.join("metadata.json"))?)?;
        Ok((m, meta.epochs_trained))
    }
    fn set_stage(&mut self, stage: TrainingStage) { self.config.0.training_stage = stage; }
    fn forward_train(&self, tokens: Tensor<TrainBackend, 2, Int>) -> (Tensor<TrainBackend, 3>, Tensor<TrainBackend, 1>) {
        let device = tokens.device();
        (self.forward(tokens), Tensor::zeros([1], &device))
    }
    fn save_run(&self, dir: &str, tokenizer: &TokenizerKind, p: &RunProgress) -> Result<()> {
        self.save(dir, tokenizer, &RefMetadata {
            dropout_rate:   self.config.dropout_rate,
            vocab_size:     tokenizer.vocab_size(),
            epochs_trained: p.epochs_trained,
            final_loss:     p.final_loss,
            val_loss:       p.val_loss,
            batch_size:     p.batch_size,
            training_stage: p.stage,
            embed_dim:      p.run_cfg.embed_dim,
            n_layers:       p.run_cfg.n_layers,
            attn_heads:     p.run_cfg.attn_heads,
            ff_dim:         p.run_cfg.ff_dim,
            max_seq_len:    p.run_cfg.max_seq_len,
        })
    }
}

impl<B: Backend> CausalLm<B> for YumonXLstmBrain<B> {
    fn logits(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> { self.forward(tokens) }
    fn generate(&self, tokenizer: &TokenizerKind, prompt: &str, max_tokens: usize, device: &B::Device) -> GenerationResult {
        self.generate_unmasked_parsed(tokenizer, prompt, max_tokens, device)
    }
}

// Raw-code encoder-decoder pretraining uses a constant BOS encoder input.
// Encoding the full source would leak future target tokens through cross-attention.
impl<B: Backend> CausalLm<B> for YumonBrain<B>
where TrainRuntime: cubecl::Runtime<Device = B::Device> {
    fn logits(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let batch = tokens.dims()[0];
        let encoder = Tensor::<B, 2, Int>::from_ints(
            TensorData::new(vec![BOS_TOKEN as i32; batch], [batch, 1]), &tokens.device());
        self.forward::<TrainRuntime>(encoder, tokens)
    }
    fn generate(&self, tokenizer: &TokenizerKind, prompt: &str, max_tokens: usize, device: &B::Device) -> GenerationResult {
        let mut tokens = vec![BOS_TOKEN]; tokens.extend(tokenizer.encode(prompt));
        let mut output = Vec::new();
        for _ in 0..max_tokens {
            let window = &tokens[tokens.len().saturating_sub(self.config.max_seq_len)..];
            let position = window.len()-1;
            let ids = window.iter().map(|&t| t as i32).collect::<Vec<_>>();
            let input = Tensor::<B, 2, Int>::from_ints(TensorData::new(ids, [1, window.len()]), device);
            let next = self.logits(input).slice([0..1, position..position+1, 0..self.config.vocab_size])
                .argmax(2).reshape([1]).into_data().convert::<i32>().to_vec::<i32>().unwrap()[0] as usize;
            if next <= EOS_TOKEN || next == 3 { break; }
            output.push(next); tokens.push(next);
        }
        let raw_output = tokenizer.decode(&output);
        GenerationResult { reply: raw_output.clone(), raw_output, action: crate::brain::samples::Action::Sit,
            motion_dir: crate::brain::samples::CardinalDir::None, parsed_emotion: String::new(), fsm_state: 0, allowed_count: None }
    }
}

// Both legacy architectures expose the same capacity/checkpoint fields. Keep
// their raw-code data path in the shared trainer, not in their Pet-specific loops.
macro_rules! code_model_adapter {
    ($model:ident, $config:ident, $metadata:ident) => {
        impl CausalLmTrain for $model<TrainBackend> {
            fn init_or_resume(run: &RunConfig, tokenizer: &TokenizerKind, directory: &std::path::Path,
                device: &<TrainBackend as Backend>::Device) -> Result<(Self, usize)> {
                let config = $config { vocab_size: tokenizer.vocab_size(), embed_dim: run.embed_dim,
                    hidden_units: run.hidden_units, n_layers: run.n_layers, attn_heads: run.attn_heads,
                    ff_dim: run.ff_dim, max_seq_len: run.max_seq_len, dropout_rate: run.dropout_rate,
                    training_stage: run.stages[0].stage };
                if !directory.join("model.bin").exists() { return Ok((config.init(device), 0)); }
                let (model, saved_tokenizer, saved) = Self::load(directory.to_str().unwrap(), device)?;
                anyhow::ensure!(saved.vocab_size == config.vocab_size && saved.embed_dim == config.embed_dim
                    && saved.hidden_units == config.hidden_units && saved.n_layers == config.n_layers
                    && saved.attn_heads == config.attn_heads && saved.ff_dim == config.ff_dim
                    && saved.max_seq_len == config.max_seq_len, "checkpoint capacity differs from requested code run");
                ensure_same_tokenizer(tokenizer, &saved_tokenizer)?;
                let metadata: $metadata = serde_json::from_slice(&std::fs::read(directory.join("metadata.json"))?)?;
                Ok((model, metadata.epochs_trained))
            }
            fn set_stage(&mut self, stage: TrainingStage) { self.config.0.training_stage = stage; }
            fn forward_train(&self, tokens: Tensor<TrainBackend, 2, Int>) -> (Tensor<TrainBackend, 3>, Tensor<TrainBackend, 1>) {
                let device = tokens.device(); (self.logits(tokens), Tensor::zeros([1], &device))
            }
            fn save_run(&self, directory: &str, tokenizer: &TokenizerKind, p: &RunProgress) -> Result<()> {
                self.save(directory, tokenizer, &$metadata { vocab_size: tokenizer.vocab_size(), epochs_trained: p.epochs_trained,
                    final_loss: p.final_loss, batch_size: p.batch_size, training_stage: p.stage, embed_dim: p.run_cfg.embed_dim,
                    hidden_units: p.run_cfg.hidden_units, n_layers: p.run_cfg.n_layers, attn_heads: p.run_cfg.attn_heads,
                    ff_dim: p.run_cfg.ff_dim, max_seq_len: p.run_cfg.max_seq_len })
            }
        }
    };
}
code_model_adapter!(YumonXLstmBrain, YumonXLstmBrainConfig, XLstmMetadata);
code_model_adapter!(YumonBrain, YumonBrainConfig, BrainMetadata);

/// Qualitative eval: every prompt through the inference model.
#[cfg(not(target_arch = "wasm32"))]
fn run_eval_prompts<M: CausalLm<InnerBackend>>(
    model:         &M,
    tokenizer:     &TokenizerKind,
    prompts:       &[String],
    stage:         TrainingStage,
    max_seq_len:   usize,
    device:        &<TrainBackend as Backend>::Device,
    is_yumon_talk: bool,
) -> Vec<EvalLogEntry> {
    prompts.iter().map(|p| {
        evaluate_prompt(p, stage, is_yumon_talk, |prompt| {
            let result = model.generate(tokenizer, prompt, max_seq_len, device);
            if stage == TrainingStage::Structured { result.reply } else { result.raw_output }
        })
    }).collect()
}

/// [prompt][separator][reply] causal-LM training for one run: stages, epochs,
/// linear LR decay, held-out validation, periodic checkpoints and eval prompts.
#[cfg(not(target_arch = "wasm32"))]
fn train_causal_lm<M>(
    run_cfg:       &RunConfig,
    run_dir:       &std::path::Path,
    tokenizer:     &TokenizerKind,
    keyword_index: &HashMap<String, Vec<usize>>,
    prompts:       &[String],
    device:        &<TrainBackend as Backend>::Device,
    mut prepared_code: Option<crate::brain::code_curriculum::PreparedCode>,
) -> Result<()>
where
    M: CausalLmTrain,
    M::InnerModule: CausalLm<InnerBackend>,
{
    let is_prepared = prepared_code.is_some();
    let raw_code = prepared_code.as_ref().map_or(false, |p| p.objective == crate::brain::sample_cache::CacheObjective::RawCode);

    println!("raw_code: {:?}", raw_code);

    let is_yumon_talk = is_yumon_talk_run(&run_cfg.name, tokenizer, raw_code);
    let run_dir_str = run_dir.to_str().unwrap();
    let (mut model, mut epochs_already_done) = M::init_or_resume(run_cfg, tokenizer, run_dir, device)?;
    if is_prepared && epochs_already_done >= run_cfg.stages[0].epochs {
        println!("Epoch budget already completed: {epochs_already_done}");
        return Ok(());
    }

    'stage_loop: for (stage_idx, stage_cfg) in run_cfg.stages.iter().enumerate() {
        model.set_stage(stage_cfg.stage);
        println!("\n🔨 Stage {}: {:?}", stage_idx + 1, stage_cfg.stage);

        let epoch_indices = prepared_code.as_ref().map(|p| (0..stage_cfg.epochs).map(|e| p.epoch_indices(e)).collect::<Vec<_>>());
        let epoch_seed = prepared_code.as_ref().map(|p| p.seed);
        let batch_plan = prepared_code.as_ref().map(|p| p.batch_plan(stage_cfg.epochs, stage_cfg.batch_size));
        let (training_samples, val_samples) = if let Some(prepared) = prepared_code.take() {
            (prepared.training, prepared.validation)
        } else {
            let mut samples = load_stage_data(stage_cfg.stage, tokenizer, keyword_index, run_cfg.max_seq_len)?;
            anyhow::ensure!(samples.len() > 1, "Not enough training samples for stage");
            let count = (samples.len() / 100).clamp(1, MAX_VAL_SAMPLES);
            let validation = samples.split_off(samples.len() - count);
            (samples, validation)
        };
        println!("Training samples: {}, validation samples: {}", training_samples.len(), val_samples.len());

        for (i, sample) in training_samples.iter().enumerate() {
            if i >= 12 { break; }
            println!("INPUT:  {:?}", tokenizer.decode(&sample.input_ids));
            println!("TARGET: {:?}", tokenizer.decode(&sample.target_labels));
            println!("input_len:     {}", sample.input_ids.iter().filter(|&&t| t != PAD_TOKEN).count());
            println!("target_active: {}", sample.target_labels.iter().filter(|&&t| t != PAD_TOKEN).count());
        }

        let mut optimizer = AdamWConfig::new()
            .with_epsilon(stage_cfg.epsilon)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
            .with_weight_decay(stage_cfg.weight_decay)
            .init();

        let mut rng = rand::thread_rng();
        use std::io::{stdout, IsTerminal};
        let mut terminal = if stdout().is_terminal() {
            let backend = CrosstermBackend::new(stdout());
            Some(Terminal::with_options(
                backend,
                TerminalOptions { viewport: Viewport::Inline(46) },
            )?)
        } else {
            None
        };

        anyhow::ensure!(stage_cfg.batch_size > 0, "batch size must be positive");
        let total_batches = training_samples.len().div_ceil(stage_cfg.batch_size);
        let mut state = TrainingState {
            loss_history: vec![],
            avg_loss_history: vec![],
            current_loss: 0.0,
            avg_loss: 0.0,
            epoch: 0,
            total_epochs: stage_cfg.epochs,
            batch: 0,
            total_batches: total_batches,
            current_lr: stage_cfg.first_lr,
            lr_history: vec![],
            global_step: 0,
            entropy: 0.0,
            entropy_history: vec![],
            top3_accuracy: None,
            avg_top3_accuracy: None,
            top3_history: vec![],
            last_reply: String::new()
        };

        let mut final_loss = 0.0f32;
        let inference_log_path = format!("{}/{}_inference_log.txt", run_dir_str, run_cfg.name);
        let chart_path = format!("{}/{}_stage_{}.png", run_dir_str, run_cfg.name, stage_idx + 1);
        let mut prev_epoch_loss: Option<f32> = None;
        let mut run_should_stop = false;
        let mut val_loss: Option<f32>;
        let vocab = tokenizer.vocab_size();
        let mut loss_stats = TokenLossStats::new(run_cfg.max_seq_len, vocab);
        let heatmap_dir = std::path::Path::new(run_dir_str).join("heatmaps");
        let heatmap_ceiling = (vocab as f32).ln();
        let log_every = if run_cfg.overfit { 10 } else { 50 };
        let total_steps = batch_plan.as_ref().map_or(stage_cfg.epochs * training_samples.len().div_ceil(stage_cfg.batch_size), |p| p.iter().sum());

        // Adaptive learning-value weighting (raw code only); state resumes with the checkpoint.
        use crate::brain::adaptive_loss::{AdaptiveLoss, CONTEXT_LABELS, ContextLabeler, FrequencyBuckets, GroupLoss};
        let mut adaptive = match run_cfg.adaptive_loss.as_ref().filter(|_| raw_code) {
            Some(cfg) => Some(AdaptiveLoss::load_or_new(run_dir, cfg, vocab)?),
            None => None,
        };
        let frequency_buckets = adaptive.as_ref()
            .map(|_| FrequencyBuckets::from_counts(&FrequencyBuckets::count_targets(&training_samples, vocab)));
        // Every adaptive mode logs CE by occurrence context, so arms compare on it.
        let labeler = adaptive.as_ref().map(|_| ContextLabeler::from_tokenizer(tokenizer, vocab));
        if let Some(a) = adaptive.as_ref() {
            let c = &a.config;
            println!("Adaptive loss: mode {}, lambda {}, warm-up {} steps, ramp {} steps, window {} steps, {} completed windows restored.",
                c.effective_mode().tag(), c.lambda_at(total_steps, total_steps), (c.warmup_fraction * total_steps as f64) as usize,
                (c.ramp_fraction * total_steps as f64) as usize, c.statistics_interval_steps, a.windows);
        }
        // Per-window sums for the diagnostics log.
        #[derive(Default)]
        struct Window { ce: f64, weighted: f64, weight: f64, tokens: usize, steps: usize, overhead: std::time::Duration, by_context: GroupLoss }
        let mut window = Window::default();
        let mut window_start = std::time::Instant::now();
        // Adaptive depth: summed per-tier reach fractions and steps since the last log line.
        let depth_names = model.depth_tier_names();
        let mut depth_window: (Vec<f64>, usize) = (Vec::new(), 0);

        // Decoder-only style: [prompt][separator][reply] packed into one causal sequence
        let sep_text = if stage_cfg.stage == TrainingStage::Structured { "\n---\n" } else { " " };
        let sep_tokens = if raw_code { Vec::new() } else { tokenizer.encode(sep_text) };
        anyhow::ensure!(sep_tokens.len() + 2 <= run_cfg.max_seq_len, "Sequence too short for separator and reply");
        // Held-out loss; with adaptive loss also unweighted CE by training frequency
        // and by occurrence context, appended to adaptive_loss.jsonl.
        let validate = |model: &M::InnerModule, epoch: usize, step: usize, print: bool| -> Result<Option<(f32, f64, usize)>> {
            let (Some(buckets), Some(labeler)) = (frequency_buckets.as_ref(), labeler.as_ref()) else {
                return Ok(causal_validation_loss(model, &val_samples, &sep_tokens, run_cfg.max_seq_len,
                    stage_cfg.batch_size, vocab, device, raw_code, None));
            };
            let (mut by_frequency, mut by_context) = (GroupLoss::default(), GroupLoss::default());
            let mut collect = |targets: &[i32], nll: &[f32]| {
                by_frequency.add_by_frequency(buckets, targets, nll);
                by_context.add_by_label(&labeler.label(targets, run_cfg.max_seq_len), nll, None);
            };
            let val = causal_validation_loss(model, &val_samples, &sep_tokens, run_cfg.max_seq_len,
                stage_cfg.batch_size, vocab, device, raw_code, Some(&mut collect));
            if let Some((loss, top3, tokens)) = val {
                let line = serde_json::json!({"validation": {"epoch": epoch, "step": step,
                    "ce": loss, "top3": top3, "tokens": tokens,
                    "ce_by_train_frequency": by_frequency.to_json(&FrequencyBuckets::LABELS, false),
                    "ce_by_context": by_context.to_json(&CONTEXT_LABELS, false)}});
                // if print { println!("adaptive validation: {line}"); }
                crate::brain::adaptive_loss::append_log(run_dir, &line)?;
            }
            Ok(val)
        };

        let start_epoch = if is_prepared { epochs_already_done } else { 0 };
        let epoch_offset = if is_prepared { 0 } else { epochs_already_done };
        'epoch_loop: for epoch in start_epoch..stage_cfg.epochs {
            state.epoch = epoch + 1;
            let mut idx: Vec<usize> = epoch_indices.as_ref()
                .map_or_else(|| (0..training_samples.len()).collect(), |all| all[epoch].clone());
            let epoch_samples = idx.len();
            if let Some(seed) = epoch_seed {
                idx.shuffle(&mut StdRng::seed_from_u64(seed.wrapping_add(epoch as u64)));
            } else { idx.shuffle(&mut rng); }
            let num_batches = idx.len().div_ceil(stage_cfg.batch_size);
            state.total_batches = num_batches;
            
            let mut epoch_loss = 0.0f32;
            let mut processed_batches = 0usize;
            let mut epoch_top3_correct = 0usize;
            let mut epoch_supervised = 0usize;
            let step_base: usize = batch_plan.as_ref().map_or(epoch * num_batches, |p| p[..epoch].iter().sum());

            for batch_num in 0..num_batches {
                let step = step_base + batch_num;
                let current_lr = {
                    let total_steps = batch_plan.as_ref().map_or(stage_cfg.epochs * num_batches, |p| p.iter().sum());
                    let t = step as f64 / total_steps as f64;
                    stage_cfg.first_lr * (1.0 - t) + stage_cfg.last_lr * t
                };

                let batch_start = batch_num * stage_cfg.batch_size;
                let batch_end = (batch_start + stage_cfg.batch_size).min(idx.len());
                let batch_idx = &idx[batch_start..batch_end];
                let current_batch_size = batch_idx.len();
                if current_batch_size == 0 { continue; }

                let (all_seq_ids, all_lang_targets, supervised) = build_moe_batch(&training_samples, batch_idx, &sep_tokens, run_cfg.max_seq_len, raw_code);
                let (all_seq_ids, all_lang_targets, seq_len) = trim_batch(all_seq_ids, all_lang_targets, current_batch_size, run_cfg.max_seq_len);

                if supervised == 0 { continue; }
                let targets_cpu = all_lang_targets.clone();
                let lang_target_t = Tensor::<TrainBackend, 1, Int>::from_ints(TensorData::new(all_lang_targets, [current_batch_size * seq_len]), device);
                let tokens_t = Tensor::<TrainBackend, 2, Int>::from_ints(TensorData::new(all_seq_ids, [current_batch_size, seq_len]), device);

                let (token_logits, aux_loss, reach) = model.forward_train_depth(tokens_t.clone());
                if !reach.is_empty() {
                    depth_window.0.resize(reach.len(), 0.0);
                    depth_window.0.iter_mut().zip(&reach).for_each(|(sum, r)| *sum += r);
                    depth_window.1 += 1;
                }
                let correct = top3_correct(
                    token_logits.clone().inner().reshape([current_batch_size * seq_len, vocab]),
                    lang_target_t.clone().inner(),
                );
                epoch_top3_correct += correct;
                epoch_supervised += supervised;
                let top3 = correct as f64 / supervised as f64;

                // Entropy, on detached logits so none of its [batch, seq, vocab]
                // intermediates join the autodiff graph.
                let log_probs = burn::tensor::activation::log_softmax(token_logits.clone().detach(), 2);
                // Per-token loss of this batch (0 at PAD) for the heatmaps; no gradient.
                let token_nll = per_token_nll(log_probs.clone(), lang_target_t.clone());
                loss_stats.add(&targets_cpu, &token_nll, seq_len);
                let probs = log_probs.clone().exp();
                let token_entropy = (probs * log_probs).sum_dim(2).neg().squeeze_dim::<2>(2);
                let non_pad_mask = tokens_t.clone().equal_elem(PAD_TOKEN as u32).bool_not().float();
                let entropy_val: f32 = (token_entropy * non_pad_mask.clone()).cast(burn::tensor::FloatDType::F32).sum()
                    .div(non_pad_mask.cast(burn::tensor::FloatDType::F32).sum())
                    .into_data().convert::<f32>().to_vec::<f32>().unwrap()[0];

                // Loss
                let logits_2d = token_logits.reshape([current_batch_size * seq_len, vocab]);
                // Weights come only from completed windows; lambda 0 (warm-up or
                // disabled) keeps the plain masked CE path exactly.
                let bookkeeping = std::time::Instant::now();
                let lambda = adaptive.as_ref().map_or(0.0, |a| a.config.lambda_at(step, total_steps));
                let labels = labeler.as_ref().map(|l| l.label(&targets_cpu, seq_len));
                let keys = adaptive.as_ref().map(|a| a.keys(&targets_cpu, labels.as_deref()));
                let row_weights = adaptive.as_ref().zip(keys.as_ref()).filter(|_| lambda > 0.0)
                    .map(|(a, keys)| a.row_weights(keys, lambda));
                if let Some(labels) = &labels {
                    window.by_context.add_by_label(labels, &token_nll, row_weights.as_deref());
                }
                let weight_sum = row_weights.as_ref().map_or(supervised as f64, |w| w.iter().map(|&x| x as f64).sum());
                let rows = current_batch_size * seq_len;
                let row_weights_t = row_weights.map(|w| Tensor::<TrainBackend, 1>::from_floats(TensorData::new(w, [rows]), device));
                window.overhead += bookkeeping.elapsed();
                let lang_loss = weighted_token_ce(logits_2d, lang_target_t, row_weights_t, supervised, stage_cfg.smoothing);

                let total_loss = lang_loss.clone() + aux_loss.cast(burn::tensor::FloatDType::F32);
                let grads = GradientsParams::from_grads(total_loss.backward(), &model);
                let grad_norm = adaptive.as_ref().filter(|a| a.closes_window_next_step())
                    .map(|_| global_grad_norm(&model, &grads));
                model = optimizer.step(current_lr, model, grads);

                let lang_val: f32 = lang_loss.inner().to_data().convert::<f32>().to_vec::<f32>().unwrap()[0];
                // Charts, epoch loss and metadata stay on unweighted CE for comparability.
                let ce_sum: f64 = targets_cpu.iter().zip(&token_nll)
                    .filter(|(t, _)| **t != PAD_TOKEN as i32).map(|(_, &l)| l as f64).sum();
                let loss_val = if lambda > 0.0 { (ce_sum / supervised as f64) as f32 } else { lang_val };
                if let Some(a) = adaptive.as_mut() {
                    let bookkeeping = std::time::Instant::now();
                    window.ce += ce_sum;
                    window.weighted += lang_val as f64 * supervised as f64;
                    window.weight += weight_sum;
                    window.tokens += supervised;
                    window.steps += 1;
                    let closed = a.observe(keys.as_ref().unwrap(), &token_nll);
                    window.overhead += bookkeeping.elapsed();
                    if closed {
                        let elapsed = window_start.elapsed().as_secs_f64();
                        let report = a.report(frequency_buckets.as_ref().unwrap());
                        let tokens = window.tokens.max(1) as f64;
                        let line = serde_json::json!({
                            "step": step + 1, "epoch": epoch + 1, "lambda": lambda,
                            "standard_ce": window.ce / tokens, "adaptive_lm_loss": window.weighted / tokens,
                            "mean_effective_weight": window.weight / tokens, "grad_norm": grad_norm,
                            "tokens_per_sec": window.tokens as f64 / elapsed,
                            "step_ms": 1e3 * elapsed / window.steps.max(1) as f64,
                            "adaptive_cpu_ms_per_step": 1e3 * window.overhead.as_secs_f64() / window.steps.max(1) as f64,
                            // Unweighted CE and mean applied weight a (1 when lambda is 0) per context.
                            "by_context": window.by_context.to_json(&CONTEXT_LABELS, true),
                            "stats": report,
                        });
                        if terminal.is_none() {
                            println!("adaptive window {} lambda {lambda:.3} ce {:.4} weighted {:.4} mean_w {:.3} w p05/p50/p95 {:.2}/{:.2}/{:.2} grad_norm {:.3}",
                                a.windows, window.ce / tokens, window.weighted / tokens, window.weight / tokens,
                                report.weight_p05, report.weight_p50, report.weight_p95, grad_norm.unwrap_or(f32::NAN));
                            let context: Vec<String> = CONTEXT_LABELS.iter().zip(&window.by_context.0)
                                .map(|(label, &(ce, w, n))| format!("{label} {:.2}@{:.2}", ce / n.max(1) as f64, w / n.max(1) as f64)).collect();
                            println!("  ce@weight by context: {}", context.join("  "));
                        }
                        if let Err(e) = crate::brain::adaptive_loss::append_log(run_dir, &line) {
                            eprintln!("⚠️  Failed to append adaptive loss log: {e}");
                        }
                        window = Window::default();
                        window_start = std::time::Instant::now();
                    }
                }
                epoch_loss += loss_val;
                processed_batches += 1;

                state.entropy = entropy_val;
                state.top3_accuracy = Some(top3);
                state.avg_top3_accuracy = Some(epoch_top3_correct as f64 / epoch_supervised as f64);
                state.current_loss = loss_val;
                state.avg_loss = epoch_loss / processed_batches as f32;
                state.batch = batch_num + 1;
                state.current_lr = current_lr;
                state.global_step += 1;
                state.loss_history.push((state.global_step as f64, loss_val as f64));
                state.avg_loss_history.push((state.global_step as f64, state.avg_loss as f64));
                state.entropy_history.push((state.global_step as f64, entropy_val as f64));
                state.lr_history.push((state.global_step as f64, current_lr));
                state.top3_history.push((state.global_step as f64, top3));

                if run_cfg.heatmap_every > 0 && (state.global_step == 1 || state.global_step % run_cfg.heatmap_every == 0) {
                    let step = state.global_step;
                    let snapshot = (|| -> Result<()> {
                        std::fs::create_dir_all(&heatmap_dir)?;
                        save_batch_heatmap(&heatmap_dir.join(format!("batch_{step:06}.png")), &targets_cpu,
                            &token_nll, current_batch_size, seq_len, heatmap_ceiling)?;
                        loss_stats.save(&heatmap_dir, step)?;
                        write_validation_token_view(&model.valid(), tokenizer, &val_samples, run_cfg.max_seq_len,
                            vocab, device, &heatmap_dir.join(format!("tokens_{step:06}.html")), heatmap_ceiling,
                            raw_code, &sep_tokens, is_yumon_talk)
                    })();
                    if let Err(e) = snapshot { eprintln!("⚠️  Failed to save loss heatmaps: {e}"); }
                    loss_stats.reset();
                }

                if let Some(term) = terminal.as_mut() {
                    term.draw(|frame| render(frame, &state))?;
                } else if state.global_step % log_every == 0 || batch_num + 1 == num_batches {
                    println!(
                        "epoch {}/{} batch {}/{} loss {:.4} avg {:.4} lr {:.2e} entropy {:.4} top3 {:.2}% avg_top3 {:.2}%",
                        state.epoch, state.total_epochs, state.batch, state.total_batches,
                        state.current_loss, state.avg_loss, state.current_lr, state.entropy,
                        100.0 * top3, 100.0 * state.avg_top3_accuracy.unwrap(),
                    );
                }
                if depth_window.1 > 0 && (state.global_step % log_every == 0 || batch_num + 1 == num_batches) {
                    let mean: Vec<f64> = depth_window.0.iter().map(|s| s / depth_window.1 as f64).collect();
                    if terminal.is_none() {
                        let parts: Vec<String> = depth_names.iter().zip(&mean)
                            .map(|(name, r)| format!("{name} {:.1}%", 100.0 * r)).collect();
                        println!("  depth reach: {}", parts.join("  "));
                    }
                    let line = serde_json::json!({"step": step + 1, "epoch": epoch + 1, "steps": depth_window.1,
                        "reach": depth_names.iter().zip(mean).map(|(n, r)| (n.clone(), serde_json::json!(r)))
                            .collect::<serde_json::Map<_, _>>()});
                    let appended = std::fs::OpenOptions::new().create(true).append(true)
                        .open(run_dir.join("adaptive_depth.jsonl"))
                        .and_then(|mut f| std::io::Write::write_all(&mut f, format!("{line}\n").as_bytes()));
                    if let Err(e) = appended { eprintln!("⚠️  Failed to append adaptive depth log: {e}"); }
                    depth_window = (Vec::new(), 0);
                }

                // Periodic save and inference every 500 batches
                if (batch_num + 1) % 500 == 0 {
                    let inference_model = model.valid();
                    let val = validate(&inference_model, epoch + 1, step + 1, terminal.is_none())?;
                    if let Some(a) = adaptive.as_ref() { a.save(run_dir)?; }
                    val_loss = val.map(|(loss, _, _)| loss);
                    model.save_run(run_dir_str, tokenizer, &RunProgress {
                        run_cfg,
                        epochs_trained: epoch_offset + epoch,
                        final_loss:     epoch_loss / processed_batches as f32,
                        val_loss,
                        batch_size:     stage_cfg.batch_size,
                        stage:          stage_cfg.stage,
                    })?;

                    let entries = run_eval_prompts(&inference_model, tokenizer, prompts, stage_cfg.stage, run_cfg.max_seq_len, device, is_yumon_talk);
                    state.last_reply = entries.first().map(|entry| {
                        if let Some(yr) = &entry.yumon_reply {
                            format!("{} [{}]", entry.reply, yr)
                        } else {
                            entry.reply.clone()
                        }
                    }).unwrap_or_default();
                    if let Err(e) = append_inference_log(&inference_log_path, &run_cfg.name, stage_idx, stage_cfg.stage, state.epoch, state.total_epochs, state.batch, state.total_batches, state.avg_loss, &entries) {
                        eprintln!("⚠️  Failed to append inference log: {}", e);
                    }
                    if let Some((loss, top3, tokens)) = val {
                        let line = format!("val_loss {loss:.4} val_top3 {:.2}% over {tokens} held-out tokens", top3 * 100.0);
                        // println!("{line}");
                        if let Err(e) = append_log_line(&inference_log_path, &line) {
                            eprintln!("⚠️  Failed to append inference log: {}", e);
                        }
                    }
                    if let Err(e) = state.save_chart_image(&chart_path) {
                        eprintln!("⚠️  Failed to save chart image: {}", e);
                    }
                }

                // Loss Threshold Exit
                if state.avg_loss < stage_cfg.loss_threshold {
                    println!("\n🎯 Loss Threshold Reached: {:.4} < {:.4}. Ending Stage.", state.avg_loss, stage_cfg.loss_threshold);
                    break;
                }
            }
            anyhow::ensure!(processed_batches > 0, "epoch contains no supervised tokens");
            final_loss = epoch_loss / processed_batches as f32;
            let inference_model = model.valid();
            let val = validate(&inference_model, epoch + 1, step_base + num_batches, terminal.is_none())?;
            val_loss = val.map(|(loss, _, _)| loss);

            // Early stop on held-out loss when available.
            let stop_loss = val_loss.unwrap_or(final_loss);
            if let Some(prev) = prev_epoch_loss {
                if !is_prepared && prev - stop_loss < MIN_EPOCH_LOSS_DROP {
                    println!(
                        "\n⏹️  Epoch loss drop {:.4} < {:.4} (prev {:.4} -> {:.4}). Finishing run early.",
                        prev - stop_loss, MIN_EPOCH_LOSS_DROP, prev, stop_loss,
                    );
                    run_should_stop = true;
                }
            }
            prev_epoch_loss = Some(stop_loss);

            // Overfit runs have many tiny epochs: checkpoint, generate and chart rarely.
            let heavy = !run_cfg.overfit || (epoch + 1) % 25 == 0 || epoch + 1 == stage_cfg.epochs;
            if heavy {
                if let Some(a) = adaptive.as_ref() { a.save(run_dir)?; }
                model.save_run(run_dir_str, tokenizer, &RunProgress {
                    run_cfg,
                    epochs_trained: epoch_offset + epoch + 1,
                    final_loss,
                    val_loss,
                    batch_size: stage_cfg.batch_size,
                    stage:      stage_cfg.stage,
                })?;

                let entries = run_eval_prompts(&inference_model, tokenizer, prompts, stage_cfg.stage, run_cfg.max_seq_len, device, is_yumon_talk);
                state.last_reply = entries.first().map(|entry| {
                    if let Some(yr) = &entry.yumon_reply {
                        format!("{} [{}]", entry.reply, yr)
                    } else {
                        entry.reply.clone()
                    }
                }).unwrap_or_default();
                if let Err(e) = append_inference_log(&inference_log_path, &run_cfg.name, stage_idx, stage_cfg.stage, state.epoch, state.total_epochs, state.batch, state.total_batches, state.avg_loss, &entries) {
                    eprintln!("⚠️  Failed to append inference log: {}", e);
                }
            }
            if let Some((loss, top3, tokens)) = val {
                let line = format!("val_loss {loss:.4} val_top3 {:.2}% over {tokens} held-out tokens (epoch end)", top3 * 100.0);
                // if heavy || !run_cfg.overfit { println!("{line}"); }
                if heavy {
                    if let Err(e) = append_log_line(&inference_log_path, &line) {
                        eprintln!("⚠️  Failed to append inference log: {}", e);
                    }
                }
            }
            if heavy {
                if let Err(e) = state.save_chart_image(&chart_path) {
                    eprintln!("⚠️  Failed to save chart image: {}", e);
                }
            }

            if run_should_stop || final_loss < stage_cfg.loss_threshold { break 'epoch_loop; }
        }
        if is_prepared { epochs_already_done = state.epoch; } else { epochs_already_done += state.epoch; }
        if let Some(term) = terminal.as_mut() {
            term.clear()?;
        }

        if let Err(e) = state.save_chart_image(&chart_path) {
            eprintln!("⚠️  Failed to save chart image: {}", e);
        } else {
            println!("📊 Chart saved to {}", chart_path);
        }

        println!("✅ Stage complete. Final loss: {:.4}", final_loss);
        if run_cfg.overfit {
            let tail = state.loss_history.len().saturating_sub(10);
            let recent = &state.loss_history[tail..];
            let last10 = recent.iter().map(|p| p.1).sum::<f64>() / recent.len().max(1) as f64;
            let verdict = if last10 < 0.1 { "memorized (expected < 0.1)" }
                else if last10 < 1.0 { "learning but not memorized yet" }
                else { "STALLED above 1: check label shifting, loss masking, padding" };
            println!("Overfit result: mean of last {} batch losses {last10:.4} - {verdict}", recent.len());
        }

        if run_should_stop {
            println!("⏹️  Run {} finished early, skipping remaining stages.", run_cfg.name);
            break 'stage_loop;
        }
    }
    Ok(())
}

/// Programmatically builds the full grid of RunConfigs to try, rather than a
/// hand-picked list. Most combinations won't converge and will be cut short
/// by MIN_EPOCH_LOSS_DROP, but this way we actually cover the space instead
/// of only the sizes/depths someone happened to type in by hand.
fn generate_run_configs(batch_size_option: usize, architecture: Architecture) -> Vec<RunConfig> {
    // ~150k training records at these sizes puts every combo below well under
    // a 20x-tokens-per-active-param ratio even for a single epoch, so capacity
    // here is gated by iGPU compute (see TrainBrain's batch_size CLI comment -
    // batch 256 was already too large at 128 hidden on this hardware), not by
    // data volume. Reduce this matrix, don't reduce the dataset, if a sweep
    // runs too slow.
    // Pet H100 starting shape: 1024 wide x 24 layers x 32 heads, context 256.
    // Four experts/top-1 gives ~1.34B total / ~436M active at vocab 16384.
    // Both causal architectures use flash attention and balanced checkpointing.
    // See configs/runpod-h100.md; peak memory must be measured on the pod.
    let sizes:       [usize; 1] = [
        // 32,
        // 64,
        // 128,
        256, // matching configs/yumon-talk-moe.json
        // 512, 
        // 1024,
    ];
    let layer_counts: [usize; 1] = [
        // 1,
        // 2,
        // 4,
        // 8,
        12,
        // 16,
        // 24,
        // 32,
    ];
    let head_counts:  [usize; 1] = [
        // 1,
        // 2,
        // 4,
        // 8,
        16,
        // 32,
        // 64,
    ];
    let seq_lens:     [usize; 1] = [
        512,
    ];
    let batch_sizes:     [usize; 1] = [
        if matches!(architecture, Architecture::Moe { .. } | Architecture::DecoderOnly) { batch_size_option } else { 4 }
    ];
    // Honor the CLI expert count and routing choice.
    let architectures = vec![architecture];
    let stages = [
        TrainingStage::Language,
        // TrainingStage::Structured
    ];

    let mut runs = Vec::new();

    for &size in &sizes {
        for &n_layers in &layer_counts {
            for &attn_heads in &head_counts {
                if size % attn_heads != 0 { continue; }
                for &max_seq_len in &seq_lens {
                    for &batch_size in &batch_sizes {
                        for &architecture in &architectures {
                            for &stage in &stages {
                                let (first_lr, last_lr) = match architecture {
                                    Architecture::Moe { .. } => (3e-4, 3e-5),
                                    Architecture::DecoderOnly    => (2e-4, 2e-5),
                                    Architecture::EncoderDecoder => (1e-3, 1e-4),
                                    Architecture::XLstm          => (1e-3, 1e-4),
                                    Architecture::Reference      => (3e-4, 3e-5),
                                };
                                let arch_tag = match architecture {
                                    Architecture::Moe { num_experts, top_k } => format!("Moe_e{num_experts}_k{top_k}"),
                                    Architecture::DecoderOnly    => "DecoderOnly".to_string(),
                                    Architecture::EncoderDecoder => "EncoderDecoder".to_string(),
                                    Architecture::XLstm          => "XLstm".to_string(),
                                    Architecture::Reference      => "Reference".to_string(),
                                };
                                let stage_tag = match stage {
                                    TrainingStage::Language   => "Language",
                                    TrainingStage::Structured => "Structured",
                                };
                                let name = format!(
                                    "{}h_{}l_{}a_{}len_b{}_{}_{}_f32",
                                    size, n_layers, attn_heads, max_seq_len, batch_size, arch_tag, stage_tag,
                                );
                                runs.push(RunConfig {
                                    name,
                                    embed_dim: size,
                                    hidden_units: size,
                                    n_layers,
                                    attn_heads,
                                    ff_dim: if size == 512 && n_layers == 12 { 1536 } else { size * 4 },
                                    max_seq_len,
                                    architecture,
                                    stages: vec![StageConfig {
                                        stage,
                                        loss_threshold: 0.01,
                                        epochs: 10,
                                        batch_size,
                                        first_lr,
                                        last_lr,
                                        weight_decay: 0.01,
                                        epsilon: 1e-7,
                                        smoothing: 0.0,
                                    }],
                                    dropout_rate: 0.05,
                                    overfit: false,
                                    heatmap_every: 0,
                                    #[cfg(not(target_arch = "wasm32"))]
                                    adaptive_loss: None,
                                    adaptive_depth: None,
                                    kv_heads: None,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    runs
}

fn load_stage_data(
    stage: TrainingStage, 
    tokenizer: &TokenizerKind, 
    keyword_index: &HashMap<String, Vec<usize>>,
    max_seq_len: usize,
) -> Result<Vec<crate::brain::samples::Sample>> {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(path) = std::env::var_os("YUMON_SAMPLE_CACHE").filter(|path| !path.is_empty()) {
        let path = std::path::PathBuf::from(path);
        println!("[SampleCache] loading prepared samples from {}", path.display());
        return crate::brain::sample_cache::load_cache(&path, tokenizer, stage, max_seq_len);
    }
    stage_data_loader(stage).load(tokenizer, keyword_index, max_seq_len)
}

/// Every source for a training stage, with the global cap and seed.
/// Also used by `train_bpe` so the tokenizer sees the training data.
pub fn stage_data_loader(stage: TrainingStage) -> DataLoader {
    let mut loader = DataLoader::new(stage);
    // match stage {
    //     TrainingStage::Language => {
    //         loader = loader
    //             .add("archive/handcrafted_pairs.txt", FileKind::Chats, None);
    //     }
    //     TrainingStage::Structured => {
    //         loader = loader
    //             .add("archive/handcrafted_pairs.txt", FileKind::Chats, None);
    //     }
    // }

    loader = loader
        // // .add("data/chatbot_arena_conversations.json",   FileKind::JsonChats, None)
        .add("data/ideas.txt",   FileKind::TxtLines, Some(10_000))
        .add("archive/arena_extract.txt",   FileKind::Chats, Some(10_000))
        // .add("data/distillchatv1.csv",   FileKind::DistillChat, None)
        // Plain text, loss on every token (cleaned extract of the simplewiki XML).
        // .add("data/wiki_extract.txt",   FileKind::Paragraphs, None) // too dense
        // .add("data/quotes.csv",   FileKind::QuotesCsv, None) // only 5 prompts used. maybe better used without prompts
        // .add("data/bible_bbe.csv", FileKind::BibleCsv, None)
        // .add("data/bible_asv.csv", FileKind::BibleCsv, None)
        // LLM-generated Q&A pairs from src/bin/gen_synthetic_data.rs - real
        // message/reply splits (ideas, Bible and stories train as plain text).
        .add("archive/synthetic/bible.txt", FileKind::Chats, None)
        .add("archive/synthetic/business.txt", FileKind::Chats, None)
        .add("archive/synthetic/universe.txt", FileKind::Chats, None)
        .add("archive/synthetic/world_basics.txt", FileKind::Chats, None)
        .add("archive/synthetic/daily_life.txt", FileKind::Chats, None)
        .add("archive/synthetic/social_life.txt", FileKind::Chats, None)
        .add("archive/synthetic/puzzles.txt", FileKind::Chats, None)
        .add("archive/synthetic/commands.txt", FileKind::Chats, None)
        .add("archive/synthetic/memory.txt", FileKind::Chats, None)
        // .add("data/creative_stories.txt", FileKind::Stories, None)
        // // .add("data/Dictionary/Oxford/Oxford_English_Dictionary.txt",   FileKind::SpecificDict, Some(50_000))
        // // .add("archive/handcrafted_pairs.txt", FileKind::Chats, None);
        // // .add("archive/ov_chats.txt", FileKind::Chats, None)
        // .add("data/The-Office-Lines-V4.csv",   FileKind::DialogueCsv, None)
        // .add("data/friends_all_episodes_clean.csv",   FileKind::FriendsCsv, None)
        // // .add("archive/ov_chats.txt", FileKind::Chats, None)
        // // .add("archive/ov_chats.txt", FileKind::Chats, None)
        .add("archive/ov_chats.txt", FileKind::Chats, None)
        // // .add("archive/you_chats.txt", FileKind::Chats, None)
        // // .add("archive/you_chats.txt", FileKind::Chats, None)
        // // .add("archive/you_chats.txt", FileKind::Chats, None)
        .add("archive/you_chats.txt", FileKind::Chats, None)
        // // .add("archive/clean_chats.txt", FileKind::Chats, None)
        // // .add("archive/clean_chats.txt", FileKind::Chats, None)
        // // .add("archive/clean_chats.txt", FileKind::Chats, None)
        .add("archive/clean_chats.txt", FileKind::Chats, None);
        // .add(vec![
        //         "data/ebooks/faa-h-8083-25c.pdf".to_string(),
        //         "data/ebooks/algor_intro.pdf".to_string(),
        //         "data/ebooks/intro_engineer.pdf".to_string(),
        //         "data/ebooks/meap.pdf".to_string(),
        //         // "data/ebooks/missiles.pdf".to_string(),
        //         "data/ebooks/os_concepts.pdf".to_string(),
        //         "data/ebooks/real-time-embedded.pdf".to_string(),
        //         "data/ebooks/riscv.pdf".to_string(),
        //         "data/ebooks/rtos.pdf".to_string(),
        //         "data/ebooks/stephen_hawking_a_brief_history_of_time.pdf".to_string(),
        //     ].join(", "), 
        //     FileKind::PDF, 
        //     None
        // );

    // AM-DeepSeek-R1-Distilled (streamed, so the file can be a partial download).
    // Not baked into the Docker image: scripts/fetch_am_deepseek.sh pulls it onto
    // the RunPod volume. Skipped silently when absent so local runs still work.
    let am_path = std::env::var("YUMON_AM_PATH")
        .unwrap_or_else(|_| "data/am_deepseek/am_0.9M.jsonl.zst".to_string());
    if std::path::Path::new(&am_path).exists() {
        let am_limit = std::env::var("YUMON_AM_LIMIT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(300_000);
        loader = loader.add(am_path, FileKind::AmDistill, Some(am_limit));
    }

    loader
        // .total_limit(2_000_000)
        .total_limit(5_000_000)
        .seed(4815162342)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run(wiki_xml: &str, vision_checkpoint: &str, out_dir: &str, epochs: usize,
    batch_size: usize, max_articles: usize) -> Result<()> {
    run_with_architecture(wiki_xml, vision_checkpoint, out_dir, epochs, batch_size,
        max_articles, Architecture::XLstm)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run_with_architecture(
    wiki_xml:          &str,
    _vision_checkpoint: &str,  // reserved for future real-image fine-tuning
    out_dir:           &str,
    epochs:            usize,
    batch_size:        usize,
    max_articles:      usize,
    architecture: Architecture,
) -> Result<()> {
    #[cfg(not(any(feature = "cuda-training", all(target_os = "linux", not(feature = "desktop")))))]
    let device = burn::backend::wgpu::WgpuDevice::default();
    #[cfg(any(feature = "cuda-training", all(target_os = "linux", not(feature = "desktop"))))]
    let device = burn::backend::cuda::CudaDevice::default();
    #[cfg(any(feature = "cuda-training", all(target_os = "linux", not(feature = "desktop"))))]
    println!("Training precision: FP32 parameters, activations, gradients and optimizer moments.");
    #[cfg(not(any(feature = "cuda-training", all(target_os = "linux", not(feature = "desktop")))))]
    println!("Training precision: FP32 (local WGPU backend).");
    let label_keywords   = build_label_keywords();
    let keyword_index    = build_keyword_index(&label_keywords);
    let tokenizer_path = std::env::var("YUMON_TOKENIZER")
        .or_else(|_| std::env::var("TOKENIZER_PATH"))
        .unwrap_or_else(|_| {
            if let Some(cache) = std::env::var_os("YUMON_SAMPLE_CACHE") {
                if cache.to_string_lossy().contains("yumon_talk") {
                    let talk_tok = std::path::Path::new("training-cache/yumon_talk_tokenizer");
                    if talk_tok.exists() {
                        return "training-cache/yumon_talk_tokenizer".to_string();
                    }
                }
            }
            "yumon_bpe".to_string()
        });
    let tokenizer = TokenizerKind::Bpe(BpeTokenizer::load(&tokenizer_path)?);

    // ── Configure Runs ──────────────────────────────────────────────────────────
    // Full grid search (see generate_run_configs) instead of a hand-picked list -
    // MIN_EPOCH_LOSS_DROP cuts non-converging configs short, so covering the whole
    // space is affordable.
    let prompts = eval_prompts();
    let causal_lm = matches!(architecture, Architecture::Moe { .. } | Architecture::DecoderOnly);
    if causal_lm {
        anyhow::ensure!(batch_size > 0 && epochs > 0, "batch size and epochs must be positive");
    }
    let mut runs = generate_run_configs(batch_size, architecture);
    if causal_lm {
        // num_experts/top_k are per-run (see generate_run_configs' own sweep),
        // so only epochs/batch_size - the training-duration knobs, not capacity
        // knobs - get stamped on uniformly here.
        for run in &mut runs {
            for stage in &mut run.stages {
                stage.epochs = epochs;
                stage.batch_size = batch_size;
            }
        }
    }

    for run_cfg in runs {
        let run_dir = std::path::Path::new(out_dir).join(&run_cfg.name);
        std::fs::create_dir_all(&run_dir)?;
        let run_dir_str = run_dir.to_str().unwrap();

        println!("\n🚀 Starting Run: {} ({:?})", run_cfg.name, run_cfg.architecture);

        match run_cfg.architecture {
        Architecture::EncoderDecoder => {

        let (mut model, mut epochs_already_done) = if std::path::Path::new(run_dir_str).join("model.bin").exists() {
            match YumonBrain::<TrainBackend>::load(run_dir_str, &device) {
                Ok((m, _tok, _config)) => {
                    let meta_json = std::fs::read_to_string(std::path::Path::new(run_dir_str).join("metadata.json"))?;
                    let meta: BrainMetadata = serde_json::from_str(&meta_json)?;
                    println!("▶️  Resuming run {} from checkpoint ({} epochs done, loss={:.4})", 
                             run_cfg.name, meta.epochs_trained, meta.final_loss);
                    (m, meta.epochs_trained)
                }
                Err(_) => {
                    let config = YumonBrainConfig {
                    // let config = YumonDecBrainConfig {
                        vocab_size: tokenizer.vocab_size(),
                        embed_dim: run_cfg.embed_dim,
                        hidden_units: run_cfg.hidden_units,
                        n_layers: run_cfg.n_layers,
                        attn_heads: run_cfg.attn_heads,
                        ff_dim: run_cfg.ff_dim,
                        max_seq_len: run_cfg.max_seq_len,
                        training_stage: run_cfg.stages.get(0).expect("Couldn't get stage").stage,
                        dropout_rate: 0.05,
                    };
                    (config.init(&device), 0)
                }
            }
        } else {
            println!("🆕 Starting fresh run: {}", run_cfg.name);
            let config = YumonBrainConfig {
            // let config = YumonDecBrainConfig {
                vocab_size: tokenizer.vocab_size(),
                embed_dim: run_cfg.embed_dim,
                hidden_units: run_cfg.hidden_units,
                n_layers: run_cfg.n_layers,
                attn_heads: run_cfg.attn_heads,
                ff_dim: run_cfg.ff_dim,
                max_seq_len: run_cfg.max_seq_len,
                training_stage: run_cfg.stages.get(0).expect("Couldn't get stage").stage,
                dropout_rate: 0.05,
            };
            (config.init(&device), 0)
        };

        'stage_loop_enc: for (stage_idx, stage_cfg) in run_cfg.stages.iter().enumerate() {
            println!("\n🔨 Stage {}: {:?}", stage_idx + 1, stage_cfg.stage);

            let training_samples = load_stage_data(stage_cfg.stage.clone(), &tokenizer, &keyword_index, run_cfg.max_seq_len)?;
            println!("Training samples: {}", training_samples.len());

            // debug print — first 12 samples
            for (i, sample) in training_samples.iter().enumerate() {
                if i >= 12 { break; }
                println!("INPUT:  {:?}", tokenizer.decode(&sample.input_ids));
                println!("TARGET: {:?}", tokenizer.decode(
                    &sample.target_labels.iter()
                        .map(|&t| if t == PAD_TOKEN { PAD_TOKEN } else { t })
                        .collect::<Vec<_>>()
                ));
                println!("input_len:     {}", sample.input_ids.iter().filter(|&&t| t != PAD_TOKEN).count());
                println!("target_active: {}", sample.target_labels.iter().filter(|&&t| t != PAD_TOKEN).count());
            }

            let mut optimizer = AdamWConfig::new()
                .with_epsilon(stage_cfg.epsilon)
                .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
                .with_weight_decay(stage_cfg.weight_decay)
                .init();

            let ce_loss = CrossEntropyLossConfig::new()
                .with_pad_tokens(Some(vec![PAD_TOKEN as usize]))
                .with_smoothing(Some(stage_cfg.smoothing))
                .init(&device);

            let mut rng = rand::thread_rng();
            use std::io::{stdout, IsTerminal};
            // Headless runs (e.g. a detached Docker/RunPod container) have no real
            // terminal attached, so fall back to plain log lines instead of the TUI.
            let mut terminal = if stdout().is_terminal() {
                let backend = CrosstermBackend::new(stdout());
                Some(Terminal::with_options(
                    backend,
                    TerminalOptions { viewport: Viewport::Inline(46) },
                )?)
            } else {
                None
            };

            let total_batches = training_samples.len() / stage_cfg.batch_size;
            let mut state = TrainingState {
                loss_history: vec![],
                avg_loss_history: vec![],
                current_loss: 0.0,
                avg_loss: 0.0,
                epoch: 0,
                total_epochs: stage_cfg.epochs,
                batch: 0,
                total_batches: total_batches,
                current_lr: stage_cfg.first_lr,
                lr_history: vec![],
                global_step: 0,
                entropy: 0.0,
                entropy_history: vec![],
                top3_accuracy: None,
                avg_top3_accuracy: None,
                top3_history: vec![],
                last_reply: String::new()
            };

            let mut final_loss = 0.0f32;
            let inference_log_path = format!("{}/{}_inference_log.txt", run_dir_str, run_cfg.name);
            let chart_path = format!("{}/{}_stage_{}.png", run_dir_str, run_cfg.name, stage_idx + 1);
            let mut prev_epoch_loss: Option<f32> = None;
            let mut run_should_stop = false;

            'epoch_loop: for epoch in 0..stage_cfg.epochs {
                state.epoch = epoch + 1;
                let mut idx: Vec<usize> = (0..training_samples.len()).collect();
                idx.shuffle(&mut rng);
                let num_batches = idx.len().max(1) / stage_cfg.batch_size;
                let mut epoch_loss = 0.0f32;

                // --- Decoder-Encoder style
                for batch_num in 0..num_batches {
                    let current_lr = {
                        let total_steps = stage_cfg.epochs * num_batches;
                        let step = epoch * num_batches + batch_num;
                        let t = step as f64 / total_steps as f64;
                        (stage_cfg.first_lr * (1.0 - t) + stage_cfg.last_lr * t)
                    };

                    // // cosine annealing
                    // let current_lr = {
                    //     let total_steps = stage_cfg.epochs * num_batches;
                    //     let step = epoch * num_batches + batch_num;
                    //     let progress = step as f64 / total_steps as f64;
                    //     let cosine = (std::f64::consts::PI * progress).cos();
                    //     (last_lr + 0.5 * (first_lr - last_lr) * (1.0 + cosine)) as f64
                    // };

                    // // exp decay
                    // let current_lr = {
                    //     let total_steps = stage_cfg.epochs * num_batches;
                    //     let step = epoch * num_batches + batch_num;
                    //     let t = step as f64 / total_steps as f64;
                    //     (first_lr as f64 * (last_lr as f64 / first_lr as f64).powf(t))
                    // };

                    let batch_start = batch_num * stage_cfg.batch_size;
                    let batch_end = (batch_start + stage_cfg.batch_size).min(training_samples.len());
                    let batch_idx = &idx[batch_start..batch_end];
                    let current_batch_size = batch_idx.len();
                    if current_batch_size == 0 { continue; }

                    let mut all_lang_targets: Vec<i32> = Vec::with_capacity(current_batch_size * run_cfg.max_seq_len);
                    let mut all_enc_ids: Vec<i32> = Vec::with_capacity(current_batch_size * run_cfg.max_seq_len);
                    let mut all_dec_input_ids: Vec<i32> = Vec::with_capacity(current_batch_size * run_cfg.max_seq_len);

                    for &i in batch_idx {
                        let sample = &training_samples[i];
                        all_enc_ids.extend(sample.input_ids.iter().map(|&t| t as i32));
                        let target_labels = &sample.target_labels;
                        let real_len = target_labels.iter().position(|&t| t == PAD_TOKEN).unwrap_or(run_cfg.max_seq_len);

                        let mut dec_input: Vec<i32> = vec![BOS_TOKEN as i32];
                        dec_input.extend(target_labels[0..real_len.saturating_sub(1)].iter().map(|&t| t as i32));
                        dec_input.resize(run_cfg.max_seq_len, PAD_TOKEN as i32);

                        let mut lang_targets: Vec<i32> = target_labels[0..real_len].iter().map(|&t| t as i32).collect();
                        lang_targets.resize(run_cfg.max_seq_len, PAD_TOKEN as i32);

                        all_dec_input_ids.extend(dec_input);
                        all_lang_targets.extend(lang_targets);
                    }

                    let lang_target_t = Tensor::<TrainBackend, 1, Int>::from_ints(TensorData::new(all_lang_targets, [current_batch_size * run_cfg.max_seq_len]), &device);
                    let enc_t = Tensor::<TrainBackend, 2, Int>::from_ints(TensorData::new(all_enc_ids, [current_batch_size, run_cfg.max_seq_len]), &device);
                    let dec_t = Tensor::<TrainBackend, 2, Int>::from_ints(TensorData::new(all_dec_input_ids, [current_batch_size, run_cfg.max_seq_len]), &device);

                    let token_logits = model.forward::<TrainRuntime>(enc_t, dec_t.clone());
                    // let token_logits = model.forward::<CudaTrainRuntime>(enc_t, dec_t.clone()); // runpod

                    // Entropy
                    let probs = burn::tensor::activation::softmax(token_logits.clone(), 2);
                    let log_probs = (probs.clone() + 1e-10).log();
                    let token_entropy = (probs * log_probs).sum_dim(2).neg().squeeze::<2>();
                    let non_pad_mask = dec_t.clone().equal_elem(PAD_TOKEN as u32).bool_not().float();
                    let entropy_val: f32 = (token_entropy * non_pad_mask.clone()).cast(burn::tensor::FloatDType::F32).sum()
                    .div(non_pad_mask.cast(burn::tensor::FloatDType::F32).sum())
                    .into_data().convert::<f32>().to_vec::<f32>().unwrap()[0];

                    // Loss
                    let vocab = tokenizer.vocab_size();
                    let logits_2d = token_logits.reshape([current_batch_size * run_cfg.max_seq_len, vocab]);
                    let lang_loss = ce_loss.forward(logits_2d, lang_target_t);

                    let grads = GradientsParams::from_grads(lang_loss.backward(), &model);
                    model = optimizer.step(current_lr, model, grads);

                    let loss_val: f32 = lang_loss.clone().inner().to_data().convert::<f32>().to_vec::<f32>().unwrap()[0];
                    epoch_loss += loss_val;

                    state.entropy = entropy_val;
                    state.current_loss = loss_val;
                    state.avg_loss = epoch_loss / (batch_num + 1) as f32;
                    state.batch = batch_num + 1;
                    state.current_lr = current_lr;
                    state.global_step += 1;
                    state.loss_history.push((state.global_step as f64, loss_val as f64));
                    state.avg_loss_history.push((state.global_step as f64, state.avg_loss as f64));
                    state.entropy_history.push((state.global_step as f64, entropy_val as f64));
                    state.lr_history.push((state.global_step as f64, current_lr));

                    if let Some(term) = terminal.as_mut() {
                        term.draw(|frame| render(frame, &state))?;
                    } else if state.global_step % 50 == 0 || batch_num + 1 == num_batches {
                        println!(
                            "epoch {}/{} batch {}/{} loss {:.4} avg {:.4} lr {:.2e} entropy {:.4}",
                            state.epoch, state.total_epochs, state.batch, state.total_batches,
                            state.current_loss, state.avg_loss, state.current_lr, state.entropy,
                        );
                    }

                    // Periodic save and inference every 500 batches
                    if (batch_num + 1) % 500 == 0 {
                        let current_final_loss = epoch_loss / (batch_num + 1) as f32;
                        let meta = BrainMetadata {
                            vocab_size:     tokenizer.vocab_size(),
                            epochs_trained: epochs_already_done + epoch, // Partial epoch progress
                            final_loss:     current_final_loss,
                            batch_size:     stage_cfg.batch_size,
                            training_stage: stage_cfg.stage.clone(),
                            embed_dim:      run_cfg.embed_dim,
                            hidden_units:   run_cfg.hidden_units,
                            n_layers:       run_cfg.n_layers,
                            attn_heads:     run_cfg.attn_heads,
                            ff_dim:         run_cfg.ff_dim,
                            max_seq_len:    run_cfg.max_seq_len,
                        };
                        model.save(run_dir_str, &tokenizer, &meta)?;

                        // Periodic inference — 5 prompts for a well-rounded qualitative read,
                        // logged to disk (appended) and the loss chart re-saved (overwritten)
                        // right away rather than only at the end of the run.
                        let inference_model = model.valid();
                        let is_yumon_talk = is_yumon_talk_run(&run_cfg.name, &tokenizer, false);
                        let mut entries = Vec::with_capacity(prompts.len());
                        for p in &prompts {
                            let entry = evaluate_prompt(p, stage_cfg.stage, is_yumon_talk, |prompt| {
                                let result = inference_model.generate_unmasked_parsed::<TrainRuntime>(&tokenizer, prompt, run_cfg.max_seq_len, &device);
                                if stage_cfg.stage == TrainingStage::Structured { result.reply } else { result.raw_output }
                            });
                            entries.push(entry);
                        }
                        state.last_reply = entries.first().map(|entry| {
                            if let Some(yr) = &entry.yumon_reply {
                                format!("{} [{}]", entry.reply, yr)
                            } else {
                                entry.reply.clone()
                            }
                        }).unwrap_or_default();
                        if let Err(e) = append_inference_log(&inference_log_path, &run_cfg.name, stage_idx, stage_cfg.stage, state.epoch, state.total_epochs, state.batch, state.total_batches, state.avg_loss, &entries) {
                            eprintln!("⚠️  Failed to append inference log: {}", e);
                        }
                        if let Err(e) = state.save_chart_image(&chart_path) {
                            eprintln!("⚠️  Failed to save chart image: {}", e);
                        }
                    }

                    // Loss Threshold Exit
                    if state.avg_loss < stage_cfg.loss_threshold {
                        println!("\n🎯 Loss Threshold Reached: {:.4} < {:.4}. Ending Stage.", state.avg_loss, stage_cfg.loss_threshold);
                        final_loss = state.avg_loss;
                        break 'epoch_loop;
                    }
                }
                final_loss = epoch_loss / num_batches.max(1) as f32;

                // Automatic run-finish: if this epoch didn't drop avg loss by at
                // least MIN_EPOCH_LOSS_DROP versus the previous epoch, this config
                // isn't converging fast enough to be worth the remaining epochs —
                // finish the run here and move on to the next RunConfig.
                if let Some(prev) = prev_epoch_loss {
                    if prev - final_loss < MIN_EPOCH_LOSS_DROP {
                        println!(
                            "\n⏹️  Epoch loss drop {:.4} < {:.4} (prev {:.4} -> {:.4}). Finishing run early.",
                            prev - final_loss, MIN_EPOCH_LOSS_DROP, prev, final_loss,
                        );
                        run_should_stop = true;
                    }
                }
                prev_epoch_loss = Some(final_loss);

                // Save checkpoint after each epoch
                let meta = BrainMetadata {
                    vocab_size:     tokenizer.vocab_size(),
                    epochs_trained: epochs_already_done + epoch + 1,
                    final_loss,
                    batch_size: stage_cfg.batch_size,
                    training_stage: stage_cfg.stage.clone(),
                    embed_dim: run_cfg.embed_dim,
                    hidden_units: run_cfg.hidden_units,
                    n_layers: run_cfg.n_layers,
                    attn_heads: run_cfg.attn_heads,
                    ff_dim: run_cfg.ff_dim,
                    max_seq_len: run_cfg.max_seq_len,
                };
                model.save(run_dir_str, &tokenizer, &meta)?;

                // periodic inference — same 5-prompt log + chart save as above
                {
                    let inference_model = model.valid();
                    let is_yumon_talk = is_yumon_talk_run(&run_cfg.name, &tokenizer, false);
                    let mut entries = Vec::with_capacity(prompts.len());
                    for p in &prompts {
                        let entry = evaluate_prompt(p, stage_cfg.stage, is_yumon_talk, |prompt| {
                            let result = inference_model.generate_unmasked_parsed::<TrainRuntime>(&tokenizer, prompt, run_cfg.max_seq_len, &device);
                            if stage_cfg.stage == TrainingStage::Structured { result.reply } else { result.raw_output }
                        });
                        entries.push(entry);
                    }
                    state.last_reply = entries.first().map(|entry| {
                        if let Some(yr) = &entry.yumon_reply {
                            format!("{} [{}]", entry.reply, yr)
                        } else {
                            entry.reply.clone()
                        }
                    }).unwrap_or_default();
                    if let Err(e) = append_inference_log(&inference_log_path, &run_cfg.name, stage_idx, stage_cfg.stage, state.epoch, state.total_epochs, state.batch, state.total_batches, state.avg_loss, &entries) {
                        eprintln!("⚠️  Failed to append inference log: {}", e);
                    }
                    if let Err(e) = state.save_chart_image(&chart_path) {
                        eprintln!("⚠️  Failed to save chart image: {}", e);
                    }
                }

                if run_should_stop { break 'epoch_loop; }
            }
            epochs_already_done += state.epoch;
            if let Some(term) = terminal.as_mut() {
                term.clear()?;
            }

            // Final chart save as a safety net (the periodic saves above already
            // keep this path current throughout training).
            if let Err(e) = state.save_chart_image(&chart_path) {
                eprintln!("⚠️  Failed to save chart image: {}", e);
            } else {
                println!("📊 Chart saved to {}", chart_path);
            }

            println!("✅ Stage complete. Final loss: {:.4}", final_loss);

            if run_should_stop {
                println!("⏹️  Run {} finished early, skipping remaining stages.", run_cfg.name);
                break 'stage_loop_enc;
            }
        }

        } // Architecture::EncoderDecoder

        Architecture::XLstm => {

        let (mut model, mut epochs_already_done) = if std::path::Path::new(run_dir_str).join("model.bin").exists() {
            match YumonXLstmBrain::<TrainBackend>::load(run_dir_str, &device) {
                Ok((m, _tok, _config)) => {
                    let meta_json = std::fs::read_to_string(std::path::Path::new(run_dir_str).join("metadata.json"))?;
                    let meta: XLstmMetadata = serde_json::from_str(&meta_json)?;
                    println!("▶️  Resuming run {} from checkpoint ({} epochs done, loss={:.4})",
                             run_cfg.name, meta.epochs_trained, meta.final_loss);
                    (m, meta.epochs_trained)
                }
                Err(_) => {
                    let config = YumonXLstmBrainConfig {
                        vocab_size: tokenizer.vocab_size(),
                        embed_dim: run_cfg.embed_dim,
                        hidden_units: run_cfg.hidden_units,
                        n_layers: run_cfg.n_layers,
                        attn_heads: run_cfg.attn_heads,
                        ff_dim: run_cfg.ff_dim,
                        max_seq_len: run_cfg.max_seq_len,
                        training_stage: run_cfg.stages.get(0).expect("Couldn't get stage").stage,
                        dropout_rate: 0.05,
                    };
                    (config.init(&device), 0)
                }
            }
        } else {
            println!("🆕 Starting fresh run: {}", run_cfg.name);
            let config = YumonXLstmBrainConfig {
                vocab_size: tokenizer.vocab_size(),
                embed_dim: run_cfg.embed_dim,
                hidden_units: run_cfg.hidden_units,
                n_layers: run_cfg.n_layers,
                attn_heads: run_cfg.attn_heads,
                ff_dim: run_cfg.ff_dim,
                max_seq_len: run_cfg.max_seq_len,
                training_stage: run_cfg.stages.get(0).expect("Couldn't get stage").stage,
                dropout_rate: 0.05,
            };
            (config.init(&device), 0)
        };

        'stage_loop_xlstm: for (stage_idx, stage_cfg) in run_cfg.stages.iter().enumerate() {
            println!("\n🔨 Stage {}: {:?}", stage_idx + 1, stage_cfg.stage);

            let training_samples = load_stage_data(stage_cfg.stage.clone(), &tokenizer, &keyword_index, run_cfg.max_seq_len)?;
            println!("Training samples: {}", training_samples.len());

            for (i, sample) in training_samples.iter().enumerate() {
                if i >= 12 { break; }
                println!("INPUT:  {:?}", tokenizer.decode(&sample.input_ids));
                println!("TARGET: {:?}", tokenizer.decode(
                    &sample.target_labels.iter()
                        .map(|&t| if t == PAD_TOKEN { PAD_TOKEN } else { t })
                        .collect::<Vec<_>>()
                ));
                println!("input_len:     {}", sample.input_ids.iter().filter(|&&t| t != PAD_TOKEN).count());
                println!("target_active: {}", sample.target_labels.iter().filter(|&&t| t != PAD_TOKEN).count());
            }

            let mut optimizer = AdamWConfig::new()
                .with_epsilon(stage_cfg.epsilon)
                .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
                .with_weight_decay(stage_cfg.weight_decay)
                .init();

            let ce_loss = CrossEntropyLossConfig::new()
                .with_pad_tokens(Some(vec![PAD_TOKEN as usize]))
                .with_smoothing(Some(stage_cfg.smoothing))
                .init(&device);

            let mut rng = rand::thread_rng();
            use std::io::{stdout, IsTerminal};
            let mut terminal = if stdout().is_terminal() {
                let backend = CrosstermBackend::new(stdout());
                Some(Terminal::with_options(
                    backend,
                    TerminalOptions { viewport: Viewport::Inline(46) },
                )?)
            } else {
                None
            };

            let total_batches = training_samples.len() / stage_cfg.batch_size;
            let mut state = TrainingState {
                loss_history: vec![],
                avg_loss_history: vec![],
                current_loss: 0.0,
                avg_loss: 0.0,
                epoch: 0,
                total_epochs: stage_cfg.epochs,
                batch: 0,
                total_batches: total_batches,
                current_lr: stage_cfg.first_lr,
                lr_history: vec![],
                global_step: 0,
                entropy: 0.0,
                entropy_history: vec![],
                top3_accuracy: None,
                avg_top3_accuracy: None,
                top3_history: vec![],
                last_reply: String::new()
            };

            let mut final_loss = 0.0f32;
            let inference_log_path = format!("{}/{}_inference_log.txt", run_dir_str, run_cfg.name);
            let chart_path = format!("{}/{}_stage_{}.png", run_dir_str, run_cfg.name, stage_idx + 1);
            let mut prev_epoch_loss: Option<f32> = None;
            let mut run_should_stop = false;

            'epoch_loop_xlstm: for epoch in 0..stage_cfg.epochs {
                state.epoch = epoch + 1;
                let mut idx: Vec<usize> = (0..training_samples.len()).collect();
                idx.shuffle(&mut rng);
                let num_batches = idx.len().max(1) / stage_cfg.batch_size;
                let mut epoch_loss = 0.0f32;

                // Decoder-only style: [prompt][separator][reply] packed into one causal sequence
                let sep_text = if stage_cfg.stage == TrainingStage::Structured { "\n---\n" } else { " " };
                let sep_tokens = tokenizer.encode(sep_text);

                for batch_num in 0..num_batches {
                    let current_lr = {
                        let total_steps = stage_cfg.epochs * num_batches;
                        let step = epoch * num_batches + batch_num;
                        let t = step as f64 / total_steps as f64;
                        (stage_cfg.first_lr * (1.0 - t) + stage_cfg.last_lr * t)
                    };

                    let batch_start = batch_num * stage_cfg.batch_size;
                    let batch_end = (batch_start + stage_cfg.batch_size).min(training_samples.len());
                    let batch_idx = &idx[batch_start..batch_end];
                    let current_batch_size = batch_idx.len();
                    if current_batch_size == 0 { continue; }

                    let (all_seq_ids, all_lang_targets, _) =
                        build_moe_batch(&training_samples, batch_idx, &sep_tokens, run_cfg.max_seq_len, false);

                    let lang_target_t = Tensor::<TrainBackend, 1, Int>::from_ints(TensorData::new(all_lang_targets, [current_batch_size * run_cfg.max_seq_len]), &device);
                    let tokens_t = Tensor::<TrainBackend, 2, Int>::from_ints(TensorData::new(all_seq_ids, [current_batch_size, run_cfg.max_seq_len]), &device);

                    let token_logits = model.forward(tokens_t.clone());

                    // Entropy
                    let probs = burn::tensor::activation::softmax(token_logits.clone(), 2);
                    let log_probs = (probs.clone() + 1e-10).log();
                    let token_entropy = (probs * log_probs).sum_dim(2).neg().squeeze::<2>();
                    let non_pad_mask = tokens_t.clone().equal_elem(PAD_TOKEN as u32).bool_not().float();
                    let entropy_val: f32 = (token_entropy * non_pad_mask.clone()).cast(burn::tensor::FloatDType::F32).sum()
                    .div(non_pad_mask.cast(burn::tensor::FloatDType::F32).sum())
                    .into_data().convert::<f32>().to_vec::<f32>().unwrap()[0];

                    // Loss
                    let vocab = tokenizer.vocab_size();
                    let logits_2d = token_logits.reshape([current_batch_size * run_cfg.max_seq_len, vocab]);
                    let lang_loss = ce_loss.forward(logits_2d, lang_target_t);

                    let grads = GradientsParams::from_grads(lang_loss.backward(), &model);
                    model = optimizer.step(current_lr, model, grads);

                    let loss_val: f32 = lang_loss.clone().inner().to_data().convert::<f32>().to_vec::<f32>().unwrap()[0];
                    epoch_loss += loss_val;

                    state.entropy = entropy_val;
                    state.current_loss = loss_val;
                    state.avg_loss = epoch_loss / (batch_num + 1) as f32;
                    state.batch = batch_num + 1;
                    state.current_lr = current_lr;
                    state.global_step += 1;
                    state.loss_history.push((state.global_step as f64, loss_val as f64));
                    state.avg_loss_history.push((state.global_step as f64, state.avg_loss as f64));
                    state.entropy_history.push((state.global_step as f64, entropy_val as f64));
                    state.lr_history.push((state.global_step as f64, current_lr));

                    if let Some(term) = terminal.as_mut() {
                        term.draw(|frame| render(frame, &state))?;
                    } else if state.global_step % 50 == 0 || batch_num + 1 == num_batches {
                        println!(
                            "epoch {}/{} batch {}/{} loss {:.4} avg {:.4} lr {:.2e} entropy {:.4}",
                            state.epoch, state.total_epochs, state.batch, state.total_batches,
                            state.current_loss, state.avg_loss, state.current_lr, state.entropy,
                        );
                    }

                    // Periodic save and inference every 500 batches
                    if (batch_num + 1) % 500 == 0 {
                        let current_final_loss = epoch_loss / (batch_num + 1) as f32;
                        let meta = XLstmMetadata {
                            vocab_size:     tokenizer.vocab_size(),
                            epochs_trained: epochs_already_done + epoch,
                            final_loss:     current_final_loss,
                            batch_size:     stage_cfg.batch_size,
                            training_stage: stage_cfg.stage.clone(),
                            embed_dim:      run_cfg.embed_dim,
                            hidden_units:   run_cfg.hidden_units,
                            n_layers:       run_cfg.n_layers,
                            attn_heads:     run_cfg.attn_heads,
                            ff_dim:         run_cfg.ff_dim,
                            max_seq_len:    run_cfg.max_seq_len,
                        };
                        model.save(run_dir_str, &tokenizer, &meta)?;

                        let inference_model = model.valid();
                        let is_yumon_talk = is_yumon_talk_run(&run_cfg.name, &tokenizer, false);
                        let mut entries = Vec::with_capacity(prompts.len());
                        for p in &prompts {
                            let entry = evaluate_prompt(p, stage_cfg.stage, is_yumon_talk, |prompt| {
                                let result = inference_model.generate_unmasked_parsed(&tokenizer, prompt, run_cfg.max_seq_len, &device);
                                if stage_cfg.stage == TrainingStage::Structured { result.reply } else { result.raw_output }
                            });
                            entries.push(entry);
                        }
                        state.last_reply = entries.first().map(|entry| {
                            if let Some(yr) = &entry.yumon_reply {
                                format!("{} [{}]", entry.reply, yr)
                            } else {
                                entry.reply.clone()
                            }
                        }).unwrap_or_default();
                        if let Err(e) = append_inference_log(&inference_log_path, &run_cfg.name, stage_idx, stage_cfg.stage, state.epoch, state.total_epochs, state.batch, state.total_batches, state.avg_loss, &entries) {
                            eprintln!("⚠️  Failed to append inference log: {}", e);
                        }
                        if let Err(e) = state.save_chart_image(&chart_path) {
                            eprintln!("⚠️  Failed to save chart image: {}", e);
                        }
                    }

                    // Loss Threshold Exit
                    if state.avg_loss < stage_cfg.loss_threshold {
                        println!("\n🎯 Loss Threshold Reached: {:.4} < {:.4}. Ending Stage.", state.avg_loss, stage_cfg.loss_threshold);
                        final_loss = state.avg_loss;
                        break 'epoch_loop_xlstm;
                    }
                }
                final_loss = epoch_loss / num_batches.max(1) as f32;

                if let Some(prev) = prev_epoch_loss {
                    if prev - final_loss < MIN_EPOCH_LOSS_DROP {
                        println!(
                            "\n⏹️  Epoch loss drop {:.4} < {:.4} (prev {:.4} -> {:.4}). Finishing run early.",
                            prev - final_loss, MIN_EPOCH_LOSS_DROP, prev, final_loss,
                        );
                        run_should_stop = true;
                    }
                }
                prev_epoch_loss = Some(final_loss);

                let meta = XLstmMetadata {
                    vocab_size:     tokenizer.vocab_size(),
                    epochs_trained: epochs_already_done + epoch + 1,
                    final_loss,
                    batch_size: stage_cfg.batch_size,
                    training_stage: stage_cfg.stage.clone(),
                    embed_dim: run_cfg.embed_dim,
                    hidden_units: run_cfg.hidden_units,
                    n_layers: run_cfg.n_layers,
                    attn_heads: run_cfg.attn_heads,
                    ff_dim: run_cfg.ff_dim,
                    max_seq_len: run_cfg.max_seq_len,
                };
                model.save(run_dir_str, &tokenizer, &meta)?;

                {
                    let inference_model = model.valid();
                    let is_yumon_talk = is_yumon_talk_run(&run_cfg.name, &tokenizer, false);
                    let mut entries = Vec::with_capacity(prompts.len());
                    for p in &prompts {
                        let entry = evaluate_prompt(p, stage_cfg.stage, is_yumon_talk, |prompt| {
                            let result = inference_model.generate_unmasked_parsed(&tokenizer, prompt, run_cfg.max_seq_len, &device);
                            if stage_cfg.stage == TrainingStage::Structured { result.reply } else { result.raw_output }
                        });
                        entries.push(entry);
                    }
                    state.last_reply = entries.first().map(|entry| {
                        if let Some(yr) = &entry.yumon_reply {
                            format!("{} [{}]", entry.reply, yr)
                        } else {
                            entry.reply.clone()
                        }
                    }).unwrap_or_default();
                    if let Err(e) = append_inference_log(&inference_log_path, &run_cfg.name, stage_idx, stage_cfg.stage, state.epoch, state.total_epochs, state.batch, state.total_batches, state.avg_loss, &entries) {
                        eprintln!("⚠️  Failed to append inference log: {}", e);
                    }
                    if let Err(e) = state.save_chart_image(&chart_path) {
                        eprintln!("⚠️  Failed to save chart image: {}", e);
                    }
                }

                if run_should_stop { break 'epoch_loop_xlstm; }
            }
            epochs_already_done += state.epoch;
            if let Some(term) = terminal.as_mut() {
                term.clear()?;
            }

            if let Err(e) = state.save_chart_image(&chart_path) {
                eprintln!("⚠️  Failed to save chart image: {}", e);
            } else {
                println!("📊 Chart saved to {}", chart_path);
            }

            println!("✅ Stage complete. Final loss: {:.4}", final_loss);

            if run_should_stop {
                println!("⏹️  Run {} finished early, skipping remaining stages.", run_cfg.name);
                break 'stage_loop_xlstm;
            }
        }

        } // Architecture::XLstm

        Architecture::Moe { .. } => {
            train_causal_lm::<YumonMoeBrain<TrainBackend>>(&run_cfg, &run_dir, &tokenizer, &keyword_index, &prompts, &device, None)?;
        }

        Architecture::DecoderOnly => {
            train_causal_lm::<YumonDecBrain<TrainBackend>>(&run_cfg, &run_dir, &tokenizer, &keyword_index, &prompts, &device, None)?;
        }

        Architecture::Reference => {
            train_causal_lm::<YumonReferenceBrain<TrainBackend>>(&run_cfg, &run_dir, &tokenizer, &keyword_index, &prompts, &device, None)?;
        }

        } // match run_cfg.architecture
    }

    println!("✅ All training runs complete.");
    Ok(())
}

// ─── Emote keyword heuristic ──────────────────────────────────────────────────

/// Simple keyword-based pseudo-label for emote head pre-training.
/// 0=angry, 1=disgust, 2=fear, 3=happy, 4=neutral, 5=sad, 6=surprise
pub fn keyword_emote_label(text: &str) -> usize {
    let lower = text.to_lowercase();
    if lower.contains("war") || lower.contains("attack") || lower.contains("conflict") {
        0
    } else if lower.contains("poison") || lower.contains("disease") || lower.contains("waste") {
        1
    } else if lower.contains("danger") || lower.contains("threat") || lower.contains("risk") {
        2
    } else if lower.contains("celebrat") || lower.contains("award") || lower.contains("success") {
        3
    } else if lower.contains("death") || lower.contains("loss") || lower.contains("victim") {
        5
    } else if lower.contains("discover") || lower.contains("unexpect") || lower.contains("sudden") {
        6
    } else {
        4 // neutral
    }
}

// ─── Progress bar ─────────────────────────────────────────────────────────────
#[cfg(not(target_arch = "wasm32"))]
pub fn make_progress(total: usize, epoch: usize, epochs: usize) -> ProgressBar {
    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner} epoch {msg} [{bar:40}] {pos}/{len} loss={prefix}")
            .unwrap()
    );
    pb.set_message(format!("{epoch}/{epochs}"));
    pb
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "../../tests/code_curriculum/training.rs"]
mod code_curriculum_training_tests;

#[cfg(test)]
mod language_run_tests {
    use super::*;

    #[test]
    fn training_grid_uses_512_token_language_context() {
        for architecture in [Architecture::XLstm, Architecture::EncoderDecoder,
            Architecture::Moe { num_experts: 4, top_k: 1 }] {
            let runs = generate_run_configs(8, architecture);
            assert!(!runs.is_empty());
            for run in runs {
                assert_eq!(run.max_seq_len, 512);
                assert!(run.name.contains("_512len_"));
                assert!(run.stages.iter().all(|s| s.stage == TrainingStage::Language));
            }
        }
    }

    #[test]
    fn trim_batch_keeps_the_longest_row_and_its_targets() {
        let p = PAD_TOKEN as i32;
        // Row 1 ends at position 3 in its ids; row 2 has a target one past its ids.
        let ids = vec![1, 7, 8, p, p, p, 1, 9, p, p, p, p];
        let targets = vec![7, 8, 2, p, p, p, 9, p, 5, p, p, p];
        let (ids, targets, seq) = trim_batch(ids, targets, 2, 6);
        assert_eq!(seq, 3);
        assert_eq!(ids, vec![1, 7, 8, 1, 9, p]);
        assert_eq!(targets, vec![7, 8, 2, 9, p, 5]);
        // A full row keeps the whole context untouched.
        let full: Vec<i32> = (1..=4).collect();
        assert_eq!(trim_batch(full.clone(), full.clone(), 1, 4), (full.clone(), full, 4));
    }

    #[test]
    fn plain_text_trains_every_token_and_pairs_mask_the_prompt() {
        use crate::brain::{BOS_TOKEN, EOS_TOKEN, samples::{Action, CardinalDir, Sample}};
        let (p, len) = (PAD_TOKEN as i32, 10);
        let sample = |input: &[usize], target: &[usize]| {
            let pad = |ids: &[usize]| { let mut v = ids.to_vec(); v.resize(len, PAD_TOKEN); v };
            Sample { input_ids: pad(input), target_labels: pad(target), action: Action::Sit,
                motion_dir: CardinalDir::None, world: Default::default(), target_json: String::new(),
                pair: Default::default() }
        };
        let samples = [sample(&[BOS_TOKEN], &[7, 8, 9, EOS_TOKEN]), sample(&[BOS_TOKEN, 5, 6], &[7, EOS_TOKEN])];
        let (ids, targets, supervised) = build_moe_batch(&samples, &[0, 1], &[4], len, false);
        let (bos, eos) = (BOS_TOKEN as i32, EOS_TOKEN as i32);
        // Plain text: no separator, BOS predicts the first token, loss through EOS.
        assert_eq!(ids[..len], [bos, 7, 8, 9, eos, p, p, p, p, p]);
        assert_eq!(targets[..len], [7, 8, 9, eos, p, p, p, p, p, p]);
        // Prompt-reply: only the separator and reply are supervised.
        assert_eq!(ids[len..], [bos, 5, 6, 4, 7, eos, p, p, p, p]);
        assert_eq!(targets[len..], [p, p, 4, 7, eos, p, p, p, p, p]);
        assert_eq!(supervised, 4 + 3);
    }
}

#[cfg(test)]
mod yumon_talk_log_tests {
    use super::*;

    #[test]
    fn test_is_yumon_talk_run_detection() {
        let tok = TokenizerKind::Bpe(BpeTokenizer::load("yumon_bpe").unwrap());
        // Run name checks
        assert!(is_yumon_talk_run("512h_12l_16a_512len_b4_Moe_e8_k2_talk_Language_f32", &tok, false));
        assert!(is_yumon_talk_run("yumon_talk_moe", &tok, false));
        assert!(!is_yumon_talk_run("kilo_code_moe", &tok, false));
        assert!(!is_yumon_talk_run("code_512h_12l_16a_Language_f32", &tok, true));
    }

    #[test]
    fn test_append_inference_log_dual_format() {
        let temp_dir = std::env::temp_dir();
        let log_path = temp_dir.join("test_yumon_inference_log.txt");
        let log_path_str = log_path.to_string_lossy().to_string();
        let _ = std::fs::remove_file(&log_path);

        let entries = vec![
            EvalLogEntry {
                prompt: "Tell me about yourself.".to_string(),
                reply: "I am a small pet.".to_string(),
                yumon_prompt: Some("talk self".to_string()),
                yumon_reply: Some("me tiny yumon".to_string()),
            },
        ];

        append_inference_log(
            &log_path_str,
            "test_run",
            0,
            TrainingStage::Language,
            1,
            10,
            50,
            100,
            2.345,
            &entries,
        ).unwrap();

        let content = std::fs::read_to_string(&log_path).unwrap();
        assert!(content.contains("[1] PROMPT (English):    Tell me about yourself."));
        assert!(content.contains("[1] PROMPT (Yumon Talk): talk self"));
        assert!(content.contains("[1] REPLY  (Yumon Talk): me tiny yumon"));
        assert!(content.contains("[1] REPLY  (English):    I am a small pet."));
        let _ = std::fs::remove_file(&log_path);
    }

    #[test]
    fn test_evaluate_prompt_roundtrip() {
        let p = "Hello world";
        let entry = evaluate_prompt(p, TrainingStage::Language, true, |prompt| {
            format!("{prompt} response")
        });
        assert_eq!(entry.prompt, "Hello world");
        assert!(entry.yumon_prompt.is_some());
        assert!(entry.yumon_reply.is_some());
        assert!(!entry.reply.is_empty());
    }
}

#[cfg(test)]
mod moe_loss_tests {
    #[test]
    fn moe_tiny_training_smoke_local() {
        type Local = burn::backend::Autodiff<
            burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>,
            burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing>;
        super::tiny_moe_smoke::<Local>(&Default::default(), burn::tensor::DType::F32).unwrap();
    }

    #[test]
    #[ignore = "requires CUDA GPU with BF16 support"]
    fn cuda_bf16_moe_tiny_training_smoke() {
        super::tiny_moe_smoke::<super::CudaBf16TrainBackend>(&Default::default(), burn::tensor::DType::BF16).unwrap();
    }

    use super::*;
    use burn::backend::Wgpu;
    type B = Wgpu;

    fn check_top3<B: Backend>() {
        let device = Default::default();
        let logits = Tensor::<B, 2>::from_floats([
            [0.0, 5.0, 4.0, 3.0, 2.0], // target 1: first
            [0.0, 5.0, 4.0, 3.0, 2.0], // target 3: third
            [0.0, 5.0, 4.0, 3.0, 2.0], // target 4: fourth
            [5.0, 4.0, 3.0, 2.0, 1.0], // padding: excluded even though first
            [1.0, 1.0, 1.0, 1.0, 1.0], // tie: ID 2 is third
            [1.0, 1.0, 1.0, 1.0, 1.0], // tie: ID 3 is fourth
        ], &device);
        let targets = Tensor::<B, 1, Int>::from_ints([1, 3, 4, PAD_TOKEN as i32, 2, 3], &device);
        assert_eq!(top3_correct(logits.clone(), targets), 3);
        let padding = Tensor::<B, 1, Int>::full([6], PAD_TOKEN as i32, &device);
        assert_eq!(top3_correct(logits, padding), 0);
        // A vocabulary smaller than k still includes every supervised target.
        let small = Tensor::<B, 2>::from_floats([[2.0, 1.0]], &device);
        assert_eq!(top3_correct(small, Tensor::from_ints([1], &device)), 1);
    }

    #[test]
    fn top3_accuracy_masks_padding_and_breaks_ties() {
        // Match the unfused backend used by the actual training loop.
        check_top3::<burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>>();
    }

    #[test]
    #[ignore = "requires a CUDA GPU with native BF16 support; no profiling"]
    fn cuda_bf16_top3_accuracy() {
        check_top3::<<CudaBf16TrainBackend as AutodiffBackend>::InnerBackend>();
    }

    #[test]
    fn moe_grid_matches_yumon_talk_moe_shape() {
        let runs = generate_run_configs(16, Architecture::Moe { num_experts: 4, top_k: 1 });
        assert_eq!(runs.len(), 1);
        let run = &runs[0];
        assert_eq!((run.embed_dim, run.n_layers, run.attn_heads, run.ff_dim), (256, 12, 16, 1024));
        assert!(matches!(run.architecture, Architecture::Moe { num_experts: 4, top_k: 1 }));
        assert_eq!(run.stages[0].batch_size, 16);
        assert_eq!(run.name, "256h_12l_16a_512len_b16_Moe_e4_k1_Language_f32");
        let custom = generate_run_configs(8, Architecture::Moe { num_experts: 8, top_k: 2 });
        assert!(matches!(custom[0].architecture, Architecture::Moe { num_experts: 8, top_k: 2 }));
    }

    #[test]
    fn moe_masked_loss_averages_supervised_tokens_only() {
        let device = Default::default();
        let logits = Tensor::<B, 2>::from_floats(
            [[2.0, 0.5, -1.0], [0.1, 0.2, 0.3], [-0.5, 1.5, 0.0], [3.0, -2.0, 1.0]], &device);
        // Rows 1 and 3 are padding.
        let targets = Tensor::<B, 1, Int>::from_ints([1, PAD_TOKEN as i32, 2, PAD_TOKEN as i32], &device);

        let row_nll = |row: [f32; 3], target: usize| {
            let max = row.iter().cloned().fold(f32::MIN, f32::max);
            let log_sum = row.iter().map(|x| (x - max).exp()).sum::<f32>().ln() + max;
            log_sum - row[target]
        };
        let expected = (row_nll([2.0, 0.5, -1.0], 1) + row_nll([-0.5, 1.5, 0.0], 2)) / 2.0;

        let ours: f32 = masked_token_ce(logits.clone(), targets.clone(), 2, 0.0).into_scalar();
        assert!((ours - expected).abs() < 1e-5, "{ours} vs {expected}");

        // Burn's padded cross-entropy divides by all 4 rows, so it reads half as large here.
        let burn: f32 = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![PAD_TOKEN]))
            .init(&device)
            .forward(logits, targets)
            .into_scalar();
        assert!((burn * 4.0 / 2.0 - expected).abs() < 1e-5, "{burn} vs {expected}");
    }

    /// Adaptive loss: sum(a * CE) / supervised, PAD ignored, unit weights equal
    /// masked CE, and each row's gradient scales by its weight.
    #[test]
    fn weighted_loss_scales_rows_and_gradients() {
        type AD = burn::backend::Autodiff<B>;
        let device = Default::default();
        let rows = [[2.0, 0.5, -1.0], [0.1, 0.2, 0.3], [-0.5, 1.5, 0.0], [3.0, -2.0, 1.0]];
        let targets = || Tensor::<AD, 1, Int>::from_ints([1, PAD_TOKEN as i32, 2, PAD_TOKEN as i32], &device);
        let scalar = |t: Tensor<AD, 1>| t.into_data().convert::<f32>().to_vec::<f32>().unwrap()[0];
        let plain = scalar(masked_token_ce(Tensor::<AD, 2>::from_floats(rows, &device), targets(), 2, 0.0));
        let ones = Tensor::<AD, 1>::from_floats([1.0, 0.0, 1.0, 0.0], &device);
        let unit = scalar(weighted_token_ce(Tensor::<AD, 2>::from_floats(rows, &device), targets(), Some(ones), 2, 0.0));
        assert!((plain - unit).abs() < 1e-6, "{plain} vs {unit}");

        let grads_for = |weights: Option<[f32; 4]>| {
            let logits = Tensor::<AD, 2>::from_floats(rows, &device).require_grad();
            let weights = weights.map(|w| Tensor::<AD, 1>::from_floats(w, &device));
            let loss = weighted_token_ce(logits.clone(), targets(), weights, 2, 0.0);
            let value = scalar(loss.clone());
            let grads = loss.backward();
            (value, logits.grad(&grads).unwrap().into_data().convert::<f32>().to_vec::<f32>().unwrap())
        };
        let (base, base_grad) = grads_for(None);
        // PAD rows carry weight 0 from row_weights; a stray weight there must still be masked.
        let (weighted, weighted_grad) = grads_for(Some([0.5, 7.0, 2.0, 7.0]));
        let nll = |row: [f32; 3], target: usize| {
            let max = row.iter().cloned().fold(f32::MIN, f32::max);
            row.iter().map(|x| (x - max).exp()).sum::<f32>().ln() + max - row[target]
        };
        let expected = (0.5 * nll(rows[0], 1) + 2.0 * nll(rows[2], 2)) / 2.0;
        assert!((weighted - expected).abs() < 1e-5, "{weighted} vs {expected}");
        assert!((base - plain).abs() < 1e-6);
        for (i, (g, b)) in weighted_grad.iter().zip(&base_grad).enumerate() {
            let scale = [0.5, 0.0, 2.0, 0.0][i / 3];
            assert!(g.is_finite() && (g - scale * b).abs() < 1e-5, "row {} grad {g} vs {}", i / 3, scale * b);
        }
    }

    /// Real Language data: samples, duplicates removed and tokens per source,
    /// one source in memory at a time. Run from the repo root with:
    /// cargo test --release --no-default-features --lib language_data_report -- --ignored --nocapture
    #[test]
    #[ignore]
    fn language_data_report() {
        let tokenizer = TokenizerKind::Bpe(BpeTokenizer::load("yumon_bpe").unwrap());
        let max_seq_len = 256;
        let sep_len = tokenizer.encode(" ").len();
        let reports = stage_data_loader(TrainingStage::Language)
            .report(&tokenizer, &HashMap::new(), max_seq_len)
            .unwrap();

        println!("\n{:<40} {:>10} {:>10} {:>14} {:>14}", "source", "samples", "dupes", "prompt tok", "trained tok");
        let (mut samples, mut dupes, mut prompt, mut trained) = (0, 0, 0, 0);
        for r in &reports {
            // Prompt-reply samples also train on their separator; plain text has none.
            let r_trained = r.target_tokens + (r.samples - r.plain_samples) * sep_len;
            println!("{:<40} {:>10} {:>10} {:>14} {:>14}", r.path, r.samples, r.duplicates, r.prompt_tokens, r_trained);
            samples += r.samples; dupes += r.duplicates; prompt += r.prompt_tokens; trained += r_trained;
        }
        println!("{:<40} {:>10} {:>10} {:>14} {:>14}", "TOTAL", samples, dupes, prompt, trained);
        println!("trained tokens / padded positions: {:.1}%", 100.0 * trained as f64 / (samples * max_seq_len) as f64);
    }

    /// Decoded wiki and quote conversations from the real files, to check that
    /// they read like chat. Run with language_data_report's command.
    #[test]
    #[ignore]
    fn language_data_samples() {
        let tokenizer = TokenizerKind::Bpe(BpeTokenizer::load("yumon_bpe").unwrap());
        for (path, kind, count) in [("data/wiki_extract.txt", FileKind::Paragraphs, 12), ("data/quotes.csv", FileKind::QuotesCsv, 6), ("data/creative_stories.txt", FileKind::Stories, 8)] {
            // 500 random conversations keep this to about a minute.
            let samples = DataLoader::new(TrainingStage::Language)
                .add(path, kind, Some(500))
                .seed(4815162342)
                .load(&tokenizer, &HashMap::new(), 256)
                .unwrap();
            println!("\n===== {path}: {} samples =====", samples.len());
            for sample in samples.iter().take(count) {
                println!("--- INPUT:\n{}\n--- TARGET:\n{}\n", tokenizer.decode(&sample.input_ids), tokenizer.decode(&sample.target_labels));
            }
        }
    }

    #[test]
    fn language_loader_removes_exact_duplicates() {
        let tokenizer = TokenizerKind::Bpe(BpeTokenizer::load("yumon_bpe").unwrap());
        let dir = std::env::temp_dir().join(format!("yumon-dedupe-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lines.txt");
        std::fs::write(&path, "a platform that helps farmers grow crops\na platform that helps farmers grow crops\na tool for writers to plan stories\n").unwrap();

        let samples = DataLoader::new(TrainingStage::Language)
            .add(path.to_str().unwrap(), FileKind::TxtLines, None)
            .add(path.to_str().unwrap(), FileKind::TxtLines, None) // same file again
            .load(&tokenizer, &HashMap::new(), 256)
            .unwrap();
        assert_eq!(samples.len(), 2);

        assert_eq!(dir.parent(), Some(std::env::temp_dir().as_path()));
        std::fs::remove_dir_all(dir).unwrap();
    }
}

/// Training-step memory and time on the local wgpu adapter. One variant per
/// process so the memory pool starts empty. Example (repo root):
/// YUMON_PROBE=dec-balanced PROBE_WIDTH=256 PROBE_LAYERS=16 PROBE_BATCH=32 \
///   cargo test --release --lib training_memory_probe -- --ignored --nocapture
/// Variants: dec-balanced, dec-plain, moe-balanced, moe-plain (no checkpointing),
/// attn-flash-{balanced,plain},
/// attn-naive-{balanced,plain} (attention alone at the model's shape).
#[cfg(test)]
mod memory_probe_tests {
    use super::*;
    use burn::backend::autodiff::{Autodiff, checkpoint::strategy::{BalancedCheckpointing, NoCheckpointing}};
    use burn_cubecl::CubeBackend;
    use cubecl::{Runtime, wgpu::{WgpuDevice, WgpuRuntime}};

    type Inner = CubeBackend<WgpuRuntime, f32, i32, u32>;

    fn env(name: &str, default: usize) -> usize {
        std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
    }

    struct Shape { width: usize, layers: usize, heads: usize, seq: usize, batch: usize, steps: usize, vocab: usize }

    const MIB: f64 = 1024.0 * 1024.0;

    /// Runs `steps` forward+backward passes; reports activation bytes held at
    /// the end of forward and pool growth (peak working set incl. fragmentation).
    fn measure<AD: AutodiffBackend<Device = WgpuDevice>>(label: &str, s: &Shape, mut step: impl FnMut(usize) -> Tensor<AD, 1>) {
        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);
        AD::sync(&device).unwrap();
        client.memory_cleanup();
        let base = client.memory_usage();
        let (mut held, mut times, mut fwd_times) = (0u64, vec![], vec![]);
        for i in 0..s.steps {
            AD::sync(&device).unwrap();
            let start = std::time::Instant::now();
            let loss = step(i);
            AD::sync(&device).unwrap();
            fwd_times.push(start.elapsed());
            held = held.max(client.memory_usage().bytes_in_use.saturating_sub(base.bytes_in_use));
            let grads = loss.backward();
            drop(grads);
            AD::sync(&device).unwrap();
            times.push(start.elapsed());
        }
        let peak = client.memory_usage().bytes_reserved.saturating_sub(base.bytes_reserved);
        let warm = &times[times.len().min(2)..];
        let avg = warm.iter().sum::<std::time::Duration>() / warm.len().max(1) as u32;
        let warm_fwd = &fwd_times[fwd_times.len().min(2)..];
        let avg_fwd = warm_fwd.iter().sum::<std::time::Duration>() / warm_fwd.len().max(1) as u32;
        println!(
            "PROBE {label} width={} layers={} heads={} seq={} batch={} | params+inputs in use {:.0} MiB | held after forward {:.0} MiB | pool growth {:.0} MiB | step {:.0} ms, forward {:.0} ms (avg of {} after warmup)",
            s.width, s.layers, s.heads, s.seq, s.batch,
            base.bytes_in_use as f64 / MIB, held as f64 / MIB, peak as f64 / MIB,
            avg.as_secs_f64() * 1e3, avg_fwd.as_secs_f64() * 1e3, warm.len(),
        );
    }

    fn tokens<AD: Backend>(s: &Shape, step: usize, device: &AD::Device) -> (Tensor<AD, 2, Int>, Tensor<AD, 1, Int>) {
        // Full-length rows (no padding): the worst case for memory.
        let ids: Vec<i32> = (0..s.batch * s.seq).map(|i| 3 + ((i * 7919 + step * 104729) % (s.vocab - 3)) as i32).collect();
        let mut targets = ids.clone();
        targets.rotate_left(1);
        (
            Tensor::from_ints(TensorData::new(ids, [s.batch, s.seq]), device),
            Tensor::from_ints(TensorData::new(targets, [s.batch * s.seq]), device),
        )
    }

    fn lm_loss<AD: Backend>(logits: Tensor<AD, 3>, targets: Tensor<AD, 1, Int>, s: &Shape) -> Tensor<AD, 1> {
        masked_token_ce(logits.reshape([s.batch * s.seq, s.vocab]), targets, s.batch * s.seq, 0.0)
    }

    fn dec<AD: AutodiffBackend<Device = WgpuDevice> + FlashAttention>(label: &str, s: &Shape) {
        let device = WgpuDevice::default();
        let config = YumonDecBrainConfig::new(s.vocab, TrainingStage::Language)
            .with_embed_dim(s.width).with_hidden_units(s.width).with_n_layers(s.layers)
            .with_attn_heads(s.heads).with_ff_dim(4 * s.width).with_max_seq_len(s.seq);
        println!("{label}: {:.1}M parameters", config.param_count() as f64 / 1e6);
        let model: YumonDecBrain<AD> = config.init(&device);
        measure::<AD>(label, s, |i| {
            let (x, y) = tokens::<AD>(s, i, &device);
            lm_loss(model.forward(x), y, s)
        });
    }

    fn moe<AD: AutodiffBackend<Device = WgpuDevice> + FlashAttention>(label: &str, s: &Shape) {
        let device = WgpuDevice::default();
        let config = YumonMoeBrainConfig::new(s.vocab, TrainingStage::Language)
            .with_embed_dim(s.width).with_hidden_units(s.width).with_n_layers(s.layers)
            .with_attn_heads(s.heads).with_ff_dim(4 * s.width).with_max_seq_len(s.seq)
            .with_num_experts(4).with_top_k(1);
        let model: YumonMoeBrain<AD> = config.init(&device);
        println!("{label}: {:.1}M parameters", model.num_params() as f64 / 1e6);
        measure::<AD>(label, s, |i| {
            let (x, y) = tokens::<AD>(s, i, &device);
            let (logits, aux, _) = model.forward_with_aux(x);
            lm_loss(logits, y, s) + aux
        });
    }

    fn attn<AD: AutodiffBackend<Device = WgpuDevice> + FlashAttention>(label: &str, flash: bool, s: &Shape) {
        let device = WgpuDevice::default();
        let hd = s.width / s.heads;
        let max_shared = WgpuRuntime::client(&device).properties().hardware.max_shared_memory_size;
        println!("{label}: head dim {hd}, shared memory {max_shared} B, {} rows per tile",
            crate::brain::flash_attn::backend::block_for_dim(hd, max_shared));
        measure::<AD>(label, s, |_| {
            let rand = || Tensor::<AD, 4>::random([s.batch, s.heads, s.seq, hd], burn::tensor::Distribution::Normal(0.0, 1.0), &device).require_grad();
            let (q, k, v) = (rand(), rand(), rand());
            let out = if flash {
                crate::brain::flash_attn::backend::causal_flash_attention(q, k, v)
            } else {
                let future = Tensor::<AD, 2>::ones([s.seq, s.seq], &device).triu(1).bool().unsqueeze::<4>();
                let scores = q.matmul(k.transpose()) / (hd as f64).sqrt();
                burn::tensor::activation::softmax(scores.mask_fill(future, -1e9), 3).matmul(v)
            };
            out.powf_scalar(2.0).mean().reshape([1])
        });
    }

    #[test]
    #[ignore]
    fn training_memory_probe() {
        let s = Shape {
            width: env("PROBE_WIDTH", 256), layers: env("PROBE_LAYERS", 16), heads: env("PROBE_HEADS", 8),
            seq: env("PROBE_SEQ", 256), batch: env("PROBE_BATCH", 32), steps: env("PROBE_STEPS", 5),
            vocab: env("PROBE_VOCAB", 16384),
        };
        let variant = std::env::var("YUMON_PROBE").unwrap_or_else(|_| "dec-balanced".into());
        match variant.as_str() {
            "dec-balanced" => dec::<Autodiff<Inner, BalancedCheckpointing>>(&variant, &s),
            "dec-plain" => dec::<Autodiff<Inner, NoCheckpointing>>(&variant, &s),
            "moe-balanced" => moe::<Autodiff<Inner, BalancedCheckpointing>>(&variant, &s),
            "moe-plain" => moe::<Autodiff<Inner, NoCheckpointing>>(&variant, &s),
            "attn-flash-balanced" => attn::<Autodiff<Inner, BalancedCheckpointing>>(&variant, true, &s),
            "attn-flash-plain" => attn::<Autodiff<Inner, NoCheckpointing>>(&variant, true, &s),
            "attn-naive-balanced" => attn::<Autodiff<Inner, BalancedCheckpointing>>(&variant, false, &s),
            "attn-naive-plain" => attn::<Autodiff<Inner, NoCheckpointing>>(&variant, false, &s),
            other => panic!("unknown YUMON_PROBE {other}"),
        }
    }
}
