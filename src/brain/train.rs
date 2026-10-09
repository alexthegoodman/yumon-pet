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
};

// Used by the local desktop training UI (src/bin/train_ui.rs), which runs on
// wgpu since dev machines typically have no CUDA GPU.
//
// Both training backends are plain CubeBackend (no burn-fusion) so the custom
// flash attention op in flash_attn/backend.rs applies, and use Burn's
// BalancedCheckpointing: memory-bound ops (elementwise, norms, activations,
// reshapes) are recomputed during backward instead of stored.
#[cfg(not(all(target_os = "linux", not(feature = "desktop"))))]
pub type TrainBackend = burn::backend::Autodiff<
    burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>,
    burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing,
>;
#[cfg(not(all(target_os = "linux", not(feature = "desktop"))))]
pub type TrainRuntime = cubecl::wgpu::WgpuRuntime;
#[cfg(all(target_os = "linux", not(feature = "desktop")))]
pub type TrainBackend = CudaTrainBackend;
#[cfg(all(target_os = "linux", not(feature = "desktop")))]
pub type TrainRuntime = CudaTrainRuntime;
// pub type TrainBackend = burn::backend::Autodiff<burn::backend::NdArray<f32>>;

// Used by the headless `train-brain` CLI path (`run`, below) — this is what
// RunPod/Docker actually runs. CUDA talks to the driver directly and needs no
// Vulkan/GL adapter, unlike wgpu, which RunPod's driver stack doesn't expose.
pub type CudaTrainBackend = burn::backend::Autodiff<
    burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime, f32, i32, u8>,
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

/// Appends one qualitative-eval snapshot (all `entries` prompt/reply pairs)
/// to the run's inference log. Appends rather than overwrites so the log
/// reads as a history of how replies evolved over the run.
fn append_inference_log(
    log_path:     &str,
    run_name:     &str,
    stage_idx:    usize,
    stage:        TrainingStage,
    epoch:        usize,
    total_epochs: usize,
    batch:        usize,
    total_batches: usize,
    avg_loss:     f32,
    entries:      &[(String, String)],
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
    for (i, (prompt, reply)) in entries.iter().enumerate() {
        writeln!(file, "[{}] PROMPT: {}", i + 1, prompt)?;
        writeln!(file, "[{}] REPLY:  {}", i + 1, reply)?;
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
/// only (PAD elsewhere). Returns token ids, targets and the supervised count.
fn build_moe_batch(
    samples:     &[crate::brain::samples::Sample],
    batch_idx:   &[usize],
    sep_tokens:  &[usize],
    max_seq_len: usize,
) -> (Vec<i32>, Vec<i32>, usize) {
    let sep_len = sep_tokens.len();
    let mut all_seq_ids: Vec<i32> = Vec::with_capacity(batch_idx.len() * max_seq_len);
    let mut all_targets: Vec<i32> = Vec::with_capacity(batch_idx.len() * max_seq_len);
    let mut supervised = 0usize;

    for &i in batch_idx {
        let sample = &samples[i];
        let input_ids = &sample.input_ids;
        let target_labels = &sample.target_labels;

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

/// Mean cross-entropy per supervised (non-PAD) target. Burn 0.20's
/// CrossEntropyLoss zeroes PAD rows but still divides by every row, which
/// scales the loss by the fraction of positions that are supervised.
fn masked_token_ce<B: Backend>(
    logits:     Tensor<B, 2>,
    targets:    Tensor<B, 1, Int>,
    supervised: usize,
    smoothing:  f32,
) -> Tensor<B, 1> {
    let [rows, _] = logits.dims();
    let mask = targets.clone().equal_elem(PAD_TOKEN as i32).bool_not().float();
    let log_probs = burn::tensor::activation::log_softmax(logits, 1);
    let nll = log_probs.clone().gather(1, targets.reshape([rows, 1])).reshape([rows]).neg();
    let per_token = if smoothing > 0.0 {
        let uniform = log_probs.mean_dim(1).reshape([rows]).neg();
        nll * (1.0 - smoothing as f64) + uniform * smoothing as f64
    } else {
        nll
    };
    (per_token * mask).sum() / supervised.max(1) as f64
}

/// Per-token loss over the held-out samples, without dropout or gradients.
/// Returns the loss and the number of supervised tokens it covers.
fn causal_validation_loss<B: Backend, M: CausalLm<B>>(
    model:       &M,
    samples:     &[crate::brain::samples::Sample],
    sep_tokens:  &[usize],
    max_seq_len: usize,
    batch_size:  usize,
    vocab:       usize,
    device:      &B::Device,
) -> Option<(f32, usize)> {
    let mut total = 0.0f64;
    let mut tokens = 0usize;
    let idx: Vec<usize> = (0..samples.len()).collect();
    for batch_idx in idx.chunks(batch_size) {
        let (seq_ids, targets, supervised) = build_moe_batch(samples, batch_idx, sep_tokens, max_seq_len);
        if supervised == 0 { continue; }
        let n = batch_idx.len();
        let tokens_t = Tensor::<B, 2, Int>::from_ints(TensorData::new(seq_ids, [n, max_seq_len]), device);
        let targets_t = Tensor::<B, 1, Int>::from_ints(TensorData::new(targets, [n * max_seq_len]), device);
        let logits = model.logits(tokens_t).reshape([n * max_seq_len, vocab]);
        let loss = masked_token_ce(logits, targets_t, supervised, 0.0)
            .into_data().convert::<f32>().to_vec::<f32>().ok()?[0];
        total += loss as f64 * supervised as f64;
        tokens += supervised;
    }
    (tokens > 0).then(|| ((total / tokens as f64) as f32, tokens))
}

// ─── Shared causal-LM training loop (MoE and dense decoder-only) ─────────────

type InnerBackend = <TrainBackend as AutodiffBackend>::InnerBackend;

/// What validation and qualitative eval need from a decoder-only model.
trait CausalLm<B: Backend> {
    fn logits(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3>;
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

impl<B: Backend> CausalLm<B> for YumonMoeBrain<B> {
    fn logits(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> { self.forward(tokens) }
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
            .with_num_experts(num_experts).with_top_k(top_k);
        println!("MoE: {} experts, top-{}; FFN parameters/layer: {} total, {} active/token (excluding router).",
            num_experts, top_k, 3 * run_cfg.embed_dim * run_cfg.ff_dim * num_experts,
            3 * run_cfg.embed_dim * run_cfg.ff_dim * top_k);
        if !run_dir.join("model.bin").exists() {
            return Ok((config.init(device), 0));
        }
        let (m, checkpoint_tokenizer, saved) = Self::load(run_dir.to_str().unwrap(), device)?;
        anyhow::ensure!(saved.vocab_size == config.vocab_size && saved.embed_dim == config.embed_dim
            && saved.ff_dim == config.ff_dim && saved.n_layers == config.n_layers
            && saved.attn_heads == config.attn_heads && saved.max_seq_len == config.max_seq_len
            && saved.num_experts == num_experts && saved.top_k == top_k,
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
            .with_ff_dim(run_cfg.ff_dim).with_max_seq_len(run_cfg.max_seq_len);
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

/// Qualitative eval: every prompt through the inference model.
#[cfg(not(target_arch = "wasm32"))]
fn run_eval_prompts<M: CausalLm<InnerBackend>>(
    model:       &M,
    tokenizer:   &TokenizerKind,
    prompts:     &[String],
    stage:       TrainingStage,
    max_seq_len: usize,
    device:      &<TrainBackend as Backend>::Device,
) -> Vec<(String, String)> {
    prompts.iter().map(|p| {
        let prompt = build_inference_prompt(stage, p);
        let result = model.generate(tokenizer, &prompt, max_seq_len, device);
        let reply = if stage == TrainingStage::Structured { result.reply } else { result.raw_output };
        (p.clone(), reply)
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
) -> Result<()>
where
    M: CausalLmTrain,
    M::InnerModule: CausalLm<InnerBackend>,
{
    let run_dir_str = run_dir.to_str().unwrap();
    let (mut model, mut epochs_already_done) = M::init_or_resume(run_cfg, tokenizer, run_dir, device)?;

    'stage_loop: for (stage_idx, stage_cfg) in run_cfg.stages.iter().enumerate() {
        model.set_stage(stage_cfg.stage);
        println!("\n🔨 Stage {}: {:?}", stage_idx + 1, stage_cfg.stage);

        let mut training_samples = load_stage_data(stage_cfg.stage.clone(), tokenizer, keyword_index, run_cfg.max_seq_len)?;
        anyhow::ensure!(training_samples.len() > 1, "Not enough training samples for stage");
        // Deduped and seed-shuffled by the loader, so the tail is a fixed held-out set.
        let val_count = (training_samples.len() / 100).clamp(1, MAX_VAL_SAMPLES);
        let val_samples = training_samples.split_off(training_samples.len() - val_count);
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
            last_reply: String::new()
        };

        let mut final_loss = 0.0f32;
        let inference_log_path = format!("{}/{}_inference_log.txt", run_dir_str, run_cfg.name);
        let chart_path = format!("{}/{}_stage_{}.png", run_dir_str, run_cfg.name, stage_idx + 1);
        let mut prev_epoch_loss: Option<f32> = None;
        let mut run_should_stop = false;
        let mut val_loss: Option<f32>;
        let vocab = tokenizer.vocab_size();

        // Decoder-only style: [prompt][separator][reply] packed into one causal sequence
        let sep_text = if stage_cfg.stage == TrainingStage::Structured { "\n---\n" } else { " " };
        let sep_tokens = tokenizer.encode(sep_text);
        anyhow::ensure!(sep_tokens.len() + 2 <= run_cfg.max_seq_len, "Sequence too short for separator and reply");

        'epoch_loop: for epoch in 0..stage_cfg.epochs {
            state.epoch = epoch + 1;
            let mut idx: Vec<usize> = (0..training_samples.len()).collect();
            idx.shuffle(&mut rng);
            let num_batches = idx.len().div_ceil(stage_cfg.batch_size);
            let mut epoch_loss = 0.0f32;
            let mut processed_batches = 0usize;

            for batch_num in 0..num_batches {
                let current_lr = {
                    let total_steps = stage_cfg.epochs * num_batches;
                    let step = epoch * num_batches + batch_num;
                    let t = step as f64 / total_steps as f64;
                    stage_cfg.first_lr * (1.0 - t) + stage_cfg.last_lr * t
                };

                let batch_start = batch_num * stage_cfg.batch_size;
                let batch_end = (batch_start + stage_cfg.batch_size).min(training_samples.len());
                let batch_idx = &idx[batch_start..batch_end];
                let current_batch_size = batch_idx.len();
                if current_batch_size == 0 { continue; }

                let (all_seq_ids, all_lang_targets, supervised) = build_moe_batch(&training_samples, batch_idx, &sep_tokens, run_cfg.max_seq_len);

                if supervised == 0 { continue; }
                let lang_target_t = Tensor::<TrainBackend, 1, Int>::from_ints(TensorData::new(all_lang_targets, [current_batch_size * run_cfg.max_seq_len]), device);
                let tokens_t = Tensor::<TrainBackend, 2, Int>::from_ints(TensorData::new(all_seq_ids, [current_batch_size, run_cfg.max_seq_len]), device);

                let (token_logits, aux_loss) = model.forward_train(tokens_t.clone());

                // Entropy, on detached logits so none of its [batch, seq, vocab]
                // intermediates join the autodiff graph.
                let probs = burn::tensor::activation::softmax(token_logits.clone().detach(), 2);
                let log_probs = (probs.clone() + 1e-10).log();
                let token_entropy = (probs * log_probs).sum_dim(2).neg().squeeze_dim::<2>(2);
                let non_pad_mask = tokens_t.clone().equal_elem(PAD_TOKEN as u32).bool_not().float();
                let entropy_val: f32 = (token_entropy * non_pad_mask.clone()).sum().div(non_pad_mask.sum()).into_scalar();

                // Loss
                let logits_2d = token_logits.reshape([current_batch_size * run_cfg.max_seq_len, vocab]);
                let lang_loss = masked_token_ce(logits_2d, lang_target_t, supervised, stage_cfg.smoothing);

                let total_loss = lang_loss.clone() + aux_loss;
                let grads = GradientsParams::from_grads(total_loss.backward(), &model);
                model = optimizer.step(current_lr, model, grads);

                let loss_val: f32 = lang_loss.inner().to_data().to_vec::<f32>().unwrap()[0];
                epoch_loss += loss_val;
                processed_batches += 1;

                state.entropy = entropy_val;
                state.current_loss = loss_val;
                state.avg_loss = epoch_loss / processed_batches as f32;
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
                    let inference_model = model.valid();
                    let val = causal_validation_loss(&inference_model, &val_samples, &sep_tokens, run_cfg.max_seq_len, stage_cfg.batch_size, vocab, device);
                    val_loss = val.map(|(loss, _)| loss);
                    model.save_run(run_dir_str, tokenizer, &RunProgress {
                        run_cfg,
                        epochs_trained: epochs_already_done + epoch,
                        final_loss:     epoch_loss / processed_batches as f32,
                        val_loss,
                        batch_size:     stage_cfg.batch_size,
                        stage:          stage_cfg.stage,
                    })?;

                    let entries = run_eval_prompts(&inference_model, tokenizer, prompts, stage_cfg.stage, run_cfg.max_seq_len, device);
                    state.last_reply = entries[0].1.clone();
                    if let Err(e) = append_inference_log(&inference_log_path, &run_cfg.name, stage_idx, stage_cfg.stage, state.epoch, state.total_epochs, state.batch, state.total_batches, state.avg_loss, &entries) {
                        eprintln!("⚠️  Failed to append inference log: {}", e);
                    }
                    if let Some((loss, tokens)) = val {
                        let line = format!("val_loss {loss:.4} over {tokens} held-out tokens");
                        println!("{line}");
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
            let val = causal_validation_loss(&inference_model, &val_samples, &sep_tokens, run_cfg.max_seq_len, stage_cfg.batch_size, vocab, device);
            val_loss = val.map(|(loss, _)| loss);

            // Early stop on held-out loss when available.
            let stop_loss = val_loss.unwrap_or(final_loss);
            if let Some(prev) = prev_epoch_loss {
                if prev - stop_loss < MIN_EPOCH_LOSS_DROP {
                    println!(
                        "\n⏹️  Epoch loss drop {:.4} < {:.4} (prev {:.4} -> {:.4}). Finishing run early.",
                        prev - stop_loss, MIN_EPOCH_LOSS_DROP, prev, stop_loss,
                    );
                    run_should_stop = true;
                }
            }
            prev_epoch_loss = Some(stop_loss);

            model.save_run(run_dir_str, tokenizer, &RunProgress {
                run_cfg,
                epochs_trained: epochs_already_done + epoch + 1,
                final_loss,
                val_loss,
                batch_size: stage_cfg.batch_size,
                stage:      stage_cfg.stage,
            })?;

            let entries = run_eval_prompts(&inference_model, tokenizer, prompts, stage_cfg.stage, run_cfg.max_seq_len, device);
            state.last_reply = entries[0].1.clone();
            if let Err(e) = append_inference_log(&inference_log_path, &run_cfg.name, stage_idx, stage_cfg.stage, state.epoch, state.total_epochs, state.batch, state.total_batches, state.avg_loss, &entries) {
                eprintln!("⚠️  Failed to append inference log: {}", e);
            }
            if let Some((loss, tokens)) = val {
                let line = format!("val_loss {loss:.4} over {tokens} held-out tokens (epoch end)");
                println!("{line}");
                if let Err(e) = append_log_line(&inference_log_path, &line) {
                    eprintln!("⚠️  Failed to append inference log: {}", e);
                }
            }
            if let Err(e) = state.save_chart_image(&chart_path) {
                eprintln!("⚠️  Failed to save chart image: {}", e);
            }

            if run_should_stop || final_loss < stage_cfg.loss_threshold { break 'epoch_loop; }
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
    // Dense decoder-only RunPod run: 1024 wide x 24 layers x 32 heads (head
    // dim 32, the flash kernel's fastest measured width), ~436M params, batch
    // 32 x 256 tokens (Dockerfile). Estimated peak ~28 GiB: weights + grads +
    // AdamW 6.5 GiB, stored activations 18.3 GiB, logits/CE 3.2 GiB - scaled
    // from training_memory_probe at 256 wide on wgpu, so confirm on the pod.
    let sizes:       [usize; 1] = [
        // 32,
        // 64,
        // 128,
        // 256, // smaller, maybe good seq length of 256
        // 512 // 220M ideal
        // 1024, // ~1.32B total / ~411M active at 24 layers, 4 experts top-1
        1024, // dense: ~436M at 24 layers
    ];
    let layer_counts: [usize; 1] = [
        // 1,
        // 2,
        // 4,
        // 8,
        // 8, // stretch: slower on iGPU - each MoE layer forces a host readback
        // 16 // discovered minimum (runpod)
        24
        // 32
    ];
    let head_counts:  [usize; 1] = [
        // 1,
        // 2,
        // 4,
        // 8,
        // 16,
        32, // head dim 32 at 1024 wide
        // 64
    ];
    let seq_lens:     [usize; 1] = [
        256, // good for memories and conversation
        // 512 // good starting seq length for code
    ];
    let batch_sizes:     [usize; 1] = [
        // 2,
        // 8
        if matches!(architecture, Architecture::Moe { .. } | Architecture::DecoderOnly) { batch_size_option } else { 16 }
        // 32
        // 64
    ];
    // For Moe, num_experts/top_k are swept here as real matrix dimensions
    // instead of the single fixed pair the CLI's --moe-experts/--moe-top-k
    // used to stamp onto every run - those two CLI flags are now unused for
    // architecture=moe (kept only for the other architectures' CLI parsing).
    let moe_expert_configs: [(usize, usize); 1] = [
        // (2, 1),
        (4, 1),
        // (4, 2),
        // (8, 1), // ~420M total at 512 wide x 16 layers
        // (8, 2),
        // (16, 2), // stretch: doubles total params, same active params as (8,2)
    ];
    let architectures: Vec<Architecture> = if matches!(architecture, Architecture::Moe { .. }) {
        moe_expert_configs
            .iter()
            .map(|&(num_experts, top_k)| Architecture::Moe { num_experts, top_k })
            .collect()
    } else {
        vec![
            // Architecture::DecoderOnly,
            // Architecture::EncoderDecoder,
            architecture,
        ]
    };
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
                                    // Architecture::Moe { .. } => (3e-4, 3e-5),
                                    Architecture::Moe { .. } => (1e-4, 1e-5),
                                    // 24 layers, no warmup, 8k tokens/step: below the usual 3e-4.
                                    Architecture::DecoderOnly    => (2e-4, 2e-5),
                                    Architecture::EncoderDecoder => (1e-3, 1e-4),
                                    // xLSTM's exponential gating is more sensitive to a hot LR
                                    // than plain attention softmax — starting below DecoderOnly's
                                    // own rate is the standard recurrent-net-with-gating caution,
                                    // not a benchmarked-optimal number.
                                    // Architecture::XLstm          => (1e-4, 1e-5),
                                    Architecture::XLstm          => (1e-3, 1e-4),
                                    // Architecture::XLstm          => (1e-2, 1e-6),
                                };
                                let arch_tag = match architecture {
                                    Architecture::Moe { num_experts, top_k } => format!("Moe_e{num_experts}_k{top_k}"),
                                    Architecture::DecoderOnly    => "DecoderOnly".to_string(),
                                    Architecture::EncoderDecoder => "EncoderDecoder".to_string(),
                                    Architecture::XLstm          => "XLstm".to_string(),
                                };
                                let stage_tag = match stage {
                                    TrainingStage::Language   => "Language",
                                    TrainingStage::Structured => "Structured",
                                };
                                let name = format!(
                                    "{}h_{}l_{}a_{}len_b{}_{}_{}",
                                    size, n_layers, attn_heads, max_seq_len, batch_size, arch_tag, stage_tag,
                                );
                                runs.push(RunConfig {
                                    name,
                                    embed_dim: size,
                                    hidden_units: size,
                                    n_layers,
                                    attn_heads,
                                    ff_dim: size * 4,
                                    max_seq_len,
                                    architecture,
                                    stages: vec![StageConfig {
                                        stage,
                                        loss_threshold: 0.01,
                                        epochs: 15,
                                        batch_size,
                                        first_lr,
                                        last_lr,
                                        weight_decay: 0.01,
                                        epsilon: 1e-7,
                                        smoothing: 0.0,
                                    }],
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
        .add("data/ideas.txt",   FileKind::TxtLines, None)
        .add("archive/arena_extract.txt",   FileKind::Chats, None)
        .add("data/distillchatv1.csv",   FileKind::DistillChat, None)
        // Plain text, loss on every token (cleaned extract of the simplewiki XML).
        // .add("data/wiki_extract.txt",   FileKind::Paragraphs, None) // too dense
        // .add("data/quotes.csv",   FileKind::QuotesCsv, None) // only 5 prompts used. maybe better used without prompts
        .add("data/bible_bbe.csv", FileKind::BibleCsv, None)
        .add("data/bible_asv.csv", FileKind::BibleCsv, None)
        // LLM-generated Q&A pairs from src/bin/gen_synthetic_data.rs — proper
        // message/reply splits instead of BibleCsv's arbitrary mid-sentence cuts.
        .add("archive/synthetic/bible.txt", FileKind::Chats, None)
        .add("archive/synthetic/business.txt", FileKind::Chats, None)
        .add("archive/synthetic/universe.txt", FileKind::Chats, None)
        .add("archive/synthetic/world_basics.txt", FileKind::Chats, None)
        .add("archive/synthetic/daily_life.txt", FileKind::Chats, None)
        .add("archive/synthetic/social_life.txt", FileKind::Chats, None)
        .add("archive/synthetic/puzzles.txt", FileKind::Chats, None)
        .add("archive/synthetic/commands.txt", FileKind::Chats, None)
        .add("archive/synthetic/memory.txt", FileKind::Chats, None)
        .add("data/creative_stories.txt", FileKind::Stories, None)
        // // .add("data/Dictionary/Oxford/Oxford_English_Dictionary.txt",   FileKind::SpecificDict, Some(50_000))
        // // .add("archive/handcrafted_pairs.txt", FileKind::Chats, None);
        // // .add("archive/ov_chats.txt", FileKind::Chats, None)
        .add("data/The-Office-Lines-V4.csv",   FileKind::DialogueCsv, None)
        .add("data/friends_all_episodes_clean.csv",   FileKind::FriendsCsv, None)
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
    #[cfg(not(all(target_os = "linux", not(feature = "desktop"))))]
    let device = burn::backend::wgpu::WgpuDevice::default();
    #[cfg(all(target_os = "linux", not(feature = "desktop")))]
    let device = burn::backend::cuda::CudaDevice::default();
    let label_keywords   = build_label_keywords();
    let keyword_index    = build_keyword_index(&label_keywords);
    let tokenizer = TokenizerKind::Bpe(BpeTokenizer::load("yumon_bpe")?);

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
                    let entropy_val: f32 = (token_entropy * non_pad_mask.clone()).sum().div(non_pad_mask.sum()).into_scalar();

                    // Loss
                    let vocab = tokenizer.vocab_size();
                    let logits_2d = token_logits.reshape([current_batch_size * run_cfg.max_seq_len, vocab]);
                    let lang_loss = ce_loss.forward(logits_2d, lang_target_t);

                    let grads = GradientsParams::from_grads(lang_loss.backward(), &model);
                    model = optimizer.step(current_lr, model, grads);

                    let loss_val: f32 = lang_loss.clone().inner().to_data().to_vec::<f32>().unwrap()[0];
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
                        let mut entries = Vec::with_capacity(prompts.len());
                        for p in &prompts {
                            let prompt = build_inference_prompt(stage_cfg.stage, p);
                            let result = inference_model.generate_unmasked_parsed::<TrainRuntime>(&tokenizer, &prompt, run_cfg.max_seq_len, &device);
                            // let result = inference_model.generate_unmasked_parsed::<CudaTrainRuntime>(&tokenizer, &prompt, run_cfg.max_seq_len, &device); // runpod
                            let reply = if stage_cfg.stage == TrainingStage::Structured { result.reply } else { result.raw_output };
                            entries.push((p.clone(), reply));
                        }
                        state.last_reply = entries[0].1.clone();
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
                    let mut entries = Vec::with_capacity(prompts.len());
                    for p in &prompts {
                        let prompt = build_inference_prompt(stage_cfg.stage, p);
                        let result = inference_model.generate_unmasked_parsed::<TrainRuntime>(&tokenizer, &prompt, run_cfg.max_seq_len, &device);
                        // let result = inference_model.generate_unmasked_parsed::<CudaTrainRuntime>(&tokenizer, &prompt, run_cfg.max_seq_len, &device); // runpod
                        let reply = if stage_cfg.stage == TrainingStage::Structured { result.reply } else { result.raw_output };
                        entries.push((p.clone(), reply));
                    }
                    state.last_reply = entries[0].1.clone();
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
                let sep_len = sep_tokens.len();

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

                    let mut all_lang_targets: Vec<i32> = Vec::with_capacity(current_batch_size * run_cfg.max_seq_len);
                    let mut all_seq_ids: Vec<i32> = Vec::with_capacity(current_batch_size * run_cfg.max_seq_len);

                    for &i in batch_idx {
                        let sample = &training_samples[i];
                        let input_ids = &sample.input_ids;
                        let target_labels = &sample.target_labels;

                        let input_len = input_ids.iter().position(|&t| t == PAD_TOKEN).unwrap_or(run_cfg.max_seq_len);
                        let target_len = target_labels.iter().position(|&t| t == PAD_TOKEN).unwrap_or(run_cfg.max_seq_len);

                        let mut full_seq = Vec::with_capacity(run_cfg.max_seq_len);
                        full_seq.extend(&input_ids[..input_len]);
                        full_seq.extend(sep_tokens.iter().map(|&t| t as usize));

                        let remaining_space = run_cfg.max_seq_len.saturating_sub(full_seq.len());
                        let actual_target_len = target_len.min(remaining_space);
                        full_seq.extend(&target_labels[..actual_target_len]);
                        full_seq.resize(run_cfg.max_seq_len, PAD_TOKEN);

                        // Loss only on the separator + target tokens, not the prompt.
                        let mut loss_targets = vec![PAD_TOKEN as i32; run_cfg.max_seq_len];

                        let start_predict_idx = input_len.saturating_sub(1);
                        let end_predict_idx = (input_len + sep_len + actual_target_len).saturating_sub(1).min(run_cfg.max_seq_len - 1);

                        for idx in start_predict_idx..end_predict_idx {
                            loss_targets[idx] = full_seq[idx + 1] as i32;
                        }

                        all_seq_ids.extend(full_seq.iter().map(|&t| t as i32));
                        all_lang_targets.extend(loss_targets);
                    }

                    let lang_target_t = Tensor::<TrainBackend, 1, Int>::from_ints(TensorData::new(all_lang_targets, [current_batch_size * run_cfg.max_seq_len]), &device);
                    let tokens_t = Tensor::<TrainBackend, 2, Int>::from_ints(TensorData::new(all_seq_ids, [current_batch_size, run_cfg.max_seq_len]), &device);

                    let token_logits = model.forward(tokens_t.clone());

                    // Entropy
                    let probs = burn::tensor::activation::softmax(token_logits.clone(), 2);
                    let log_probs = (probs.clone() + 1e-10).log();
                    let token_entropy = (probs * log_probs).sum_dim(2).neg().squeeze::<2>();
                    let non_pad_mask = tokens_t.clone().equal_elem(PAD_TOKEN as u32).bool_not().float();
                    let entropy_val: f32 = (token_entropy * non_pad_mask.clone()).sum().div(non_pad_mask.sum()).into_scalar();

                    // Loss
                    let vocab = tokenizer.vocab_size();
                    let logits_2d = token_logits.reshape([current_batch_size * run_cfg.max_seq_len, vocab]);
                    let lang_loss = ce_loss.forward(logits_2d, lang_target_t);

                    let grads = GradientsParams::from_grads(lang_loss.backward(), &model);
                    model = optimizer.step(current_lr, model, grads);

                    let loss_val: f32 = lang_loss.clone().inner().to_data().to_vec::<f32>().unwrap()[0];
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
                        let mut entries = Vec::with_capacity(prompts.len());
                        for p in &prompts {
                            let prompt = build_inference_prompt(stage_cfg.stage, p);
                            let result = inference_model.generate_unmasked_parsed(&tokenizer, &prompt, run_cfg.max_seq_len, &device);
                            let reply = if stage_cfg.stage == TrainingStage::Structured { result.reply } else { result.raw_output };
                            entries.push((p.clone(), reply));
                        }
                        state.last_reply = entries[0].1.clone();
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
                    let mut entries = Vec::with_capacity(prompts.len());
                    for p in &prompts {
                        let prompt = build_inference_prompt(stage_cfg.stage, p);
                        let result = inference_model.generate_unmasked_parsed(&tokenizer, &prompt, run_cfg.max_seq_len, &device);
                        let reply = if stage_cfg.stage == TrainingStage::Structured { result.reply } else { result.raw_output };
                        entries.push((p.clone(), reply));
                    }
                    state.last_reply = entries[0].1.clone();
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
            train_causal_lm::<YumonMoeBrain<TrainBackend>>(&run_cfg, &run_dir, &tokenizer, &keyword_index, &prompts, &device)?;
        }

        Architecture::DecoderOnly => {
            train_causal_lm::<YumonDecBrain<TrainBackend>>(&run_cfg, &run_dir, &tokenizer, &keyword_index, &prompts, &device)?;
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
}

#[cfg(test)]
mod moe_loss_tests {
    use super::*;
    use burn::backend::Wgpu;
    type B = Wgpu;

    #[test]
    fn moe_grid_is_1024_wide_24_layers_four_experts_top1() {
        let runs = generate_run_configs(16, Architecture::Moe { num_experts: 4, top_k: 1 });
        assert_eq!(runs.len(), 1);
        let run = &runs[0];
        assert_eq!((run.embed_dim, run.n_layers, run.attn_heads, run.ff_dim), (1024, 24, 8, 4096));
        assert!(matches!(run.architecture, Architecture::Moe { num_experts: 4, top_k: 1 }));
        assert_eq!(run.stages[0].batch_size, 16);
        assert_eq!(run.name, "1024h_24l_8a_512len_b16_Moe_e4_k1_Language");
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
            // Every sample also trains on its separator.
            let r_trained = r.target_tokens + r.samples * sep_len;
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
/// Variants: dec-balanced, dec-plain, moe-balanced, moe-fusion (the previous
/// setup: fusion wgpu, no checkpointing), attn-flash-{balanced,plain},
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

    fn moe<AD: AutodiffBackend<Device = WgpuDevice>>(label: &str, s: &Shape) {
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
            "moe-fusion" => moe::<Autodiff<burn::backend::Wgpu>>(&variant, &s),
            "attn-flash-balanced" => attn::<Autodiff<Inner, BalancedCheckpointing>>(&variant, true, &s),
            "attn-flash-plain" => attn::<Autodiff<Inner, NoCheckpointing>>(&variant, true, &s),
            "attn-naive-balanced" => attn::<Autodiff<Inner, BalancedCheckpointing>>(&variant, false, &s),
            "attn-naive-plain" => attn::<Autodiff<Inner, NoCheckpointing>>(&variant, false, &s),
            other => panic!("unknown YUMON_PROBE {other}"),
        }
    }
}
