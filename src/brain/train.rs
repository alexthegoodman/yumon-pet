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

// #[cfg(not(target_arch = "wasm32"))]
// use crate::brain::{decoder_model::{BrainDecMetadata, YumonDecBrain, YumonDecBrainConfig}};

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
pub type TrainBackend = burn::backend::Autodiff<burn::backend::Wgpu>;
pub type TrainRuntime = cubecl::wgpu::WgpuRuntime;
// pub type TrainBackend = burn::backend::Autodiff<burn::backend::NdArray<f32>>;

// Used by the headless `train-brain` CLI path (`run`, below) — this is what
// RunPod/Docker actually runs. CUDA talks to the driver directly and needs no
// Vulkan/GL adapter, unlike wgpu, which RunPod's driver stack doesn't expose.
pub type CudaTrainBackend = burn::backend::Autodiff<burn::backend::Cuda>;
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
    let sizes:       [usize; 1] = [
        // 32,
        // 64,
        // 128,
        256, // stretch: ~28M total / ~9.5M active params at 4 layers/8 experts
        // 512
        // 1024
    ];
    let layer_counts: [usize; 1] = [
        // 1,
        // 2,
        // 4,
        // 8,
        // 8, // stretch: slower on iGPU - each MoE layer forces a host readback
        16
    ];
    let head_counts:  [usize; 1] = [
        // 1,
        // 2,
        4,
        // 8,
        // 16,
        // 32
        // 64
    ];
    let seq_lens:     [usize; 1] = [
        32,
        // 64,
        // 128
    ];
    let batch_sizes:     [usize; 1] = [
        // 2,
        // 8
        if matches!(architecture, Architecture::Moe { .. }) { batch_size_option } else { 16 }
        // 32
        // 64
    ];
    // For Moe, num_experts/top_k are swept here as real matrix dimensions
    // instead of the single fixed pair the CLI's --moe-experts/--moe-top-k
    // used to stamp onto every run - those two CLI flags are now unused for
    // architecture=moe (kept only for the other architectures' CLI parsing).
    let moe_expert_configs: [(usize, usize); 4] = [
        // (2, 1),
        (4, 1),
        (4, 2),
        (8, 1),
        (8, 2),
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
                                    Architecture::DecoderOnly    => (3e-4, 3e-5),
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
        // .add("data/ideas.txt",   FileKind::TxtLines, Some(25_000))
        // .add("archive/arena_extract.txt",   FileKind::Chats, Some(25_000))
        // .add("data/distillchatv1.csv",   FileKind::DistillChat, Some(25_000))
        // .add("data/wiki_extract.txt",   FileKind::Txt, Some(250_000))
        // .add("data/bible_bbe.csv", FileKind::BibleCsv, None)
        // .add("data/bible_asv.csv", FileKind::BibleCsv, None)
        // LLM-generated Q&A pairs from src/bin/gen_synthetic_data.rs — proper
        // message/reply splits instead of BibleCsv's arbitrary mid-sentence cuts.
        .add("data/synthetic/bible.txt", FileKind::Chats, None)
        .add("data/synthetic/business.txt", FileKind::Chats, None)
        .add("data/synthetic/universe.txt", FileKind::Chats, None)
        // // .add("data/creative_stories.txt", FileKind::Txt, Some(50_000)) // good but gets split
        // // .add("data/Dictionary/Oxford/Oxford_English_Dictionary.txt",   FileKind::SpecificDict, Some(50_000))
        // // .add("archive/handcrafted_pairs.txt", FileKind::Chats, None);
        // // .add("archive/ov_chats.txt", FileKind::Chats, None)
        // .add("data/The-Office-Lines-V4.csv",   FileKind::DialogueCsv, Some(25_000))
        // .add("data/friends_all_episodes_clean.csv",   FileKind::FriendsCsv, Some(25_000))
        // // .add("archive/ov_chats.txt", FileKind::Chats, None)
        // // .add("archive/ov_chats.txt", FileKind::Chats, None)
        // .add("archive/ov_chats.txt", FileKind::Chats, None)
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

    loader
        // .total_limit(2_000_000)
        .total_limit(5_000_000)
        .seed(4815162342)
        .load(tokenizer, keyword_index, max_seq_len)
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
    let device = burn::backend::wgpu::WgpuDevice::default();
    // let device = burn::backend::cuda::CudaDevice::default(); // for runpod
    let label_keywords   = build_label_keywords();
    let keyword_index    = build_keyword_index(&label_keywords);
    let tokenizer = TokenizerKind::Bpe(BpeTokenizer::load("yumon_bpe")?);

    // ── Configure Runs ──────────────────────────────────────────────────────────
    // Full grid search (see generate_run_configs) instead of a hand-picked list -
    // MIN_EPOCH_LOSS_DROP cuts non-converging configs short, so covering the whole
    // space is affordable.
    let prompts = eval_prompts();
    if matches!(architecture, Architecture::Moe { .. }) {
        anyhow::ensure!(batch_size > 0 && epochs > 0, "batch size and epochs must be positive");
    }
    let mut runs = generate_run_configs(batch_size, architecture);
    if matches!(architecture, Architecture::Moe { .. }) {
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
                    TerminalOptions { viewport: Viewport::Inline(26) },
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
                    TerminalOptions { viewport: Viewport::Inline(26) },
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

        Architecture::Moe { num_experts, top_k } => {

        let config = YumonMoeBrainConfig::new(tokenizer.vocab_size(), run_cfg.stages[0].stage)
            .with_embed_dim(run_cfg.embed_dim).with_hidden_units(run_cfg.hidden_units)
            .with_n_layers(run_cfg.n_layers).with_attn_heads(run_cfg.attn_heads)
            .with_ff_dim(run_cfg.ff_dim).with_max_seq_len(run_cfg.max_seq_len)
            .with_num_experts(num_experts).with_top_k(top_k);
        println!("MoE: {} experts, top-{}; FFN parameters/layer: {} total, {} active/token (excluding router).",
            num_experts, top_k, 3 * run_cfg.embed_dim * run_cfg.ff_dim * num_experts,
            3 * run_cfg.embed_dim * run_cfg.ff_dim * top_k);
        let (mut model, mut epochs_already_done) = if run_dir.join("model.bin").exists() {
            let (m, checkpoint_tokenizer, saved) = YumonMoeBrain::<TrainBackend>::load(run_dir_str, &device)?;
            anyhow::ensure!(saved.vocab_size == config.vocab_size && saved.embed_dim == config.embed_dim
                && saved.ff_dim == config.ff_dim && saved.n_layers == config.n_layers
                && saved.attn_heads == config.attn_heads && saved.max_seq_len == config.max_seq_len
                && saved.num_experts == num_experts && saved.top_k == top_k,
                "MoE checkpoint configuration differs from requested run");
            let current_json = match &tokenizer { TokenizerKind::Bpe(t) => t.inner.to_string(false).map_err(|e| anyhow::anyhow!("{e}"))?, _ => unreachable!() };
            let saved_json = match &checkpoint_tokenizer { TokenizerKind::Bpe(t) => t.inner.to_string(false).map_err(|e| anyhow::anyhow!("{e}"))?, _ => unreachable!() };
            anyhow::ensure!(serde_json::from_str::<serde_json::Value>(&current_json)? == serde_json::from_str::<serde_json::Value>(&saved_json)?,
                "MoE checkpoint tokenizer differs from training tokenizer");
            let meta: MoeMetadata = serde_json::from_str(&std::fs::read_to_string(run_dir.join("metadata.json"))?)?;
            (m, meta.epochs_trained)
        } else { (config.init(&device), 0) };

        'stage_loop_moe: for (stage_idx, stage_cfg) in run_cfg.stages.iter().enumerate() {
            model.config.0.training_stage = stage_cfg.stage;
            println!("\n🔨 Stage {}: {:?}", stage_idx + 1, stage_cfg.stage);

            let training_samples = load_stage_data(stage_cfg.stage.clone(), &tokenizer, &keyword_index, run_cfg.max_seq_len)?;
            anyhow::ensure!(!training_samples.is_empty(), "No training samples for MoE stage");
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
                    TerminalOptions { viewport: Viewport::Inline(26) },
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

            'epoch_loop_moe: for epoch in 0..stage_cfg.epochs {
                state.epoch = epoch + 1;
                let mut idx: Vec<usize> = (0..training_samples.len()).collect();
                idx.shuffle(&mut rng);
                let num_batches = idx.len().div_ceil(stage_cfg.batch_size);
                let mut epoch_loss = 0.0f32;
                let mut processed_batches = 0usize;

                // Decoder-only style: [prompt][separator][reply] packed into one causal sequence
                let sep_text = if stage_cfg.stage == TrainingStage::Structured { "\n---\n" } else { " " };
                let sep_tokens = tokenizer.encode(sep_text);
                let sep_len = sep_tokens.len();
                anyhow::ensure!(sep_len + 2 <= run_cfg.max_seq_len, "Sequence too short for separator and reply");

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

                        let input_len = input_ids.iter().position(|&t| t == PAD_TOKEN).unwrap_or(input_ids.len())
                            .min(run_cfg.max_seq_len - sep_len - 1);
                        let target_len = target_labels.iter().position(|&t| t == PAD_TOKEN).unwrap_or(target_labels.len());

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

                    if all_lang_targets.iter().all(|&id| id == PAD_TOKEN as i32) { continue; }
                    let lang_target_t = Tensor::<TrainBackend, 1, Int>::from_ints(TensorData::new(all_lang_targets, [current_batch_size * run_cfg.max_seq_len]), &device);
                    let tokens_t = Tensor::<TrainBackend, 2, Int>::from_ints(TensorData::new(all_seq_ids, [current_batch_size, run_cfg.max_seq_len]), &device);

                    let (token_logits, aux_loss, expert_counts) = model.forward_with_aux(tokens_t.clone());

                    // Entropy
                    let probs = burn::tensor::activation::softmax(token_logits.clone(), 2);
                    let log_probs = (probs.clone() + 1e-10).log();
                    let token_entropy = (probs * log_probs).sum_dim(2).neg().squeeze_dim::<2>(2);
                    let non_pad_mask = tokens_t.clone().equal_elem(PAD_TOKEN as u32).bool_not().float();
                    let entropy_val: f32 = (token_entropy * non_pad_mask.clone()).sum().div(non_pad_mask.sum()).into_scalar();

                    // Loss
                    let vocab = tokenizer.vocab_size();
                    let logits_2d = token_logits.reshape([current_batch_size * run_cfg.max_seq_len, vocab]);
                    let lang_loss = ce_loss.forward(logits_2d, lang_target_t);

                    let total_loss = lang_loss.clone() + aux_loss.clone();
                    let grads = GradientsParams::from_grads(total_loss.backward(), &model);
                    model = optimizer.step(current_lr, model, grads);

                    let loss_val: f32 = lang_loss.clone().inner().to_data().to_vec::<f32>().unwrap()[0];
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

                    // if batch_num % 100 == 0 {
                    //     println!("MoE auxiliary loss {:.5}; dispatched rows per layer/expert: {:?}", aux_loss.into_scalar(), expert_counts);
                    // }

                    // Periodic save and inference every 500 batches
                    if (batch_num + 1) % 500 == 0 {
                        let current_final_loss = epoch_loss / processed_batches as f32;
                        let meta = MoeMetadata {
                            num_experts, top_k,
                            aux_loss_weight: model.config.aux_loss_weight,
                            z_loss_weight: model.config.z_loss_weight,
                            dropout_rate: model.config.dropout_rate,
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
                        break;
                    }
                }
                anyhow::ensure!(processed_batches > 0, "MoE epoch contains no supervised tokens");
                final_loss = epoch_loss / processed_batches as f32;

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

                let meta = MoeMetadata {
                            num_experts, top_k,
                            aux_loss_weight: model.config.aux_loss_weight,
                            z_loss_weight: model.config.z_loss_weight,
                            dropout_rate: model.config.dropout_rate,
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

                if run_should_stop || final_loss < stage_cfg.loss_threshold { break 'epoch_loop_moe; }
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
                break 'stage_loop_moe;
            }
        }

        } // Architecture::Moe

        // Architecture::DecoderOnly => {

        // let (mut model, mut epochs_already_done) = if std::path::Path::new(run_dir_str).join("model.bin").exists() {
        //     match YumonDecBrain::<TrainBackend>::load(run_dir_str, &device) {
        //         Ok((m, _tok, _config)) => {
        //             let meta_json = std::fs::read_to_string(std::path::Path::new(run_dir_str).join("metadata.json"))?;
        //             let meta: BrainDecMetadata = serde_json::from_str(&meta_json)?;
        //             println!("▶️  Resuming run {} from checkpoint ({} epochs done, loss={:.4})",
        //                      run_cfg.name, meta.epochs_trained, meta.final_loss);
        //             (m, meta.epochs_trained)
        //         }
        //         Err(_) => {
        //             let config = YumonDecBrainConfig {
        //                 vocab_size: tokenizer.vocab_size(),
        //                 embed_dim: run_cfg.embed_dim,
        //                 hidden_units: run_cfg.hidden_units,
        //                 n_layers: run_cfg.n_layers,
        //                 attn_heads: run_cfg.attn_heads,
        //                 ff_dim: run_cfg.ff_dim,
        //                 max_seq_len: run_cfg.max_seq_len,
        //                 training_stage: run_cfg.stages.get(0).expect("Couldn't get stage").stage,
        //                 dropout_rate: 0.05,
        //             };
        //             (config.init(&device), 0)
        //         }
        //     }
        // } else {
        //     println!("🆕 Starting fresh run: {}", run_cfg.name);
        //     let config = YumonDecBrainConfig {
        //         vocab_size: tokenizer.vocab_size(),
        //         embed_dim: run_cfg.embed_dim,
        //         hidden_units: run_cfg.hidden_units,
        //         n_layers: run_cfg.n_layers,
        //         attn_heads: run_cfg.attn_heads,
        //         ff_dim: run_cfg.ff_dim,
        //         max_seq_len: run_cfg.max_seq_len,
        //         training_stage: run_cfg.stages.get(0).expect("Couldn't get stage").stage,
        //         dropout_rate: 0.05,
        //     };
        //     (config.init(&device), 0)
        // };

        // 'stage_loop_dec: for (stage_idx, stage_cfg) in run_cfg.stages.iter().enumerate() {
        //     println!("\n🔨 Stage {}: {:?}", stage_idx + 1, stage_cfg.stage);

        //     let training_samples = load_stage_data(stage_cfg.stage.clone(), &tokenizer, &keyword_index, run_cfg.max_seq_len)?;
        //     println!("Training samples: {}", training_samples.len());

        //     // debug print — first 12 samples
        //     for (i, sample) in training_samples.iter().enumerate() {
        //         if i >= 12 { break; }
        //         println!("INPUT:  {:?}", tokenizer.decode(&sample.input_ids));
        //         println!("TARGET: {:?}", tokenizer.decode(
        //             &sample.target_labels.iter()
        //                 .map(|&t| if t == PAD_TOKEN { PAD_TOKEN } else { t })
        //                 .collect::<Vec<_>>()
        //         ));
        //         println!("input_len:     {}", sample.input_ids.iter().filter(|&&t| t != PAD_TOKEN).count());
        //         println!("target_active: {}", sample.target_labels.iter().filter(|&&t| t != PAD_TOKEN).count());
        //     }

        //     let mut optimizer = AdamWConfig::new()
        //         .with_epsilon(stage_cfg.epsilon)
        //         .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        //         .with_weight_decay(stage_cfg.weight_decay)
        //         .init();

        //     let ce_loss = CrossEntropyLossConfig::new()
        //         .with_pad_tokens(Some(vec![PAD_TOKEN as usize]))
        //         .with_smoothing(Some(stage_cfg.smoothing))
        //         .init(&device);

        //     let mut rng = rand::thread_rng();
        //     use std::io::{stdout, IsTerminal};
        //     let mut terminal = if stdout().is_terminal() {
        //         let backend = CrosstermBackend::new(stdout());
        //         Some(Terminal::with_options(
        //             backend,
        //             TerminalOptions { viewport: Viewport::Inline(26) },
        //         )?)
        //     } else {
        //         None
        //     };

        //     let total_batches = training_samples.len() / stage_cfg.batch_size;
        //     let mut state = TrainingState {
        //         loss_history: vec![],
        //         avg_loss_history: vec![],
        //         current_loss: 0.0,
        //         avg_loss: 0.0,
        //         epoch: 0,
        //         total_epochs: stage_cfg.epochs,
        //         batch: 0,
        //         total_batches: total_batches,
        //         current_lr: stage_cfg.first_lr,
        //         lr_history: vec![],
        //         global_step: 0,
        //         entropy: 0.0,
        //         entropy_history: vec![],
        //         last_reply: String::new()
        //     };

        //     let mut final_loss = 0.0f32;
        //     let inference_log_path = format!("{}/{}_inference_log.txt", run_dir_str, run_cfg.name);
        //     let chart_path = format!("{}/{}_stage_{}.png", run_dir_str, run_cfg.name, stage_idx + 1);
        //     let mut prev_epoch_loss: Option<f32> = None;
        //     let mut run_should_stop = false;

        //     'epoch_loop_dec: for epoch in 0..stage_cfg.epochs {
        //         state.epoch = epoch + 1;
        //         let mut idx: Vec<usize> = (0..training_samples.len()).collect();
        //         idx.shuffle(&mut rng);
        //         let num_batches = idx.len().max(1) / stage_cfg.batch_size;
        //         let mut epoch_loss = 0.0f32;

        //         // --- Decoder-only style: [prompt][separator][reply] packed into one causal sequence
        //         let sep_text = if stage_cfg.stage == TrainingStage::Structured { "\n---\n" } else { " " };
        //         let sep_tokens = tokenizer.encode(sep_text);
        //         let sep_len = sep_tokens.len();

        //         for batch_num in 0..num_batches {
        //             let current_lr = {
        //                 let total_steps = stage_cfg.epochs * num_batches;
        //                 let step = epoch * num_batches + batch_num;
        //                 let t = step as f64 / total_steps as f64;
        //                 (stage_cfg.first_lr * (1.0 - t) + stage_cfg.last_lr * t)
        //             };

        //             let batch_start = batch_num * stage_cfg.batch_size;
        //             let batch_end = (batch_start + stage_cfg.batch_size).min(training_samples.len());
        //             let batch_idx = &idx[batch_start..batch_end];
        //             let current_batch_size = batch_idx.len();
        //             if current_batch_size == 0 { continue; }

        //             let mut all_lang_targets: Vec<i32> = Vec::with_capacity(current_batch_size * run_cfg.max_seq_len);
        //             let mut all_seq_ids: Vec<i32> = Vec::with_capacity(current_batch_size * run_cfg.max_seq_len);

        //             for &i in batch_idx {
        //                 let sample = &training_samples[i];
        //                 let input_ids = &sample.input_ids;
        //                 let target_labels = &sample.target_labels;

        //                 // Find actual length of input (up to PAD)
        //                 let input_len = input_ids.iter().position(|&t| t == PAD_TOKEN).unwrap_or(run_cfg.max_seq_len);
        //                 let target_len = target_labels.iter().position(|&t| t == PAD_TOKEN).unwrap_or(run_cfg.max_seq_len);

        //                 // Construct single sequence: [Input] [Separator] [Target]
        //                 let mut full_seq = Vec::with_capacity(run_cfg.max_seq_len);
        //                 full_seq.extend(&input_ids[..input_len]);
        //                 full_seq.extend(sep_tokens.iter().map(|&t| t as usize));

        //                 let remaining_space = run_cfg.max_seq_len.saturating_sub(full_seq.len());
        //                 let actual_target_len = target_len.min(remaining_space);
        //                 full_seq.extend(&target_labels[..actual_target_len]);
        //                 full_seq.resize(run_cfg.max_seq_len, PAD_TOKEN);

        //                 // Targets for loss: shifted left by 1. We only want loss on the
        //                 // separator + target tokens, not the prompt tokens.
        //                 let mut loss_targets = vec![PAD_TOKEN as i32; run_cfg.max_seq_len];

        //                 let start_predict_idx = input_len.saturating_sub(1);
        //                 let end_predict_idx = (input_len + sep_len + actual_target_len).saturating_sub(1).min(run_cfg.max_seq_len - 1);

        //                 for idx in start_predict_idx..end_predict_idx {
        //                     loss_targets[idx] = full_seq[idx + 1] as i32;
        //                 }

        //                 all_seq_ids.extend(full_seq.iter().map(|&t| t as i32));
        //                 all_lang_targets.extend(loss_targets);
        //             }

        //             let lang_target_t = Tensor::<TrainBackend, 1, Int>::from_ints(TensorData::new(all_lang_targets, [current_batch_size * run_cfg.max_seq_len]), &device);
        //             let tokens_t = Tensor::<TrainBackend, 2, Int>::from_ints(TensorData::new(all_seq_ids, [current_batch_size, run_cfg.max_seq_len]), &device);

        //             let token_logits = model.forward::<TrainRuntime>(tokens_t.clone());

        //             // Entropy
        //             let probs = burn::tensor::activation::softmax(token_logits.clone(), 2);
        //             let log_probs = (probs.clone() + 1e-10).log();
        //             let token_entropy = (probs * log_probs).sum_dim(2).neg().squeeze::<2>();
        //             let non_pad_mask = tokens_t.clone().equal_elem(PAD_TOKEN as u32).bool_not().float();
        //             let entropy_val: f32 = (token_entropy * non_pad_mask.clone()).sum().div(non_pad_mask.sum()).into_scalar();

        //             // Loss
        //             let vocab = tokenizer.vocab_size();
        //             let logits_2d = token_logits.reshape([current_batch_size * run_cfg.max_seq_len, vocab]);
        //             let lang_loss = ce_loss.forward(logits_2d, lang_target_t);

        //             let grads = GradientsParams::from_grads(lang_loss.backward(), &model);
        //             model = optimizer.step(current_lr, model, grads);

        //             let loss_val: f32 = lang_loss.clone().inner().to_data().to_vec::<f32>().unwrap()[0];
        //             epoch_loss += loss_val;

        //             state.entropy = entropy_val;
        //             state.current_loss = loss_val;
        //             state.avg_loss = epoch_loss / (batch_num + 1) as f32;
        //             state.batch = batch_num + 1;
        //             state.current_lr = current_lr;
        //             state.global_step += 1;
        //             state.loss_history.push((state.global_step as f64, loss_val as f64));
        //             state.avg_loss_history.push((state.global_step as f64, state.avg_loss as f64));
        //             state.entropy_history.push((state.global_step as f64, entropy_val as f64));
        //             state.lr_history.push((state.global_step as f64, current_lr));

        //             if let Some(term) = terminal.as_mut() {
        //                 term.draw(|frame| render(frame, &state))?;
        //             } else if state.global_step % 50 == 0 || batch_num + 1 == num_batches {
        //                 println!(
        //                     "epoch {}/{} batch {}/{} loss {:.4} avg {:.4} lr {:.2e} entropy {:.4}",
        //                     state.epoch, state.total_epochs, state.batch, state.total_batches,
        //                     state.current_loss, state.avg_loss, state.current_lr, state.entropy,
        //                 );
        //             }

        //             // Periodic save and inference every 500 batches
        //             if (batch_num + 1) % 500 == 0 {
        //                 let current_final_loss = epoch_loss / (batch_num + 1) as f32;
        //                 let meta = BrainDecMetadata {
        //                     vocab_size:     tokenizer.vocab_size(),
        //                     epochs_trained: epochs_already_done + epoch, // Partial epoch progress
        //                     final_loss:     current_final_loss,
        //                     batch_size:     stage_cfg.batch_size,
        //                     training_stage: stage_cfg.stage.clone(),
        //                     embed_dim:      run_cfg.embed_dim,
        //                     hidden_units:   run_cfg.hidden_units,
        //                     n_layers:       run_cfg.n_layers,
        //                     attn_heads:     run_cfg.attn_heads,
        //                     ff_dim:         run_cfg.ff_dim,
        //                     max_seq_len:    run_cfg.max_seq_len,
        //                 };
        //                 model.save(run_dir_str, &tokenizer, &meta)?;

        //                 // Periodic inference — 5 prompts, logged (appended) and the
        //                 // loss chart re-saved (overwritten) right away.
        //                 let inference_model = model.valid();
        //                 let mut entries = Vec::with_capacity(prompts.len());
        //                 for p in &prompts {
        //                     let prompt = build_inference_prompt(stage_cfg.stage, p);
        //                     let result = inference_model.generate_unmasked_parsed::<TrainRuntime>(&tokenizer, &prompt, run_cfg.max_seq_len, &device);
        //                     let reply = if stage_cfg.stage == TrainingStage::Structured { result.reply } else { result.raw_output };
        //                     entries.push((p.clone(), reply));
        //                 }
        //                 state.last_reply = entries[0].1.clone();
        //                 if let Err(e) = append_inference_log(&inference_log_path, &run_cfg.name, stage_idx, stage_cfg.stage, state.epoch, state.total_epochs, state.batch, state.total_batches, state.avg_loss, &entries) {
        //                     eprintln!("⚠️  Failed to append inference log: {}", e);
        //                 }
        //                 if let Err(e) = state.save_chart_image(&chart_path) {
        //                     eprintln!("⚠️  Failed to save chart image: {}", e);
        //                 }
        //             }

        //             // Loss Threshold Exit
        //             if state.avg_loss < stage_cfg.loss_threshold {
        //                 println!("\n🎯 Loss Threshold Reached: {:.4} < {:.4}. Ending Stage.", state.avg_loss, stage_cfg.loss_threshold);
        //                 final_loss = state.avg_loss;
        //                 break 'epoch_loop_dec;
        //             }
        //         }
        //         final_loss = epoch_loss / num_batches.max(1) as f32;

        //         // Automatic run-finish: if this epoch didn't drop avg loss by at
        //         // least MIN_EPOCH_LOSS_DROP versus the previous epoch, this config
        //         // isn't converging fast enough to be worth the remaining epochs —
        //         // finish the run here and move on to the next RunConfig.
        //         if let Some(prev) = prev_epoch_loss {
        //             if prev - final_loss < MIN_EPOCH_LOSS_DROP {
        //                 println!(
        //                     "\n⏹️  Epoch loss drop {:.4} < {:.4} (prev {:.4} -> {:.4}). Finishing run early.",
        //                     prev - final_loss, MIN_EPOCH_LOSS_DROP, prev, final_loss,
        //                 );
        //                 run_should_stop = true;
        //             }
        //         }
        //         prev_epoch_loss = Some(final_loss);

        //         // Save checkpoint after each epoch
        //         let meta = BrainDecMetadata {
        //             vocab_size:     tokenizer.vocab_size(),
        //             epochs_trained: epochs_already_done + epoch + 1,
        //             final_loss,
        //             batch_size: stage_cfg.batch_size,
        //             training_stage: stage_cfg.stage.clone(),
        //             embed_dim: run_cfg.embed_dim,
        //             hidden_units: run_cfg.hidden_units,
        //             n_layers: run_cfg.n_layers,
        //             attn_heads: run_cfg.attn_heads,
        //             ff_dim: run_cfg.ff_dim,
        //             max_seq_len: run_cfg.max_seq_len,
        //         };
        //         model.save(run_dir_str, &tokenizer, &meta)?;

        //         // periodic inference — same 5-prompt log + chart save as above
        //         {
        //             let inference_model = model.valid();
        //             let mut entries = Vec::with_capacity(prompts.len());
        //             for p in &prompts {
        //                 let prompt = build_inference_prompt(stage_cfg.stage, p);
        //                 let result = inference_model.generate_unmasked_parsed::<TrainRuntime>(&tokenizer, &prompt, run_cfg.max_seq_len, &device);
        //                 let reply = if stage_cfg.stage == TrainingStage::Structured { result.reply } else { result.raw_output };
        //                 entries.push((p.clone(), reply));
        //             }
        //             state.last_reply = entries[0].1.clone();
        //             if let Err(e) = append_inference_log(&inference_log_path, &run_cfg.name, stage_idx, stage_cfg.stage, state.epoch, state.total_epochs, state.batch, state.total_batches, state.avg_loss, &entries) {
        //                 eprintln!("⚠️  Failed to append inference log: {}", e);
        //             }
        //             if let Err(e) = state.save_chart_image(&chart_path) {
        //                 eprintln!("⚠️  Failed to save chart image: {}", e);
        //             }
        //         }

        //         if run_should_stop { break 'epoch_loop_dec; }
        //     }
        //     epochs_already_done += state.epoch;
        //     if let Some(term) = terminal.as_mut() {
        //         term.clear()?;
        //     }

        //     // Final chart save as a safety net (the periodic saves above already
        //     // keep this path current throughout training).
        //     if let Err(e) = state.save_chart_image(&chart_path) {
        //         eprintln!("⚠️  Failed to save chart image: {}", e);
        //     } else {
        //         println!("📊 Chart saved to {}", chart_path);
        //     }

        //     println!("✅ Stage complete. Final loss: {:.4}", final_loss);

        //     if run_should_stop {
        //         println!("⏹️  Run {} finished early, skipping remaining stages.", run_cfg.name);
        //         break 'stage_loop_dec;
        //     }
        // }

        // } // Architecture::DecoderOnly
        _ => {},
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