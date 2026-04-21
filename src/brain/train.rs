/// Yumon Brain training loop

use anyhow::Result;
use burn::{
    grad_clipping::GradientClippingConfig, module::AutodiffModule, nn::loss::CrossEntropyLossConfig, optim::{AdamConfig, AdamWConfig, GradientsParams, Optimizer}, prelude::*, tensor::{Int, TensorData, backend::AutodiffBackend}
};
#[cfg(target_os = "windows")]
use rand::{Rng, rngs::StdRng, seq::SliceRandom, thread_rng};
#[cfg(target_os = "windows")]
use indicatif::{ProgressBar, ProgressStyle};
#[cfg(target_os = "windows")]
use ratatui::{Terminal, TerminalOptions, Viewport, prelude::CrosstermBackend};
use std::collections::HashMap;
#[cfg(target_os = "windows")]
use rand::SeedableRng;

use crate::{brain::{PAD_TOKEN, bpe::{BpeTokenizer, CL_ID, CR_ID, TokenizerKind}, chart::{TrainingState}, loader::{DataLoader, FileKind}, samples::{TrainingStage, WorldContext, prepare_paired_samples_split, prepare_paired_samples_split_sep}}, vision::{CIFAR_CLASSES, EMOTE_CLASSES, EMOTE_NAMES}};

#[cfg(target_os = "windows")]
use crate::brain::chart::render;
#[cfg(target_os = "windows")]
use crate::brain::mdx::{load_dictionary_sentences, load_handcrafted_sentences, load_qa_pairs, load_qa_singles, load_txt_sentences};
#[cfg(target_os = "windows")]
use crate::brain::pdf::{load_pdf_ebook_sentences, load_pdfs};
#[cfg(target_os = "windows")]
use crate::brain::wiki::{save_sentence_pairs_to_file};

use crate::brain::{
    // CONTEXT_DIMS,
    tokenizer::{Tokenizer, BOS_TOKEN, EOS_TOKEN},
    model::{YumonBrain, YumonBrainConfig, BrainMetadata, GenerationResult},
};

pub type TrainBackend = burn::backend::Autodiff<burn::backend::Wgpu>;
// pub type TrainBackend = burn::backend::Autodiff<burn::backend::NdArray<f32>>;

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

pub struct RunConfig {
    pub name: String,
    pub embed_dim: usize,
    pub hidden_units: usize,
    pub n_layers: usize,
    pub attn_heads: usize,
    pub ff_dim: usize,
    pub max_seq_len: usize,
    pub stages: Vec<StageConfig>,
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
        .add("data/arena_extract.txt",   FileKind::Chats, Some(10_000))
        .add("data/distillchatv1.csv",   FileKind::DistillChat, Some(50_000))
        // // .add("data/wiki_extract.txt",   FileKind::Txt, None)
        // .add("data/creative_stories.txt", FileKind::Txt, None)
        // // .add("data/Dictionary/Oxford/Oxford_English_Dictionary.txt",   FileKind::SpecificDict, None)
        // .add("archive/handcrafted_pairs.txt", FileKind::Chats, None);
        .add("archive/ov_chats.txt", FileKind::Chats, None)
        .add("archive/ov_chats.txt", FileKind::Chats, None)
        .add("archive/ov_chats.txt", FileKind::Chats, None)
        .add("archive/you_chats.txt", FileKind::Chats, None)
        .add("archive/you_chats.txt", FileKind::Chats, None)
        .add("archive/you_chats.txt", FileKind::Chats, None)
        .add("archive/clean_chats.txt", FileKind::Chats, None)
        .add("archive/clean_chats.txt", FileKind::Chats, None)
        .add("archive/clean_chats.txt", FileKind::Chats, None);
        // .add(vec![
        //         "data/ebooks/faa-h-8083-25c.pdf".to_string(),
        //         "data/ebooks/algor_intro.pdf".to_string(),
        //         "data/ebooks/intro_engineer.pdf".to_string(),
        //         "data/ebooks/meap.pdf".to_string(),
        //         "data/ebooks/missiles.pdf".to_string(),
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
        .total_limit(100_000)
        .seed(4815162342)
        .load(tokenizer, keyword_index, max_seq_len)
}

#[cfg(target_os = "windows")]
pub fn run(
    wiki_xml:          &str,
    _vision_checkpoint: &str,  // reserved for future real-image fine-tuning
    out_dir:           &str,
    epochs:            usize,
    batch_size:        usize,
    max_articles:      usize,
) -> Result<()> {
    let device = burn::backend::wgpu::WgpuDevice::default();
    let label_keywords   = build_label_keywords();
    let keyword_index    = build_keyword_index(&label_keywords);
    let tokenizer = TokenizerKind::Bpe(BpeTokenizer::load("yumon_bpe")?);

    // ── Configure Runs ──────────────────────────────────────────────────────────
    // Configurations explore three axes: model size (tiny→large), depth (shallow→deep),
    // and training aggressiveness (fast→careful). Names reflect the dominant characteristic.
    let runs = vec![

        // does not learn
        // RunConfig {
        //     name: "64h_2l_2a_180len".to_string(),
        //     embed_dim: 64, 
        //     hidden_units: 64, 
        //     n_layers: 2, 
        //     attn_heads: 2, 
        //     ff_dim: 256, 
        //     max_seq_len: 180,
        //     stages: vec![
        //         StageConfig { stage: TrainingStage::Language,   loss_threshold: 0.1, epochs: 10, batch_size, first_lr: 3e-5, last_lr: 1e-7, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
        //         StageConfig { stage: TrainingStage::Structured, loss_threshold: 0.22, epochs: 10, batch_size, first_lr: 3e-5, last_lr: 1e-7, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
        //     ],
        // },
        
        // memorizes extremely well. outputs memorized sentences regardless of input prompt though, not usually relevant to input prompt
        // RunConfig {
        //     name: "128h_2l_2a_180len".to_string(),
        //     embed_dim: 128, 
        //     hidden_units: 128, 
        //     n_layers: 2, 
        //     attn_heads: 2, 
        //     ff_dim: 512, 
        //     // max_seq_len: 200,
        //     max_seq_len: 180,
        //     stages: vec![
        //         StageConfig { stage: TrainingStage::Language,   loss_threshold: 0.05, epochs: 3, batch_size, first_lr: 1e-3, last_lr: 1e-7, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
        //         StageConfig { stage: TrainingStage::Structured, loss_threshold: 0.1, epochs: 3, batch_size, first_lr: 1e-3, last_lr: 1e-7, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
        //     ],
        // },

        // memorizes little, outputs odd, slightly garbled responses that are somewhat relevant to the input prompt
        // RunConfig {
        //     name: "256h_2l_4a_180len".to_string(),
        //     embed_dim: 256, 
        //     hidden_units: 256, 
        //     n_layers: 2, 
        //     attn_heads: 4, 
        //     ff_dim: 1024,
        //     max_seq_len: 180,
        //     // max_seq_len: 600,
        //     stages: vec![
        //         StageConfig { stage: TrainingStage::Language,   loss_threshold: 0.05, epochs: 12, batch_size, first_lr: 1e-4, last_lr: 1e-6, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
        //         StageConfig { stage: TrainingStage::Structured, loss_threshold: 0.1, epochs: 12, batch_size, first_lr: 1e-4, last_lr: 1e-6, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
        //     ],
        // },

        // trying with better data now for generalization and relevancy
        RunConfig {
            name: "512h_1l_8a_180len".to_string(),
            embed_dim: 512, 
            hidden_units: 512, 
            n_layers: 3,
            attn_heads: 8, 
            ff_dim: 2048, 
            max_seq_len: 180,
            stages: vec![
                StageConfig { stage: TrainingStage::Language,   loss_threshold: 0.05, epochs: 3, batch_size, first_lr: 1e-3, last_lr: 1e-7, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
                StageConfig { stage: TrainingStage::Structured, loss_threshold: 0.1, epochs: 3, batch_size, first_lr: 1e-3, last_lr: 1e-7, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
            ],
        },

        // deeper for complexity, wider doesnt help
        // RunConfig {
        //     name: "256h_12l_4a_180len".to_string(),
        //     embed_dim: 256, 
        //     hidden_units: 256, 
        //     n_layers: 12,
        //     attn_heads: 4, 
        //     ff_dim: 1024,
        //     max_seq_len: 180,
        //     stages: vec![
        //         StageConfig { stage: TrainingStage::Language,   loss_threshold: 0.05, epochs: 12, batch_size, first_lr: 1e-4, last_lr: 1e-6, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
        //         StageConfig { stage: TrainingStage::Structured, loss_threshold: 0.1, epochs: 12, batch_size, first_lr: 1e-4, last_lr: 1e-6, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
        //     ],
        // },

        // memorizes little, very similar to 256, maybe slightly more relevant, but overall about the same
        // RunConfig {
        //     name: "1024h_8l_16a_180len".to_string(),
        //     embed_dim: 1024, 
        //     hidden_units: 1024, 
        //     n_layers: 8, 
        //     attn_heads: 16, 
        //     ff_dim: 2048, 
        //     max_seq_len: 180,
        //     stages: vec![
        //         StageConfig { stage: TrainingStage::Language,   loss_threshold: 0.1, epochs: 10, batch_size, first_lr: 3e-5, last_lr: 1e-7, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
        //         StageConfig { stage: TrainingStage::Structured, loss_threshold: 0.22, epochs: 10, batch_size, first_lr: 3e-5, last_lr: 1e-7, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
        //     ],
        // },

        // RunConfig {
        //     name: "384h_4l_6a_160len".to_string(),
        //     embed_dim: 384, 
        //     hidden_units: 384, 
        //     n_layers: 4, 
        //     attn_heads: 6, 
        //     ff_dim: 1536, 
        //     max_seq_len: 160,
        //     stages: vec![
        //         StageConfig { stage: TrainingStage::Language,   loss_threshold: 0.05, epochs: 2, batch_size, first_lr: 1e-4, last_lr: 1e-6, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
        //         StageConfig { stage: TrainingStage::Structured, loss_threshold: 0.1, epochs: 2, batch_size, first_lr: 1e-4, last_lr: 1e-6, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
        //     ],
        // },

        // RunConfig {
        //     name: "512h_4l_8a_140len".to_string(),
        //     embed_dim: 512, 
        //     hidden_units: 512, 
        //     n_layers: 4, 
        //     attn_heads: 8, 
        //     ff_dim: 2048, 
        //     max_seq_len: 140,
        //     stages: vec![
        //         StageConfig { stage: TrainingStage::Language,   loss_threshold: 0.05, epochs: 2, batch_size, first_lr: 5e-5, last_lr: 1e-7, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
        //         StageConfig { stage: TrainingStage::Structured, loss_threshold: 0.1, epochs: 2, batch_size, first_lr: 5e-5, last_lr: 1e-7, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
        //     ],
        // },

        // // ── Small model, deep — tests whether more layers help at small width ─
        // RunConfig {
        //     name: "small_deep_balanced".to_string(),
        //     embed_dim: 256, hidden_units: 256, n_layers: 6, attn_heads: 4, ff_dim: 1024, max_seq_len: 320,
        //     stages: vec![
        //         StageConfig { stage: TrainingStage::Language,   loss_threshold: 0.1, epochs: 10, batch_size, first_lr: 1e-4, last_lr: 1e-6, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
        //         StageConfig { stage: TrainingStage::Structured, loss_threshold: 0.22, epochs: 10, batch_size, first_lr: 1e-4, last_lr: 1e-6, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
        //     ],
        // },

        // // ── Medium model, moderate training ────────────────────────────────
        // RunConfig {
        //     name: "medium_balanced".to_string(),
        //     embed_dim: 384, hidden_units: 384, n_layers: 4, attn_heads: 6, ff_dim: 1536, max_seq_len: 320,
        //     stages: vec![
        //         StageConfig { stage: TrainingStage::Language,   loss_threshold: 0.1, epochs: 10, batch_size, first_lr: 5e-5, last_lr: 1e-7, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
        //         StageConfig { stage: TrainingStage::Structured, loss_threshold: 0.22, epochs: 10, batch_size, first_lr: 5e-5, last_lr: 1e-7, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
        //     ],
        // },

        // // ── Medium model, high weight decay — tests regularisation ──────────
        // RunConfig {
        //     name: "medium_high_wd".to_string(),
        //     embed_dim: 384, hidden_units: 384, n_layers: 4, attn_heads: 6, ff_dim: 1536, max_seq_len: 320,
        //     stages: vec![
        //         StageConfig { stage: TrainingStage::Language,   loss_threshold: 0.1, epochs: 10, batch_size, first_lr: 5e-5, last_lr: 1e-7, weight_decay: 0.1,  epsilon: 1e-7, smoothing: 0.1 },
        //         StageConfig { stage: TrainingStage::Structured, loss_threshold: 0.22, epochs: 10, batch_size, first_lr: 5e-5, last_lr: 1e-7, weight_decay: 0.1,  epsilon: 1e-7, smoothing: 0.1 },
        //     ],
        // },

        // // ── Medium model, label smoothing sweep — tests soft targets ────────
        // RunConfig {
        //     name: "medium_smooth02".to_string(),
        //     embed_dim: 384, hidden_units: 384, n_layers: 4, attn_heads: 6, ff_dim: 1536, max_seq_len: 320,
        //     stages: vec![
        //         StageConfig { stage: TrainingStage::Language,   loss_threshold: 0.1, epochs: 10, batch_size, first_lr: 5e-5, last_lr: 1e-7, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.2 },
        //         StageConfig { stage: TrainingStage::Structured, loss_threshold: 0.22, epochs: 10, batch_size, first_lr: 5e-5, last_lr: 1e-7, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.2 },
        //     ],
        // },

        // // ── Large model, moderate training ─────────────────────────────────
        // RunConfig {
        //     name: "large_balanced".to_string(),
        //     embed_dim: 512, hidden_units: 512, n_layers: 4, attn_heads: 8, ff_dim: 2048, max_seq_len: 320,
        //     stages: vec![
        //         StageConfig { stage: TrainingStage::Language,   loss_threshold: 0.1, epochs: 10, batch_size, first_lr: 5e-5, last_lr: 1e-7, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
        //         StageConfig { stage: TrainingStage::Structured, loss_threshold: 0.22, epochs: 10, batch_size, first_lr: 5e-5, last_lr: 1e-7, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
        //     ],
        // },

        // // ── Large model, deep — high capacity ceiling ───────────────────────
        // RunConfig {
        //     name: "large_deep".to_string(),
        //     embed_dim: 512, hidden_units: 512, n_layers: 8, attn_heads: 8, ff_dim: 2048, max_seq_len: 320,
        //     stages: vec![
        //         StageConfig { stage: TrainingStage::Language,   loss_threshold: 0.1, epochs: 10, batch_size, first_lr: 3e-5, last_lr: 1e-7, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
        //         StageConfig { stage: TrainingStage::Structured, loss_threshold: 0.22, epochs: 10, batch_size, first_lr: 3e-5, last_lr: 1e-7, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
        //     ],
        // },

        // // ── Long context — tests seq-len sensitivity at medium size ─────────
        // // RunConfig {
        // //     name: "medium_long_ctx".to_string(),
        // //     embed_dim: 384, hidden_units: 384, n_layers: 4, attn_heads: 6, ff_dim: 1536, max_seq_len: 512,
        // //     stages: vec![
        // //         StageConfig { stage: TrainingStage::Language,   loss_threshold: 0.08, epochs: 12, batch_size, first_lr: 5e-5, last_lr: 1e-7, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
        // //         StageConfig { stage: TrainingStage::Structured, loss_threshold: 0.04, epochs: 12, batch_size, first_lr: 5e-5, last_lr: 1e-7, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
        // //     ],
        // // },

        // // ── Careful/slow: small model, long training, low lr ───────────────
        // RunConfig {
        //     name: "small_deep_careful".to_string(),
        //     embed_dim: 256, hidden_units: 256, n_layers: 4, attn_heads: 4, ff_dim: 1024, max_seq_len: 256,
        //     stages: vec![
        //         StageConfig { stage: TrainingStage::Language,   loss_threshold: 0.1, epochs: 20, batch_size, first_lr: 1e-5, last_lr: 1e-8, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
        //         StageConfig { stage: TrainingStage::Structured, loss_threshold: 0.22, epochs: 20, batch_size, first_lr: 1e-5, last_lr: 1e-8, weight_decay: 0.01, epsilon: 1e-7, smoothing: 0.1 },
        //     ],
        // },
    ];

    for run_cfg in runs {
        let run_dir = std::path::Path::new(out_dir).join(&run_cfg.name);
        std::fs::create_dir_all(&run_dir)?;
        let run_dir_str = run_dir.to_str().unwrap();

        println!("\n🚀 Starting Run: {}", run_cfg.name);

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

        for (stage_idx, stage_cfg) in run_cfg.stages.iter().enumerate() {
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
            use std::io::stdout;
            let backend = CrosstermBackend::new(stdout());
            let mut terminal = Terminal::with_options(
                backend,
                TerminalOptions { viewport: Viewport::Inline(26) },
            )?;

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

            'epoch_loop: for epoch in 0..stage_cfg.epochs {
                state.epoch = epoch + 1;
                let mut idx: Vec<usize> = (0..training_samples.len()).collect();
                idx.shuffle(&mut rng);
                let num_batches = idx.len().max(1) / stage_cfg.batch_size;
                let mut epoch_loss = 0.0f32;

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

                    let token_logits = model.forward(enc_t, dec_t.clone());

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

                    terminal.draw(|frame| render(frame, &state))?;

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

                        // Periodic inference
                        let inference_model = model.valid();
                        let prompt_text = "What is the universe?".to_string();
                        let prompt = if stage_cfg.stage == TrainingStage::Structured { 
                            serde_json::to_string_pretty(&serde_json::json!({
                                "obstacle_dir": "none", "building_dir": "none", "resource_dir": "none", "message": prompt_text,
                            })).unwrap() 
                        } else { prompt_text };
                        let result = inference_model.generate_unmasked_parsed(&tokenizer, &prompt, run_cfg.max_seq_len, &device);
                        state.last_reply = if stage_cfg.stage == TrainingStage::Structured { result.reply } else { result.raw_output };
                    }

                    // Loss Threshold Exit
                    if state.avg_loss < stage_cfg.loss_threshold {
                        println!("\n🎯 Loss Threshold Reached: {:.4} < {:.4}. Ending Stage.", state.avg_loss, stage_cfg.loss_threshold);
                        final_loss = state.avg_loss;
                        break 'epoch_loop;
                    }
                }
                final_loss = epoch_loss / num_batches.max(1) as f32;

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

                // periodic inference
                {
                    let inference_model = model.valid();
                    let prompt_text = "What is the universe?".to_string();
                    let prompt = if stage_cfg.stage == TrainingStage::Structured { 
                        serde_json::to_string_pretty(&serde_json::json!({
                            "obstacle_dir": "none", "building_dir": "none", "resource_dir": "none", "message": prompt_text,
                        })).unwrap() 
                    } else { prompt_text };
                    let result = inference_model.generate_unmasked_parsed(&tokenizer, &prompt, run_cfg.max_seq_len, &device);
                    state.last_reply = if stage_cfg.stage == TrainingStage::Structured { result.reply } else { result.raw_output };
                }
            }
            epochs_already_done += state.epoch;
            terminal.clear()?;

            // Save Chart Image at end of stage
            let chart_path = format!("{}/{}_stage_{}.png", run_dir_str, run_cfg.name, stage_idx + 1);
            if let Err(e) = state.save_chart_image(&chart_path) {
                eprintln!("⚠️  Failed to save chart image: {}", e);
            } else {
                println!("📊 Chart saved to {}", chart_path);
            }

            println!("✅ Stage complete. Final loss: {:.4}", final_loss);
        }
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
#[cfg(target_os = "windows")]
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