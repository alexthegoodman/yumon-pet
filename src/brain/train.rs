/// Yumon Brain training loop — Language model pre-training on SimpleWiki.
///
/// Training objective:
///   Standard next-character prediction (cross-entropy) over each sentence,
///   with a label-indexed vision context vector injected at every timestep.
///
/// Vision context during pre-training:
///   Rather than sampling random context (which the LSTM would learn to ignore),
///   we build an inverted index from CIFAR-100 fine label keywords → sentence indices
///   at load time. Each sentence is then assigned a peaked class-probability vector
///   where the matched label carries most of the probability mass, with small Gaussian
///   noise added to simulate realistic CNN output uncertainty.
///
///   Sentences matching multiple labels share mass across all matches.
///   Sentences matching no label receive a near-uniform + noise distribution
///   (still honest — the model simply sees "nothing recognised strongly").
///
///   This means the LSTM genuinely learns:
///     "when class_probs peaks at index 3 (bear), sentences are about bears"
///   and at inference the real CNN probabilities slot straight in.
///
/// Peak strength tuning:
///   PEAK_LOGIT controls how dominant the matched label is before softmax.
///   A value of 4.0 produces ~85% mass on the top class with 99 competitors,
///   which is a plausible strong-but-not-certain CNN output.
///   Increase toward 6.0 for sharper conditioning; decrease toward 2.0
///   for softer conditioning if the LSTM overfits the label too hard.
///
/// Emote head training:
///   For each sentence, we assign a pseudo-emote label based on simple keyword
///   heuristics. The emote head loss is added at the final timestep only.

use anyhow::Result;
use burn::{
    grad_clipping::GradientClippingConfig, module::AutodiffModule, nn::loss::CrossEntropyLossConfig, optim::{AdamConfig, AdamWConfig, GradientsParams, Optimizer}, prelude::*, tensor::{Int, TensorData, backend::AutodiffBackend}
};
use rand::{Rng, seq::SliceRandom, thread_rng};
use indicatif::{ProgressBar, ProgressStyle};
use ratatui::{Terminal, TerminalOptions, Viewport, prelude::CrosstermBackend};
use std::collections::HashMap;

use crate::{brain::{PAD_TOKEN, bpe::{BpeTokenizer, CL_ID, CR_ID, TokenizerKind}, chart::{TrainingState, render}, mdx::{load_csv_bible, load_csv_qna, load_csv_quotes, load_dictionary_sentences, load_handcrafted_sentences, load_mdx_sentences, load_notion_sentences, load_txt_sentences}, model::CONTEXT_DIMS, pdf::load_pdf_ebook_sentences, samples::{TrainingStage, WorldContext, prepare_paired_samples}}, vision::{CIFAR_CLASSES, EMOTE_CLASSES, EMOTE_NAMES}};
use crate::brain::{
    // CONTEXT_DIMS,
    tokenizer::{Tokenizer, BOS_TOKEN, EOS_TOKEN},
    model::{YumonBrain, YumonBrainConfig, BrainMetadata, GenerationResult},
    wiki::load_wiki_sentences,
};

pub type TrainBackend = burn::backend::Autodiff<burn::backend::Wgpu>;
// pub type TrainBackend = burn::backend::Autodiff<burn::backend::NdArray<f32>>;

// Max sequence length during training (characters)
// pub const MAX_SEQ_LEN:  usize = 120;
// pub const MAX_SEQ_LEN:  usize = 25;
pub const MAX_SEQ_LEN:  usize = 100;
// pub const MAX_SEQ_LEN:  usize = 80; // better for outlines structured output?
// pub const MAX_SEQ_LEN:  usize = 60; // lighter to train on iGPU
// pub const MAX_SEQ_LEN:  usize = 40; // even lower with bpe
// Max vocab size
const MAX_VOCAB:    usize = 256;
// Emote head loss weight (much lighter than language loss)
const EMOTE_WEIGHT: f32   = 0.2;
// Logit assigned to matched CIFAR labels before softmax.
// 4.0 → ~85% probability mass on top class across 100 classes.
const PEAK_LOGIT:   f32   = 4.0;
// Std-dev of Gaussian noise added to logits before softmax (CNN jitter simulation).
const NOISE_STD:    f32   = 0.3;

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

// ─── Context vector construction ─────────────────────────────────────────────

/// Build a peaked class-probability vector for a sentence.
///
/// For each matched class index, assign logit = PEAK_LOGIT.
/// All other indices get logit = 0.
/// Then add Gaussian noise (std = NOISE_STD) to every logit and apply softmax.
///
/// If no classes matched, all logits are 0 + noise → near-uniform distribution.
pub fn peaked_class_probs(matched: &[usize], rng: &mut impl Rng) -> Vec<f32> {
    let mut logits = vec![0.0f32; CIFAR_CLASSES];
    for &ci in matched {
        logits[ci] = PEAK_LOGIT;
    }
    // Add jitter to simulate CNN uncertainty
    for l in logits.iter_mut() {
        *l += rng.sample::<f32, _>(rand::distributions::Standard) * NOISE_STD * 2.0 - NOISE_STD;
    }
    softmax(&logits)
}

/// Build the full 114-dim context vector for a training sample.
/// emote_idx: the sentence's keyword-derived emote label (0..6).
// pub fn build_context(
//     class_probs: &[f32],
//     emote_idx:   usize,
//     rng:         &mut impl Rng,
// ) -> Vec<f32> {
//     let mut ctx = Vec::with_capacity(CONTEXT_DIMS);

//     // class_probs [100]
//     ctx.extend_from_slice(class_probs);

//     // emote_probs [7]: peaked on emote_idx + noise, to match inference distribution
//     let mut emote_logits = vec![0.0f32; EMOTE_CLASSES];
//     emote_logits[emote_idx] = PEAK_LOGIT;
//     for l in emote_logits.iter_mut() {
//         *l += rng.sample::<f32, _>(rand::distributions::Standard) * NOISE_STD * 2.0 - NOISE_STD;
//     }
//     ctx.extend(softmax(&emote_logits));

//     // user_emote_onehot [7]
//     let mut onehot = vec![0.0f32; EMOTE_CLASSES];
//     onehot[emote_idx] = 1.0;
//     ctx.extend(onehot);

//     ctx
// }

pub fn build_context(
    class_probs: &[f32],
    emote_label: usize,
    world:       &WorldContext,   // ← new
    rng:         &mut impl rand::Rng,
) -> Vec<f32> {
    let mut ctx = Vec::with_capacity(CONTEXT_DIMS);

    // class_probs (100)
    ctx.extend_from_slice(class_probs);

    // user_emote_probs (7) — one-hot with small noise
    let mut emote_probs = vec![0.0f32; 7];
    emote_probs[emote_label.min(6)] = 1.0;
    ctx.extend_from_slice(&emote_probs);

    // user_emote_onehot (7) — same for training
    ctx.extend_from_slice(&emote_probs);

    // world spatial context (18)
    ctx.extend_from_slice(&world.to_context_slice());

    debug_assert_eq!(ctx.len(), CONTEXT_DIMS);
    ctx
}

pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum = exps.iter().sum::<f32>();
    exps.iter().map(|e| e / sum).collect()
}

// ─── Main training entry point ────────────────────────────────────────────────

pub fn run(
    wiki_xml:          &str,
    _vision_checkpoint: &str,  // reserved for future real-image fine-tuning
    out_dir:           &str,
    epochs:            usize,
    batch_size:        usize,
    max_articles:      usize,
) -> Result<()> {
    let device: <TrainBackend as Backend>::Device = Default::default();

    // ── Build label keyword index ─────────────────────────────────────────────
    let label_keywords   = build_label_keywords();
    let keyword_index    = build_keyword_index(&label_keywords);
    println!("🏷  CIFAR-100 keyword index: {} unique keywords across {} classes",
             keyword_index.len(), CIFAR_CLASSES);

    // ── Load + tokenize wiki corpus ───────────────────────────────────────────
    let mut sentences = Vec::new();

    let mut wiki_sentences = load_wiki_sentences(wiki_xml, max_articles)?;

    for (i, sent) in wiki_sentences.iter().enumerate() {
        if (i < 12) {
            println!("WIKI: {:?}", sent);
        } else {
            break;
        }
    }

    let mut mdx_sentences = load_mdx_sentences("data/(poems)/")?;

    for (i, sent) in mdx_sentences.iter().enumerate() {
        if (i < 12) {
            println!("MDX: {:?}", sent);
        } else {
            break;
        }
    }

    let mut quote_sentences = load_csv_quotes("data/quotes.csv")?;

    for (i, sent) in quote_sentences.iter().enumerate() {
        if (i < 12) {
            println!("QUOTE: {:?}", sent);
        } else {
            break;
        }
    }

    let mut dict_sentences = load_dictionary_sentences("data/Dictionary/Oxford/Oxford_English_Dictionary.txt")?;

    for (i, sent) in dict_sentences.iter().enumerate() {
        if (i < 12) {
            println!("DICT: {:?}", sent);
        }
    }

    let mut qna_sentences = load_csv_qna("data/AI.csv")?;

    for (i, sent) in qna_sentences.iter().enumerate() {
        if (i < 12) {
            println!("Q&A: {:?}", sent);
        } else {
            break;
        }
    }

    let mut bible_verses = load_csv_bible("data/bible_bbe.csv")?;

    for (i, sent) in bible_verses.iter().enumerate() {
        if (i < 12) {
            println!("Verse: {:?}", sent);
        } else {
            break;
        }
    }

    let mut handcrafted = load_handcrafted_sentences("data/handcrafted.txt")?;

    for (i, sent) in handcrafted.iter().enumerate() {
        if (i < 12) {
            println!("handcrafted: {:?}", sent);
        } else {
            break;
        }
    }

    let mut notions = load_notion_sentences("data/notion/")?;

    for (i, sent) in notions.iter().enumerate() {
        if (i < 12) {
            println!("notion: {:?}", sent);
        }
    }

    // let personals = load_notion_sentences("data/personal/")?;

    // for (i, sent) in personals.iter().enumerate() {
    //     if (i < 12) {
    //         println!("personal: {:?}", sent);
    //     }
    // }

    // let mut ebooks = load_pdf_ebook_sentences("data/algorithms_ebook.pdf")?;

    // for (i, sent) in ebooks.iter().enumerate() {
    //     if (i < 12) {
    //         println!("ebook: {:?}", sent);
    //     }
    // }

    // let mut ebooks = load_pdf_ebook_sentences("data/stephen_hawking_a_brief_history_of_time.pdf")?;

    // for (i, sent) in ebooks.iter().enumerate() {
    //     if (i < 12) {
    //         println!("ebook: {:?}", sent);
    //     }
    // }

    let mut txt = load_txt_sentences("data/creative_stories.txt")?;

    for (i, sent) in txt.iter().enumerate() {
        if (i < 12) {
            println!("txt: {:?}", sent);
        }
    }

    // let mut ebooks2 = load_pdf_ebook_sentences("data/Perspectives.pdf")?;

    // for (i, sent) in ebooks2.iter().enumerate() {
    //     if (i < 12) {
    //         println!("ebook: {:?}", sent);
    //     }
    // }

    sentences.extend(mdx_sentences.clone());
    // sentences.extend(quote_sentences.clone());
    // sentences.extend(qna_sentences.clone());
    sentences.extend(handcrafted.clone());
    // sentences.extend(wiki_sentences.clone());
    // sentences.extend(dict_sentences.clone()); // wasteful cloning

    let full_text: String = sentences.join(" ");
    println!("Building vocabulary from {} chars...", full_text.len());
    // let tokenizer = Tokenizer::build_from_text(&full_text, MAX_VOCAB);

    let use_bpe = true;

    let tokenizer = if use_bpe {
        TokenizerKind::Bpe(BpeTokenizer::load("yumon_bpe")?)
    } else {
        let dir = std::path::Path::new(out_dir);
        TokenizerKind::Char(Tokenizer::build_from_text(&full_text, MAX_VOCAB))
    };

    println!("Vocabulary size: {}", tokenizer.vocab_size());

    // ── Prepare samples with label-indexed context ────────────────────────────
    let mut training_samples = Vec::new();

    let mut rng = thread_rng();

     // optional: makes keyword matching faster
    wiki_sentences.shuffle(&mut rng);
    wiki_sentences.truncate(8192);

    dict_sentences.shuffle(&mut rng);
    dict_sentences.truncate(8192);

    quote_sentences.shuffle(&mut rng);
    quote_sentences.truncate(8192);

    qna_sentences.shuffle(&mut rng);
    qna_sentences.truncate(8192);

    mdx_sentences.shuffle(&mut rng);
    mdx_sentences.truncate(8192);

    bible_verses.shuffle(&mut rng);
    bible_verses.truncate(8192);

    handcrafted.shuffle(&mut rng);
    handcrafted.truncate(8192);

    notions.shuffle(&mut rng);
    notions.truncate(8192);

    txt.shuffle(&mut rng);
    txt.truncate(8192);

    // ebooks.shuffle(&mut rng);
    // ebooks.truncate(8192);

    // ebooks2.shuffle(&mut rng);
    // ebooks2.truncate(8192);

    // let training_stage = TrainingStage::Language; // first
    let training_stage = TrainingStage::Structured; // fine-tune

    let mut mdx_samples = prepare_paired_samples(&mdx_sentences, &tokenizer, &keyword_index, &mut rng, 1, 2, training_stage);
    let mut quote_samples = prepare_paired_samples(&quote_sentences, &tokenizer, &keyword_index, &mut rng, 1, 2, training_stage);
    let mut qna_samples = prepare_paired_samples(&qna_sentences, &tokenizer, &keyword_index, &mut rng, 1, 2, training_stage);
    let mut wiki_samples = prepare_paired_samples(&wiki_sentences, &tokenizer, &keyword_index, &mut rng, 1, 2, training_stage);
    let mut dict_samples = prepare_paired_samples(&dict_sentences, &tokenizer, &keyword_index, &mut rng, 1, 2, training_stage);
    let mut bible_samples = prepare_paired_samples(&bible_verses, &tokenizer, &keyword_index, &mut rng, 1, 2, training_stage);
    let mut handcrafted_samples = prepare_paired_samples(&handcrafted, &tokenizer, &keyword_index, &mut rng, 1, 2, training_stage);
    let mut notion_samples = prepare_paired_samples(&notions, &tokenizer, &keyword_index, &mut rng, 1, 2, training_stage);
    // let personal_samples = prepare_samples(&personals, &tokenizer, &keyword_index);
    // let mut ebook_samples = prepare_paired_samples(&ebooks, &tokenizer, &keyword_index, &mut rng, 1, 2, training_stage);
    let mut txt_samples = prepare_paired_samples(&txt, &tokenizer, &keyword_index, &mut rng, 1, 2, training_stage);
    // let mut ebooks2_samples = prepare_paired_samples(&ebooks2, &tokenizer, &keyword_index, &mut rng, 1, 2);

    // println!(
    //     "Samples lengths: {} {} {} {} {}",
    //     mdx_samples.len(),
    //     // quote_samples.len(),
    //     // qna_samples.len(),
    //     // wiki_samples.len(),
    //     bible_samples.len(),
    //     handcrafted_samples.len(),
    //     notion_samples.len(),
    //     // personal_samples.len(),
    //     txt_samples.len(),
    //     // ebooks2_samples.len()
    // );

    // bible_samples.shuffle(&mut rng);
    // bible_samples.truncate(2048);
    // // bible_samples.truncate(4096);

    // // wiki_samples.shuffle(&mut rng);
    // // wiki_samples.truncate(2048);

    // txt_samples.shuffle(&mut rng);
    // // ebook_samples.truncate(2048);
    // txt_samples.truncate(4096);

    // // ebooks2_samples.shuffle(&mut rng);
    // // ebooks2_samples.truncate(2048);
    // // // ebooks2_samples.truncate(4096);

    // notion_samples.shuffle(&mut rng);
    // notion_samples.truncate(2048);
    // // notion_samples.truncate(4096);
    
    training_samples.extend(wiki_samples); // Yumon expresses that he is confused by wiki material
    training_samples.extend(quote_samples);
    training_samples.extend(dict_samples);
    training_samples.extend(qna_samples);
    training_samples.extend(bible_samples);
    training_samples.extend(notion_samples);
    training_samples.extend(txt_samples);
    // training_samples.extend(ebooks2_samples);
    
    training_samples.shuffle(&mut rng);
    training_samples.truncate(65536);
    // training_samples.truncate(16384); // maybe at 128 hidden size? maybe need 256?
    // training_samples.truncate(8192); // limit total for now
    // training_samples.truncate(1024); 
    // training_samples.truncate(2048); 

    training_samples.extend(mdx_samples);
    training_samples.extend(handcrafted_samples); // always add after to include all of these
    // training_samples.extend(personal_samples);

    println!(
        "Training samples: {}",
        training_samples.len()
    );

    for (i, sample) in training_samples.iter().enumerate() {
        if (i < 12) {
            println!("Sample: {:?}", sample.target_json);
            println!("input:  {:?}", tokenizer.decode(&sample.input_ids));
            println!("target: {:?}", tokenizer.decode(&sample.target_labels
                .iter()
                .map(|&t| if t == PAD_TOKEN { PAD_TOKEN } else { t })
                .collect::<Vec<_>>()));
            println!("input_len: {}", sample.input_ids.iter().filter(|&&t| t != PAD_TOKEN).count());
            println!("target_active: {}", sample.target_labels.iter().filter(|&&t| t != PAD_TOKEN).count());

        } else {
            break;
        }
    }

    // ── Init model + optimizer ────────────────────────────────────────────────
    // ── Resume from checkpoint if one exists ─────────────────────────────────
    let checkpoint_meta = std::path::Path::new(out_dir).join("metadata.json");
    let checkpoint_model = std::path::Path::new(out_dir).join("model.bin"); // BinFileRecorder ext

    let (mut model, epochs_already_done) = if checkpoint_model.exists() && checkpoint_meta.exists() {
        match YumonBrain::<TrainBackend>::load(out_dir, &device) {
            Ok((m, _tok)) => {
                let meta_json = std::fs::read_to_string(&checkpoint_meta)?;
                let meta: BrainMetadata = serde_json::from_str(&meta_json)?;
                println!("▶️  Resuming from checkpoint ({} epochs done, loss={:.4})",
                         meta.epochs_trained, meta.final_loss);
                (m, meta.epochs_trained)
            }
            Err(e) => {
                eprintln!("⚠️  Checkpoint found but failed to load ({e}) — starting fresh.");
                (YumonBrainConfig::new(tokenizer.vocab_size()).init(&device), 0)
            }
        }
    } else {
        println!("🆕 No checkpoint found — starting fresh.");
        (YumonBrainConfig::new(tokenizer.vocab_size()).init(&device), 0)
    };

    if epochs_already_done >= epochs {
        println!("✅ Already trained for {epochs_already_done} epochs (requested {epochs}). Nothing to do.");
        return Ok(());
    }

    let remaining_epochs = epochs - epochs_already_done;
    println!("📅 Training for {remaining_epochs} more epoch(s) (to reach {epochs} total).");

    // let mut optimizer = AdamConfig::new()
    //     .with_epsilon(1e-7)
    //     .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(1e-5)))
    //     .init();

    let mut optimizer = AdamWConfig::new()
        .with_epsilon(1e-7)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        // .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(1e-5)))
        .init();

    // let ce_loss = CrossEntropyLossConfig::new().init(&device);
    let ce_loss = CrossEntropyLossConfig::new()
        .with_pad_tokens(Some(vec![PAD_TOKEN as usize]))
        .with_smoothing(Some(0.1))
        .init(&device);
    let mut rng = rand::thread_rng();
    let mut final_loss = 0.0f32;
    
    // lr over time
    // let first_lr = 1e-4;
    let first_lr = 0.001;
    let last_lr = 1e-8;

    // let first_lr = 1e-6;
    // let last_lr = 1e-7;

    // let first_lr = 1e-7;
    // let last_lr = 1e-8;

    // let first_lr = 0.0001;
    // let last_lr = 1e-8;

    use std::io::stdout;
    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::with_options(
        backend,
        TerminalOptions { viewport: Viewport::Inline(16) },
    )?;

    let total_batches = training_samples.len() / batch_size;

    let mut state = TrainingState {
        loss_history: vec![],
        avg_loss_history: vec![],
        current_loss: 0.0,
        avg_loss: 0.0,
        epoch: epochs_already_done + 1,
        total_epochs: epochs,
        batch: 0,
        total_batches,          // ← same value your loop uses for num_batches
        current_lr: first_lr as f64,
        global_step: 0,
    };

    // ── Training loop ─────────────────────────────────────────────────────────
    for epoch in 0..remaining_epochs {
        let absolute_epoch = epochs_already_done + epoch + 1;

        let mut idx: Vec<usize> = (0..training_samples.len()).collect();
        use rand::seq::SliceRandom;
        idx.shuffle(&mut rng);

        let num_batches = idx.len().max(1) / batch_size;
        // let pb = make_progress(num_batches, absolute_epoch, epochs);
        let mut epoch_loss = 0.0f32;

        for batch_num in 0..num_batches {
            // cosine annealing
            // let current_lr = {
            //     let total_steps = remaining_epochs * (training_samples.len() / batch_size);
            //     let step = epoch * (training_samples.len() / batch_size) + batch_num;
            //     let progress = step as f64 / total_steps as f64;
            //     let cosine = (std::f64::consts::PI * progress).cos();
            //     (last_lr + 0.5 * (first_lr - last_lr) * (1.0 + cosine)) as f64
            // };

            // // linear
            let current_lr = {
                let total_steps = remaining_epochs * (training_samples.len() / batch_size);
                let step = epoch * (training_samples.len() / batch_size) + batch_num;
                let t = step as f64 / total_steps as f64;
                (first_lr as f64 * (1.0 - t) + last_lr as f64 * t)
            };

            // // exp decay
            // let current_lr = {
            //     let total_steps = remaining_epochs * (training_samples.len() / batch_size);
            //     let step = epoch * (training_samples.len() / batch_size) + batch_num;
            //     let t = step as f64 / total_steps as f64;
            //     (first_lr as f64 * (last_lr as f64 / first_lr as f64).powf(t))
            // };

            let batch_start = batch_num * batch_size;
            let batch_end = (batch_start + batch_size).min(training_samples.len());
            let batch_idx = &idx[batch_start..batch_end];
            let current_batch_size = batch_idx.len();  // in case last batch is smaller

            if current_batch_size == 0 { continue; }

            // ── Collect everything into flat vectors (fast) ─────────────────────
            // let mut all_ids: Vec<i32> = Vec::with_capacity(current_batch_size * MAX_SEQ_LEN);
            let mut all_contexts: Vec<f32> = Vec::with_capacity(current_batch_size * CONTEXT_DIMS);
            let mut all_lang_targets: Vec<i32> = Vec::with_capacity(current_batch_size * MAX_SEQ_LEN);
            let mut all_emote_targets: Vec<i32> = Vec::with_capacity(current_batch_size);
            let mut all_enc_ids: Vec<i32> = Vec::with_capacity(current_batch_size * MAX_SEQ_LEN);
            // let mut all_dec_ids: Vec<i32> = Vec::with_capacity(current_batch_size * MAX_SEQ_LEN);
            let mut all_dec_input_ids: Vec<i32> = Vec::with_capacity(current_batch_size * MAX_SEQ_LEN);

            for &i in batch_idx {
                let sample = &training_samples[i];

                // all_ids.extend(sample.input_ids.iter().map(|&t| t as i32));

                // // ← pass sample.world instead of rebuilding from scratch
                let class_probs = peaked_class_probs(&sample.matched_classes, &mut rng);
                let ctx_flat = build_context(&class_probs, sample.emote_label, &sample.world, &mut rng);
                all_contexts.extend(ctx_flat);

                // all_lang_targets.extend(sample.target_labels.iter().map(|&t| t as i32));
                all_emote_targets.push(sample.emote_label as i32);

                all_enc_ids.extend(sample.input_ids.iter().map(|&t| t as i32));
                // all_dec_ids.extend(sample.target_labels.iter().map(|&t| t as i32));

                // Inside the for &i in batch_idx loop, after let sample = &training_samples[i];

                let target_labels = &sample.target_labels;  // [tok0, tok1, ..., EOS, PAD, PAD, ...]

                // Find real length (includes EOS, stops before first PAD)
                let real_len = target_labels.iter()
                    .position(|&t| t == PAD_TOKEN)
                    .unwrap_or(MAX_SEQ_LEN);

                // ── Decoder INPUT (exactly like generate_* loops) ─────────────
                let mut dec_input: Vec<i32> = vec![BOS_TOKEN as i32];
                dec_input.extend(
                    target_labels[0..real_len.saturating_sub(1)]
                        .iter()
                        .map(|&t| t as i32)
                );
                dec_input.resize(MAX_SEQ_LEN, PAD_TOKEN as i32);

                // ── Loss TARGETS (shifted correctly, includes first token + EOS) ──
                let mut lang_targets: Vec<i32> = target_labels[0..real_len]  // ← CHANGED: start at 0
                    .iter()
                    .map(|&t| t as i32)
                    .collect();
                lang_targets.resize(MAX_SEQ_LEN, PAD_TOKEN as i32);

                all_dec_input_ids.extend(dec_input);
                all_lang_targets.extend(lang_targets);
            }

            // ── Stack into real batched tensors (ONE allocation) ───────────────
            // let ids_t = Tensor::<TrainBackend, 2, Int>::from_ints(
            //     TensorData::new(all_ids, [current_batch_size, MAX_SEQ_LEN]),
            //     &device,
            // );

            let context_t = Tensor::<TrainBackend, 2>::from_floats(
                TensorData::new(all_contexts, [current_batch_size, CONTEXT_DIMS]),
                &device,
            );

            let lang_target_t = Tensor::<TrainBackend, 1, Int>::from_ints(
                TensorData::new(all_lang_targets, [current_batch_size * MAX_SEQ_LEN]),
                &device,
            );

            let emote_target_t = Tensor::<TrainBackend, 1, Int>::from_ints(
                TensorData::new(all_emote_targets, [current_batch_size]),
                &device,
            );

            let enc_t = Tensor::<TrainBackend, 2, Int>::from_ints(
                TensorData::new(all_enc_ids, [current_batch_size, MAX_SEQ_LEN]),
                &device,
            );

            // let dec_t = Tensor::<TrainBackend, 2, Int>::from_ints(
            //     TensorData::new(all_dec_ids, [current_batch_size, MAX_SEQ_LEN]),
            //     &device,
            // );

            let dec_t = Tensor::<TrainBackend, 2, Int>::from_ints(  // now shifted input
                TensorData::new(all_dec_input_ids, [current_batch_size, MAX_SEQ_LEN]),
                &device,
            );

            // ── SINGLE forward pass (this is where the 20× speedup happens) ─────
            // let (token_logits, emote_logits) = model.forward(ids_t, context_t);

            let (token_logits, emote_logits) = model.forward(enc_t, dec_t, context_t);

            // ── Loss (now automatically batched) ───────────────────────────────
            let vocab = tokenizer.vocab_size();
            let logits_2d = token_logits.reshape([current_batch_size * MAX_SEQ_LEN, vocab]);  // [B*S, vocab]

            let lang_loss = ce_loss.forward(logits_2d, lang_target_t);

            let emote_loss = ce_loss.forward(emote_logits, emote_target_t);  // [B, C] and [B]

            let total_loss = lang_loss + emote_loss.mul_scalar(EMOTE_WEIGHT);

            // Backward + step (exactly one per real batch)
            let grads = GradientsParams::from_grads(total_loss.backward(), &model);
            model = optimizer.step(
                current_lr, 
                // 1e-6,
                model, 
                grads
            );

            // Logging (same feel as before)
            let loss_val: f32 = total_loss.clone().inner().to_data().to_vec::<f32>().unwrap()[0];
            epoch_loss += loss_val;

            let batches_done = (batch_num + 1) as f32;
            let avg_loss = epoch_loss / batches_done;

            // pb.set_prefix(format!("{:.4} avg_loss={:.4} current_lr={:.7}", loss_val, avg_loss, current_lr));
            // pb.inc(1);

            // instead of pb.set_prefix(...) and pb.inc(1)
            // state.current_loss = loss_val;
            // let loss_scale = 100.0; // 1 for normal, 100.0 for structured? helps see minor improvements in TUI
            let loss_scale = 1.0; 
            state.current_loss = loss_val * loss_scale; // Don't scale the actual loss used for backprop, just the display value.
            state.avg_loss = (epoch_loss * loss_scale) / (batch_num + 1) as f32; // Don't scale the actual loss used for backprop, just the display value.
            state.batch = batch_num + 1;
            state.current_lr = current_lr;
            state.global_step += 1;
            state.loss_history.push((state.global_step as f64, loss_val as f64 * loss_scale as f64)); // Don't scale the actual loss used for backprop, just the display value.
            state.avg_loss_history.push((state.global_step as f64, state.avg_loss as f64));

            // optional: sliding window so chart doesn't compress
            // if state.loss_history.len() > 500 {
            //     state.loss_history.remove(0);
            //     state.avg_loss_history.remove(0);
            // }

            terminal.draw(|frame| render(frame, &state))?;
        }

        final_loss = epoch_loss / num_batches.max(1) as f32;
        // pb.finish_with_message(format!("epoch {absolute_epoch}/{epochs}  loss={final_loss:.4}"));

        let meta = BrainMetadata {
            vocab_size:     tokenizer.vocab_size(),
            epochs_trained: absolute_epoch,
            final_loss,
        };
        model.valid().save(out_dir, &tokenizer, &meta)?;

        // --- Periodic Inference ---
        // {
        //     let inference_model = model.valid();
        //     println!("\n🔍 Running inference for epoch {absolute_epoch}...");

        //     // Pick a few target classes to see if the model is learning the context
        //     // 0: apple, 3: bear, 8: bicycle, 23: cloud, 4: neutral
        //     let test_prompts = [
        //         (0, "Apple", "The apple"),
        //         (3, "Bear", "The bear"),
        //         (8, "Bicycle", "A bicycle"),
        //         (23, "Cloud", "The cloud"),
        //         (4, "Neutral", "Hello"),
        //     ];

        //     let label_keywords   = build_label_keywords();
        //     let keyword_index    = build_keyword_index(&label_keywords);

        //     for (class_idx, label, seed) in test_prompts {
        //         let mut inf_rng = rand::thread_rng();
        //         let emote_label     = keyword_emote_label(seed);
        //         let matched_classes = matched_classes(seed, &keyword_index);
        //         // We use fixed context for comparison across epochs
        //         let class_probs = peaked_class_probs(&matched_classes, &mut inf_rng);
        //         let emote_probs = peaked_class_probs(&[0, 0, 0, 1, 0, 0, 0], &mut inf_rng); // neutral user emote

        //         let res = inference_model.generate(
        //             &tokenizer,
        //             &class_probs,
        //             &emote_probs,
        //             4, // neutral user emote onehot
        //             seed,
        //             30,
        //             &device,
        //         );

        //         println!("  [{label}] {seed} -> {} (emote: {})",
        //             res.reply.replace("\n", " "),
        //             EMOTE_NAMES[res.yumon_emote_idx]
        //         );
        //     }
        //     println!();
        // }
    }

    // after the epoch loop
    terminal.clear()?;

    println!("✅ Brain training complete. Final loss: {final_loss:.4}");
    Ok(())
}

// ─── Sample preparation ───────────────────────────────────────────────────────

// struct Sample {
//     input_ids:       Vec<usize>,
//     target_ids:      Vec<usize>,
//     emote_label:     usize,
//     /// CIFAR-100 class indices whose keywords appear in this sentence.
//     /// Empty = no label matched → near-uniform context at train time.
//     matched_classes: Vec<usize>,
//     target_labels: Vec<usize>,
//     pair: Vec<String>
// }

// fn prepare_samples(
//     sentences:     &[String],
//     tokenizer:     &TokenizerKind,
//     keyword_index: &HashMap<String, Vec<usize>>,
// ) -> Vec<Sample> {
//     let mut samples = Vec::new();

//     for sentence in sentences {
//         let bad_words = vec!["sex", "drug", "kill", "rape"];

//         // Check if the main_string contains any of the strings in the vector
//         let found = bad_words
//             .iter()
//             .any(|&substring| sentence.contains(substring));

//         if found { continue; }

//         let encoded = tokenizer.encode(sentence);
//         if encoded.len() < (MAX_SEQ_LEN - 30) || encoded.len() > MAX_SEQ_LEN { continue; }

//         // input:  [BOS, t0, t1, ..., t_{n-1}]
//         // labels: [t0,  t1, ..., t_{n-1}, EOS]  (one-ahead shift)
//         let input_ids: Vec<usize> = std::iter::once(BOS_TOKEN)
//             .chain(encoded.iter().cloned().take(MAX_SEQ_LEN - 1))
//             .collect();

//         let target_labels: Vec<usize> = encoded.iter().cloned().take(MAX_SEQ_LEN - 1)
//             .chain(std::iter::once(EOS_TOKEN))
//             .collect();

//         let pad = |mut v: Vec<usize>| -> Vec<usize> {
//             v.resize(MAX_SEQ_LEN, PAD_TOKEN);
//             v
//         };

//         let input_ids = pad(input_ids);
//         let target_labels = pad(target_labels);

//         // if samples.len() < 12 {
//         //     println!("input_ids {:?}", input_ids);
//         //     println!("target_labels {:?}", target_labels);
//         // }

//         // assert!(input_ids.len() == MAX_SEQ_LEN, 
//         //     "input_ids wrong len: {} (encoded len was {})", input_ids.len(), encoded.len());

//         let emote_label     = keyword_emote_label(sentence);
//         let matched_classes = matched_classes(sentence, keyword_index);

//         samples.push(Sample {
//             pair:            vec![sentence.clone()],
//             input_ids,
//             target_ids:      vec![],          // unused in completion mode
//             target_labels,
//             emote_label,
//             matched_classes,
//         });
//     }

//     samples
// }

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