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
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::{backend::AutodiffBackend, Int, TensorData},
    module::AutodiffModule,
};
use rand::Rng;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;

use crate::vision::{CIFAR_CLASSES, EMOTE_CLASSES};
use crate::brain::{
    CONTEXT_DIMS,
    tokenizer::{Tokenizer, BOS_TOKEN, EOS_TOKEN},
    model::{YumonBrain, YumonBrainConfig, BrainMetadata},
    wiki::load_wiki_sentences,
};

pub type TrainBackend = burn::backend::Autodiff<burn::backend::Wgpu>;

// Max sequence length during training (characters)
// const MAX_SEQ_LEN:  usize = 120;
const MAX_SEQ_LEN:  usize = 60; // easier to train on iGPU
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
fn build_label_keywords() -> Vec<Vec<String>> {
    CIFAR_FINE_LABELS.iter().map(|label| {
        label.split('_')
             .map(|w| w.to_lowercase())
             .filter(|w| w.len() >= 3) // skip tiny words like "a", "of"
             .collect()
    }).collect()
}

/// Maps keyword → list of CIFAR class indices that contain it.
fn build_keyword_index(label_keywords: &[Vec<String>]) -> HashMap<String, Vec<usize>> {
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
fn matched_classes(sentence: &str, keyword_index: &HashMap<String, Vec<usize>>) -> Vec<usize> {
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
fn whole_word_match(text: &str, kw: &str) -> bool {
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
fn peaked_class_probs(matched: &[usize], rng: &mut impl Rng) -> Vec<f32> {
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
fn build_context(
    class_probs: &[f32],
    emote_idx:   usize,
    rng:         &mut impl Rng,
) -> Vec<f32> {
    let mut ctx = Vec::with_capacity(CONTEXT_DIMS);

    // class_probs [100]
    ctx.extend_from_slice(class_probs);

    // emote_probs [7]: peaked on emote_idx + noise, to match inference distribution
    let mut emote_logits = vec![0.0f32; EMOTE_CLASSES];
    emote_logits[emote_idx] = PEAK_LOGIT;
    for l in emote_logits.iter_mut() {
        *l += rng.sample::<f32, _>(rand::distributions::Standard) * NOISE_STD * 2.0 - NOISE_STD;
    }
    ctx.extend(softmax(&emote_logits));

    // user_emote_onehot [7]
    let mut onehot = vec![0.0f32; EMOTE_CLASSES];
    onehot[emote_idx] = 1.0;
    ctx.extend(onehot);

    ctx
}

fn softmax(logits: &[f32]) -> Vec<f32> {
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
    let sentences = load_wiki_sentences(wiki_xml, max_articles)?;

    let full_text: String = sentences.join(" ");
    println!("Building vocabulary from {} chars...", full_text.len());
    let tokenizer = Tokenizer::build_from_text(&full_text, MAX_VOCAB);
    println!("Vocabulary size: {}", tokenizer.vocab_size);

    // ── Prepare samples with label-indexed context ────────────────────────────
    let training_samples = prepare_samples(&sentences, &tokenizer, &keyword_index);

    // for (i, sample) in training_samples.iter().enumerate() {
    //     if i < 10 {
    //         println!("sample example: {:?} {:?} {:?} {:?}", sample.emote_label, sample.input_ids, sample.matched_classes, sample.target_ids);
    //     }
    // }

    let labelled   = training_samples.iter().filter(|s| !s.matched_classes.is_empty()).count();
    let unlabelled = training_samples.len() - labelled;
    println!(
        "Training samples: {}  ({} with CIFAR label match, {} near-uniform)",
        training_samples.len(), labelled, unlabelled
    );

    // ── Init model + optimizer ────────────────────────────────────────────────
    let mut model: YumonBrain<TrainBackend> = YumonBrainConfig::new(tokenizer.vocab_size)
        .init(&device);
    let mut optimizer = AdamConfig::new()
        .with_epsilon(1e-7)
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(1e-5)))
        .init();

    let ce_loss = CrossEntropyLossConfig::new().init(&device);
    let mut rng = rand::thread_rng();
    let mut final_loss = 0.0f32;

    // ── Training loop ─────────────────────────────────────────────────────────
    for epoch in 0..epochs {
        let mut idx: Vec<usize> = (0..training_samples.len()).collect();
        use rand::seq::SliceRandom;
        idx.shuffle(&mut rng);

        let num_batches = idx.len().max(1) / batch_size;
        let pb = make_progress(num_batches, epoch + 1, epochs);
        let mut epoch_loss = 0.0f32;

        for batch_num in 0..num_batches {
            let batch_idx = &idx[batch_num * batch_size..(batch_num + 1) * batch_size];

            let mut batch_loss_tensors: Vec<Tensor<TrainBackend, 1>> = Vec::new();
            let mut batch_loss_sum = 0.0f32;

            for &i in batch_idx {
                let sample  = &training_samples[i];
                let seq_len = sample.input_ids.len();
                if seq_len < 2 { continue; }

                // Build peaked class probs from this sentence's matched labels,
                // with fresh noise every forward pass (data augmentation for free).
                let class_probs = peaked_class_probs(&sample.matched_classes, &mut rng);
                let ctx_flat    = build_context(&class_probs, sample.emote_label, &mut rng);

                let context_t = Tensor::<TrainBackend, 2>::from_floats(
                    TensorData::new(ctx_flat, [1, CONTEXT_DIMS]),
                    &device,
                );

                let ids_flat: Vec<i32> = sample.input_ids.iter().map(|&t| t as i32).collect();
                let ids_t = Tensor::<TrainBackend, 2, Int>::from_ints(
                    TensorData::new(ids_flat, [1, seq_len]),
                    &device,
                );

                let (token_logits, emote_logits) = model.forward(ids_t, context_t);

                // Language loss
                let vocab     = tokenizer.vocab_size;
                let logits_2d = token_logits.reshape([seq_len, vocab]);
                let targets: Vec<i32> = sample.target_ids.iter().map(|&t| t as i32).collect();
                let target_t  = Tensor::<TrainBackend, 1, Int>::from_ints(
                    TensorData::new(targets, [seq_len]),
                    &device,
                );
                let lang_loss = ce_loss.forward(logits_2d, target_t);

                // Emote loss — last timestep only
                let emote_target_t = Tensor::<TrainBackend, 1, Int>::from_ints(
                    TensorData::new(vec![sample.emote_label as i32], [1]),
                    &device,
                );
                let emote_loss = ce_loss.forward(emote_logits, emote_target_t);

                let total     = lang_loss + emote_loss.mul_scalar(EMOTE_WEIGHT);
                let loss_val: f32 = total.clone().inner().to_data()
                    .to_vec::<f32>().unwrap()[0];

                batch_loss_sum += loss_val;
                batch_loss_tensors.push(total);
            }

            if batch_loss_tensors.is_empty() { continue; }

            let n        = batch_loss_tensors.len() as f32;
            let combined = batch_loss_tensors.into_iter()
                .reduce(|a, b| a + b)
                .unwrap()
                .div_scalar(n);

            let grads = GradientsParams::from_grads(combined.backward(), &model);
            model     = optimizer.step(3e-4, model, grads);

            let avg = batch_loss_sum / n;
            epoch_loss += avg;
            pb.set_prefix(format!("{:.4}", avg));
            pb.inc(1);
        }

        final_loss = epoch_loss / num_batches.max(1) as f32;
        pb.finish_with_message(format!("epoch {}/{epochs}  loss={final_loss:.4}", epoch + 1));

        let meta = BrainMetadata {
            vocab_size:     tokenizer.vocab_size,
            epochs_trained: epoch + 1,
            final_loss,
        };
        model.valid().save(out_dir, &tokenizer, &meta)?;
    }

    println!("✅ Brain training complete. Final loss: {final_loss:.4}");
    Ok(())
}

// ─── Sample preparation ───────────────────────────────────────────────────────

struct Sample {
    input_ids:       Vec<usize>,
    target_ids:      Vec<usize>,
    emote_label:     usize,
    /// CIFAR-100 class indices whose keywords appear in this sentence.
    /// Empty = no label matched → near-uniform context at train time.
    matched_classes: Vec<usize>,
}

fn prepare_samples(
    sentences:     &[String],
    tokenizer:     &Tokenizer,
    keyword_index: &HashMap<String, Vec<usize>>,
) -> Vec<Sample> {
    let mut samples = Vec::new();
    let mut label_counts = HashMap::new();

    for s in sentences {
        let encoded = tokenizer.encode(s);
        if encoded.len() < 3 { continue; }

        let truncated: Vec<usize> = std::iter::once(BOS_TOKEN)
            .chain(encoded.iter().cloned().take(MAX_SEQ_LEN - 2))
            .chain(std::iter::once(EOS_TOKEN))
            .collect();

        let input_ids  = truncated[..truncated.len() - 1].to_vec();
        let target_ids = truncated[1..].to_vec();

        let emote_label     = keyword_emote_label(s);
        let matched_classes = matched_classes(s, keyword_index);

        if matched_classes.len() > 0 {
            if label_counts.get(&matched_classes[0]).is_none() {
                label_counts.insert(matched_classes[0], 0);
            }

            // we can train on non-matches for broader knowledge later on possibly
            if let Some(current) = label_counts.get(&matched_classes[0]) {
                if *current < 200 {
                    label_counts.insert(matched_classes[0], *current + 1);
                    samples.push(Sample { input_ids, target_ids, emote_label, matched_classes });
                }
            }
        }
    }

    samples
}

// ─── Emote keyword heuristic ──────────────────────────────────────────────────

/// Simple keyword-based pseudo-label for emote head pre-training.
/// 0=angry, 1=disgust, 2=fear, 3=happy, 4=neutral, 5=sad, 6=surprise
fn keyword_emote_label(text: &str) -> usize {
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

fn make_progress(total: usize, epoch: usize, epochs: usize) -> ProgressBar {
    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner} epoch {msg} [{bar:40}] {pos}/{len} loss={prefix}")
            .unwrap()
    );
    pb.set_message(format!("{epoch}/{epochs}"));
    pb
}
