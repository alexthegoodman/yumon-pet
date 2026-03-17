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
use rand::Rng;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;

use crate::{brain::{PAD_TOKEN, bpe::{BpeTokenizer, TokenizerKind}, mdx::{load_csv_qna, load_csv_quotes, load_dictionary_sentences, load_mdx_sentences}}, vision::{CIFAR_CLASSES, EMOTE_CLASSES, EMOTE_NAMES}};
use crate::brain::{
    CONTEXT_DIMS,
    tokenizer::{Tokenizer, BOS_TOKEN, EOS_TOKEN},
    model::{YumonBrain, YumonBrainConfig, BrainMetadata, GenerationResult},
    wiki::load_wiki_sentences,
};

pub type TrainBackend = burn::backend::Autodiff<burn::backend::Wgpu>;
// pub type TrainBackend = burn::backend::Autodiff<burn::backend::NdArray<f32>>;

// Max sequence length during training (characters)
// const MAX_SEQ_LEN:  usize = 120;
// pub const MAX_SEQ_LEN:  usize = 60; // lighter to train on iGPU
pub const MAX_SEQ_LEN:  usize = 30; // even lower with bpe
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
    let mut sentences = Vec::new();

    let wiki_sentences = load_wiki_sentences(wiki_xml, max_articles)?;

    for (i, sent) in wiki_sentences.iter().enumerate() {
        if (i < 12) {
            println!("WIKI: {:?}", sent);
        }
    }

    let mdx_sentences = load_mdx_sentences("data/(poems)/")?;

    for (i, sent) in mdx_sentences.iter().enumerate() {
        if (i < 12) {
            println!("MDX: {:?}", sent);
        }
    }

    let quote_sentences = load_csv_quotes("data/quotes.csv")?;

    for (i, sent) in quote_sentences.iter().enumerate() {
        if (i < 12) {
            println!("QUOTE: {:?}", sent);
        }
    }

    let dict_sentences = load_dictionary_sentences("data/Dictionary/Oxford/Oxford_English_Dictionary.txt")?;

    for (i, sent) in dict_sentences.iter().enumerate() {
        if (i < 12) {
            println!("DICT: {:?}", sent);
        }
    }

    let qna_sentences = load_csv_qna("data/AI.csv")?;

    for (i, sent) in qna_sentences.iter().enumerate() {
        if (i < 12) {
            println!("Q&A: {:?}", sent);
        }
    }
    
    sentences.extend(wiki_sentences);
    sentences.extend(mdx_sentences);
    sentences.extend(quote_sentences);
    sentences.extend(dict_sentences);
    sentences.extend(qna_sentences);
    // let sentences = mdx_sentences;

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
    let training_samples = prepare_samples(&sentences, &tokenizer, &keyword_index);

    for (i, sample) in training_samples.iter().enumerate() {
        if (i < 12) {
            println!("Sample: {:?}", sample.pair);
        }
    }

    // let labelled   = training_samples.iter().filter(|s| !s.matched_classes.is_empty()).count();
    // let unlabelled = training_samples.len() - labelled;
    println!(
        "Training samples: {}",
        training_samples.len()
    );

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
        .init(&device);
    let mut rng = rand::thread_rng();
    let mut final_loss = 0.0f32;
    
    // lr over time
    let first_lr = 0.0001;
    let last_lr = 1e-8;

    // Compute LR from absolute epoch number — stateless, resume-safe
    let lr_for_epoch = |abs_epoch: usize| -> f64 {
        let t = abs_epoch as f64 / epochs as f64;         // 0.0 → 1.0
        let t = t.clamp(0.0, 1.0);
        first_lr + (last_lr - first_lr) * t               // linear interp
    };

    // ── Training loop ─────────────────────────────────────────────────────────
    for epoch in 0..remaining_epochs {
        let absolute_epoch = epochs_already_done + epoch + 1;

        let current_lr = lr_for_epoch(absolute_epoch);
        // let current_lr = 1e-6;

        println!("current_lr {:?}", current_lr);

        let mut idx: Vec<usize> = (0..training_samples.len()).collect();
        use rand::seq::SliceRandom;
        idx.shuffle(&mut rng);

        let num_batches = idx.len().max(1) / batch_size;
        let pb = make_progress(num_batches, absolute_epoch, epochs);
        let mut epoch_loss = 0.0f32;

        // for batch_num in 0..num_batches {
        //     let batch_idx = &idx[batch_num * batch_size..(batch_num + 1) * batch_size];

        //     let mut batch_loss_tensors: Vec<Tensor<TrainBackend, 1>> = Vec::new();
        //     let mut batch_loss_sum = 0.0f32;

            // for &i in batch_idx {
            //     let sample  = &training_samples[i];
            //     // let seq_len = sample.input_ids.len();
            //     let seq_len = MAX_SEQ_LEN;
            //     if seq_len < 2 { continue; }

            //     let class_probs = peaked_class_probs(&sample.matched_classes, &mut rng);
            //     let ctx_flat    = build_context(&class_probs, sample.emote_label, &mut rng);

            //     let context_t = Tensor::<TrainBackend, 2>::from_floats(
            //         TensorData::new(ctx_flat, [1, CONTEXT_DIMS]),
            //         &device,
            //     );

            //     let ids_flat: Vec<i32> = sample.input_ids.iter().map(|&t| t as i32).collect();
            //     // println!("ids_flat len={}, seq_len={}", ids_flat.len(), seq_len);
            //     let ids_t = Tensor::<TrainBackend, 2, Int>::from_ints(
            //         TensorData::new(ids_flat, [1, seq_len]),
            //         &device,
            //     );

            //     let (token_logits, emote_logits, lstm_state) = model.forward(ids_t, context_t, None);

            //     // Language loss
            //     let vocab     = tokenizer.vocab_size();
            //     let logits_2d = token_logits.reshape([seq_len, vocab]);

            //     // let targets: Vec<i32> = sample.target_ids.iter().map(|&t| t as i32).collect();
            //     let targets: Vec<i32> = sample.target_labels.iter().map(|&t| t as i32).collect();

            //     let target_t  = Tensor::<TrainBackend, 1, Int>::from_ints(
            //         TensorData::new(targets, [seq_len]),
            //         &device,
            //     );
            //     let lang_loss = ce_loss.forward(logits_2d, target_t);

            //     // Emote loss — last timestep only
            //     let emote_target_t = Tensor::<TrainBackend, 1, Int>::from_ints(
            //         TensorData::new(vec![sample.emote_label as i32], [1]),
            //         &device,
            //     );
            //     let emote_loss = ce_loss.forward(emote_logits, emote_target_t);

            //     let total     = lang_loss + emote_loss.mul_scalar(EMOTE_WEIGHT);
            //     let loss_val: f32 = total.clone().inner().to_data()
            //         .to_vec::<f32>().unwrap()[0];

            //     batch_loss_sum += loss_val;
            //     batch_loss_tensors.push(total);
            // }

            // if batch_loss_tensors.is_empty() { continue; }

            // let n        = batch_loss_tensors.len() as f32;
            // let combined = batch_loss_tensors.into_iter()
            //     .reduce(|a, b| a + b)
            //     .unwrap()
            //     .div_scalar(n);

            // let grads = GradientsParams::from_grads(combined.backward(), &model);
            // model     = optimizer.step(
            //     current_lr,
            //     // 0.001,
            //     // 1e-4,
            //     // 3e-5, 
            //     // 1e-5,
            //     // 3e-6,
            //     // 3e-7,
            //     // 3e-8,
            //     // 0.001, // standard?
            //     model, 
            //     grads
            // );

            // let avg = batch_loss_sum / n;
            // epoch_loss += avg;

            // let batches_done = (batch_num + 1) as f32;
            // let avg_loss = epoch_loss / batches_done;

            // pb.set_prefix(format!("{:.4} avg_loss={:.4}", avg, avg_loss));
            // pb.inc(1);
        // }

        for batch_num in 0..num_batches {
            let batch_start = batch_num * batch_size;
            let batch_end = (batch_start + batch_size).min(training_samples.len());
            let batch_idx = &idx[batch_start..batch_end];
            let current_batch_size = batch_idx.len();  // in case last batch is smaller

            if current_batch_size == 0 { continue; }

            // ── Collect everything into flat vectors (fast) ─────────────────────
            let mut all_ids: Vec<i32> = Vec::with_capacity(current_batch_size * MAX_SEQ_LEN);
            let mut all_contexts: Vec<f32> = Vec::with_capacity(current_batch_size * CONTEXT_DIMS);
            let mut all_lang_targets: Vec<i32> = Vec::with_capacity(current_batch_size * MAX_SEQ_LEN);
            let mut all_emote_targets: Vec<i32> = Vec::with_capacity(current_batch_size);

            for &i in batch_idx {
                let sample = &training_samples[i];

                // input ids
                all_ids.extend(sample.input_ids.iter().map(|&t| t as i32));

                // context (still per-sample because of rng noise — cheap)
                let class_probs = peaked_class_probs(&sample.matched_classes, &mut rng);
                let ctx_flat = build_context(&class_probs, sample.emote_label, &mut rng);
                all_contexts.extend(ctx_flat);

                // language targets
                all_lang_targets.extend(sample.target_labels.iter().map(|&t| t as i32));

                // emote target
                all_emote_targets.push(sample.emote_label as i32);
            }

            // ── Stack into real batched tensors (ONE allocation) ───────────────
            let ids_t = Tensor::<TrainBackend, 2, Int>::from_ints(
                TensorData::new(all_ids, [current_batch_size, MAX_SEQ_LEN]),
                &device,
            );

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

            // ── SINGLE forward pass (this is where the 20× speedup happens) ─────
            let (token_logits, emote_logits, _lstm_state) = model.forward(ids_t, context_t, None);

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
                // 1e-4,
                model, 
                grads
            );

            // Logging (same feel as before)
            let loss_val: f32 = total_loss.clone().inner().to_data().to_vec::<f32>().unwrap()[0];
            epoch_loss += loss_val;

            let batches_done = (batch_num + 1) as f32;
            let avg_loss = epoch_loss / batches_done;

            pb.set_prefix(format!("{:.4} avg_loss={:.4}", loss_val, avg_loss));
            pb.inc(1);
        }

        final_loss = epoch_loss / num_batches.max(1) as f32;
        pb.finish_with_message(format!("epoch {absolute_epoch}/{epochs}  loss={final_loss:.4} current_lr={current_lr:.4}"));

        let meta = BrainMetadata {
            vocab_size:     tokenizer.vocab_size(),
            epochs_trained: absolute_epoch,
            final_loss,
        };
        model.valid().save(out_dir, &tokenizer, &meta)?;

        // --- Periodic Inference ---
        {
            let inference_model = model.valid();
            println!("\n🔍 Running inference for epoch {absolute_epoch}...");

            // Pick a few target classes to see if the model is learning the context
            // 0: apple, 3: bear, 8: bicycle, 23: cloud, 4: neutral
            let test_prompts = [
                (0, "Apple", "The apple"),
                (3, "Bear", "The bear"),
                (8, "Bicycle", "A bicycle"),
                (23, "Cloud", "The cloud"),
                (4, "Neutral", "Hello"),
            ];

            for (class_idx, label, seed) in test_prompts {
                let mut inf_rng = rand::thread_rng();
                // We use fixed context for comparison across epochs
                let class_probs = peaked_class_probs(&[class_idx], &mut inf_rng);
                let emote_probs = peaked_class_probs(&[4], &mut inf_rng); // neutral user emote

                let res = inference_model.generate(
                    &tokenizer,
                    &class_probs,
                    &emote_probs,
                    4, // neutral user emote onehot
                    seed,
                    30,
                    &device,
                );

                println!("  [{label}] {seed} -> {} (emote: {})",
                    res.reply.replace("\n", " "),
                    EMOTE_NAMES[res.yumon_emote_idx]
                );
            }
            println!();
        }
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
    target_labels: Vec<usize>,
    pair: Vec<String>
}

// fn prepare_samples(
//     sentences:     &[String],
//     tokenizer:     &TokenizerKind,
//     keyword_index: &HashMap<String, Vec<usize>>,
// ) -> Vec<Sample> {
//     let mut samples = Vec::new();
//     let mut label_counts = HashMap::new();

//     for s in sentences {
//         let encoded = tokenizer.encode(s);
//         if encoded.len() < 3 { continue; }

//         let truncated: Vec<usize> = std::iter::once(BOS_TOKEN)
//             .chain(encoded.iter().cloned().take(MAX_SEQ_LEN - 2))
//             .chain(std::iter::once(EOS_TOKEN))
//             .collect();

//         let input_ids  = truncated[..truncated.len() - 1].to_vec();
//         let target_ids = truncated[1..].to_vec();

//         let emote_label     = keyword_emote_label(s);
//         let matched_classes = matched_classes(s, keyword_index);

//         if matched_classes.len() > 0 {
//             if label_counts.get(&matched_classes[0]).is_none() {
//                 label_counts.insert(matched_classes[0], 0);
//             }

//             // we can train on non-matches for broader knowledge later on possibly
//             if let Some(current) = label_counts.get(&matched_classes[0]) {
//                 if *current < 200 {
//                     label_counts.insert(matched_classes[0], *current + 1);
//                     samples.push(Sample { input_ids, target_ids, emote_label, matched_classes });
//                 }
//             }
//         }
//     }

//     samples
// }


// fn prepare_samples(
//     sentences:     &[String],
//     tokenizer:     &TokenizerKind,
//     keyword_index: &HashMap<String, Vec<usize>>,
// ) -> Vec<Sample> {
//     let mut samples = Vec::new();

//     for pair in sentences.chunks(2) {
//         if pair.len() < 2 { continue; }

//         let input_encoded  = tokenizer.encode(&pair[0]);
//         let target_encoded = tokenizer.encode(&pair[1]);

//         if input_encoded.len() < 10 || target_encoded.len() < 10 { continue; }
//         if input_encoded.len() > 400 || target_encoded.len() > 400 { continue; }

//         let input_ids: Vec<usize> = std::iter::once(BOS_TOKEN)
//             .chain(input_encoded.iter().cloned().take(MAX_SEQ_LEN - 2))
//             .chain(std::iter::once(EOS_TOKEN))
//             .collect();

//         let target_ids: Vec<usize> = std::iter::once(BOS_TOKEN)
//             .chain(target_encoded.iter().cloned().take(MAX_SEQ_LEN - 2))
//             .chain(std::iter::once(EOS_TOKEN))
//             .collect();

//         let emote_label     = keyword_emote_label(&pair[0]);
//         let matched_classes = matched_classes(&pair[0], keyword_index);

//         // if matched_classes.is_empty() { continue; }

//         let pad = |mut v: Vec<usize>| -> Vec<usize> {
//             v.resize(MAX_SEQ_LEN, PAD_TOKEN);
//             v
//         };

//         let input_ids  = pad(input_ids);
//         let target_ids = pad(target_ids);

//         let target_labels: Vec<usize> = target_encoded.iter().cloned().take(MAX_SEQ_LEN - 1)
//             .chain(std::iter::once(EOS_TOKEN))
//             .collect();
//         let target_labels = pad(target_labels);

//         samples.push(Sample { pair: pair.to_vec(), input_ids, target_ids, emote_label, matched_classes, target_labels });
//     }

//     samples
// }


fn prepare_samples(
    sentences:     &[String],
    tokenizer:     &TokenizerKind,
    keyword_index: &HashMap<String, Vec<usize>>,
) -> Vec<Sample> {
    let mut samples = Vec::new();

    for sentence in sentences {
        let encoded = tokenizer.encode(sentence);
        if encoded.len() < 25 || encoded.len() > 30 { continue; }

        // input:  [BOS, t0, t1, ..., t_{n-1}]
        // labels: [t0,  t1, ..., t_{n-1}, EOS]  (one-ahead shift)
        let input_ids: Vec<usize> = std::iter::once(BOS_TOKEN)
            .chain(encoded.iter().cloned().take(MAX_SEQ_LEN - 1))
            .collect();

        let target_labels: Vec<usize> = encoded.iter().cloned().take(MAX_SEQ_LEN - 1)
            .chain(std::iter::once(EOS_TOKEN))
            .collect();

        let pad = |mut v: Vec<usize>| -> Vec<usize> {
            v.resize(MAX_SEQ_LEN, PAD_TOKEN);
            v
        };

        let input_ids = pad(input_ids);
        let target_labels = pad(target_labels);

        if samples.len() < 12 {
            println!("input_ids {:?}", input_ids);
            println!("target_labels {:?}", target_labels);
        }

        // assert!(input_ids.len() == MAX_SEQ_LEN, 
        //     "input_ids wrong len: {} (encoded len was {})", input_ids.len(), encoded.len());

        let emote_label     = keyword_emote_label(sentence);
        let matched_classes = matched_classes(sentence, keyword_index);

        samples.push(Sample {
            pair:            vec![sentence.clone()],
            input_ids,
            target_ids:      vec![],          // unused in completion mode
            target_labels,
            emote_label,
            matched_classes,
        });
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