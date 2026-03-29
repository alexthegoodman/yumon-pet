/// Yumon Brain training loop

use anyhow::Result;
use burn::{
    grad_clipping::GradientClippingConfig, module::AutodiffModule, nn::loss::CrossEntropyLossConfig, optim::{AdamConfig, AdamWConfig, GradientsParams, Optimizer}, prelude::*, tensor::{Int, TensorData, backend::AutodiffBackend}
};
use rand::{Rng, rngs::StdRng, seq::SliceRandom, thread_rng};
use indicatif::{ProgressBar, ProgressStyle};
use ratatui::{Terminal, TerminalOptions, Viewport, prelude::CrosstermBackend};
use std::collections::HashMap;
use rand::SeedableRng;

use crate::{brain::{PAD_TOKEN, bpe::{BpeTokenizer, CL_ID, CR_ID, TokenizerKind}, chart::{TrainingState, render}, loader::{DataLoader, FileKind}, mdx::{load_csv_bible, load_csv_qna, load_csv_quotes, load_dictionary_sentences, load_handcrafted_sentences, load_mdx_sentences, load_notion_sentences, load_qa_pairs, load_qa_singles, load_txt_sentences}, pdf::{load_pdf_ebook_sentences, load_pdfs}, samples::{TrainingStage, WorldContext, prepare_paired_samples_split, prepare_paired_samples_split_sep}, wiki::save_sentences_to_file}, vision::{CIFAR_CLASSES, EMOTE_CLASSES, EMOTE_NAMES}};
use crate::brain::{
    // CONTEXT_DIMS,
    tokenizer::{Tokenizer, BOS_TOKEN, EOS_TOKEN},
    model::{YumonBrain, YumonBrainConfig, BrainMetadata, GenerationResult},
    wiki::load_wiki_sentences,
};

pub type TrainBackend = burn::backend::Autodiff<burn::backend::Wgpu>;
// pub type TrainBackend = burn::backend::Autodiff<burn::backend::NdArray<f32>>;

// Max sequence length during training (tokens)
// pub const MAX_SEQ_LEN:  usize = 120;
// pub const MAX_SEQ_LEN:  usize = 25;
// pub const MAX_SEQ_LEN:  usize = 512;
// pub const MAX_SEQ_LEN:  usize = 1024;
// pub const MAX_SEQ_LEN:  usize = 256;
pub const MAX_SEQ_LEN_CHARS:  usize = 180;
pub const MAX_SEQ_LEN:  usize = 180;
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

pub fn run(
    wiki_xml:          &str,
    _vision_checkpoint: &str,  // reserved for future real-image fine-tuning
    out_dir:           &str,
    epochs:            usize,
    batch_size:        usize,
    max_articles:      usize,
) -> Result<()> {
    // let device: <TrainBackend as Backend>::Device = Default::default();
    let device = burn::backend::wgpu::WgpuDevice::default();

    // ── Build label keyword index ─────────────────────────────────────────────
    let label_keywords   = build_label_keywords();
    let keyword_index    = build_keyword_index(&label_keywords);
    println!("🏷  CIFAR-100 keyword index: {} unique keywords across {} classes",
             keyword_index.len(), CIFAR_CLASSES);

    let tokenizer = TokenizerKind::Bpe(BpeTokenizer::load("yumon_bpe")?);

    let training_stage = TrainingStage::Language;
    // let training_stage = TrainingStage::Structured;
 
    let training_samples = DataLoader::new(training_stage)
        // per-file limits — None means "take everything"
        .add("data/(poems)/",                   FileKind::Mdx,         None)
        .add("data/bible_bbe.csv",              FileKind::BibleCsv,    Some(1024))
        .add("archive/handcrafted.txt",         FileKind::Handcrafted, None)
        .add("archive/handcrafted_pairs.txt",   FileKind::QaPairs, None)
        .add("data/qa_journal.txt",             FileKind::QaPairs, None)
        // global cap after merging all sources
        .total_limit(4096)
        // reproducible seed (same default as before: 4815162342)
        .seed(4815162342)
        .load(&tokenizer, &keyword_index)?;
    
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
        .with_weight_decay(0.01)
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
    // let first_lr = 0.003; // even better for 8?
    let first_lr = 0.001; // batch sizes like 8
    // let first_lr = 0.0001; // for batch size 4?
    // let first_lr = 0.000003;
    // let first_lr = 3e-6; // flat immediately
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
        TerminalOptions { viewport: Viewport::Inline(26) },
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
        entropy: 0.0,
        entropy_history: vec![],
        last_reply: String::new()
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
            // let mut all_contexts: Vec<f32> = Vec::with_capacity(current_batch_size * CONTEXT_DIMS);
            let mut all_lang_targets: Vec<i32> = Vec::with_capacity(current_batch_size * MAX_SEQ_LEN);
            // let mut all_emote_targets: Vec<i32> = Vec::with_capacity(current_batch_size);
            let mut all_enc_ids: Vec<i32> = Vec::with_capacity(current_batch_size * MAX_SEQ_LEN);
            // let mut all_dec_ids: Vec<i32> = Vec::with_capacity(current_batch_size * MAX_SEQ_LEN);
            let mut all_dec_input_ids: Vec<i32> = Vec::with_capacity(current_batch_size * MAX_SEQ_LEN);

            for &i in batch_idx {
                let sample = &training_samples[i];

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

                // input_ids is the full sequence [BOS, sent_a..., sent_b..., EOS, PAD...]
                // target_labels is already shifted  [sent_a..., sent_b..., EOS, PAD...]
                // all_dec_input_ids.extend(sample.input_ids.iter().map(|&t| t as i32));
                // all_lang_targets.extend(sample.target_labels.iter().map(|&t| t as i32));
            }

            // ── Stack into real batched tensors (ONE allocation) ───────────────
            // let ids_t = Tensor::<TrainBackend, 2, Int>::from_ints(
            //     TensorData::new(all_ids, [current_batch_size, MAX_SEQ_LEN]),
            //     &device,
            // );

            // let context_t = Tensor::<TrainBackend, 2>::from_floats(
            //     TensorData::new(all_contexts, [current_batch_size, CONTEXT_DIMS]),
            //     &device,
            // );

            let lang_target_t = Tensor::<TrainBackend, 1, Int>::from_ints(
                TensorData::new(all_lang_targets, [current_batch_size * MAX_SEQ_LEN]),
                &device,
            );

            // let emote_target_t = Tensor::<TrainBackend, 1, Int>::from_ints(
            //     TensorData::new(all_emote_targets, [current_batch_size]),
            //     &device,
            // );

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

            let token_logits = model.forward(enc_t, dec_t.clone());
            // let token_logits = model.forward(dec_t);

            // ── Entropy (predictive) ──────────────────────────────────────────────
            let probs = burn::tensor::activation::softmax(token_logits.clone(), 2); // [B, S, vocab]
            let log_probs = (probs.clone() + 1e-10).log();                          // avoid log(0)

            let token_entropy = (probs * log_probs)
                .sum_dim(2)
                .neg()
                .squeeze::<2>();  // [B, S, 1] → [B, S]

            let non_pad_mask = dec_t
                .clone()
                .equal_elem(PAD_TOKEN as u32)
                .bool_not()
                .float();  // [B, S]

            let entropy_val: f32 = (token_entropy * non_pad_mask.clone())
                .sum()
                .div(non_pad_mask.sum())
                .into_scalar();

            // ── Loss (now automatically batched) ───────────────────────────────
            let vocab = tokenizer.vocab_size();
            let logits_2d = token_logits.reshape([current_batch_size * MAX_SEQ_LEN, vocab]);  // [B*S, vocab]

            let lang_loss = ce_loss.forward(logits_2d, lang_target_t);

            // let emote_loss = ce_loss.forward(emote_logits, emote_target_t);  // [B, C] and [B]

            // let total_loss = lang_loss + emote_loss.mul_scalar(EMOTE_WEIGHT);
            let total_loss  = lang_loss;

            // // Backward + step (exactly one per real batch)
            let grads = GradientsParams::from_grads(total_loss.backward(), &model);
            model = optimizer.step(
                current_lr, 
                // 1e-6,
                model, 
                grads
            );

            // // Before step
            // let w_before: Vec<f32> = model.token_head.weight.val()
            //     .to_data().to_vec().unwrap();

            // let grads = GradientsParams::from_grads(total_loss.backward(), &model);
            // model = optimizer.step(current_lr, model, grads);

            // // After step  
            // let w_after: Vec<f32> = model.token_head.weight.val()
            //     .to_data().to_vec().unwrap();

            // println!("weight delta: {:?}", 
            //     w_before.iter().zip(&w_after)
            //     .map(|(a,b)| (b-a).abs())
            //     .take(5)
            //     .collect::<Vec<_>>()
            // );

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
            state.entropy = entropy_val;
            state.current_loss = loss_val * loss_scale; // Don't scale the actual loss used for backprop, just the display value.
            state.avg_loss = (epoch_loss * loss_scale) / (batch_num + 1) as f32; // Don't scale the actual loss used for backprop, just the display value.
            state.batch = batch_num + 1;
            state.current_lr = current_lr;
            state.global_step += 1;
            state.loss_history.push((state.global_step as f64, loss_val as f64 * loss_scale as f64)); // Don't scale the actual loss used for backprop, just the display value.
            state.avg_loss_history.push((state.global_step as f64, state.avg_loss as f64));
            state.entropy_history.push((state.global_step as f64, entropy_val as f64));

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

        // periodic inference
        {
            let inference_model = model.valid();

            let prompt_text = "What is the universe?".to_string();

            let prompt = if training_stage == TrainingStage::Structured { 
                serde_json::to_string_pretty(&serde_json::json!({
                    "obstacle_dir": "none",
                    "building_dir": "none",
                    "resource_dir": "none",
                    "message":      prompt_text,
                }))
                .unwrap() 
            } else {
                prompt_text
            };

            let result = inference_model.generate_unmasked_parsed(
                &tokenizer,
                &prompt,
                MAX_SEQ_LEN,
                &device,
            );
            
            state.last_reply = if training_stage == TrainingStage::Structured { result.reply } else { result.raw_output };
        }
    }

    // after the epoch loop
    terminal.clear()?;

    println!("✅ Brain training complete. Final loss: {final_loss:.4}");
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