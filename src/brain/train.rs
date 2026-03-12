/// Yumon Brain training loop — Language model pre-training on SimpleWiki.
///
/// Training objective:
///   Standard next-character prediction (cross-entropy) over each sentence,
///   with a randomly sampled vision context vector injected at every timestep.
///
/// Vision context during pre-training:
///   We don't have paired image–text data yet, so we sample random class/emote
///   probability vectors from a Dirichlet distribution. This trains the LSTM
///   to condition on vision context without requiring paired data.
///   Fine-tuning with real pairs comes later.
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

use crate::vision::{CIFAR_CLASSES, EMOTE_CLASSES};
use crate::brain::{
    CONTEXT_DIMS,
    tokenizer::{Tokenizer, BOS_TOKEN, EOS_TOKEN},
    model::{YumonBrain, YumonBrainConfig, BrainMetadata},
    wiki::load_wiki_sentences,
};

pub type TrainBackend = burn::backend::Autodiff<burn::backend::NdArray<f32>>;

// Max sequence length during training (characters)
const MAX_SEQ_LEN: usize = 120;
// Max vocab size
const MAX_VOCAB:   usize = 256;
// Emote head loss weight (much lighter than language loss)
const EMOTE_WEIGHT: f32  = 0.2;

pub fn run(
    wiki_xml:          &str,
    _vision_checkpoint: &str,  // reserved for future fine-tuning
    out_dir:           &str,
    epochs:            usize,
    batch_size:        usize,
    max_articles:      usize,
) -> Result<()> {
    let device: <TrainBackend as Backend>::Device = Default::default();

    // ── Load + tokenize wiki corpus ───────────────────────────────────────────
    let sentences = load_wiki_sentences(wiki_xml, max_articles)?;

    // Build vocabulary from entire corpus
    let full_text: String = sentences.join(" ");
    println!("Building vocabulary from {} chars...", full_text.len());
    let tokenizer = Tokenizer::build_from_text(&full_text, MAX_VOCAB);
    println!("Vocabulary size: {}", tokenizer.vocab_size);

    // Tokenize all sentences into (input_ids, target_ids, emote_label) triples
    let training_samples = prepare_samples(&sentences, &tokenizer);
    println!("Training samples: {}", training_samples.len());

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
        // Shuffle sample indices
        let mut idx: Vec<usize> = (0..training_samples.len()).collect();
        use rand::seq::SliceRandom;
        idx.shuffle(&mut rng);

        let num_batches = idx.len().max(1) / batch_size;
        let pb = make_progress(num_batches, epoch + 1, epochs);
        let mut epoch_loss = 0.0f32;

        for batch_num in 0..num_batches {
            let batch_idx = &idx[batch_num * batch_size..(batch_num + 1) * batch_size];

            // For simplicity, process each sample individually (seq lens vary).
            // A production version would bucket by length and pad.
            let mut batch_loss_tensors: Vec<Tensor<TrainBackend, 1>> = Vec::new();
            let mut batch_loss_sum = 0.0f32;

            for &i in batch_idx {
                let (ref input_ids, ref target_ids, emote_label) = training_samples[i];
                let seq_len = input_ids.len();
                if seq_len < 2 { continue; }

                // Random vision context (Dirichlet-like via normalised exponentials)
                let ctx_flat = random_context(&mut rng);
                let context_t = Tensor::<TrainBackend, 2>::from_floats(
                    TensorData::new(ctx_flat, [1, CONTEXT_DIMS]),
                    &device,
                );

                // Token IDs → [1, seq_len]
                let ids_flat: Vec<i32> = input_ids.iter().map(|&t| t as i32).collect();
                let ids_t = Tensor::<TrainBackend, 2, Int>::from_ints(
                    TensorData::new(ids_flat, [1, seq_len]),
                    &device,
                );

                let (token_logits, emote_logits) = model.forward(ids_t, context_t);

                // Language loss — predict target_ids at each position
                // token_logits: [1, seq_len, vocab] → [seq_len, vocab]
                let vocab = tokenizer.vocab_size;
                let logits_2d = token_logits.reshape([seq_len, vocab]);
                let targets: Vec<i32> = target_ids.iter().map(|&t| t as i32).collect();
                let target_t = Tensor::<TrainBackend, 1, Int>::from_ints(
                    TensorData::new(targets, [seq_len]),
                    &device,
                );
                let lang_loss = ce_loss.forward(logits_2d, target_t);

                // Emote loss — at last timestep
                let emote_target_t = Tensor::<TrainBackend, 1, Int>::from_ints(
                    TensorData::new(vec![emote_label as i32], [1]),
                    &device,
                );
                let emote_loss = ce_loss.forward(emote_logits, emote_target_t);

                let total = lang_loss + emote_loss.mul_scalar(EMOTE_WEIGHT);
                let loss_val: f32 = total.clone().inner().to_data()
                    .to_vec::<f32>().unwrap()[0];

                batch_loss_sum += loss_val;
                batch_loss_tensors.push(total);
            }

            if batch_loss_tensors.is_empty() { continue; }

            let n = batch_loss_tensors.len() as f32;
            let combined = batch_loss_tensors.into_iter()
                .reduce(|a, b| a + b)
                .unwrap()
                .div_scalar(n);

            let grads = GradientsParams::from_grads(combined.backward(), &model);
            model = optimizer.step(3e-4, model, grads);

            let avg = batch_loss_sum / n;
            epoch_loss += avg;
            pb.set_prefix(format!("{:.4}", avg));
            pb.inc(1);
        }

        final_loss = epoch_loss / num_batches.max(1) as f32;
        pb.finish_with_message(format!("epoch {}/{epochs}  loss={final_loss:.4}", epoch + 1));

        // Save checkpoint after every epoch
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

/// (input_token_ids, target_token_ids, emote_label)
type Sample = (Vec<usize>, Vec<usize>, usize);

fn prepare_samples(sentences: &[String], tokenizer: &Tokenizer) -> Vec<Sample> {
    let mut samples = Vec::new();
    for s in sentences {
        let encoded = tokenizer.encode(s);
        if encoded.len() < 3 { continue; }

        // Truncate to MAX_SEQ_LEN
        let truncated: Vec<usize> = std::iter::once(BOS_TOKEN)
            .chain(encoded.iter().cloned().take(MAX_SEQ_LEN - 2))
            .chain(std::iter::once(EOS_TOKEN))
            .collect();

        let input_ids: Vec<usize> = truncated[..truncated.len() - 1].to_vec();
        let target_ids: Vec<usize> = truncated[1..].to_vec();

        // Heuristic emote label based on keywords
        let emote_label = keyword_emote_label(s);

        samples.push((input_ids, target_ids, emote_label));
    }
    samples
}

/// Very simple keyword-based pseudo-label for emote head pre-training.
fn keyword_emote_label(text: &str) -> usize {
    let lower = text.to_lowercase();
    // 0=angry, 1=disgust, 2=fear, 3=happy, 4=neutral, 5=sad, 6=surprise
    if lower.contains("war") || lower.contains("attack") || lower.contains("conflict") {
        0 // angry
    } else if lower.contains("poison") || lower.contains("disease") || lower.contains("waste") {
        1 // disgust
    } else if lower.contains("danger") || lower.contains("threat") || lower.contains("risk") {
        2 // fear
    } else if lower.contains("celebrat") || lower.contains("award") || lower.contains("success") {
        3 // happy
    } else if lower.contains("death") || lower.contains("loss") || lower.contains("victim") {
        5 // sad
    } else if lower.contains("discover") || lower.contains("unexpect") || lower.contains("sudden") {
        6 // surprise
    } else {
        4 // neutral (most wiki text)
    }
}

/// Sample a random context vector from a Dirichlet-like distribution.
/// Returns a Vec<f32> of length CONTEXT_DIMS = 114.
fn random_context(rng: &mut impl Rng) -> Vec<f32> {
    let mut ctx = Vec::with_capacity(CONTEXT_DIMS);

    // class_probs: random simplex point over 100 classes
    ctx.extend(random_simplex(rng, CIFAR_CLASSES));
    // emote_probs: random simplex over 7 emotes (skew toward neutral)
    ctx.extend(random_simplex(rng, EMOTE_CLASSES));
    // user_emote_onehot: pick one at random
    let chosen = rng.gen_range(0..EMOTE_CLASSES);
    let mut onehot = vec![0.0f32; EMOTE_CLASSES];
    onehot[chosen] = 1.0;
    ctx.extend(onehot);

    ctx
}

fn random_simplex(rng: &mut impl Rng, n: usize) -> Vec<f32> {
    let raw: Vec<f32> = (0..n).map(|_| rng.r#gen::<f32>().max(1e-9).ln().neg()).collect();
    let sum: f32 = raw.iter().sum();
    raw.iter().map(|x| x / sum).collect()
}

trait Neg { fn neg(self) -> Self; }
impl Neg for f32 { fn neg(self) -> Self { -self } }

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
