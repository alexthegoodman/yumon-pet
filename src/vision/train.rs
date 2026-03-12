/// Vision CNN training loop.
///
/// Strategy:
///   - Each batch alternates between a CIFAR-100 sub-batch and a FER2013 sub-batch.
///   - CIFAR loss  = cross-entropy on classification_head
///   - FER loss    = cross-entropy on emote_head
///   - Total loss  = cifar_loss + fer_loss  (equal weighting)
///
/// The shared backbone learns general visual representations while each head
/// specialises on its own task simultaneously.

use anyhow::Result;
use burn::{
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::{backend::AutodiffBackend, Int, TensorData},
    module::AutodiffModule,
};
use rand::seq::SliceRandom;
use rand::thread_rng;
use indicatif::{ProgressBar, ProgressStyle};

use super::{
    cifar::{CifarDataset, collate_batch as cifar_collate},
    fer::{FerDataset, collate_batch as fer_collate},
    model::{VisionModel, VisionModelConfig, VisionMetadata},
    IMG_SIZE, CIFAR_CLASSES, EMOTE_CLASSES,
};

pub type TrainBackend = burn::backend::Autodiff<burn::backend::NdArray<f32>>;

pub fn run(
    cifar_dir:  &str,
    fer_dir:    &str,
    out_dir:    &str,
    epochs:     usize,
    batch_size: usize,
) -> Result<()> {
    let device: <TrainBackend as Backend>::Device = Default::default();

    // ── Load datasets ─────────────────────────────────────────────────────────
    let cifar_train = CifarDataset::load(&format!("{cifar_dir}/train.bin"))?;
    let fer_train   = FerDataset::load(&format!("{fer_dir}/train"))?;
    let fer_test    = FerDataset::load(&format!("{fer_dir}/test"))?;

    println!("CIFAR-100 train: {} | FER train: {} | FER test: {}",
             cifar_train.len(), fer_train.len(), fer_test.len());

    // ── Init model + optimizer ────────────────────────────────────────────────
    let mut model: VisionModel<TrainBackend> = VisionModelConfig::new().init(&device);
    let mut optimizer = AdamConfig::new()
        .with_epsilon(1e-7)
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(1e-4)))
        .init();

    let ce_loss = CrossEntropyLossConfig::new().init(&device);

    let mut best_emote_acc = 0.0f32;

    // ── Training loop ─────────────────────────────────────────────────────────
    for epoch in 0..epochs {
        let mut cifar_idx: Vec<usize> = (0..cifar_train.len()).collect();
        let mut fer_idx:   Vec<usize> = (0..fer_train.len()).collect();
        cifar_idx.shuffle(&mut thread_rng());
        fer_idx.shuffle(&mut thread_rng());

        let num_batches = cifar_idx.len().max(fer_idx.len()) / batch_size;
        let pb = ProgressBar::new(num_batches as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner} epoch {msg} [{bar:40}] {pos}/{len} loss={prefix}")
            .unwrap());
        pb.set_message(format!("{}/{epochs}", epoch + 1));

        let mut epoch_loss   = 0.0f32;
        let mut cifar_cursor = 0usize;
        let mut fer_cursor   = 0usize;

        for _batch in 0..num_batches {
            // ── CIFAR sub-batch ───────────────────────────────────────────
            let ci_end = (cifar_cursor + batch_size).min(cifar_idx.len());
            let ci_batch: Vec<&_> = cifar_idx[cifar_cursor..ci_end]
                .iter().map(|&i| &cifar_train.records[i]).collect();
            cifar_cursor = ci_end % cifar_idx.len();

            let (ci_pixels, ci_labels) = cifar_collate(&ci_batch);
            let bs = ci_batch.len();

            let img_t = Tensor::<TrainBackend, 4>::from_floats(
                TensorData::new(ci_pixels, [bs, 3, IMG_SIZE, IMG_SIZE]),
                &device,
            );
            let lbl_t = Tensor::<TrainBackend, 1, Int>::from_ints(
                TensorData::new(ci_labels.iter().map(|&x| x as i32).collect::<Vec<_>>(), [bs]),
                &device,
            );

            let (class_logits, _emote_logits) = model.forward(img_t.clone());
            let cifar_loss = ce_loss.forward(class_logits, lbl_t);

            // ── FER sub-batch ─────────────────────────────────────────────
            let fe_end = (fer_cursor + batch_size).min(fer_idx.len());
            let fe_batch: Vec<&_> = fer_idx[fer_cursor..fe_end]
                .iter().map(|&i| &fer_train.records[i]).collect();
            fer_cursor = fe_end % fer_idx.len();

            let (fe_pixels, fe_labels) = fer_collate(&fe_batch);
            let fbs = fe_batch.len();

            let fer_img_t = Tensor::<TrainBackend, 4>::from_floats(
                TensorData::new(fe_pixels, [fbs, 3, IMG_SIZE, IMG_SIZE]),
                &device,
            );
            let fer_lbl_t = Tensor::<TrainBackend, 1, Int>::from_ints(
                TensorData::new(fe_labels.iter().map(|&x| x as i32).collect::<Vec<_>>(), [fbs]),
                &device,
            );

            let (_, emote_logits) = model.forward(fer_img_t);
            let fer_loss = ce_loss.forward(emote_logits, fer_lbl_t);

            // ── Combined loss + step ──────────────────────────────────────
            let total_loss = cifar_loss + fer_loss;
            let loss_val: f32 = total_loss.clone().inner().to_data()
                .to_vec::<f32>().unwrap()[0];
            epoch_loss += loss_val;

            let grads  = GradientsParams::from_grads(total_loss.backward(), &model);
            model = optimizer.step(3e-4, model, grads);

            pb.set_prefix(format!("{:.4}", loss_val));
            pb.inc(1);
        }

        let avg_loss = epoch_loss / num_batches as f32;
        pb.finish_with_message(format!("epoch {}/{epochs} done — avg loss {avg_loss:.4}", epoch + 1));

        // ── Emote validation accuracy ─────────────────────────────────────
        let emote_acc = eval_emote_acc(&model, &fer_test, batch_size, &device);
        println!("  emote val acc: {:.2}%", emote_acc * 100.0);

        if emote_acc > best_emote_acc {
            best_emote_acc = emote_acc;
            let meta = VisionMetadata {
                epochs_trained:  epoch + 1,
                best_class_acc:  0.0, // placeholder; add CIFAR eval if desired
                best_emote_acc,
                cifar_classes:   CIFAR_CLASSES,
                emote_classes:   EMOTE_CLASSES,
            };
            model.valid().save(out_dir, &meta)?;
        }
    }

    println!("✅ Vision training complete. Best emote acc: {:.2}%", best_emote_acc * 100.0);
    Ok(())
}

// ─── Evaluation ───────────────────────────────────────────────────────────────

fn eval_emote_acc<B: AutodiffBackend>(
    model:      &VisionModel<B>,
    dataset:    &FerDataset,
    batch_size: usize,
    device:     &B::Device,
) -> f32 {
    let valid  = model.clone().valid();
    let mut correct = 0usize;
    let mut total   = 0usize;

    let idx: Vec<usize> = (0..dataset.len()).collect();

    for chunk in idx.chunks(batch_size) {
        let batch: Vec<&_> = chunk.iter().map(|&i| &dataset.records[i]).collect();
        let (pixels, labels) = fer_collate(&batch);
        let bs = batch.len();

        let img_t = Tensor::<B, 4>::from_floats(
            TensorData::new(pixels, [bs, 3, IMG_SIZE, IMG_SIZE]),
            device,
        );

        let (_, emote_logits) = valid.forward(img_t.inner());
        let preds: Vec<f32>   = emote_logits.to_data().to_vec().unwrap();

        for (i, lbl) in labels.iter().enumerate() {
            let logits_slice = &preds[i * EMOTE_CLASSES..(i + 1) * EMOTE_CLASSES];
            let pred = argmax(logits_slice);
            if pred == *lbl { correct += 1; }
            total += 1;
        }
    }

    correct as f32 / total as f32
}

fn argmax(arr: &[f32]) -> usize {
    arr.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}
