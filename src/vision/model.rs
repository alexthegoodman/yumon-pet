/// Yumon Vision CNN — Dual-head architecture
///
/// Shared backbone:
///   Conv(3→32, k3, p1) → BN → ReLU → MaxPool(2)       [32 → 16]
///   Conv(32→64, k3, p1) → BN → ReLU → MaxPool(2)      [16 →  8]
///   Conv(64→128, k3, p1) → BN → ReLU → AdaptiveAvg    [ 8 →  4]
///   Flatten → Dense(512, ReLU) → Dropout(0.4)
///   Dense(256, ReLU)
///
/// Heads:
///   classification_head: Dense(100)   — CIFAR-100 fine labels (softmax at inference)
///   emote_head:          Dense(7)     — FER2013 emotions       (softmax at inference)

use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
        BatchNorm, BatchNormConfig,
        Dropout, DropoutConfig,
        Linear, LinearConfig,
    },
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::TensorData,
};
use serde::{Serialize, Deserialize};
use anyhow::Result;

use crate::vision::{CIFAR_CLASSES, EMOTE_CLASSES, IMG_SIZE};

// ─── Model ────────────────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct VisionModel<B: Backend> {
    // Block 1
    conv1:  Conv2d<B>,
    bn1:    BatchNorm<B, 2>,

    // Block 2
    conv2:  Conv2d<B>,
    bn2:    BatchNorm<B, 2>,

    // Block 3
    conv3:  Conv2d<B>,
    bn3:    BatchNorm<B, 2>,

    pool:       MaxPool2d,
    avg_pool:   AdaptiveAvgPool2d,
    dropout:    Dropout,

    // Shared dense layers
    fc1:    Linear<B>,
    fc2:    Linear<B>,

    // Heads
    classification_head: Linear<B>,
    emote_head:          Linear<B>,
}

#[derive(Config, Debug)]
pub struct VisionModelConfig {
    #[config(default = 0.4)]
    pub dropout_rate: f64,
}

impl VisionModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VisionModel<B> {
        VisionModel {
            conv1: Conv2dConfig::new([3, 32], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same).init(device),
            bn1:   BatchNormConfig::new(32).init(device),

            conv2: Conv2dConfig::new([32, 64], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same).init(device),
            bn2:   BatchNormConfig::new(64).init(device),

            conv3: Conv2dConfig::new([64, 128], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same).init(device),
            bn3:   BatchNormConfig::new(128).init(device),

            pool:     MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            avg_pool: AdaptiveAvgPool2dConfig::new([4, 4]).init(),
            dropout:  DropoutConfig::new(self.dropout_rate).init(),

            // 128 * 4 * 4 = 2048
            fc1: LinearConfig::new(128 * 4 * 4, 512).init(device),
            fc2: LinearConfig::new(512, 256).init(device),

            classification_head: LinearConfig::new(256, CIFAR_CLASSES).init(device),
            emote_head:          LinearConfig::new(256, EMOTE_CLASSES).init(device),
        }
    }
}

impl<B: Backend> VisionModel<B> {
    /// Forward pass.
    /// Input:  [batch, 3, IMG_SIZE, IMG_SIZE]
    /// Output: (class_logits [batch, 100], emote_logits [batch, 7])
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        use burn::tensor::activation::relu;

        // Block 1: conv → BN → ReLU → pool
        let x = relu(self.bn1.forward(self.conv1.forward(x)));
        let x = self.pool.forward(x);

        // Block 2
        let x = relu(self.bn2.forward(self.conv2.forward(x)));
        let x = self.pool.forward(x);

        // Block 3
        let x = relu(self.bn3.forward(self.conv3.forward(x)));
        let x = self.avg_pool.forward(x);

        // Flatten → [batch, 128*4*4]
        let [batch, c, h, w] = x.dims();
        let x = x.reshape([batch, c * h * w]);

        let x = self.dropout.forward(x);
        let x = relu(self.fc1.forward(x));
        let x = relu(self.fc2.forward(x));

        let class_logits = self.classification_head.forward(x.clone());
        let emote_logits = self.emote_head.forward(x);

        (class_logits, emote_logits)
    }

    /// Inference: returns softmax probability vectors.
    pub fn infer(&self, img: Tensor<B, 4>) -> (Vec<f32>, Vec<f32>) {
        let (cl, el) = self.forward(img);

        // Take first item in batch
        let [_b, nc] = cl.dims();
        let [_b2, ne] = el.dims();

        let cl = cl.slice([0..1, 0..nc]).reshape([nc]);
        let el = el.slice([0..1, 0..ne]).reshape([ne]);

        let class_probs = softmax_vec(cl.to_data().to_vec::<f32>().unwrap());
        let emote_probs = softmax_vec(el.to_data().to_vec::<f32>().unwrap());

        (class_probs, emote_probs)
    }

    // ── Checkpoint I/O ─────────────────────────────────────────────────────

    pub fn save(&self, directory: &str, metadata: &VisionMetadata) -> Result<()> {
        let dir = std::path::Path::new(directory);
        std::fs::create_dir_all(dir)?;

        let json = serde_json::to_string_pretty(metadata)?;
        std::fs::write(dir.join("metadata.json"), json)?;

        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        self.clone().save_file(dir.join("model"), &recorder)
            .map_err(|e| anyhow::anyhow!("save_file: {e:?}"))?;

        println!("✅ Vision checkpoint saved → {directory}");
        Ok(())
    }

    pub fn load(directory: &str, device: &B::Device) -> Result<Self> {
        let dir = std::path::Path::new(directory);
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let record   = recorder.load(dir.join("model").into(), device)
            .map_err(|e| anyhow::anyhow!("load: {e:?}"))?;
        let model    = VisionModelConfig::new().init::<B>(device).load_record(record);
        Ok(model)
    }
}

// ─── Metadata ─────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct VisionMetadata {
    pub epochs_trained:     usize,
    pub best_class_acc:     f32,
    pub best_emote_acc:     f32,
    pub cifar_classes:      usize,
    pub emote_classes:      usize,
}

// ─── Helper ───────────────────────────────────────────────────────────────────

fn softmax_vec(logits: Vec<f32>) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum = exps.iter().sum::<f32>();
    exps.iter().map(|e| e / sum).collect()
}
