// flow.rs - Rectified Flow (Flow Matching), 30-step Euler ODE Sampler, and EMA tracking for VisionDiT

use std::collections::VecDeque;
use std::path::Path;
use anyhow::Result;
use burn::{
    backend::autodiff::Autodiff,
    grad_clipping::GradientClippingConfig,
    module::{Module, ModuleMapper, ModuleVisitor, Param},
    optim::{AdamW, AdamWConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{Distribution, backend::AutodiffBackend},
};

use super::dit::{VisionDiT, VisionDiTConfig};
use crate::brain::flash_attn::backend::FlashAttention;

// ─── Rectified Flow (Flow Matching) ──────────────────────────────────────────

pub struct RectifiedFlow;

impl RectifiedFlow {
    /// Computes the Flow Matching loss:
    ///   x_t = (1 - t) x_0 + t z_1
    ///   v_target = z_1 - x_0
    ///   L = MSE(v_theta(x_t, t) - v_target)
    pub fn training_step<B: FlashAttention>(
        model: &VisionDiT<B>,
        x_0: Tensor<B, 4>,
    ) -> (Tensor<B, 1>, Tensor<B, 4>, Tensor<B, 4>) {
        let [b, c, h, w] = x_0.dims();
        let device = x_0.device();

        // Sample Gaussian noise z_1 ~ N(0, I)
        let z_1 = Tensor::<B, 4>::random([b, c, h, w], Distribution::Normal(0.0, 1.0), &device);

        // Sample timesteps t ~ Uniform(0, 1)
        let t = Tensor::<B, 1>::random([b], Distribution::Uniform(0.0, 1.0), &device);

        // Linear interpolation: x_t = (1 - t) x_0 + t z_1
        let t_expand = t.clone().reshape([b, 1, 1, 1]);
        let x_t = x_0.clone() * (t_expand.clone().neg() + 1.0) + z_1.clone() * t_expand;

        // Target velocity: d(x_t)/dt = z_1 - x_0
        let v_target = z_1 - x_0;

        // Model prediction
        let v_pred = model.forward(x_t, t);

        // MSE velocity loss
        let loss = (v_pred.clone() - v_target.clone()).powf_scalar(2.0).mean();

        (loss, v_pred, v_target)
    }
}

// ─── 30-Step Euler ODE Sampler ────────────────────────────────────────────────

pub struct EulerSampler {
    pub num_steps: usize,
}

impl Default for EulerSampler {
    fn default() -> Self {
        Self { num_steps: 30 }
    }
}

impl EulerSampler {
    pub fn new(num_steps: usize) -> Self {
        Self { num_steps }
    }

    /// Runs Euler integration from t = 1.0 (pure Gaussian noise) down to t = 0.0 (image space).
    /// Returns generated images [B, 3, 64, 64] clamped to [-1.0, 1.0].
    pub fn sample<B: FlashAttention>(
        &self,
        model: &VisionDiT<B>,
        batch_size: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let [h, w, c] = [model.img_size, model.img_size, model.in_channels];
        let mut x = Tensor::<B, 4>::random([batch_size, c, h, w], Distribution::Normal(0.0, 1.0), device);

        let dt = 1.0_f32 / (self.num_steps as f32);

        for step in 0..self.num_steps {
            let t_val = 1.0_f32 - (step as f32) * dt;
            let t_vec = vec![t_val; batch_size];
            let t_tensor = Tensor::<B, 1>::from_data(
                burn::tensor::TensorData::new(t_vec, [batch_size]),
                device,
            );

            // Velocity dx/dt = v_theta(x, t)
            let v = model.forward(x.clone(), t_tensor);

            // Step backwards towards data: x_{t - dt} = x_t - v * dt
            x = x - v * dt;
        }

        // Clamp to valid pixel range [-1.0, 1.0]
        x.clamp(-1.0, 1.0)
    }

    /// Reconstructs a clean sample starting from a specific noisy point z_1 (for overfit tests).
    pub fn sample_from<B: FlashAttention>(
        &self,
        model: &VisionDiT<B>,
        z_1: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let [batch_size, _, _, _] = z_1.dims();
        let device = z_1.device();
        let mut x = z_1;
        let dt = 1.0_f32 / (self.num_steps as f32);

        for step in 0..self.num_steps {
            let t_val = 1.0_f32 - (step as f32) * dt;
            let t_vec = vec![t_val; batch_size];
            let t_tensor = Tensor::<B, 1>::from_data(
                burn::tensor::TensorData::new(t_vec, [batch_size]),
                &device,
            );

            let v = model.forward(x.clone(), t_tensor);
            x = x - v * dt;
        }

        x.clamp(-1.0, 1.0)
    }
}

// ─── Exponential Moving Average (EMA) Tracker ─────────────────────────────────

struct TensorCollector<B: Backend> {
    tensors: VecDeque<Tensor<B, 1>>,
}

impl<B: Backend> ModuleVisitor<B> for TensorCollector<B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let n: usize = param.val().dims().iter().product();
        self.tensors.push_back(param.val().detach().reshape([n]));
    }
}

struct EmaApplier<B: Backend> {
    src_tensors: VecDeque<Tensor<B, 1>>,
    decay: f64,
}

impl<B: Backend> ModuleMapper<B> for EmaApplier<B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let src = self.src_tensors.pop_front().expect("matched parameter count").reshape(param.val().dims());
        let updated = (param.val().detach() * self.decay + src * (1.0 - self.decay)).detach();
        Param::from_tensor(updated)
    }
}

pub struct EmaTracker<B: Backend> {
    pub decay: f64,
    pub ema_model: VisionDiT<B>,
}

impl<B: Backend> EmaTracker<B> {
    pub fn new(model: &VisionDiT<B>, decay: f64) -> Self {
        Self {
            decay,
            ema_model: model.clone(),
        }
    }

    /// Updates shadow EMA weights:
    ///   theta_EMA <- decay * theta_EMA + (1 - decay) * theta
    pub fn update(&mut self, current_model: &VisionDiT<B>) {
        let mut collector = TensorCollector { tensors: VecDeque::new() };
        current_model.visit(&mut collector);

        let mut applier = EmaApplier {
            src_tensors: collector.tensors,
            decay: self.decay,
        };
        self.ema_model = self.ema_model.clone().map(&mut applier);
    }

    /// Saves the EMA model checkpoint to a binary file.
    pub fn save_checkpoint(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        recorder.record(self.ema_model.clone().into_record(), path.to_path_buf())?;
        Ok(())
    }

    /// Loads EMA weights from a binary file.
    pub fn load_checkpoint(config: &VisionDiTConfig, path: &Path, device: &B::Device) -> Result<VisionDiT<B>> {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        let record = recorder.load(path.to_path_buf(), device)?;
        let model = config.init(device).load_record(record);
        Ok(model)
    }
}

// ─── Trainer Routine ─────────────────────────────────────────────────────────

pub struct DiTTrainer<B: AutodiffBackend + FlashAttention> {
    pub model: VisionDiT<B>,
    pub ema: EmaTracker<B>,
    pub optimizer: OptimizerAdaptor<AdamW, VisionDiT<B>, B>,
    pub lr: f64,
}

impl<B: AutodiffBackend + FlashAttention> DiTTrainer<B> {
    pub fn new(
        config: &VisionDiTConfig,
        lr: f64,
        weight_decay: f32,
        grad_clip: f32,
        ema_decay: f64,
        device: &B::Device,
    ) -> Self {
        let model = config.init(device);
        let ema = EmaTracker::new(&model, ema_decay);
        let optimizer = AdamWConfig::new()
            .with_weight_decay(weight_decay)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(grad_clip)))
            .init();

        Self {
            model,
            ema,
            optimizer,
            lr,
        }
    }

    /// Performs a single optimization step on a batch of [B, 3, 64, 64] images.
    /// Returns the loss scalar as f32.
    pub fn step(&mut self, batch: Tensor<B, 4>) -> f32 {
        let (loss, _, _) = RectifiedFlow::training_step(&self.model, batch);
        let loss_val = loss.clone().into_scalar().elem::<f32>();

        let grads = GradientsParams::from_grads(loss.backward(), &self.model);
        self.model = self.optimizer.step(self.lr, self.model.clone(), grads);

        // Update EMA shadow weights
        self.ema.update(&self.model);

        loss_val
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::autodiff::{Autodiff, checkpoint::strategy::BalancedCheckpointing};
    use burn_cubecl::CubeBackend;
    use cubecl::wgpu::WgpuRuntime;

    type Inner = CubeBackend<WgpuRuntime, f32, i32, u32>;
    type AD = Autodiff<Inner, BalancedCheckpointing>;

    #[test]
    fn test_rectified_flow_step_and_euler_sampling() {
        let device = Default::default();
        let config = VisionDiTConfig::micro();
        let mut trainer = DiTTrainer::<AD>::new(
            &config,
            1e-3,
            0.0,
            1.0,
            0.99,
            &device,
        );

        let batch_size = 2;
        let images = Tensor::<AD, 4>::random(
            [batch_size, 3, 64, 64],
            Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        let initial_loss = trainer.step(images.clone());
        assert!(initial_loss.is_finite(), "initial loss must be finite: {initial_loss}");

        let sampler = EulerSampler::new(5);
        let samples = sampler.sample(&trainer.ema.ema_model, batch_size, &device);
        assert_eq!(samples.dims(), [batch_size, 3, 64, 64]);
        let max_val = samples.clone().abs().max().into_scalar().elem::<f32>();
        assert!(max_val <= 1.0 + 1e-4, "sampled values must be within [-1, 1], got {max_val}");
    }

    #[test]
    fn test_single_batch_overfit() {
        let device = Default::default();
        let config = VisionDiTConfig::micro();
        let mut trainer = DiTTrainer::<AD>::new(
            &config,
            1e-3,
            0.0,
            1.0,
            0.999,
            &device,
        );

        let data_dir = std::path::Path::new("data/DiffusionDb/images-64");
        let images = if data_dir.exists() {
            let files = crate::vision::discover_images(data_dir).unwrap();
            let take = files.into_iter().take(4).collect::<Vec<_>>();
            crate::vision::load_image_batch::<AD>(&take, &device).unwrap()
        } else {
            Tensor::<AD, 4>::random([4, 3, 64, 64], Distribution::Uniform(-1.0, 1.0), &device)
        };

        let mut first_loss = 0.0f32;
        let mut last_loss = 0.0f32;
        for step in 0..40 {
            let loss = trainer.step(images.clone());
            if step == 0 {
                first_loss = loss;
            }
            last_loss = loss;
        }

        assert!(first_loss.is_finite());
        assert!(last_loss.is_finite());
        assert!(last_loss < first_loss, "loss must drop steadily: {last_loss} < {first_loss}");

        // 30-step Euler sampler reconstructs from EMA shadow weights
        let sampler = EulerSampler::new(30);
        let samples = sampler.sample(&trainer.ema.ema_model, 4, &device);
        assert_eq!(samples.dims(), [4, 3, 64, 64]);
    }
}
