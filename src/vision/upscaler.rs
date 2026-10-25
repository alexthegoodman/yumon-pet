//! 2D Convolutional Neural Upscaler for Pass 4: 32x32 -> 256x256 (8x Super-Resolution).
//!
//! Replaces heavy autoregressive sequence decoding with a lightweight, feed-forward
//! deep residual convolutional network. 
//!
//! Features:
//! - 3-stage progressive upsampling (32 -> 64 -> 128 -> 256)
//! - Deep Residual Backbone with long skip connection
//! - Gradient / edge preservation loss for high-frequency micro-textures

use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        PaddingConfig2d,
    },
    prelude::*,
    tensor::{activation::relu, Tensor},
};

/// Residual block with two 3x3 convolutions and a residual scaling factor.
#[derive(Module, Debug)]
pub struct ResBlock<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
}

impl<B: Backend> ResBlock<B> {
    pub fn new(channels: usize, device: &B::Device) -> Self {
        Self {
            conv1: Conv2dConfig::new([channels, channels], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv2: Conv2dConfig::new([channels, channels], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let res = self.conv1.forward(x.clone());
        let res = relu(res);
        let res = self.conv2.forward(res);
        // Residual addition with 0.2 scaling for training stability
        x + res * 0.2
    }
}

/// 2x Upsampling block using transposed convolution and refinement convolution.
#[derive(Module, Debug)]
pub struct UpsampleBlock<B: Backend> {
    deconv: ConvTranspose2d<B>,
    refine: Conv2d<B>,
}

impl<B: Backend> UpsampleBlock<B> {
    pub fn new(channels: usize, device: &B::Device) -> Self {
        Self {
            // Kernel 4x4, stride 2, padding 1 performs exact 2x spatial enlargement
            deconv: ConvTranspose2dConfig::new([channels, channels], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device),
            refine: Conv2dConfig::new([channels, channels], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.deconv.forward(x);
        let x = relu(x);
        let x = self.refine.forward(x);
        relu(x)
    }
}

/// Configuration for the Pass 4 Upscaler model.
#[derive(Config, Debug)]
pub struct UpscalerModelConfig {
    #[config(default = 64)]
    pub channels: usize,
    #[config(default = 6)]
    pub num_res_blocks: usize,
}

impl UpscalerModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> UpscalerModel<B> {
        let stem = Conv2dConfig::new([3, self.channels], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        let mut res_blocks = Vec::with_capacity(self.num_res_blocks);
        for _ in 0..self.num_res_blocks {
            res_blocks.push(ResBlock::new(self.channels, device));
        }

        let conv_after_res = Conv2dConfig::new([self.channels, self.channels], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        let up1 = UpsampleBlock::new(self.channels, device);
        let up2 = UpsampleBlock::new(self.channels, device);
        let up3 = UpsampleBlock::new(self.channels, device);

        let out_conv = Conv2dConfig::new([self.channels, 3], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        UpscalerModel {
            stem,
            res_blocks,
            conv_after_res,
            up1,
            up2,
            up3,
            out_conv,
        }
    }
}

/// The Pass 4 Feedforward Generative Upscaler (32x32 -> 256x256).
#[derive(Module, Debug)]
pub struct UpscalerModel<B: Backend> {
    stem: Conv2d<B>,
    res_blocks: Vec<ResBlock<B>>,
    conv_after_res: Conv2d<B>,
    up1: UpsampleBlock<B>,
    up2: UpsampleBlock<B>,
    up3: UpsampleBlock<B>,
    out_conv: Conv2d<B>,
}

impl<B: Backend> UpscalerModel<B> {
    /// Forward pass:
    /// Input:  `[batch, 3, 32, 32]` (RGB values normalized to [-1.0, 1.0])
    /// Output: `[batch, 3, 256, 256]` (Super-resolved RGB values in [-1.0, 1.0])
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // 1. Initial feature extraction
        let f0 = self.stem.forward(x);
        let f0 = relu(f0);

        // 2. Residual backbone
        let mut f = f0.clone();
        for block in &self.res_blocks {
            f = block.forward(f);
        }

        // 3. Long skip connection
        let f = self.conv_after_res.forward(f) + f0;

        // 4. Progressive 3-stage upsampling (32 -> 64 -> 128 -> 256)
        let f = self.up1.forward(f);
        let f = self.up2.forward(f);
        let f = self.up3.forward(f);

        // 5. Output RGB projection (clamped to [-1.0, 1.0] via tanh)
        let out = self.out_conv.forward(f);
        burn::tensor::activation::tanh(out)
    }

    /// Compute composite training loss:
    /// Loss = L1_pixel_loss + 0.5 * High_frequency_gradient_loss
    pub fn loss(&self, pred: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
        let [_b, _c, h, w] = pred.dims();

        // 1. Charbonnier / Smooth L1 loss: sqrt((pred - target)^2 + eps^2)
        let diff = pred.clone() - target.clone();
        let charbonnier = (diff.clone().powf_scalar(2.0) + 1e-6).sqrt().mean();

        // 2. High-Frequency Gradient Loss (finite differences along X and Y)
        // Horizontal difference: [B, C, H, W-1]
        let pred_dx = pred.clone().slice([0.._b, 0.._c, 0..h, 1..w])
            - pred.clone().slice([0.._b, 0.._c, 0..h, 0..w - 1]);
        let target_dx = target.clone().slice([0.._b, 0.._c, 0..h, 1..w])
            - target.clone().slice([0.._b, 0.._c, 0..h, 0..w - 1]);
        let grad_x_loss = (pred_dx - target_dx).abs().mean();

        // Vertical difference: [B, C, H-1, W]
        let pred_dy = pred.clone().slice([0.._b, 0.._c, 1..h, 0..w])
            - pred.clone().slice([0.._b, 0.._c, 0..h - 1, 0..w]);
        let target_dy = target.clone().slice([0.._b, 0.._c, 1..h, 0..w])
            - target.clone().slice([0.._b, 0.._c, 0..h - 1, 0..w]);
        let grad_y_loss = (pred_dy - target_dy).abs().mean();

        let grad_loss = (grad_x_loss + grad_y_loss) * 0.5;

        // Composite Loss: fidelity + edge crispness
        charbonnier + grad_loss * 0.5
    }
}
