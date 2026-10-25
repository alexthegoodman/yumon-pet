//! 2D Convolutional Neural Upscaler for Pass 4: 32x32 -> 256x256 (8x Super-Resolution).
//!
//! Uses a lightweight Deep Residual feature extractor at 32x32, followed by a
//! progressive 3-stage transposed convolution pyramid (32 -> 64 -> 128 -> 256).
//!
//! Optimized for high-throughput GPU training on WGPU/CubeCL and pure L1 loss.

use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        PaddingConfig2d,
    },
    prelude::*,
    tensor::{activation::relu, Tensor},
};

/// Residual block with two 3x3 convolutions and 0.2 residual scaling.
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
        let res = relu(self.conv1.forward(x.clone()));
        x + self.conv2.forward(res) * 0.2
    }
}

/// Configuration for the Pass 4 Upscaler model.
#[derive(Config, Debug)]
pub struct UpscalerModelConfig {
    #[config(default = 32)]
    pub channels: usize,
}

impl UpscalerModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> UpscalerModel<B> {
        let c = self.channels;
        let c_half = (c / 2).max(16);

        // Feature extraction at native 32x32 resolution
        let stem = Conv2dConfig::new([3, c], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        let res1 = ResBlock::new(c, device);
        let res2 = ResBlock::new(c, device);
        let res3 = ResBlock::new(c, device);

        // Progressive 3-stage upsampling (32 -> 64 -> 128 -> 256)
        let up1 = ConvTranspose2dConfig::new([c, c], [4, 4])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .init(device);

        let up2 = ConvTranspose2dConfig::new([c, c_half], [4, 4])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .init(device);

        let up3 = ConvTranspose2dConfig::new([c_half, c_half], [4, 4])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .init(device);

        let out_conv = Conv2dConfig::new([c_half, 3], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        UpscalerModel {
            stem,
            res1,
            res2,
            res3,
            up1,
            up2,
            up3,
            out_conv,
        }
    }
}

/// The Pass 4 Generative Upscaler (32x32 -> 256x256).
#[derive(Module, Debug)]
pub struct UpscalerModel<B: Backend> {
    stem: Conv2d<B>,
    res1: ResBlock<B>,
    res2: ResBlock<B>,
    res3: ResBlock<B>,
    up1: ConvTranspose2d<B>,
    up2: ConvTranspose2d<B>,
    up3: ConvTranspose2d<B>,
    out_conv: Conv2d<B>,
}

impl<B: Backend> UpscalerModel<B> {
    /// Forward pass:
    /// Input:  `[batch, 3, 32, 32]` (RGB values in [-1.0, 1.0])
    /// Output: `[batch, 3, 256, 256]` (Super-resolved RGB values in [-1.0, 1.0])
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let f = relu(self.stem.forward(x));
        let f = self.res1.forward(f);
        let f = self.res2.forward(f);
        let f = self.res3.forward(f);

        let f = relu(self.up1.forward(f)); // 64x64
        let f = relu(self.up2.forward(f)); // 128x128
        let f = relu(self.up3.forward(f)); // 256x256

        burn::tensor::activation::tanh(self.out_conv.forward(f))
    }
}
