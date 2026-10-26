//! 2D Convolutional Neural Upscaler: 64x64/32x32 -> 256x256 (Grayscale to Grayscale).
//!
//! Takes a 1-channel grayscale input at 64x64 (or 32x32) and produces a crisp
//! 1-channel grayscale super-resolved output at 256x256.
//!
//! Lightweight, ultra-fast convolutional residual architecture optimized for
//! high-frequency edge and texture synthesis on WGPU/CubeCL with Huber and LSGAN losses.

use burn::{
    module::{Ignored, Module},
    nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        PaddingConfig2d,
    },
    prelude::*,
    tensor::{
        activation::{leaky_relu, relu, tanh},
        Tensor,
    },
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

/// Configuration for the Pass 4/5 Upscaler model.
#[derive(Config, Debug)]
pub struct UpscalerModelConfig {
    #[config(default = 64)]
    pub scales_from: usize,

    #[config(default = 32)]
    pub channels: usize,

    #[config(default = 3)]
    pub num_res_blocks: usize,

    #[config(default = 0)]
    pub num_refine_blocks_per_layer: usize,
}

impl UpscalerModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> UpscalerModel<B> {
        let c = self.channels;
        let c_half = (c / 2).max(16);

        // 1-channel grayscale input
        let stem = Conv2dConfig::new([1, c], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        let res_blocks = make_res_blocks(
            self.num_res_blocks,
            c,
            device,
        );

        // If scales_from <= 32, we add an initial 2x upsampling stage (32 -> 64)
        let (up0, refine_blocks0) = if self.scales_from <= 32 {
            let up = ConvTranspose2dConfig::new([c, c], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device);
            let refine = make_res_blocks(self.num_refine_blocks_per_layer, c, device);
            (Some(up), refine)
        } else {
            (None, Vec::new())
        };

        // Stage 1: 64 -> 128
        let up1 = ConvTranspose2dConfig::new([c, c_half], [4, 4])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .init(device);
        let refine_blocks1 = make_res_blocks(self.num_refine_blocks_per_layer, c_half, device);

        // Stage 2: 128 -> 256
        let up2 = ConvTranspose2dConfig::new([c_half, c_half], [4, 4])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .init(device);
        let refine_blocks2 = make_res_blocks(self.num_refine_blocks_per_layer, c_half, device);

        // Final output convolution to 1-channel grayscale
        let out_conv = Conv2dConfig::new([c_half, 1], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        UpscalerModel {
            stem,
            res_blocks,
            up0,
            refine_blocks0,
            up1,
            refine_blocks1,
            up2,
            refine_blocks2,
            out_conv,
            config: Ignored(self.clone()),
        }
    }
}

fn make_res_blocks<B: Backend>(
    count: usize,
    channels: usize,
    device: &B::Device,
) -> Vec<ResBlock<B>> {
    (0..count)
        .map(|_| ResBlock::new(channels, device))
        .collect()
}

/// The Generative Upscaler (64x64/32x32 -> 256x256 Grayscale).
#[derive(Module, Debug)]
pub struct UpscalerModel<B: Backend> {
    stem: Conv2d<B>,
    res_blocks: Vec<ResBlock<B>>,

    up0: Option<ConvTranspose2d<B>>,
    refine_blocks0: Vec<ResBlock<B>>,

    up1: ConvTranspose2d<B>,
    refine_blocks1: Vec<ResBlock<B>>,

    up2: ConvTranspose2d<B>,
    refine_blocks2: Vec<ResBlock<B>>,

    out_conv: Conv2d<B>,

    pub config: Ignored<UpscalerModelConfig>,
}

impl<B: Backend> UpscalerModel<B> {
    /// Forward pass:
    /// Input:  `[batch, 1, H, W]` (Grayscale values in [-1.0, 1.0])
    /// Output: `[batch, 1, 256, 256]` (Super-resolved Grayscale values in [-1.0, 1.0])
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut f = relu(self.stem.forward(x));

        for block in &self.res_blocks {
            f = block.forward(f);
        }

        // Optional 32 -> 64 upsampling
        if let Some(up0) = &self.up0 {
            f = relu(up0.forward(f));
            for block in &self.refine_blocks0 {
                f = block.forward(f);
            }
        }

        // 64 -> 128 upsampling
        f = relu(self.up1.forward(f));
        for block in &self.refine_blocks1 {
            f = block.forward(f);
        }

        // 128 -> 256 upsampling
        f = relu(self.up2.forward(f));
        for block in &self.refine_blocks2 {
            f = block.forward(f);
        }

        tanh(self.out_conv.forward(f))
    }
}

/// Configuration for the PatchGAN Discriminator for 256x256 grayscale images.
#[derive(Config, Debug)]
pub struct UpscalerDiscriminatorConfig {
    #[config(default = 32)]
    pub base_channels: usize,
}

impl UpscalerDiscriminatorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> UpscalerDiscriminator<B> {
        let c = self.base_channels;
        // 1-channel grayscale input
        let conv1 = Conv2dConfig::new([1, c], [4, 4])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let conv2 = Conv2dConfig::new([c, c * 2], [4, 4])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let conv3 = Conv2dConfig::new([c * 2, c * 4], [4, 4])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let conv4 = Conv2dConfig::new([c * 4, c * 8], [4, 4])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let out_conv = Conv2dConfig::new([c * 8, 1], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        UpscalerDiscriminator {
            conv1,
            conv2,
            conv3,
            conv4,
            out_conv,
        }
    }
}

/// PatchGAN Discriminator for 256x256 grayscale images.
/// Evaluates overlapping local patches and outputs a spatial grid of logits [batch, 1, 16, 16].
#[derive(Module, Debug)]
pub struct UpscalerDiscriminator<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    conv4: Conv2d<B>,
    out_conv: Conv2d<B>,
}

impl<B: Backend> UpscalerDiscriminator<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let (logits, _) = self.forward_with_features(x);
        logits
    }

    /// Forward pass returning patch logits and intermediate feature maps for feature-matching loss.
    pub fn forward_with_features(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Vec<Tensor<B, 4>>) {
        let f1 = leaky_relu(self.conv1.forward(x), 0.2);
        let f2 = leaky_relu(self.conv2.forward(f1.clone()), 0.2);
        let f3 = leaky_relu(self.conv3.forward(f2.clone()), 0.2);
        let f4 = leaky_relu(self.conv4.forward(f3.clone()), 0.2);
        let logits = self.out_conv.forward(f4.clone());

        (logits, vec![f1, f2, f3, f4])
    }
}
