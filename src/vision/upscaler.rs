//! 2D Convolutional Neural Upscaler: 64x64/32x32 -> 256x256.
//!
//! Takes a 3-channel RGB (or 1-channel grayscale) input at 64x64 (or 32x32) and produces
//! a crisp 3-channel RGB (or 1-channel grayscale) super-resolved output at 256x256.
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

    #[config(default = 3)]
    pub in_channels: usize,

    #[config(default = 3)]
    pub out_channels: usize,

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

        let stem = Conv2dConfig::new([self.in_channels, c], [3, 3])
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

        // Final output convolution
        let out_conv = Conv2dConfig::new([c_half, self.out_channels], [3, 3])
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

/// Configuration for the PatchGAN Discriminator for 256x256 images.
#[derive(Config, Debug)]
pub struct UpscalerDiscriminatorConfig {
    #[config(default = 3)]
    pub in_channels: usize,

    #[config(default = 32)]
    pub base_channels: usize,
}

impl UpscalerDiscriminatorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> UpscalerDiscriminator<B> {
        let c = self.base_channels;
        let conv1 = Conv2dConfig::new([self.in_channels, c], [4, 4])
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

/// Default FSQ quantization levels: [8, 6, 5, 5] -> 8 * 6 * 5 * 5 = 1,200 codebook states.
pub fn default_fsq_levels() -> Vec<usize> {
    vec![8, 6, 5, 5]
}

/// Finite Scalar Quantizer (FSQ).
///
/// Projects continuous latent activations to discrete levels using bounded tanh and
/// Straight-Through Estimator (STE) rounding.
///
/// Features:
/// - Zero codebook collapse: discrete levels are fixed and uniformly distributed.
/// - No commitment loss or auxiliary codebook losses required.
/// - Pure element-wise vectorized GPU computation on CubeCL/WGPU.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FsqQuantizer {
    pub levels: Vec<usize>,
}

impl FsqQuantizer {
    pub fn new(levels: Vec<usize>) -> Self {
        assert!(!levels.is_empty(), "FSQ levels cannot be empty");
        for &l in &levels {
            assert!(l >= 2, "FSQ level must be >= 2");
        }
        Self { levels }
    }

    /// Number of latent channels required (one channel per level dimension).
    pub fn num_channels(&self) -> usize {
        self.levels.len()
    }

    /// Total number of discrete codebook states (product of all dimension levels).
    pub fn codebook_size(&self) -> usize {
        self.levels.iter().product()
    }

    /// Forward pass: quantizes continuous activations [batch, d, H, W] to discrete levels in [-1.0, 1.0].
    pub fn forward<B: Backend>(&self, z: Tensor<B, 4>) -> Tensor<B, 4> {
        let device = z.device();
        let levels_f32: Vec<f32> = self.levels.iter().map(|&l| l as f32).collect();
        let half_widths: Vec<f32> = levels_f32.iter().map(|&l| (l - 1.0) / 2.0).collect();
        let inv_widths: Vec<f32> = half_widths.iter().map(|&w| 1.0 / w).collect();

        let s: Tensor<B, 4> = Tensor::<B, 1>::from_floats(half_widths.as_slice(), &device)
            .reshape([1, self.levels.len(), 1, 1]);
        let s_inv: Tensor<B, 4> = Tensor::<B, 1>::from_floats(inv_widths.as_slice(), &device)
            .reshape([1, self.levels.len(), 1, 1]);

        let u = tanh(z) * s;
        let u_round = u.clone().round();
        let u_ste = u.clone() + (u_round - u).detach();
        u_ste * s_inv
    }
}

/// Configuration for the FSQ Codebook Quantized Upscaler (64x64 -> 256x256).
#[derive(Config, Debug)]
pub struct CodebookUpscalerModelConfig {
    #[config(default = 64)]
    pub scales_from: usize,

    #[config(default = 3)]
    pub in_channels: usize,

    #[config(default = 3)]
    pub out_channels: usize,

    #[config(default = 64)]
    pub channels: usize,

    #[config(default = 4)]
    pub num_res_blocks: usize,

    #[config(default = 1)]
    pub num_refine_blocks_per_layer: usize,

    #[config(default = "default_fsq_levels()")]
    pub fsq_levels: Vec<usize>,
}

impl CodebookUpscalerModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CodebookUpscalerModel<B> {
        let c = self.channels;
        let c_half = (c / 2).max(16);
        let fsq_ch = self.fsq_levels.len();

        let stem = Conv2dConfig::new([self.in_channels, c], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        let res_blocks = make_res_blocks(self.num_res_blocks, c, device);

        // FSQ Quantization Bottleneck
        let to_fsq = Conv2dConfig::new([c, fsq_ch], [1, 1]).init(device);
        let fsq = Ignored(FsqQuantizer::new(self.fsq_levels.clone()));
        let from_fsq = Conv2dConfig::new([fsq_ch, c], [1, 1]).init(device);
        let post_fsq_blocks = make_res_blocks(self.num_refine_blocks_per_layer.max(1), c, device);

        // Optional 32 -> 64 upsampling
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

        // Final output convolution
        let out_conv = Conv2dConfig::new([c_half, self.out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        CodebookUpscalerModel {
            stem,
            res_blocks,
            to_fsq,
            fsq,
            from_fsq,
            post_fsq_blocks,
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

/// The FSQ Codebook Quantized Upscaler (64x64/32x32 -> 256x256).
///
/// Enforces discrete codebook quantization in the latent bottleneck to eliminate
/// regression-to-the-mean blur and synthesize sharp, plausible high-frequency details.
#[derive(Module, Debug)]
pub struct CodebookUpscalerModel<B: Backend> {
    stem: Conv2d<B>,
    res_blocks: Vec<ResBlock<B>>,

    to_fsq: Conv2d<B>,
    pub fsq: Ignored<FsqQuantizer>,
    from_fsq: Conv2d<B>,
    post_fsq_blocks: Vec<ResBlock<B>>,

    up0: Option<ConvTranspose2d<B>>,
    refine_blocks0: Vec<ResBlock<B>>,

    up1: ConvTranspose2d<B>,
    refine_blocks1: Vec<ResBlock<B>>,

    up2: ConvTranspose2d<B>,
    refine_blocks2: Vec<ResBlock<B>>,

    out_conv: Conv2d<B>,

    pub config: Ignored<CodebookUpscalerModelConfig>,
}

impl<B: Backend> CodebookUpscalerModel<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut f = relu(self.stem.forward(x));

        for block in &self.res_blocks {
            f = block.forward(f);
        }

        // Discrete FSQ Bottleneck: forces discrete prototype representation
        let latent = self.to_fsq.forward(f);
        let quantized = self.fsq.forward(latent);
        f = relu(self.from_fsq.forward(quantized));

        for block in &self.post_fsq_blocks {
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

    /// Extract discrete quantized latents [batch, num_levels, H, W].
    pub fn encode_latents(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut f = relu(self.stem.forward(x));
        for block in &self.res_blocks {
            f = block.forward(f);
        }
        let latent = self.to_fsq.forward(f);
        self.fsq.forward(latent)
    }

    /// Decode from discrete quantized latents [batch, num_levels, H, W] -> 256x256.
    pub fn decode_latents(&self, quantized: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut f = relu(self.from_fsq.forward(quantized));
        for block in &self.post_fsq_blocks {
            f = block.forward(f);
        }

        if let Some(up0) = &self.up0 {
            f = relu(up0.forward(f));
            for block in &self.refine_blocks0 {
                f = block.forward(f);
            }
        }

        f = relu(self.up1.forward(f));
        for block in &self.refine_blocks1 {
            f = block.forward(f);
        }

        f = relu(self.up2.forward(f));
        for block in &self.refine_blocks2 {
            f = block.forward(f);
        }

        tanh(self.out_conv.forward(f))
    }
}

/// Unified enum wrapper supporting both Legacy continuous Upscaler and new Codebook Upscaler.
pub enum AnyUpscaler<B: Backend> {
    Standard(UpscalerModel<B>),
    Codebook(CodebookUpscalerModel<B>),
}

impl<B: Backend> AnyUpscaler<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Self::Standard(m) => m.forward(x),
            Self::Codebook(m) => m.forward(x),
        }
    }

    pub fn in_channels(&self) -> usize {
        match self {
            Self::Standard(m) => m.config.in_channels,
            Self::Codebook(m) => m.config.in_channels,
        }
    }

    pub fn out_channels(&self) -> usize {
        match self {
            Self::Standard(m) => m.config.out_channels,
            Self::Codebook(m) => m.config.out_channels,
        }
    }

    pub fn is_codebook(&self) -> bool {
        matches!(self, Self::Codebook(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_cubecl::CubeBackend;
    use cubecl::wgpu::WgpuRuntime;

    type TestBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;

    #[test]
    fn test_tensor_round() {
        let device = Default::default();
        let t: Tensor<TestBackend, 1> = Tensor::from_floats([1.2, 2.7, -1.6], &device);
        let _r = t.round();
    }

    #[test]
    fn test_vectorized_fsq() {
        let device = Default::default();
        let fsq = FsqQuantizer::new(vec![8, 6, 5, 5]);
        assert_eq!(fsq.num_channels(), 4);
        assert_eq!(fsq.codebook_size(), 1200);

        let z: Tensor<TestBackend, 4> = Tensor::zeros([1, 4, 4, 4], &device);
        let z_q = fsq.forward(z);

        let out = z_q.into_data();
        let vals: Vec<f32> = out.convert::<f32>().to_vec().unwrap();
        assert_eq!(vals.len(), 64);
        for v in vals {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_codebook_upscaler_forward() {
        let device = Default::default();
        let config = CodebookUpscalerModelConfig::new()
            .with_channels(16)
            .with_in_channels(3)
            .with_out_channels(3)
            .with_num_res_blocks(1)
            .with_num_refine_blocks_per_layer(0);

        let model: CodebookUpscalerModel<TestBackend> = config.init(&device);
        let input: Tensor<TestBackend, 4> = Tensor::zeros([1, 3, 64, 64], &device);
        let output = model.forward(input);
        assert_eq!(output.dims(), [1, 3, 256, 256]);
    }
}


