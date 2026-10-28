//! Vanilla Deep Convolutional GAN (DCGAN) for 64x64 RGB image synthesis.
//!
//! A faithful re-implementation of the original Radford, Metz & Chintala (2015)
//! DCGAN architecture and training recipe, adapted to the Burn/WGPU backend.
//!
//! Unlike the multi-pass upscaler, this is a *single* unconditional generator
//! that maps a latent noise vector `z ~ N(0, 1)` of shape `[batch, nz]` directly
//! to a `[batch, 3, 64, 64]` image in `[-1, 1]`.
//!
//! Architecture (matching the paper's Table 1 / reference implementation):
//!
//! Generator:
//! ```text
//! z [B, 100] -> ConvT(100 -> 512, k4, s1, p0) + ReLU                  -> [B, 512, 4, 4]
//!            -> ConvT(512 -> 256, k4, s2, p1) + BN + ReLU              -> [B, 256, 8, 8]
//!            -> ConvT(256 -> 128, k4, s2, p1) + BN + ReLU              -> [B, 128, 16, 16]
//!            -> ConvT(128 ->  64, k4, s2, p1) + BN + ReLU              -> [B, 64, 32, 32]
//!            -> ConvT(64  ->   3, k4, s2, p1) + Tanh                    -> [B, 3, 64, 64]
//! ```
//!
//! Discriminator:
//! ```text
//! x [B, 3, 64, 64] -> Conv(3 -> 64, k4, s2, p1) + LeakyReLU(0.2)           -> [B, 64, 32, 32]
//!                   -> Conv(64 -> 128, k4, s2, p1) + BN + LeakyReLU(0.2)    -> [B, 128, 16, 16]
//!                   -> Conv(128 -> 256, k4, s2, p1) + BN + LeakyReLU(0.2)   -> [B, 256, 8, 8]
//!                   -> Conv(256 -> 512, k4, s2, p1) + BN + LeakyReLU(0.2)   -> [B, 512, 4, 4]
//!                   -> Conv(512 -> 1, k4, s1, p0) + Sigmoid                 -> [B, 1, 1, 1]
//! ```
//!
//! Recipe details preserved from the original paper:
//! - All weights initialized from `N(0, 0.02)`.
//! - No bias in any convolution (bias is folded into BatchNorm / omitted).
//! - BatchNorm everywhere in both nets except the generator output and the
//!   discriminator's first and last layers.
//! - ReLU in the generator (tanh output), LeakyReLU(0.2) in the discriminator.
//! - Binary cross-entropy objective with the non-saturating generator loss.

use burn::{
    module::{Ignored, Module},
    nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        BatchNorm, BatchNormConfig, Initializer, PaddingConfig2d,
    },
    prelude::*,
    tensor::{
        activation::{leaky_relu, relu, sigmoid, tanh},
        Tensor,
    },
};

/// DCGAN default latent (noise) vector length.
pub const DCGAN_NOISE_DIM: usize = 100;
/// DCGAN default generator feature-map width.
pub const DCGAN_GEN_FEATURES: usize = 64;
/// DCGAN default discriminator feature-map width.
pub const DCGAN_DISC_FEATURES: usize = 64;
/// DCGAN default output image resolution (square).
pub const DCGAN_IMAGE_SIZE: usize = 64;
/// DCGAN default number of output image channels.
pub const DCGAN_CHANNELS: usize = 3;
/// DCGAN default discriminator LeakyReLU slope.
pub const DCGAN_LRELU_SLOPE: f64 = 0.2;
/// DCGAN default weight standard deviation for `N(0, 0.02)` init.
pub const DCGAN_INIT_STD: f64 = 0.02;

/// `N(0, 0.02)` weight initializer used across both networks (Radford et al.).
fn dcgan_init() -> Initializer {
    Initializer::Normal {
        mean: 0.0,
        std: DCGAN_INIT_STD,
    }
}

/// Configuration for the DCGAN generator.
#[derive(Config, Debug)]
pub struct DcganGeneratorConfig {
    /// Length of the latent noise vector `z`.
    #[config(default = "DCGAN_NOISE_DIM")]
    pub noise_dim: usize,

    /// Generator feature-map width (`ngf`).
    #[config(default = "DCGAN_GEN_FEATURES")]
    pub ngf: usize,

    /// Number of output image channels (`nc`, 3 for RGB).
    #[config(default = "DCGAN_CHANNELS")]
    pub out_channels: usize,

    /// Square output resolution (`64`).
    #[config(default = "DCGAN_IMAGE_SIZE")]
    pub image_size: usize,
}

impl DcganGeneratorConfig {
    /// Build the generator with the DCGAN `N(0, 0.02)` init and bias-free convolutions.
    pub fn init<B: Backend>(&self, device: &B::Device) -> DcganGenerator<B> {
        let nz = self.noise_dim;
        let ngf = self.ngf;
        let nc = self.out_channels;

        // Project + reshape latent to a 4x4 feature grid (no BatchNorm here).
        let project = ConvTranspose2dConfig::new([nz, ngf * 8], [4, 4])
            .with_stride([1, 1])
            .with_padding([0, 0])
            .with_bias(false)
            .with_initializer(dcgan_init())
            .init(device);

        // 4x4 -> 8x8
        let deconv1 = ConvTranspose2dConfig::new([ngf * 8, ngf * 4], [4, 4])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .with_bias(false)
            .with_initializer(dcgan_init())
            .init(device);
        let bn1 = BatchNormConfig::new(ngf * 4).init(device);

        // 8x8 -> 16x16
        let deconv2 = ConvTranspose2dConfig::new([ngf * 4, ngf * 2], [4, 4])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .with_bias(false)
            .with_initializer(dcgan_init())
            .init(device);
        let bn2 = BatchNormConfig::new(ngf * 2).init(device);

        // 16x16 -> 32x32
        let deconv3 = ConvTranspose2dConfig::new([ngf * 2, ngf], [4, 4])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .with_bias(false)
            .with_initializer(dcgan_init())
            .init(device);
        let bn3 = BatchNormConfig::new(ngf).init(device);

        // 32x32 -> 64x64 (Tanh output, no BatchNorm)
        let out = ConvTranspose2dConfig::new([ngf, nc], [4, 4])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .with_bias(false)
            .with_initializer(dcgan_init())
            .init(device);

        DcganGenerator {
            project,
            deconv1,
            bn1,
            deconv2,
            bn2,
            deconv3,
            bn3,
            out,
            config: Ignored(self.clone()),
        }
    }
}

/// The DCGAN generator: maps latent `z` to a `[batch, 3, 64, 64]` image in `[-1, 1]`.
#[derive(Module, Debug)]
pub struct DcganGenerator<B: Backend> {
    project: ConvTranspose2d<B>,
    deconv1: ConvTranspose2d<B>,
    bn1: BatchNorm<B>,
    deconv2: ConvTranspose2d<B>,
    bn2: BatchNorm<B>,
    deconv3: ConvTranspose2d<B>,
    bn3: BatchNorm<B>,
    out: ConvTranspose2d<B>,
    pub config: Ignored<DcganGeneratorConfig>,
}

impl<B: Backend> DcganGenerator<B> {
    /// Forward pass.
    ///
    /// Input:  `[batch, noise_dim]` latent vectors.
    /// Output: `[batch, out_channels, image_size, image_size]` pixels in `[-1, 1]`.
    pub fn forward(&self, z: Tensor<B, 2>) -> Tensor<B, 4> {
        let [batch, _] = z.dims();
        let z4 = z.reshape([batch, self.config.noise_dim, 1, 1]);

        let mut f = relu(self.project.forward(z4)); // [B, ngf*8, 4, 4]

        f = relu(self.bn1.forward(self.deconv1.forward(f))); // [B, ngf*4, 8, 8]
        f = relu(self.bn2.forward(self.deconv2.forward(f))); // [B, ngf*2, 16, 16]
        f = relu(self.bn3.forward(self.deconv3.forward(f))); // [B, ngf, 32, 32]

        tanh(self.out.forward(f)) // [B, 3, 64, 64]
    }

    /// Length of the latent noise vector this generator expects.
    pub fn noise_dim(&self) -> usize {
        self.config.noise_dim
    }
}

/// Configuration for the DCGAN discriminator.
#[derive(Config, Debug)]
pub struct DcganDiscriminatorConfig {
    /// Number of input image channels (`nc`, 3 for RGB).
    #[config(default = "DCGAN_CHANNELS")]
    pub in_channels: usize,

    /// Discriminator feature-map width (`ndf`).
    #[config(default = "DCGAN_DISC_FEATURES")]
    pub ndf: usize,

    /// Square input resolution (`64`).
    #[config(default = "DCGAN_IMAGE_SIZE")]
    pub image_size: usize,
}

impl DcganDiscriminatorConfig {
    /// Build the discriminator with the DCGAN `N(0, 0.02)` init and bias-free convolutions.
    pub fn init<B: Backend>(&self, device: &B::Device) -> DcganDiscriminator<B> {
        let nc = self.in_channels;
        let ndf = self.ndf;

        // 64x64 -> 32x32 (no BatchNorm on the input layer).
        let conv1 = Conv2dConfig::new([nc, ndf], [4, 4])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false)
            .with_initializer(dcgan_init())
            .init(device);

        // 32x32 -> 16x16
        let conv2 = Conv2dConfig::new([ndf, ndf * 2], [4, 4])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false)
            .with_initializer(dcgan_init())
            .init(device);
        let bn1 = BatchNormConfig::new(ndf * 2).init(device);

        // 16x16 -> 8x8
        let conv3 = Conv2dConfig::new([ndf * 2, ndf * 4], [4, 4])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false)
            .with_initializer(dcgan_init())
            .init(device);
        let bn2 = BatchNormConfig::new(ndf * 4).init(device);

        // 8x8 -> 4x4
        let conv4 = Conv2dConfig::new([ndf * 4, ndf * 8], [4, 4])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false)
            .with_initializer(dcgan_init())
            .init(device);
        let bn3 = BatchNormConfig::new(ndf * 8).init(device);

        // 4x4 -> 1x1 (Sigmoid probability, no BatchNorm).
        let out = Conv2dConfig::new([ndf * 8, 1], [4, 4])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_bias(false)
            .with_initializer(dcgan_init())
            .init(device);

        DcganDiscriminator {
            conv1,
            conv2,
            bn1,
            conv3,
            bn2,
            conv4,
            bn3,
            out,
            config: Ignored(self.clone()),
        }
    }
}

/// The DCGAN discriminator: a binary classifier outputting a single sigmoid probability.
#[derive(Module, Debug)]
pub struct DcganDiscriminator<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    bn1: BatchNorm<B>,
    conv3: Conv2d<B>,
    bn2: BatchNorm<B>,
    conv4: Conv2d<B>,
    bn3: BatchNorm<B>,
    out: Conv2d<B>,
    pub config: Ignored<DcganDiscriminatorConfig>,
}

impl<B: Backend> DcganDiscriminator<B> {
    /// Forward pass.
    ///
    /// Input:  `[batch, in_channels, image_size, image_size]`.
    /// Output: `[batch, 1, 1, 1]` sigmoid probability of being real.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut f = leaky_relu(self.conv1.forward(x), DCGAN_LRELU_SLOPE); // [B, ndf, 32, 32]

        f = leaky_relu(
            self.bn1.forward(self.conv2.forward(f)),
            DCGAN_LRELU_SLOPE,
        ); // [B, ndf*2, 16, 16]
        f = leaky_relu(
            self.bn2.forward(self.conv3.forward(f)),
            DCGAN_LRELU_SLOPE,
        ); // [B, ndf*4, 8, 8]
        f = leaky_relu(
            self.bn3.forward(self.conv4.forward(f)),
            DCGAN_LRELU_SLOPE,
        ); // [B, ndf*8, 4, 4]

        sigmoid(self.out.forward(f)) // [B, 1, 1, 1]
    }
}

/// DCGAN discriminator loss: `-E[log D(x)] - E[log(1 - D(G(z)))]`.
///
/// Inputs are already-sigmoid discriminator outputs of shape `[batch, 1, 1, 1]`.
/// Returns a scalar loss tensor.
pub fn discriminator_loss<B: Backend>(
    real_out: Tensor<B, 4>,
    fake_out: Tensor<B, 4>,
) -> Tensor<B, 1> {
    let batch = real_out.dims()[0];
    let real = real_out.reshape([batch]);
    let fake = fake_out.reshape([batch]);

    let eps = 1e-8f32;
    let log_real = real.clamp_min(eps).log().mean();
    let log_inv_fake = fake
        .neg()
        .add_scalar(1.0)
        .clamp_min(eps)
        .log()
        .mean();

    (log_real + log_inv_fake).neg()
}

/// DCGAN generator loss (non-saturating): `-E[log D(G(z))]`.
///
/// Inputs are already-sigmoid discriminator outputs of shape `[batch, 1, 1, 1]`.
/// Returns a scalar loss tensor.
pub fn generator_loss<B: Backend>(fake_out: Tensor<B, 4>) -> Tensor<B, 1> {
    let batch = fake_out.dims()[0];
    fake_out.reshape([batch]).clamp_min(1e-8).log().mean().neg()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_cubecl::CubeBackend;
    use cubecl::wgpu::WgpuRuntime;

    type TestBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;

    #[test]
    fn test_generator_output_shape() {
        let device = Default::default();
        let generator: DcganGenerator<TestBackend> = DcganGeneratorConfig::new().init(&device);

        let z: Tensor<TestBackend, 2> = Tensor::zeros([4, DCGAN_NOISE_DIM], &device);
        let out = generator.forward(z);
        assert_eq!(out.dims(), [4, DCGAN_CHANNELS, DCGAN_IMAGE_SIZE, DCGAN_IMAGE_SIZE]);
    }

    #[test]
    fn test_generator_custom_ngf_shape() {
        let device = Default::default();
        let generator: DcganGenerator<TestBackend> = DcganGeneratorConfig::new()
            .with_ngf(32)
            .init(&device);

        let z: Tensor<TestBackend, 2> = Tensor::zeros([2, DCGAN_NOISE_DIM], &device);
        let out = generator.forward(z);
        assert_eq!(out.dims(), [2, DCGAN_CHANNELS, DCGAN_IMAGE_SIZE, DCGAN_IMAGE_SIZE]);
    }

    #[test]
    fn test_discriminator_output_shape() {
        let device = Default::default();
        let disc: DcganDiscriminator<TestBackend> = DcganDiscriminatorConfig::new().init(&device);

        let x: Tensor<TestBackend, 4> =
            Tensor::zeros([4, DCGAN_CHANNELS, DCGAN_IMAGE_SIZE, DCGAN_IMAGE_SIZE], &device);
        let out = disc.forward(x);
        assert_eq!(out.dims(), [4, 1, 1, 1]);
    }

    #[test]
    fn test_generator_grayscale_output() {
        let device = Default::default();
        let generator: DcganGenerator<TestBackend> = DcganGeneratorConfig::new()
            .with_out_channels(1)
            .init(&device);

        let z: Tensor<TestBackend, 2> = Tensor::zeros([1, DCGAN_NOISE_DIM], &device);
        let out = generator.forward(z);
        assert_eq!(out.dims(), [1, 1, DCGAN_IMAGE_SIZE, DCGAN_IMAGE_SIZE]);
    }
}
