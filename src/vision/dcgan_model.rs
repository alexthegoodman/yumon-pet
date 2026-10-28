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
//! z [B, 100] -> ConvT(100 -> 512, k4, s1, p0) + BN + ReLU              -> [B, 512, 4, 4]
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
//!                   -> Conv(512 -> 1, k4, s1, p0) [Logits / Sigmoid]        -> [B, 1, 1, 1]
//! ```
//!
//! Recipe details preserved from the original paper:
//! - All weights initialized from `N(0, 0.02)`.
//! - No bias in any convolution (bias is folded into BatchNorm / omitted).
//! - BatchNorm everywhere in both nets except the generator output and the
//!   discriminator's first and last layers.
//! - ReLU in the generator (tanh output), LeakyReLU(0.2) in the discriminator.
//! - Binary cross-entropy with logits objective using the non-saturating generator loss.

use burn::{
    module::{Ignored, Module},
    nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        BatchNorm, BatchNormConfig, Initializer, PaddingConfig2d,
    },
    prelude::*,
    tensor::{
        activation::{leaky_relu, relu, sigmoid, tanh},
        Tensor, TensorData,
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

        // Project + reshape latent to a 4x4 feature grid with BatchNorm.
        let project = ConvTranspose2dConfig::new([nz, ngf * 8], [4, 4])
            .with_stride([1, 1])
            .with_padding([0, 0])
            .with_bias(false)
            .with_initializer(dcgan_init())
            .init(device);
        let bn0 = BatchNormConfig::new(ngf * 8).init(device);

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
            bn0,
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
    bn0: BatchNorm<B>,
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

        let mut f = relu(self.bn0.forward(self.project.forward(z4))); // [B, ngf*8, 4, 4]

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
    /// Forward pass returning raw unactivated logits: `[batch, 1, 1, 1]`.
    pub fn forward_logits(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
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

        self.out.forward(f) // [B, 1, 1, 1] raw logits
    }

    /// Forward pass returning sigmoid probability of being real: `[batch, 1, 1, 1]` in `[0, 1]`.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        sigmoid(self.forward_logits(x))
    }
}

/// Numerically stable binary cross-entropy with logits for target = 1:
/// `-log(sigmoid(x)) = softplus(-x) = max(-x, 0) + log(1 + exp(-|x|))`.
fn bce_with_logits_target_1<B: Backend>(logits: Tensor<B, 1>) -> Tensor<B, 1> {
    let neg_x = logits.clone().neg();
    let abs_x = logits.abs();
    (neg_x.clamp_min(0.0) + (abs_x.neg().exp().add_scalar(1.0)).log()).mean()
}

/// Numerically stable binary cross-entropy with logits for target = 0:
/// `-log(1 - sigmoid(x)) = softplus(x) = max(x, 0) + log(1 + exp(-|x|))`.
fn bce_with_logits_target_0<B: Backend>(logits: Tensor<B, 1>) -> Tensor<B, 1> {
    let abs_x = logits.clone().abs();
    (logits.clamp_min(0.0) + (abs_x.neg().exp().add_scalar(1.0)).log()).mean()
}

/// DCGAN discriminator loss: `-E[log D(x)] - E[log(1 - D(G(z)))]`.
///
/// Inputs are raw discriminator logits of shape `[batch, 1, 1, 1]`.
/// Uses numerically stable softplus computation to prevent vanishing or exploding gradients.
pub fn discriminator_loss<B: Backend>(
    real_logits: Tensor<B, 4>,
    fake_logits: Tensor<B, 4>,
) -> Tensor<B, 1> {
    let batch_real = real_logits.dims()[0];
    let batch_fake = fake_logits.dims()[0];
    let real = real_logits.reshape([batch_real]);
    let fake = fake_logits.reshape([batch_fake]);

    bce_with_logits_target_1(real) + bce_with_logits_target_0(fake)
}

/// DCGAN generator loss (non-saturating): `-E[log D(G(z))]`.
///
/// Inputs are raw discriminator logits of shape `[batch, 1, 1, 1]`.
/// Uses numerically stable softplus computation: gradient magnitude never collapses to zero.
pub fn generator_loss<B: Backend>(fake_logits: Tensor<B, 4>) -> Tensor<B, 1> {
    let batch = fake_logits.dims()[0];
    let fake = fake_logits.reshape([batch]);
    bce_with_logits_target_1(fake)
}

/// Sample standard normal Gaussian vector `z ~ N(0, 1)` using Box-Muller transform.
pub fn sample_noise_vec(count: usize, rng: &mut impl rand::Rng) -> Vec<f32> {
    let mut flat = Vec::with_capacity(count);
    for _ in 0..(count + 1) / 2 {
        let u1: f32 = rng.gen_range(1e-7f32..1.0);
        let u2: f32 = rng.gen_range(0.0f32..1.0);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        flat.push(r * theta.cos());
        if flat.len() < count {
            flat.push(r * theta.sin());
        }
    }
    flat
}

/// Sample a batch of latent vectors `z ~ N(0, 1)` of shape `[batch, dim]`.
pub fn sample_noise<B: Backend>(
    batch: usize,
    dim: usize,
    device: &B::Device,
    rng: &mut impl rand::Rng,
) -> Tensor<B, 2> {
    let flat = sample_noise_vec(batch * dim, rng);
    Tensor::from_floats(TensorData::new(flat, [batch, dim]), device)
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
        let out = disc.forward(x.clone());
        assert_eq!(out.dims(), [4, 1, 1, 1]);
        let logits = disc.forward_logits(x);
        assert_eq!(logits.dims(), [4, 1, 1, 1]);
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

    #[test]
    fn test_loss_numerical_stability() {
        let device = Default::default();
        // Test with extreme logits that would previously underflow/overflow sigmoid & log
        let real_logits: Tensor<TestBackend, 4> = Tensor::from_floats([[[[50.0f32]]], [[[-50.0f32]]]], &device);
        let fake_logits: Tensor<TestBackend, 4> = Tensor::from_floats([[[[-50.0f32]]], [[[50.0f32]]]], &device);

        let d_loss = discriminator_loss(real_logits, fake_logits.clone());
        let g_loss = generator_loss(fake_logits);

        let d_val: f32 = d_loss.into_data().convert::<f32>().to_vec().unwrap()[0];
        let g_val: f32 = g_loss.into_data().convert::<f32>().to_vec().unwrap()[0];

        assert!(d_val.is_finite(), "Discriminator loss must be finite even on extreme logits: {d_val}");
        assert!(g_val.is_finite(), "Generator loss must be finite even on extreme logits: {g_val}");
        assert!(d_val >= 0.0, "Discriminator loss must be non-negative");
        assert!(g_val >= 0.0, "Generator loss must be non-negative");
    }

    #[test]
    fn test_gaussian_noise_sampling() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
        let n = 10_000;
        let samples = sample_noise_vec(n, &mut rng);
        assert_eq!(samples.len(), n);

        let mean: f32 = samples.iter().sum::<f32>() / n as f32;
        let variance: f32 = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n as f32;

        // Standard normal should have mean ~ 0.0 and variance ~ 1.0
        assert!((mean.abs()) < 0.05, "Mean should be close to 0: {mean}");
        assert!((variance - 1.0).abs() < 0.08, "Variance should be close to 1: {variance}");
    }
}
