//! 2D Convolutional Neural Upscaler for Pass 4: 64x64/32x32 -> 256x256.
//!
//! Supports:
//! - Multi-scale inputs (`scales_from`: 64 or 32, default 64).
//! - Joint colorization & super-resolution (`colorizes`: bool, default true: 1-channel grayscale -> 3-channel RGB).
//! - Cross-attention text conditioning (`use_text_conditioning`: bool, default true)
//!   allowing BPE prompt embeddings to steer high-frequency texture and color synthesis.
//! - Progressive transposed convolution pyramid with residual refinement.
//! - Optimized for high-throughput GPU training on WGPU/CubeCL with Huber and LSGAN losses.

use burn::{
    module::{Ignored, Module},
    nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        Embedding, EmbeddingConfig, Linear, LinearConfig,
        PaddingConfig2d,
    },
    prelude::*,
    tensor::{
        activation::{leaky_relu, relu, softmax, tanh},
        Int, Tensor,
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

/// Multi-Head Cross-Attention module.
/// Projects visual spatial tokens (queries) and text conditioning tokens (keys/values),
/// computing scaled dot-product cross attention.
#[derive(Config, Debug)]
pub struct CrossAttentionConfig {
    pub visual_dim: usize,
    pub text_dim: usize,
    #[config(default = 4)]
    pub n_heads: usize,
}

impl CrossAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CrossAttention<B> {
        let n_heads = self.n_heads.min(self.visual_dim).max(1);
        let n_heads = if self.visual_dim % n_heads == 0 {
            n_heads
        } else {
            1
        };
        let head_dim = self.visual_dim / n_heads;

        CrossAttention {
            q: LinearConfig::new(self.visual_dim, self.visual_dim).with_bias(false).init(device),
            k: LinearConfig::new(self.text_dim, self.visual_dim).with_bias(false).init(device),
            v: LinearConfig::new(self.text_dim, self.visual_dim).with_bias(false).init(device),
            o: LinearConfig::new(self.visual_dim, self.visual_dim).with_bias(false).init(device),
            n_heads,
            head_dim,
        }
    }
}

#[derive(Module, Debug)]
pub struct CrossAttention<B: Backend> {
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    o: Linear<B>,
    n_heads: usize,
    head_dim: usize,
}

impl<B: Backend> CrossAttention<B> {
    /// visual: `[batch, num_visual_tokens, visual_dim]`
    /// text:   `[batch, num_text_tokens, text_dim]`
    pub fn forward(&self, visual: Tensor<B, 3>, text: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, num_v, _] = visual.dims();
        let [_, num_t, _] = text.dims();

        let q = self
            .q
            .forward(visual)
            .reshape([batch, num_v, self.n_heads, self.head_dim])
            .swap_dims(1, 2); // [batch, heads, num_v, head_dim]

        let k = self
            .k
            .forward(text.clone())
            .reshape([batch, num_t, self.n_heads, self.head_dim])
            .swap_dims(1, 2); // [batch, heads, num_t, head_dim]

        let v = self
            .v
            .forward(text)
            .reshape([batch, num_t, self.n_heads, self.head_dim])
            .swap_dims(1, 2); // [batch, heads, num_t, head_dim]

        let scale = (self.head_dim as f64).powf(-0.5);
        let qk = q.matmul(k.swap_dims(2, 3)) * scale; // [batch, heads, num_v, num_t]

        let weights = softmax(qk, 3);
        let out = weights
            .matmul(v) // [batch, heads, num_v, head_dim]
            .swap_dims(1, 2) // [batch, num_v, heads, head_dim]
            .reshape([batch, num_v, self.n_heads * self.head_dim]); // [batch, num_v, visual_dim]

        self.o.forward(out)
    }
}

/// Configuration for the Pass 4 Upscaler model.
#[derive(Config, Debug)]
pub struct UpscalerModelConfig {
    #[config(default = 64)]
    pub scales_from: usize,

    #[config(default = true)]
    pub colorizes: bool,

    #[config(default = 32)]
    pub channels: usize,

    #[config(default = 3)]
    pub num_res_blocks: usize,

    #[config(default = 0)]
    pub num_refine_blocks_per_layer: usize,

    #[config(default = true)]
    pub use_text_conditioning: bool,

    #[config(default = 16384)]
    pub vocab_size: usize,

    #[config(default = 64)]
    pub text_dim: usize,
}

impl UpscalerModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> UpscalerModel<B> {
        let c = self.channels;
        let c_half = (c / 2).max(16);
        let in_channels = if self.colorizes { 1 } else { 3 };

        let stem = Conv2dConfig::new([in_channels, c], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        let res_blocks = make_res_blocks(
            self.num_res_blocks,
            c,
            device,
        );

        let (text_embedding, cross_attn) = if self.use_text_conditioning {
            let emb = EmbeddingConfig::new(self.vocab_size, self.text_dim).init(device);
            let attn = CrossAttentionConfig::new(c, self.text_dim)
                .with_n_heads(4)
                .init(device);
            (Some(emb), Some(attn))
        } else {
            (None, None)
        };

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

        // Final output convolution to 3-channel RGB (always 3 channels)
        let out_conv = Conv2dConfig::new([c_half, 3], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        UpscalerModel {
            stem,
            res_blocks,
            text_embedding,
            cross_attn,
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

/// The Pass 4 Generative Upscaler (64x64/32x32 -> 256x256 RGB).
#[derive(Module, Debug)]
pub struct UpscalerModel<B: Backend> {
    stem: Conv2d<B>,
    res_blocks: Vec<ResBlock<B>>,

    text_embedding: Option<Embedding<B>>,
    cross_attn: Option<CrossAttention<B>>,

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
    /// Unconditional forward pass (delegates to `forward_with_text` with `None`).
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward_with_text(x, None)
    }

    /// Forward pass with optional text token conditioning:
    /// Input:  `[batch, in_channels, H, W]` (values in [-1.0, 1.0])
    /// Text:   Optional `[batch, seq_len]` Int token IDs
    /// Output: `[batch, 3, 256, 256]` (Super-resolved and colorized RGB values in [-1.0, 1.0])
    pub fn forward_with_text(
        &self,
        x: Tensor<B, 4>,
        text_tokens: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 4> {
        let [batch, _c, h, w] = x.dims();
        let mut f = relu(self.stem.forward(x));

        for block in &self.res_blocks {
            f = block.forward(f);
        }

        // Cross-attention text conditioning at bottleneck
        if let (Some(embed), Some(attn)) = (&self.text_embedding, &self.cross_attn) {
            let text_emb = match text_tokens {
                Some(tokens) => embed.forward(tokens),
                None => {
                    // Default unconditional token: single PAD token (id 0) per sample
                    let pad = Tensor::zeros([batch, 1], &f.device());
                    embed.forward(pad)
                }
            };

            // Flatten spatial dimensions [B, C, H, W] -> [B, H*W, C]
            let num_spatial = h * w;
            let c = f.dims()[1];
            let visual_tokens = f.clone().reshape([batch, c, num_spatial]).swap_dims(1, 2);

            let attn_out = attn.forward(visual_tokens, text_emb);

            // Reshape back [B, H*W, C] -> [B, C, H, W] and residual add with 0.2 scale
            let f_attn = attn_out.swap_dims(1, 2).reshape([batch, c, h, w]);
            f = f + f_attn * 0.2;
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
    #[config(default = 32)]
    pub base_channels: usize,
}

impl UpscalerDiscriminatorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> UpscalerDiscriminator<B> {
        let c = self.base_channels;
        let conv1 = Conv2dConfig::new([3, c], [4, 4])
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

/// PatchGAN Discriminator for 256x256 images.
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
