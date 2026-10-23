//! Lightweight Linear Patch Tokenizer for 64x64 Canvas Input.
//!
//! Converts a [batch, 3, 64, 64] image into [batch, 256, embed_dim] visual patch tokens.
//! Projects non-overlapping 4x4 patches directly into embed_dim via a strided 4x4 convolution,
//! followed by RMSNorm. Spatial 2D coordinate reasoning is delegated to M-RoPE in attention.

use burn::{
    nn::conv::{Conv2d, Conv2dConfig},
    prelude::*,
};
use crate::brain::model::{RMSNorm, RMSNormConfig};

#[derive(Config, Debug)]
pub struct VisionPatchEncoderConfig {
    pub embed_dim: usize,
    #[config(default = 4)]
    pub patch_size: usize,
    #[config(default = 64)]
    pub canvas_size: usize,
}

impl VisionPatchEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VisionPatchEncoder<B> {
        let grid = self.canvas_size / self.patch_size;
        let num_patches = grid * grid;

        // Direct patch projection downsampling 3 channels to embed_dim with stride = patch_size
        let patch_proj = Conv2dConfig::new([3, self.embed_dim], [self.patch_size, self.patch_size])
            .with_stride([self.patch_size, self.patch_size])
            .with_bias(false)
            .init(device);

        let norm = RMSNormConfig::new(self.embed_dim).init(device);

        VisionPatchEncoder {
            patch_proj,
            norm,
            embed_dim: self.embed_dim,
            grid_size: grid,
            num_patches,
        }
    }
}

#[derive(Module, Debug)]
pub struct VisionPatchEncoder<B: Backend> {
    patch_proj: Conv2d<B>,
    norm: RMSNorm<B>,
    pub embed_dim: usize,
    pub grid_size: usize,
    pub num_patches: usize,
}

impl<B: Backend> VisionPatchEncoder<B> {
    /// Forward pass.
    /// Input:  [batch, 3, 64, 64] float canvas normalized to [-1.0, 1.0].
    /// Output: [batch, 256, embed_dim] visual patch tokens.
    pub fn forward(&self, canvas: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, _c, _h, _w] = canvas.dims();

        // 1. Project 4x4 patches directly: [batch, 3, 64, 64] -> [batch, embed_dim, 16, 16]
        let patches = self.patch_proj.forward(canvas);

        // 2. Reshape to [batch, embed_dim, 256] then swap to [batch, 256, embed_dim]
        let x = patches
            .reshape([batch, self.embed_dim, self.num_patches])
            .swap_dims(1, 2);

        // 3. Normalize
        self.norm.forward(x)
    }
}
