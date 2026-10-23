//! 2D Coordinate-Aware Linear Patch Tokenizer for 64x64 Canvas Input
//! with Conv2D Multi-Scale Edge Stem and 2D Spatial Patch Mixer.
//!
//! Converts a [batch, 3, 64, 64] image into [batch, 256, embed_dim] visual patch tokens.
//! Features:
//! 1. Multi-scale 3x3 Conv2D stem: computes overlapping local spatial derivatives and
//!    edge orientation across sub-patch boundaries on the 64x64 canvas.
//! 2. 4x4 non-overlapping patch projection into embed_dim.
//! 3. 2D spatial Conv2D mixer across the 16x16 macro-patch grid to bridge row-wrap
//!    discontinuities and stabilize MoE routing along continuous edges.
//! 4. Learned 2D coordinate embeddings (row + col) and RMSNorm.

use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Embedding, EmbeddingConfig,
        PaddingConfig2d,
    },
    prelude::*,
    tensor::TensorData,
};
use crate::brain::model::{RMSNorm, RMSNormConfig};
use super::paint_action::{CANVAS_SIZE, GRID_SIZE, NUM_PATCHES, PATCH_SIZE};

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

        // 1. Multi-scale 3x3 Conv2D stem: extracts sub-patch edge continuity
        let stem_conv1 = Conv2dConfig::new([3, 16], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let stem_conv2 = Conv2dConfig::new([16, 16], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        // 2. Patch projection downsampling 16 channels to embed_dim
        let patch_proj = Conv2dConfig::new([16, self.embed_dim], [self.patch_size, self.patch_size])
            .with_stride([self.patch_size, self.patch_size])
            .with_bias(false)
            .init(device);

        // 3. 2D spatial Conv2D mixer on the 16x16 macro-patch grid
        let spatial_conv = Conv2dConfig::new([self.embed_dim, self.embed_dim], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        let row_embed = EmbeddingConfig::new(grid, self.embed_dim).init(device);
        let col_embed = EmbeddingConfig::new(grid, self.embed_dim).init(device);
        let norm = RMSNormConfig::new(self.embed_dim).init(device);

        // Precompute row and column index vectors for fast addition
        let mut row_indices = Vec::with_capacity(num_patches);
        let mut col_indices = Vec::with_capacity(num_patches);

        for r in 0..grid {
            for c in 0..grid {
                row_indices.push(r as i32);
                col_indices.push(c as i32);
            }
        }

        let row_ids = Tensor::<B, 1, Int>::from_ints(
            TensorData::new(row_indices, [num_patches]),
            device,
        );
        let col_ids = Tensor::<B, 1, Int>::from_ints(
            TensorData::new(col_indices, [num_patches]),
            device,
        );

        VisionPatchEncoder {
            stem_conv1,
            stem_conv2,
            patch_proj,
            spatial_conv,
            row_embed,
            col_embed,
            norm,
            row_ids,
            col_ids,
            embed_dim: self.embed_dim,
            grid_size: grid,
            num_patches,
        }
    }
}

#[derive(Module, Debug)]
pub struct VisionPatchEncoder<B: Backend> {
    stem_conv1: Conv2d<B>,
    stem_conv2: Conv2d<B>,
    patch_proj: Conv2d<B>,
    spatial_conv: Conv2d<B>,
    row_embed: Embedding<B>,
    col_embed: Embedding<B>,
    norm: RMSNorm<B>,
    row_ids: Tensor<B, 1, Int>,
    col_ids: Tensor<B, 1, Int>,
    pub embed_dim: usize,
    pub grid_size: usize,
    pub num_patches: usize,
}

impl<B: Backend> VisionPatchEncoder<B> {
    /// Forward pass.
    /// Input:  [batch, 3, 64, 64] float canvas normalized to [-1.0, 1.0].
    /// Output: [batch, 256, embed_dim] visual patch tokens with 2D coordinate embeddings.
    pub fn forward(&self, canvas: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, _c, _h, _w] = canvas.dims();

        // 1. Multi-scale 3x3 Conv2D edge stem: extracts overlapping gradients across boundaries
        let h = self.stem_conv1.forward(canvas);
        let h = burn::tensor::activation::gelu(h);
        let stem_out = self.stem_conv2.forward(h);
        let stem_out = burn::tensor::activation::gelu(stem_out);

        // 2. Project 4x4 patches -> [batch, embed_dim, 16, 16]
        let patches = self.patch_proj.forward(stem_out);

        // 3. 2D spatial Conv2D mixer across the 16x16 macro-patch grid + residual connection
        let mixed = self.spatial_conv.forward(patches.clone());
        let patches = patches + burn::tensor::activation::gelu(mixed);

        // 4. Reshape to [batch, embed_dim, 256] then swap to [batch, 256, embed_dim]
        let x = patches
            .reshape([batch, self.embed_dim, self.num_patches])
            .swap_dims(1, 2);

        // 5. Lookup 2D coordinate embeddings [1, 256, embed_dim]
        let row_pos = self.row_embed.forward(self.row_ids.clone().unsqueeze_dim(0));
        let col_pos = self.col_embed.forward(self.col_ids.clone().unsqueeze_dim(0));

        // 6. Add 2D position embeddings and normalize
        let x = x + row_pos + col_pos;
        self.norm.forward(x)
    }
}
