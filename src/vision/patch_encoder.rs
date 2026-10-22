//! 2D Coordinate-Aware Linear Patch Tokenizer for 64x64 Canvas Input.
//!
//! Converts a [batch, 3, 64, 64] image into [batch, 256, embed_dim] visual patch tokens.
//! Zero spatial pooling: all 48 pixel values per 4x4 patch are linearly projected,
//! followed by learned 2D coordinate embeddings (row + col) and RMSNorm.

use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Embedding, EmbeddingConfig,
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

        // Conv2d with kernel=patch_size, stride=patch_size is equivalent to linear patch projection
        let patch_proj = Conv2dConfig::new([3, self.embed_dim], [self.patch_size, self.patch_size])
            .with_stride([self.patch_size, self.patch_size])
            .with_bias(false)
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
            patch_proj,
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
    patch_proj: Conv2d<B>,
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

        // 1. Linearly project non-overlapping 4x4 patches -> [batch, embed_dim, 16, 16]
        let patches = self.patch_proj.forward(canvas);

        // 2. Reshape to [batch, embed_dim, 256] then swap to [batch, 256, embed_dim]
        let x = patches
            .reshape([batch, self.embed_dim, self.num_patches])
            .swap_dims(1, 2);

        // 3. Lookup 2D coordinate embeddings [1, 256, embed_dim]
        let row_pos = self.row_embed.forward(self.row_ids.clone().unsqueeze_dim(0));
        let col_pos = self.col_embed.forward(self.col_ids.clone().unsqueeze_dim(0));

        // 4. Add 2D position embeddings and normalize
        let x = x + row_pos + col_pos;
        self.norm.forward(x)
    }
}
