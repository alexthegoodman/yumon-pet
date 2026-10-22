// dit.rs - Standalone 64x64 Diffusion Vision Transformer (DiT)
//
// Pure-Transformer isotropic backbone trained via Rectified Flow (Flow Matching).
// Reuses bidirectional Flash Attention and normalization/MLP primitives.

use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Initializer, Linear, LinearConfig},
    prelude::*,
    tensor::{TensorData, activation::silu},
};
use serde::{Deserialize, Serialize};

use crate::brain::{
    flash_attn::backend::{FlashAttention, flash_attention},
    model::{MLP, MLPConfig, RMSNorm, RMSNormConfig},
};

// ─── Timestep Embedder ────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct TimestepEmbedder<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    freq_dim: usize,
}

impl<B: Backend> TimestepEmbedder<B> {
    pub fn new(freq_dim: usize, d_model: usize, device: &B::Device) -> Self {
        Self {
            linear1: LinearConfig::new(freq_dim, d_model).init(device),
            linear2: LinearConfig::new(d_model, d_model).init(device),
            freq_dim,
        }
    }

    /// t: [batch] continuous timesteps in [0, 1].
    /// Returns conditioning vector [batch, d_model].
    pub fn forward(&self, t: Tensor<B, 1>) -> Tensor<B, 2> {
        let device = t.device();
        let batch = t.dims()[0];
        let half = self.freq_dim / 2;

        let mut freqs = Vec::with_capacity(half);
        for i in 0..half {
            let exponent = -((i as f64) / (half as f64)) * (10000.0_f64).ln();
            freqs.push(exponent.exp() as f32);
        }
        let freqs = Tensor::<B, 2>::from_data(
            TensorData::new(freqs, [1, half]),
            &device,
        );

        // Continuous diffusion frequency scale t * 1000.0
        let t_scaled = (t * 1000.0).unsqueeze_dim::<2>(1); // [batch, 1]
        let args = t_scaled.matmul(freqs); // [batch, half]
        let sin = args.clone().sin();
        let cos = args.cos();
        let emb = Tensor::cat(vec![sin, cos], 1); // [batch, freq_dim]

        let h = silu(self.linear1.forward(emb));
        self.linear2.forward(h)
    }
}

// ─── 2D Sinusoidal Positional Embeddings ──────────────────────────────────────

pub fn build_2d_sinusoidal_pos_embed<B: Backend>(
    grid_size: usize,
    d_model: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    assert_eq!(d_model % 4, 0, "d_model must be divisible by 4 for 2D sin/cos");
    let n_tokens = grid_size * grid_size;
    let coord_dim = d_model / 2;
    let half_coord = coord_dim / 2;

    let mut freqs = Vec::with_capacity(half_coord);
    for i in 0..half_coord {
        let omega = 1.0_f64 / (10000.0_f64).powf((2 * i) as f64 / coord_dim as f64);
        freqs.push(omega as f32);
    }

    let mut table = Vec::with_capacity(n_tokens * d_model);
    for y in 0..grid_size {
        for x in 0..grid_size {
            // Y coordinates (first d_model / 2)
            for &omega in &freqs {
                table.push(((y as f32) * omega).sin());
            }
            for &omega in &freqs {
                table.push(((y as f32) * omega).cos());
            }
            // X coordinates (second d_model / 2)
            for &omega in &freqs {
                table.push(((x as f32) * omega).sin());
            }
            for &omega in &freqs {
                table.push(((x as f32) * omega).cos());
            }
        }
    }

    Tensor::from_data(
        TensorData::new(table, [1, n_tokens, d_model]),
        device,
    )
}

// ─── AdaLN-Zero DiT Block ─────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct DiTBlock<B: Backend> {
    ada_ln:   Linear<B>,
    norm1:    RMSNorm<B>,
    q_proj:   Linear<B>,
    k_proj:   Linear<B>,
    v_proj:   Linear<B>,
    q_norm:   RMSNorm<B>,
    k_norm:   RMSNorm<B>,
    o_proj:   Linear<B>,
    norm2:    RMSNorm<B>,
    mlp:      MLP<B>,
    n_heads:  usize,
    d_model:  usize,
}

impl<B: Backend> DiTBlock<B> {
    pub fn new(
        d_model: usize,
        d_hidden: usize,
        n_heads: usize,
        device: &B::Device,
    ) -> Self {
        assert_eq!(d_model % n_heads, 0, "d_model must be divisible by n_heads");
        let head_dim = d_model / n_heads;

        // AdaLN-Zero modulation: 6 * d_model outputs, initialized to zero
        let ada_ln = LinearConfig::new(d_model, 6 * d_model)
            .with_initializer(Initializer::Zeros)
            .init(device);

        let norm1 = RMSNormConfig::new(d_model)
            .with_elementwise_affine(false)
            .init(device);

        let q_proj = LinearConfig::new(d_model, d_model).with_bias(false).init(device);
        let k_proj = LinearConfig::new(d_model, d_model).with_bias(false).init(device);
        let v_proj = LinearConfig::new(d_model, d_model).with_bias(false).init(device);
        let q_norm = RMSNormConfig::new(head_dim).with_elementwise_affine(true).init(device);
        let k_norm = RMSNormConfig::new(head_dim).with_elementwise_affine(true).init(device);
        let o_proj = LinearConfig::new(d_model, d_model).with_bias(false).init(device);

        let norm2 = RMSNormConfig::new(d_model)
            .with_elementwise_affine(false)
            .init(device);

        let mlp = MLPConfig::new(d_model, d_hidden).init(device);

        Self {
            ada_ln,
            norm1,
            q_proj,
            k_proj,
            v_proj,
            q_norm,
            k_norm,
            o_proj,
            norm2,
            mlp,
            n_heads,
            d_model,
        }
    }
}

impl<B: FlashAttention> DiTBlock<B> {
    /// x: [batch, n_tokens, d_model]
    /// c: [batch, d_model] timestep conditioning vector
    pub fn forward(&self, x: Tensor<B, 3>, c: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, n, d_model] = x.dims();
        let head_dim = d_model / self.n_heads;

        // 6 modulation projections
        let mod_params = self.ada_ln.forward(c).unsqueeze_dim::<3>(1); // [batch, 1, 6 * d_model]
        let shift_msa = mod_params.clone().narrow(2, 0 * d_model, d_model);
        let scale_msa = mod_params.clone().narrow(2, 1 * d_model, d_model);
        let gate_msa  = mod_params.clone().narrow(2, 2 * d_model, d_model);
        let shift_mlp = mod_params.clone().narrow(2, 3 * d_model, d_model);
        let scale_mlp = mod_params.clone().narrow(2, 4 * d_model, d_model);
        let gate_mlp  = mod_params.narrow(2, 5 * d_model, d_model);

        // 1. Attention with AdaLN modulation and QK-Norm
        let norm_x1 = self.norm1.forward(x.clone());
        let x_mod1 = norm_x1 * (scale_msa + 1.0) + shift_msa;

        let split_heads = |t: Tensor<B, 3>| {
            t.reshape([batch, n, self.n_heads, head_dim]).swap_dims(1, 2)
        };
        let q = self.q_norm.forward(split_heads(self.q_proj.forward(x_mod1.clone())));
        let k = self.k_norm.forward(split_heads(self.k_proj.forward(x_mod1.clone())));
        let v = split_heads(self.v_proj.forward(x_mod1));

        // Bidirectional flash attention (causal = false)
        let attn = flash_attention(q, k, v, false);
        let attn = attn.swap_dims(1, 2).reshape([batch, n, d_model]);
        let attn_out = self.o_proj.forward(attn);
        let x = x + attn_out * gate_msa;

        // 2. SwiGLU MLP with AdaLN modulation
        let norm_x2 = self.norm2.forward(x.clone());
        let x_mod2 = norm_x2 * (scale_mlp + 1.0) + shift_mlp;
        let mlp_out = self.mlp.forward(x_mod2);
        let x = x + mlp_out * gate_mlp;

        x
    }
}

// ─── Standalone VisionDiT Model ──────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct VisionDiTConfig {
    #[config(default = 64)]
    pub img_size: usize,
    #[config(default = 4)]
    pub patch_size: usize,
    #[config(default = 3)]
    pub in_channels: usize,
    #[config(default = 128)]
    pub d_model: usize,
    #[config(default = 256)]
    pub d_hidden: usize,
    #[config(default = 4)]
    pub n_layers: usize,
    #[config(default = 4)]
    pub n_heads: usize,
    #[config(default = 256)]
    pub time_embed_dim: usize,
}

impl VisionDiTConfig {
    /// Lightweight preset suitable for fast iteration and tests (~400k parameters)
    pub fn micro() -> Self {
        Self {
            img_size: 64,
            patch_size: 4,
            in_channels: 3,
            d_model: 128,
            d_hidden: 256,
            n_layers: 4,
            n_heads: 4,
            time_embed_dim: 256,
        }
    }

    /// Tiny preset with stronger capacity (~2.5M parameters)
    pub fn tiny() -> Self {
        Self {
            img_size: 64,
            patch_size: 4,
            in_channels: 3,
            d_model: 256,
            d_hidden: 512,
            n_layers: 6,
            n_heads: 8,
            time_embed_dim: 256,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> VisionDiT<B> {
        let grid_size = self.img_size / self.patch_size;
        let patch_dim = self.in_channels * self.patch_size * self.patch_size;

        let patch_embed = LinearConfig::new(patch_dim, self.d_model).init(device);
        let time_embed = TimestepEmbedder::new(self.time_embed_dim, self.d_model, device);
        let pos_tensor = build_2d_sinusoidal_pos_embed::<B>(grid_size, self.d_model, device);
        let pos_embed = Param::from_tensor(pos_tensor);

        let blocks = (0..self.n_layers)
            .map(|_| DiTBlock::new(self.d_model, self.d_hidden, self.n_heads, device))
            .collect();

        // Final AdaLN-Zero modulation: 2 * d_model (shift, scale), initialized to zero
        let final_ada_ln = LinearConfig::new(self.d_model, 2 * self.d_model)
            .with_initializer(Initializer::Zeros)
            .init(device);

        let final_norm = RMSNormConfig::new(self.d_model)
            .with_elementwise_affine(false)
            .init(device);

        // Final linear projection to patch features, initialized to zero
        let final_linear = LinearConfig::new(self.d_model, patch_dim)
            .with_initializer(Initializer::Zeros)
            .init(device);

        VisionDiT {
            patch_size: self.patch_size,
            img_size: self.img_size,
            in_channels: self.in_channels,
            d_model: self.d_model,
            patch_embed,
            time_embed,
            pos_embed,
            blocks,
            final_ada_ln,
            final_norm,
            final_linear,
        }
    }
}

#[derive(Module, Debug)]
pub struct VisionDiT<B: Backend> {
    pub patch_size: usize,
    pub img_size: usize,
    pub in_channels: usize,
    pub d_model: usize,
    patch_embed: Linear<B>,
    time_embed: TimestepEmbedder<B>,
    pos_embed: Param<Tensor<B, 3>>,
    blocks: Vec<DiTBlock<B>>,
    final_ada_ln: Linear<B>,
    final_norm: RMSNorm<B>,
    final_linear: Linear<B>,
}

impl<B: Backend> VisionDiT<B> {
    /// Converts [B, C, H, W] to [B, (H/P)*(W/P), C*P*P]
    pub fn patchify(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [b, c, h, w] = x.dims();
        let p = self.patch_size;
        let gh = h / p;
        let gw = w / p;

        // [B, C, gh, P, gw, P]
        let x = x.reshape([b, c, gh, p, gw, p]);
        // Swap to [B, gh, gw, C, P, P]
        let x = x.swap_dims(1, 2)  // [B, gh, C, P, gw, P]
                 .swap_dims(2, 4)  // [B, gh, gw, P, C, P]
                 .swap_dims(3, 4); // [B, gh, gw, C, P, P]
        // Flatten to [B, gh * gw, C * P * P]
        x.reshape([b, gh * gw, c * p * p])
    }

    /// Inverts [B, (H/P)*(W/P), C*P*P] back to [B, C, H, W]
    pub fn unpatchify(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [b, _, _] = x.dims();
        let p = self.patch_size;
        let gh = self.img_size / p;
        let gw = self.img_size / p;
        let c = self.in_channels;

        // Reshape to [B, gh, gw, C, P, P]
        let x = x.reshape([b, gh, gw, c, p, p]);
        // Inverse permute back to [B, C, gh, P, gw, P]
        let x = x.swap_dims(3, 4)  // [B, gh, gw, P, C, P]
                 .swap_dims(2, 4)  // [B, gh, C, P, gw, P]
                 .swap_dims(1, 2); // [B, C, gh, P, gw, P]
        // Flatten to [B, C, gh * P, gw * P]
        x.reshape([b, c, gh * p, gw * p])
    }
}

impl<B: FlashAttention> VisionDiT<B> {
    /// x: [batch, 3, 64, 64] image or intermediate noisy image
    /// t: [batch] continuous timesteps in [0, 1]
    /// Returns predicted velocity [batch, 3, 64, 64]
    pub fn forward(&self, x: Tensor<B, 4>, t: Tensor<B, 1>) -> Tensor<B, 4> {
        let patches = self.patchify(x); // [batch, n_tokens, patch_dim]
        let mut h = self.patch_embed.forward(patches) + self.pos_embed.val();
        let c = self.time_embed.forward(t); // [batch, d_model]

        for block in &self.blocks {
            h = block.forward(h, c.clone());
        }

        // Final AdaLN modulation
        let mod_params = self.final_ada_ln.forward(c).unsqueeze_dim::<3>(1); // [batch, 1, 2 * d_model]
        let shift = mod_params.clone().narrow(2, 0, self.d_model);
        let scale = mod_params.narrow(2, self.d_model, self.d_model);
        let h = self.final_norm.forward(h) * (scale + 1.0) + shift;

        let out_patches = self.final_linear.forward(h); // [batch, n_tokens, patch_dim]
        self.unpatchify(out_patches)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn_cubecl::CubeBackend;
    use cubecl::wgpu::WgpuRuntime;

    type TestBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;

    #[test]
    fn test_patchify_unpatchify_exact_inverse() {
        let device = Default::default();
        let config = VisionDiTConfig::micro();
        let model: VisionDiT<TestBackend> = config.init(&device);

        let img = Tensor::<TestBackend, 4>::random(
            [2, 3, 64, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let patches = model.patchify(img.clone());
        assert_eq!(patches.dims(), [2, 256, 48]);

        let reconstructed = model.unpatchify(patches);
        assert_eq!(reconstructed.dims(), [2, 3, 64, 64]);

        let diff = (img - reconstructed).abs().max().into_scalar();
        assert!(diff < 1e-6, "reconstruction max diff: {diff}");
    }

    #[test]
    fn test_dit_initialization_identity_zero_output() {
        let device = Default::default();
        let config = VisionDiTConfig::micro();
        let model: VisionDiT<TestBackend> = config.init(&device);

        let batch_size = 2;
        let x = Tensor::<TestBackend, 4>::random(
            [batch_size, 3, 64, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let t = Tensor::<TestBackend, 1>::from_floats([0.2, 0.8], &device);

        let velocity = model.forward(x, t);
        assert_eq!(velocity.dims(), [batch_size, 3, 64, 64]);

        let max_val = velocity.abs().max().into_scalar();
        assert_eq!(max_val, 0.0, "initial output must be strictly 0.0, got {max_val}");
    }
}
