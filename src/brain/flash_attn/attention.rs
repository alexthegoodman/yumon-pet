// attention.rs — drop-in FlashAttention integration
//
// Strategy: keep every projection, RoPE call, mask construction, and
// module structure exactly as-is. Only the inner "qk → softmax → weighted
// sum" section is replaced with flash_attn_forward via ops.rs.
//
// The bridge function `flash_attn_from_tensors` converts Burn tensors to
// raw slices, calls the CubeCL kernel, and returns a Burn tensor — so
// the surrounding Burn code never needs to know a custom kernel ran.

use burn::{
    config::Config, module::Module, nn::{Initializer, Linear, LinearConfig, RotaryEncoding}, tensor::{Bool, Device, Tensor, backend::Backend}
};
use burn::backend::wgpu::{WgpuDevice, WgpuRuntime};

use crate::brain::{flash_attn::bridge_ops::{launch_forward, read_f32, read_f32_async}, model::{MLP, MLPConfig, RMSNorm, RMSNormConfig}};

// ── Kernel bridge ─────────────────────────────────────────────────────────────
//
// Takes post-RoPE Q, K, V as Burn tensors [batch, heads, seq_q, d_k],
// an optional additive float mask [batch, 1, seq_q, seq_k], and returns
// the attended output [batch, heads, seq_q, d_v].
//
// Masked path:   scores are materialised on-GPU via Burn ops (standard
//                matmul+softmax), preserving -inf masking correctness.
//                The N×N matrix is unavoidable here until the kernel
//                gains a mask tensor argument.
//
// Unmasked path: full CubeCL tiled kernel — no N×N matrix, O(N) memory.

fn flash_attn_from_tensors<B: Backend<Device = WgpuDevice>>(
    q:       Tensor<B, 4>,
    k:       Tensor<B, 4>,
    v:       Tensor<B, 4>,
    mask:    Option<Tensor<B, 4>>,
    device:  &B::Device, 
    block_q: usize,
    block_k: usize,
) -> Tensor<B, 4> {
    let q_device = q.device();
    let [batch, heads, seq_q, d_k] = q.dims();
    let [_, _, seq_k, d_v]         = v.dims();

    if let Some(m) = mask {
        // Masked path — standard Burn ops, correct -inf handling
        let scale  = (d_k as f64).sqrt().recip();
        let scores = q.matmul(k.transpose()) * scale + m;
        let w      = burn::tensor::activation::softmax(scores, 3);
        return w.matmul(v);
    }

    // Unmasked path — CubeCL FlashAttention kernel
    let bh = batch * heads;

    let q_r: Tensor<B, 3> = q.reshape([bh, seq_q, d_k]);
    let k_r: Tensor<B, 3> = k.reshape([bh, seq_k, d_k]);
    let v_r: Tensor<B, 3> = v.reshape([bh, seq_k, d_v]);

    let q_data: Vec<f32> = q_r.into_data().to_vec().unwrap();
    let k_data: Vec<f32> = k_r.into_data().to_vec().unwrap();
    let v_data: Vec<f32> = v_r.into_data().to_vec().unwrap();

    let result = launch_forward::<WgpuRuntime>(
        device,
        &q_data, &k_data, &v_data,
        bh, seq_q, seq_k, d_k, d_v,
        block_q, block_k,
    );

    let out_data = read_f32::<WgpuRuntime>(device, result.out, bh * seq_q * d_v);

    Tensor::<B, 4>::from_floats(out_data.as_slice(), &q_device)
        .reshape([batch, heads, seq_q, d_v])
}

async fn flash_attn_from_tensors_async<B: Backend<Device = WgpuDevice>>(
    q:       Tensor<B, 4>,
    k:       Tensor<B, 4>,
    v:       Tensor<B, 4>,
    mask:    Option<Tensor<B, 4>>,
    device:  &B::Device, 
    block_q: usize,
    block_k: usize,
) -> Tensor<B, 4> {
    let q_device = q.device();
    let [batch, heads, seq_q, d_k] = q.dims();
    let [_, _, seq_k, d_v]         = v.dims();

    if let Some(m) = mask {
        // Masked path — standard Burn ops, correct -inf handling
        let scale  = (d_k as f64).sqrt().recip();
        let scores = q.matmul(k.transpose()) * scale + m;
        let w      = burn::tensor::activation::softmax(scores, 3);
        return w.matmul(v);
    }

    // Unmasked path — CubeCL FlashAttention kernel
    let bh = batch * heads;

    let q_r: Tensor<B, 3> = q.reshape([bh, seq_q, d_k]);
    let k_r: Tensor<B, 3> = k.reshape([bh, seq_k, d_k]);
    let v_r: Tensor<B, 3> = v.reshape([bh, seq_k, d_v]);

    let q_data: Vec<f32> = q_r.into_data_async().await.expect("Couldn't get data").to_vec().unwrap();
    let k_data: Vec<f32> = k_r.into_data_async().await.expect("Couldn't get data").to_vec().unwrap();
    let v_data: Vec<f32> = v_r.into_data_async().await.expect("Couldn't get data").to_vec().unwrap();

    let result = launch_forward::<WgpuRuntime>(
        device,
        &q_data, &k_data, &v_data,
        bh, seq_q, seq_k, d_k, d_v,
        block_q, block_k,
    );

    let out_data = read_f32_async::<WgpuRuntime>(device, result.out, bh * seq_q * d_v).await;

    Tensor::<B, 4>::from_floats(out_data.as_slice(), &q_device)
        .reshape([batch, heads, seq_q, d_v])
}

// ── Shared mask builder ───────────────────────────────────────────────────────
//
// Merges causal + padding masks into one additive [batch, 1, seq_q, seq_k]
// float tensor, or returns None if neither is present.

fn build_mask<B: Backend>(
    causal:   Option<Tensor<B, 2>>,
    pad_mask: Option<Tensor<B, 2, Bool>>,
    seq_q:    usize,
    seq_k:    usize,
) -> Option<Tensor<B, 4>> {
    let mut combined: Option<Tensor<B, 4>> = None;

    if let Some(cm) = causal {
        // [seq, seq] → [1, 1, seq_q, seq_k]
        let m = cm.slice([0..seq_q, 0..seq_k]).unsqueeze::<4>();
        combined = Some(m);
    }

    if let Some(pm) = pad_mask {
        // [batch, seq_k] → [batch, 1, 1, seq_k] additive float
        let pm_f = pm
            .unsqueeze_dim::<3>(1)
            .unsqueeze_dim::<4>(2)
            .float()
            .mul_scalar(-1e9_f32);

        combined = Some(match combined {
            Some(c) => c + pm_f,
            None    => pm_f,
        });
    }

    combined
}

// ── Causal mask helper (unchanged) ───────────────────────────────────────────

pub fn causal_mask<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 2> {
    use std::f32::NEG_INFINITY;
    Tensor::full([seq_len, seq_len], NEG_INFINITY, device).triu(1)
}

// ── Self-Attention ────────────────────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct SelfAttentionConfig {
    d_model: usize,
    n_heads: usize,
    #[config(default = 64)]
    block_q: usize,
    #[config(default = 64)]
    block_k: usize,
}

impl SelfAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SelfAttention<B> {
        assert!(self.d_model % self.n_heads == 0);
        let head_dim = self.d_model / self.n_heads;
        let proj = |i, o| LinearConfig::new(i, o)
            .with_initializer(Initializer::XavierUniform { gain: 1.0 })
            .with_bias(false)
            .init(device);
        SelfAttention {
            q: proj(self.d_model, self.d_model),
            k: proj(self.d_model, self.d_model),
            v: proj(self.d_model, self.d_model),
            o: proj(self.d_model, self.d_model),
            n_heads:  self.n_heads,
            head_dim,
            block_q:  self.block_q,
            block_k:  self.block_k,
        }
    }
}

#[derive(Module, Debug)]
pub struct SelfAttention<B: Backend> {
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    o: Linear<B>,
    n_heads:  usize,
    head_dim: usize,
    block_q:  usize,
    block_k:  usize,
}

impl<B: Backend<Device = WgpuDevice>> SelfAttention<B> {
    pub fn forward(
        &self,
        x:           Tensor<B, 3>,
        rope:        &RotaryEncoding<B>,
        causal_mask: Option<Tensor<B, 2>>,
        pad_mask:    Option<Tensor<B, 2, Bool>>,
        device:      &B::Device,
    ) -> Tensor<B, 3> {
        let [batch, seq, _] = x.dims();

        let reshape = |t: Tensor<B, 3>| {
            t.reshape([batch, seq, self.n_heads, self.head_dim])
             .swap_dims(1, 2)
        };

        // RoPE applied exactly as in your original
        let q = rope.forward(reshape(self.q.forward(x.clone())));
        let k = rope.forward(reshape(self.k.forward(x.clone())));
        let v =              reshape(self.v.forward(x));

        let mask = build_mask(causal_mask, pad_mask, seq, seq);

        let out = flash_attn_from_tensors(q, k, v, mask, device, self.block_q, self.block_k);

        self.o.forward(out.swap_dims(1, 2).flatten(2, 3))
    }

    pub async fn forward_async(
        &self,
        x:           Tensor<B, 3>,
        rope:        &RotaryEncoding<B>,
        causal_mask: Option<Tensor<B, 2>>,
        pad_mask:    Option<Tensor<B, 2, Bool>>,
        device:      &B::Device,
    ) -> Tensor<B, 3> {
        let [batch, seq, _] = x.dims();

        let reshape = |t: Tensor<B, 3>| {
            t.reshape([batch, seq, self.n_heads, self.head_dim])
             .swap_dims(1, 2)
        };

        // RoPE applied exactly as in your original
        let q = rope.forward(reshape(self.q.forward(x.clone())));
        let k = rope.forward(reshape(self.k.forward(x.clone())));
        let v =              reshape(self.v.forward(x));

        let mask = build_mask(causal_mask, pad_mask, seq, seq);

        let out = flash_attn_from_tensors_async(q, k, v, mask, device, self.block_q, self.block_k).await;

        self.o.forward(out.swap_dims(1, 2).flatten(2, 3))
    }
}

// ── Cross-Attention ───────────────────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct CrossAttentionConfig {
    d_model: usize,
    n_heads: usize,
    #[config(default = 64)]
    block_q: usize,
    #[config(default = 64)]
    block_k: usize,
}

impl CrossAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CrossAttentionBlock<B> {
        assert!(self.d_model % self.n_heads == 0);
        let head_dim = self.d_model / self.n_heads;
        let proj = |i, o| LinearConfig::new(i, o).with_bias(false).init(device);
        CrossAttentionBlock {
            q:       proj(self.d_model, self.d_model),
            k:       proj(self.d_model, self.d_model),
            v:       proj(self.d_model, self.d_model),
            o:       proj(self.d_model, self.d_model),
            gate:    LinearConfig::new(self.d_model, self.n_heads)
                         .with_bias(true).init(device),
            n_heads:  self.n_heads,
            head_dim,
            block_q:  self.block_q,
            block_k:  self.block_k,
        }
    }
}

#[derive(Module, Debug)]
pub struct CrossAttentionBlock<B: Backend> {
    q:        Linear<B>,
    k:        Linear<B>,
    v:        Linear<B>,
    o:        Linear<B>,
    gate:     Linear<B>,
    n_heads:  usize,
    head_dim: usize,
    block_q:  usize,
    block_k:  usize,
}

impl<B: Backend<Device = WgpuDevice>> CrossAttentionBlock<B> {
    pub fn forward(
        &self,
        x:            Tensor<B, 3>,
        memory:       Tensor<B, 3>,
        rope:         &RotaryEncoding<B>,
        enc_pad_mask: Option<Tensor<B, 2, Bool>>,
        device:       &B::Device,
    ) -> Tensor<B, 3> {
        let [batch, dec_len, _] = x.dims();
        let [_, enc_len, _]     = memory.dims();

        let reshape_dec = |t: Tensor<B, 3>| {
            t.reshape([batch, dec_len, self.n_heads, self.head_dim])
             .swap_dims(1, 2)
        };
        let reshape_enc = |t: Tensor<B, 3>| {
            t.reshape([batch, enc_len, self.n_heads, self.head_dim])
             .swap_dims(1, 2)
        };

        // Q from decoder, K/V from encoder — same as your original
        let q = reshape_dec(self.q.forward(x.clone()));
        let k = reshape_enc(self.k.forward(memory.clone()));
        let v = reshape_enc(self.v.forward(memory));

        // No causal mask on cross-attention, only optional encoder padding
        let mask = build_mask(None, enc_pad_mask, dec_len, enc_len);

        let out = flash_attn_from_tensors(q, k, v, mask, device, self.block_q, self.block_k);

        self.o.forward(out.swap_dims(1, 2).flatten(2, 3))
    }

    pub async fn forward_async(
        &self,
        x:            Tensor<B, 3>,
        memory:       Tensor<B, 3>,
        rope:         &RotaryEncoding<B>,
        enc_pad_mask: Option<Tensor<B, 2, Bool>>,
        device:       &B::Device,
    ) -> Tensor<B, 3> {
        let [batch, dec_len, _] = x.dims();
        let [_, enc_len, _]     = memory.dims();

        let reshape_dec = |t: Tensor<B, 3>| {
            t.reshape([batch, dec_len, self.n_heads, self.head_dim])
             .swap_dims(1, 2)
        };
        let reshape_enc = |t: Tensor<B, 3>| {
            t.reshape([batch, enc_len, self.n_heads, self.head_dim])
             .swap_dims(1, 2)
        };

        // Q from decoder, K/V from encoder — same as your original
        let q = reshape_dec(self.q.forward(x.clone()));
        let k = reshape_enc(self.k.forward(memory.clone()));
        let v = reshape_enc(self.v.forward(memory));

        // No causal mask on cross-attention, only optional encoder padding
        let mask = build_mask(None, enc_pad_mask, dec_len, enc_len);

        let out = flash_attn_from_tensors_async(q, k, v, mask, device, self.block_q, self.block_k).await;

        self.o.forward(out.swap_dims(1, 2).flatten(2, 3))
    }
}

// ── EncoderBlock ─────────────────────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct EncoderBlockConfig {
    pub d_model:  usize,
    pub d_hidden: usize,
    pub n_heads:  usize,
    #[config(default = 64)]
    pub block_q:  usize,
    #[config(default = 64)]
    pub block_k:  usize,
}
 
impl EncoderBlockConfig {
    pub fn init<B: Backend<Device = WgpuDevice>>(&self, device: &B::Device) -> EncoderBlock<B> {
        EncoderBlock {
            attn_norm: RMSNormConfig::new(self.d_model).init(device),
            attn:      SelfAttentionConfig::new(self.d_model, self.n_heads)
                           .with_block_q(self.block_q)
                           .with_block_k(self.block_k)
                           .init(device),
            mlp_norm:  RMSNormConfig::new(self.d_model).init(device),
            mlp:       MLPConfig::new(self.d_model, self.d_hidden).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct EncoderBlock<B: Backend<Device = WgpuDevice>> {
    pub attn_norm: RMSNorm<B>,
    pub attn:      SelfAttention<B>,
    pub mlp_norm:  RMSNorm<B>,
    pub mlp:       MLP<B>,
}

impl<B: Backend<Device = WgpuDevice>> EncoderBlock<B> {
    pub fn forward(
        &self,
        x:        Tensor<B, 3>,
        rope:     &RotaryEncoding<B>,
        pad_mask: Option<Tensor<B, 2, Bool>>,
        // device:       &B::Device,
    ) -> Tensor<B, 3> {
        let x_device = x.device();
        let x = x.clone()
            + self.attn.forward(
                self.attn_norm.forward(x), rope, None, pad_mask, &x_device,
            );
        x.clone() + self.mlp.forward(self.mlp_norm.forward(x))
    }

    pub async fn forward_async(
        &self,
        x:        Tensor<B, 3>,
        rope:     &RotaryEncoding<B>,
        pad_mask: Option<Tensor<B, 2, Bool>>,
        // device:       &B::Device,
    ) -> Tensor<B, 3> {
        let x_device = x.device();
        let x = x.clone()
            + self.attn.forward_async(
                self.attn_norm.forward(x), rope, None, pad_mask, &x_device,
            ).await;
        x.clone() + self.mlp.forward(self.mlp_norm.forward(x))
    }
}

// ── DecoderBlock ─────────────────────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct DecoderBlockConfig {
    pub d_model:  usize,
    pub d_hidden: usize,
    pub n_heads:  usize,
    #[config(default = 64)]
    pub block_q:  usize,
    #[config(default = 64)]
    pub block_k:  usize,
}
 
impl DecoderBlockConfig {
    pub fn init<B: Backend<Device = WgpuDevice>>(&self, device: &B::Device) -> DecoderBlock<B> {
        let attn_cfg = || SelfAttentionConfig::new(self.d_model, self.n_heads)
            .with_block_q(self.block_q)
            .with_block_k(self.block_k);
        let cross_cfg = || CrossAttentionConfig::new(self.d_model, self.n_heads)
            .with_block_q(self.block_q)
            .with_block_k(self.block_k);
        DecoderBlock {
            self_attn_norm:  RMSNormConfig::new(self.d_model).init(device),
            self_attn:       attn_cfg().init(device),
            cross_attn_norm: RMSNormConfig::new(self.d_model).init(device),
            cross_attn:      cross_cfg().init(device),
            mlp_norm:        RMSNormConfig::new(self.d_model).init(device),
            mlp:             MLPConfig::new(self.d_model, self.d_hidden).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct DecoderBlock<B: Backend<Device = WgpuDevice>> {
    pub self_attn_norm:  RMSNorm<B>,
    pub self_attn:       SelfAttention<B>,
    pub cross_attn_norm: RMSNorm<B>,
    pub cross_attn:      CrossAttentionBlock<B>,
    pub mlp_norm:        RMSNorm<B>,
    pub mlp:             MLP<B>,
}

impl<B: Backend<Device = WgpuDevice>> DecoderBlock<B> {
    pub fn forward(
        &self,
        x:            Tensor<B, 3>,
        memory:       Tensor<B, 3>,
        rope:         &RotaryEncoding<B>,
        causal_mask:  Tensor<B, 2>,
        dec_pad_mask: Option<Tensor<B, 2, Bool>>,
        enc_pad_mask: Option<Tensor<B, 2, Bool>>,
        // device:       &B::Device,
    ) -> Tensor<B, 3> {
        let x_device = x.device();
        let x = x.clone()
            + self.self_attn.forward(
                self.self_attn_norm.forward(x),
                rope,
                Some(causal_mask),
                dec_pad_mask,
                &x_device,
            );
        let x = x.clone()
            + self.cross_attn.forward(
                self.cross_attn_norm.forward(x.clone()),
                memory,
                rope,
                enc_pad_mask,
                &x_device,
            );
        x.clone() + self.mlp.forward(self.mlp_norm.forward(x))
    }

    pub async fn forward_async(
        &self,
        x:            Tensor<B, 3>,
        memory:       Tensor<B, 3>,
        rope:         &RotaryEncoding<B>,
        causal_mask:  Tensor<B, 2>,
        dec_pad_mask: Option<Tensor<B, 2, Bool>>,
        enc_pad_mask: Option<Tensor<B, 2, Bool>>,
        // device:       &B::Device,
    ) -> Tensor<B, 3> {
        let x_device = x.device();
        let x = x.clone()
            + self.self_attn.forward_async(
                self.self_attn_norm.forward(x),
                rope,
                Some(causal_mask),
                dec_pad_mask,
                &x_device,
            ).await;
        let x = x.clone()
            + self.cross_attn.forward_async(
                self.cross_attn_norm.forward(x.clone()),
                memory,
                rope,
                enc_pad_mask,
                &x_device,
            ).await;
        x.clone() + self.mlp.forward(self.mlp_norm.forward(x))
    }
}