// ═════════════════════════════════════════════════════════════════════════════
// Causal mask helper
// ═════════════════════════════════════════════════════════════════════════════

// use burn::nn::RotaryEncoding;

// use crate::brain::model::{MLP, MLPConfig, RMSNorm, RMSNormConfig};
// use burn::{
//     config::Config, module::Module, nn::{Initializer, Linear, LinearConfig}, tensor::{Bool, Device, Tensor, backend::Backend}
// };
// use burn::backend::wgpu::{WgpuDevice, WgpuRuntime};


// pub fn causal_mask<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 2> {
//     use std::f32::NEG_INFINITY;
//     Tensor::full([seq_len, seq_len], NEG_INFINITY, device).triu(1)
// }


// // ═════════════════════════════════════════════════════════════════════════════
// // Multi-Head Self-Attention  (with RoPE, optional causal mask, optional pad mask)
// // ═════════════════════════════════════════════════════════════════════════════

// #[derive(Config, Debug)]
// pub struct SelfAttentionConfig {
//     d_model:  usize,
//     n_heads:  usize,
// }

// impl SelfAttentionConfig {
//     pub fn init<B: Backend>(&self, device: &B::Device) -> SelfAttention<B> {
//         assert!(self.d_model % self.n_heads == 0);
//         let head_dim = self.d_model / self.n_heads;
//         SelfAttention {
//             q: LinearConfig::new(self.d_model, self.d_model)
//             .with_initializer(Initializer::XavierUniform { gain: 1.0 })
//             .with_bias(false).init(device),
//             k: LinearConfig::new(self.d_model, self.d_model)
//             .with_initializer(Initializer::XavierUniform { gain: 1.0 })
//             .with_bias(false).init(device),
//             v: LinearConfig::new(self.d_model, self.d_model)
//             .with_initializer(Initializer::XavierUniform { gain: 1.0 })
//             .with_bias(false).init(device),
//             o: LinearConfig::new(self.d_model, self.d_model)
//             .with_initializer(Initializer::XavierUniform { gain: 1.0 })
//             .with_bias(false).init(device),
//             n_heads: self.n_heads,
//             head_dim,
//         }
//     }
// }

// #[derive(Module, Debug)]
// pub struct SelfAttention<B: Backend> {
//     q: Linear<B>,
//     k: Linear<B>,
//     v: Linear<B>,
//     o: Linear<B>,
//     n_heads:  usize,
//     head_dim: usize,
// }

// impl<B: Backend> SelfAttention<B> {
//     /// x:          [batch, seq, d_model]
//     /// rope:       shared RotaryEncoding
//     /// causal_mask: optional [seq, seq] additive mask (NEG_INFINITY upper triangle)
//     /// pad_mask:   optional [batch, seq] bool — true where padded
//     pub fn forward(
//         &self,
//         x:           Tensor<B, 3>,
//         rope:        &RotaryEncoding<B>,
//         causal_mask: Option<Tensor<B, 2>>,
//         pad_mask:    Option<Tensor<B, 2, Bool>>,
//     ) -> Tensor<B, 3> {
//         let [batch, seq, _] = x.dims();
//         let scale = (self.head_dim as f64).powf(-0.25);

//         let reshape = |t: Tensor<B, 3>| {
//             t.reshape([batch, seq, self.n_heads, self.head_dim])
//              .swap_dims(1, 2)          // [batch, heads, seq, head_dim]
//         };

//         // let q = rope.forward(reshape(self.q.forward(x.clone()))) * scale;
//         // let k = rope.forward(reshape(self.k.forward(x.clone()))) * scale;
//         let q = rope.forward(reshape(self.q.forward(x.clone())));
//         let k = rope.forward(reshape(self.k.forward(x.clone())));
//         let v =              reshape(self.v.forward(x));

//         let mut qk = q.matmul(k.transpose()); // [batch, heads, seq, seq]

//         // additive causal mask
//         if let Some(mask) = causal_mask {
//             qk = qk + mask.slice([0..seq, 0..seq]).unsqueeze::<4>();
//         }

//         // padding mask: set attention to -inf where key is a pad token
//         if let Some(pmask) = pad_mask {
//             // pmask: [batch, seq] → [batch, 1, 1, seq]
//             let pmask_f = pmask
//                 .unsqueeze_dim::<3>(1)
//                 .unsqueeze_dim::<4>(2)
//                 .float()
//                 .mul_scalar(-1e9);
//             qk = qk + pmask_f;
//         }

//         let w = burn::tensor::activation::softmax(qk, 3);
//         let out = w.matmul(v)              // [batch, heads, seq, head_dim]
//             .swap_dims(1, 2)               // [batch, seq, heads, head_dim]
//             .flatten(2, 3);                // [batch, seq, d_model]

//         self.o.forward(out)
//     }
// }


// // ═════════════════════════════════════════════════════════════════════════════
// // Cross-Attention  (Q from decoder, K/V from encoder memory)
// // ═════════════════════════════════════════════════════════════════════════════

// #[derive(Config, Debug)]
// pub struct CrossAttentionConfig {
//     d_model: usize,
//     n_heads: usize,
// }

// impl CrossAttentionConfig {
//     pub fn init<B: Backend>(&self, device: &B::Device) -> CrossAttentionBlock<B> {
//         assert!(self.d_model % self.n_heads == 0);
//         let head_dim = self.d_model / self.n_heads;
//         CrossAttentionBlock {
//             q: LinearConfig::new(self.d_model, self.d_model).with_bias(false).init(device),
//             k: LinearConfig::new(self.d_model, self.d_model).with_bias(false).init(device),
//             v: LinearConfig::new(self.d_model, self.d_model).with_bias(false).init(device),
//             o: LinearConfig::new(self.d_model, self.d_model).with_bias(false).init(device),
//             gate: LinearConfig::new(self.d_model, self.n_heads).with_bias(true).init(device),
//             n_heads: self.n_heads,
//             head_dim,
//         }
//     }
// }

// #[derive(Module, Debug)]
// pub struct CrossAttentionBlock<B: Backend> {
//     q: Linear<B>,
//     k: Linear<B>,
//     v: Linear<B>,
//     o: Linear<B>,
//     n_heads:  usize,
//     head_dim: usize,
//     gate: Linear<B>,      // new: d_model → n_heads
// }

// impl<B: Backend> CrossAttentionBlock<B> {
//     /// x:      [batch, dec_len, d_model]  — decoder states
//     /// memory: [batch, enc_len, d_model]  — encoder output
//     /// rope:   shared RotaryEncoding (applied to both Q and K)
//     /// enc_pad_mask: optional [batch, enc_len] bool — true where encoder input was padded
//     pub fn forward(
//         &self,
//         x:            Tensor<B, 3>,
//         memory:       Tensor<B, 3>,
//         rope:         &RotaryEncoding<B>,
//         enc_pad_mask: Option<Tensor<B, 2, Bool>>,
//     ) -> Tensor<B, 3> {
//         let [batch, dec_len, _] = x.dims();
//         let [_, enc_len, _]     = memory.dims();
//         let scale = (self.head_dim as f64).powf(-0.25);

//         let reshape_dec = |t: Tensor<B, 3>| {
//             t.reshape([batch, dec_len, self.n_heads, self.head_dim])
//              .swap_dims(1, 2)
//         };
//         let reshape_enc = |t: Tensor<B, 3>| {
//             t.reshape([batch, enc_len, self.n_heads, self.head_dim])
//              .swap_dims(1, 2)
//         };

//         // Q from decoder, K/V from memory; RoPE on Q and K
//         // let q = rope.forward(reshape_dec(self.q.forward(x)))      * scale;
//         // let k = rope.forward(reshape_enc(self.k.forward(memory.clone()))) * scale;
//         // perhaps rope should only be used in the selfattention not crossattention?
//         // let q = reshape_dec(self.q.forward(x.clone()))      * scale;
//         // let k = reshape_enc(self.k.forward(memory.clone())) * scale;
//         let q = reshape_dec(self.q.forward(x.clone()));
//         let k = reshape_enc(self.k.forward(memory.clone()));
//         let v = reshape_enc(self.v.forward(memory));

//         let mut qk = q.matmul(k.transpose()); // [batch, heads, dec_len, enc_len]

//         if let Some(pmask) = enc_pad_mask {
//             let pmask_f = pmask
//                 .unsqueeze_dim::<3>(1)
//                 .unsqueeze_dim::<4>(2)
//                 .float()
//                 .mul_scalar(-1e9);
//             qk = qk + pmask_f;
//         }

//         // normal only softmax
//         let w = burn::tensor::activation::softmax(qk, 3);
//         let out = w.matmul(v)
//             .swap_dims(1, 2)
//             .flatten(2, 3);

//         self.o.forward(out)

//         // alternative, softmax with sigmoid gate
//         // let w = burn::tensor::activation::softmax(qk, 3);
    
//         // // Keep as 4D [batch, heads, dec_len, head_dim]
//         // let out_4d = w.matmul(v)
//         //     .swap_dims(1, 2); // [batch, dec_len, heads, head_dim]

//         // // Gate: [batch, dec_len, n_heads] → [batch, dec_len, n_heads, 1]
//         // let gates = sigmoid(self.gate.forward(x))
//         //     .reshape([batch, dec_len, self.n_heads, 1]);

//         // // Broadcast over head_dim, then flatten
//         // let out = (out_4d * gates).flatten(2, 3); // [batch, dec_len, d_model]

//         // self.o.forward(out)
//     }
// }


// // ═════════════════════════════════════════════════════════════════════════════
// // Encoder Block
// //   pre-norm → self-attn → residual
// //   pre-norm → MLP       → residual
// // ═════════════════════════════════════════════════════════════════════════════

// #[derive(Config, Debug)]
// pub struct EncoderBlockConfig {
//     d_model:  usize,
//     d_hidden: usize,
//     n_heads:  usize,
// }

// impl EncoderBlockConfig {
//     pub fn init<B: Backend>(&self, device: &B::Device) -> EncoderBlock<B> {
//         EncoderBlock {
//             attn_norm: RMSNormConfig::new(self.d_model).init(device),
//             attn:      SelfAttentionConfig::new(self.d_model, self.n_heads).init(device),
//             mlp_norm:  RMSNormConfig::new(self.d_model).init(device),
//             mlp:       MLPConfig::new(self.d_model, self.d_hidden).init(device),
//             // mlp: MoEConfig::new(self.d_model, self.d_hidden, 4).init(device), // doesnt help, not with soft
//         }
//     }
// }

// #[derive(Module, Debug)]
// pub struct EncoderBlock<B: Backend> {
//     attn_norm: RMSNorm<B>,
//     attn:      SelfAttention<B>,
//     mlp_norm:  RMSNorm<B>,
//     mlp:       MLP<B>,
//     // mlp: MoE<B>,
// }

// impl<B: Backend> EncoderBlock<B> {
//     pub fn forward(
//         &self,
//         x:        Tensor<B, 3>,
//         rope:     &RotaryEncoding<B>,
//         pad_mask: Option<Tensor<B, 2, Bool>>,
//     ) -> Tensor<B, 3> {
//         // self-attention with residual
//         let x = x.clone()
//             + self.attn.forward(self.attn_norm.forward(x), rope, None, pad_mask);
//         // MLP with residual
//         x.clone() + self.mlp.forward(self.mlp_norm.forward(x))
//     }
// }


// // ═════════════════════════════════════════════════════════════════════════════
// // Decoder Block
// //   pre-norm → masked self-attn  → residual
// //   pre-norm → cross-attn        → residual
// //   pre-norm → MLP               → residual
// // ═════════════════════════════════════════════════════════════════════════════

// #[derive(Config, Debug)]
// pub struct DecoderBlockConfig {
//     d_model:  usize,
//     d_hidden: usize,
//     n_heads:  usize,
// }

// impl DecoderBlockConfig {
//     pub fn init<B: Backend>(&self, device: &B::Device) -> DecoderBlock<B> {
//         DecoderBlock {
//             self_attn_norm:  RMSNormConfig::new(self.d_model).init(device),
//             self_attn:       SelfAttentionConfig::new(self.d_model, self.n_heads).init(device),
//             cross_attn_norm: RMSNormConfig::new(self.d_model).init(device),
//             cross_attn:      CrossAttentionConfig::new(self.d_model, self.n_heads).init(device),
//             mlp_norm:        RMSNormConfig::new(self.d_model).init(device),
//             mlp:             MLPConfig::new(self.d_model, self.d_hidden).init(device),
//             // mlp: MoEConfig::new(self.d_model, self.d_hidden, 4).init(device),
//         }
//     }
// }

// #[derive(Module, Debug)]
// pub struct DecoderBlock<B: Backend> {
//     self_attn_norm:  RMSNorm<B>,
//     self_attn:       SelfAttention<B>,
//     cross_attn_norm: RMSNorm<B>,
//     cross_attn:      CrossAttentionBlock<B>,
//     mlp_norm:        RMSNorm<B>,
//     mlp:             MLP<B>,
//     // mlp:             MoE<B>
// }

// impl<B: Backend> DecoderBlock<B> {
//     pub fn forward(
//         &self,
//         x:            Tensor<B, 3>,
//         memory:       Tensor<B, 3>,
//         rope:         &RotaryEncoding<B>,
//         causal_mask:  Tensor<B, 2>,
//         dec_pad_mask: Option<Tensor<B, 2, Bool>>,
//         enc_pad_mask: Option<Tensor<B, 2, Bool>>,
//     ) -> Tensor<B, 3> {
//         // masked self-attention
//         let x = x.clone()
//             + self.self_attn.forward(
//                 self.self_attn_norm.forward(x),
//                 rope,
//                 Some(causal_mask),
//                 dec_pad_mask,
//             );
//         // cross-attention
//         let x = x.clone()
//             + self.cross_attn.forward(
//                 self.cross_attn_norm.forward(x.clone()),
//                 memory,
//                 rope,
//                 enc_pad_mask,
//             );
//         // MLP
//         x.clone() + self.mlp.forward(self.mlp_norm.forward(x))
//     }
// }