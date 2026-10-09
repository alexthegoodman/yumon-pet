// backend.rs - causal FlashAttention as a Burn backend op
//
// `FlashAttentionKernels` is the raw forward/backward pair, implemented for
// burn-cubecl's `CubeBackend` (wgpu and CUDA, no fusion) on device-resident
// tensors. `FlashAttention` is the differentiable op the model calls; on
// `Autodiff` it registers a compute-bound backward step that checkpoints
// Q/K/V through the checkpointer, so it composes with BalancedCheckpointing
// like Burn's own matmul does. The output and log-sum-exp are kept as state
// (O(seq) per row), the [seq, seq] score matrix never exists.

use burn::{
    backend::autodiff::{
        Autodiff, NodeId,
        checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind},
    },
    prelude::*,
    tensor::{TensorPrimitive, ops::FloatTensor},
};
use burn_cubecl::{
    BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement,
    kernel::into_contiguous, ops::numeric::empty_device, tensor::CubeTensor,
};
use cubecl::prelude::{CubeCount, CubeDim, ScalarArg};

use super::kernel::{causal_bwd_dkdv, causal_bwd_dq, causal_fwd};

/// Rows per tile for a head width: two staged [block, dim] f32 tiles must fit
/// the device's shared memory (WebGPU's default is 16 KiB; native wgpu and
/// CUDA devices usually allow 32-64 KiB). Capped at 64 units per cube and
/// rounded down to a multiple of 8 (subgroup width) when possible.
pub fn block_for_dim(dim: usize, max_shared_bytes: usize) -> usize {
    let block = (max_shared_bytes / 4 / (2 * dim)).clamp(1, 64);
    if block >= 8 { block / 8 * 8 } else { block }
}

pub trait FlashAttentionKernels: Backend {
    /// q, k, v: [batch, heads, seq, dim]. Returns (output, lse [batch, heads, seq]).
    fn causal_flash_fwd(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>);

    /// Returns (dq, dk, dv).
    fn causal_flash_bwd(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        out: FloatTensor<Self>,
        lse: FloatTensor<Self>,
        d_out: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>);
}

pub trait FlashAttention: Backend {
    /// softmax(q kᵀ / √dim + causal mask) v, for q, k, v: [batch, heads, seq, dim].
    fn causal_flash_attention(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
}

/// Causal self-attention over [batch, heads, seq, dim] without materializing
/// the [seq, seq] scores. Scaling by 1/√dim happens inside the kernel.
pub fn causal_flash_attention<B: FlashAttention>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
) -> Tensor<B, 4> {
    assert_eq!(q.dims(), k.dims(), "flash attention: q/k shape mismatch");
    assert_eq!(q.dims(), v.dims(), "flash attention: q/v shape mismatch");
    Tensor::from_primitive(TensorPrimitive::Float(B::causal_flash_attention(
        q.into_primitive().tensor(),
        k.into_primitive().tensor(),
        v.into_primitive().tensor(),
    )))
}

// ── CubeBackend: kernel launches ─────────────────────────────────────────────

struct Layout {
    bh:    usize,
    seq:   usize,
    dim:   usize,
    block: usize,
    tiles: usize,
}

fn layout<R: CubeRuntime>(q: &CubeTensor<R>) -> Layout {
    let [batch, heads, seq, dim] = q.shape.dims::<4>();
    assert!(seq > 0 && dim > 0, "flash attention: empty input");
    assert!(matches!(q.dtype, burn::tensor::DType::F32 | burn::tensor::DType::BF16), "flash attention supports FP32 and BF16");
    let block = block_for_dim(dim, q.client.properties().hardware.max_shared_memory_size);
    assert!(block <= q.client.properties().hardware.max_units_per_cube as usize);
    Layout { bh: batch * heads, seq, dim, block, tiles: seq.div_ceil(block) }
}

fn launch_fwd<R: CubeRuntime, E: FloatElement>(
    q: CubeTensor<R>, k: CubeTensor<R>, v: CubeTensor<R>,
) -> (CubeTensor<R>, CubeTensor<R>) {
    let (q, k, v) = (into_contiguous(q), into_contiguous(k), into_contiguous(v));
    let shape = q.shape.clone();
    let [batch, heads, seq, _] = shape.dims::<4>();
    let n = layout(&q);
    let out = empty_device::<R, E>(q.client.clone(), q.device.clone(), shape);
    let lse = empty_device::<R, f32>(q.client.clone(), q.device.clone(), [batch, heads, seq].into());

    causal_fwd::launch::<E, R>(
        &q.client,
        CubeCount::Static(n.bh as u32, n.tiles as u32, 1),
        CubeDim::new_1d(n.block as u32),
        q.as_tensor_arg(1),
        k.as_tensor_arg(1),
        v.as_tensor_arg(1),
        out.as_tensor_arg(1),
        lse.as_tensor_arg(1),
        ScalarArg::new((n.dim as f32).sqrt().recip()),
        ScalarArg::new(n.seq),
        n.block,
        n.dim,
        n.block * n.dim,
    )
    .expect("flash attention forward launch");

    (out, lse)
}

fn launch_bwd<R: CubeRuntime, E: FloatElement>(
    q: CubeTensor<R>, k: CubeTensor<R>, v: CubeTensor<R>,
    out: CubeTensor<R>, lse: CubeTensor<R>, d_out: CubeTensor<R>,
) -> (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>) {
    let (q, k, v) = (into_contiguous(q), into_contiguous(k), into_contiguous(v));
    let (out, lse, d_out) = (into_contiguous(out), into_contiguous(lse), into_contiguous(d_out));
    let n = layout(&q);
    let client = q.client.clone();
    let device = q.device.clone();
    let scale = (n.dim as f32).sqrt().recip();
    let delta = empty_device::<R, f32>(client.clone(), device.clone(), lse.shape.clone());
    let d_q = empty_device::<R, E>(client.clone(), device.clone(), q.shape.clone());
    let d_k = empty_device::<R, E>(client.clone(), device.clone(), q.shape.clone());
    let d_v = empty_device::<R, E>(client.clone(), device.clone(), q.shape.clone());
    let count = CubeCount::Static(n.bh as u32, n.tiles as u32, 1);

    // Same stream: delta is complete before causal_bwd_dkdv reads it.
    causal_bwd_dq::launch::<E, R>(
        &client,
        count.clone(),
        CubeDim::new_1d(n.block as u32),
        q.as_tensor_arg(1),
        k.as_tensor_arg(1),
        v.as_tensor_arg(1),
        out.as_tensor_arg(1),
        d_out.as_tensor_arg(1),
        lse.as_tensor_arg(1),
        delta.as_tensor_arg(1),
        d_q.as_tensor_arg(1),
        ScalarArg::new(scale),
        ScalarArg::new(n.seq),
        n.block,
        n.dim,
        n.block * n.dim,
    )
    .expect("flash attention dQ launch");

    causal_bwd_dkdv::launch::<E, R>(
        &client,
        count,
        CubeDim::new_1d(n.block as u32),
        q.as_tensor_arg(1),
        k.as_tensor_arg(1),
        v.as_tensor_arg(1),
        d_out.as_tensor_arg(1),
        lse.as_tensor_arg(1),
        delta.as_tensor_arg(1),
        d_k.as_tensor_arg(1),
        d_v.as_tensor_arg(1),
        ScalarArg::new(scale),
        ScalarArg::new(n.seq),
        ScalarArg::new(n.tiles),
        n.block,
        n.dim,
        n.block * n.dim,
    )
    .expect("flash attention dK/dV launch");

    (d_q, d_k, d_v)
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> FlashAttentionKernels
    for CubeBackend<R, F, I, BT>
{
    fn causal_flash_fwd(
        q: FloatTensor<Self>, k: FloatTensor<Self>, v: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        assert_eq!(q.dtype, k.dtype, "flash attention: q/k dtype mismatch");
        assert_eq!(q.dtype, v.dtype, "flash attention: q/v dtype mismatch");
        match q.dtype {
            burn::tensor::DType::F32 => launch_fwd::<R, f32>(q, k, v),
            burn::tensor::DType::BF16 => launch_fwd::<R, burn::tensor::bf16>(q, k, v),
            dtype => panic!("flash attention: unsupported dtype {dtype:?}"),
        }
    }

    fn causal_flash_bwd(
        q: FloatTensor<Self>, k: FloatTensor<Self>, v: FloatTensor<Self>,
        out: FloatTensor<Self>, lse: FloatTensor<Self>, d_out: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        for tensor in [&k, &v, &out, &d_out] {
            assert_eq!(q.dtype, tensor.dtype, "flash attention: backward dtype mismatch");
        }
        assert_eq!(lse.dtype, burn::tensor::DType::F32);
        match q.dtype {
            burn::tensor::DType::F32 => launch_bwd::<R, f32>(q, k, v, out, lse, d_out),
            burn::tensor::DType::BF16 => launch_bwd::<R, burn::tensor::bf16>(q, k, v, out, lse, d_out),
            dtype => panic!("flash attention: unsupported dtype {dtype:?}"),
        }
    }
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> FlashAttention
    for CubeBackend<R, F, I, BT>
{
    fn causal_flash_attention(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        Self::causal_flash_fwd(q, k, v).0
    }
}

// ── Autodiff: backward step ──────────────────────────────────────────────────

impl<B: FlashAttentionKernels, C: CheckpointStrategy> FlashAttention for Autodiff<B, C> {
    fn causal_flash_attention(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct CausalFlashBackward;

        impl<B: FlashAttentionKernels> Backward<B, 3> for CausalFlashBackward {
            // Q/K/V by checkpoint id (kept or recomputed per strategy), O and LSE by value.
            type State = (NodeId, NodeId, NodeId, FloatTensor<B>, FloatTensor<B>);

            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_q, node_k, node_v] = ops.parents;
                let d_out = grads.consume::<B>(&ops.node);
                let (q_id, k_id, v_id, out, lse) = ops.state;
                let q = checkpointer.retrieve_node_output(q_id);
                let k = checkpointer.retrieve_node_output(k_id);
                let v = checkpointer.retrieve_node_output(v_id);

                let (d_q, d_k, d_v) = B::causal_flash_bwd(q, k, v, out, lse, d_out);

                if let Some(node) = node_q { grads.register::<B>(node.id, d_q); }
                if let Some(node) = node_k { grads.register::<B>(node.id, d_k); }
                if let Some(node) = node_v { grads.register::<B>(node.id, d_v); }
            }
        }

        match CausalFlashBackward
            .prepare::<C>([q.node.clone(), k.node.clone(), v.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let q_id = prep.checkpoint(&q);
                let k_id = prep.checkpoint(&k);
                let v_id = prep.checkpoint(&v);
                let (out, lse) = B::causal_flash_fwd(q.primitive, k.primitive, v.primitive);
                prep.finish((q_id, k_id, v_id, out.clone(), lse), out)
            }
            OpsKind::UnTracked(prep) => {
                prep.finish(B::causal_flash_fwd(q.primitive, k.primitive, v.primitive).0)
            }
        }
    }
}
