// ops.rs — Launch wrappers for the CubeCL FlashAttention kernels
//
// Rewritten to match cubecl 0.9 API:
//   - Generic over R: Runtime throughout
//   - R::client(device) for client acquisition
//   - client.empty(bytes) / client.create_from_slice for allocation
//   - TensorArg::from_raw_parts for kernel arguments
//   - Strides computed explicitly (row-major)

use burn::backend::wgpu::{WgpuDevice, WgpuRuntime};
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::brain::flash_attn::kernel::{flash_attn_backward, flash_attn_forward};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Byte size of an f32 tensor with given number of elements.
fn f32_bytes(n: usize) -> usize {
    n * core::mem::size_of::<f32>()
}

/// Row-major strides for a 3D tensor [d0, d1, d2].
/// stride[0] = d1*d2, stride[1] = d2, stride[2] = 1
fn strides_3d(d0: usize, d1: usize, d2: usize) -> [usize; 3] {
    [d1 * d2, d2, 1]
}

/// Row-major strides for a 1D tensor.
fn strides_1d(_d0: usize) -> [usize; 1] {
    [1]
}

// ── Forward pass ─────────────────────────────────────────────────────────────
//
// q, k, v are pre-reshaped to [batch*heads, seq, d] and provided as
// raw f32 slices. Returns (output, lse) handles.
//
// Returns:
//   out_handle : cubecl Handle for [bh, seq_q, d_v] f32 tensor
//   lse_handle : cubecl Handle for [bh, seq_q]      f32 tensor

pub struct ForwardResult {
    pub out: Handle,
    pub lse: Handle,
    /// Shapes saved for backward
    pub bh:    usize,
    pub seq_q: usize,
    pub seq_k: usize,
    pub d_k:   usize,
    pub d_v:   usize,
}

pub fn launch_forward<R: Runtime>(
    device:  &R::Device,
    q_data:  &[f32],   // [bh, seq_q, d_k]
    k_data:  &[f32],   // [bh, seq_k, d_k]
    v_data:  &[f32],   // [bh, seq_k, d_v]
    bh:      usize,
    seq_q:   usize,
    seq_k:   usize,
    d_k:     usize,
    d_v:     usize,
    block_q: usize,
    block_k: usize,
) -> ForwardResult {
    let client = R::client(device);

    // Upload input tensors
    let q_handle = client.create_from_slice(f32::as_bytes(q_data));
    let k_handle = client.create_from_slice(f32::as_bytes(k_data));
    let v_handle = client.create_from_slice(f32::as_bytes(v_data));

    // Allocate outputs
    let out_handle = client.empty(f32_bytes(bh * seq_q * d_v));
    let lse_handle = client.empty(f32_bytes(bh * seq_q));

    let scale = (d_k as f32).sqrt().recip();
    let num_q_tiles = seq_q.div_ceil(block_q);

    // Strides
    let [sq0, sq1, sq2] = strides_3d(bh, seq_q, d_k);
    let [sk0, sk1, sk2] = strides_3d(bh, seq_k, d_k);
    let [sv0, sv1, sv2] = strides_3d(bh, seq_k, d_v);
    let [so0, so1, so2] = strides_3d(bh, seq_q, d_v);
    let [sl0]           = strides_1d(bh * seq_q);

    unsafe {
        flash_attn_forward::launch::<R>(
            &client,
            CubeCount::Static(bh as u32, num_q_tiles as u32, 1),
            CubeDim::new_1d(block_q as u32),
            // q
            TensorArg::from_raw_parts::<f32>(
                &q_handle,
                &[sq0, sq1, sq2],
                &[bh, seq_q, d_k],
                1,
            ),
            // k
            TensorArg::from_raw_parts::<f32>(
                &k_handle,
                &[sk0, sk1, sk2],
                &[bh, seq_k, d_k],
                1,
            ),
            // v
            TensorArg::from_raw_parts::<f32>(
                &v_handle,
                &[sv0, sv1, sv2],
                &[bh, seq_k, d_v],
                1,
            ),
            // out
            TensorArg::from_raw_parts::<f32>(
                &out_handle,
                &[so0, so1, so2],
                &[bh, seq_q, d_v],
                1,
            ),
            // lse
            TensorArg::from_raw_parts::<f32>(
                &lse_handle,
                &[sl0],
                &[bh * seq_q],
                1,
            ),
            ScalarArg::new(scale),
            ScalarArg::new(seq_q as usize),
            ScalarArg::new(seq_k as usize),
            ScalarArg::new(d_k as usize),
            ScalarArg::new(d_v as usize),
            ScalarArg::new(block_q as usize),
            ScalarArg::new(block_k as usize),
        );
    }

    ForwardResult { out: out_handle, lse: lse_handle, bh, seq_q, seq_k, d_k, d_v }
}

// ── Backward pass ─────────────────────────────────────────────────────────────

pub struct BackwardResult {
    pub d_q: Handle,
    pub d_k: Handle,
    pub d_v: Handle,
}

pub fn launch_backward<R: Runtime>(
    device:    &R::Device,
    q_data:    &[f32],
    k_data:    &[f32],
    v_data:    &[f32],
    out_data:  &[f32],
    dout_data: &[f32],
    lse_data:  &[f32],
    bh:        usize,
    seq_q:     usize,
    seq_k:     usize,
    d_k:       usize,
    d_v:       usize,
    block_q:   usize,
    block_k:   usize,
) -> BackwardResult {
    let client = R::client(device);

    let q_handle    = client.create_from_slice(f32::as_bytes(q_data));
    let k_handle    = client.create_from_slice(f32::as_bytes(k_data));
    let v_handle    = client.create_from_slice(f32::as_bytes(v_data));
    let out_handle  = client.create_from_slice(f32::as_bytes(out_data));
    let dout_handle = client.create_from_slice(f32::as_bytes(dout_data));
    let lse_handle  = client.create_from_slice(f32::as_bytes(lse_data));

    // Zero-init gradient buffers
    let zeros_qk = vec![0.0f32; bh * seq_q * d_k];
    let zeros_kk = vec![0.0f32; bh * seq_k * d_k];
    let zeros_kv = vec![0.0f32; bh * seq_k * d_v];
    let dq_handle = client.create_from_slice(f32::as_bytes(&zeros_qk));
    let dk_handle = client.create_from_slice(f32::as_bytes(&zeros_kk));
    let dv_handle = client.create_from_slice(f32::as_bytes(&zeros_kv));

    let scale       = (d_k as f32).sqrt().recip();
    let num_q_tiles = seq_q.div_ceil(block_q);

    let [sq0, sq1, sq2]  = strides_3d(bh, seq_q, d_k);
    let [sk0, sk1, sk2]  = strides_3d(bh, seq_k, d_k);
    let [sv0, sv1, sv2]  = strides_3d(bh, seq_k, d_v);
    let [so0, so1, so2]  = strides_3d(bh, seq_q, d_v);
    let [sl0]            = strides_1d(bh * seq_q);

    unsafe {
        flash_attn_backward::launch::<R>(
            &client,
            CubeCount::Static(bh as u32, num_q_tiles as u32, 1),
            CubeDim::new_1d(block_q as u32),
            TensorArg::from_raw_parts::<f32>(&q_handle,    &[sq0, sq1, sq2], &[bh, seq_q, d_k], 1),
            TensorArg::from_raw_parts::<f32>(&k_handle,    &[sk0, sk1, sk2], &[bh, seq_k, d_k], 1),
            TensorArg::from_raw_parts::<f32>(&v_handle,    &[sv0, sv1, sv2], &[bh, seq_k, d_v], 1),
            TensorArg::from_raw_parts::<f32>(&out_handle,  &[so0, so1, so2], &[bh, seq_q, d_v], 1),
            TensorArg::from_raw_parts::<f32>(&dout_handle, &[so0, so1, so2], &[bh, seq_q, d_v], 1),
            TensorArg::from_raw_parts::<f32>(&lse_handle,  &[sl0],           &[bh * seq_q],      1),
            TensorArg::from_raw_parts::<f32>(&dq_handle,   &[sq0, sq1, sq2], &[bh, seq_q, d_k], 1),
            TensorArg::from_raw_parts::<f32>(&dk_handle,   &[sk0, sk1, sk2], &[bh, seq_k, d_k], 1),
            TensorArg::from_raw_parts::<f32>(&dv_handle,   &[sv0, sv1, sv2], &[bh, seq_k, d_v], 1),
            ScalarArg::new(scale),
            ScalarArg::new(seq_q as usize),
            ScalarArg::new(seq_k as usize),
            ScalarArg::new(d_k as usize),
            ScalarArg::new(d_v as usize),
            ScalarArg::new(block_q as usize),
            ScalarArg::new(block_k as usize),
        );
    }

    BackwardResult { d_q: dq_handle, d_k: dk_handle, d_v: dv_handle }
}

// ── Read-back helper ──────────────────────────────────────────────────────────
//
// Synchronously reads a handle back to a Vec<f32>.
// Use only for testing / debugging — not in the training hot path.

pub fn read_f32<R: Runtime>(device: &R::Device, handle: Handle, n: usize) -> Vec<f32> {
    let client = R::client(device);
    let bytes  = client.read_one(handle);
    f32::from_bytes(&bytes)[..n].to_vec()
}

// ── Convenience type alias for your Burn WGPU backend ────────────────────────

// pub type WgpuForwardResult  = ForwardResult<WgpuRuntime>;
// pub type WgpuBackwardResult = BackwardResult<WgpuRuntime>;

/// Entry point typed for Burn's WgpuDevice — call this from your Burn modules.
pub fn forward_wgpu(
    device:  &WgpuDevice,
    q:       &[f32],
    k:       &[f32],
    v:       &[f32],
    bh:      usize,
    seq_q:   usize,
    seq_k:   usize,
    d_k:     usize,
    d_v:     usize,
    block_q: usize,
    block_k: usize,
) -> ForwardResult {
    launch_forward::<WgpuRuntime>(
        device, q, k, v, bh, seq_q, seq_k, d_k, d_v, block_q, block_k,
    )
}