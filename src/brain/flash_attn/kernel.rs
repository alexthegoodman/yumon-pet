// kernel.rs - CubeCL causal FlashAttention (forward + two-pass backward)
//
// Tensors are contiguous [batch*heads, seq, dim] FP32 or BF16.
// Storage uses F; shared tiles, accumulators, LSE and delta stay FP32. Self-attention only
// (seq_q == seq_k) and always causal: query row i sees keys 0..=i.
//
// Launch config for every kernel:
//   cube_count = (batch * heads, ceil(seq / block), 1)
//   cube_dim   = (block, 1, 1) - one unit per row of the tile
//
// K/V (forward, dQ) or Q/dO (dK/dV) tiles are staged in shared memory, so
// `block * dim` floats per tile must fit twice in the shared budget (see
// `block_for_dim` in backend.rs). Every unit reads the same tile row at the
// same time (a broadcast). The unit's own row and its accumulators live in
// `dim`-wide local arrays; every `dim` loop is unrolled so those stay in
// registers. Measured on a UHD 770 (dim 32, seq 256): rolled loops 420 ms per
// layer forward+backward, unrolled 65 ms. Staging the own row in shared
// memory instead was slower (182 ms, 199 ms with padded rows).
//
// Units past the end of the sequence still load tiles and hit every
// sync_cube(); they skip the math and the writes.
//
// The backward is split so every unit owns the rows it writes:
//   causal_bwd_dq   - one unit per query row: delta = rowsum(dO * O), dQ
//   causal_bwd_dkdv - one unit per key row:   dK, dV
// No atomics, no zero-initialized outputs.
//
// CubeCL note: `let mut x = 0.0_f32` (or f32::NEG_INFINITY) expands to a
// constant that can't be reassigned, so mutable scalars start from
// f32::from_int(0) / f32::min_value(). min_value() also stands in for -inf
// as the masked score: exp(min_value - m) underflows to exactly 0.

use cubecl::prelude::*;

#[cube(launch)]
pub fn causal_fwd<F: Float>(
    q:     &Tensor<F>,
    k:     &Tensor<F>,
    v:     &Tensor<F>,
    out:   &mut Tensor<F>,
    lse:   &mut Tensor<f32>, // [bh, seq], log-sum-exp of the scaled scores
    scale: f32,
    seq:   usize,
    #[comptime] block: usize,
    #[comptime] dim:   usize,
    #[comptime] tile:  usize, // block * dim
) {
    let bh     = CUBE_POS_X as usize;
    let q_tile = CUBE_POS_Y as usize;
    let t      = UNIT_POS_X as usize;
    let row    = q_tile * block + t;
    let live   = row < seq;
    let base   = bh * seq * dim;

    let mut k_s = SharedMemory::<f32>::new(tile);
    let mut v_s = SharedMemory::<f32>::new(tile);
    let mut q_r = Array::<f32>::new(dim);
    let mut acc = Array::<f32>::new(dim);
    let mut s   = Array::<f32>::new(block);

    #[unroll]
    for d in 0..dim {
        let mut x = f32::from_int(0);
        if live { x = f32::cast_from(q[base + row * dim + d]); }
        q_r[d] = x * scale;
        acc[d] = f32::from_int(0);
    }

    let mut m = f32::min_value();
    let mut l = f32::from_int(0);

    // Causal: key tiles past this query tile are fully masked, skip them.
    for kt in 0..q_tile + 1 {
        let k_row = kt * block + t;
        #[unroll]
        for d in 0..dim {
            let mut kx = f32::from_int(0);
            let mut vx = f32::from_int(0);
            if k_row < seq {
                kx = f32::cast_from(k[base + k_row * dim + d]);
                vx = f32::cast_from(v[base + k_row * dim + d]);
            }
            k_s[t * dim + d] = kx;
            v_s[t * dim + d] = vx;
        }
        sync_cube();

        if live {
            // Every processed tile holds at least key kt*block <= row, so
            // tile_max is a real score and m leaves min_value after the first
            // tile. Masked scores stay at min_value, so exp(score - m_new) == 0.
            let mut tile_max = f32::min_value();
            for j in 0..block {
                let mut score = f32::min_value();
                if kt * block + j <= row {
                    let mut dot = f32::from_int(0);
                    #[unroll]
                    for d in 0..dim {
                        dot += q_r[d] * k_s[j * dim + d];
                    }
                    score = dot;
                }
                s[j] = score;
                if score > tile_max { tile_max = score; }
            }

            let mut m_new = m;
            if tile_max > m_new { m_new = tile_max; }
            let correction = Exp::exp(m - m_new);
            l *= correction;
            #[unroll]
            for d in 0..dim {
                acc[d] *= correction;
            }
            for j in 0..block {
                let p = Exp::exp(s[j] - m_new);
                l += p;
                #[unroll]
                for d in 0..dim {
                    acc[d] += p * v_s[j * dim + d];
                }
            }
            m = m_new;
        }
        sync_cube();
    }

    if live {
        let inv = 1.0_f32 / l;
        #[unroll]
        for d in 0..dim {
            out[base + row * dim + d] = F::cast_from(acc[d] * inv);
        }
        lse[bh * seq + row] = m + Log::ln(l);
    }
}

#[cube(launch)]
pub fn causal_bwd_dq<F: Float>(
    q:     &Tensor<F>,
    k:     &Tensor<F>,
    v:     &Tensor<F>,
    out:   &Tensor<F>,
    d_out: &Tensor<F>,
    lse:   &Tensor<f32>,
    delta: &mut Tensor<f32>, // [bh, seq], written here, read by causal_bwd_dkdv
    d_q:   &mut Tensor<F>,
    scale: f32,
    seq:   usize,
    #[comptime] block: usize,
    #[comptime] dim:   usize,
    #[comptime] tile:  usize,
) {
    let bh     = CUBE_POS_X as usize;
    let q_tile = CUBE_POS_Y as usize;
    let t      = UNIT_POS_X as usize;
    let row    = q_tile * block + t;
    let live   = row < seq;
    let base   = bh * seq * dim;

    let mut k_s  = SharedMemory::<f32>::new(tile);
    let mut v_s  = SharedMemory::<f32>::new(tile);
    let mut q_r  = Array::<f32>::new(dim);
    let mut do_r = Array::<f32>::new(dim);
    let mut dq   = Array::<f32>::new(dim);

    let mut delta_r = f32::from_int(0);
    let mut lse_r = f32::from_int(0);
    #[unroll]
    for d in 0..dim {
        let mut qx = f32::from_int(0);
        let mut gx = f32::from_int(0);
        if live {
            qx = f32::cast_from(q[base + row * dim + d]);
            gx = f32::cast_from(d_out[base + row * dim + d]);
            delta_r += gx * f32::cast_from(out[base + row * dim + d]);
        }
        q_r[d]  = qx;
        do_r[d] = gx;
        dq[d]   = f32::from_int(0);
    }
    if live {
        lse_r = lse[bh * seq + row];
        delta[bh * seq + row] = delta_r;
    }

    for kt in 0..q_tile + 1 {
        let k_row = kt * block + t;
        #[unroll]
        for d in 0..dim {
            let mut kx = f32::from_int(0);
            let mut vx = f32::from_int(0);
            if k_row < seq {
                kx = f32::cast_from(k[base + k_row * dim + d]);
                vx = f32::cast_from(v[base + k_row * dim + d]);
            }
            k_s[t * dim + d] = kx;
            v_s[t * dim + d] = vx;
        }
        sync_cube();

        if live {
            for j in 0..block {
                if kt * block + j <= row {
                    let mut dot = f32::from_int(0);
                    let mut dp = f32::from_int(0);
                    #[unroll]
                    for d in 0..dim {
                        dot += q_r[d] * k_s[j * dim + d];
                        dp  += do_r[d] * v_s[j * dim + d];
                    }
                    let p  = Exp::exp(dot * scale - lse_r);
                    let ds = p * (dp - delta_r);
                    #[unroll]
                    for d in 0..dim {
                        dq[d] += ds * k_s[j * dim + d];
                    }
                }
            }
        }
        sync_cube();
    }

    if live {
        #[unroll]
        for d in 0..dim {
            d_q[base + row * dim + d] = F::cast_from(dq[d] * scale);
        }
    }
}

#[cube(launch)]
pub fn causal_bwd_dkdv<F: Float>(
    q:     &Tensor<F>,
    k:     &Tensor<F>,
    v:     &Tensor<F>,
    d_out: &Tensor<F>,
    lse:   &Tensor<f32>,
    delta: &Tensor<f32>,
    d_k:   &mut Tensor<F>,
    d_v:   &mut Tensor<F>,
    scale: f32,
    seq:   usize,
    tiles: usize, // ceil(seq / block)
    #[comptime] block: usize,
    #[comptime] dim:   usize,
    #[comptime] tile:  usize,
) {
    let bh     = CUBE_POS_X as usize;
    let k_tile = CUBE_POS_Y as usize;
    let t      = UNIT_POS_X as usize;
    let col    = k_tile * block + t;
    let live   = col < seq;
    let base   = bh * seq * dim;

    let mut q_s  = SharedMemory::<f32>::new(tile);
    let mut do_s = SharedMemory::<f32>::new(tile);
    let mut k_r  = Array::<f32>::new(dim);
    let mut v_r  = Array::<f32>::new(dim);
    let mut dk   = Array::<f32>::new(dim);
    let mut dv   = Array::<f32>::new(dim);

    #[unroll]
    for d in 0..dim {
        let mut kx = f32::from_int(0);
        let mut vx = f32::from_int(0);
        if live {
            kx = f32::cast_from(k[base + col * dim + d]);
            vx = f32::cast_from(v[base + col * dim + d]);
        }
        k_r[d] = kx;
        v_r[d] = vx;
        dk[d]  = f32::from_int(0);
        dv[d]  = f32::from_int(0);
    }

    // Causal: only query tiles at or after this key tile see these keys.
    for qt in k_tile..tiles {
        let q_row = qt * block + t;
        #[unroll]
        for d in 0..dim {
            let mut qx = f32::from_int(0);
            let mut gx = f32::from_int(0);
            if q_row < seq {
                qx = f32::cast_from(q[base + q_row * dim + d]);
                gx = f32::cast_from(d_out[base + q_row * dim + d]);
            }
            q_s[t * dim + d]  = qx;
            do_s[t * dim + d] = gx;
        }
        sync_cube();

        if live {
            for i in 0..block {
                let r = qt * block + i;
                if r >= col && r < seq {
                    let mut dot = f32::from_int(0);
                    let mut dp = f32::from_int(0);
                    #[unroll]
                    for d in 0..dim {
                        dot += k_r[d] * q_s[i * dim + d];
                        dp  += v_r[d] * do_s[i * dim + d];
                    }
                    let p  = Exp::exp(dot * scale - lse[bh * seq + r]);
                    let ds = p * (dp - delta[bh * seq + r]);
                    #[unroll]
                    for d in 0..dim {
                        dv[d] += p * do_s[i * dim + d];
                        dk[d] += ds * q_s[i * dim + d];
                    }
                }
            }
        }
        sync_cube();
    }

    if live {
        #[unroll]
        for d in 0..dim {
            d_k[base + col * dim + d] = F::cast_from(dk[d] * scale);
            d_v[base + col * dim + d] = F::cast_from(dv[d]);
        }
    }
}
