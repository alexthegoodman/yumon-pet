
// ** analysis #2 **

use std::env;
use std::fs::File;
use std::io::{BufReader, Read, Write};

const VOCAB_SIZE: usize = 4096;
const EMBED_DIM: usize = 64;
const FF_DIM: usize = 256;
const NUM_EXPERTS: usize = 4;

#[derive(Clone)]
struct BurnTensor {
    index: usize,
    data_offset: usize, // Exact byte offset where the tensor's f32 slice starts
    num_floats: usize,
    floats: Vec<f32>,
}

/// Extracts exact Burn tensors by combining the proven contiguous-float scanner
/// with Burn's little-endian byte-length prefix verification & tail-alignment snapping.
fn extract_burn_tensors(bytes: &[u8]) -> Vec<BurnTensor> {
    let mut tensors = Vec::new();
    let mut pos = 0;

    while pos + 4 <= bytes.len() {
        let f0 = f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        if f0.is_finite() && f0.abs() < 30.0 && (f0 == 0.0 || f0.abs() > 1e-6) {
            let start = pos;
            let mut count = 0usize;
            while pos + 4 <= bytes.len() {
                let v = f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
                if v.is_finite() && v.abs() < 30.0 && (v == 0.0 || v.abs() > 1e-9) {
                    count += 1;
                    pos += 4;
                } else {
                    break;
                }
            }

            let end_byte = start + count * 4;

            // Snap to exact architecture tensor sizes (262144, 16384, 4096, 256)
            // If `count` is off by 1-3 floats (e.g. 4097 or 16383), check where the exact tensor boundary is:
            let target_floats = if (262140..=262150).contains(&count) {
                Some(262144usize)
            } else if (16380..=16390).contains(&count) {
                Some(16384usize)
            } else if (4094..=4102).contains(&count) {
                Some(4096usize)
            } else if (254..=260).contains(&count) {
                Some(256usize)
            } else {
                None
            };

            if let Some(exact_len) = target_floats {
                let exact_bytes = exact_len * 4;
                // Find the exact start offset around `start` (within +/- 16 bytes)
                // by checking if a little-endian u32/u64 length prefix (either `exact_bytes` or `exact_len`) sits right before it,
                // OR by anchoring to the end of the contiguous float block if `count >= exact_len`.
                let mut best_start = if count >= exact_len {
                    // Header bytes at the front caused extra 1-3 floats; the end of the float array is exact!
                    end_byte - exact_bytes
                } else {
                    // 1-2 small floats at the very start were skipped (< 1e-6); anchor to the end!
                    end_byte.saturating_sub(exact_bytes)
                };

                // Double check +/- 16 bytes around `start` for an explicit LE u32 byte-length or element-count prefix
                let search_min = start.saturating_sub(16);
                let search_max = (start + 16).min(bytes.len().saturating_sub(exact_bytes));
                for cand in search_min..=search_max {
                    if cand >= 4 {
                        let prev_u32 = u32::from_le_bytes(bytes[cand - 4..cand].try_into().unwrap()) as usize;
                        if prev_u32 == exact_bytes || prev_u32 == exact_len {
                            best_start = cand;
                            break;
                        }
                    }
                }

                if best_start + exact_bytes <= bytes.len() {
                    let raw_slice = &bytes[best_start..best_start + exact_bytes];
                    let floats: Vec<f32> = raw_slice
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect();

                    tensors.push(BurnTensor {
                        index: tensors.len(),
                        data_offset: best_start,
                        num_floats: exact_len,
                        floats,
                    });
                }
            }
            continue;
        }
        pos += 1;
    }

    tensors
}

/// Computes top-K singular values and singular vectors (sigma, u, v) via deflated power iteration
fn svd_top_k(w: &[f32], rows: usize, cols: usize, k: usize) -> Vec<(f32, Vec<f32>, Vec<f32>)> {
    let mut results: Vec<(f32, Vec<f32>, Vec<f32>)> = Vec::with_capacity(k);

    for rank in 0..k {
        let mut v: Vec<f32> = (0..cols)
            .map(|j| (((j + rank * 17) as f32 * 1.6180339).sin()) + 0.05)
            .collect();
        normalize(&mut v);
        let mut u = vec![0.0f32; rows];

        for _ in 0..40 {
            for r in 0..rows {
                let row = &w[r * cols..(r + 1) * cols];
                let mut dot = 0.0f32;
                for c in 0..cols {
                    dot += row[c] * v[c];
                }
                u[r] = dot;
            }
            for (s_prev, u_prev, v_prev) in &results {
                let vd: f32 = v_prev.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                for r in 0..rows {
                    u[r] -= s_prev * u_prev[r] * vd;
                }
            }
            normalize(&mut u);

            for c in 0..cols {
                let mut dot = 0.0f32;
                for r in 0..rows {
                    dot += w[r * cols + c] * u[r];
                }
                v[c] = dot;
            }
            for (s_prev, u_prev, v_prev) in &results {
                let ud: f32 = u_prev.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
                for c in 0..cols {
                    v[c] -= s_prev * v_prev[c] * ud;
                }
            }
            normalize(&mut v);
        }

        let mut sigma = 0.0f32;
        for r in 0..rows {
            let row = &w[r * cols..(r + 1) * cols];
            let mut dot = 0.0f32;
            for c in 0..cols {
                dot += row[c] * v[c];
            }
            sigma += u[r] * dot;
        }
        for (s_prev, u_prev, v_prev) in &results {
            let ud: f32 = u_prev.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
            let vd: f32 = v_prev.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            sigma -= s_prev * ud * vd;
        }

        results.push((sigma.abs(), u, v));
    }

    results
}

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-12 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

fn l2_norm(w: &[f32]) -> f32 {
    w.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt() as f32
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| (x as f64) * (y as f64)).sum();
    let na = l2_norm(a) as f64;
    let nb = l2_norm(b) as f64;
    if na * nb > 1e-12 {
        (dot / (na * nb)) as f32
    } else {
        0.0
    }
}

fn matmul_64x64(a: &[f32], b: &[f32], transpose_a: bool) -> Vec<f32> {
    let mut out = vec![0.0f32; 64 * 64];
    for i in 0..64 {
        for j in 0..64 {
            let mut sum = 0.0f32;
            for k in 0..64 {
                let a_val = if transpose_a { a[k * 64 + i] } else { a[i * 64 + k] };
                sum += a_val * b[k * 64 + j];
            }
            out[i * 64 + j] = sum;
        }
    }
    out
}

fn analyze_embeddings_and_head(t_in: &BurnTensor, t_out: &BurnTensor) {
    println!("\n==================================================================");
    println!(" 1. INPUT EMBEDDING vs. OUTPUT LM_HEAD FORENSICS (4096 x 64)");
    println!("==================================================================");

    for (label, t) in [
        ("Input Embedding (Tensor #00)", t_in),
        ("Output LM_Head  (Last 4096x64)", t_out),
    ] {
        let l2 = l2_norm(&t.floats);
        let svd = svd_top_k(&t.floats, VOCAB_SIZE, EMBED_DIM, 5);
        let (s1, s2, s3, s4, s5) = (svd[0].0, svd[1].0, svd[2].0, svd[3].0, svd[4].0);
        let rank1_pct = (s1 * s1) / (l2 * l2) * 100.0;

        println!("  [{}] @ byte {}", label, t.data_offset);
        println!(
            "    Frobenius L2 Norm: {:.2} | Top-5 Singular Values: [{:.2}, {:.2}, {:.2}, {:.2}, {:.2}]",
            l2, s1, s2, s3, s4, s5
        );
        println!(
            "    s1/s2 Ratio: {:.2}x | Rank-1 Energy: {:.1}% | Top-5 Cumulative Energy: {:.1}%",
            s1 / s2.max(1e-6),
            rank1_pct,
            svd.iter().map(|(s, _, _)| s * s).sum::<f32>() / (l2 * l2) * 100.0
        );

        let u1 = &svd[0].1;
        let mut token_proj: Vec<(usize, f32, f32)> = (0..VOCAB_SIZE)
            .map(|tok_id| {
                let row = &t.floats[tok_id * EMBED_DIM..(tok_id + 1) * EMBED_DIM];
                let norm = l2_norm(row);
                (tok_id, u1[tok_id].abs() * s1, norm)
            })
            .collect();
        token_proj.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        print!("    Top 10 Token IDs driving Rank-1 Attractor (ID: s1_proj / row_norm): ");
        for (id, proj, norm) in token_proj.iter().take(10) {
            print!("#{}({:.2}/{:.2}) ", id, proj, norm);
        }
        println!("\n");
    }

    let emb_head_cos = cosine_sim(&t_in.floats, &t_out.floats);
    println!(
        "  Global Cosine Similarity between Input Embedding & Output LM_Head: {:.4}\n",
        emb_head_cos
    );
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).map(|s| s.as_str()).unwrap_or("model.bin");
    let fix_mode = args.iter().any(|a| a == "--fix-collapse");

    let mut reader = BufReader::new(File::open(path)?);
    let mut raw_bytes = Vec::new();
    reader.read_to_end(&mut raw_bytes)?;

    let tensors = extract_burn_tensors(&raw_bytes);
    println!("==================================================================");
    println!(
        " EXACT BURN TENSOR EXTRACTOR | Snapped {} exact tensors in {}",
        tensors.len(),
        path
    );
    println!("==================================================================");

    let embeds: Vec<&BurnTensor> = tensors.iter().filter(|t| t.num_floats == 262144).collect();
    let routers: Vec<&BurnTensor> = tensors.iter().filter(|t| t.num_floats == 256).collect();
    let attns: Vec<&BurnTensor> = tensors.iter().filter(|t| t.num_floats == 4096).collect();
    let ffns: Vec<&BurnTensor> = tensors.iter().filter(|t| t.num_floats == 16384).collect();

    println!(
        "  Summary: {} Embeddings (4096x64), {} Attn Matrices (64x64), {} MoE Routers (4x64), {} Expert FFNs (64x256)",
        embeds.len(),
        attns.len(),
        routers.len(),
        ffns.len()
    );

    if embeds.len() >= 2 {
        analyze_embeddings_and_head(embeds[0], embeds[embeds.len() - 1]);
    }

    println!("==================================================================");
    println!(" 2. ATTENTION HEAD & VIRTUAL CIRCUIT ANALYSIS (64x64)");
    println!("==================================================================");
    for (l_idx, chunk) in attns.chunks(4).enumerate() {
        println!("  --- Layer {} Attention ---", l_idx);
        let names = ["W_Q", "W_K", "W_V", "W_O"];
        for (i, &t) in chunk.iter().enumerate() {
            let l2 = l2_norm(&t.floats);
            let svd = svd_top_k(&t.floats, 64, 64, 2);
            let (s1, s2) = (svd[0].0, svd[1].0);
            let rank1 = (s1 * s1) / (l2 * l2) * 100.0;
            println!(
                "    {} (Tensor #{:02} @ {:7}) | L2={:5.2} | s1={:5.2}, s2={:5.2} (s1/s2={:4.2}x) | Rank1={:4.1}%",
                names.get(i).unwrap_or(&"Attn"),
                t.index,
                t.data_offset,
                l2,
                s1,
                s2,
                s1 / s2.max(1e-6),
                rank1
            );
        }
        if chunk.len() == 4 {
            let w_qk = matmul_64x64(&chunk[0].floats, &chunk[1].floats, true);
            let w_ov = matmul_64x64(&chunk[2].floats, &chunk[3].floats, false);
            let qk_svd = svd_top_k(&w_qk, 64, 64, 2);
            let ov_svd = svd_top_k(&w_ov, 64, 64, 2);
            let qk_l2 = l2_norm(&w_qk);
            let ov_l2 = l2_norm(&w_ov);
            println!(
                "    -> Virtual W_QK: L2={:5.2}, s1/s2={:4.2}x, Rank1={:4.1}% | Virtual W_OV: L2={:5.2}, s1/s2={:4.2}x, Rank1={:4.1}%",
                qk_l2,
                qk_svd[0].0 / qk_svd[1].0.max(1e-6),
                (qk_svd[0].0 * qk_svd[0].0) / (qk_l2 * qk_l2) * 100.0,
                ov_l2,
                ov_svd[0].0 / ov_svd[1].0.max(1e-6),
                (ov_svd[0].0 * ov_svd[0].0) / (ov_l2 * ov_l2) * 100.0
            );
        }
    }

    println!("\n==================================================================");
    println!(" 3. MoE ROUTER & EXPERT SPECIALIZATION ANALYSIS (4 Layers)");
    println!("==================================================================");
    for l_idx in 0..routers.len() {
        let r = routers[l_idx];
        println!("  --- Layer {} MoE (Router Tensor #{:02}) ---", l_idx, r.index);
        print!("    Router Gate Row Norms: ");
        for e in 0..NUM_EXPERTS {
            let row = &r.floats[e * EMBED_DIM..(e + 1) * EMBED_DIM];
            print!("E{}:{:.3}  ", e, l2_norm(row));
        }
        println!();

        print!("    Router Gate Cosine Sims: ");
        for i in 0..NUM_EXPERTS {
            for j in (i + 1)..NUM_EXPERTS {
                let ri = &r.floats[i * EMBED_DIM..(i + 1) * EMBED_DIM];
                let rj = &r.floats[j * EMBED_DIM..(j + 1) * EMBED_DIM];
                print!("E{}-E{}:{:+.2} ", i, j, cosine_sim(ri, rj));
            }
        }
        println!();

        let layer_ffns = &ffns[l_idx * 12..((l_idx + 1) * 12).min(ffns.len())];
        if layer_ffns.len() == 12 {
            for e in 0..NUM_EXPERTS {
                let m0 = layer_ffns[e * 3];
                let m1 = layer_ffns[e * 3 + 1];
                let m2 = layer_ffns[e * 3 + 2];
                let s_m0 = svd_top_k(&m0.floats, 64, 256, 2);
                let s_m1 = svd_top_k(&m1.floats, 64, 256, 2);
                let s_m2 = svd_top_k(&m2.floats, 64, 256, 2);

                println!(
                    "    Expert {}: M0(#{:02}) L2={:5.2} s1/s2={:4.2}x | M1(#{:02}) L2={:5.2} s1/s2={:4.2}x | M2(#{:02}) L2={:5.2} s1/s2={:4.2}x",
                    e,
                    m0.index, l2_norm(&m0.floats), s_m0[0].0 / s_m0[1].0.max(1e-6),
                    m1.index, l2_norm(&m1.floats), s_m1[0].0 / s_m1[1].0.max(1e-6),
                    m2.index, l2_norm(&m2.floats), s_m2[0].0 / s_m2[1].0.max(1e-6),
                );
            }

            // Check if experts are clones or distinct in weight space!
            let e0_m1 = &layer_ffns[1].floats;
            let e1_m1 = &layer_ffns[4].floats;
            let e2_m1 = &layer_ffns[7].floats;
            let e3_m1 = &layer_ffns[10].floats;
            println!(
                "    Inter-Expert M1 Cosine Sim: E0-E1:{:+.3}, E0-E2:{:+.3}, E0-E3:{:+.3}, E1-E2:{:+.3}",
                cosine_sim(e0_m1, e1_m1),
                cosine_sim(e0_m1, e2_m1),
                cosine_sim(e0_m1, e3_m1),
                cosine_sim(e1_m1, e2_m1)
            );
        }
    }

    if fix_mode && embeds.len() >= 2 {
        println!("\n==================================================================");
        println!(" 4. SURGICAL WEIGHT-SPACE INTERVENTION (--fix-collapse)");
        println!("==================================================================");
        let mut patched_bytes = raw_bytes.clone();

        for &target in &[embeds[0], embeds[embeds.len() - 1]] {
            let svd = svd_top_k(&target.floats, VOCAB_SIZE, EMBED_DIM, 3);
            let (s1, ref u1, ref v1) = svd[0];
            let s2 = svd[1].0;

            if s1 > 2.0 * s2 {
                let target_s1 = 1.15 * s2;
                let remove_scale = s1 - target_s1;
                println!(
                    "  [+] Patching Tensor #{:02} @ byte {}: Reducing dominant sigma1 from {:.2} -> {:.2} (s2={:.2})",
                    target.index, target.data_offset, s1, target_s1, s2
                );

                let mut new_floats = target.floats.clone();
                for r in 0..VOCAB_SIZE {
                    for c in 0..EMBED_DIM {
                        new_floats[r * EMBED_DIM + c] -= remove_scale * u1[r] * v1[c];
                    }
                }

                for (k, &val) in new_floats.iter().enumerate() {
                    let b = val.to_le_bytes();
                    let off = target.data_offset + k * 4;
                    patched_bytes[off..off + 4].copy_from_slice(&b);
                }
            }
        }

        let out_path = format!("{}.patched", path);
        let mut out_file = File::create(&out_path)?;
        out_file.write_all(&patched_bytes)?;
        println!("  [SUCCESS] Wrote surgically de-collapsed checkpoint to: {}", out_path);
    }

    Ok(())
}

