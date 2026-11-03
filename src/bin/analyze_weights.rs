// ** analysis #2 **

use std::env;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub ff_dim: usize,
    pub num_experts: usize,
    pub num_shared_experts: usize,
    pub n_layers: usize,
    pub attn_heads: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 4096,
            embed_dim: 64,
            ff_dim: 256,
            num_experts: 4,
            num_shared_experts: 0,
            n_layers: 4,
            attn_heads: 4,
        }
    }
}

impl ModelConfig {
    pub fn update_from_json(&mut self, val: &serde_json::Value) {
        let extract = |k: &str| -> Option<usize> {
            val.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .or_else(|| {
                    val.get("config")
                        .and_then(|c| c.get(k))
                        .and_then(|x| x.as_u64())
                        .map(|x| x as usize)
                })
        };
        if let Some(v) = extract("vocab_size") { self.vocab_size = v; }
        if let Some(v) = extract("embed_dim") { self.embed_dim = v; }
        if let Some(v) = extract("ff_dim") { self.ff_dim = v; }
        if let Some(v) = extract("num_experts") { self.num_experts = v; }
        if let Some(v) = extract("num_shared_experts") { self.num_shared_experts = v; }
        if let Some(v) = extract("n_layers") { self.n_layers = v; }
        if let Some(v) = extract("attn_heads") { self.attn_heads = v; }
    }
}

fn try_load_dir_config(dir: &Path, config: &mut ModelConfig) {
    for name in &["metadata.json", "code-config.json"] {
        let path = dir.join(name);
        if path.is_file() {
            if let Ok(content) = std::fs::read_to_string(&path) {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                    config.update_from_json(&val);
                    return;
                }
            }
        }
    }
    // Search 1 level of subdirectories for metadata.json
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                for name in &["metadata.json", "code-config.json"] {
                    let sub = p.join(name);
                    if sub.is_file() {
                        if let Ok(content) = std::fs::read_to_string(&sub) {
                            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                                config.update_from_json(&val);
                                return;
                            }
                        }
                    }
                }
            }
        }
    }
    // Also check parent directory
    if let Some(parent) = dir.parent() {
        for name in &["metadata.json", "code-config.json"] {
            let path = parent.join(name);
            if path.is_file() {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                        config.update_from_json(&val);
                        return;
                    }
                }
            }
        }
    }
}

fn find_model_bin_in_dir(dir: &Path) -> Option<PathBuf> {
    let direct = dir.join("model.bin");
    if direct.is_file() {
        return Some(direct);
    }
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                let sub = p.join("model.bin");
                if sub.is_file() {
                    return Some(sub);
                }
            }
        }
    }
    None
}

fn resolve_model_and_config(args: &[String]) -> std::io::Result<(PathBuf, ModelConfig)> {
    let mut config = ModelConfig::default();
    let mut model_path: Option<PathBuf> = None;
    let mut explicit_config_path: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--config" && i + 1 < args.len() {
            explicit_config_path = Some(PathBuf::from(&args[i + 1]));
            i += 2;
            continue;
        } else if arg == "--embed-dim" && i + 1 < args.len() {
            if let Ok(v) = args[i + 1].parse() { config.embed_dim = v; }
            i += 2;
            continue;
        } else if arg == "--ff-dim" && i + 1 < args.len() {
            if let Ok(v) = args[i + 1].parse() { config.ff_dim = v; }
            i += 2;
            continue;
        } else if (arg == "--experts" || arg == "--num-experts") && i + 1 < args.len() {
            if let Ok(v) = args[i + 1].parse() { config.num_experts = v; }
            i += 2;
            continue;
        } else if (arg == "--layers" || arg == "--n-layers") && i + 1 < args.len() {
            if let Ok(v) = args[i + 1].parse() { config.n_layers = v; }
            i += 2;
            continue;
        } else if arg == "--vocab" && i + 1 < args.len() {
            if let Ok(v) = args[i + 1].parse() { config.vocab_size = v; }
            i += 2;
            continue;
        } else if (arg == "--shared-experts" || arg == "--num-shared-experts") && i + 1 < args.len() {
            if let Ok(v) = args[i + 1].parse() { config.num_shared_experts = v; }
            i += 2;
            continue;
        } else if !arg.starts_with("--") && model_path.is_none() {
            model_path = Some(PathBuf::from(arg));
        }
        i += 1;
    }

    if let Some(ref cp) = explicit_config_path {
        if let Ok(content) = std::fs::read_to_string(cp) {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                config.update_from_json(&val);
            }
        }
    }

    let final_model_path = match model_path {
        Some(p) => {
            if p.is_file() {
                if p.extension().map_or(false, |ext| ext == "json") {
                    if let Ok(content) = std::fs::read_to_string(&p) {
                        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                            config.update_from_json(&val);
                            let out_dir = val.get("out_dir")
                                .and_then(|s| s.as_str())
                                .or_else(|| val.get("config").and_then(|c| c.get("out_dir")).and_then(|s| s.as_str()));
                            if let Some(dir_str) = out_dir {
                                let dir = PathBuf::from(dir_str);
                                find_model_bin_in_dir(&dir).unwrap_or(p)
                            } else {
                                p
                            }
                        } else {
                            p
                        }
                    } else {
                        p
                    }
                } else {
                    if explicit_config_path.is_none() {
                        if let Some(parent) = p.parent() {
                            try_load_dir_config(parent, &mut config);
                        }
                    }
                    p
                }
            } else if p.is_dir() {
                if explicit_config_path.is_none() {
                    try_load_dir_config(&p, &mut config);
                }
                find_model_bin_in_dir(&p).unwrap_or_else(|| p.join("model.bin"))
            } else {
                p
            }
        }
        None => {
            let default_path = PathBuf::from("model.bin");
            if explicit_config_path.is_none() {
                try_load_dir_config(&PathBuf::from("."), &mut config);
            }
            default_path
        }
    };

    Ok((final_model_path, config))
}

#[derive(Clone)]
struct BurnTensor {
    index: usize,
    data_offset: usize, // Exact byte offset where the tensor's f32 slice starts
    num_floats: usize,
    floats: Vec<f32>,
}

/// Extracts exact Burn tensors by combining the contiguous-float scanner
/// with Burn's little-endian byte-length prefix verification & tail-alignment snapping.
fn extract_burn_tensors(bytes: &[u8], config: &ModelConfig) -> Vec<BurnTensor> {
    let mut tensors = Vec::new();
    let mut pos = 0;

    let embed_size = config.vocab_size * config.embed_dim;
    let ffn_size = config.embed_dim * config.ff_dim;
    let attn_size = config.embed_dim * config.embed_dim;
    let router_size = config.num_experts * config.embed_dim;
    let norm_size = config.embed_dim;

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

            // Snap to exact architecture tensor sizes
            let target_floats = if (embed_size.saturating_sub(32)..=embed_size + 32).contains(&count) {
                Some(embed_size)
            } else if (ffn_size.saturating_sub(32)..=ffn_size + 32).contains(&count) {
                Some(ffn_size)
            } else if (attn_size.saturating_sub(32)..=attn_size + 32).contains(&count) {
                Some(attn_size)
            } else if (router_size.saturating_sub(32)..=router_size + 32).contains(&count) {
                Some(router_size)
            } else if (norm_size.saturating_sub(4)..=norm_size + 4).contains(&count) {
                Some(norm_size)
            } else {
                None
            };

            if let Some(exact_len) = target_floats {
                let exact_bytes = exact_len * 4;
                let mut best_start = if count >= exact_len {
                    end_byte - exact_bytes
                } else {
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

fn matmul_square(a: &[f32], b: &[f32], dim: usize, transpose_a: bool) -> Vec<f32> {
    let mut out = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut sum = 0.0f32;
            for k in 0..dim {
                let a_val = if transpose_a { a[k * dim + i] } else { a[i * dim + k] };
                sum += a_val * b[k * dim + j];
            }
            out[i * dim + j] = sum;
        }
    }
    out
}

fn analyze_embeddings_and_head(t_in: &BurnTensor, t_out: &BurnTensor, config: &ModelConfig) {
    println!("\n==================================================================");
    println!(" 1. INPUT EMBEDDING vs. OUTPUT LM_HEAD FORENSICS ({} x {})", config.vocab_size, config.embed_dim);
    println!("==================================================================");

    for (label, t) in [
        ("Input Embedding (Tensor #00)", t_in),
        ("Output LM_Head  (Last token_head)", t_out),
    ] {
        let l2 = l2_norm(&t.floats);
        let svd = svd_top_k(&t.floats, config.vocab_size, config.embed_dim, 5);
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
        let mut token_proj: Vec<(usize, f32, f32)> = (0..config.vocab_size)
            .map(|tok_id| {
                let row = &t.floats[tok_id * config.embed_dim..(tok_id + 1) * config.embed_dim];
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
    let fix_mode = args.iter().any(|a| a == "--fix-collapse");

    let (path_buf, config) = resolve_model_and_config(&args)?;
    let path = path_buf.to_str().unwrap_or("model.bin");

    println!("==================================================================");
    println!(" MODEL WEIGHT ANALYZER");
    println!("   Model File:         {}", path);
    println!("   Vocab Size:         {}", config.vocab_size);
    println!("   Embedding Dim:      {}", config.embed_dim);
    println!("   FFN Dim:            {}", config.ff_dim);
    println!("   Num Experts:        {} routed{}", config.num_experts, if config.num_shared_experts > 0 { format!(" (+ {} shared)", config.num_shared_experts) } else { "".to_string() });
    println!("   Num Layers:         {}", config.n_layers);
    println!("   Attention Heads:    {}", config.attn_heads);
    println!("==================================================================");

    let mut reader = BufReader::new(File::open(path)?);
    let mut raw_bytes = Vec::new();
    reader.read_to_end(&mut raw_bytes)?;

    let tensors = extract_burn_tensors(&raw_bytes, &config);
    println!(
        " EXACT BURN TENSOR EXTRACTOR | Snapped {} exact tensors in {}",
        tensors.len(),
        path
    );
    println!("==================================================================");

    let embed_size = config.vocab_size * config.embed_dim;
    let router_size = config.num_experts * config.embed_dim;
    let attn_size = config.embed_dim * config.embed_dim;
    let ffn_size = config.embed_dim * config.ff_dim;

    let embeds: Vec<&BurnTensor> = tensors.iter().filter(|t| t.num_floats == embed_size).collect();
    let routers: Vec<&BurnTensor> = tensors.iter().filter(|t| t.num_floats == router_size).collect();
    let attns: Vec<&BurnTensor> = tensors.iter().filter(|t| t.num_floats == attn_size).collect();
    let ffns: Vec<&BurnTensor> = tensors.iter().filter(|t| t.num_floats == ffn_size).collect();

    println!(
        "  Summary: {} Embeddings ({}x{}), {} Attn Matrices ({}x{}), {} MoE Routers ({}x{}), {} Expert FFNs ({}x{})",
        embeds.len(), config.vocab_size, config.embed_dim,
        attns.len(), config.embed_dim, config.embed_dim,
        routers.len(), config.num_experts, config.embed_dim,
        ffns.len(), config.embed_dim, config.ff_dim
    );

    if embeds.len() >= 2 {
        analyze_embeddings_and_head(embeds[0], embeds[embeds.len() - 1], &config);
    } else if embeds.len() == 1 {
        analyze_embeddings_and_head(embeds[0], embeds[0], &config);
    }

    println!("==================================================================");
    println!(" 2. ATTENTION HEAD & VIRTUAL CIRCUIT ANALYSIS ({}x{})", config.embed_dim, config.embed_dim);
    println!("==================================================================");
    for (l_idx, chunk) in attns.chunks(4).enumerate() {
        println!("  --- Layer {} Attention ---", l_idx);
        let names = ["W_Q", "W_K", "W_V", "W_O"];
        for (i, &t) in chunk.iter().enumerate() {
            let l2 = l2_norm(&t.floats);
            let svd = svd_top_k(&t.floats, config.embed_dim, config.embed_dim, 2);
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
            let w_qk = matmul_square(&chunk[0].floats, &chunk[1].floats, config.embed_dim, true);
            let w_ov = matmul_square(&chunk[2].floats, &chunk[3].floats, config.embed_dim, false);
            let qk_svd = svd_top_k(&w_qk, config.embed_dim, config.embed_dim, 2);
            let ov_svd = svd_top_k(&w_ov, config.embed_dim, config.embed_dim, 2);
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
    println!(" 3. MoE ROUTER & EXPERT SPECIALIZATION ANALYSIS ({} Routers, {} Experts)", routers.len(), config.num_experts);
    println!("==================================================================");
    let routed_ffns_per_layer = config.num_experts * 3;
    let shared_base = routers.len() * routed_ffns_per_layer;
    let shared_ffns_per_layer = config.num_shared_experts * 3;

    for l_idx in 0..routers.len() {
        let r = routers[l_idx];
        println!("  --- Layer {} MoE (Router Tensor #{:02}) ---", l_idx, r.index);
        print!("    Router Gate Row Norms: ");
        for e in 0..config.num_experts {
            if (e + 1) * config.embed_dim <= r.floats.len() {
                let row = &r.floats[e * config.embed_dim..(e + 1) * config.embed_dim];
                print!("E{}:{:.3}  ", e, l2_norm(row));
            }
        }
        println!();

        print!("    Router Gate Cosine Sims: ");
        for i in 0..config.num_experts.min(8) {
            for j in (i + 1)..config.num_experts.min(8) {
                if (j + 1) * config.embed_dim <= r.floats.len() {
                    let ri = &r.floats[i * config.embed_dim..(i + 1) * config.embed_dim];
                    let rj = &r.floats[j * config.embed_dim..(j + 1) * config.embed_dim];
                    print!("E{}-E{}:{:+.2} ", i, j, cosine_sim(ri, rj));
                }
            }
        }
        println!();

        let routed_start = l_idx * routed_ffns_per_layer;
        let routed_end = (routed_start + routed_ffns_per_layer).min(ffns.len());
        if routed_start < ffns.len() {
            let layer_ffns = &ffns[routed_start..routed_end];
            let num_exp = (layer_ffns.len() / 3).min(config.num_experts);

            for e in 0..num_exp {
                let m0 = layer_ffns[e * 3];
                let m1 = layer_ffns[e * 3 + 1];
                let m2 = layer_ffns[e * 3 + 2];
                let s_m0 = svd_top_k(&m0.floats, config.embed_dim, config.ff_dim, 2);
                let s_m1 = svd_top_k(&m1.floats, config.ff_dim, config.embed_dim, 2);
                let s_m2 = svd_top_k(&m2.floats, config.embed_dim, config.ff_dim, 2);

                println!(
                    "    Expert {}: M0(#{:02}) L2={:5.2} s1/s2={:4.2}x | M1(#{:02}) L2={:5.2} s1/s2={:4.2}x | M2(#{:02}) L2={:5.2} s1/s2={:4.2}x",
                    e,
                    m0.index, l2_norm(&m0.floats), s_m0[0].0 / s_m0[1].0.max(1e-6),
                    m1.index, l2_norm(&m1.floats), s_m1[0].0 / s_m1[1].0.max(1e-6),
                    m2.index, l2_norm(&m2.floats), s_m2[0].0 / s_m2[1].0.max(1e-6),
                );
            }

            // Check if experts are clones or distinct in weight space
            if num_exp >= 2 {
                print!("    Inter-Expert M1 Cosine Sim: ");
                let compare_limit = num_exp.min(4);
                for i in 0..compare_limit {
                    for j in (i + 1)..compare_limit {
                        let mi = &layer_ffns[i * 3 + 1].floats;
                        let mj = &layer_ffns[j * 3 + 1].floats;
                        print!("E{}-E{}:{:+.3} ", i, j, cosine_sim(mi, mj));
                    }
                }
                println!();
            }
        }

        // Shared expert check
        if config.num_shared_experts > 0 {
            let shared_start = shared_base + l_idx * shared_ffns_per_layer;
            let shared_end = shared_start + shared_ffns_per_layer;
            if shared_start < ffns.len() && shared_end <= ffns.len() {
                let sm0 = ffns[shared_start];
                let sm1 = ffns[shared_start + 1];
                let sm2 = ffns[shared_start + 2];
                let s_sm0 = svd_top_k(&sm0.floats, config.embed_dim, config.ff_dim, 2);
                let s_sm1 = svd_top_k(&sm1.floats, config.ff_dim, config.embed_dim, 2);
                let s_sm2 = svd_top_k(&sm2.floats, config.embed_dim, config.ff_dim, 2);
                println!(
                    "    Shared Expert: M0(#{:02}) L2={:5.2} s1/s2={:4.2}x | M1(#{:02}) L2={:5.2} s1/s2={:4.2}x | M2(#{:02}) L2={:5.2} s1/s2={:4.2}x",
                    sm0.index, l2_norm(&sm0.floats), s_sm0[0].0 / s_sm0[1].0.max(1e-6),
                    sm1.index, l2_norm(&sm1.floats), s_sm1[0].0 / s_sm1[1].0.max(1e-6),
                    sm2.index, l2_norm(&sm2.floats), s_sm2[0].0 / s_sm2[1].0.max(1e-6),
                );
            }
        }
    }

    if fix_mode && embeds.len() >= 2 {
        println!("\n==================================================================");
        println!(" 4. SURGICAL WEIGHT-SPACE INTERVENTION (--fix-collapse)");
        println!("==================================================================");
        let mut patched_bytes = raw_bytes.clone();

        for &target in &[embeds[0], embeds[embeds.len() - 1]] {
            let svd = svd_top_k(&target.floats, config.vocab_size, config.embed_dim, 3);
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
                for r in 0..config.vocab_size {
                    for c in 0..config.embed_dim {
                        new_floats[r * config.embed_dim + c] -= remove_scale * u1[r] * v1[c];
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
