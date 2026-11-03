use std::env;
use std::fs::File;
use std::io::{BufReader, Read};
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
    // Search 1 level of subdirectories
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

fn is_plausible_weight(val: f32) -> bool {
    val.is_finite() && val.abs() < 50.0
}

/// Inspects the first bytes of the file for magic headers, strings, or bincode struct fields
fn inspect_file_header(bytes: &[u8]) {
    println!("=== 1. FILE HEADER FORENSICS (First 128 bytes) ===");
    let preview_len = bytes.len().min(128);
    
    // Print printable ASCII characters (useful if safetensors, JSON, or String names exist)
    print!("  ASCII preview: \"");
    for &b in &bytes[0..preview_len] {
        if (32..=126).contains(&b) {
            print!("{}", b as char);
        } else {
            print!(".");
        }
    }
    println!("\"");

    // Check first few u64 / u32 values at offsets 0..32
    print!("  First u64s (little-endian): ");
    for offset in (0..32.min(bytes.len().saturating_sub(8))).step_by(8) {
        let v = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        print!("[+{}]={}", offset, v);
        if v < 10_000_000 {
            print!(" (valid len!)");
        }
        print!("  ");
    }
    println!();
}

/// Checks which byte alignment (0, 1, 2, or 3) contains the valid f32 weights!
fn scan_byte_alignments(bytes: &[u8]) -> usize {
    println!("\n=== 2. BYTE-ALIGNMENT DETECTOR (Why Remainder = 3) ===");
    let mut best_align = 0;
    let mut best_valid_count = 0;

    for align in 0..4 {
        let slice = &bytes[align..];
        let total_chunks = slice.len() / 4;
        let mut valid_weights = 0usize;
        let mut zero_weights = 0usize;
        let mut extreme_or_nan = 0usize;

        for chunk in slice.chunks_exact(4) {
            let v = f32::from_le_bytes(chunk.try_into().unwrap());
            if v == 0.0 {
                zero_weights += 1;
            } else if is_plausible_weight(v) && v.abs() > 1e-7 {
                valid_weights += 1;
            } else {
                extreme_or_nan += 1;
            }
        }

        let valid_pct = (valid_weights + zero_weights) as f64 * 100.0 / total_chunks as f64;
        println!(
            "  Alignment offset +{}: {:8} valid weights ({:5.1}%), {:7} zeros, {:8} corrupt/header floats",
            align, valid_weights, valid_pct, zero_weights, extreme_or_nan
        );

        if valid_weights > best_valid_count {
            best_valid_count = valid_weights;
            best_align = align;
        }
    }
    println!("  --> Dominant f32 byte alignment is +{}!", best_align);
    best_align
}

/// Scans the file for contiguous runs of valid f32 weights across ALL 4 alignments
/// (in case variable-length strings or bools shift alignment mid-file!)
struct ContiguousTensor {
    byte_offset: usize,
    alignment: usize,
    floats: Vec<f32>,
}

fn extract_all_valid_tensors(bytes: &[u8]) -> Vec<ContiguousTensor> {
    let mut tensors = Vec::new();
    let min_tensor_floats = 64; // Ignore tiny accidental matches; look for >= 64 floats

    let mut pos = 0;
    while pos + 4 <= bytes.len() {
        // Check if a valid run of floats starts at `pos`
        let f0 = f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        if is_plausible_weight(f0) && (f0 == 0.0 || f0.abs() > 1e-6) {
            let start = pos;
            let mut run = Vec::new();
            while pos + 4 <= bytes.len() {
                let v = f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
                // Allow normal weights in (-30.0, 30.0), stop when hitting a u64 header or NaN/Inf
                if v.is_finite() && v.abs() < 30.0 && (v == 0.0 || v.abs() > 1e-9) {
                    run.push(v);
                    pos += 4;
                } else {
                    break;
                }
            }

            if run.len() >= min_tensor_floats {
                // Check if the 8 bytes right before `start` was a bincode u64 length prefix!
                let mut actual_start = start;
                if start >= 8 {
                    let prev_u64 = u64::from_le_bytes(bytes[start - 8..start].try_into().unwrap()) as usize;
                    if prev_u64 == run.len() || prev_u64 + 1 == run.len() {
                        actual_start = start;
                    }
                }
                tensors.push(ContiguousTensor {
                    byte_offset: actual_start,
                    alignment: actual_start % 4,
                    floats: run,
                });
                continue;
            }
        }
        pos += 1;
    }
    tensors
}

/// Fast power iteration for top 2 singular values
fn top2_singular_values(w: &[f32], rows: usize, cols: usize) -> (f32, f32, f32) {
    let sq_sum: f64 = w.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let l2 = sq_sum.sqrt() as f32;
    if sq_sum < 1e-12 {
        return (0.0, 0.0, 0.0);
    }

    let (s1, u1, v1) = power_iter(w, rows, cols, None, 30);
    let (s2, _, _) = power_iter(w, rows, cols, Some((s1, &u1, &v1)), 30);
    (l2, s1, s2)
}

fn power_iter(
    w: &[f32],
    rows: usize,
    cols: usize,
    deflate: Option<(f32, &[f32], &[f32])>,
    iters: usize,
) -> (f32, Vec<f32>, Vec<f32>) {
    let mut v: Vec<f32> = (0..cols).map(|j| ((j as f32 * 1.618).sin()) + 0.1).collect();
    norm_vec(&mut v);
    let mut u = vec![0.0f32; rows];

    for _ in 0..iters {
        for r in 0..rows {
            let mut dot = 0.0f32;
            let row = &w[r * cols..(r + 1) * cols];
            for c in 0..cols {
                dot += row[c] * v[c];
            }
            if let Some((s1, u1, v1)) = deflate {
                let vd: f32 = v1.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                dot -= s1 * u1[r] * vd;
            }
            u[r] = dot;
        }
        norm_vec(&mut u);

        for c in 0..cols {
            let mut dot = 0.0f32;
            for r in 0..rows {
                dot += w[r * cols + c] * u[r];
            }
            if let Some((s1, u1, v1)) = deflate {
                let ud: f32 = u1.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
                dot -= s1 * v1[c] * ud;
            }
            v[c] = dot;
        }
        norm_vec(&mut v);
    }

    let mut sigma = 0.0f32;
    for r in 0..rows {
        let mut dot = 0.0f32;
        for c in 0..cols {
            dot += w[r * cols + c] * v[c];
        }
        if let Some((s1, u1, v1)) = deflate {
            let vd: f32 = v1.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            dot -= s1 * u1[r] * vd;
        }
        sigma += u[r] * dot;
    }
    (sigma.abs(), u, v)
}

fn norm_vec(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-12 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

fn guess_shape(len: usize, config: &ModelConfig) -> String {
    if len == config.vocab_size * config.embed_dim {
        return format!("{}x{} (Embedding/LM_Head)", config.vocab_size, config.embed_dim);
    }
    if len == config.embed_dim * config.embed_dim {
        return format!("{}x{} (Attn Proj)", config.embed_dim, config.embed_dim);
    }
    if len == 3 * config.embed_dim * config.embed_dim {
        return format!("{}x{} (Fused QKV)", 3 * config.embed_dim, config.embed_dim);
    }
    if len == config.embed_dim * config.ff_dim {
        return format!("{}x{} (Expert FFN)", config.embed_dim, config.ff_dim);
    }
    if len == config.num_experts * config.embed_dim {
        return format!("{}x{} (MoE Router)", config.num_experts, config.embed_dim);
    }
    if len == config.embed_dim {
        return format!("{} (LayerNorm/Bias)", config.embed_dim);
    }
    if len == config.ff_dim {
        return format!("{} (FFN Bias)", config.ff_dim);
    }
    if len % config.embed_dim == 0 {
        return format!("{}x{}", len / config.embed_dim, config.embed_dim);
    }
    format!("{} floats", len)
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let (path_buf, config) = resolve_model_and_config(&args)?;
    let path = path_buf.to_str().unwrap_or("model.bin");

    println!("==================================================================");
    println!(" PASS 1: MODEL WEIGHT FORENSICS & RAW SCANNER");
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

    inspect_file_header(&raw_bytes);
    let _best_align = scan_byte_alignments(&raw_bytes);

    println!("\n=== 3. AUTOMATIC TENSOR EXTRACTION & SPECTRAL HEALTH ===");
    let tensors = extract_all_valid_tensors(&raw_bytes);
    let total_extracted_floats: usize = tensors.iter().map(|t| t.floats.len()).sum();
    println!(
        "  Discovered {} contiguous weight blocks totaling {} floats ({:.1}% of file bytes)",
        tensors.len(),
        total_extracted_floats,
        (total_extracted_floats * 4) as f64 * 100.0 / raw_bytes.len() as f64
    );

    let mut small_vec_count = 0;
    let mut row_norm_count = 0;

    for (i, t) in tensors.iter().enumerate() {
        let len = t.floats.len();
        if len == config.embed_dim {
            row_norm_count += 1;
            continue;
        }
        if len < config.embed_dim * 4 {
            small_vec_count += 1;
            continue;
        }

        let prefix_u64 = if t.byte_offset >= 8 {
            u64::from_le_bytes(raw_bytes[t.byte_offset - 8..t.byte_offset].try_into().unwrap())
        } else {
            0
        };

        // Determine matrix dimensions for SVD
        let (rows, cols) = if len % config.embed_dim == 0 && len >= config.embed_dim * 4 {
            (len / config.embed_dim, config.embed_dim)
        } else {
            (1, len)
        };

        let (l2, s1, s2) = if rows > 1 {
            top2_singular_values(&t.floats, rows, cols)
        } else {
            let l2 = t.floats.iter().map(|x| x * x).sum::<f32>().sqrt();
            (l2, l2, 0.0)
        };

        let rank1_pct = if l2 > 1e-6 {
            (s1 * s1) / (l2 * l2) * 100.0
        } else {
            0.0
        };
        let s1_s2_ratio = if s2 > 1e-6 { s1 / s2 } else { 0.0 };

        let status = if rows > 1 && s1_s2_ratio < 1.08 && rank1_pct < 8.0 {
            "[UNLEARNED / RANDOM SPECTRUM]"
        } else if rank1_pct > 25.0 {
            "[COLLAPSED / ANISOTROPIC]"
        } else {
            "[LEARNED STRUCTURE]"
        };

        println!(
            "  Tensor #{:03} | byte {:7} (align +{}) | prev_u64={:7} | {:24} | L2={:6.2} | s1={:5.2}, s2={:5.2} (s1/s2={:4.2}x) | Rank1={:4.1}% {}",
            i,
            t.byte_offset,
            t.alignment,
            prefix_u64,
            guess_shape(len, &config),
            l2,
            s1,
            s2,
            s1_s2_ratio,
            rank1_pct,
            status
        );
    }

    if row_norm_count > 0 || small_vec_count > 0 {
        println!(
            "\n  [+] Also found {} standalone {}-float rows/norms and {} small vectors (<{} floats).",
            row_norm_count, config.embed_dim, small_vec_count, config.embed_dim * 4
        );
    }

    Ok(())
}
