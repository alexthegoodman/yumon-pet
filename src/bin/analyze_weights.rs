// use std::env;
// use std::fs::File;
// use std::io::{BufReader, Read};

// const VOCAB_SIZE: usize = 4096;
// const EMBED_DIM: usize = 64;
// const FF_DIM: usize = 256;
// const NUM_EXPERTS: usize = 4;

// fn is_plausible_weight(val: f32) -> bool {
//     val.is_finite() && val.abs() < 50.0
// }

// /// Inspects the first bytes of the file for magic headers, strings, or bincode struct fields
// fn inspect_file_header(bytes: &[u8]) {
//     println!("=== 1. FILE HEADER FORENSICS (First 128 bytes) ===");
//     let preview_len = bytes.len().min(128);
    
//     // Print printable ASCII characters (useful if safetensors, JSON, or String names exist)
//     print!("  ASCII preview: \"");
//     for &b in &bytes[0..preview_len] {
//         if (32..=126).contains(&b) {
//             print!("{}", b as char);
//         } else {
//             print!(".");
//         }
//     }
//     println!("\"");

//     // Check first few u64 / u32 values at offsets 0..32
//     print!("  First u64s (little-endian): ");
//     for offset in (0..32.min(bytes.len().saturating_sub(8))).step_by(8) {
//         let v = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
//         print!("[+{}]={}", offset, v);
//         if v < 10_000_000 {
//             print!(" (valid len!)");
//         }
//         print!("  ");
//     }
//     println!();
// }

// /// Checks which byte alignment (0, 1, 2, or 3) contains the valid f32 weights!
// fn scan_byte_alignments(bytes: &[u8]) -> usize {
//     println!("\n=== 2. BYTE-ALIGNMENT DETECTOR (Why Remainder = 3) ===");
//     let mut best_align = 0;
//     let mut best_valid_count = 0;

//     for align in 0..4 {
//         let slice = &bytes[align..];
//         let total_chunks = slice.len() / 4;
//         let mut valid_weights = 0usize;
//         let mut zero_weights = 0usize;
//         let mut extreme_or_nan = 0usize;

//         for chunk in slice.chunks_exact(4) {
//             let v = f32::from_le_bytes(chunk.try_into().unwrap());
//             if v == 0.0 {
//                 zero_weights += 1;
//             } else if is_plausible_weight(v) && v.abs() > 1e-7 {
//                 valid_weights += 1;
//             } else {
//                 extreme_or_nan += 1;
//             }
//         }

//         let valid_pct = (valid_weights + zero_weights) as f64 * 100.0 / total_chunks as f64;
//         println!(
//             "  Alignment offset +{}: {:8} valid weights ({:5.1}%), {:7} zeros, {:8} corrupt/header floats",
//             align, valid_weights, valid_pct, zero_weights, extreme_or_nan
//         );

//         if valid_weights > best_valid_count {
//             best_valid_count = valid_weights;
//             best_align = align;
//         }
//     }
//     println!("  --> Dominant f32 byte alignment is +{}!", best_align);
//     best_align
// }

// /// Scans the file for contiguous runs of valid f32 weights across ALL 4 alignments
// /// (in case variable-length strings or bools shift alignment mid-file!)
// struct ContiguousTensor {
//     byte_offset: usize,
//     alignment: usize,
//     floats: Vec<f32>,
// }

// fn extract_all_valid_tensors(bytes: &[u8]) -> Vec<ContiguousTensor> {
//     let mut tensors = Vec::new();
//     let min_tensor_floats = 64; // Ignore tiny accidental matches; look for >= 64 floats

//     let mut pos = 0;
//     while pos + 4 <= bytes.len() {
//         // Check if a valid run of floats starts at `pos`
//         let f0 = f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
//         if is_plausible_weight(f0) && (f0 == 0.0 || f0.abs() > 1e-6) {
//             let start = pos;
//             let mut run = Vec::new();
//             while pos + 4 <= bytes.len() {
//                 let v = f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
//                 // Allow normal weights in (-30.0, 30.0), stop when hitting a u64 header or NaN/Inf
//                 if v.is_finite() && v.abs() < 30.0 && (v == 0.0 || v.abs() > 1e-9) {
//                     run.push(v);
//                     pos += 4;
//                 } else {
//                     break;
//                 }
//             }

//             if run.len() >= min_tensor_floats {
//                 // Check if the 8 bytes right before `start` was a bincode u64 length prefix!
//                 let mut actual_start = start;
//                 if start >= 8 {
//                     let prev_u64 = u64::from_le_bytes(bytes[start - 8..start].try_into().unwrap()) as usize;
//                     if prev_u64 == run.len() || prev_u64 + 1 == run.len() {
//                         actual_start = start;
//                     }
//                 }
//                 tensors.push(ContiguousTensor {
//                     byte_offset: actual_start,
//                     alignment: actual_start % 4,
//                     floats: run,
//                 });
//                 continue;
//             }
//         }
//         pos += 1;
//     }
//     tensors
// }

// /// Fast power iteration for top 2 singular values
// fn top2_singular_values(w: &[f32], rows: usize, cols: usize) -> (f32, f32, f32) {
//     let sq_sum: f64 = w.iter().map(|&x| (x as f64) * (x as f64)).sum();
//     let l2 = sq_sum.sqrt() as f32;
//     if sq_sum < 1e-12 {
//         return (0.0, 0.0, 0.0);
//     }

//     let (s1, u1, v1) = power_iter(w, rows, cols, None, 30);
//     let (s2, _, _) = power_iter(w, rows, cols, Some((s1, &u1, &v1)), 30);
//     (l2, s1, s2)
// }

// fn power_iter(
//     w: &[f32],
//     rows: usize,
//     cols: usize,
//     deflate: Option<(f32, &[f32], &[f32])>,
//     iters: usize,
// ) -> (f32, Vec<f32>, Vec<f32>) {
//     let mut v: Vec<f32> = (0..cols).map(|j| ((j as f32 * 1.618).sin()) + 0.1).collect();
//     norm_vec(&mut v);
//     let mut u = vec![0.0f32; rows];

//     for _ in 0..iters {
//         for r in 0..rows {
//             let mut dot = 0.0f32;
//             let row = &w[r * cols..(r + 1) * cols];
//             for c in 0..cols {
//                 dot += row[c] * v[c];
//             }
//             if let Some((s1, u1, v1)) = deflate {
//                 let vd: f32 = v1.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
//                 dot -= s1 * u1[r] * vd;
//             }
//             u[r] = dot;
//         }
//         norm_vec(&mut u);

//         for c in 0..cols {
//             let mut dot = 0.0f32;
//             for r in 0..rows {
//                 dot += w[r * cols + c] * u[r];
//             }
//             if let Some((s1, u1, v1)) = deflate {
//                 let ud: f32 = u1.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
//                 dot -= s1 * v1[c] * ud;
//             }
//             v[c] = dot;
//         }
//         norm_vec(&mut v);
//     }

//     let mut sigma = 0.0f32;
//     for r in 0..rows {
//         let mut dot = 0.0f32;
//         for c in 0..cols {
//             dot += w[r * cols + c] * v[c];
//         }
//         if let Some((s1, u1, v1)) = deflate {
//             let vd: f32 = v1.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
//             dot -= s1 * u1[r] * vd;
//         }
//         sigma += u[r] * dot;
//     }
//     (sigma.abs(), u, v)
// }

// fn norm_vec(v: &mut [f32]) {
//     let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
//     if n > 1e-12 {
//         for x in v.iter_mut() {
//             *x /= n;
//         }
//     }
// }

// fn guess_shape(len: usize) -> String {
//     if len == VOCAB_SIZE * EMBED_DIM {
//         return format!("{}x{} (Embedding/LM_Head)", VOCAB_SIZE, EMBED_DIM);
//     }
//     if len == EMBED_DIM * EMBED_DIM {
//         return "64x64 (Attn Proj)".to_string();
//     }
//     if len == 3 * EMBED_DIM * EMBED_DIM {
//         return "192x64 (Fused QKV)".to_string();
//     }
//     if len == EMBED_DIM * FF_DIM {
//         return "64x256 (Expert FFN)".to_string();
//     }
//     if len == NUM_EXPERTS * EMBED_DIM {
//         return "4x64 (MoE Router)".to_string();
//     }
//     if len == EMBED_DIM {
//         return "64 (LayerNorm/Bias)".to_string();
//     }
//     if len == FF_DIM {
//         return "256 (FFN Bias)".to_string();
//     }
//     if len % EMBED_DIM == 0 {
//         return format!("{}x{}", len / EMBED_DIM, EMBED_DIM);
//     }
//     format!("{} floats", len)
// }

// fn main() -> std::io::Result<()> {
//     let args: Vec<String> = env::args().collect();
//     let path = args.get(1).map(|s| s.as_str()).unwrap_or("model.bin");

//     let mut reader = BufReader::new(File::open(path)?);
//     let mut raw_bytes = Vec::new();
//     reader.read_to_end(&mut raw_bytes)?;

//     inspect_file_header(&raw_bytes);
//     let _best_align = scan_byte_alignments(&raw_bytes);

//     println!("\n=== 3. AUTOMATIC TENSOR EXTRACTION & SPECTRAL HEALTH ===");
//     let tensors = extract_all_valid_tensors(&raw_bytes);
//     let total_extracted_floats: usize = tensors.iter().map(|t| t.floats.len()).sum();
//     println!(
//         "  Discovered {} contiguous weight blocks totaling {} floats ({:.1}% of file bytes)",
//         tensors.len(),
//         total_extracted_floats,
//         (total_extracted_floats * 4) as f64 * 100.0 / raw_bytes.len() as f64
//     );

//     // Group or display major tensors (>= 256 floats) and small vectors summary
//     let mut small_vec_count = 0;
//     let mut row_64_count = 0;

//     for (i, t) in tensors.iter().enumerate() {
//         let len = t.floats.len();
//         if len == 64 {
//             row_64_count += 1;
//             continue;
//         }
//         if len < 256 {
//             small_vec_count += 1;
//             continue;
//         }

//         // Check if there is a u64 right before `t.byte_offset`
//         let prefix_u64 = if t.byte_offset >= 8 {
//             u64::from_le_bytes(raw_bytes[t.byte_offset - 8..t.byte_offset].try_into().unwrap())
//         } else {
//             0
//         };

//         // Determine matrix dimensions for SVD
//         let (rows, cols) = if len % EMBED_DIM == 0 && len >= EMBED_DIM * 4 {
//             (len / EMBED_DIM, EMBED_DIM)
//         } else {
//             (1, len)
//         };

//         let (l2, s1, s2) = if rows > 1 {
//             top2_singular_values(&t.floats, rows, cols)
//         } else {
//             let l2 = t.floats.iter().map(|x| x * x).sum::<f32>().sqrt();
//             (l2, l2, 0.0)
//         };

//         let rank1_pct = if l2 > 1e-6 {
//             (s1 * s1) / (l2 * l2) * 100.0
//         } else {
//             0.0
//         };
//         let s1_s2_ratio = if s2 > 1e-6 { s1 / s2 } else { 0.0 };

//         // Diagnose whether the matrix looks like untrained random noise (s1/s2 ~ 1.02) vs learned structure
//         let status = if rows > 1 && s1_s2_ratio < 1.08 && rank1_pct < 8.0 {
//             "[UNLEARNED / RANDOM SPECTRUM]"
//         } else if rank1_pct > 25.0 {
//             "[COLLAPSED / ANISOTROPIC]"
//         } else {
//             "[LEARNED STRUCTURE]"
//         };

//         println!(
//             "  Tensor #{:03} | byte {:7} (align +{}) | prev_u64={:7} | {:24} | L2={:6.2} | s1={:5.2}, s2={:5.2} (s1/s2={:4.2}x) | Rank1={:4.1}% {}",
//             i,
//             t.byte_offset,
//             t.alignment,
//             prefix_u64,
//             guess_shape(len),
//             l2,
//             s1,
//             s2,
//             s1_s2_ratio,
//             rank1_pct,
//             status
//         );
//     }

//     if row_64_count > 0 || small_vec_count > 0 {
//         println!(
//             "\n  [+] Also found {} standalone 64-float rows and {} small vectors (<256 floats).",
//             row_64_count, small_vec_count
//         );
//         if row_64_count > 500 {
//             println!(
//                 "      NOTE: Having {} separate 64-float rows means your matrices are serialized as `Vec<Vec<f32>>` (nested Vecs with 8-byte length headers per row)!",
//                 row_64_count
//             );
//         }
//     }

//     Ok(())
// }

// analysis #2

// use std::env;
// use std::fs::File;
// use std::io::{BufReader, Read, Write};

// const VOCAB_SIZE: usize = 4096;
// const EMBED_DIM: usize = 64;
// const FF_DIM: usize = 256;
// const NUM_EXPERTS: usize = 4;

// #[derive(Clone)]
// struct BurnTensor {
//     index: usize,
//     data_offset: usize, // Exact byte offset where the tensor's f32 slice starts
//     num_floats: usize,
//     floats: Vec<f32>,
// }

// /// Extracts exact Burn tensors by combining the proven contiguous-float scanner
// /// with Burn's little-endian byte-length prefix verification & tail-alignment snapping.
// fn extract_burn_tensors(bytes: &[u8]) -> Vec<BurnTensor> {
//     let mut tensors = Vec::new();
//     let mut pos = 0;

//     while pos + 4 <= bytes.len() {
//         let f0 = f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
//         if f0.is_finite() && f0.abs() < 30.0 && (f0 == 0.0 || f0.abs() > 1e-6) {
//             let start = pos;
//             let mut count = 0usize;
//             while pos + 4 <= bytes.len() {
//                 let v = f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
//                 if v.is_finite() && v.abs() < 30.0 && (v == 0.0 || v.abs() > 1e-9) {
//                     count += 1;
//                     pos += 4;
//                 } else {
//                     break;
//                 }
//             }

//             let end_byte = start + count * 4;

//             // Snap to exact architecture tensor sizes (262144, 16384, 4096, 256)
//             // If `count` is off by 1-3 floats (e.g. 4097 or 16383), check where the exact tensor boundary is:
//             let target_floats = if (262140..=262150).contains(&count) {
//                 Some(262144usize)
//             } else if (16380..=16390).contains(&count) {
//                 Some(16384usize)
//             } else if (4094..=4102).contains(&count) {
//                 Some(4096usize)
//             } else if (254..=260).contains(&count) {
//                 Some(256usize)
//             } else {
//                 None
//             };

//             if let Some(exact_len) = target_floats {
//                 let exact_bytes = exact_len * 4;
//                 // Find the exact start offset around `start` (within +/- 16 bytes)
//                 // by checking if a little-endian u32/u64 length prefix (either `exact_bytes` or `exact_len`) sits right before it,
//                 // OR by anchoring to the end of the contiguous float block if `count >= exact_len`.
//                 let mut best_start = if count >= exact_len {
//                     // Header bytes at the front caused extra 1-3 floats; the end of the float array is exact!
//                     end_byte - exact_bytes
//                 } else {
//                     // 1-2 small floats at the very start were skipped (< 1e-6); anchor to the end!
//                     end_byte.saturating_sub(exact_bytes)
//                 };

//                 // Double check +/- 16 bytes around `start` for an explicit LE u32 byte-length or element-count prefix
//                 let search_min = start.saturating_sub(16);
//                 let search_max = (start + 16).min(bytes.len().saturating_sub(exact_bytes));
//                 for cand in search_min..=search_max {
//                     if cand >= 4 {
//                         let prev_u32 = u32::from_le_bytes(bytes[cand - 4..cand].try_into().unwrap()) as usize;
//                         if prev_u32 == exact_bytes || prev_u32 == exact_len {
//                             best_start = cand;
//                             break;
//                         }
//                     }
//                 }

//                 if best_start + exact_bytes <= bytes.len() {
//                     let raw_slice = &bytes[best_start..best_start + exact_bytes];
//                     let floats: Vec<f32> = raw_slice
//                         .chunks_exact(4)
//                         .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
//                         .collect();

//                     tensors.push(BurnTensor {
//                         index: tensors.len(),
//                         data_offset: best_start,
//                         num_floats: exact_len,
//                         floats,
//                     });
//                 }
//             }
//             continue;
//         }
//         pos += 1;
//     }

//     tensors
// }

// /// Computes top-K singular values and singular vectors (sigma, u, v) via deflated power iteration
// fn svd_top_k(w: &[f32], rows: usize, cols: usize, k: usize) -> Vec<(f32, Vec<f32>, Vec<f32>)> {
//     let mut results: Vec<(f32, Vec<f32>, Vec<f32>)> = Vec::with_capacity(k);

//     for rank in 0..k {
//         let mut v: Vec<f32> = (0..cols)
//             .map(|j| (((j + rank * 17) as f32 * 1.6180339).sin()) + 0.05)
//             .collect();
//         normalize(&mut v);
//         let mut u = vec![0.0f32; rows];

//         for _ in 0..40 {
//             for r in 0..rows {
//                 let row = &w[r * cols..(r + 1) * cols];
//                 let mut dot = 0.0f32;
//                 for c in 0..cols {
//                     dot += row[c] * v[c];
//                 }
//                 u[r] = dot;
//             }
//             for (s_prev, u_prev, v_prev) in &results {
//                 let vd: f32 = v_prev.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
//                 for r in 0..rows {
//                     u[r] -= s_prev * u_prev[r] * vd;
//                 }
//             }
//             normalize(&mut u);

//             for c in 0..cols {
//                 let mut dot = 0.0f32;
//                 for r in 0..rows {
//                     dot += w[r * cols + c] * u[r];
//                 }
//                 v[c] = dot;
//             }
//             for (s_prev, u_prev, v_prev) in &results {
//                 let ud: f32 = u_prev.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
//                 for c in 0..cols {
//                     v[c] -= s_prev * v_prev[c] * ud;
//                 }
//             }
//             normalize(&mut v);
//         }

//         let mut sigma = 0.0f32;
//         for r in 0..rows {
//             let row = &w[r * cols..(r + 1) * cols];
//             let mut dot = 0.0f32;
//             for c in 0..cols {
//                 dot += row[c] * v[c];
//             }
//             sigma += u[r] * dot;
//         }
//         for (s_prev, u_prev, v_prev) in &results {
//             let ud: f32 = u_prev.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
//             let vd: f32 = v_prev.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
//             sigma -= s_prev * ud * vd;
//         }

//         results.push((sigma.abs(), u, v));
//     }

//     results
// }

// fn normalize(v: &mut [f32]) {
//     let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
//     if n > 1e-12 {
//         for x in v.iter_mut() {
//             *x /= n;
//         }
//     }
// }

// fn l2_norm(w: &[f32]) -> f32 {
//     w.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt() as f32
// }

// fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
//     let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| (x as f64) * (y as f64)).sum();
//     let na = l2_norm(a) as f64;
//     let nb = l2_norm(b) as f64;
//     if na * nb > 1e-12 {
//         (dot / (na * nb)) as f32
//     } else {
//         0.0
//     }
// }

// fn matmul_64x64(a: &[f32], b: &[f32], transpose_a: bool) -> Vec<f32> {
//     let mut out = vec![0.0f32; 64 * 64];
//     for i in 0..64 {
//         for j in 0..64 {
//             let mut sum = 0.0f32;
//             for k in 0..64 {
//                 let a_val = if transpose_a { a[k * 64 + i] } else { a[i * 64 + k] };
//                 sum += a_val * b[k * 64 + j];
//             }
//             out[i * 64 + j] = sum;
//         }
//     }
//     out
// }

// fn analyze_embeddings_and_head(t_in: &BurnTensor, t_out: &BurnTensor) {
//     println!("\n==================================================================");
//     println!(" 1. INPUT EMBEDDING vs. OUTPUT LM_HEAD FORENSICS (4096 x 64)");
//     println!("==================================================================");

//     for (label, t) in [
//         ("Input Embedding (Tensor #00)", t_in),
//         ("Output LM_Head  (Last 4096x64)", t_out),
//     ] {
//         let l2 = l2_norm(&t.floats);
//         let svd = svd_top_k(&t.floats, VOCAB_SIZE, EMBED_DIM, 5);
//         let (s1, s2, s3, s4, s5) = (svd[0].0, svd[1].0, svd[2].0, svd[3].0, svd[4].0);
//         let rank1_pct = (s1 * s1) / (l2 * l2) * 100.0;

//         println!("  [{}] @ byte {}", label, t.data_offset);
//         println!(
//             "    Frobenius L2 Norm: {:.2} | Top-5 Singular Values: [{:.2}, {:.2}, {:.2}, {:.2}, {:.2}]",
//             l2, s1, s2, s3, s4, s5
//         );
//         println!(
//             "    s1/s2 Ratio: {:.2}x | Rank-1 Energy: {:.1}% | Top-5 Cumulative Energy: {:.1}%",
//             s1 / s2.max(1e-6),
//             rank1_pct,
//             svd.iter().map(|(s, _, _)| s * s).sum::<f32>() / (l2 * l2) * 100.0
//         );

//         let u1 = &svd[0].1;
//         let mut token_proj: Vec<(usize, f32, f32)> = (0..VOCAB_SIZE)
//             .map(|tok_id| {
//                 let row = &t.floats[tok_id * EMBED_DIM..(tok_id + 1) * EMBED_DIM];
//                 let norm = l2_norm(row);
//                 (tok_id, u1[tok_id].abs() * s1, norm)
//             })
//             .collect();
//         token_proj.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

//         print!("    Top 10 Token IDs driving Rank-1 Attractor (ID: s1_proj / row_norm): ");
//         for (id, proj, norm) in token_proj.iter().take(10) {
//             print!("#{}({:.2}/{:.2}) ", id, proj, norm);
//         }
//         println!("\n");
//     }

//     let emb_head_cos = cosine_sim(&t_in.floats, &t_out.floats);
//     println!(
//         "  Global Cosine Similarity between Input Embedding & Output LM_Head: {:.4}\n",
//         emb_head_cos
//     );
// }

// fn main() -> std::io::Result<()> {
//     let args: Vec<String> = env::args().collect();
//     let path = args.get(1).map(|s| s.as_str()).unwrap_or("model.bin");
//     let fix_mode = args.iter().any(|a| a == "--fix-collapse");

//     let mut reader = BufReader::new(File::open(path)?);
//     let mut raw_bytes = Vec::new();
//     reader.read_to_end(&mut raw_bytes)?;

//     let tensors = extract_burn_tensors(&raw_bytes);
//     println!("==================================================================");
//     println!(
//         " EXACT BURN TENSOR EXTRACTOR | Snapped {} exact tensors in {}",
//         tensors.len(),
//         path
//     );
//     println!("==================================================================");

//     let embeds: Vec<&BurnTensor> = tensors.iter().filter(|t| t.num_floats == 262144).collect();
//     let routers: Vec<&BurnTensor> = tensors.iter().filter(|t| t.num_floats == 256).collect();
//     let attns: Vec<&BurnTensor> = tensors.iter().filter(|t| t.num_floats == 4096).collect();
//     let ffns: Vec<&BurnTensor> = tensors.iter().filter(|t| t.num_floats == 16384).collect();

//     println!(
//         "  Summary: {} Embeddings (4096x64), {} Attn Matrices (64x64), {} MoE Routers (4x64), {} Expert FFNs (64x256)",
//         embeds.len(),
//         attns.len(),
//         routers.len(),
//         ffns.len()
//     );

//     if embeds.len() >= 2 {
//         analyze_embeddings_and_head(embeds[0], embeds[embeds.len() - 1]);
//     }

//     println!("==================================================================");
//     println!(" 2. ATTENTION HEAD & VIRTUAL CIRCUIT ANALYSIS (64x64)");
//     println!("==================================================================");
//     for (l_idx, chunk) in attns.chunks(4).enumerate() {
//         println!("  --- Layer {} Attention ---", l_idx);
//         let names = ["W_Q", "W_K", "W_V", "W_O"];
//         for (i, &t) in chunk.iter().enumerate() {
//             let l2 = l2_norm(&t.floats);
//             let svd = svd_top_k(&t.floats, 64, 64, 2);
//             let (s1, s2) = (svd[0].0, svd[1].0);
//             let rank1 = (s1 * s1) / (l2 * l2) * 100.0;
//             println!(
//                 "    {} (Tensor #{:02} @ {:7}) | L2={:5.2} | s1={:5.2}, s2={:5.2} (s1/s2={:4.2}x) | Rank1={:4.1}%",
//                 names.get(i).unwrap_or(&"Attn"),
//                 t.index,
//                 t.data_offset,
//                 l2,
//                 s1,
//                 s2,
//                 s1 / s2.max(1e-6),
//                 rank1
//             );
//         }
//         if chunk.len() == 4 {
//             let w_qk = matmul_64x64(&chunk[0].floats, &chunk[1].floats, true);
//             let w_ov = matmul_64x64(&chunk[2].floats, &chunk[3].floats, false);
//             let qk_svd = svd_top_k(&w_qk, 64, 64, 2);
//             let ov_svd = svd_top_k(&w_ov, 64, 64, 2);
//             let qk_l2 = l2_norm(&w_qk);
//             let ov_l2 = l2_norm(&w_ov);
//             println!(
//                 "    -> Virtual W_QK: L2={:5.2}, s1/s2={:4.2}x, Rank1={:4.1}% | Virtual W_OV: L2={:5.2}, s1/s2={:4.2}x, Rank1={:4.1}%",
//                 qk_l2,
//                 qk_svd[0].0 / qk_svd[1].0.max(1e-6),
//                 (qk_svd[0].0 * qk_svd[0].0) / (qk_l2 * qk_l2) * 100.0,
//                 ov_l2,
//                 ov_svd[0].0 / ov_svd[1].0.max(1e-6),
//                 (ov_svd[0].0 * ov_svd[0].0) / (ov_l2 * ov_l2) * 100.0
//             );
//         }
//     }

//     println!("\n==================================================================");
//     println!(" 3. MoE ROUTER & EXPERT SPECIALIZATION ANALYSIS (4 Layers)");
//     println!("==================================================================");
//     for l_idx in 0..routers.len() {
//         let r = routers[l_idx];
//         println!("  --- Layer {} MoE (Router Tensor #{:02}) ---", l_idx, r.index);
//         print!("    Router Gate Row Norms: ");
//         for e in 0..NUM_EXPERTS {
//             let row = &r.floats[e * EMBED_DIM..(e + 1) * EMBED_DIM];
//             print!("E{}:{:.3}  ", e, l2_norm(row));
//         }
//         println!();

//         print!("    Router Gate Cosine Sims: ");
//         for i in 0..NUM_EXPERTS {
//             for j in (i + 1)..NUM_EXPERTS {
//                 let ri = &r.floats[i * EMBED_DIM..(i + 1) * EMBED_DIM];
//                 let rj = &r.floats[j * EMBED_DIM..(j + 1) * EMBED_DIM];
//                 print!("E{}-E{}:{:+.2} ", i, j, cosine_sim(ri, rj));
//             }
//         }
//         println!();

//         let layer_ffns = &ffns[l_idx * 12..((l_idx + 1) * 12).min(ffns.len())];
//         if layer_ffns.len() == 12 {
//             for e in 0..NUM_EXPERTS {
//                 let m0 = layer_ffns[e * 3];
//                 let m1 = layer_ffns[e * 3 + 1];
//                 let m2 = layer_ffns[e * 3 + 2];
//                 let s_m0 = svd_top_k(&m0.floats, 64, 256, 2);
//                 let s_m1 = svd_top_k(&m1.floats, 64, 256, 2);
//                 let s_m2 = svd_top_k(&m2.floats, 64, 256, 2);

//                 println!(
//                     "    Expert {}: M0(#{:02}) L2={:5.2} s1/s2={:4.2}x | M1(#{:02}) L2={:5.2} s1/s2={:4.2}x | M2(#{:02}) L2={:5.2} s1/s2={:4.2}x",
//                     e,
//                     m0.index, l2_norm(&m0.floats), s_m0[0].0 / s_m0[1].0.max(1e-6),
//                     m1.index, l2_norm(&m1.floats), s_m1[0].0 / s_m1[1].0.max(1e-6),
//                     m2.index, l2_norm(&m2.floats), s_m2[0].0 / s_m2[1].0.max(1e-6),
//                 );
//             }

//             // Check if experts are clones or distinct in weight space!
//             let e0_m1 = &layer_ffns[1].floats;
//             let e1_m1 = &layer_ffns[4].floats;
//             let e2_m1 = &layer_ffns[7].floats;
//             let e3_m1 = &layer_ffns[10].floats;
//             println!(
//                 "    Inter-Expert M1 Cosine Sim: E0-E1:{:+.3}, E0-E2:{:+.3}, E0-E3:{:+.3}, E1-E2:{:+.3}",
//                 cosine_sim(e0_m1, e1_m1),
//                 cosine_sim(e0_m1, e2_m1),
//                 cosine_sim(e0_m1, e3_m1),
//                 cosine_sim(e1_m1, e2_m1)
//             );
//         }
//     }

//     if fix_mode && embeds.len() >= 2 {
//         println!("\n==================================================================");
//         println!(" 4. SURGICAL WEIGHT-SPACE INTERVENTION (--fix-collapse)");
//         println!("==================================================================");
//         let mut patched_bytes = raw_bytes.clone();

//         for &target in &[embeds[0], embeds[embeds.len() - 1]] {
//             let svd = svd_top_k(&target.floats, VOCAB_SIZE, EMBED_DIM, 3);
//             let (s1, ref u1, ref v1) = svd[0];
//             let s2 = svd[1].0;

//             if s1 > 2.0 * s2 {
//                 let target_s1 = 1.15 * s2;
//                 let remove_scale = s1 - target_s1;
//                 println!(
//                     "  [+] Patching Tensor #{:02} @ byte {}: Reducing dominant sigma1 from {:.2} -> {:.2} (s2={:.2})",
//                     target.index, target.data_offset, s1, target_s1, s2
//                 );

//                 let mut new_floats = target.floats.clone();
//                 for r in 0..VOCAB_SIZE {
//                     for c in 0..EMBED_DIM {
//                         new_floats[r * EMBED_DIM + c] -= remove_scale * u1[r] * v1[c];
//                     }
//                 }

//                 for (k, &val) in new_floats.iter().enumerate() {
//                     let b = val.to_le_bytes();
//                     let off = target.data_offset + k * 4;
//                     patched_bytes[off..off + 4].copy_from_slice(&b);
//                 }
//             }
//         }

//         let out_path = format!("{}.patched", path);
//         let mut out_file = File::create(&out_path)?;
//         out_file.write_all(&patched_bytes)?;
//         println!("  [SUCCESS] Wrote surgically de-collapsed checkpoint to: {}", out_path);
//     }

//     Ok(())
// }

// analysis 3

use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufReader, Read, Write};

const VOCAB_SIZE: usize = 4096;
const EMBED_DIM: usize = 64;
const FF_DIM: usize = 256;
const NUM_EXPERTS: usize = 4;

#[derive(Clone)]
struct BurnTensor {
    index: usize,
    name_trail: String, // Extracted ASCII field path from Burn's MessagePack metadata before the tensor
    data_offset: usize,
    num_floats: usize,
    floats: Vec<f32>,
}

/// Scans backwards from `offset` (up to 160 bytes) to grab any readable ASCII field names
/// serialized by Burn's BinFileRecorder (e.g., "layers", "attention", "query", "weight", "gamma").
fn extract_burn_field_path(bytes: &[u8], offset: usize, prev_end: usize) -> String {
    let start = prev_end.max(offset.saturating_sub(180));
    let slice = &bytes[start..offset];
    let mut words = Vec::new();
    let mut cur = String::new();

    for &b in slice {
        if (b.is_ascii_alphanumeric() || b == b'_') && b != b'f' {
            cur.push(b as char);
        } else {
            if cur.len() >= 3
                && cur != "param"
                && cur != "value"
                && cur != "bytes"
                && cur != "shape"
                && cur != "32"
            {
                words.push(cur.clone());
            }
            cur.clear();
        }
    }
    if cur.len() >= 3 {
        words.push(cur);
    }
    if words.is_empty() {
        "tensor".to_string()
    } else {
        words.join(".")
    }
}

fn extract_burn_tensors(bytes: &[u8]) -> Vec<BurnTensor> {
    let mut tensors = Vec::new();
    let mut pos = 0;
    let mut last_tensor_end = 0usize;

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
                let mut best_start = end_byte.saturating_sub(exact_bytes);

                let search_min = start.saturating_sub(16);
                let search_max = (start + 16).min(bytes.len().saturating_sub(exact_bytes));
                for cand in search_min..=search_max {
                    if cand >= 4 {
                        let prev_u32 =
                            u32::from_le_bytes(bytes[cand - 4..cand].try_into().unwrap()) as usize;
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

                    let name_trail = extract_burn_field_path(bytes, best_start, last_tensor_end);
                    last_tensor_end = best_start + exact_bytes;

                    tensors.push(BurnTensor {
                        index: tensors.len(),
                        name_trail,
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

/// Optional simple vocab loader: parses either a line-by-line text vocab or HuggingFace/custom JSON `"token": id`
fn load_vocab_map(vocab_path: Option<&str>) -> HashMap<usize, String> {
    let mut map = HashMap::new();
    let Some(path) = vocab_path else { return map };
    let Ok(content) = fs::read_to_string(path) else {
        return map;
    };

    // Try simple `"token": 1234` or `"1234": "token"` patterns first
    if content.trim_start().starts_with('{') || content.contains("\"vocab\"") {
        for line in content.lines() {
            let trimmed = line.trim().trim_end_matches(',');
            if let Some((k, v)) = trimmed.split_once(':') {
                let k_clean = k.trim().trim_matches('"');
                let v_clean = v.trim().trim_matches('"');
                if let Ok(id) = v_clean.parse::<usize>() {
                    map.insert(id, k_clean.to_string());
                } else if let Ok(id) = k_clean.parse::<usize>() {
                    map.insert(id, v_clean.to_string());
                }
            }
        }
    } else {
        for (idx, line) in content.lines().enumerate() {
            map.insert(idx, line.trim().to_string());
        }
    }
    println!("  [+] Loaded {} token strings from {}", map.len(), path);
    map
}

fn svd_top_k(w: &[f32], rows: usize, cols: usize, k: usize) -> Vec<(f32, Vec<f32>, Vec<f32>)> {
    let mut results: Vec<(f32, Vec<f32>, Vec<f32>)> = Vec::with_capacity(k);

    for rank in 0..k {
        let mut v: Vec<f32> = (0..cols)
            .map(|j| (((j + rank * 17) as f32 * 1.6180339).sin()) + 0.05)
            .collect();
        normalize(&mut v);
        let mut u = vec![0.0f32; rows];

        for _ in 0..45 {
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

/// Surgically dampens top singular value `s1` down to `target_ratio * s2` in-place inside `patched_bytes`
fn dampen_rank1_in_place(
    patched_bytes: &mut [u8],
    t: &BurnTensor,
    rows: usize,
    cols: usize,
    trigger_ratio: f32,
    target_ratio: f32,
) -> bool {
    let svd = svd_top_k(&t.floats, rows, cols, 2);
    let (s1, ref u1, ref v1) = svd[0];
    let s2 = svd[1].0;

    if s1 > trigger_ratio * s2 {
        let new_s1 = target_ratio * s2;
        let remove = s1 - new_s1;
        println!(
            "  [PATCH] Tensor #{:02} ({}) @ byte {}: Dampening s1 {:.2} -> {:.2} (s2={:.2}, was {:.2}x)",
            t.index,
            t.name_trail,
            t.data_offset,
            s1,
            new_s1,
            s2,
            s1 / s2.max(1e-6)
        );

        let mut new_floats = t.floats.clone();
        for r in 0..rows {
            for c in 0..cols {
                new_floats[r * cols + c] -= remove * u1[r] * v1[c];
            }
        }

        for (k, &val) in new_floats.iter().enumerate() {
            let b = val.to_le_bytes();
            let off = t.data_offset + k * 4;
            patched_bytes[off..off + 4].copy_from_slice(&b);
        }
        return true;
    }
    false
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).map(|s| s.as_str()).unwrap_or("model.bin");
    let fix_mode = args.iter().any(|a| a == "--fix-collapse");
    let vocab_arg = args
        .iter()
        .find(|a| a.ends_with(".json") || a.ends_with(".txt"))
        .map(|s| s.as_str());

    let vocab_map = load_vocab_map(vocab_arg);

    let mut reader = BufReader::new(File::open(path)?);
    let mut raw_bytes = Vec::new();
    reader.read_to_end(&mut raw_bytes)?;

    let tensors = extract_burn_tensors(&raw_bytes);

    println!("==================================================================");
    println!(" 1. BURN FIELD NAMES & ANOMALOUS TENSOR MAP ({} Tensors)", tensors.len());
    println!("==================================================================");
    for t in &tensors {
        // Print all non-FFN tensors + the first FFN of each layer so we see the exact Burn struct names!
        if t.num_floats != 16384 || t.index == 6 || t.index == 23 || t.index == 40 || t.index == 57 {
            let (r, c) = match t.num_floats {
                262144 => (4096, 64),
                16384 => (64, 256),
                4096 => (64, 64),
                256 => (4, 64),
                _ => (1, t.num_floats),
            };
            let svd = svd_top_k(&t.floats, r, c, 2);
            let l2 = l2_norm(&t.floats);
            let (s1, s2) = (svd[0].0, svd[1].0);
            println!(
                "  #{:02} | {:4}x{:<3} | L2={:6.2} | s1/s2={:4.2}x | Rank1={:4.1}% | Path: {}",
                t.index,
                r,
                c,
                l2,
                s1 / s2.max(1e-6),
                (s1 * s1) / (l2 * l2) * 100.0,
                t.name_trail
            );
        }
    }

    // Decode the top attractor tokens in Output LM_Head (#69) along both +u1 and -u1!
    if let Some(lm_head) = tensors.iter().rev().find(|t| t.num_floats == 262144) {
        println!("\n==================================================================");
        println!(" 2. OUTPUT LM_HEAD (#{:02}) SINGULAR VECTOR DECODING", lm_head.index);
        println!("==================================================================");
        let svd = svd_top_k(&lm_head.floats, VOCAB_SIZE, EMBED_DIM, 3);
        for (rank_idx, (sigma, u, _v)) in svd.iter().enumerate() {
            let mut pos_toks: Vec<(usize, f32)> = u.iter().cloned().enumerate().collect();
            pos_toks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            print!("  Singular Vector u{} (sigma={:.2}) [+ pole]: ", rank_idx + 1, sigma);
            for &(id, val) in pos_toks.iter().take(8) {
                let word = vocab_map.get(&id).map(|s| s.as_str()).unwrap_or("");
                print!("#{}{}(\"{}\":{:+.2}) ", id, "", word, val * sigma);
            }
            println!();

            print!("  Singular Vector u{} (sigma={:.2}) [- pole]: ", rank_idx + 1, sigma);
            for &(id, val) in pos_toks.iter().rev().take(8) {
                let word = vocab_map.get(&id).map(|s| s.as_str()).unwrap_or("");
                print!("#{}{}(\"{}\":{:+.2}) ", id, "", word, val * sigma);
            }
            println!("\n");
        }
    }

    // 3. Full Surgical De-Collapse Mode
    if fix_mode {
        println!("==================================================================");
        println!(" 3. EXECUTING SURGICAL WEIGHT DE-COLLAPSE (--fix-collapse)");
        println!("==================================================================");
        let mut patched_bytes = raw_bytes.clone();
        let mut patch_count = 0;

        for t in &tensors {
            match t.num_floats {
                262144 => {
                    // Dampen LM_Head Rank-1 attractor (4.29x -> 1.15x)
                    if dampen_rank1_in_place(&mut patched_bytes, t, 4096, 64, 1.8, 1.15) {
                        patch_count += 1;
                    }
                }
                4096 => {
                    // Dampen collapsed W_Q, W_K, and Tensor #70 (where s1/s2 > 1.50x -> 1.20x)
                    // This breaks the 93.7% W_QK attention-sink blindness!
                    if dampen_rank1_in_place(&mut patched_bytes, t, 64, 64, 1.50, 1.20) {
                        patch_count += 1;
                    }
                }
                _ => {}
            }
        }

        let out_path = format!("{}.patched", path);
        let mut out_file = File::create(&out_path)?;
        out_file.write_all(&patched_bytes)?;
        println!(
            "\n  [SUCCESS] Surgically patched {} collapsed matrices and wrote: {}",
            patch_count, out_path
        );
        println!("  --> Swap `model.bin` with `model.bin.patched` (keep a backup!) and run your generation test prompts!");
    } else {
        println!("  [TIP] Run with `--fix-collapse` (and optionally pass your `vocab.json` path) to generate `model.bin.patched`!");
    }

    Ok(())
}