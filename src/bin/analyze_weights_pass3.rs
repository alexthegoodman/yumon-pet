// ** analysis 3 **

// use std::collections::HashMap;
// use std::env;
// use std::fs::{self, File};
// use std::io::{BufReader, Read, Write};

// const VOCAB_SIZE: usize = 4096;
// const EMBED_DIM: usize = 64;
// const FF_DIM: usize = 256;
// const NUM_EXPERTS: usize = 4;

// #[derive(Clone)]
// struct BurnTensor {
//     index: usize,
//     name_trail: String, // Extracted ASCII field path from Burn's MessagePack metadata before the tensor
//     data_offset: usize,
//     num_floats: usize,
//     floats: Vec<f32>,
// }

// /// Scans backwards from `offset` (up to 160 bytes) to grab any readable ASCII field names
// /// serialized by Burn's BinFileRecorder (e.g., "layers", "attention", "query", "weight", "gamma").
// fn extract_burn_field_path(bytes: &[u8], offset: usize, prev_end: usize) -> String {
//     let start = prev_end.max(offset.saturating_sub(180));
//     let slice = &bytes[start..offset];
//     let mut words = Vec::new();
//     let mut cur = String::new();

//     for &b in slice {
//         if (b.is_ascii_alphanumeric() || b == b'_') && b != b'f' {
//             cur.push(b as char);
//         } else {
//             if cur.len() >= 3
//                 && cur != "param"
//                 && cur != "value"
//                 && cur != "bytes"
//                 && cur != "shape"
//                 && cur != "32"
//             {
//                 words.push(cur.clone());
//             }
//             cur.clear();
//         }
//     }
//     if cur.len() >= 3 {
//         words.push(cur);
//     }
//     if words.is_empty() {
//         "tensor".to_string()
//     } else {
//         words.join(".")
//     }
// }

// fn extract_burn_tensors(bytes: &[u8]) -> Vec<BurnTensor> {
//     let mut tensors = Vec::new();
//     let mut pos = 0;
//     let mut last_tensor_end = 0usize;

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
//                 let mut best_start = end_byte.saturating_sub(exact_bytes);

//                 let search_min = start.saturating_sub(16);
//                 let search_max = (start + 16).min(bytes.len().saturating_sub(exact_bytes));
//                 for cand in search_min..=search_max {
//                     if cand >= 4 {
//                         let prev_u32 =
//                             u32::from_le_bytes(bytes[cand - 4..cand].try_into().unwrap()) as usize;
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

//                     let name_trail = extract_burn_field_path(bytes, best_start, last_tensor_end);
//                     last_tensor_end = best_start + exact_bytes;

//                     tensors.push(BurnTensor {
//                         index: tensors.len(),
//                         name_trail,
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

// /// Optional simple vocab loader: parses either a line-by-line text vocab or HuggingFace/custom JSON `"token": id`
// fn load_vocab_map(vocab_path: Option<&str>) -> HashMap<usize, String> {
//     let mut map = HashMap::new();
//     let Some(path) = vocab_path else { return map };
//     let Ok(content) = fs::read_to_string(path) else {
//         return map;
//     };

//     // Try simple `"token": 1234` or `"1234": "token"` patterns first
//     if content.trim_start().starts_with('{') || content.contains("\"vocab\"") {
//         for line in content.lines() {
//             let trimmed = line.trim().trim_end_matches(',');
//             if let Some((k, v)) = trimmed.split_once(':') {
//                 let k_clean = k.trim().trim_matches('"');
//                 let v_clean = v.trim().trim_matches('"');
//                 if let Ok(id) = v_clean.parse::<usize>() {
//                     map.insert(id, k_clean.to_string());
//                 } else if let Ok(id) = k_clean.parse::<usize>() {
//                     map.insert(id, v_clean.to_string());
//                 }
//             }
//         }
//     } else {
//         for (idx, line) in content.lines().enumerate() {
//             map.insert(idx, line.trim().to_string());
//         }
//     }
//     println!("  [+] Loaded {} token strings from {}", map.len(), path);
//     map
// }

// fn svd_top_k(w: &[f32], rows: usize, cols: usize, k: usize) -> Vec<(f32, Vec<f32>, Vec<f32>)> {
//     let mut results: Vec<(f32, Vec<f32>, Vec<f32>)> = Vec::with_capacity(k);

//     for rank in 0..k {
//         let mut v: Vec<f32> = (0..cols)
//             .map(|j| (((j + rank * 17) as f32 * 1.6180339).sin()) + 0.05)
//             .collect();
//         normalize(&mut v);
//         let mut u = vec![0.0f32; rows];

//         for _ in 0..45 {
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

// /// Surgically dampens top singular value `s1` down to `target_ratio * s2` in-place inside `patched_bytes`
// fn dampen_rank1_in_place(
//     patched_bytes: &mut [u8],
//     t: &BurnTensor,
//     rows: usize,
//     cols: usize,
//     trigger_ratio: f32,
//     target_ratio: f32,
// ) -> bool {
//     let svd = svd_top_k(&t.floats, rows, cols, 2);
//     let (s1, ref u1, ref v1) = svd[0];
//     let s2 = svd[1].0;

//     if s1 > trigger_ratio * s2 {
//         let new_s1 = target_ratio * s2;
//         let remove = s1 - new_s1;
//         println!(
//             "  [PATCH] Tensor #{:02} ({}) @ byte {}: Dampening s1 {:.2} -> {:.2} (s2={:.2}, was {:.2}x)",
//             t.index,
//             t.name_trail,
//             t.data_offset,
//             s1,
//             new_s1,
//             s2,
//             s1 / s2.max(1e-6)
//         );

//         let mut new_floats = t.floats.clone();
//         for r in 0..rows {
//             for c in 0..cols {
//                 new_floats[r * cols + c] -= remove * u1[r] * v1[c];
//             }
//         }

//         for (k, &val) in new_floats.iter().enumerate() {
//             let b = val.to_le_bytes();
//             let off = t.data_offset + k * 4;
//             patched_bytes[off..off + 4].copy_from_slice(&b);
//         }
//         return true;
//     }
//     false
// }

// fn main() -> std::io::Result<()> {
//     let args: Vec<String> = env::args().collect();
//     let path = args.get(1).map(|s| s.as_str()).unwrap_or("model.bin");
//     let fix_mode = args.iter().any(|a| a == "--fix-collapse");
//     let vocab_arg = args
//         .iter()
//         .find(|a| a.ends_with(".json") || a.ends_with(".txt"))
//         .map(|s| s.as_str());

//     let vocab_map = load_vocab_map(vocab_arg);

//     let mut reader = BufReader::new(File::open(path)?);
//     let mut raw_bytes = Vec::new();
//     reader.read_to_end(&mut raw_bytes)?;

//     let tensors = extract_burn_tensors(&raw_bytes);

//     println!("==================================================================");
//     println!(" 1. BURN FIELD NAMES & ANOMALOUS TENSOR MAP ({} Tensors)", tensors.len());
//     println!("==================================================================");
//     for t in &tensors {
//         // Print all non-FFN tensors + the first FFN of each layer so we see the exact Burn struct names!
//         if t.num_floats != 16384 || t.index == 6 || t.index == 23 || t.index == 40 || t.index == 57 {
//             let (r, c) = match t.num_floats {
//                 262144 => (4096, 64),
//                 16384 => (64, 256),
//                 4096 => (64, 64),
//                 256 => (4, 64),
//                 _ => (1, t.num_floats),
//             };
//             let svd = svd_top_k(&t.floats, r, c, 2);
//             let l2 = l2_norm(&t.floats);
//             let (s1, s2) = (svd[0].0, svd[1].0);
//             println!(
//                 "  #{:02} | {:4}x{:<3} | L2={:6.2} | s1/s2={:4.2}x | Rank1={:4.1}% | Path: {}",
//                 t.index,
//                 r,
//                 c,
//                 l2,
//                 s1 / s2.max(1e-6),
//                 (s1 * s1) / (l2 * l2) * 100.0,
//                 t.name_trail
//             );
//         }
//     }

//     // Decode the top attractor tokens in Output LM_Head (#69) along both +u1 and -u1!
//     if let Some(lm_head) = tensors.iter().rev().find(|t| t.num_floats == 262144) {
//         println!("\n==================================================================");
//         println!(" 2. OUTPUT LM_HEAD (#{:02}) SINGULAR VECTOR DECODING", lm_head.index);
//         println!("==================================================================");
//         let svd = svd_top_k(&lm_head.floats, VOCAB_SIZE, EMBED_DIM, 3);
//         for (rank_idx, (sigma, u, _v)) in svd.iter().enumerate() {
//             let mut pos_toks: Vec<(usize, f32)> = u.iter().cloned().enumerate().collect();
//             pos_toks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

//             print!("  Singular Vector u{} (sigma={:.2}) [+ pole]: ", rank_idx + 1, sigma);
//             for &(id, val) in pos_toks.iter().take(8) {
//                 let word = vocab_map.get(&id).map(|s| s.as_str()).unwrap_or("");
//                 print!("#{}{}(\"{}\":{:+.2}) ", id, "", word, val * sigma);
//             }
//             println!();

//             print!("  Singular Vector u{} (sigma={:.2}) [- pole]: ", rank_idx + 1, sigma);
//             for &(id, val) in pos_toks.iter().rev().take(8) {
//                 let word = vocab_map.get(&id).map(|s| s.as_str()).unwrap_or("");
//                 print!("#{}{}(\"{}\":{:+.2}) ", id, "", word, val * sigma);
//             }
//             println!("\n");
//         }
//     }

//     // 3. Full Surgical De-Collapse Mode
//     if fix_mode {
//         println!("==================================================================");
//         println!(" 3. EXECUTING SURGICAL WEIGHT DE-COLLAPSE (--fix-collapse)");
//         println!("==================================================================");
//         let mut patched_bytes = raw_bytes.clone();
//         let mut patch_count = 0;

//         for t in &tensors {
//             match t.num_floats {
//                 262144 => {
//                     // Dampen LM_Head Rank-1 attractor (4.29x -> 1.15x)
//                     if dampen_rank1_in_place(&mut patched_bytes, t, 4096, 64, 1.8, 1.15) {
//                         patch_count += 1;
//                     }
//                 }
//                 4096 => {
//                     // Dampen collapsed W_Q, W_K, and Tensor #70 (where s1/s2 > 1.50x -> 1.20x)
//                     // This breaks the 93.7% W_QK attention-sink blindness!
//                     if dampen_rank1_in_place(&mut patched_bytes, t, 64, 64, 1.50, 1.20) {
//                         patch_count += 1;
//                     }
//                 }
//                 _ => {}
//             }
//         }

//         let out_path = format!("{}.patched", path);
//         let mut out_file = File::create(&out_path)?;
//         out_file.write_all(&patched_bytes)?;
//         println!(
//             "\n  [SUCCESS] Surgically patched {} collapsed matrices and wrote: {}",
//             patch_count, out_path
//         );
//         println!("  --> Swap `model.bin` with `model.bin.patched` (keep a backup!) and run your generation test prompts!");
//     } else {
//         println!("  [TIP] Run with `--fix-collapse` (and optionally pass your `vocab.json` path) to generate `model.bin.patched`!");
//     }

//     Ok(())
// }


fn main() {}
