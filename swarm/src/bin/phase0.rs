//! Phase 0: extraction over code.bin, no training.
//! Checks the extractor (byte-exact rebuild, skeleton and fill parse rates), counts
//! instances per kind, measures prompt lengths and marker token costs, and selects
//! each role's output vocabulary on training files, scoring coverage on held-out files.
//!
//! cargo run --release --bin phase0 -- [--limit N] [--roles roles/kinds.json,roles/coarse.json]
use anyhow::Result;
use rayon::prelude::*;
use std::{collections::BTreeMap, path::PathBuf};
use swarm::{
    dataset::{self, Chunk, FILL_SEP},
    roles::RoleSet,
    skeleton::{self, KINDS},
};
use tokenizers::Tokenizer;

/// 2,048 code BPE ids plus the 25 swarm special tokens (`make_tokenizer`).
const VOCAB: usize = 2048 + 25;
const EOS: u32 = 2;

#[derive(Default)]
struct Stats {
    chunks: usize,
    unparsed: usize,
    mismatched: usize,
    nodes: usize,
    skel_fail: BTreeMap<String, usize>,
    fill_fail: BTreeMap<String, usize>,
    count: BTreeMap<String, usize>,
    depth: BTreeMap<usize, usize>,
    // per kind: token histogram of targets
    hist: BTreeMap<String, Vec<u64>>,
    // per kind: target token lengths, prompt (context + sep + target) lengths
    fill_len: BTreeMap<String, Vec<u32>>,
    prompt_len: BTreeMap<String, Vec<u32>>,
}

impl Stats {
    fn merge(mut self, o: Stats) -> Stats {
        self.chunks += o.chunks;
        self.unparsed += o.unparsed;
        self.mismatched += o.mismatched;
        self.nodes += o.nodes;
        for (k, v) in o.skel_fail {
            *self.skel_fail.entry(k).or_default() += v;
        }
        for (k, v) in o.fill_fail {
            *self.fill_fail.entry(k).or_default() += v;
        }
        for (k, v) in o.count {
            *self.count.entry(k).or_default() += v;
        }
        for (k, v) in o.depth {
            *self.depth.entry(k).or_default() += v;
        }
        for (k, v) in o.hist {
            let h = self.hist.entry(k).or_insert_with(|| vec![0; VOCAB]);
            for (a, b) in h.iter_mut().zip(v) {
                *a += b;
            }
        }
        for (k, v) in o.fill_len {
            self.fill_len.entry(k).or_default().extend(v);
        }
        for (k, v) in o.prompt_len {
            self.prompt_len.entry(k).or_default().extend(v);
        }
        self
    }
}

fn ids(tok: &Tokenizer, text: &str) -> Vec<u32> {
    tok.encode(text, false).map(|e| e.get_ids().to_vec()).unwrap_or_default()
}

fn pct(v: &mut Vec<u32>, p: f64) -> u32 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[((v.len() - 1) as f64 * p) as usize]
}

/// Training files when `train_only`, held-out files otherwise.
fn scan(chunks: &[Chunk], tok: &Tokenizer, train_only: bool) -> Stats {
    let sep = ids(tok, FILL_SEP).len();
    chunks
        .par_iter()
        .map(|chunk| {
            let mut s = Stats::default();
            if train_only == dataset::is_validation(&chunk.file) {
                return s;
            }
            s.chunks = 1;
            let Ok(root) = dataset::extract_chunk(chunk) else {
                s.unparsed = 1;
                return s;
            };
            if root.assemble() != chunk.text {
                s.mismatched = 1;
            }
            root.walk(&mut Vec::new(), &mut |n, path| {
                s.nodes += 1;
                if !skeleton::skeleton_parses(n) {
                    *s.skel_fail.entry(n.kind.clone()).or_default() += 1;
                }
                if !path.is_empty() && !skeleton::fill_parses(&n.kind, &n.assemble()) {
                    *s.fill_fail.entry(n.kind.clone()).or_default() += 1;
                }
            });
            for inst in dataset::instances(&root) {
                *s.count.entry(inst.kind.clone()).or_default() += 1;
                *s.depth.entry(inst.depth).or_default() += 1;
                let target = ids(tok, &inst.target);
                let h = s.hist.entry(inst.kind.clone()).or_insert_with(|| vec![0; VOCAB]);
                for &t in &target {
                    h[t as usize] += 1;
                }
                h[EOS as usize] += 1;
                let ctx = ids(tok, &inst.context).len();
                s.fill_len.entry(inst.kind.clone()).or_default().push(target.len() as u32 + 1);
                s.prompt_len
                    .entry(inst.kind.clone())
                    .or_default()
                    .push((1 + ctx + sep + target.len() + 1) as u32);
            }
            s
        })
        .reduce(Stats::default, Stats::merge)
}

/// The role's rows: EOS plus the most frequent target tokens, `out_vocab` in total.
fn role_vocab(hist: &[u64], out_vocab: usize) -> Vec<u32> {
    let mut order: Vec<u32> = (0..VOCAB as u32).filter(|&t| hist[t as usize] > 0 && t != EOS).collect();
    order.sort_by_key(|&t| std::cmp::Reverse(hist[t as usize]));
    let mut v = vec![EOS];
    v.extend(order.into_iter().take(out_vocab - 1));
    v
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let limit = arg("--limit").map(|v| v.parse()).transpose()?;
    let role_files = arg("--roles").unwrap_or("roles/kinds.json,roles/coarse.json".into());
    let cache = PathBuf::from(arg("--cache").unwrap_or("../training-cache/code.bin".into()));
    let tokenizer = PathBuf::from(arg("--tokenizer").unwrap_or(
        "tokenizer.json".into(),
    ));
    let out = PathBuf::from(arg("--out").unwrap_or("out/phase0".into()));
    std::fs::create_dir_all(&out)?;
    let tok = Tokenizer::from_file(&tokenizer).map_err(|e| anyhow::anyhow!("{e}"))?;

    let t = std::time::Instant::now();
    let chunks = dataset::load_chunks(&cache, &tokenizer, limit)?;
    println!("loaded {} chunks in {:.1}s", chunks.len(), t.elapsed().as_secs_f64());
    let val_files: std::collections::HashSet<_> =
        chunks.iter().filter(|c| dataset::is_validation(&c.file)).map(|c| &c.file).collect();
    let all_files: std::collections::HashSet<_> = chunks.iter().map(|c| &c.file).collect();
    println!("files {} (held out {})", all_files.len(), val_files.len());

    println!("\nmarkers:");
    for k in KINDS {
        for text in [skeleton::marker(k), skeleton::target_marker(k)] {
            anyhow::ensure!(ids(&tok, &text).len() == 1, "{text} is not one token; run make_tokenizer");
        }
    }
    anyhow::ensure!(ids(&tok, FILL_SEP).len() == 1, "{FILL_SEP} is not one token");
    println!("  all {} open/target markers and the fill separator are 1 token", 2 * KINDS.len());
    let clashes = chunks.iter().filter(|c| skeleton::special_tokens().iter().any(|t| c.text.contains(t.as_str()))).count();
    println!("  chunks that already contain a marker string: {clashes}");
    let markers: Vec<serde_json::Value> = Vec::new();


    let t = std::time::Instant::now();
    let mut train = scan(&chunks, &tok, true);
    println!("\nscanned training files in {:.1}s", t.elapsed().as_secs_f64());
    let val = scan(&chunks, &tok, false);
    println!(
        "chunks: train {} (unparsed {}, rebuild mismatches {}), held-out {} (unparsed {}, mismatches {})",
        train.chunks, train.unparsed, train.mismatched, val.chunks, val.unparsed, val.mismatched
    );
    println!("nodes: {} train", train.nodes);
    println!("skeleton parse failures: {:?}\nfill parse failures: {:?}", train.skel_fail, train.fill_fail);

    println!(
        "\n{:10} {:>9} {:>9} {:>6} {:>6} {:>6} | prompt tokens p50 p90 p99 >512",
        "kind", "train", "held-out", "fill50", "fill90", "fill99"
    );
    let mut kind_rows = vec![];
    for k in KINDS {
        let n = train.count.get(k).copied().unwrap_or(0);
        let nv = val.count.get(k).copied().unwrap_or(0);
        let fl = train.fill_len.entry(k.into()).or_default();
        let (f50, f90, f99) = (pct(fl, 0.5), pct(fl, 0.9), pct(fl, 0.99));
        let pl = train.prompt_len.entry(k.into()).or_default();
        let (p50, p90, p99) = (pct(pl, 0.5), pct(pl, 0.9), pct(pl, 0.99));
        let over = pl.iter().filter(|&&l| l > 512).count() as f64 / pl.len().max(1) as f64;
        println!("{k:10} {n:9} {nv:9} {f50:6} {f90:6} {f99:6} | {p50:5} {p90:5} {p99:5} {:5.1}%", over * 100.0);
        kind_rows.push(serde_json::json!({"kind": k, "train": n, "held_out": nv, "fill_p50": f50, "fill_p90": f90,
            "fill_p99": f99, "prompt_p50": p50, "prompt_p90": p90, "prompt_p99": p99, "prompt_over_512": over}));
    }
    println!("instances by depth: {:?}", train.depth);

    // Coverage per role set: vocabulary chosen on training targets, scored on held-out files.
    let val_chunks: Vec<&Chunk> = chunks.iter().filter(|c| dataset::is_validation(&c.file)).collect();
    let mut sets_json = vec![];
    for path in role_files.split(',') {
        let set = RoleSet::load(std::path::Path::new(path))?;
        println!("\nrole set {} ({} roles), {} output rows each", set.name, set.roles.len(), set.roles[0].out_vocab);
        let vocabs: Vec<std::collections::HashSet<u32>> = set
            .roles
            .iter()
            .map(|r| {
                let mut h = vec![0u64; VOCAB];
                for k in &r.kinds {
                    if let Some(kh) = train.hist.get(k) {
                        for (a, b) in h.iter_mut().zip(kh) {
                            *a += b;
                        }
                    }
                }
                role_vocab(&h, r.out_vocab).into_iter().collect()
            })
            .collect();
        let n_roles = set.roles.len();
        let results = val_chunks
            .par_iter()
            .map(|chunk| {
                let mut r = vec![(0usize, 0usize, 0u64, 0u64); n_roles];
                let Ok(root) = dataset::extract_chunk(chunk) else { return r };
                for inst in dataset::instances(&root) {
                    let Some(ri) = set.roles.iter().position(|x| x.kinds.contains(&inst.kind)) else { continue };
                    let target = ids(&tok, &inst.target);
                    let inside = target.iter().filter(|t| vocabs[ri].contains(t)).count();
                    r[ri].0 += 1;
                    r[ri].1 += (inside == target.len()) as usize;
                    r[ri].2 += target.len() as u64;
                    r[ri].3 += inside as u64;
                }
                r
            })
            .reduce(
                || vec![(0, 0, 0, 0); n_roles],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b) {
                        x.0 += y.0;
                        x.1 += y.1;
                        x.2 += y.2;
                        x.3 += y.3;
                    }
                    a
                },
            );
        let mut roles_json = vec![];
        println!("{:10} {:>8} {:>13} {:>14}  kinds", "role", "held-out", "fills in-role", "tokens in-role");
        for (r, (n, exact, tt, ti)) in set.roles.iter().zip(&results) {
            let train_n: usize = r.kinds.iter().map(|k| train.count.get(k).copied().unwrap_or(0)).sum();
            let (fe, te) = (*exact as f64 / (*n).max(1) as f64, *ti as f64 / (*tt).max(1) as f64);
            println!("{:10} {:8} {:12.1}% {:13.1}%  {:?} (train {})", r.name, n, fe * 100.0, te * 100.0, r.kinds, train_n);
            roles_json.push(serde_json::json!({"role": r.name, "kinds": r.kinds, "train": train_n, "held_out": n,
                "fills_in_role": fe, "tokens_in_role": te}));
        }
        let vocab_json: Vec<_> = set
            .roles
            .iter()
            .zip(&vocabs)
            .map(|(r, v)| {
                let mut v: Vec<u32> = v.iter().copied().collect();
                v.sort();
                serde_json::json!({"role": r.name, "ids": v})
            })
            .collect();
        std::fs::write(out.join(format!("vocab-{}.json", set.name)), serde_json::to_string(&vocab_json)?)?;
        sets_json.push(serde_json::json!({"set": set.name, "roles": roles_json}));
    }

    // Output-vocabulary sweep: how many held-out fills fit entirely inside N rows.
    println!("
fills fully inside the role's rows, by output rows (one role per kind):");
    let sizes = [256usize, 512, 1024, VOCAB];
    let kinds_set = RoleSet::load(std::path::Path::new("roles/kinds.json"))?;
    print!("{:10}", "kind");
    for n in sizes { print!(" {n:>7}"); }
    println!();
    let mut sweep_json = vec![];
    for r in &kinds_set.roles {
        let hist = train.hist.get(&r.kinds[0]).cloned().unwrap_or_else(|| vec![0; VOCAB]);
        let sets: Vec<std::collections::HashSet<u32>> =
            sizes.iter().map(|&n| role_vocab(&hist, n).into_iter().collect()).collect();
        let (total, inside) = val_chunks
            .par_iter()
            .map(|chunk| {
                let mut tot = 0usize;
                let mut ins = vec![0usize; sizes.len()];
                if let Ok(root) = dataset::extract_chunk(chunk) {
                    for inst in dataset::instances(&root).iter().filter(|i| i.kind == r.kinds[0]) {
                        let target = ids(&tok, &inst.target);
                        tot += 1;
                        for (j, set) in sets.iter().enumerate() {
                            ins[j] += target.iter().all(|t| set.contains(t)) as usize;
                        }
                    }
                }
                (tot, ins)
            })
            .reduce(|| (0, vec![0; sizes.len()]), |a, b| (a.0 + b.0, a.1.iter().zip(&b.1).map(|(x, y)| x + y).collect()));
        print!("{:10}", r.name);
        let row: Vec<f64> = inside.iter().map(|&i| i as f64 / total.max(1) as f64).collect();
        for v in &row { print!(" {:6.1}%", v * 100.0); }
        println!("  (n {total})");
        sweep_json.push(serde_json::json!({"kind": r.name, "held_out": total, "sizes": sizes, "fills_in_role": row}));
    }

    let mut ex = String::new();
    for chunk in chunks.iter().filter(|c| c.text.len() > 200 && c.text.len() < 700).take(3) {
        if let Ok(root) = dataset::extract_chunk(chunk) {
            ex += &format!("=== {}\n--- source\n{}\n--- skeleton\n{}\n", chunk.file, chunk.text, root.skeleton());
            for i in dataset::instances(&root).iter().take(4) {
                ex += &format!("--- worker {} (depth {}) target: {}\n{}\n", i.kind, i.depth, i.target, i.context);
            }
        }
    }
    std::fs::write(out.join("examples.txt"), ex)?;
    std::fs::write(
        out.join("report.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "chunks_train": train.chunks, "chunks_held_out": val.chunks, "unparsed": train.unparsed + val.unparsed,
            "rebuild_mismatches": train.mismatched + val.mismatched, "markers": markers, "kinds": kind_rows,
            "role_sets": sets_json, "vocab_sweep": sweep_json, "skeleton_parse_failures": train.skel_fail, "fill_parse_failures": train.fill_fail,
        }))?,
    )?;
    println!("\nwrote {}", out.display());
    Ok(())
}
