//! Phase 1: oracle swarm over held-out chunks, then one-hole swaps per kind.
//!
//! 1. Dispatcher + oracle fills must rebuild every held-out chunk byte for byte.
//! 2. `cargo check` on a sample of rebuilt chunks gives the ceiling (chunks are cut
//!    from files, so many fail on unresolved names even with ground truth). Only
//!    chunks that check clean are used below, so every error is the swap's.
//! 3. Per kind: in each clean chunk, one hole of that kind is written by a wrong but
//!    parseable fill from another file; report parse rate, check rate, error codes
//!    and how often the error is charged to the swapped hole's bytes.
//!
//! cargo run --release --bin oracle -- [--limit N] [--check N] [--max-chars 200]
use anyhow::Result;
use std::{collections::BTreeMap, path::PathBuf};
use swarm::{
    check::{self, Report},
    dataset::{self, Chunk},
    skeleton::{self, KINDS, Node},
    swarm::{self as sw, Oracle, SwapOne},
};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn pct(a: usize, b: usize) -> f64 {
    if b == 0 { 0.0 } else { 100.0 * a as f64 / b as f64 }
}

fn top_codes(counts: &BTreeMap<String, usize>, n: usize) -> String {
    let mut v: Vec<_> = counts.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1));
    v.iter().take(n).map(|(c, k)| format!("{c}:{k}")).collect::<Vec<_>>().join(" ")
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let limit = arg("--limit").map(|v| v.parse()).transpose()?;
    let check_n: usize = arg("--check").map(|v| v.parse()).transpose()?.unwrap_or(400);
    let max_chars: usize = arg("--max-chars").map(|v| v.parse()).transpose()?.unwrap_or(200);
    let tokenizer = PathBuf::from(arg("--tokenizer").unwrap_or("tokenizer.json".into()));
    let scratch = PathBuf::from(arg("--scratch").unwrap_or("out/scratch".into()));
    let chunks = dataset::load_chunks(&PathBuf::from("../training-cache/code.bin"), &tokenizer, limit)?;
    println!("{} chunks", chunks.len());

    let (val, train): (Vec<&Chunk>, Vec<&Chunk>) = chunks.iter().partition(|c| dataset::is_validation(&c.file));
    // Donor pool: closed fills from training files.
    let mut pool: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for c in &train {
        let Ok(root) = dataset::extract_chunk(c) else { continue };
        root.walk(&mut Vec::new(), &mut |n, path| {
            if path.is_empty() {
                return;
            }
            let kind = KINDS.iter().find(|k| **k == n.kind).unwrap();
            let v = pool.entry(kind).or_default();
            let text = n.assemble();
            if v.len() < 5000 && text.len() <= max_chars && skeleton::fill_parses(&n.kind, &text) {
                v.push(text);
            }
        });
    }

    // 1. oracle swarm over every held-out chunk
    let mut trees: Vec<(&Chunk, Node)> = Vec::new();
    let (mut exact, mut parsed, mut stats) = (0, 0, sw::RunStats::default());
    let mut open_left = 0;
    for c in &val {
        let Ok(root) = dataset::extract_chunk(c) else { continue };
        let (slot, st) = sw::run_oracle(&root);
        stats.fills += st.fills;
        stats.retries += st.retries;
        stats.rejected += st.rejected;
        stats.gave_up += st.gave_up;
        open_left += slot.open_slots();
        match slot.assemble() {
            Some((text, spans)) => {
                if text == c.text {
                    exact += 1;
                }
                if syn::parse_file(&text).is_ok() {
                    parsed += 1;
                }
                assert!(spans.iter().all(|s| s.start <= s.end && s.end <= text.len()));
            }
            None => {}
        }
        trees.push((c, root));
    }
    println!(
        "oracle: {} held-out chunks, {exact} rebuilt exactly ({:.2}%), {parsed} parse, {} fills, {} retries, {} rejected, {} gave up, {open_left} open slots left",
        trees.len(),
        pct(exact, trees.len()),
        stats.fills,
        stats.retries,
        stats.rejected,
        stats.gave_up
    );

    // 2. cargo check ceiling on a sample
    let sample: Vec<&(&Chunk, Node)> = trees.iter().take(check_n).collect();
    let texts: Vec<String> = sample.iter().map(|(c, _)| c.text.clone()).collect();
    let t = std::time::Instant::now();
    let report = check::check_all(&scratch, &texts)?;
    let mut codes: BTreeMap<String, usize> = BTreeMap::new();
    for e in report.values().flatten() {
        *codes.entry(e.code.clone()).or_default() += 1;
    }
    let clean: Vec<&(&Chunk, Node)> =
        sample.iter().enumerate().filter(|(i, _)| !report.contains_key(i)).map(|(_, x)| *x).collect();
    println!(
        "cargo check on {} oracle chunks: {} clean ({:.1}%), {:.1}s; error codes: {}",
        sample.len(),
        clean.len(),
        pct(clean.len(), sample.len()),
        t.elapsed().as_secs_f64(),
        top_codes(&codes, 8)
    );

    // 3. one swapped hole per kind
    println!("\nkind        chunks  parse%  clean%  charged%  codes");
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let mut rows = Vec::new();
    for kind in KINDS {
        let Some(donors) = pool.get(kind).filter(|d| !d.is_empty()) else { continue };
        let mut cands: Vec<(String, Vec<sw::Span>, Vec<usize>)> = Vec::new();
        let (mut tried, mut unparsed) = (0, 0);
        for (c, root) in &clean {
            let mut paths: Vec<(Vec<usize>, String)> = Vec::new();
            root.walk(&mut Vec::new(), &mut |n, p| {
                if !p.is_empty() && n.kind == kind {
                    paths.push((p.to_vec(), n.assemble()));
                }
            });
            if paths.is_empty() {
                continue;
            }
            let (path, original) = paths.swap_remove(rng.below(paths.len()));
            let Some(text) = (0..8).map(|_| &donors[rng.below(donors.len())]).find(|d| **d != original) else {
                continue;
            };
            tried += 1;
            let mut filler = SwapOne {
                truth: Oracle(root),
                path: path.clone(),
                text: text.clone(),
                role: kind.to_string(),
            };
            let (slot, _) = sw::run(&mut filler, "item", &sw::Config::default());
            let Some((out, spans)) = slot.assemble() else { continue };
            let _ = c;
            if syn::parse_file(&out).is_err() {
                unparsed += 1;
                continue;
            }
            cands.push((out, spans, path));
        }
        let texts: Vec<String> = cands.iter().map(|c| c.0.clone()).collect();
        let report: Report = if texts.is_empty() { Report::new() } else { check::check_all(&scratch, &texts)? };
        let (mut errs, mut charged) = (0, 0);
        let mut codes: BTreeMap<String, usize> = BTreeMap::new();
        for (i, es) in &report {
            for e in es {
                errs += 1;
                *codes.entry(e.code.clone()).or_default() += 1;
                if sw::charge(&cands[*i].1, e.start, e.end).is_some_and(|s| s.path == cands[*i].2) {
                    charged += 1;
                }
            }
        }
        let n_clean = texts.len() - report.len();
        println!(
            "{kind:<11} {tried:>6}  {:>6.1}  {:>6.1}  {:>8.1}  {}",
            pct(tried - unparsed, tried),
            pct(n_clean, tried),
            pct(charged, errs),
            top_codes(&codes, 4)
        );
        rows.push(serde_json::json!({
            "kind": kind, "chunks": tried, "unparsed": unparsed, "clean": n_clean,
            "errors": errs, "charged_to_swap": charged, "codes": codes,
        }));
    }
    std::fs::create_dir_all("out/phase1")?;
    std::fs::write(
        "out/phase1/report.json",
        serde_json::to_string_pretty(&serde_json::json!({
            "held_out_chunks": trees.len(), "exact": exact, "parsed": parsed,
            "fills": stats.fills, "check_sample": sample.len(), "check_clean": clean.len(),
            "oracle_codes": codes, "swaps": rows,
        }))?,
    )?;
    Ok(())
}
