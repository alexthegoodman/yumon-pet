//! Phase 6a: hard negatives for the critic, written by the current swarm.
//!
//! cargo run --release --bin gen_negatives -- --level hole|program [--n 20000]
//!   [--out out/critic/neg_<level>.jsonl] [--workers ../checkpoints/swarm]
//!   [--planner ../checkpoints/swarm/planner] [--temp 1.0] [--top-k 40] [--limit-chunks N]
//!
//! hole:    for random instances of every kind, the owning worker samples a fill for the real
//!          context; fills that `syn` accepts and differ from the truth are kept.
//! program: the planner and workers write a whole program for a documented chunk's doc comment
//!          (greedy first tries, like `run_swarm`); complete programs that differ are kept.
//!
//! Files of both splits are used; `train_critic` splits by file, so held-out negatives only
//! ever score held-out files. The output is a JSON line per negative (`critic::NegRecord`).
use anyhow::{Context, Result};
use std::{io::Write, path::PathBuf, time::Instant};
use swarm::{
    critic::{NegRecord, Rng},
    dataset,
    planner::{Planner, ROOT, SwarmFiller, doc_prefix},
    roles::RoleSet,
    skeleton::KINDS,
    swarm::{self as sw, Config, check_fill},
    worker::{Worker, encode, load_tokenizer},
};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let level = arg("--level").context("--level hole|program is required")?;
    let n: usize = arg("--n").map(|v| v.parse()).transpose()?.unwrap_or(if level == "hole" { 20000 } else { 3000 });
    let temp: f32 = arg("--temp").map(|v| v.parse()).transpose()?.unwrap_or(1.0);
    let top_k: usize = arg("--top-k").map(|v| v.parse()).transpose()?.unwrap_or(40);
    let workers_dir = PathBuf::from(arg("--workers").unwrap_or("../checkpoints/swarm".into()));
    let planner_dir = PathBuf::from(arg("--planner").unwrap_or("../checkpoints/swarm/planner".into()));
    let roles = RoleSet::load(&PathBuf::from(arg("--roles").unwrap_or("roles/kinds.json".into())))?;
    let tokenizer = PathBuf::from("tokenizer.json");
    let out_path = PathBuf::from(arg("--out").unwrap_or(format!("out/critic/neg_{level}.jsonl")));
    if let Some(d) = out_path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let tok = load_tokenizer(&tokenizer)?;
    let _ = encode(&tok, "x");

    let mut workers = Vec::new();
    for r in roles.roles.iter().filter(|r| !r.written) {
        let dir = workers_dir.join(&r.name);
        if dir.join("model.bin").exists() {
            workers.push(Worker::load(&dir)?);
        } else {
            println!("no worker checkpoint for {} ({})", r.name, dir.display());
        }
    }
    let chunks = dataset::load_chunks(
        &PathBuf::from("../training-cache/code.bin"),
        &tokenizer,
        arg("--limit-chunks").map(|v| v.parse()).transpose()?,
    )?;
    let mut order: Vec<usize> = (0..chunks.len()).collect();
    Rng(0x9e37_79b9_7f4a_7c15).shuffle(&mut order);
    let mut out = std::io::BufWriter::new(std::fs::File::create(&out_path)?);
    let t = Instant::now();
    let mut written = 0usize;
    let mut tried = 0usize;

    if level == "hole" {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        for &ci in &order {
            if written >= n {
                break;
            }
            let chunk = &chunks[ci];
            let Ok(root) = dataset::extract_chunk(chunk) else { continue };
            let insts = dataset::instances(&root);
            if insts.is_empty() {
                continue;
            }
            // two instances per chunk keeps the kinds spread over many files
            for _ in 0..2 {
                let inst = &insts[rng.below(insts.len())];
                let Some(w) = workers.iter().find(|w| w.handles(&inst.kind)) else { continue };
                let prompt = w.prompt_for(&inst.context);
                tried += 1;
                let Some((ids, _)) = w.rollout_ids(&prompt, rng.next(), temp, top_k) else { continue };
                let bad = w.decode(&ids);
                if bad == inst.target || !check_fill(&inst.kind, &bad) {
                    continue;
                }
                let rec = NegRecord {
                    level: "hole".into(), kind: inst.kind.clone(), file: chunk.file.clone(), context: inst.context.clone(),
                    truth: inst.target.clone(), bad,
                };
                writeln!(out, "{}", serde_json::to_string(&rec)?)?;
                written += 1;
            }
            if tried % 500 < 2 {
                println!("{written}/{n} negatives from {tried} samples ({:.0}s)", t.elapsed().as_secs_f64());
            }
        }
    } else {
        let planner = Planner::load(&planner_dir).with_context(|| format!("planner at {}", planner_dir.display()))?;
        for &ci in &order {
            if written >= n {
                break;
            }
            let chunk = &chunks[ci];
            let doc = doc_prefix(&chunk.text);
            if doc.is_empty() || dataset::extract_chunk(chunk).is_err() {
                continue;
            }
            tried += 1;
            let mut filler = SwarmFiller::new(&planner, doc.to_owned(), &workers);
            let (slot, _) = sw::run(&mut filler, ROOT, &Config { max_fills: 300, max_retries: 2 });
            let Some((text, _)) = slot.assemble() else { continue };
            let body = text.strip_prefix(doc).unwrap_or(&text).to_owned();
            let truth = chunk.text[doc.len()..].to_owned();
            if body == truth || syn::parse_file(&text).is_err() {
                continue;
            }
            let rec = NegRecord {
                level: "program".into(), kind: "program".into(), file: chunk.file.clone(), context: doc.to_owned(), truth, bad: body,
            };
            writeln!(out, "{}", serde_json::to_string(&rec)?)?;
            written += 1;
            if written % 25 == 0 {
                println!("{written}/{n} programs from {tried} docs ({:.0}s)", t.elapsed().as_secs_f64());
                out.flush()?;
            }
        }
    }
    out.flush()?;
    println!("wrote {written} {level} negatives to {} ({tried} tried, {:.0}s); kinds: {:?}", out_path.display(), t.elapsed().as_secs_f64(), KINDS);
    Ok(())
}
