//! Phase 3/4: the swarm end to end. A doc comment goes to the planner, which writes the
//! root skeleton; the dispatcher routes every hole to the trained worker of its kind,
//! splices, checks each fill with `syn`, and the finished text goes through `cargo check`.
//!
//! cargo run --release --bin run_swarm -- --doc "Sums the visible widths of the rows."
//! cargo run --release --bin run_swarm -- [--held-out 50] [--seed 0] [--check]
//!   [--planner ../checkpoints/swarm/planner] [--workers ../checkpoints/swarm]
//!   [--roles roles/kinds.json] [--scratch out/scratch] [--limit-chunks N]
//!   [--verbose] (one line per hole) [--max-fills 300] [--no-check]
//!   [--retries 4] [--temp 0.8] [--top-k 40] (attempt 0 is greedy, retries sample)
//!
//! `--doc` prints one program. `--held-out N` runs N documented chunks from held-out files
//! (spread evenly), compares the plan with the real skeleton, `syn`-parses and (default)
//! `cargo check`s the results, and writes `out/phase3/samples.md` and `out/phase3/report.json`.
use anyhow::{Context, Result};
use std::{collections::BTreeMap, path::PathBuf, time::Instant};
use swarm::{
    check::{self, Report},
    dataset,
    planner::{Planner, ROOT, SwarmFiller, as_doc, doc_prefix, plan_parses, shape},
    roles::RoleSet,
    skeleton::{self, KINDS},
    swarm::{self as sw, Config},
    worker::Worker,
};

fn pct(a: usize, b: usize) -> f64 {
    if b == 0 { 0.0 } else { 100.0 * a as f64 / b as f64 }
}

struct Outcome {
    doc: String,
    truth: Option<String>,
    plan: Option<String>,
    program: Option<String>,
    stats: sw::RunStats,
    open: usize,
    by: BTreeMap<String, usize>,
}

/// Prints each hole as it is written (kind, depth, output size, time) so a long run can be watched.
struct Logged<F: sw::Filler> {
    inner: F,
    verbose: bool,
    n: usize,
}

impl<F: sw::Filler> sw::Filler for Logged<F> {
    fn fill(&mut self, req: &sw::Request) -> Option<String> {
        let t = Instant::now();
        let out = self.inner.fill(req);
        self.n += 1;
        if self.verbose {
            println!(
                "    hole {:>3} {:<9} depth {} ctx {:>4} chars -> {:>4} chars in {:>5.0} ms",
                self.n, req.kind, req.path.len(), req.context.len(), out.as_ref().map_or(0, |o| o.len()),
                t.elapsed().as_secs_f64() * 1000.0
            );
        }
        if self.verbose {
            if let Some(o) = &out {
                if !sw::check_fill(req.kind, o) {
                    let why = match sw::parse_fill(req.kind, o) {
                        Err(e) => e.to_string(),
                        Ok(_) => "syn rejects it".into(),
                    };
                    let clip: String = o.chars().take(160).collect::<String>().replace('\n', "\\n");
                    println!("        REJECTED ({why}): {clip}");
                }
            }
        }
        out
    }
    fn name(&self, kind: &str) -> &str {
        self.inner.name(kind)
    }
}

fn run_one(planner: &Planner, workers: &[Worker], doc: &str, verbose: bool, max_fills: usize, retries: usize) -> Outcome {
    let mut filler = Logged { inner: SwarmFiller::new(planner, doc.to_owned(), workers), verbose, n: 0 };
    let (slot, stats) = sw::run(&mut filler, ROOT, &Config { max_fills, max_retries: retries });
    let plan = slot.fill.as_ref().map(|f| {
        let mut t = String::new();
        for (i, seg) in f.segments.iter().enumerate() {
            t.push_str(seg);
            if let Some(c) = f.children.get(i) {
                t.push_str(&skeleton::marker(&c.kind));
            }
        }
        t
    });
    let mut by = BTreeMap::new();
    let program = slot.assemble().map(|(text, spans)| {
        for s in &spans {
            *by.entry(s.by.clone()).or_default() += 1;
        }
        text
    });
    Outcome { doc: doc.to_owned(), truth: None, plan, program, stats, open: slot.open_slots(), by }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let planner_dir = PathBuf::from(arg("--planner").unwrap_or("../checkpoints/swarm/planner".into()));
    let workers_dir = PathBuf::from(arg("--workers").unwrap_or("../checkpoints/swarm".into()));
    let roles = RoleSet::load(&PathBuf::from(arg("--roles").unwrap_or("roles/kinds.json".into())))?;
    let scratch = PathBuf::from(arg("--scratch").unwrap_or("out/scratch".into()));
    let tokenizer = PathBuf::from("tokenizer.json");
    let verbose = flag("--verbose");
    // a looping planner or worker can keep spawning holes; the library default is 4000
    let max_fills: usize = arg("--max-fills").map(|v| v.parse()).transpose()?.unwrap_or(300);
    // first try is greedy; retries sample (temperature, top-k) so they differ from it
    let retries: usize = arg("--retries").map(|v| v.parse()).transpose()?.unwrap_or(4);
    let temperature: f32 = arg("--temp").map(|v| v.parse()).transpose()?.unwrap_or(0.8);
    let top_k: usize = arg("--top-k").map(|v| v.parse()).transpose()?.unwrap_or(40);

    let mut planner = Planner::load(&planner_dir).with_context(|| format!("planner at {}", planner_dir.display()))?;
    planner.worker.temperature = temperature;
    planner.worker.top_k = top_k;
    let mut workers = Vec::new();
    for r in roles.roles.iter().filter(|r| !r.written) {
        let dir = workers_dir.join(&r.name);
        if dir.join("model.bin").exists() {
            let mut w = Worker::load(&dir)?;
            w.temperature = temperature;
            w.top_k = top_k;
            workers.push(w);
        } else {
            println!("no worker checkpoint for {} ({})", r.name, dir.display());
        }
    }
    println!(
        "planner {} epochs, held-out loss {:?}; workers: {}",
        planner.worker.meta.epochs_trained,
        planner.worker.meta.val_loss,
        workers.iter().map(|w| w.meta.role.as_str()).collect::<Vec<_>>().join(" ")
    );

    if let Some(doc) = arg("--doc") {
        let doc = as_doc(&doc);
        let o = run_one(&planner, &workers, &doc, true, max_fills, retries);
        println!("\n--- plan\n{}", o.plan.as_deref().unwrap_or("(no plan)"));
        println!(
            "\n--- program ({} fills, {} retries, {} rejected, {} gave up, {} holes left open)\n{}",
            o.stats.fills,
            o.stats.retries,
            o.stats.rejected,
            o.stats.gave_up,
            o.open,
            o.program.as_deref().unwrap_or("(incomplete: some holes were never filled)")
        );
        if let Some(p) = &o.program {
            println!("--- syn parse: {}", syn::parse_file(p).is_ok());
            if !flag("--no-check") {
                let r: Report = check::check_all(&scratch, &[p.clone()])?;
                match r.get(&0) {
                    None => println!("--- cargo check: clean"),
                    Some(es) => {
                        println!("--- cargo check: {} errors", es.len());
                        for e in es.iter().take(8) {
                            println!("    {}", e.code);
                        }
                    }
                }
            }
        }
        return Ok(());
    }

    // held-out documented chunks
    let n: usize = arg("--held-out").map(|v| v.parse()).transpose()?.unwrap_or(50);
    let seed: usize = arg("--seed").map(|v| v.parse()).transpose()?.unwrap_or(0);
    let chunks = dataset::load_chunks(
        &PathBuf::from("../training-cache/code.bin"),
        &tokenizer,
        arg("--limit-chunks").map(|v| v.parse()).transpose()?,
    )?;
    let docs: Vec<&dataset::Chunk> = chunks
        .iter()
        .filter(|c| dataset::is_validation(&c.file) && !doc_prefix(&c.text).is_empty() && dataset::extract_chunk(c).is_ok())
        .collect();
    let picks: Vec<&&dataset::Chunk> = if docs.len() <= n {
        docs.iter().collect()
    } else {
        (0..n).map(|i| &docs[(i * docs.len() / n + seed) % docs.len()]).collect()
    };
    println!("{} held-out documented chunks, running {}", docs.len(), picks.len());

    let t = Instant::now();
    let mut outs = Vec::new();
    for (i, c) in picks.iter().enumerate() {
        let doc = doc_prefix(&c.text);
        let t1 = Instant::now();
        let mut o = run_one(&planner, &workers, doc, verbose, max_fills, retries);
        println!(
            "[{:>3}/{}] {:>5.1}s  fills {:>3} retries {:>3} gave up {:>2} open {:>2}  {}",
            i + 1, picks.len(), t1.elapsed().as_secs_f64(), o.stats.fills, o.stats.retries, o.stats.gave_up, o.open,
            doc.lines().next().unwrap_or("").chars().take(70).collect::<String>()
        );
        let root = dataset::extract_chunk(c)?;
        o.truth = root.skeleton().strip_prefix(doc).map(str::to_owned);
        outs.push((o, c.text.clone()));
    }
    println!("swarm runs: {:.1}s ({:.1}s each)", t.elapsed().as_secs_f64(), t.elapsed().as_secs_f64() / picks.len().max(1) as f64);

    // planner alone
    let plan_body = |o: &Outcome| o.plan.as_deref().and_then(|p| p.strip_prefix(o.doc.as_str())).map(str::to_owned);
    let n_all = outs.len();
    let planned = outs.iter().filter(|(o, _)| o.plan.is_some()).count();
    let plan_ok = outs.iter().filter(|(o, _)| plan_body(o).is_some_and(|b| plan_parses(&o.doc, &b))).count();
    let exact = outs.iter().filter(|(o, _)| plan_body(o).is_some() && plan_body(o) == o.truth).count();
    let same_shape = outs
        .iter()
        .filter(|(o, _)| plan_body(o).is_some_and(|b| o.truth.as_deref().is_some_and(|t| shape(&b) == shape(t))))
        .count();
    println!("\nplanner alone, {n_all} docs");
    println!("  planned (root accepted by the dispatcher): {:.1}%", pct(planned, n_all));
    println!("  skeleton parses:                           {:.1}%", pct(plan_ok, n_all));
    println!("  same hole shape as the real code:          {:.1}%", pct(same_shape, n_all));
    println!("  exact skeleton:                            {:.1}%", pct(exact, n_all));

    // workers in the loop
    let complete: Vec<&(Outcome, String)> = outs.iter().filter(|(o, _)| o.program.is_some()).collect();
    let parse_ok = complete.iter().filter(|(o, _)| syn::parse_file(o.program.as_ref().unwrap()).is_ok()).count();
    let (mut fills, mut retries, mut rejected, mut gave_up, mut open) = (0, 0, 0, 0, 0);
    let mut by: BTreeMap<String, usize> = BTreeMap::new();
    for (o, _) in &outs {
        fills += o.stats.fills;
        retries += o.stats.retries;
        rejected += o.stats.rejected;
        gave_up += o.stats.gave_up;
        open += o.open;
        for (k, v) in &o.by {
            *by.entry(k.clone()).or_default() += v;
        }
    }
    println!("\nfull swarm, {n_all} docs");
    println!("  programs complete (every hole filled):     {:.1}% ({}/{n_all})", pct(complete.len(), n_all), complete.len());
    println!("  complete and parse as a file:              {:.1}% of complete", pct(parse_ok, complete.len()));
    println!("  fills {fills}, retries {retries}, rejected {rejected}, gave up {gave_up}, holes left open {open}");
    println!("  fills by role: {by:?}");

    let mut clean = 0;
    let mut codes: BTreeMap<String, usize> = BTreeMap::new();
    let mut checked = 0;
    let mut status: Vec<&str> = vec!["incomplete"; outs.len()];
    if !flag("--no-check") {
        let idx: Vec<usize> = (0..outs.len())
            .filter(|&i| outs[i].0.program.as_ref().is_some_and(|p| syn::parse_file(p).is_ok()))
            .collect();
        let texts: Vec<String> = idx.iter().map(|&i| outs[i].0.program.clone().unwrap()).collect();
        for &i in &idx {
            status[i] = "clean";
        }
        for (i, o) in outs.iter().enumerate() {
            if o.0.program.is_some() && !idx.contains(&i) {
                status[i] = "no-parse";
            }
        }
        if !texts.is_empty() {
            let t = Instant::now();
            let report: Report = check::check_all(&scratch, &texts)?;
            checked = texts.len();
            clean = checked - report.len();
            for (j, es) in &report {
                status[idx[*j]] = "errors";
                for e in es {
                    *codes.entry(e.code.clone()).or_default() += 1;
                }
            }
            println!("  cargo check on {checked} files: {clean} clean ({:.1}% of checked, {:.1}% of all) in {:.1}s",
                pct(clean, checked), pct(clean, n_all), t.elapsed().as_secs_f64());
            let mut top: Vec<_> = codes.iter().collect();
            top.sort_by(|a, b| b.1.cmp(a.1));
            println!("  error codes: {}", top.iter().take(8).map(|(c, k)| format!("{c}:{k}")).collect::<Vec<_>>().join(" "));
        }
    }

    std::fs::create_dir_all("out/phase3")?;
    let mut md = String::from("# Swarm samples (held-out documented chunks)\n\n");
    for (i, (o, original)) in outs.iter().enumerate() {
        md.push_str(&format!(
            "## {i}: {}\n\n```rust\n{}```\n\nplanned:\n\n```rust\n{}\n```\n\nprogram ({} fills, {} retries, {} gave up, {} open):\n\n```rust\n{}\n```\n\noriginal:\n\n```rust\n{}\n```\n\n",
            status[i], o.doc, o.plan.as_deref().unwrap_or("(none)"), o.stats.fills, o.stats.retries, o.stats.gave_up,
            o.open, o.program.as_deref().unwrap_or("(incomplete)"), original
        ));
    }
    std::fs::write("out/phase3/samples.md", md)?;
    std::fs::write(
        "out/phase3/report.json",
        serde_json::to_string_pretty(&serde_json::json!({
            "docs": n_all, "planned": planned, "plan_parses": plan_ok, "same_shape": same_shape, "plan_exact": exact,
            "complete": complete.len(), "parse": parse_ok, "checked": checked, "clean": clean,
            "fills": fills, "retries": retries, "rejected": rejected, "gave_up": gave_up, "open": open,
            "by_role": by, "codes": codes, "kinds": KINDS,
        }))?,
    )?;
    println!("\nwrote out/phase3/samples.md and report.json");
    Ok(())
}
