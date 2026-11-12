//! Phase 2 evaluation of a trained role, on held-out files.
//!
//! 1. Greedy fills for held-out instances of the role's kinds: exact match against the
//!    ground-truth fill and `syn` parse rate, per kind.
//! 2. The role swapped into the oracle swarm: in held-out chunks that `cargo check` clean
//!    with ground truth, one hole of the role's kinds is written by the worker and
//!    everything else stays perfect. Reports parse and `cargo check` rates, the share
//!    that are exact (clean by construction), the clean rate among fills that differ
//!    from the truth, error codes, and how often an error is charged to the swap.
//!
//! cargo run --release --bin eval_role -- --role block [--roles roles/kinds.json]
//!   [--ckpt ../checkpoints/swarm/<role>] [--samples 300] [--check 1500] [--limit-chunks N]
use anyhow::{Context, Result};
use std::{collections::BTreeMap, path::PathBuf};
use swarm::{
    check::{self, Report},
    dataset::{self, Chunk},
    roles::RoleSet,
    skeleton::{KINDS, Node},
    swarm::{self as sw, Oracle, SwapOne, check_fill},
    worker::{self, RoleData, Worker},
};

fn pct(a: usize, b: usize) -> f64 {
    if b == 0 { 0.0 } else { 100.0 * a as f64 / b as f64 }
}

struct Rng(u64);
impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

fn node_at<'a>(root: &'a Node, path: &[usize]) -> &'a Node {
    path.iter().fold(root, |n, &i| &n.children[i])
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let num = |name: &str, d: usize| -> Result<usize> { Ok(arg(name).map(|v| v.parse()).transpose()?.unwrap_or(d)) };
    let role_name = arg("--role").context("--role NAME is required")?;
    let roles = RoleSet::load(&PathBuf::from(arg("--roles").unwrap_or("roles/kinds.json".into())))?;
    let role = roles.role(&role_name).with_context(|| format!("no role {role_name}"))?.clone();
    let ckpt = PathBuf::from(arg("--ckpt").unwrap_or(format!("../checkpoints/swarm/{role_name}")));
    let tokenizer = PathBuf::from(arg("--tokenizer").unwrap_or("tokenizer.json".into()));
    let cache = PathBuf::from(arg("--cache").unwrap_or("../training-cache/code.bin".into()));
    let scratch = PathBuf::from(arg("--scratch").unwrap_or("out/scratch".into()));
    let samples = num("--samples", 300)?;
    let check_n = num("--check", 1500)?;
    let tok = worker::load_tokenizer(&tokenizer)?;
    let w = Worker::load(&ckpt)?;
    println!(
        "worker {} ({} params, {} epochs, held-out loss {:?}, top-1 {:?}, top-3 {:?})",
        w.meta.role, w.meta.params, w.meta.epochs_trained, w.meta.val_loss, w.meta.val_top1, w.meta.val_top3
    );

    let chunks = dataset::load_chunks(&cache, &tokenizer, arg("--limit-chunks").map(|v| v.parse()).transpose()?)?;

    // 1. greedy fills on held-out instances
    let data = RoleData::build(&chunks, &tok, &role)?;
    if data.rows.ids != w.meta.rows {
        println!("warning: rows rebuilt from the cache differ from the checkpoint's (cache or tokenizer changed)");
    }
    let n = data.val.len();
    let picks: Vec<&worker::Sample> =
        if n <= samples { data.val.iter().collect() } else { (0..samples).map(|i| &data.val[i * n / samples]).collect() };
    let mut per: BTreeMap<&str, [usize; 4]> = BTreeMap::new();
    let t = std::time::Instant::now();
    for s in &picks {
        let kind = KINDS[s.kind as usize];
        let out = w.fill_ids(&s.prompt()).unwrap_or_default();
        let truth: Vec<u32> = s.fill().iter().map(|&x| x as u32).collect();
        let text = w.decode(&out);
        let e = per.entry(kind).or_default();
        e[0] += 1;
        e[1] += (out == truth) as usize;
        e[2] += check_fill(kind, &text) as usize;
        e[3] += s.in_role as usize;
    }
    println!("\ngreedy fills, {} of {n} held-out instances ({:.1}s)", picks.len(), t.elapsed().as_secs_f64());
    println!("kind        n   exact%  parse%  truth-in-rows%");
    for (k, e) in &per {
        println!("{k:<10} {:>4}  {:>6.1}  {:>6.1}  {:>6.1}", e[0], pct(e[1], e[0]), pct(e[2], e[0]), pct(e[3], e[0]));
    }

    // 2. swap into the oracle swarm
    let val: Vec<&Chunk> = chunks.iter().filter(|c| dataset::is_validation(&c.file)).collect();
    let trees: Vec<(&Chunk, Node)> =
        val.iter().take(check_n).filter_map(|c| dataset::extract_chunk(c).ok().map(|r| (*c, r))).collect();
    let texts: Vec<String> = trees.iter().map(|(c, _)| c.text.clone()).collect();
    let t = std::time::Instant::now();
    let base: Report = check::check_all(&scratch, &texts)?;
    let clean: Vec<&(&Chunk, Node)> =
        trees.iter().enumerate().filter(|(i, _)| !base.contains_key(i)).map(|(_, x)| x).collect();
    println!("\noracle check: {} of {} held-out chunks compile ({:.1}s)", clean.len(), trees.len(), t.elapsed().as_secs_f64());

    let mut rng = Rng(0x9e3779b97f4a7c15);
    println!("\nswap into oracle swarm (one hole per clean chunk and kind)");
    println!("kind        holes  reshaped%  parse%  exact%  clean%  clean%-if-different  charged%  codes");
    println!("(parse%, exact%, clean% are of holes whose sub-hole kinds match the truth; reshaped% are not checked)");
    let mut rows_json = Vec::new();
    for kind in role.kinds.iter().map(String::as_str) {
        let mut cands: Vec<(String, Vec<sw::Span>, Vec<usize>, bool)> = Vec::new();
        let (mut tried, mut reshaped) = (0, 0);
        for (_, root) in &clean {
            let mut paths: Vec<Vec<usize>> = Vec::new();
            root.walk(&mut Vec::new(), &mut |n, p| {
                if !p.is_empty() && n.kind == kind {
                    paths.push(p.to_vec());
                }
            });
            if paths.is_empty() {
                continue;
            }
            let path = paths.swap_remove(rng.below(paths.len()));
            tried += 1;
            let truth_node = node_at(root, &path);
            let Some(text) = w.fill(&worker::context_of(root, &path)) else { continue };
            // The oracle fills sub-holes by path, so it can only finish a fill whose
            // holes have the same kinds as the truth's. Others are counted, not checked.
            let same_shape = sw::parse_fill(kind, &text).is_ok_and(|f| {
                f.children.iter().map(|c| &c.kind).eq(truth_node.children.iter().map(|c| &c.kind))
            });
            if !same_shape {
                reshaped += 1;
                continue;
            }
            let exact = text == truth_node.skeleton();
            let mut filler = SwapOne { truth: Oracle(root), path: path.clone(), text, role: kind.to_string() };
            let (slot, _) = sw::run(&mut filler, "item", &sw::Config::default());
            let Some((out, spans)) = slot.assemble() else { continue };
            if syn::parse_file(&out).is_err() {
                continue;
            }
            cands.push((out, spans, path, exact));
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
        let shaped = tried - reshaped;
        let exact = cands.iter().filter(|c| c.3).count();
        let n_clean = texts.len() - report.len();
        let diff = cands.len() - exact;
        let diff_clean = cands.iter().enumerate().filter(|(i, c)| !c.3 && !report.contains_key(i)).count();
        let mut top: Vec<_> = codes.iter().collect();
        top.sort_by(|a, b| b.1.cmp(a.1));
        let top = top.iter().take(4).map(|(c, k)| format!("{c}:{k}")).collect::<Vec<_>>().join(" ");
        println!(
            "{kind:<10} {tried:>6}  {:>9.1}  {:>6.1}  {:>6.1}  {:>6.1}  {:>19.1}  {:>8.1}  {top}",
            pct(reshaped, tried),
            pct(cands.len(), shaped),
            pct(exact, shaped),
            pct(n_clean, shaped),
            pct(diff_clean, diff),
            pct(charged, errs)
        );
        rows_json.push(serde_json::json!({
            "kind": kind, "holes": tried, "reshaped": reshaped, "parsed": cands.len(), "exact": exact, "clean": n_clean,
            "different": diff, "different_clean": diff_clean, "errors": errs, "charged": charged, "codes": codes,
        }));
    }
    std::fs::create_dir_all("out/phase2")?;
    std::fs::write(
        format!("out/phase2/{role_name}.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "role": role_name, "meta": w.meta,
            "greedy": per.iter().map(|(k, e)| serde_json::json!({
                "kind": k, "n": e[0], "exact": e[1], "parsed": e[2], "truth_in_rows": e[3]})).collect::<Vec<_>>(),
            "clean_chunks": clean.len(), "checked_chunks": trees.len(), "swaps": rows_json,
        }))?,
    )?;
    println!("\nwrote out/phase2/{role_name}.json");
    Ok(())
}
