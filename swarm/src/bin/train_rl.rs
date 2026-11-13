//! Phase 6b: RL for the swarm against a frozen critic (docs/SWARM_RL.md).
//!
//! cargo run --release --bin train_rl -- --critic-hole ../checkpoints/swarm/critic_hole
//!   [--critic-program ../checkpoints/swarm/critic_program] [--workers ../checkpoints/swarm]
//!   [--planner ../checkpoints/swarm/planner] [--out ../checkpoints/swarm-rl] [--iters 100]
//!   [--docs 8] [--group 4] [--lr 2e-5] [--kl 0.02] [--temp 1.0] [--top-k 0] [--retries 2]
//!   [--max-fills 150] [--max-batch 64] [--min-traces 8] [--save-every 5] [--seed 1]
//!   [--only block,expr,...] (train just these roles; others stay frozen, `planner` included)
//!   [--w-critic 1] [--w-syntax 1] [--w-compile 0.5] [--w-program 0.5] [--w-clean 0.5]
//!   [--lambda 0.5] [--no-check] (skip cargo check) [--resume] [--limit-chunks N]
//!
//! Every iteration: `docs` documented training chunks, `group` sampled rollouts of the whole
//! swarm each; judge them (see `rl.rs`); each role takes one policy-gradient step on the
//! traces it wrote. Held-out evaluation is `run_swarm --planner <out>/planner --workers <out>`,
//! which also reports `cargo check` and the critic, so reward hacking shows up there.
//! Checkpoints go to `--out/<role>/`; the sources are never touched. `rl.csv` logs each iteration.
use anyhow::{Context, Result, ensure};
use burn::{
    grad_clipping::GradientClippingConfig,
    optim::AdamWConfig,
    prelude::*,
};
use std::{collections::BTreeMap, io::Write, path::{Path, PathBuf}, time::Instant};
use swarm::{
    check::{self, Report},
    critic::{Critic, Rng},
    dataset::{self, Chunk},
    planner::{Planner, ROOT, doc_prefix},
    rl::{self, Policy, RlFiller, Trace, Weights, advantages, update, value_rollout},
    roles::RoleSet,
    swarm::{self as sw, Config},
    worker::{Infer, Rows, RoleMeta, Train, Worker, save_role},
};
use yumon_pet::brain::decoder_model::YumonDecBrain;

/// Copy a role checkpoint into `dst` with dropout switched off, so the policy-gradient
/// forward matches the (dropout-free) rollouts.
fn stage(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for e in std::fs::read_dir(src)? {
        let e = e?;
        if e.file_type()?.is_file() {
            std::fs::copy(e.path(), dst.join(e.file_name()))?;
        }
    }
    let p = dst.join("metadata.json");
    let mut v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p)?)?;
    v["dropout_rate"] = serde_json::json!(0.0);
    std::fs::write(&p, serde_json::to_string_pretty(&v)?)?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let num = |name: &str, d: usize| -> Result<usize> { Ok(arg(name).map(|v| v.parse()).transpose()?.unwrap_or(d)) };
    let fnum = |name: &str, d: f64| -> Result<f64> { Ok(arg(name).map(|v| v.parse()).transpose()?.unwrap_or(d)) };
    let iters = num("--iters", 100)?;
    let n_docs = num("--docs", 8)?;
    let group = num("--group", 4)?.max(2);
    let max_fills = num("--max-fills", 150)?;
    let retries = num("--retries", 2)?;
    let max_batch = num("--max-batch", 64)?;
    let min_traces = num("--min-traces", 8)?;
    let save_every = num("--save-every", 5)?.max(1);
    let seed = num("--seed", 1)? as u64;
    let (lr, beta, temp) = (fnum("--lr", 2e-5)?, fnum("--kl", 0.02)? as f32, fnum("--temp", 1.0)? as f32);
    let top_k = num("--top-k", 0)?;
    let w = Weights {
        critic: fnum("--w-critic", 1.0)? as f32,
        syntax: fnum("--w-syntax", 1.0)? as f32,
        compile: fnum("--w-compile", 0.5)? as f32,
        program: fnum("--w-program", 0.5)? as f32,
        clean: fnum("--w-clean", 0.5)? as f32,
        lambda: fnum("--lambda", 0.5)? as f32,
    };
    let tokenizer = PathBuf::from("tokenizer.json");
    let workers_dir = PathBuf::from(arg("--workers").unwrap_or("../checkpoints/swarm".into()));
    let planner_dir = PathBuf::from(arg("--planner").unwrap_or("../checkpoints/swarm/planner".into()));
    let out_dir = PathBuf::from(arg("--out").unwrap_or("../checkpoints/swarm-rl".into()));
    let scratch = PathBuf::from(arg("--scratch").unwrap_or("out/scratch".into()));
    let roles = RoleSet::load(&PathBuf::from(arg("--roles").unwrap_or("roles/kinds.json".into())))?;
    let only: Option<Vec<String>> = arg("--only").map(|s| s.split(',').map(str::to_owned).collect());
    let hole_critic = Critic::load(&PathBuf::from(arg("--critic-hole").context("--critic-hole DIR is required")?))?;
    let prog_critic = arg("--critic-program").map(|p| Critic::load(&PathBuf::from(p))).transpose()?;
    println!(
        "critics: hole level held-out accuracy {:?}; program level {}",
        hole_critic.meta.by_neg,
        prog_critic.as_ref().map_or("none".into(), |c| format!("{:?}", c.meta.by_neg))
    );

    // roles: sources -> staged copies in out_dir (resume continues from them)
    let mut sources: Vec<(String, PathBuf)> = vec![("planner".into(), planner_dir)];
    for r in roles.roles.iter().filter(|r| !r.written) {
        let d = workers_dir.join(&r.name);
        if d.join("model.bin").exists() {
            sources.push((r.name.clone(), d));
        } else {
            println!("no worker checkpoint for {} ({})", r.name, d.display());
        }
    }
    let device: <Train as Backend>::Device = Default::default();
    let mut planner: Option<Planner> = None;
    let mut workers: Vec<Worker> = Vec::new();
    let mut policies: Vec<Policy> = Vec::new();
    for (name, src) in &sources {
        let dst = out_dir.join(name);
        if !(flag("--resume") && dst.join("model.bin").exists()) {
            stage(src, &dst)?;
            // staged copies keep the original role.json
        }
        let mut worker = Worker::load(&dst)?;
        let train_it = only.as_ref().is_none_or(|o| o.contains(name));
        if train_it {
            let (model, _, cfg) = YumonDecBrain::<Train>::load(dst.to_str().unwrap(), &device)?;
            let rows = Rows { ids: worker.meta.rows.clone() };
            let opt = AdamWConfig::new()
                .with_epsilon(1e-7)
                .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
                .with_weight_decay(0.0)
                .init();
            let rows_i32: Vec<i32> = rows.ids.iter().map(|&r| r as i32).collect();
            policies.push(Policy {
                role: name.clone(),
                meta: worker.meta.clone(),
                reference: worker.model().clone(),
                rows_train: Tensor::<Train, 1, Int>::from_ints(rows_i32.as_slice(), &device),
                rows_infer: Tensor::<Infer, 1, Int>::from_ints(rows_i32.as_slice(), &Default::default()),
                pos: rows.position(),
                rows,
                model,
                opt,
            });
            let _ = cfg;
        }
        worker.temperature = temp;
        if name == "planner" {
            planner = Some(Planner { worker });
        } else {
            workers.push(worker);
        }
    }
    let mut planner = planner.context("no planner checkpoint")?;
    println!(
        "training {} roles: {}; frozen: {}",
        policies.len(),
        policies.iter().map(|p| p.role.as_str()).collect::<Vec<_>>().join(" "),
        sources.iter().map(|s| s.0.as_str()).filter(|n| !policies.iter().any(|p| p.role == *n)).collect::<Vec<_>>().join(" ")
    );
    ensure!(!policies.is_empty(), "no role to train (check --only)");

    // documents: documented chunks of training files only
    let chunks = dataset::load_chunks(
        &PathBuf::from("../training-cache/code.bin"),
        &tokenizer,
        arg("--limit-chunks").map(|v| v.parse()).transpose()?,
    )?;
    let mut docs: Vec<&Chunk> = chunks
        .iter()
        .filter(|c| !dataset::is_validation(&c.file) && !doc_prefix(&c.text).is_empty() && dataset::extract_chunk(c).is_ok())
        .collect();
    Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1).shuffle(&mut docs);
    println!("{} documented training chunks; {n_docs} docs x {group} rollouts per iteration", docs.len());
    ensure!(docs.len() >= n_docs, "not enough documents");

    std::fs::create_dir_all(&out_dir)?;
    let csv = out_dir.join("rl.csv");
    if !csv.exists() {
        std::fs::write(&csv, "iter,secs,traces,accepted,complete,clean,hole_critic,prog_critic,mean_value,kl,loss\n")?;
    }
    let started = Instant::now();
    let mut cursor = 0usize;
    let mut rng = Rng(seed | 1);
    for iter in 1..=iters {
        let t0 = Instant::now();
        // 1. rollouts
        let mut rollouts: Vec<(usize, String, Vec<Trace>, Option<(String, Vec<sw::Span>)>, bool)> = Vec::new();
        for d in 0..n_docs {
            let chunk = docs[(cursor + d) % docs.len()];
            let doc = doc_prefix(&chunk.text).to_owned();
            for g in 0..group {
                let mut filler = RlFiller {
                    planner: &planner,
                    doc: doc.clone(),
                    workers: &workers,
                    doc_index: d,
                    rollout: g,
                    salt: rng.next(),
                    temperature: temp,
                    top_k,
                    traces: Vec::new(),
                };
                let (slot, _) = sw::run(&mut filler, ROOT, &Config { max_fills, max_retries: retries });
                filler.finish();
                let program = slot.assemble();
                let parses = program.as_ref().is_some_and(|(t, _)| syn::parse_file(t).is_ok());
                rollouts.push((d, doc.clone(), filler.traces, program, parses));
            }
            println!("  doc {}/{n_docs} rolled out ({:.0}s)", d + 1, t0.elapsed().as_secs_f64());
        }
        cursor += n_docs;

        // 2. cargo check on the programs that parse
        let mut reports: Report = Report::new();
        let idx: Vec<usize> = (0..rollouts.len()).filter(|&i| rollouts[i].4).collect();
        if !flag("--no-check") && !idx.is_empty() {
            let texts: Vec<String> = idx.iter().map(|&i| rollouts[i].3.as_ref().unwrap().0.clone()).collect();
            let r = check::check_all(&scratch, &texts)?;
            for (j, es) in r {
                reports.insert(idx[j], es);
            }
        }
        let checked = !flag("--no-check");

        // 3. values and advantages
        let (mut acc, mut total, mut complete, mut clean, mut hs, mut hn, mut ps, mut pn) = (0, 0, 0, 0, 0.0f32, 0usize, 0.0f32, 0usize);
        let mut role_names: Vec<String> = Vec::new();
        let mut triples: Vec<(usize, usize, f32)> = Vec::new();
        let mut owners: Vec<(usize, usize)> = Vec::new(); // (rollout, trace) per triple
        let mut value_sum = 0.0f32;
        for (ri, (d, doc, traces, program, parses)) in rollouts.iter().enumerate() {
            let errs: Option<&[check::CheckError]> = if checked && *parses { Some(reports.get(&ri).map_or(&[][..], |v| v.as_slice())) } else { None };
            let j = value_rollout(traces, doc, &hole_critic, prog_critic.as_ref(), program.as_ref(), *parses, errs, &w);
            total += traces.len();
            acc += traces.iter().filter(|t| t.accepted).count();
            complete += j.complete as usize;
            clean += (j.errors == Some(0)) as usize;
            hs += j.hole_sum;
            hn += j.hole_n;
            if let Some(p) = j.program_r {
                ps += p;
                pn += 1;
            }
            for (ti, t) in traces.iter().enumerate() {
                let r = match role_names.iter().position(|n| *n == t.role) {
                    Some(r) => r,
                    None => {
                        role_names.push(t.role.clone());
                        role_names.len() - 1
                    }
                };
                triples.push((*d, r, j.values[ti]));
                owners.push((ri, ti));
                value_sum += j.values[ti];
            }
        }
        let adv = advantages(&triples);

        // 4. one step per role
        let (mut kl_sum, mut loss_sum, mut steps) = (0.0f32, 0.0f32, 0usize);
        let mut report: BTreeMap<String, String> = BTreeMap::new();
        for p in policies.iter_mut() {
            let Some(r) = role_names.iter().position(|n| *n == p.role) else { continue };
            let mut mine: Vec<(rl::Trace, f32)> = triples
                .iter()
                .zip(&owners)
                .zip(&adv)
                .filter(|((t, _), _)| t.1 == r)
                .map(|((_, o), a)| (rollouts[o.0].2[o.1].clone(), *a))
                .collect();
            if mine.len() < min_traces {
                continue;
            }
            rng.shuffle(&mut mine);
            mine.truncate(max_batch);
            let items: Vec<_> = mine.iter().map(|(t, a)| (t.sample(), *a)).collect();
            let s = update(p, &items, temp, beta, lr)?;
            kl_sum += s.kl;
            loss_sum += s.loss;
            steps += 1;
            report.insert(p.role.clone(), format!("n {} kl {:.4}", s.n, s.kl));
        }
        // 5. rollouts use the new weights
        for p in &policies {
            let m = p.snapshot();
            if p.role == "planner" {
                planner.worker.set_model(m);
            } else if let Some(w) = workers.iter_mut().find(|w| w.meta.role == p.role) {
                w.set_model(m);
            }
        }
        let nr = rollouts.len();
        let secs = t0.elapsed().as_secs_f64();
        println!(
            "iter {iter}/{iters} {secs:.0}s | traces {total}, accepted {:.0}% | complete {complete}/{nr}, clean {clean}/{nr} | hole critic {:.2}, program critic {:.2} | value {:.3} | kl {:.4} | {}",
            100.0 * acc as f32 / total.max(1) as f32,
            hs / hn.max(1) as f32,
            ps / pn.max(1) as f32,
            value_sum / total.max(1) as f32,
            kl_sum / steps.max(1) as f32,
            report.iter().map(|(k, v)| format!("{k}: {v}")).collect::<Vec<_>>().join(", ")
        );
        let mut f = std::fs::OpenOptions::new().append(true).open(&csv)?;
        writeln!(
            f, "{iter},{:.0},{total},{acc},{complete},{clean},{:.4},{:.4},{:.4},{:.5},{:.5}",
            started.elapsed().as_secs_f64(), hs / hn.max(1) as f32, ps / pn.max(1) as f32,
            value_sum / total.max(1) as f32, kl_sum / steps.max(1) as f32, loss_sum / steps.max(1) as f32
        )?;
        if iter % save_every == 0 || iter == iters {
            for p in &policies {
                let infer = p.snapshot();
                let meta = RoleMeta { epochs_trained: p.meta.epochs_trained, ..p.meta.clone() };
                save_role(&infer, &out_dir.join(&p.role), &tokenizer, &meta, 0, 0.0)?;
            }
            println!("  saved {} roles to {}", policies.len(), out_dir.display());
        }
    }
    Ok(())
}
