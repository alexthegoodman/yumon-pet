//! Phase 2: fine-tune one role (a separate small decoder per role).
//!
//! The role starts from the set's base checkpoint (sugar-maniac, vocab grown to 2,073
//! rows) or from its own `model` shape in the role file, reads the whole vocabulary and
//! writes only its own rows of the tied table. Loss is on the fill tokens only.
//!
//! cargo run --release --bin train_role -- --role block [--roles roles/kinds.json]
//!   [--out ../checkpoints/swarm] [--epochs N] [--batch N] [--lr F] [--last-lr F]
//!   [--max-samples N] [--limit-chunks N] [--eval-samples 64] [--val-samples 2000]
//!   [--log-every 25] [--steps N] [--resume] [--check]
//!
//! `--check` prepares the data and builds the model, prints counts, trains nothing.
//! `--steps N` stops after N optimiser steps (smoke runs). Progress goes to stdout;
//! `train.csv` and `val.csv` in the role directory are appended as it goes. The best
//! validation epoch is what `<out>/<role>/` holds (and what `--resume` continues from).
use anyhow::{Context, Result, ensure};
use burn::{
    grad_clipping::GradientClippingConfig,
    module::AutodiffModule,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
};
use std::{io::Write, path::PathBuf, time::Instant};
use swarm::{
    dataset,
    roles::RoleSet,
    swarm::check_fill,
    worker::{self, Batch, Infer, Rows, RoleData, RoleMeta, Sample, Train, make_batch},
};
use yumon_pet::brain::decoder_model::YumonDecBrain;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, (self.next() % (i as u64 + 1)) as usize);
        }
    }
}

fn append(path: &PathBuf, header: &str, line: &str) -> Result<()> {
    let new = !path.exists();
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    if new {
        writeln!(f, "{header}")?;
    }
    writeln!(f, "{line}")?;
    Ok(())
}

fn lr_at(step: usize, total: usize, first: f64, last: f64) -> f64 {
    let warm = 100.min(total / 10).max(1);
    if step < warm {
        return first * (step + 1) as f64 / warm as f64;
    }
    let t = (step - warm) as f64 / (total - warm).max(1) as f64;
    last + (first - last) * 0.5 * (1.0 + (std::f64::consts::PI * t).cos())
}

/// Batches for one epoch: shuffled, length-bucketed in windows of 64 batches.
fn epoch_batches(n: usize, batch: usize, cap: Option<usize>, data: &[Sample], rng: &mut Rng) -> Vec<Vec<usize>> {
    let mut idx: Vec<usize> = (0..n).collect();
    rng.shuffle(&mut idx);
    if let Some(c) = cap {
        idx.truncate(c);
    }
    let mut batches = Vec::new();
    for window in idx.chunks(batch * 64) {
        let mut w = window.to_vec();
        w.sort_by_key(|&i| data[i].ids.len());
        batches.extend(w.chunks(batch).map(|c| c.to_vec()));
    }
    rng.shuffle(&mut batches);
    batches
}

struct Val {
    loss: f32,
    top1: f32,
    top3: f32,
    samples: usize,
}

fn validate(model: &YumonDecBrain<Infer>, val: &[&Sample], rows: &Rows, batch: usize) -> Val {
    let device: <Infer as Backend>::Device = Default::default();
    let pos = rows.position();
    let rows_t = Tensor::<Infer, 1, Int>::from_ints(
        rows.ids.iter().map(|&r| r as i32).collect::<Vec<_>>().as_slice(),
        &device,
    );
    let (mut loss, mut t1, mut t3, mut n) = (0.0f64, 0.0f64, 0.0f64, 0usize);
    for chunk in val.chunks(batch) {
        let b = make_batch(chunk, &pos);
        if b.supervised == 0 {
            continue;
        }
        let s = worker::score(model, &b, &rows_t, &device);
        let v = |t: Tensor<Infer, 1>| t.into_data().convert::<f32>().to_vec::<f32>().unwrap()[0] as f64;
        loss += v(s.loss) * s.supervised as f64;
        t1 += v(s.top1);
        t3 += v(s.top3);
        n += s.supervised;
    }
    let d = n.max(1) as f64;
    Val { loss: (loss / d) as f32, top1: (t1 / d) as f32, top3: (t3 / d) as f32, samples: val.len() }
}

/// Greedy fills for `samples`: exact match and `syn` parse rates, and a couple of examples.
fn generate(model: &YumonDecBrain<Infer>, samples: &[&Sample], rows: &Rows, tok: &tokenizers::Tokenizer,
    kinds: &[String], show: usize) -> (f32, f32) {
    let device: <Infer as Backend>::Device = Default::default();
    let (mut exact, mut parsed) = (0, 0);
    for (i, s) in samples.iter().enumerate() {
        let prompt = s.prompt();
        let max_new = worker::CONTEXT - prompt.len() - 1;
        let out = model.greedy_rows(&prompt, &rows.ids, worker::EOS, max_new, &device);
        let truth: Vec<u32> = s.fill().iter().map(|&t| t as u32).collect();
        let text = tok.decode(&out, false).unwrap_or_default();
        let kind = &kinds[0];
        let kind = swarm::skeleton::KINDS.get(s.kind as usize).copied().unwrap_or(kind.as_str());
        exact += (out == truth) as usize;
        parsed += check_fill(kind, &text) as usize;
        if i < show {
            let want = tok.decode(&truth, false).unwrap_or_default();
            let clip = |t: &str| t.chars().take(110).collect::<String>().replace('\n', "\\n");
            println!("    [{kind}] want: {}\n    [{kind}] got:  {}", clip(&want), clip(&text));
        }
    }
    let d = samples.len().max(1) as f32;
    (100.0 * exact as f32 / d, 100.0 * parsed as f32 / d)
}

fn even_subset<'a>(v: &'a [Sample], n: usize, filter: impl Fn(&Sample) -> bool) -> Vec<&'a Sample> {
    let all: Vec<&Sample> = v.iter().filter(|s| filter(s)).collect();
    if all.len() <= n {
        return all;
    }
    (0..n).map(|i| all[i * all.len() / n]).collect()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let num = |name: &str| -> Result<Option<usize>> { Ok(arg(name).map(|v| v.parse()).transpose()?) };
    let role_name = arg("--role").context("--role NAME is required (see roles/kinds.json)")?;
    let roles = RoleSet::load(&PathBuf::from(arg("--roles").unwrap_or("roles/kinds.json".into())))?;
    let role = roles.role(&role_name).with_context(|| format!("no role {role_name} in set {}", roles.name))?.clone();
    ensure!(!role.written, "role {role_name} is written, not trained");
    let mut cfg = roles.resolve(&role);
    if let Some(v) = num("--epochs")? {
        cfg.epochs = v;
    }
    if let Some(v) = num("--batch")? {
        cfg.batch = v;
    }
    if let Some(v) = arg("--lr") {
        cfg.first_lr = v.parse()?;
    }
    if let Some(v) = arg("--last-lr") {
        cfg.last_lr = v.parse()?;
    }
    if let Some(v) = num("--max-samples")? {
        cfg.max_samples = Some(v);
    }
    let eval_samples = num("--eval-samples")?.unwrap_or(64);
    let val_samples = num("--val-samples")?.unwrap_or(2000);
    let log_every = num("--log-every")?.unwrap_or(25).max(1);
    let max_steps = num("--steps")?;
    let tokenizer = PathBuf::from(arg("--tokenizer").unwrap_or("tokenizer.json".into()));
    let cache = PathBuf::from(arg("--cache").unwrap_or("../training-cache/code.bin".into()));
    let out = PathBuf::from(arg("--out").unwrap_or("../checkpoints/swarm".into())).join(&role.name);
    let tok = worker::load_tokenizer(&tokenizer)?;

    let t = Instant::now();
    let chunks = dataset::load_chunks(&cache, &tokenizer, num("--limit-chunks")?)?;
    println!("loaded {} chunks in {:.1}s", chunks.len(), t.elapsed().as_secs_f64());
    let t = Instant::now();
    let data = RoleData::build(&chunks, &tok, &role)?;
    drop(chunks);
    println!(
        "role {} kinds {:?}: {} train, {} held-out samples ({:.1}s); {} output rows; dropped {} train fills outside the rows, {} over {} tokens",
        role.name, role.kinds, data.train.len(), data.val.len(), t.elapsed().as_secs_f64(), data.rows.ids.len(),
        data.train_out_of_role, data.too_long, worker::CONTEXT
    );
    let val_in_role = data.val.iter().filter(|s| s.in_role).count();
    println!(
        "held-out fills inside the rows: {}/{} ({:.1}%)",
        val_in_role,
        data.val.len(),
        100.0 * val_in_role as f64 / data.val.len().max(1) as f64
    );
    ensure!(!data.train.is_empty(), "no training samples for role {}", role.name);

    let device: <Train as Backend>::Device = Default::default();
    let resumed = flag("--resume") && out.join("model.bin").exists();
    let (mut model, params) = if resumed {
        let (m, _, c) = YumonDecBrain::<Train>::load(out.to_str().unwrap(), &device)?;
        println!("resuming from {}", out.display());
        (m, c.param_count())
    } else {
        worker::init_model(&role, &roles.base, &device)?
    };
    let start_epoch = if resumed {
        serde_json::from_str::<RoleMeta>(&std::fs::read_to_string(out.join("role.json"))?)?.epochs_trained
    } else {
        0
    };
    println!(
        "model: {} params, {} epochs x {} samples, batch {}, lr {:.1e} -> {:.1e}",
        params,
        cfg.epochs,
        cfg.max_samples.map_or(data.train.len(), |m| m.min(data.train.len())),
        cfg.batch,
        cfg.first_lr,
        cfg.last_lr
    );
    if flag("--check") {
        println!("--check: nothing trained");
        return Ok(());
    }

    std::fs::create_dir_all(&out)?;
    let rows_train = Tensor::<Train, 1, Int>::from_ints(
        data.rows.ids.iter().map(|&r| r as i32).collect::<Vec<_>>().as_slice(),
        &device,
    );
    let pos = data.rows.position();
    let val_set = even_subset(&data.val, val_samples, |s| s.in_role);
    let gen_set = even_subset(&data.val, eval_samples, |_| true);
    let per_epoch = cfg.max_samples.map_or(data.train.len(), |m| m.min(data.train.len())) / cfg.batch;
    let total = per_epoch * cfg.epochs;
    let mut optimizer = AdamWConfig::new()
        .with_epsilon(1e-7)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .with_weight_decay(cfg.weight_decay as f32)
        .init();
    let mut rng = Rng(0x2545_f491_4f6c_dd1d ^ (start_epoch as u64 + 1));
    let mut best = f32::MAX;
    if resumed {
        if let Ok(m) = serde_json::from_str::<RoleMeta>(&std::fs::read_to_string(out.join("role.json"))?) {
            best = m.val_loss.unwrap_or(f32::MAX);
        }
    }
    let started = Instant::now();
    let mut step = 0usize;
    'epochs: for epoch in 1..=cfg.epochs {
        let batches = epoch_batches(data.train.len(), cfg.batch, cfg.max_samples, &data.train, &mut rng);
        let (mut run_loss, mut run_n, mut tokens) = (0.0f64, 0usize, 0usize);
        let epoch_start = Instant::now();
        for ids in &batches {
            let refs: Vec<&Sample> = ids.iter().map(|&i| &data.train[i]).collect();
            let batch: Batch = make_batch(&refs, &pos);
            if batch.supervised == 0 {
                continue;
            }
            let lr = lr_at(step, total, cfg.first_lr, cfg.last_lr);
            let s = worker::score::<Train>(&model, &batch, &rows_train, &device);
            let loss_v = s.loss.clone().into_data().convert::<f32>().to_vec::<f32>()?[0];
            ensure!(loss_v.is_finite(), "non-finite loss at step {step}");
            let grads = GradientsParams::from_grads(s.loss.backward(), &model);
            model = optimizer.step(lr, model, grads);
            step += 1;
            run_loss += loss_v as f64;
            run_n += 1;
            tokens += batch.rows * batch.len;
            if step % log_every == 0 {
                let avg = run_loss / run_n as f64;
                let secs = started.elapsed().as_secs_f64();
                let eta = secs / step as f64 * total.saturating_sub(step) as f64;
                println!(
                    "epoch {}/{} step {}/{} loss {:.4} lr {:.2e} {:.1} step/s {:.0} tok/s eta {:.0}s",
                    start_epoch + epoch, start_epoch + cfg.epochs, step, total, avg, lr,
                    step as f64 / secs, tokens as f64 / epoch_start.elapsed().as_secs_f64(), eta
                );
                append(&out.join("train.csv"), "epoch,step,loss,lr", &format!("{},{step},{avg:.5},{lr:.3e}", start_epoch + epoch))?;
                run_loss = 0.0;
                run_n = 0;
            }
            if max_steps.is_some_and(|m| step >= m) {
                break;
            }
        }
        // end of epoch: held-out loss, accuracy, greedy fills, checkpoint
        let infer = model.valid();
        let v = validate(&infer, &val_set, &data.rows, cfg.batch);
        println!("epoch {} held-out ({} samples): loss {:.4}, top-1 {:.1}%, top-3 {:.1}%", start_epoch + epoch,
            v.samples, v.loss, 100.0 * v.top1, 100.0 * v.top3);
        let (exact, parsed) = generate(&infer, &gen_set, &data.rows, &tok, &role.kinds, 2);
        println!("  greedy on {} held-out fills: exact {exact:.1}%, parses {parsed:.1}%", gen_set.len());
        append(
            &out.join("val.csv"),
            "epoch,val_loss,top1,top3,exact,parse",
            &format!("{},{:.5},{:.4},{:.4},{exact:.2},{parsed:.2}", start_epoch + epoch, v.loss, v.top1, v.top3),
        )?;
        if v.samples == 0 || v.loss < best {
            best = v.loss;
            let meta = RoleMeta {
                role: role.name.clone(),
                kinds: role.kinds.clone(),
                rows: data.rows.ids.clone(),
                epochs_trained: start_epoch + epoch,
                val_loss: Some(v.loss),
                val_top1: Some(v.top1),
                val_top3: Some(v.top3),
                params,
                train_samples: data.train.len(),
            };
            worker::save_role(&infer, &out, &tokenizer, &meta, cfg.batch, (run_loss / run_n.max(1) as f64) as f32)?;
            println!("  saved {} (best held-out loss)", out.display());
        }
        if max_steps.is_some_and(|m| step >= m) {
            break 'epochs;
        }
    }
    println!("done: {} steps in {:.0}s", step, started.elapsed().as_secs_f64());
    Ok(())
}
