//! Phase 6a: train the reward model (the "discriminator" of docs/SWARM_RL.md).
//!
//! cargo run --release --bin train_critic -- --level hole|program [--negatives out/critic/neg.jsonl]
//!   [--out ../checkpoints/swarm/critic_<level>] [--init <checkpoint>] [--k 3] [--keep 0.15]
//!   [--epochs 2] [--batch 8] [--lr 3e-4] [--last-lr 3e-5] [--swarm-repeat 3] [--val-items 3000]
//!   [--embed 96 --layers 6 --heads 4 --ff 384] (default: from-scratch critic of that shape;
//!   --init <checkpoint> starts from a pretrained decoder instead)
//!   [--limit-chunks N] [--steps N] [--log-every 25] [--check]
//!
//! Each step takes `batch` real fills, adds `k` wrong ones to each and trains the critic to
//! rank the real one first. Held-out files (`dataset::is_validation`) are never trained on,
//! and their negatives come from held-out files only. After each epoch it prints ranking
//! accuracy per negative kind (`other` = random same-kind fill, `same` = another hole of the
//! same chunk/file, `swarm` = what the swarm wrote) and keeps the epoch with the best mean.
use anyhow::{Context, Result, ensure};
use burn::{
    grad_clipping::GradientClippingConfig,
    module::AutodiffModule,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
};
use std::{path::PathBuf, time::Instant};
use swarm::{
    critic::{self, Item, Rng, load_negatives, make_group, pool_of, save_critic, score_batch},
    dataset,
    roles::{Role, RoleSet},
    worker::{self, Train, encode},
};
use yumon_pet::brain::decoder_model::YumonDecBrain;

fn lr_at(step: usize, total: usize, first: f64, last: f64) -> f64 {
    let warm = 100.min(total / 10).max(1);
    if step < warm {
        return first * (step + 1) as f64 / warm as f64;
    }
    let t = (step - warm) as f64 / (total - warm).max(1) as f64;
    last + (first - last) * 0.5 * (1.0 + (std::f64::consts::PI * t).cos())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let num = |name: &str, d: usize| -> Result<usize> { Ok(arg(name).map(|v| v.parse()).transpose()?.unwrap_or(d)) };
    let fnum = |name: &str, d: f64| -> Result<f64> { Ok(arg(name).map(|v| v.parse()).transpose()?.unwrap_or(d)) };
    let level = arg("--level").context("--level hole|program is required")?;
    ensure!(level == "hole" || level == "program", "--level is hole or program");
    let k = num("--k", 3)?;
    let epochs = num("--epochs", 2)?;
    let batch = num("--batch", 8)?;
    let log_every = num("--log-every", 25)?.max(1);
    let val_items = num("--val-items", 3000)?;
    let swarm_repeat = num("--swarm-repeat", 3)?.max(1);
    let keep = fnum("--keep", 0.15)?;
    let (first_lr, last_lr) = (fnum("--lr", 3e-4)?, fnum("--last-lr", 3e-5)?);
    let max_steps = arg("--steps").map(|v| v.parse::<usize>()).transpose()?;
    let tokenizer = PathBuf::from(arg("--tokenizer").unwrap_or("tokenizer.json".into()));
    let cache = PathBuf::from(arg("--cache").unwrap_or("../training-cache/code.bin".into()));
    let out = PathBuf::from(arg("--out").unwrap_or(format!("../checkpoints/swarm/critic_{level}")));
    let roles = RoleSet::load(&PathBuf::from(arg("--roles").unwrap_or("roles/kinds.json".into())))?;
    let tok = worker::load_tokenizer(&tokenizer)?;
    let fill_id = encode(&tok, swarm::skeleton::FILL)[0];

    let t = Instant::now();
    let chunks = dataset::load_chunks(&cache, &tokenizer, arg("--limit-chunks").map(|v| v.parse()).transpose()?)?;
    let mut items: Vec<Item> = if level == "hole" { critic::hole_items(&chunks, keep) } else { critic::program_items(&chunks) };
    let from_chunks = items.len();
    drop(chunks);
    let mut swarm_items = 0;
    if let Some(p) = arg("--negatives") {
        let extra = load_negatives(&PathBuf::from(&p), &level)?;
        swarm_items = extra.len();
        items.extend(extra);
    }
    let (mut train, mut val): (Vec<Item>, Vec<Item>) = items.into_iter().partition(|i| !dataset::is_validation(&i.file));
    let train_swarm = train.iter().filter(|i| i.hard.iter().any(|(n, _)| *n == critic::NegKind::Swarm)).count();
    let val_swarm = val.iter().filter(|i| i.hard.iter().any(|(n, _)| *n == critic::NegKind::Swarm)).count();
    println!(
        "{level} level: {from_chunks} items from chunks + {swarm_items} swarm negatives ({:.1}s); train {} ({train_swarm} swarm), held-out {} ({val_swarm} swarm)",
        t.elapsed().as_secs_f64(), train.len(), val.len()
    );
    ensure!(!train.is_empty(), "no training items");
    let pool_train = pool_of(&train);
    let pool_val = pool_of(&val);
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    rng.shuffle(&mut val);
    // keep every swarm item of the held-out set first: they are the hard ones
    val.sort_by_key(|i| !i.hard.iter().any(|(n, _)| *n == critic::NegKind::Swarm));
    val.truncate(val_items.max(val_swarm));
    let val_refs: Vec<&Item> = val.iter().collect();

    let device: <Train as Backend>::Device = Default::default();
    // `--embed N` trains a critic of its own shape from scratch instead of a checkpoint
    let spec = (arg("--init").is_none()).then(|| arg("--embed").unwrap_or("96".into())).map(|e| -> Result<_> {
        let embed_dim: usize = e.parse()?;
        Ok(swarm::roles::ModelSpec {
            checkpoint: None,
            embed_dim,
            n_layers: num("--layers", 6)?,
            attn_heads: num("--heads", 4)?,
            ff_dim: num("--ff", embed_dim * 4)?,
            token_embed_dim: num("--tok-embed", embed_dim)?,
            jl_ffn: 0,
        })
    }).transpose()?;
    let role = Role {
        name: "critic".into(), kinds: Vec::new(), crop_tokens: 448, out_vocab: worker::VOCAB, written: false, model: spec,
        train: Default::default(),
    };
    let base = arg("--init").unwrap_or(roles.base.clone());
    let (mut model, params): (YumonDecBrain<Train>, usize) = worker::init_model(&role, &base, &device)?;
    println!("critic: {params} params from {base}; k {k} negatives, batch {batch} items ({} sequences)", batch * (1 + k));
    if flag("--check") {
        println!("--check: nothing trained");
        return Ok(());
    }

    // order: every item once, swarm items `swarm_repeat` times
    let order_of = |rng: &mut Rng| {
        let mut o: Vec<usize> = Vec::new();
        for (i, it) in train.iter().enumerate() {
            let rep = if it.hard.iter().any(|(n, _)| *n == critic::NegKind::Swarm) { swarm_repeat } else { 1 };
            o.extend(std::iter::repeat(i).take(rep));
        }
        rng.shuffle(&mut o);
        o
    };
    let per_epoch = order_of(&mut rng).len() / batch;
    let total = per_epoch * epochs;
    let mut optimizer = AdamWConfig::new()
        .with_epsilon(1e-7)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .with_weight_decay(0.01)
        .init();
    let mut best = -1.0f32;
    let mut step = 0usize;
    let started = Instant::now();
    std::fs::create_dir_all(&out)?;
    'epochs: for epoch in 1..=epochs {
        let order = order_of(&mut rng);
        let (mut run_loss, mut run_acc, mut run_n) = (0.0f64, 0.0f64, 0usize);
        for ids in order.chunks(batch) {
            let refs: Vec<&Item> = ids.iter().map(|&i| &train[i]).collect();
            let g = make_group(&refs, &pool_train, &tok, fill_id, k, &mut rng);
            if g.items == 0 {
                continue;
            }
            let lr = lr_at(step, total, first_lr, last_lr);
            let scores = score_batch::<Train>(&model, &g.seqs, &device).reshape([g.items, 1 + k]);
            let lp = burn::tensor::activation::log_softmax(scores.clone(), 1);
            let loss = lp.narrow(1, 0, 1).mean().neg();
            let loss_v = critic::to_vec(loss.clone())[0];
            ensure!(loss_v.is_finite(), "non-finite loss at step {step}");
            let s = scores.into_data().convert::<f32>().to_vec::<f32>().unwrap();
            let wins = (0..g.items).map(|i| (1..=k).filter(|&j| s[i * (1 + k)] > s[i * (1 + k) + j]).count()).sum::<usize>();
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(lr, model, grads);
            step += 1;
            run_loss += loss_v as f64;
            run_acc += wins as f64 / (g.items * k) as f64;
            run_n += 1;
            if step % log_every == 0 {
                let secs = started.elapsed().as_secs_f64();
                println!(
                    "epoch {epoch}/{epochs} step {step}/{total} loss {:.4} train acc {:.1}% lr {lr:.2e} {:.1} step/s eta {:.0}s",
                    run_loss / run_n as f64, 100.0 * run_acc / run_n as f64, step as f64 / secs,
                    secs / step as f64 * total.saturating_sub(step) as f64
                );
                run_loss = 0.0;
                run_acc = 0.0;
                run_n = 0;
            }
            if max_steps.is_some_and(|m| step >= m) {
                break;
            }
        }
        let infer = model.valid();
        let ev = critic::evaluate(&infer, &val_refs, &pool_val, &tok, fill_id, k, 16, &level, 17);
        println!("epoch {epoch} held-out ({} items): ranking accuracy by negative: {:?}  mean {:.3}", val_refs.len(), ev.meta.by_neg, ev.score);
        for (kind, c) in &ev.meta.calib {
            println!("    {kind:<10} n {:>5}  acc {:.3}  true {:>7.3}  wrong {:>7.3}", c.n, c.accuracy, c.true_mean, c.neg_mean);
        }
        if ev.score > best {
            best = ev.score;
            save_critic(&infer, &out, &tokenizer, &ev.meta, epoch, params, batch, 0.0)?;
            println!("  saved {} (best held-out accuracy)", out.display());
        }
        if max_steps.is_some_and(|m| step >= m) {
            break 'epochs;
        }
    }
    println!("done: {step} steps in {:.0}s", started.elapsed().as_secs_f64());
    Ok(())
}
