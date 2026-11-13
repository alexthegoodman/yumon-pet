//! Phase 6b: RL for the whole swarm (see docs/SWARM_RL.md).
//!
//! One rollout = the planner and the workers writing a program for a doc comment, sampled
//! (not greedy) so a group of rollouts for one doc differ. Every attempt of every hole is a
//! `Trace` with the token ids that were sampled, because the policy gradient needs exactly
//! those. Rewards are cut per hole and handed to the role that wrote it:
//!
//! - rejected attempt (`syn` refused it): `-syntax`, nothing else;
//! - accepted fill: `critic * hole critic` (does this fill belong in its context);
//! - shared by every accepted fill of a program: `program * program critic` (does the whole
//!   program match the doc), a bonus when `cargo check` is clean, and `-program` when the
//!   program is incomplete or does not parse as a file;
//! - `cargo check` errors, charged to the deepest fill whose bytes contain the error span
//!   (`-compile * min(errors, 3) / 3`);
//! - tree credit: a fill also receives `lambda` times the mean value of the fills that
//!   filled its own sub-holes, so a block that sets up good children is rewarded for it.
//!
//! The value of a trace becomes an advantage against its own group: the mean of the same
//! role's traces for the same doc (or the role's batch mean when there are fewer than two),
//! divided by the role's spread. Each role then takes one REINFORCE step on its own
//! weights, with a KL term against the frozen starting weights.
use crate::{
    check::CheckError,
    critic::Critic,
    planner::{Planner, ROOT, doc_prefix},
    skeleton::KINDS,
    swarm::{Filler, Request, Span, charge, check_fill},
    worker::{Batch, EOS, Infer, Rows, RoleMeta, Sample, Train, Worker, make_batch},
};
use anyhow::{Result, ensure};
use burn::{
    module::AutodiffModule,
    optim::{GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    prelude::*,
};
use std::collections::HashMap;
use yumon_pet::brain::decoder_model::YumonDecBrain;

/// One attempt at one hole.
#[derive(Debug, Clone)]
pub struct Trace {
    pub doc: usize,
    pub rollout: usize,
    /// `planner` for the root, else the worker's role name.
    pub role: String,
    pub kind: String,
    pub path: Vec<usize>,
    pub attempt: usize,
    pub context: String,
    /// What the model wrote (the planner's is the skeleton without the doc).
    pub text: String,
    pub prompt: Vec<u32>,
    pub ids: Vec<u32>,
    /// The model ended with EOS (false = cut off at the context limit).
    pub ended: bool,
    /// Passed the dispatcher's `syn` check (set after the run).
    pub accepted: bool,
}

impl Trace {
    /// Prompt + sampled tokens (+ EOS), loss only on the sampled part.
    pub fn sample(&self) -> Sample {
        let mut ids = self.prompt.clone();
        let prompt_len = ids.len();
        ids.extend(&self.ids);
        if self.ended {
            ids.push(EOS);
        }
        Sample { ids: ids.iter().map(|&t| t as u16).collect(), prompt_len, kind: 0, in_role: true }
    }
}

fn seed_for(salt: u64, path: &[usize], attempt: usize) -> u64 {
    let mut s = salt ^ 0xcbf2_9ce4_8422_2325;
    for &p in path {
        s = (s ^ (p as u64 + 1)).wrapping_mul(0x100_0000_01b3);
    }
    s ^ (attempt as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// Dispatcher filler that samples every attempt and records it.
pub struct RlFiller<'a> {
    pub planner: &'a Planner,
    pub doc: String,
    pub workers: &'a [Worker],
    pub doc_index: usize,
    pub rollout: usize,
    pub salt: u64,
    pub temperature: f32,
    /// 0 = the whole softmax.
    pub top_k: usize,
    pub traces: Vec<Trace>,
}

impl RlFiller<'_> {
    fn worker(&self, kind: &str) -> Option<&Worker> {
        self.workers.iter().find(|w| w.handles(kind))
    }

    /// Mark which traces the dispatcher's `syn` check passed.
    pub fn finish(&mut self) {
        for t in &mut self.traces {
            t.accepted = if t.path.is_empty() { check_fill(ROOT, &format!("{}{}", self.doc, t.text)) } else { check_fill(&t.kind, &t.text) };
        }
    }
}

impl Filler for RlFiller<'_> {
    fn fill(&mut self, req: &Request) -> Option<String> {
        let root = req.path.is_empty();
        let (w, ctx) = if root { (&self.planner.worker, self.doc.as_str()) } else { (self.worker(req.kind)?, req.context) };
        let prompt = w.prompt_for(ctx);
        let k = if self.top_k == 0 { usize::MAX } else { self.top_k };
        let (ids, ended) = w.rollout_ids(&prompt, seed_for(self.salt, req.path, req.attempt), self.temperature, k)?;
        let text = w.decode(&ids);
        self.traces.push(Trace {
            doc: self.doc_index,
            rollout: self.rollout,
            role: if root { "planner".into() } else { w.meta.role.clone() },
            kind: req.kind.to_owned(),
            path: req.path.to_vec(),
            attempt: req.attempt,
            context: req.context.to_owned(),
            text: text.clone(),
            prompt,
            ids,
            ended,
            accepted: false,
        });
        Some(if root { format!("{}{}", self.doc, text) } else { text })
    }
    fn name(&self, kind: &str) -> &str {
        if kind == ROOT { "planner" } else { self.worker(kind).map_or("none", |w| w.meta.role.as_str()) }
    }
}

#[derive(Debug, Clone)]
pub struct Weights {
    pub critic: f32,
    pub syntax: f32,
    pub compile: f32,
    pub program: f32,
    pub clean: f32,
    pub lambda: f32,
}

impl Default for Weights {
    fn default() -> Self {
        Self { critic: 1.0, syntax: 1.0, compile: 0.5, program: 0.5, clean: 0.5, lambda: 0.5 }
    }
}

/// What judging one rollout produced.
#[derive(Debug, Clone, Default)]
pub struct Judged {
    /// Value per trace of the rollout, in trace order.
    pub values: Vec<f32>,
    pub hole_sum: f32,
    pub hole_n: usize,
    pub program_r: Option<f32>,
    pub complete: bool,
    pub parses: bool,
    pub errors: Option<usize>,
}

/// Tree credit: a trace's value is its own reward plus `lambda` times the mean value of the
/// accepted fills of its sub-holes. `paths[i]` is None for rejected attempts (no tree).
pub fn propagate(paths: &[Option<&[usize]>], base: &[f32], lambda: f32) -> Vec<f32> {
    let mut at: HashMap<&[usize], usize> = HashMap::new();
    for (i, p) in paths.iter().enumerate() {
        if let Some(p) = p {
            at.insert(*p, i);
        }
    }
    let mut order: Vec<usize> = (0..paths.len()).filter(|&i| paths[i].is_some()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(paths[i].unwrap().len()));
    let mut value = base.to_vec();
    let mut kids: Vec<(f32, usize)> = vec![(0.0, 0); paths.len()];
    for i in order {
        let (s, n) = kids[i];
        if n > 0 {
            value[i] = base[i] + lambda * s / n as f32;
        }
        let p = paths[i].unwrap();
        if let Some((_, parent)) = p.split_last() {
            if let Some(&j) = at.get(parent) {
                kids[j].0 += value[i];
                kids[j].1 += 1;
            }
        }
    }
    value
}

/// Judge one rollout. `program` is the assembled text and spans (None = holes left open);
/// `errors` the `cargo check` result for it (None = not checked).
pub fn value_rollout(traces: &[Trace], doc: &str, hole: &Critic, prog: Option<&Critic>, program: Option<&(String, Vec<Span>)>,
    parses: bool, errors: Option<&[CheckError]>, w: &Weights) -> Judged {
    let n = traces.len();
    let mut local = vec![0.0f32; n];
    let mut out = Judged { complete: program.is_some(), parses, ..Default::default() };

    let idx: Vec<usize> = (0..n).filter(|&i| traces[i].accepted && !traces[i].path.is_empty()).collect();
    let pairs: Vec<(&str, &str)> = idx.iter().map(|&i| (traces[i].context.as_str(), traces[i].text.as_str())).collect();
    for (&i, raw) in idx.iter().zip(hole.score(&pairs)) {
        let r = raw.map_or(0.0, |x| hole.reward(&traces[i].kind, x));
        local[i] = w.critic * r;
        out.hole_sum += r;
        out.hole_n += 1;
    }
    for (i, t) in traces.iter().enumerate() {
        if !t.accepted {
            local[i] = -w.syntax;
        }
    }

    let mut shared = -w.program;
    let mut hits: HashMap<&[usize], usize> = HashMap::new();
    if let (Some((text, spans)), true) = (program, parses) {
        let body = text.strip_prefix(doc_prefix(doc)).unwrap_or(text);
        let r = prog.map_or(0.0, |c| c.score(&[(doc, body)])[0].map_or(0.0, |x| c.reward(crate::critic::PROGRAM, x)));
        out.program_r = Some(r);
        shared = w.program * r;
        if let Some(errs) = errors {
            out.errors = Some(errs.len());
            if errs.is_empty() {
                shared += w.clean;
            }
            for e in errs {
                if let Some(s) = charge(spans, e.start, e.end) {
                    *hits.entry(s.path.as_slice()).or_default() += 1;
                }
            }
        }
    }

    let mut base = vec![0.0f32; n];
    for i in 0..n {
        base[i] = if traces[i].accepted {
            let c = hits.get(traces[i].path.as_slice()).copied().unwrap_or(0).min(3) as f32 / 3.0;
            local[i] + shared - w.compile * c
        } else {
            local[i]
        };
    }
    let paths: Vec<Option<&[usize]>> = traces.iter().map(|t| t.accepted.then_some(t.path.as_slice())).collect();
    out.values = propagate(&paths, &base, w.lambda);
    out
}

/// Advantages for `(group, role, value)` triples: value minus the mean of the same role in
/// the same group (the role's batch mean when the group has under two), over the role's
/// spread of those residuals, clipped to +-3.
pub fn advantages(items: &[(usize, usize, f32)]) -> Vec<f32> {
    let mut role_sum: HashMap<usize, (f32, usize)> = HashMap::new();
    let mut group_sum: HashMap<(usize, usize), (f32, usize)> = HashMap::new();
    for &(g, r, v) in items {
        let a = role_sum.entry(r).or_default();
        a.0 += v;
        a.1 += 1;
        let b = group_sum.entry((g, r)).or_default();
        b.0 += v;
        b.1 += 1;
    }
    let base = |g: usize, r: usize| {
        let (s, n) = group_sum[&(g, r)];
        if n >= 2 {
            s / n as f32
        } else {
            let (s, n) = role_sum[&r];
            s / n as f32
        }
    };
    let mut var: HashMap<usize, (f32, usize)> = HashMap::new();
    for &(g, r, v) in items {
        let e = var.entry(r).or_default();
        e.0 += (v - base(g, r)).powi(2);
        e.1 += 1;
    }
    items
        .iter()
        .map(|&(g, r, v)| {
            let (s, n) = var[&r];
            let std = (s / n as f32).sqrt().max(0.05);
            ((v - base(g, r)) / std).clamp(-3.0, 3.0)
        })
        .collect()
}

pub type Opt = OptimizerAdaptor<burn::optim::AdamW, YumonDecBrain<Train>, Train>;

/// One role being trained: weights, optimiser, frozen reference and its output rows.
pub struct Policy {
    pub role: String,
    pub meta: RoleMeta,
    pub model: YumonDecBrain<Train>,
    pub opt: Opt,
    pub reference: YumonDecBrain<Infer>,
    pub rows: Rows,
    pub rows_train: Tensor<Train, 1, Int>,
    pub rows_infer: Tensor<Infer, 1, Int>,
    pub pos: Vec<i32>,
}

/// log p of every target token over the role's rows at `temp` (masked positions are junk).
pub fn token_logps<B: Backend>(model: &YumonDecBrain<B>, batch: &Batch, rows: &Tensor<B, 1, Int>, temp: f32,
    device: &B::Device) -> Tensor<B, 1>
where
    B: yumon_pet::brain::flash_attn::backend::FlashAttention,
{
    let n = batch.rows * batch.len;
    let inputs = Tensor::<B, 2, Int>::from_data(TensorData::new(batch.inputs.clone(), [batch.rows, batch.len]), device);
    let targets = Tensor::<B, 2, Int>::from_data(TensorData::new(batch.targets.clone(), [n, 1]), device);
    let width = rows.dims()[0];
    let logits = model.logits_rows(inputs, rows.clone()).reshape([n, width]) / temp.max(1e-3) as f64;
    burn::tensor::activation::log_softmax(logits, 1).gather(1, targets).reshape([n])
}

#[derive(Debug, Clone, Default)]
pub struct UpdateStats {
    pub loss: f32,
    pub kl: f32,
    pub n: usize,
    pub tokens: usize,
}

/// One REINFORCE step: `-sum A_i * mean_t log p(token_t)` plus `beta * KL` to the reference.
pub fn update(p: &mut Policy, items: &[(Sample, f32)], temp: f32, beta: f32, lr: f64) -> Result<UpdateStats> {
    let device: <Train as Backend>::Device = Default::default();
    let refs: Vec<&Sample> = items.iter().map(|(s, _)| s).collect();
    let batch = make_batch(&refs, &p.pos);
    ensure!(batch.supervised > 0, "nothing to learn from");
    let (b, len) = (batch.rows, batch.len);
    let n = b * len;
    // per-token weights: mean over a trace's own tokens, mean over traces
    let mut adv_w = vec![0.0f32; n];
    let mut kl_w = vec![0.0f32; n];
    for r in 0..b {
        let toks: f32 = batch.mask[r * len..(r + 1) * len].iter().sum();
        if toks == 0.0 {
            continue;
        }
        for i in 0..len {
            if batch.mask[r * len + i] > 0.0 {
                adv_w[r * len + i] = items[r].1 / (toks * b as f32);
                kl_w[r * len + i] = 1.0 / (toks * b as f32);
            }
        }
    }
    let ref_lp = token_logps::<Infer>(&p.reference, &batch, &p.rows_infer, temp, &Default::default());
    let ref_lp = Tensor::<Train, 1>::from_data(ref_lp.into_data(), &device);
    let lp = token_logps::<Train>(&p.model, &batch, &p.rows_train, temp, &device);
    let aw = Tensor::<Train, 1>::from_data(TensorData::new(adv_w, [n]), &device);
    let kw = Tensor::<Train, 1>::from_data(TensorData::new(kl_w, [n]), &device);
    let pg = (lp.clone() * aw).sum().neg();
    let d = ref_lp - lp;
    let kl = ((d.clone().exp() - d - 1.0) * kw).sum();
    let loss = pg + kl.clone() * beta as f64;
    let loss_v = loss.clone().into_data().convert::<f32>().to_vec::<f32>().unwrap()[0];
    let kl_v = kl.into_data().convert::<f32>().to_vec::<f32>().unwrap()[0];
    ensure!(loss_v.is_finite(), "non-finite RL loss");
    let grads = GradientsParams::from_grads(loss.backward(), &p.model);
    let model = p.model.clone();
    p.model = p.opt.step(lr, model, grads);
    Ok(UpdateStats { loss: loss_v, kl: kl_v, n: b, tokens: batch.supervised })
}

impl Policy {
    /// Inference copy of the current weights, for the next rollouts.
    pub fn snapshot(&self) -> YumonDecBrain<Infer> {
        self.model.clone().valid()
    }
}

#[allow(dead_code)]
fn _kinds() -> usize {
    KINDS.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credit_flows_up_from_children() {
        // root -> [0] -> [0,0]; value(leaf) = 1, block [0] has base 0, root base 0
        let paths: Vec<Option<&[usize]>> = vec![Some(&[]), Some(&[0]), Some(&[0, 0]), None];
        let base = [0.0, 0.0, 1.0, -1.0];
        let v = propagate(&paths, &base, 0.5);
        assert_eq!(v[2], 1.0);
        assert_eq!(v[1], 0.5);
        assert_eq!(v[0], 0.25);
        assert_eq!(v[3], -1.0, "rejected attempts take no part in the tree");
        assert_eq!(propagate(&paths, &base, 0.0), base);
    }

    #[test]
    fn credit_averages_siblings() {
        let paths: Vec<Option<&[usize]>> = vec![Some(&[]), Some(&[0]), Some(&[1])];
        let v = propagate(&paths, &[0.0, 1.0, -1.0], 1.0);
        assert_eq!(v[0], 0.0);
    }

    #[test]
    fn advantages_are_relative_to_the_group() {
        // one role, two groups: the same gap gives the same advantage whatever the level
        let a = advantages(&[(0, 0, 1.0), (0, 0, 2.0), (1, 0, 11.0), (1, 0, 12.0)]);
        assert!((a[0] + a[1]).abs() < 1e-5 && a[1] > 0.0);
        assert!((a[0] - a[2]).abs() < 1e-5 && (a[1] - a[3]).abs() < 1e-5);
        // equal values carry no signal
        assert!(advantages(&[(0, 0, 1.0), (0, 0, 1.0)]).iter().all(|x| *x == 0.0));
    }

    #[test]
    fn a_lone_trace_uses_the_role_mean() {
        let a = advantages(&[(0, 0, 0.0), (1, 0, 2.0)]);
        assert!(a[0] < 0.0 && a[1] > 0.0);
    }

    #[test]
    fn trace_sample_appends_eos_only_when_the_model_ended() {
        let t = |ended| Trace {
            doc: 0, rollout: 0, role: "block".into(), kind: "block".into(), path: vec![0], attempt: 0, context: String::new(),
            text: String::new(), prompt: vec![1, 5, 6], ids: vec![9, 8], ended, accepted: true,
        };
        let s = t(true).sample();
        assert_eq!((s.prompt_len, s.ids.as_slice()), (3, &[1u16, 5, 6, 9, 8, 2][..]));
        assert_eq!(t(false).sample().ids.len(), 5);
    }
}
