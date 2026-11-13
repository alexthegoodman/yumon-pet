//! Phase 6a: the learned reward model (see docs/SWARM_RL.md).
//!
//! A critic is a small decoder that reads `[BOS] context [<fill>] fill [EOS]` and writes one
//! number: the logit of the EOS row of the tied table at the last position. No new head, so
//! any decoder checkpoint can be a critic. It is trained listwise: for every real fill
//! (`truth`) the critic sees K wrong ones and must rank the real one first.
//!
//! Two levels share all of this code:
//! - `hole`: context = the worker's prompt (chunk with the target open), fill = a hole's fill.
//!   Scores "does this fill belong here". This is the per-worker reward.
//! - `program`: context = the doc comment, fill = everything the swarm wrote below it.
//!   Scores "does this program do what the doc says". This is the shared reward.
//!
//! Negatives, easiest to hardest: `other` (same kind, random other chunk), `same` (same kind,
//! another hole of the same chunk or another chunk of the same file) and `swarm` (what the
//! current swarm wrote for that context, from `gen_negatives`). Held-out files are split by
//! `dataset::is_validation`, so the critic is always scored on code it never saw.
use crate::{
    dataset::Chunk,
    planner::doc_prefix,
    worker::{BOS, CONTEXT, EOS, Infer, PAD, RoleMeta, encode, load_tokenizer, save_role},
};
use anyhow::{Context, Result};
use burn::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};
use tokenizers::Tokenizer;
use yumon_pet::brain::decoder_model::YumonDecBrain;

pub const PROGRAM: &str = "program";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NegKind {
    Other,
    Same,
    Swarm,
}

impl NegKind {
    pub fn name(self) -> &'static str {
        match self {
            NegKind::Other => "other",
            NegKind::Same => "same",
            NegKind::Swarm => "swarm",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Item {
    /// Hole kind, or `program`.
    pub kind: String,
    pub file: String,
    pub context: String,
    pub truth: String,
    /// Wrong fills for this exact context.
    pub hard: Vec<(NegKind, String)>,
}

/// One line of `gen_negatives` output: what the swarm wrote where the truth was `truth`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegRecord {
    pub level: String,
    pub kind: String,
    pub file: String,
    pub context: String,
    pub truth: String,
    pub bad: String,
}

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
}

/// Deterministic uniform in [0, 1) from two integers (which instances to keep).
fn unit(a: u64, b: u64) -> f64 {
    let mut x = a.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ b.wrapping_add(0x7f4a_7c15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// Hole-level items: every kept instance of every chunk, with up to two `same` negatives
/// (another hole of that kind in the same chunk whose fill differs).
pub fn hole_items(chunks: &[Chunk], keep: f64) -> Vec<Item> {
    chunks
        .par_iter()
        .enumerate()
        .flat_map_iter(|(ci, chunk)| {
            let mut out = Vec::new();
            let Ok(root) = crate::dataset::extract_chunk(chunk) else { return out };
            let insts = crate::dataset::instances(&root);
            for (i, inst) in insts.iter().enumerate() {
                if unit(ci as u64, i as u64) >= keep {
                    continue;
                }
                let mut hard = Vec::new();
                for off in 1..insts.len() {
                    let o = &insts[(i + off) % insts.len()];
                    if o.kind == inst.kind && o.target != inst.target && !hard.iter().any(|(_, t)| t == &o.target) {
                        hard.push((NegKind::Same, o.target.clone()));
                        if hard.len() == 2 {
                            break;
                        }
                    }
                }
                out.push(Item {
                    kind: inst.kind.clone(),
                    file: chunk.file.clone(),
                    context: inst.context.clone(),
                    truth: inst.target.clone(),
                    hard,
                });
            }
            out
        })
        .collect()
}

/// Program-level items: doc comment -> the rest of the chunk; the `same` negative is the
/// body of another chunk of the same file.
pub fn program_items(chunks: &[Chunk]) -> Vec<Item> {
    let mut by_file: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    let docs: Vec<Option<(&str, &str)>> = chunks
        .iter()
        .map(|c| {
            let d = doc_prefix(&c.text);
            (!d.is_empty() && c.text.len() > d.len()).then(|| (d, &c.text[d.len()..]))
        })
        .collect();
    for (i, d) in docs.iter().enumerate() {
        if d.is_some() {
            by_file.entry(chunks[i].file.as_str()).or_default().push(i);
        }
    }
    let mut out = Vec::new();
    for (i, d) in docs.iter().enumerate() {
        let Some((doc, body)) = d else { continue };
        if crate::dataset::extract_chunk(&chunks[i]).is_err() {
            continue;
        }
        let mut hard = Vec::new();
        if let Some(sibs) = by_file.get(chunks[i].file.as_str()) {
            for &j in sibs.iter().filter(|&&j| j != i).take(2) {
                hard.push((NegKind::Same, docs[j].unwrap().1.to_owned()));
            }
        }
        out.push(Item {
            kind: PROGRAM.into(),
            file: chunks[i].file.clone(),
            context: (*doc).to_owned(),
            truth: (*body).to_owned(),
            hard,
        });
    }
    out
}

pub fn load_negatives(path: &Path, level: &str) -> Result<Vec<Item>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut out = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let r: NegRecord = serde_json::from_str(line)?;
        if r.level != level || r.bad == r.truth {
            continue;
        }
        out.push(Item { kind: r.kind, file: r.file, context: r.context, truth: r.truth, hard: vec![(NegKind::Swarm, r.bad)] });
    }
    Ok(out)
}

/// Random same-kind fills for `other` negatives, per split.
pub type Pool = BTreeMap<String, Vec<String>>;

pub fn pool_of(items: &[Item]) -> Pool {
    let mut p = Pool::new();
    for it in items {
        p.entry(it.kind.clone()).or_default().push(it.truth.clone());
    }
    p
}

/// `[BOS] context [<fill>] fill [EOS]`; None if the context alone leaves no room, or the
/// fill does not fit and `clip` is false.
pub fn encode_pair(tok: &Tokenizer, fill_id: u32, context: &str, fill: &str, clip: bool) -> Option<Vec<u32>> {
    let mut ids = vec![BOS];
    ids.extend(encode(tok, context));
    ids.push(fill_id);
    if ids.len() + 1 >= CONTEXT {
        return None;
    }
    let f = encode(tok, fill);
    let room = CONTEXT - ids.len() - 1;
    if f.len() > room {
        if !clip {
            return None;
        }
        ids.extend(&f[..room]);
    } else {
        ids.extend(f);
    }
    ids.push(EOS);
    Some(ids)
}

/// One scalar per sequence: the EOS-row logit at the last token.
pub fn score_batch<B: Backend>(model: &YumonDecBrain<B>, seqs: &[Vec<u32>], device: &B::Device) -> Tensor<B, 1>
where
    B: yumon_pet::brain::flash_attn::backend::FlashAttention,
{
    let b = seqs.len();
    let len = seqs.iter().map(Vec::len).max().unwrap_or(1).max(1);
    let mut inputs = vec![PAD as i32; b * len];
    let mut last = vec![0i32; b];
    for (r, s) in seqs.iter().enumerate() {
        for (i, &t) in s.iter().enumerate() {
            inputs[r * len + i] = t as i32;
        }
        last[r] = s.len() as i32 - 1;
    }
    let inputs = Tensor::<B, 2, Int>::from_data(TensorData::new(inputs, [b, len]), device);
    let last = Tensor::<B, 2, Int>::from_data(TensorData::new(last, [b, 1]), device);
    let rows = Tensor::<B, 1, Int>::from_ints([EOS as i32], device);
    model.logits_rows(inputs, rows).reshape([b, len]).gather(1, last).reshape([b])
}

/// A list of sequences per item: the truth first, then `k` negatives.
pub struct Group {
    pub seqs: Vec<Vec<u32>>,
    /// Kind of each negative, in order (len = items * k).
    pub neg: Vec<NegKind>,
    pub kinds: Vec<String>,
    pub items: usize,
    pub k: usize,
}

pub fn make_group(items: &[&Item], pool: &Pool, tok: &Tokenizer, fill_id: u32, k: usize, rng: &mut Rng) -> Group {
    let mut g = Group { seqs: Vec::new(), neg: Vec::new(), kinds: Vec::new(), items: 0, k };
    for it in items {
        let Some(pos) = encode_pair(tok, fill_id, &it.context, &it.truth, false) else { continue };
        let mut hard: Vec<&(NegKind, String)> = it.hard.iter().filter(|(_, t)| t != &it.truth).collect();
        rng.shuffle(&mut hard);
        // prefer the hardest kinds, keep the order random within one kind
        hard.sort_by_key(|(n, _)| std::cmp::Reverse(*n));
        let mut negs: Vec<(NegKind, &str)> = hard.into_iter().take(k).map(|(n, t)| (*n, t.as_str())).collect();
        if let Some(p) = pool.get(&it.kind).filter(|p| !p.is_empty()) {
            let mut tries = 0;
            while negs.len() < k && tries < 20 {
                tries += 1;
                let t = &p[rng.below(p.len())];
                if t != &it.truth {
                    negs.push((NegKind::Other, t));
                }
            }
        }
        if negs.len() < k {
            continue;
        }
        let mut seqs = vec![pos];
        let mut kinds = Vec::new();
        let mut ok = true;
        for (n, t) in &negs {
            match encode_pair(tok, fill_id, &it.context, t, true) {
                Some(s) => {
                    seqs.push(s);
                    kinds.push(*n);
                }
                None => ok = false,
            }
        }
        if !ok {
            continue;
        }
        g.seqs.extend(seqs);
        g.neg.extend(kinds);
        g.kinds.push(it.kind.clone());
        g.items += 1;
    }
    g
}

pub fn to_vec<B: Backend>(t: Tensor<B, 1>) -> Vec<f32> {
    t.into_data().convert::<f32>().to_vec::<f32>().unwrap()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Calib {
    /// Mean raw score of real fills of this kind, and of wrong ones (the swarm's when there
    /// are enough, else all). The reward is 0 at `neg_mean` and 1 at `true_mean`.
    pub true_mean: f32,
    pub neg_mean: f32,
    pub n: usize,
    /// How often the real fill outscored a wrong one.
    pub accuracy: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CriticMeta {
    pub level: String,
    pub calib: BTreeMap<String, Calib>,
    /// Accuracy per negative kind on held-out files.
    pub by_neg: BTreeMap<String, f32>,
    pub val_items: usize,
}

#[derive(Default)]
struct KindAcc {
    n: usize,
    true_sum: f64,
    neg_all: (f64, usize),
    neg_swarm: (f64, usize),
    wins: usize,
    cmps: usize,
}

pub struct EvalOut {
    pub meta: CriticMeta,
    /// Macro mean of the per-negative-kind accuracies: the number checkpoints are chosen by.
    pub score: f32,
}

/// Held-out ranking accuracy per negative kind and the calibration table.
pub fn evaluate(model: &YumonDecBrain<Infer>, items: &[&Item], pool: &Pool, tok: &Tokenizer, fill_id: u32, k: usize,
    batch: usize, level: &str, seed: u64) -> EvalOut {
    let device: <Infer as Backend>::Device = Default::default();
    let mut rng = Rng(seed | 1);
    let mut by_neg: BTreeMap<NegKind, (usize, usize)> = BTreeMap::new();
    let mut acc: BTreeMap<String, KindAcc> = BTreeMap::new();
    for chunk in items.chunks(batch) {
        let g = make_group(chunk, pool, tok, fill_id, k, &mut rng);
        if g.items == 0 {
            continue;
        }
        let s = to_vec(score_batch::<Infer>(model, &g.seqs, &device));
        for i in 0..g.items {
            let row = &s[i * (1 + k)..(i + 1) * (1 + k)];
            let a = acc.entry(g.kinds[i].clone()).or_default();
            a.n += 1;
            a.true_sum += row[0] as f64;
            for j in 0..k {
                let nk = g.neg[i * k + j];
                let win = row[0] > row[1 + j];
                let e = by_neg.entry(nk).or_default();
                e.0 += win as usize;
                e.1 += 1;
                a.wins += win as usize;
                a.cmps += 1;
                a.neg_all.0 += row[1 + j] as f64;
                a.neg_all.1 += 1;
                if nk == NegKind::Swarm {
                    a.neg_swarm.0 += row[1 + j] as f64;
                    a.neg_swarm.1 += 1;
                }
            }
        }
    }
    let mut meta = CriticMeta { level: level.to_owned(), val_items: items.len(), ..Default::default() };
    for (kind, a) in &acc {
        let neg = if a.neg_swarm.1 >= 20 { a.neg_swarm } else { a.neg_all };
        meta.calib.insert(
            kind.clone(),
            Calib {
                true_mean: (a.true_sum / a.n.max(1) as f64) as f32,
                neg_mean: (neg.0 / neg.1.max(1) as f64) as f32,
                n: a.n,
                accuracy: a.wins as f32 / a.cmps.max(1) as f32,
            },
        );
    }
    for (nk, (w, t)) in &by_neg {
        meta.by_neg.insert(nk.name().into(), *w as f32 / (*t).max(1) as f32);
    }
    let score = if meta.by_neg.is_empty() { 0.0 } else { meta.by_neg.values().sum::<f32>() / meta.by_neg.len() as f32 };
    EvalOut { meta, score }
}

pub fn save_critic(model: &YumonDecBrain<Infer>, dir: &Path, tokenizer_json: &Path, meta: &CriticMeta, epochs: usize,
    params: usize, batch: usize, loss: f32) -> Result<()> {
    let rm = RoleMeta {
        role: "critic".into(),
        kinds: Vec::new(),
        rows: vec![EOS],
        epochs_trained: epochs,
        val_loss: None,
        val_top1: None,
        val_top3: None,
        params,
        train_samples: 0,
    };
    save_role(model, dir, tokenizer_json, &rm, batch, loss)?;
    std::fs::write(dir.join("critic.json"), serde_json::to_string_pretty(meta)?)?;
    Ok(())
}

/// A trained critic for inference.
pub struct Critic {
    model: YumonDecBrain<Infer>,
    tok: Tokenizer,
    fill_id: u32,
    device: <Infer as Backend>::Device,
    pub meta: CriticMeta,
}

impl Critic {
    pub fn load(dir: &Path) -> Result<Critic> {
        let device: <Infer as Backend>::Device = Default::default();
        let (model, _, _) = YumonDecBrain::<Infer>::load(dir.to_str().context("path")?, &device)?;
        let tok = load_tokenizer(&dir.join("tokenizer.json"))?;
        let fill_id = encode(&tok, crate::skeleton::FILL)[0];
        let meta: CriticMeta = serde_json::from_str(&std::fs::read_to_string(dir.join("critic.json"))?)?;
        Ok(Critic { model, tok, fill_id, device, meta })
    }

    /// Raw scores for (kind is only used by `reward`) context/fill pairs; None when the
    /// context leaves no room. Over-long fills are clipped.
    pub fn score(&self, pairs: &[(&str, &str)]) -> Vec<Option<f32>> {
        let mut out = vec![None; pairs.len()];
        let enc: Vec<(usize, Vec<u32>)> = pairs
            .iter()
            .enumerate()
            .filter_map(|(i, (c, f))| encode_pair(&self.tok, self.fill_id, c, f, true).map(|s| (i, s)))
            .collect();
        for part in enc.chunks(16) {
            let seqs: Vec<Vec<u32>> = part.iter().map(|(_, s)| s.clone()).collect();
            let v = to_vec(score_batch::<Infer>(&self.model, &seqs, &self.device));
            for ((i, _), x) in part.iter().zip(v) {
                out[*i] = Some(x);
            }
        }
        out
    }

    /// Raw score -> reward: 0 at the typical wrong fill of this kind, 1 at the typical real
    /// one, clamped to [-1.5, 2] so one overconfident score cannot dominate.
    pub fn reward(&self, kind: &str, raw: f32) -> f32 {
        let c = self.meta.calib.get(kind).cloned().unwrap_or(Calib { true_mean: 1.0, neg_mean: 0.0, n: 0, accuracy: 0.0 });
        let span = (c.true_mean - c.neg_mean).abs().max(0.25);
        ((raw - c.neg_mean) / span).clamp(-1.5, 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_below_stays_in_range_and_shuffle_keeps_items() {
        let mut r = Rng(7);
        assert!((0..100).all(|_| r.below(5) < 5));
        let mut v: Vec<u32> = (0..20).collect();
        r.shuffle(&mut v);
        v.sort_unstable();
        assert_eq!(v, (0..20).collect::<Vec<_>>());
    }

    #[test]
    fn unit_is_in_range_and_roughly_uniform() {
        let n = (0..10_000).filter(|&i| unit(3, i) < 0.25).count();
        assert!((2200..2800).contains(&n), "{n}");
    }

    #[test]
    fn program_items_pair_doc_with_body() {
        let c = |f: &str, t: &str| Chunk { file: f.into(), text: t.into() };
        let chunks = vec![
            c("a.rs", "/// Adds.\nfn add(a: u32, b: u32) -> u32 {\n    a + b\n}\n"),
            c("a.rs", "/// Subs.\nfn sub(a: u32, b: u32) -> u32 {\n    a - b\n}\n"),
            c("b.rs", "fn undocumented() {}\n"),
        ];
        let items = program_items(&chunks);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].context, "/// Adds.\n");
        assert!(items[0].truth.starts_with("fn add"));
        assert!(items[0].hard[0].1.starts_with("fn sub"));
    }

    #[test]
    fn negatives_load_by_level_and_skip_equal_ones() {
        let dir = std::env::temp_dir().join("swarm_neg_test.jsonl");
        let rec = |level: &str, bad: &str| {
            serde_json::to_string(&NegRecord {
                level: level.into(), kind: "expr".into(), file: "f".into(), context: "c".into(), truth: "t".into(), bad: bad.into(),
            })
            .unwrap()
        };
        std::fs::write(&dir, format!("{}\n{}\n{}\n", rec("hole", "x"), rec("hole", "t"), rec("program", "y"))).unwrap();
        let items = load_negatives(&dir, "hole").unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].hard[0], (NegKind::Swarm, "x".into()));
    }
}
