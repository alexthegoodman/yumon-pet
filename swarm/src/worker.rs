//! Phase 2: trained workers. Per-role data (prompt = BOS context `<fill>`, then the fill
//! and EOS, loss on the fill only), the role's output rows, batching, and the inference
//! side (`Worker`: greedy fill of one hole) that plugs into the dispatcher as a `Filler`.
//!
//! A role reads the full 2,073-token vocabulary but its logits are only the `rows` of
//! the tied table, so every role checkpoint is a whole small decoder (roles may differ
//! in size) plus `role.json` with its rows.
use crate::{
    dataset::{self, Chunk, FILL_SEP},
    roles::{ModelSpec, Role},
    skeleton::{self, KINDS, Node},
    swarm::{Filler, Oracle, Request},
};
use anyhow::{Context, Result, ensure};
use burn::{prelude::*, tensor::backend::AutodiffBackend};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokenizers::Tokenizer;
use yumon_pet::brain::{
    bpe::{BpeTokenizer, TokenizerKind},
    decoder_model::{DecMetadata, YumonDecBrain, YumonDecBrainConfig},
    jl::JlSet,
    samples::TrainingStage,
};

pub type Train = yumon_pet::brain::train::TrainBackend;
pub type Infer = <Train as AutodiffBackend>::InnerBackend;

/// Base code BPE ids plus the 25 swarm specials.
pub const VOCAB: usize = 2048 + 25;
pub const PAD: u32 = 0;
pub const BOS: u32 = 1;
pub const EOS: u32 = 2;
pub const CONTEXT: usize = 512;

pub fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
}

pub fn encode(tok: &Tokenizer, text: &str) -> Vec<u32> {
    tok.encode(text, false).map(|e| e.get_ids().to_vec()).unwrap_or_default()
}

/// One training example: `[BOS] context [<fill>] fill [EOS]`; loss starts at `prompt_len`.
#[derive(Debug, Clone)]
pub struct Sample {
    pub ids: Vec<u16>,
    pub prompt_len: usize,
    pub kind: u8,
    /// Every fill token is among the role's rows (always true for training samples).
    pub in_role: bool,
}

impl Sample {
    pub fn fill(&self) -> &[u16] {
        &self.ids[self.prompt_len..self.ids.len() - 1]
    }
    pub fn prompt(&self) -> Vec<u32> {
        self.ids[..self.prompt_len].iter().map(|&t| t as u32).collect()
    }
}

/// The role's rows (ids into the full table) and the inverse map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rows {
    pub ids: Vec<u32>,
}

impl Rows {
    /// EOS, every hole marker, then the most frequent target tokens, `n` in total.
    /// `n >= VOCAB` keeps the whole table.
    pub fn select(hist: &[u64], n: usize) -> Rows {
        if n >= VOCAB {
            return Rows { ids: (0..VOCAB as u32).collect() };
        }
        let mut ids = vec![EOS];
        for k in KINDS {
            ids.push(2048 + KINDS.iter().position(|x| x == &k).unwrap() as u32);
        }
        let mut order: Vec<u32> = (0..VOCAB as u32).filter(|&t| hist[t as usize] > 0 && !ids.contains(&t)).collect();
        order.sort_by_key(|&t| std::cmp::Reverse(hist[t as usize]));
        ids.extend(order.into_iter().take(n.saturating_sub(ids.len())));
        ids.sort_unstable();
        Rows { ids }
    }
    pub fn position(&self) -> Vec<i32> {
        let mut inv = vec![-1; VOCAB];
        for (i, &t) in self.ids.iter().enumerate() {
            inv[t as usize] = i as i32;
        }
        inv
    }
    pub fn eos_index(&self) -> usize {
        self.ids.iter().position(|&t| t == EOS).expect("EOS is always a row")
    }
}

/// Per-role data cut from `code.bin` chunks: training files and held-out files.
pub struct RoleData {
    pub kinds: Vec<String>,
    pub rows: Rows,
    pub train: Vec<Sample>,
    pub val: Vec<Sample>,
    /// Training fills with a token outside the rows, dropped from training.
    pub train_out_of_role: usize,
    /// Samples over the 512 context, dropped everywhere.
    pub too_long: usize,
}

struct Raw {
    ids: Vec<u16>,
    prompt_len: usize,
    kind: u8,
    val: bool,
}

impl RoleData {
    pub fn build(chunks: &[Chunk], tok: &Tokenizer, role: &Role) -> Result<RoleData> {
        let fill_id = encode(tok, FILL_SEP);
        ensure!(fill_id.len() == 1, "{FILL_SEP} is not one token; run make_tokenizer");
        let fill_id = fill_id[0];
        let kinds: Vec<u8> = role
            .kinds
            .iter()
            .map(|k| KINDS.iter().position(|x| x == k).map(|i| i as u8).with_context(|| format!("kind {k}")))
            .collect::<Result<_>>()?;
        let raws: Vec<Raw> = chunks
            .par_iter()
            .flat_map_iter(|chunk| {
                let val = dataset::is_validation(&chunk.file);
                let mut out = Vec::new();
                let Ok(root) = dataset::extract_chunk(chunk) else { return out };
                for inst in dataset::instances(&root) {
                    let Some(k) = KINDS.iter().position(|x| *x == inst.kind) else { continue };
                    if !kinds.contains(&(k as u8)) {
                        continue;
                    }
                    let mut ids = vec![BOS];
                    ids.extend(encode(tok, &inst.context));
                    ids.push(fill_id);
                    let prompt_len = ids.len();
                    ids.extend(encode(tok, &inst.target));
                    ids.push(EOS);
                    out.push(Raw { ids: ids.iter().map(|&t| t as u16).collect(), prompt_len, kind: k as u8, val });
                }
                out
            })
            .collect();

        let mut hist = vec![0u64; VOCAB];
        for r in raws.iter().filter(|r| !r.val) {
            for &t in &r.ids[r.prompt_len..] {
                hist[t as usize] += 1;
            }
        }
        let rows = Rows::select(&hist, role.out_vocab);
        let pos = rows.position();
        let (mut train, mut val) = (Vec::new(), Vec::new());
        let (mut out_of_role, mut too_long) = (0, 0);
        for r in raws {
            if r.ids.len() - 1 > CONTEXT {
                too_long += 1;
                continue;
            }
            let in_role = r.ids[r.prompt_len..].iter().all(|&t| pos[t as usize] >= 0);
            let s = Sample { ids: r.ids, prompt_len: r.prompt_len, kind: r.kind, in_role };
            if r.val {
                val.push(s);
            } else if in_role {
                train.push(s);
            } else {
                out_of_role += 1;
            }
        }
        Ok(RoleData { kinds: role.kinds.clone(), rows, train, val, train_out_of_role: out_of_role, too_long })
    }
}

/// A right-padded batch: inputs `[b, len]`, targets as row indices, and a 0/1 mask over
/// fill positions (the prompt carries no loss).
pub struct Batch {
    pub rows: usize,
    pub len: usize,
    pub inputs: Vec<i32>,
    pub targets: Vec<i32>,
    pub mask: Vec<f32>,
    pub supervised: usize,
}

pub fn make_batch(samples: &[&Sample], pos: &[i32]) -> Batch {
    let len = samples.iter().map(|s| s.ids.len() - 1).max().unwrap_or(1).max(1);
    let b = samples.len();
    let (mut inputs, mut targets, mut mask) = (vec![PAD as i32; b * len], vec![0i32; b * len], vec![0f32; b * len]);
    let mut supervised = 0;
    for (r, s) in samples.iter().enumerate() {
        for i in 0..s.ids.len() - 1 {
            inputs[r * len + i] = s.ids[i] as i32;
            // position i predicts ids[i + 1]; supervised from the first fill token on
            if i + 1 >= s.prompt_len {
                let p = pos[s.ids[i + 1] as usize];
                if p >= 0 {
                    targets[r * len + i] = p;
                    mask[r * len + i] = 1.0;
                    supervised += 1;
                }
            }
        }
    }
    Batch { rows: b, len, inputs, targets, mask, supervised }
}

/// Masked fill loss and the number of supervised targets that are top-1 / top-3.
pub struct Scored<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub top1: Tensor<B, 1>,
    pub top3: Tensor<B, 1>,
    pub supervised: usize,
}

pub fn score<B: Backend>(model: &YumonDecBrain<B>, batch: &Batch, rows: &Tensor<B, 1, Int>, device: &B::Device) -> Scored<B>
where
    B: yumon_pet::brain::flash_attn::backend::FlashAttention,
{
    let n = batch.rows * batch.len;
    let inputs = Tensor::<B, 2, Int>::from_data(TensorData::new(batch.inputs.clone(), [batch.rows, batch.len]), device);
    let targets = Tensor::<B, 1, Int>::from_data(TensorData::new(batch.targets.clone(), [n]), device);
    let mask = Tensor::<B, 1>::from_data(TensorData::new(batch.mask.clone(), [n]), device);
    let width = rows.dims()[0];
    let logits = model.logits_rows(inputs, rows.clone()).reshape([n, width]);
    let log_probs = burn::tensor::activation::log_softmax(logits.clone(), 1);
    let nll = log_probs.gather(1, targets.clone().reshape([n, 1])).reshape([n]).neg();
    let loss = (nll * mask.clone()).sum() / batch.supervised.max(1) as f64;
    // rank of the target among the logits: how many are strictly greater
    let logits = logits.detach();
    let tscore = logits.clone().gather(1, targets.reshape([n, 1]));
    let above = logits.greater(tscore).int().sum_dim(1).reshape([n]);
    let top1 = above.clone().lower_elem(1).float().mul(mask.clone()).sum();
    let top3 = above.lower_elem(3).float().mul(mask).sum();
    Scored { loss, top1, top3, supervised: batch.supervised }
}

// ---------------------------------------------------------------------------------
// model construction and checkpoints

/// `role.json` next to a role's `model.bin`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleMeta {
    pub role: String,
    pub kinds: Vec<String>,
    pub rows: Vec<u32>,
    pub epochs_trained: usize,
    pub val_loss: Option<f32>,
    pub val_top1: Option<f32>,
    pub val_top3: Option<f32>,
    pub params: usize,
    pub train_samples: usize,
}

/// The shape a role starts from: its own spec or the set's base checkpoint.
pub fn init_model(role: &Role, base: &str, device: &<Train as Backend>::Device) -> Result<(YumonDecBrain<Train>, usize)> {
    match &role.model {
        Some(ModelSpec { checkpoint: None, embed_dim, n_layers, attn_heads, ff_dim, token_embed_dim, jl_ffn }) => {
            let mut config = YumonDecBrainConfig::new(VOCAB, TrainingStage::Language)
                .with_embed_dim(*embed_dim)
                .with_hidden_units(*embed_dim)
                .with_n_layers(*n_layers)
                .with_attn_heads(*attn_heads)
                .with_ff_dim(*ff_dim)
                .with_max_seq_len(CONTEXT)
                .with_dropout_rate(0.05)
                .with_tie_embeddings(true)
                .with_token_embed_dim(Some(*token_embed_dim));
            if *jl_ffn > 0 {
                config = config.with_jl(Some(JlSet::parse(&format!("ffn:{jl_ffn}@0"))?));
            }
            let model = config.init::<Train>(device);
            let params = config.param_count();
            Ok((model, params))
        }
        spec => {
            let dir = spec.as_ref().and_then(|m| m.checkpoint.clone()).unwrap_or_else(|| base.to_owned());
            let (model, _, cfg) = YumonDecBrain::<Train>::load(&dir, device).with_context(|| format!("loading {dir}"))?;
            ensure!(cfg.tie_embeddings, "{dir}: the checkpoint is not tied; roles need a tied table");
            let model = model.grow_vocab(VOCAB, 0.02, device);
            let mut cfg = cfg;
            cfg.vocab_size = VOCAB;
            Ok((model, cfg.param_count()))
        }
    }
}

pub fn save_role(model: &YumonDecBrain<Infer>, dir: &Path, tokenizer_json: &Path, meta: &RoleMeta,
    batch_size: usize, final_loss: f32) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::copy(tokenizer_json, dir.join("tokenizer.json"))?;
    let c = &model.config.0;
    let md = DecMetadata {
        dropout_rate: c.dropout_rate,
        vocab_size: c.vocab_size,
        epochs_trained: meta.epochs_trained,
        final_loss,
        val_loss: meta.val_loss,
        batch_size,
        training_stage: c.training_stage,
        embed_dim: c.embed_dim,
        hidden_units: c.hidden_units,
        n_layers: c.n_layers,
        attn_heads: c.attn_heads,
        ff_dim: c.ff_dim,
        max_seq_len: c.max_seq_len,
        jl: c.jl.clone(),
        tie_embeddings: c.tie_embeddings,
        token_embed_dim: c.token_embed_dim,
    };
    let tok = TokenizerKind::Bpe(BpeTokenizer::load(dir.to_str().context("path")?)?);
    model.save(dir.to_str().context("path")?, &tok, &md)?;
    std::fs::write(dir.join("role.json"), serde_json::to_string_pretty(meta)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------------
// inference

/// A trained role: model, rows and tokenizer. Writes one hole at a time, greedily.
pub struct Worker {
    pub meta: RoleMeta,
    model: YumonDecBrain<Infer>,
    tok: Tokenizer,
    rows: Vec<u32>,
    fill_id: u32,
    device: <Infer as Backend>::Device,
}

impl Worker {
    pub fn load(dir: &Path) -> Result<Worker> {
        let device: <Infer as Backend>::Device = Default::default();
        let meta: RoleMeta = serde_json::from_str(&std::fs::read_to_string(dir.join("role.json"))?)?;
        let (model, _, cfg) = YumonDecBrain::<Infer>::load(dir.to_str().context("path")?, &device)?;
        ensure!(cfg.vocab_size == VOCAB, "{}: vocab {} (expected {VOCAB})", dir.display(), cfg.vocab_size);
        let tok = load_tokenizer(&dir.join("tokenizer.json"))?;
        let fill_id = encode(&tok, FILL_SEP)[0];
        Ok(Worker { rows: meta.rows.clone(), meta, model, tok, fill_id, device })
    }

    pub fn handles(&self, kind: &str) -> bool {
        self.meta.kinds.iter().any(|k| k == kind)
    }

    /// Greedy fill for a context (`<target_KIND>` marks the hole); `None` when the
    /// prompt leaves no room.
    pub fn fill_ids(&self, prompt: &[u32]) -> Option<Vec<u32>> {
        if prompt.len() + 2 > CONTEXT {
            return None;
        }
        let max_new = CONTEXT - prompt.len() - 1;
        Some(self.model.greedy_rows(prompt, &self.rows, EOS, max_new, &self.device))
    }

    pub fn prompt_for(&self, context: &str) -> Vec<u32> {
        let mut p = vec![BOS];
        p.extend(encode(&self.tok, context));
        p.push(self.fill_id);
        p
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.tok.decode(ids, false).unwrap_or_default()
    }

    pub fn fill(&self, context: &str) -> Option<String> {
        let ids = self.fill_ids(&self.prompt_for(context))?;
        Some(self.decode(&ids))
    }
}

/// Dispatcher filler: each hole goes to the worker that owns its kind. Kinds without a
/// worker are given up on (or written by `truth` when it is set, for role-by-role tests).
pub struct ModelFiller<'a> {
    pub workers: &'a [Worker],
    pub truth: Option<Oracle<'a>>,
}

impl Filler for ModelFiller<'_> {
    fn fill(&mut self, req: &Request) -> Option<String> {
        match self.workers.iter().find(|w| w.handles(req.kind)) {
            Some(w) => w.fill(req.context),
            None => self.truth.as_mut().and_then(|t| t.fill(req)),
        }
    }
    fn name(&self, kind: &str) -> &str {
        self.workers.iter().find(|w| w.handles(kind)).map_or("oracle", |w| w.meta.role.as_str())
    }
}

/// Oracle everywhere except one hole, which the worker writes.
pub struct WorkerSwap<'a> {
    pub truth: Oracle<'a>,
    pub path: Vec<usize>,
    pub worker: &'a Worker,
}

impl Filler for WorkerSwap<'_> {
    fn fill(&mut self, req: &Request) -> Option<String> {
        if req.path == self.path.as_slice() { self.worker.fill(req.context) } else { self.truth.fill(req) }
    }
    fn name(&self, kind: &str) -> &str {
        if self.worker.handles(kind) { "swap" } else { "oracle" }
    }
}

/// Context text of a hole in a ground-truth tree (what the dispatcher would send).
pub fn context_of(root: &Node, path: &[usize]) -> String {
    root.context(path)
}

#[allow(dead_code)]
fn _kinds_used() {
    let _ = skeleton::KINDS;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_keep_eos_markers_and_the_frequent() {
        let mut hist = vec![0u64; VOCAB];
        hist[10] = 5;
        hist[11] = 9;
        hist[12] = 1;
        let r = Rows::select(&hist, 16);
        assert!(r.ids.contains(&EOS));
        for m in 2048..2060 {
            assert!(r.ids.contains(&m));
        }
        assert!(r.ids.contains(&11) && r.ids.contains(&10));
        assert!(r.ids.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(Rows::select(&hist, VOCAB).ids.len(), VOCAB);
    }

    #[test]
    fn batch_masks_the_prompt() {
        // BOS a b <fill> x y EOS
        let s = Sample { ids: vec![1, 40, 41, 2060, 50, 51, 2], prompt_len: 4, kind: 0, in_role: true };
        let rows = Rows { ids: vec![2, 50, 51] };
        let pos = rows.position();
        let b = make_batch(&[&s], &pos);
        assert_eq!(b.len, 6);
        assert_eq!(b.supervised, 3);
        // position 3 (<fill>) predicts x, 4 predicts y, 5 predicts EOS
        assert_eq!(b.mask, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        assert_eq!(&b.targets[3..], &[1, 2, 0]);
    }

    fn tiny() -> YumonDecBrain<Train> {
        let role = Role {
            name: "t".into(), kinds: vec!["block".into()], crop_tokens: 448, out_vocab: 64, written: false,
            model: Some(ModelSpec { checkpoint: None, embed_dim: 16, n_layers: 2, attn_heads: 2, ff_dim: 32,
                token_embed_dim: 16, jl_ffn: 0 }),
            train: Default::default(),
        };
        init_model(&role, "", &Default::default()).unwrap().0
    }

    fn vec_of<B: Backend>(t: Tensor<B, 3>) -> Vec<f32> {
        t.into_data().convert::<f32>().to_vec::<f32>().unwrap()
    }

    #[test]
    fn greedy_matches_the_full_forward() {
        use burn::module::AutodiffModule;
        let model = tiny().valid();
        let device: <Infer as Backend>::Device = Default::default();
        let rows: Vec<u32> = (0..VOCAB as u32).step_by(7).collect();
        let rows_t = Tensor::<Infer, 1, Int>::from_ints(rows.iter().map(|&r| r as i32).collect::<Vec<_>>().as_slice(), &device);
        let mut seq: Vec<u32> = vec![1, 40, 41, 300, 2060];
        let got = model.greedy_rows(&seq, &rows, u32::MAX, 6, &device);
        assert_eq!(got.len(), 6);
        for want in got {
            let t = seq.len();
            let ids = Tensor::<Infer, 2, Int>::from_data(
                TensorData::new(seq.iter().map(|&x| x as i32).collect::<Vec<_>>(), [1, t]), &device);
            let logits = vec_of(model.logits_rows(ids, rows_t.clone()));
            let last = &logits[(t - 1) * rows.len()..];
            let best = (0..rows.len()).max_by(|&a, &b| last[a].partial_cmp(&last[b]).unwrap()).unwrap();
            assert_eq!(rows[best], want, "cached greedy and full forward disagree at length {t}");
            seq.push(want);
        }
    }

    #[test]
    fn growing_the_vocab_keeps_old_rows() {
        use burn::module::AutodiffModule;
        let device: <Train as Backend>::Device = Default::default();
        let model = tiny();
        let base = model.clone().valid();
        let grown = model.grow_vocab(VOCAB + 5, 0.02, &device).valid();
        let rows: Vec<i32> = (0..VOCAB as i32).collect();
        let rows_t = Tensor::<Infer, 1, Int>::from_ints(rows.as_slice(), &device);
        let ids = Tensor::<Infer, 2, Int>::from_ints([[1, 5, 9, 12]], &device);
        let a = vec_of(base.logits_rows(ids.clone(), rows_t.clone()));
        let b = vec_of(grown.logits_rows(ids, rows_t));
        assert_eq!(a.len(), b.len());
        let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "old rows changed by {worst}");
    }
}
