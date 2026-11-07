//! WIR decompiles of a dense decoder's SwiGLU MLPs (e.g. yumon-base-dec-sunrise).
//!
//! Each block's MLP w2(silu(w1 x) * w3 x) is a WIR program of `gate` units. The
//! pipeline decompiles it with `wir`, compiles the WIR text back to weights and runs
//! those inside the real model:
//!
//! - `record`: per layer, the SwiGLU weights as a wir weights file
//!   (`layer{l}_swiglu.json`: W1 W3 [ff, d], W2 [d, ff], zero biases) and a
//!   `wir verify` probe of real MLP inputs/outputs (`layer{l}_probe.json`).
//! - `eval`: next-token loss and top-1 agreement of the model with MLPs replaced by
//!   WIR weight files (`--mlp 0=layer0.json`, ReLU or SwiGLU), each layer alone and
//!   all together.
//!
//! Eval samples are spread evenly over the cache.

use anyhow::{Context, Result, ensure};
use burn::{prelude::*, tensor::{TensorData, activation::log_softmax}};
use clap::{Parser, Subcommand};
use serde_json::json;
use std::{collections::BTreeMap, path::{Path, PathBuf}};

use yumon_pet::brain::{
    PAD_TOKEN,
    decoder_model::{WirMlp, YumonDecBrain},
    sample_cache::load_cache,
    samples::{is_plain_text, language_separator_for},
};

type B = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;

#[derive(Parser, Debug)]
#[command(about = "Record, replace and evaluate a dense decoder's MLPs for WIR")]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Args, Debug, Clone)]
struct Common {
    #[arg(long, default_value = "checkpoints/yumon-base-dec-sunrise/code_128h_6l_4a_512ff_256len_b16_DecoderOnly_f32_tied")]
    checkpoint: PathBuf,
    #[arg(long, default_value = "training-cache/pretrain/kilo_pretrain.bin")]
    cache: PathBuf,
    /// Held-out samples for probes and evaluation.
    #[arg(long, default_value_t = 256)]
    eval: usize,
    #[arg(long, default_value_t = 16)]
    batch: usize,
    /// Shift every eval sample index by this much, to score on samples other than
    /// the ones a probe (and anything calibrated on it) came from.
    #[arg(long, default_value_t = 0)]
    offset: usize,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    Record {
        #[command(flatten)]
        common: Common,
        /// Rows per layer in each `wir verify` probe.
        #[arg(long, default_value_t = 2048)]
        probe: usize,
        #[arg(long, default_value = "tmp/wir-sunrise")]
        out: PathBuf,
    },
    Eval {
        #[command(flatten)]
        common: Common,
        /// `layer=weights.json` (wir weights JSON), repeatable.
        #[arg(long = "mlp")]
        mlps: Vec<String>,
    },
}

struct Seq {
    tokens: Vec<i32>,
    /// Next-token targets, PAD where training puts no loss (prompt part and padding).
    targets: Vec<i32>,
}

/// `[prompt][separator][reply]` as in training (`build_moe_batch`; plain text has no
/// separator); loss from the last prompt token on.
fn build(input: &[usize], target: &[usize], sep: &[usize], max: usize) -> Seq {
    let input_len = input.iter().position(|&t| t == PAD_TOKEN).unwrap_or(input.len()).min(max - sep.len() - 1);
    let target_len = target.iter().position(|&t| t == PAD_TOKEN).unwrap_or(target.len());
    let mut tokens: Vec<i32> = input[..input_len].iter().chain(sep).map(|&t| t as i32).collect();
    let room = max - tokens.len();
    tokens.extend(target[..target_len.min(room)].iter().map(|&t| t as i32));
    let mut targets = vec![PAD_TOKEN as i32; tokens.len()];
    for i in input_len.saturating_sub(1)..tokens.len().saturating_sub(1) {
        targets[i] = tokens[i + 1];
    }
    Seq { tokens, targets }
}

/// Model and eval sequences.
fn setup(c: &Common, device: &<B as Backend>::Device) -> Result<(YumonDecBrain<B>, Vec<Seq>)> {
    let (model, tokenizer, config) = YumonDecBrain::<B>::load(c.checkpoint.to_str().unwrap(), device)?;
    let samples = load_cache(&c.cache, &tokenizer, config.training_stage, config.max_seq_len)?;
    let n = samples.len();
    ensure!(c.eval > 0 && c.eval <= n, "cache has {n} samples, asked for {}", c.eval);
    let eval_idx: Vec<usize> = (0..c.eval).map(|i| ((2 * i + 1) * n / (2 * c.eval) + c.offset) % n).collect();
    let sep = tokenizer.encode(language_separator_for(&tokenizer));
    let seq = |i: &usize| {
        let s = &samples[*i];
        let sep = if is_plain_text(s) { &[][..] } else { &sep[..] };
        build(&s.input_ids, &s.target_labels, sep, config.max_seq_len)
    };
    let eval: Vec<Seq> = eval_idx.iter().map(seq).collect();
    let tokens: usize = eval.iter().map(|q| q.tokens.len()).sum();
    println!("{n} cached samples; eval {} ({tokens} tokens)", eval.len());
    Ok((model, eval))
}

/// Right-padded batch tensor and its width.
fn batch_tokens(seqs: &[Seq], device: &<B as Backend>::Device) -> (Tensor<B, 2, Int>, usize) {
    let width = seqs.iter().map(|s| s.tokens.len()).max().unwrap();
    let mut flat = Vec::with_capacity(seqs.len() * width);
    for s in seqs {
        flat.extend(&s.tokens);
        flat.extend(std::iter::repeat(PAD_TOKEN as i32).take(width - s.tokens.len()));
    }
    (Tensor::from_data(TensorData::new(flat, [seqs.len(), width]), device), width)
}

fn floats<const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data().convert::<f32>().to_vec().unwrap()
}

/// Rows of a [batch, width, d] tensor at the real (non-pad) positions.
fn real_rows(t: Tensor<B, 3>, seqs: &[Seq], out: &mut Vec<f32>) {
    let [_, width, d] = t.dims();
    let v = floats(t);
    for (b, s) in seqs.iter().enumerate() {
        out.extend_from_slice(&v[b * width * d..(b * width + s.tokens.len()) * d]);
    }
}

/// [rows, cols] from a Burn [in, out] weight, transposed to nn.Linear [out, in].
fn linear_layout(w: Tensor<B, 2>) -> (Vec<usize>, Vec<f32>) {
    let [i, o] = w.dims();
    (vec![o, i], floats(w.transpose()))
}

fn record(c: &Common, probe: usize, out: &Path) -> Result<()> {
    let device = Default::default();
    let (model, eval_seqs) = setup(c, &device)?;
    std::fs::create_dir_all(out)?;
    let layers = model.config.n_layers;

    // The probe forward must be the model's forward.
    let (t, _) = batch_tokens(&eval_seqs[..c.batch.min(eval_seqs.len())], &device);
    let (probe_logits, _) = model.forward_probe(t.clone(), &[]);
    let diff = floats((probe_logits - model.forward(t)).abs().max());
    println!("forward_probe vs forward: max |diff| {:.3e}", diff[0]);

    let d = model.config.embed_dim;
    for l in 0..layers {
        let (w1, w3, w2) = model.mlp_weights(l);
        let ff = w1.dims()[1];
        let rows = |w: Tensor<B, 2>| -> Vec<Vec<f32>> {
            let (shape, v) = linear_layout(w);
            v.chunks(shape[1]).map(<[f32]>::to_vec).collect()
        };
        let weights = json!({
            "format": "wir-weights",
            "source": format!("{} layer {l} SwiGLU MLP", c.checkpoint.display()),
            "input": "x",
            "W1": rows(w1), "b1": vec![0.0; ff],
            "W3": rows(w3), "b3": vec![0.0; ff],
            "W2": rows(w2), "b2": vec![0.0; d],
        });
        std::fs::write(out.join(format!("layer{l}_swiglu.json")), serde_json::to_string(&weights)?)?;
    }
    println!("wrote {layers} layer weight files");

    // Probes: held-out rows, an even stride over every real eval position.
    let mut px = vec![Vec::new(); layers];
    let mut py = vec![Vec::new(); layers];
    for chunk in eval_seqs.chunks(c.batch) {
        let (t, _) = batch_tokens(chunk, &device);
        let (_, trace) = model.forward_probe(t, &[]);
        for (l, (x, y)) in trace.into_iter().enumerate() {
            real_rows(x, chunk, &mut px[l]);
            real_rows(y, chunk, &mut py[l]);
        }
    }
    let total = px[0].len() / d;
    let pick: Vec<usize> = (0..probe.min(total)).map(|i| i * total / probe.min(total)).collect();
    for l in 0..layers {
        let rows = |v: &[f32]| -> Vec<Vec<f32>> { pick.iter().map(|&r| v[r * d..(r + 1) * d].to_vec()).collect() };
        let probe = json!({
            "source": format!("{} layer {l} MLP (ffn_norm output -> SwiGLU output), held-out eval tokens", c.checkpoint.display()),
            "inputs": rows(&px[l]),
            "outputs": rows(&py[l]),
        });
        std::fs::write(out.join(format!("layer{l}_probe.json")), serde_json::to_string(&probe)?)?;
    }
    println!("wrote probes: {} of {total} held-out rows per layer", pick.len());
    Ok(())
}

fn load_wir_mlp(path: &Path, device: &<B as Backend>::Device) -> Result<WirMlp<B>> {
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).with_context(|| path.display().to_string())?)?;
    let matrix = |k: &str| -> Result<(Vec<f32>, [usize; 2])> {
        let rows = v[k].as_array().with_context(|| format!("{}: no {k}", path.display()))?;
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32)).collect();
        let shape = [rows.len(), flat.len() / rows.len()];
        Ok((flat, shape))
    };
    let vector = |k: &str| -> Result<Vec<f32>> {
        Ok(v[k].as_array().with_context(|| format!("{}: no {k}", path.display()))?.iter().map(|x| x.as_f64().unwrap() as f32).collect())
    };
    let ((w1, s1), (w2, s2)) = (matrix("W1")?, matrix("W2")?);
    let (b1, b2) = (vector("b1")?, vector("b2")?);
    let up = match v.get("W3") {
        Some(_) => {
            let ((w3, s3), b3) = (matrix("W3")?, vector("b3")?);
            Some((
                Tensor::from_data(TensorData::new(w3, s3), device),
                Tensor::from_data(TensorData::new(b3.clone(), [b3.len()]), device),
            ))
        }
        None => None,
    };
    Ok(WirMlp {
        up,
        w1: Tensor::from_data(TensorData::new(w1, s1), device),
        b1: Tensor::from_data(TensorData::new(b1.clone(), [b1.len()]), device),
        w2: Tensor::from_data(TensorData::new(w2, s2), device),
        b2: Tensor::from_data(TensorData::new(b2.clone(), [b2.len()]), device),
    })
}

#[derive(Default)]
struct Score {
    loss_sum: f64,
    loss_n: usize,
    agree: usize,
    positions: usize,
}

/// Loss on training-supervised positions; top-1 agreement with `reference` argmaxes on
/// every real position (`reference` is filled when empty).
fn score(model: &YumonDecBrain<B>, seqs: &[Seq], batch: usize, mlps: &[Option<WirMlp<B>>], reference: &mut Vec<i32>) -> Score {
    let device = Default::default();
    let fill = reference.is_empty();
    let mut s = Score::default();
    let mut pos = 0;
    for chunk in seqs.chunks(batch) {
        let (t, width) = batch_tokens(chunk, &device);
        let (logits, _) = model.forward_probe(t, mlps);
        let logp = log_softmax(logits, 2);
        let top: Vec<i32> = logp.clone().argmax(2).into_data().convert::<i32>().to_vec().unwrap();
        let mut tg = Vec::with_capacity(chunk.len() * width);
        for q in chunk {
            tg.extend(q.targets.iter().map(|&t| t.max(0)));
            tg.extend(std::iter::repeat(0).take(width - q.targets.len()));
        }
        let tg = Tensor::<B, 3, Int>::from_data(TensorData::new(tg, [chunk.len(), width, 1]), &device);
        let picked = floats(logp.gather(2, tg));
        for (b, q) in chunk.iter().enumerate() {
            for i in 0..q.tokens.len() {
                if q.targets[i] != PAD_TOKEN as i32 {
                    s.loss_sum -= picked[b * width + i] as f64;
                    s.loss_n += 1;
                }
                let a = top[b * width + i];
                if fill {
                    reference.push(a);
                }
                s.agree += (reference[pos] == a) as usize;
                s.positions += 1;
                pos += 1;
            }
        }
    }
    s
}

fn eval(c: &Common, specs: &[String]) -> Result<()> {
    let device = Default::default();
    let (model, eval_seqs) = setup(c, &device)?;
    let layers = model.config.n_layers;
    let mut replacements: BTreeMap<usize, WirMlp<B>> = BTreeMap::new();
    for s in specs {
        let (l, p) = s.split_once('=').with_context(|| format!("--mlp `{s}` should be layer=path"))?;
        let l: usize = l.parse()?;
        ensure!(l < layers, "layer {l} out of range (model has {layers})");
        replacements.insert(l, load_wir_mlp(Path::new(p), &device)?);
    }
    let mut reference = Vec::new();
    let base = score(&model, &eval_seqs, c.batch, &[], &mut reference);
    let base_loss = base.loss_sum / base.loss_n as f64;
    println!("{} eval positions, {} supervised", base.positions, base.loss_n);
    println!("{:<14} {:>9} {:>9} {:>10}", "model", "loss", "d loss", "top-1 same");
    println!("{:<14} {:>9.4} {:>9} {:>10}", "original", base_loss, "-", "100.00%");
    let report = |name: &str, s: &Score| {
        let loss = s.loss_sum / s.loss_n as f64;
        println!("{:<14} {:>9.4} {:>+9.4} {:>9.2}%", name, loss, loss - base_loss, 100.0 * s.agree as f64 / s.positions as f64);
    };
    let keys: Vec<usize> = replacements.keys().copied().collect();
    let take = |only: Option<usize>, r: &mut BTreeMap<usize, WirMlp<B>>| -> Vec<Option<WirMlp<B>>> {
        (0..layers).map(|l| if only.is_none_or(|o| o == l) { r.remove(&l) } else { None }).collect()
    };
    for &l in &keys {
        let mlps = take(Some(l), &mut replacements);
        report(&format!("layer {l} WIR"), &score(&model, &eval_seqs, c.batch, &mlps, &mut reference));
        for (k, m) in mlps.into_iter().enumerate() {
            if let Some(m) = m {
                replacements.insert(k, m);
            }
        }
    }
    if keys.len() > 1 {
        let mlps = take(None, &mut replacements);
        report(&format!("{} layers WIR", keys.len()), &score(&model, &eval_seqs, c.batch, &mlps, &mut reference));
    }
    Ok(())
}

fn main() -> Result<()> {
    match Args::parse().cmd {
        Cmd::Record { common, probe, out } => record(&common, probe, &out),
        Cmd::Eval { common, mlps } => eval(&common, &mlps),
    }
}
