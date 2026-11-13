//! Single-model baseline for phase 3: the sugar-maniac checkpoint (one ~187k decoder,
//! no swarm) gets the same held-out doc comments as `run_swarm` and writes the whole item.
//! Same retry budget: greedy first, then sampled tries, first one that `syn` parses wins.
//!
//! cargo run --release --bin baseline -- [--held-out 25] [--seed 0] [--retries 4]
//!   [--temp 0.8] [--top-k 40] [--ckpt ../checkpoints/yumon-code-dec-sugar-maniac]
use anyhow::{Context, Result};
use std::path::PathBuf;
use swarm::{
    check, dataset,
    planner::doc_prefix,
    worker::{BOS, EOS, Infer, encode, load_tokenizer},
};
use yumon_pet::brain::decoder_model::YumonDecBrain;

fn pct(a: usize, b: usize) -> f64 {
    if b == 0 { 0.0 } else { 100.0 * a as f64 / b as f64 }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let n: usize = arg("--held-out").map(|v| v.parse()).transpose()?.unwrap_or(25);
    let seed: usize = arg("--seed").map(|v| v.parse()).transpose()?.unwrap_or(0);
    let retries: usize = arg("--retries").map(|v| v.parse()).transpose()?.unwrap_or(4);
    let temp: f32 = arg("--temp").map(|v| v.parse()).transpose()?.unwrap_or(0.8);
    let top_k: usize = arg("--top-k").map(|v| v.parse()).transpose()?.unwrap_or(40);
    let ckpt = arg("--ckpt").unwrap_or("../checkpoints/yumon-code-dec-sugar-maniac".into());
    let scratch = PathBuf::from(arg("--scratch").unwrap_or("out/scratch".into()));

    let device: <Infer as burn::tensor::backend::Backend>::Device = Default::default();
    let (model, _, cfg) = YumonDecBrain::<Infer>::load(&ckpt, &device).with_context(|| ckpt.clone())?;
    println!("baseline {ckpt}: vocab {}, {} params", cfg.vocab_size, cfg.param_count());
    let tok = load_tokenizer(&PathBuf::from("tokenizer.json"))?;
    let rows: Vec<u32> = (0..cfg.vocab_size as u32).collect();

    let chunks = dataset::load_chunks(&PathBuf::from("../training-cache/code.bin"), &PathBuf::from("tokenizer.json"), None)?;
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

    let (mut parsed, mut tries, mut ended) = (0, 0, 0);
    let mut texts = Vec::new();
    let mut md = String::from("# Baseline samples (sugar-maniac alone)\n\n");
    for (i, c) in picks.iter().enumerate() {
        let doc = doc_prefix(&c.text);
        let mut prompt = vec![BOS];
        prompt.extend(encode(&tok, doc));
        let max_new = 511 - prompt.len();
        let mut best = String::new();
        let mut ok = false;
        for attempt in 0..=retries {
            tries += 1;
            let ids = if attempt == 0 {
                model.greedy_rows(&prompt, &rows, EOS, max_new, &device)
            } else {
                let mut s = 0xcbf2_9ce4_8422_2325u64;
                for &t in &prompt { s = (s ^ t as u64).wrapping_mul(0x100_0000_01b3); }
                s ^= (attempt as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                model.sample_rows(&prompt, &rows, EOS, max_new, temp, top_k, s, &device)
            };
            let hit_eos = ids.len() < max_new;
            best = format!("{doc}{}", tok.decode(&ids, false).unwrap_or_default());
            if syn::parse_file(&best).is_ok() {
                ok = true;
                if hit_eos { ended += 1; }
                break;
            }
        }
        println!("[{:>3}/{}] parses {ok}", i + 1, picks.len());
        if ok {
            parsed += 1;
            texts.push((i, best.clone()));
        }
        md.push_str(&format!("## {i}: parses {ok}\n\n```rust\n{best}\n```\n\noriginal:\n\n```rust\n{}\n```\n\n", c.text));
    }
    let mut clean = 0;
    let mut codes = std::collections::BTreeMap::<String, usize>::new();
    if !texts.is_empty() {
        let just: Vec<String> = texts.iter().map(|t| t.1.clone()).collect();
        let report = check::check_all(&scratch, &just)?;
        clean = just.len() - report.len();
        for es in report.values() {
            for e in es { *codes.entry(e.code.clone()).or_default() += 1; }
        }
    }
    println!("\ndocs {}: syn-parse {parsed} ({:.0}%), ended with EOS {ended}, cargo-check clean {clean} ({:.0}% of all), decodes {tries}",
        picks.len(), pct(parsed, picks.len()), pct(clean, picks.len()));
    println!("error codes: {codes:?}");
    std::fs::create_dir_all("out/phase3")?;
    std::fs::write("out/phase3/baseline.md", md)?;
    Ok(())
}
