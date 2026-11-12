//! Print fills that do not parse in their wrapper, for debugging the extractor.
//! cargo run --release --bin show_failures -- [--limit N] [--kind pat] [--show 15]
use anyhow::Result;
use std::path::PathBuf;
use swarm::{dataset, skeleton};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let limit = arg("--limit").map(|v| v.parse()).transpose()?.unwrap_or(3000usize);
    let kind = arg("--kind");
    let show: usize = arg("--show").map(|v| v.parse()).transpose()?.unwrap_or(15);
    let cache = PathBuf::from("../training-cache/code.bin");
    let tokenizer = PathBuf::from(arg("--tokenizer").unwrap_or(
        "../checkpoints/yumon-code-dec-sugar-maniac/code_32h_4l_2a_128ff_512len_b16_DecoderOnly_f32_curriculum_documented_tiers_v3_tok64_jl_ffn16_at0_tied/tokenizer.json".into(),
    ));
    let chunks = dataset::load_chunks(&cache, &tokenizer, Some(limit))?;
    let mut shown = 0;
    for chunk in &chunks {
        let Ok(root) = dataset::extract_chunk(chunk) else { continue };
        root.walk(&mut Vec::new(), &mut |n, path| {
            if shown >= show || path.is_empty() || kind.as_deref().is_some_and(|k| k != n.kind) {
                return;
            }
            let fill = n.assemble();
            if !skeleton::fill_parses(&n.kind, &fill) {
                shown += 1;
                println!("[fill {}] {:?}", n.kind, fill);
            } else if !skeleton::skeleton_parses(n) {
                shown += 1;
                println!("[skeleton {}] {:?}", n.kind, n.skeleton());
            }
        });
    }
    Ok(())
}
