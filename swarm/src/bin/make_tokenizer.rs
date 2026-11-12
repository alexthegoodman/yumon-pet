//! Build `swarm/tokenizer.json`: the code BPE plus the swarm special tokens
//! (one open-hole token and one target token per kind, and the fill separator),
//! appended as ids 2048.. so every base id is unchanged.
//!
//! cargo run --release --bin make_tokenizer -- [--base PATH] [--out tokenizer.json]
use anyhow::{Result, ensure};
use swarm::skeleton::special_tokens;
use tokenizers::{AddedToken, Tokenizer};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let base = arg("--base").unwrap_or(
        "../yumon_code_bpe/tokenizer.json".into(),
    );
    let out = arg("--out").unwrap_or("tokenizer.json".into());
    let mut tok = Tokenizer::from_file(&base).map_err(|e| anyhow::anyhow!("{e}"))?;
    let base_size = tok.get_vocab_size(true);
    ensure!(base_size == 2048, "expected the 2,048-token code BPE, got {base_size}");
    let specials = special_tokens();
    let added: Vec<AddedToken> = specials.iter().map(|s| AddedToken::from(s.as_str(), true)).collect();
    ensure!(tok.add_special_tokens(&added) == specials.len(), "some tokens already existed");
    for (i, s) in specials.iter().enumerate() {
        ensure!(tok.token_to_id(s) == Some((base_size + i) as u32), "{s} did not get id {}", base_size + i);
        let ids = tok.encode(s.as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?;
        ensure!(ids.get_ids() == [(base_size + i) as u32], "{s} is not one token: {:?}", ids.get_ids());
    }
    tok.save(&out, false).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("wrote {out}: {} + {} special tokens = {}", base_size, specials.len(), tok.get_vocab_size(true));
    println!("ids {}..{}: {:?}", base_size, base_size + specials.len() - 1, specials);
    Ok(())
}
