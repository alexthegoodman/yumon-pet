//! Read-only inspection of any prepared Rust training cache.
#[cfg(not(target_arch = "wasm32"))]
use {
    anyhow::{Result, ensure},
    clap::Parser,
    std::path::PathBuf,
    yumon_pet::brain::{
        bpe::TokenizerKind,
        code_corpus,
        sample_cache::{self, CacheObjective},
    },
};

#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser)]
#[command(about = "Print source samples from a Rust complexity bucket without modifying it")]
struct Args {
    /// Bucket .bin file (also accepts the original code.bin).
    #[arg(long, default_value = "training-cache/code_buckets/tier_1_low.bin")]
    bucket: PathBuf,
    /// Matching code tokenizer directory.
    #[arg(long, default_value = "yumon_code_bpe")]
    tokenizer: String,
    /// Number of source samples to print.
    #[arg(long, default_value_t = 5)]
    limit: usize,
    /// Skip this many samples in the bucket's original order.
    #[arg(long, default_value_t = 0)]
    offset: usize,
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<()> {
    let args = Args::parse();
    let metadata = sample_cache::read_metadata(&args.bucket)?;
    ensure!(
        metadata.objective == CacheObjective::RawCode,
        "requires a raw-code cache"
    );
    println!(
        "Bucket: {}\nSamples: {} | context: {} | seed: {}",
        args.bucket.display(),
        metadata.samples,
        metadata.max_seq_len,
        metadata.seed
    );
    ensure!(
        args.offset < metadata.samples,
        "--offset must be smaller than {}",
        metadata.samples
    );
    if args.limit == 0 {
        return Ok(());
    }
    let tokenizer = code_corpus::load_tokenizer(&args.tokenizer)?;
    let samples = sample_cache::load_cache_with_objective(
        &args.bucket,
        &tokenizer,
        metadata.stage,
        metadata.max_seq_len,
        CacheObjective::RawCode,
    )?;
    let TokenizerKind::Bpe(tokenizer) = tokenizer else {
        anyhow::bail!("requires a code BPE tokenizer")
    };
    for (index, sample) in samples
        .iter()
        .enumerate()
        .skip(args.offset)
        .take(args.limit)
    {
        let ids = sample
            .target_labels
            .iter()
            .map(|&id| u32::try_from(id))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let source = tokenizer.decode(&ids)?;
        println!(
            "\n=== Sample {} / {} | {} | {} ===\n{}",
            index + 1,
            metadata.samples,
            sample.pair.0,
            sample.pair.1,
            source
        );
    }
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("Code bucket inspection requires a native build.");
}
