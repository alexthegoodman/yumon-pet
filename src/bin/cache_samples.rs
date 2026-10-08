#[cfg(not(target_arch = "wasm32"))]
use anyhow::{Context, Result};
#[cfg(not(target_arch = "wasm32"))]
use clap::{Parser, ValueEnum};
#[cfg(not(target_arch = "wasm32"))]
use yumon_pet::brain::{
    bpe::{BpeTokenizer, TokenizerKind},
    sample_cache,
    samples::TrainingStage,
    train::{build_keyword_index, build_label_keywords, stage_data_loader},
};

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, ValueEnum)]
enum Stage {
    Language,
    Structured,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser)]
#[command(about = "Prepare training samples locally for cached Docker/RunPod training")]
struct Args {
    #[arg(long, default_value = "training-cache/samples.bin")]
    output: std::path::PathBuf,
    #[arg(long, default_value = "yumon_bpe")]
    tokenizer: String,
    #[arg(long, value_enum, default_value = "language")]
    stage: Stage,
    /// Must match the training grid's context length.
    #[arg(long, default_value_t = 256)]
    max_seq_len: usize,
    #[arg(long, default_value_t = 4815162342)]
    seed: u64,
    /// Optional cap after the usual merge, shuffle, and deduplication.
    #[arg(long)]
    limit: Option<usize>,
    /// Print metadata for an existing output file without loading corpora or a tokenizer.
    #[arg(long)]
    inspect: bool,
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<()> {
    let args = Args::parse();
    if args.inspect {
        println!(
            "{}",
            serde_json::to_string_pretty(&sample_cache::read_metadata(&args.output)?)?
        );
        return Ok(());
    }
    anyhow::ensure!(args.max_seq_len >= 2, "--max-seq-len must be at least 2");
    anyhow::ensure!(
        !args.output.exists(),
        "{} already exists; choose a new --output",
        args.output.display()
    );
    let stage = match args.stage {
        Stage::Language => TrainingStage::Language,
        Stage::Structured => TrainingStage::Structured,
    };
    let tokenizer = TokenizerKind::Bpe(BpeTokenizer::load(&args.tokenizer)?);
    let keyword_index = build_keyword_index(&build_label_keywords());
    let mut loader = stage_data_loader(stage).seed(args.seed);
    if let Some(limit) = args.limit {
        loader = loader.total_limit(limit);
    }
    let started = std::time::Instant::now();
    let samples = loader
        .load(&tokenizer, &keyword_index, args.max_seq_len)
        .context("preparing samples from the configured training sources")?;
    let metadata = sample_cache::write_cache(
        &args.output,
        &tokenizer,
        stage,
        args.max_seq_len,
        args.seed,
        &samples,
    )?;
    println!(
        "Cached {} samples to {} ({:.1} MiB) in {:.1}s",
        metadata.samples,
        args.output.display(),
        std::fs::metadata(&args.output)?.len() as f64 / 1048576.0,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("Sample caches require a native build.");
}
