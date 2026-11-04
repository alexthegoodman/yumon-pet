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
    /// Raw Rust preparation using the same JSON configuration as train-code.
    #[arg(long, conflicts_with_all = ["output", "tokenizer", "stage", "max_seq_len", "seed"])]
    code_config: Option<std::path::PathBuf>,
    /// Override the Rust source folder in --code-config.
    #[arg(long, requires = "code_config")]
    source: Option<std::path::PathBuf>,
    /// Create a new case-preserving code tokenizer before caching (CPU only).
    #[arg(long, requires = "code_config")]
    train_code_tokenizer: bool,
    /// Build the config's `docs` stage from the toolchain's rust-docs books
    /// (CPU only; uses the existing Code tokenizer and leaves code caches alone).
    #[arg(long, requires = "code_config", conflicts_with_all = ["train_code_tokenizer", "source", "inspect"])]
    rust_docs: bool,
    /// mdBook HTML root; defaults to `$(rustc --print sysroot)/share/doc/rust/html`.
    #[arg(long, requires = "rust_docs")]
    docs_html: Option<std::path::PathBuf>,
    /// Comma-separated book directories (default: the stable books except unstable-book).
    #[arg(long, requires = "rust_docs", value_delimiter = ',')]
    books: Option<Vec<String>>,
    /// Write (or --inspect) the config's `pretrain_cache` instead of `cache`, chunked
    /// at the config's max_seq_len (e.g. code_long.bin for a longer context).
    #[arg(long, requires = "code_config", conflicts_with_all = ["rust_docs", "train_code_tokenizer"])]
    pretrain_cache: bool,
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
    /// Number of prepared samples to print (0 disables previews).
    #[arg(long, default_value_t = 50)]
    preview_samples: usize,
    /// Print metadata for an existing output file without loading corpora or a tokenizer.
    #[arg(long)]
    inspect: bool,
    /// Comma-separated list of text/chat files to cache.
    #[arg(long, value_delimiter = ',')]
    files: Option<Vec<std::path::PathBuf>>,
    /// Source file or directory of text files to cache.
    #[arg(long, alias = "data-dir")]
    dataset: Option<std::path::PathBuf>,
    /// Cache objective: "prompt_reply" (default), "pretrain", "sft", "dpo".
    #[arg(long, default_value = "prompt_reply")]
    objective: String,
    /// Overwrite existing output cache.
    #[arg(long, short = 'f')]
    force: bool,
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(path) = &args.code_config {
        use yumon_pet::brain::code_corpus;
        let mut config = code_corpus::CodeConfig::load(path)?;
        if let Some(source) = args.source { config.source = source; }
        if args.rust_docs {
            use yumon_pet::brain::code_docs;
            let output = config.docs.clone()
                .context("--rust-docs needs a \"docs\" directory in the code config")?;
            let (html, toolchain) = match args.docs_html {
                Some(html) => (html, "unknown (--docs-html)".to_owned()),
                None => code_docs::toolchain_html()?,
            };
            let books = args.books.unwrap_or_else(|| code_docs::DEFAULT_BOOKS.map(String::from).to_vec());
            let tokenizer = code_corpus::load_tokenizer(&config.tokenizer)?;
            let report = code_docs::prepare(&config, &tokenizer, &html, &toolchain, &books,
                &output, args.preview_samples)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            println!("Cached Rust docs ({toolchain}) to {}", output.display());
            return Ok(());
        }
        if args.pretrain_cache {
            config.cache = config.pretrain_cache.clone()
                .context("--pretrain-cache needs a \"pretrain_cache\" path in the code config")?;
        }
        if args.inspect {
            println!("{}", serde_json::to_string_pretty(&sample_cache::read_metadata(&config.cache)?)?);
            return Ok(());
        }
        anyhow::ensure!(!config.cache.exists(), "{} already exists; choose a new cache path in the config", config.cache.display());
        let files = code_corpus::rust_files(&config)?;
        if args.train_code_tokenizer {
            code_corpus::train_tokenizer(&files, config.vocab_size, &config.tokenizer)?;
        }
        let tokenizer = code_corpus::load_tokenizer(&config.tokenizer)?;
        let samples = code_corpus::prepare(&config, &files, &tokenizer, args.limit)?;
        for (i, sample) in samples.iter().take(args.preview_samples).enumerate() {
            let tokens = sample.target_labels.iter()
                .filter(|&&id| id != yumon_pet::brain::PAD_TOKEN).count();
            println!("\n=== Code sample {} / {} | {} ({}) | {} target tokens (including EOS) ===",
                i + 1, samples.len(), sample.pair.0, sample.pair.1, tokens);
            println!("[BOS]\n{}\n[EOS]", tokenizer.decode(&sample.target_labels));
        }
        let metadata = sample_cache::write_cache_with_objective(&config.cache, &tokenizer,
            TrainingStage::Language, config.max_seq_len, config.seed, &samples,
            sample_cache::CacheObjective::RawCode)?;
        println!("{}", serde_json::to_string_pretty(&metadata)?);
        println!("Cached Rust code to {}", config.cache.display());
        return Ok(());
    }
    if args.inspect {
        println!(
            "{}",
            serde_json::to_string_pretty(&sample_cache::read_metadata(&args.output)?)?
        );
        return Ok(());
    }
    anyhow::ensure!(args.max_seq_len >= 2, "--max-seq-len must be at least 2");
    if args.output.exists() {
        if args.force {
            std::fs::remove_file(&args.output)?;
        } else {
            anyhow::bail!("{} already exists; pass --force to overwrite", args.output.display());
        }
    }
    let stage = match args.stage {
        Stage::Language => TrainingStage::Language,
        Stage::Structured => TrainingStage::Structured,
    };
    let objective = match args.objective.to_lowercase().as_str() {
        "prompt_reply" | "prompt-reply" => sample_cache::CacheObjective::PromptReply,
        "pretrain" | "pretrain_causal" | "pretrain-causal" => sample_cache::CacheObjective::PretrainCausal,
        "sft" | "sft_instruction" | "sft-instruction" => sample_cache::CacheObjective::SftInstruction,
        "dpo" | "dpo_preference" | "dpo-preference" => sample_cache::CacheObjective::DpoPreference,
        "code" | "raw_code" | "raw-code" => sample_cache::CacheObjective::RawCode,
        other => anyhow::bail!("unknown objective '{other}'; choose prompt_reply, pretrain, sft, dpo"),
    };
    let file_kind = match objective {
        sample_cache::CacheObjective::SftInstruction => yumon_pet::brain::loader::FileKind::SyntheticSft,
        sample_cache::CacheObjective::DpoPreference => yumon_pet::brain::loader::FileKind::SyntheticDpo,
        _ => yumon_pet::brain::loader::FileKind::Chats,
    };
    let tokenizer = TokenizerKind::Bpe(BpeTokenizer::load(&args.tokenizer)?);
    let keyword_index = build_keyword_index(&build_label_keywords());
    let mut loader = if let Some(files) = &args.files {
        let mut l = yumon_pet::brain::loader::DataLoader::new(stage).seed(args.seed);
        for f in files {
            anyhow::ensure!(f.exists(), "input file does not exist: {}", f.display());
            l = l.add(f.to_string_lossy().to_string(), file_kind, None);
        }
        l
    } else if let Some(source) = &args.dataset {
        let mut l = yumon_pet::brain::loader::DataLoader::new(stage).seed(args.seed);
        if source.is_dir() {
            for entry in std::fs::read_dir(source)? {
                let path = entry?.path();
                if path.extension().map_or(false, |ext| ext == "txt" || ext == "csv") {
                    l = l.add(path.to_string_lossy().to_string(), file_kind, None);
                }
            }
        } else {
            anyhow::ensure!(source.exists(), "source file does not exist: {}", source.display());
            l = l.add(source.to_string_lossy().to_string(), file_kind, None);
        }
        l
    } else {
        stage_data_loader(stage).seed(args.seed)
    };
    if let Some(limit) = args.limit {
        loader = loader.total_limit(limit);
    }
    let started = std::time::Instant::now();
    let samples = loader
        .load(&tokenizer, &keyword_index, args.max_seq_len)
        .context("preparing samples from the configured training sources")?;

    for sample in samples.iter().take(args.preview_samples) {
        println!("input: {}", tokenizer.decode(&sample.input_ids));
        println!("target: {}", tokenizer.decode(&sample.target_labels));
    }

    let metadata = sample_cache::write_cache_with_objective(
        &args.output,
        &tokenizer,
        stage,
        args.max_seq_len,
        args.seed,
        &samples,
        objective,
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
