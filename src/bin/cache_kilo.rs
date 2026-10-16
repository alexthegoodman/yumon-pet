//! Convert raw Rust training chunks into the Kilo language and cache to `kilo_code.bin`.

#[cfg(not(target_arch = "wasm32"))]
use {
    anyhow::{Context, Result, ensure},
    clap::Parser,
    std::path::PathBuf,
    yumon_pet::brain::{
        BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
        bpe::{BpeTokenizer, TokenizerKind},
        code_corpus,
        kilo,
        sample_cache::{self, CacheObjective},
        samples::{Action, CardinalDir, Sample, TrainingStage, WorldContext},
    },
};

#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser)]
#[command(about = "Convert Rust code chunks into the Kilo intermediate language (1024 tokens) and save to kilo_code.bin")]
struct Args {
    /// Existing Rust raw-code cache (default: training-cache/code.bin).
    #[arg(long, default_value = "training-cache/code.bin")]
    input_cache: PathBuf,
    /// Matching Rust code BPE tokenizer directory.
    #[arg(long, default_value = "yumon_code_bpe")]
    tokenizer: String,
    /// Output Kilo cache path.
    #[arg(long, default_value = "training-cache/kilo_code.bin")]
    output: PathBuf,
    /// Context length (must be >= 2).
    #[arg(long, default_value_t = 512)]
    max_seq_len: usize,
    /// Random seed for metadata.
    #[arg(long, default_value_t = 4815162342)]
    seed: u64,
    /// Limit number of input samples to convert (None = all).
    #[arg(long)]
    limit: Option<usize>,
    /// Number of sample previews to print.
    #[arg(long, default_value_t = 5)]
    preview_samples: usize,
    /// Overwrite existing output cache if present.
    #[arg(long)]
    force: bool,
    /// Print metadata for an existing kilo cache without loading corpora.
    #[arg(long)]
    inspect: bool,
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<()> {
    let args = Args::parse();

    if args.inspect {
        let meta = sample_cache::read_metadata(&args.output)?;
        println!("{}", serde_json::to_string_pretty(&meta)?);
        return Ok(());
    }

    if args.output.exists() {
        if args.force {
            std::fs::remove_file(&args.output)?;
        } else {
            anyhow::bail!("{} already exists; pass --force to overwrite", args.output.display());
        }
    }

    ensure!(args.input_cache.exists(), "Input cache does not exist: {}", args.input_cache.display());
    ensure!(args.max_seq_len >= 2, "max_seq_len must be >= 2");

    println!("Loading input Rust code cache from {} ...", args.input_cache.display());
    let input_meta = sample_cache::read_metadata(&args.input_cache)?;
    ensure!(input_meta.objective == CacheObjective::RawCode, "Input cache must be a RawCode cache");

    let bpe_tokenizer = code_corpus::load_tokenizer(&args.tokenizer)?;
    let TokenizerKind::Bpe(ref bpe) = bpe_tokenizer else {
        anyhow::bail!("Input cache requires code BPE tokenizer");
    };

    let input_samples = sample_cache::load_cache_with_objective(
        &args.input_cache,
        &bpe_tokenizer,
        input_meta.stage,
        input_meta.max_seq_len,
        CacheObjective::RawCode,
    )?;

    let total_available = input_samples.len();
    let samples_to_process = match args.limit {
        Some(lim) => lim.min(total_available),
        None => total_available,
    };

    println!("Processing {} / {} Rust code samples into Kilo language...", samples_to_process, total_available);

    let budget = args.max_seq_len - 1; // 1 for BOS, 1 for EOS
    let mut kilo_samples = Vec::new();
    let mut preview_list = Vec::new();

    let mut unparsed_count = 0;
    let mut unk_token_count = 0;
    let mut oversized_count = 0;
    let mut roundtrip_failed_count = 0;

    let started = std::time::Instant::now();

    for (idx, sample) in input_samples.iter().take(samples_to_process).enumerate() {
        let ids: Vec<u32> = sample.target_labels.iter()
            .take_while(|&&id| id != PAD_TOKEN)
            .map(|&id| id as u32)
            .collect();

        // Decode source code
        let raw_code = match bpe.decode(&ids) {
            Ok(s) => s,
            Err(_) => {
                unparsed_count += 1;
                continue;
            }
        };

        // Convert to Kilo
        let kilo_code = match kilo::rust_to_kilo(&raw_code) {
            Ok(k) => k,
            Err(e) => {
                let err_str = e.to_string();
                if unk_token_count + unparsed_count < 5 {
                    eprintln!("[Kilo Debug] Failed {}:{} -> {}", sample.pair.0, sample.pair.1, err_str);
                }
                if err_str.contains("UNK") {
                    unk_token_count += 1;
                } else {
                    unparsed_count += 1;
                }
                continue;
            }
        };

        // Verify round-trip back to valid Rust
        let rust_reconstructed = match kilo::kilo_to_rust(&kilo_code) {
            Ok(r) => r,
            Err(_) => {
                roundtrip_failed_count += 1;
                continue;
            }
        };

        // Tokenize Kilo code into IDs
        let token_ids = kilo::tokenize_kilo(&kilo_code);
        if token_ids.len() > budget {
            oversized_count += 1;
            continue;
        }

        // Store preview if needed
        if preview_list.len() < args.preview_samples {
            preview_list.push((sample.pair.clone(), raw_code.clone(), kilo_code.clone(), rust_reconstructed, token_ids.len()));
        }

        // Build sample
        let mut input_ids = vec![BOS_TOKEN];
        input_ids.extend(token_ids.iter().copied());
        let mut target_labels: Vec<usize> = token_ids.iter().copied().collect();
        target_labels.push(EOS_TOKEN);

        input_ids.resize(args.max_seq_len, PAD_TOKEN);
        target_labels.resize(args.max_seq_len, PAD_TOKEN);

        // Validate sample format against raw-code requirements
        let end = target_labels.iter().position(|&t| t == PAD_TOKEN).unwrap_or(target_labels.len());
        if end < 2
            || input_ids[0] != BOS_TOKEN
            || target_labels[end - 1] != EOS_TOKEN
            || input_ids[1..end] != target_labels[..end - 1]
            || target_labels[..end - 1].iter().any(|&id| id <= 3)
            || !input_ids[end..].iter().all(|&id| id == PAD_TOKEN)
            || !target_labels[end..].iter().all(|&id| id == PAD_TOKEN)
            || sample.pair.0.is_empty()
        {
            if let Some(bad_tok) = target_labels[..end - 1].iter().find(|&&id| id <= 3) {
                eprintln!("[Validation debug] sample has token <= 3: id={bad_tok} in {}:{}", sample.pair.0, sample.pair.1);
            }
            continue;
        }

        kilo_samples.push(Sample {
            input_ids,
            target_labels,
            action: Action::Sit,
            motion_dir: CardinalDir::None,
            world: WorldContext::default(),
            target_json: String::new(),
            pair: sample.pair.clone(),
        });

        if (idx + 1) % 10000 == 0 || idx + 1 == samples_to_process {
            let pct = (kilo_samples.len() as f64 / (idx + 1) as f64) * 100.0;
            println!("  [{:>6} / {:>6}] Converted: {:>6} (Coverage: {:.2}%)", idx + 1, samples_to_process, kilo_samples.len(), pct);
        }
    }

    let elapsed = started.elapsed().as_secs_f64();
    let coverage_pct = (kilo_samples.len() as f64 / samples_to_process as f64) * 100.0;

    println!("\n=== Kilo Conversion Results ===");
    println!("Total Rust items considered: {}", samples_to_process);
    println!("Successfully converted to Kilo: {} ({:.2}% coverage)", kilo_samples.len(), coverage_pct);
    println!("Skipped due to unparseable / unsupported constructs: {}", unparsed_count);
    println!("Skipped due to out-of-budget tokens: {}", unk_token_count);
    println!("Skipped due to roundtrip verification failure: {}", roundtrip_failed_count);
    println!("Skipped due to sequence length > {}: {}", budget, oversized_count);
    println!("Total processing time: {:.2}s ({:.1} items/s)", elapsed, samples_to_process as f64 / elapsed.max(0.001));

    ensure!(
        coverage_pct >= 50.0 || samples_to_process < 10,
        "Coverage was {:.2}%, which is below the 50% requirement! Please refine Kilo mappings.",
        coverage_pct
    );

    println!("\n=== Converted Sample Previews ===");
    for (i, (pair, original, kilo_src, reconstructed, n_toks)) in preview_list.iter().enumerate() {
        println!("\n--- Sample {} [{}:{}] ({} Kilo tokens) ---", i + 1, pair.0, pair.1, n_toks);
        println!(">>> ORIGINAL RUST <<<\n{}", original.trim());
        println!("\n>>> KILO INTERMEDIARY CODE <<<\n{}", kilo_src.trim());
        println!("\n>>> RECONSTRUCTED RUST <<<\n{}", reconstructed.trim());
    }

    println!("\nWriting Kilo cache to {} ...", args.output.display());
    let kilo_tok = BpeTokenizer::load("training-cache/kilo_tokenizer")
        .context("loading training-cache/kilo_tokenizer (generate with build_kilo_tokenizer.py)")?;
    let metadata = sample_cache::write_cache_with_objective(
        &args.output,
        &TokenizerKind::Bpe(kilo_tok),
        TrainingStage::Language,
        args.max_seq_len,
        args.seed,
        &kilo_samples,
        CacheObjective::RawCode,
    )?;

    let out_size_mb = std::fs::metadata(&args.output)?.len() as f64 / 1048576.0;
    println!("Cached {} Kilo samples to {} ({:.2} MiB)", metadata.samples, args.output.display(), out_size_mb);

    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("Kilo caching requires a native build.");
}
