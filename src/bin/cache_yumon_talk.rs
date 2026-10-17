//! Convert raw English training samples into the Yumon Talk language and cache to `yumon_talk_samples.bin`.

#[cfg(not(target_arch = "wasm32"))]
use {
    anyhow::{Context, Result, ensure},
    clap::Parser,
    rand::SeedableRng,
    std::path::PathBuf,
    yumon_pet::brain::{
        BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
        bpe::{BpeTokenizer, TokenizerKind},
        sample_cache,
        samples::Sample,
        yumon_talk,
    },
};

#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser)]
#[command(about = "Convert natural language samples into the Yumon Talk language (1024 tokens) and save to yumon_talk_samples.bin")]
struct Args {
    /// Existing natural language cache (default: training-cache/samples.bin).
    #[arg(long, default_value = "training-cache/samples.bin")]
    input_cache: PathBuf,
    /// Matching BPE tokenizer directory.
    #[arg(long, default_value = "yumon_bpe")]
    tokenizer: String,
    /// Output Yumon Talk cache path.
    #[arg(long, default_value = "training-cache/yumon_talk_samples.bin")]
    output: PathBuf,
    /// Path to dynamic Yumon Talk vocabulary JSON file (default: training-cache/yumon_talk_vocab.json).
    #[arg(long, default_value = "training-cache/yumon_talk_vocab.json")]
    vocab: PathBuf,
    /// Context length (must be >= 2).
    #[arg(long, default_value_t = 256)]
    max_seq_len: usize,
    /// Random seed for sampling.
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
    /// Print metadata for an existing cache without converting.
    #[arg(long)]
    inspect: bool,
    /// Minimum required conversion rate percentage (default: 50.0%).
    #[arg(long, default_value_t = 50.0)]
    min_coverage: f64,
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

    if args.vocab.exists() {
        println!("Loading Yumon Talk dynamic vocabulary from {} ...", args.vocab.display());
        yumon_talk::load_global_vocab_file(&args.vocab)?;
    } else {
        println!("Notice: Vocab file {} not found; using built-in default vocabulary.", args.vocab.display());
    }

    println!("Loading input samples from {} ...", args.input_cache.display());
    let (input_meta, input_samples) = sample_cache::load_cache_lenient(
        &args.input_cache,
        args.max_seq_len,
    )?;

    let talk_tokenizer_dir = args.vocab.parent().unwrap_or(std::path::Path::new(".")).join("yumon_talk_tokenizer");
    let target_tokenizer = if talk_tokenizer_dir.join("tokenizer.json").exists() {
        println!("Using Yumon Talk tokenizer from {} ...", talk_tokenizer_dir.display());
        TokenizerKind::Bpe(
            BpeTokenizer::load(talk_tokenizer_dir.to_str().unwrap())
                .with_context(|| format!("loading tokenizer from {}", talk_tokenizer_dir.display()))?
        )
    } else {
        println!("Using base tokenizer from {} ...", args.tokenizer);
        TokenizerKind::Bpe(
            BpeTokenizer::load(&args.tokenizer)
                .with_context(|| format!("loading tokenizer from {}", args.tokenizer))?
        )
    };

    let total_available = input_samples.len();
    let samples_to_process = match args.limit {
        Some(lim) => lim.min(total_available),
        None => total_available,
    };

    println!("Processing {} / {} samples into Yumon Talk language...", samples_to_process, total_available);

    let budget = args.max_seq_len - 1; // 1 for BOS, 1 for EOS
    let mut yumon_samples = Vec::new();
    let mut preview_list = Vec::new();

    let mut unk_token_count = 0;
    let mut oversized_count = 0;
    let mut empty_count = 0;
    let mut roundtrip_failed_count = 0;

    let started = std::time::Instant::now();
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);

    for (idx, sample) in input_samples.iter().take(samples_to_process).enumerate() {
        // Extract original text from pair or token decode
        let (prompt_raw, reply_raw) = if !sample.pair.0.is_empty() || !sample.pair.1.is_empty() {
            (sample.pair.0.clone(), sample.pair.1.clone())
        } else {
            let p_ids: Vec<u32> = sample.input_ids.iter()
                .take_while(|&&id| id != PAD_TOKEN)
                .map(|&id| id as u32)
                .collect();
            let r_ids: Vec<u32> = sample.target_labels.iter()
                .take_while(|&&id| id != PAD_TOKEN)
                .map(|&id| id as u32)
                .collect();
            let p = match target_tokenizer {
                TokenizerKind::Bpe(ref bpe) => bpe.decode(&p_ids).unwrap_or_default(),
                _ => String::new(),
            };
            let r = match target_tokenizer {
                TokenizerKind::Bpe(ref bpe) => bpe.decode(&r_ids).unwrap_or_default(),
                _ => String::new(),
            };
            (p, r)
        };

        let combined_english = if !prompt_raw.is_empty() && !reply_raw.is_empty() {
            format!("{} {}", prompt_raw.trim(), reply_raw.trim())
        } else if !reply_raw.is_empty() {
            reply_raw.trim().to_string()
        } else {
            prompt_raw.trim().to_string()
        };

        if combined_english.is_empty() {
            empty_count += 1;
            continue;
        }

        // Convert English to Yumon Talk
        let (yumon_str, ctx) = match yumon_talk::english_to_yumon_with_map(&combined_english) {
            Ok(res) => res,
            Err(e) => {
                if unk_token_count < 5 {
                    eprintln!("[DEBUG ERROR sample {idx}] {e}");
                }
                unk_token_count += 1;
                continue;
            }
        };

        // Roundtrip back to English with random synonym sampling
        let reconstructed_english = match yumon_talk::yumon_to_english_with_context(&yumon_str, &ctx, &mut rng) {
            Ok(eng) => eng,
            Err(_) => {
                roundtrip_failed_count += 1;
                continue;
            }
        };

        // Tokenize Yumon Talk into IDs
        let token_ids = yumon_talk::tokenize_yumon(&yumon_str);
        if token_ids.len() > budget {
            oversized_count += 1;
            continue;
        }

        if token_ids.iter().any(|&id| id == yumon_talk::UNK_ID) {
            if unk_token_count < 3 {
                eprintln!("[DEBUG UNK sample {idx}] yumon_str = {:?}", yumon_str);
            }
            unk_token_count += 1;
            continue;
        }

        if preview_list.len() < args.preview_samples {
            preview_list.push((
                combined_english.clone(),
                yumon_str.clone(),
                reconstructed_english.clone(),
                token_ids.len(),
            ));
        }

        let mut input_ids = vec![BOS_TOKEN];
        input_ids.extend(token_ids.iter().copied());
        let mut target_labels: Vec<usize> = token_ids.iter().copied().collect();
        target_labels.push(EOS_TOKEN);

        input_ids.resize(args.max_seq_len, PAD_TOKEN);
        target_labels.resize(args.max_seq_len, PAD_TOKEN);

        yumon_samples.push(Sample {
            input_ids,
            target_labels,
            action: sample.action,
            motion_dir: sample.motion_dir,
            world: sample.world,
            target_json: sample.target_json.clone(),
            pair: (combined_english, yumon_str),
        });

        if (idx + 1) % 5000 == 0 || idx + 1 == samples_to_process {
            let pct = (yumon_samples.len() as f64 / (idx + 1) as f64) * 100.0;
            println!("  [{:>6} / {:>6}] Converted: {:>6} (Coverage: {:.2}%)", idx + 1, samples_to_process, yumon_samples.len(), pct);
        }
    }

    let elapsed = started.elapsed().as_secs_f64();
    let coverage_pct = (yumon_samples.len() as f64 / samples_to_process as f64) * 100.0;

    println!("\n=== Yumon Talk Conversion Results ===");
    println!("Total samples considered: {}", samples_to_process);
    println!("Successfully converted to Yumon Talk: {} ({:.2}% coverage)", yumon_samples.len(), coverage_pct);
    println!("Skipped due to UNK tokens: {}", unk_token_count);
    println!("Skipped due to roundtrip failure: {}", roundtrip_failed_count);
    println!("Skipped due to sequence length > {}: {}", budget, oversized_count);
    println!("Skipped due to empty text: {}", empty_count);
    println!("Total processing time: {:.2}s ({:.1} samples/s)", elapsed, samples_to_process as f64 / elapsed.max(0.001));

    ensure!(
        coverage_pct >= args.min_coverage || samples_to_process < 10,
        "Coverage was {:.2}%, which is below the {:.1}% requirement! Please refine Yumon Talk vocabulary and mappings.",
        coverage_pct,
        args.min_coverage
    );

    println!("\n=== Converted Sample Previews ===");
    for (i, (original, yumon, reconstructed, n_toks)) in preview_list.iter().enumerate() {
        println!("\n--- Sample {} ({} Yumon Talk tokens) ---", i + 1, n_toks);
        println!(">>> ORIGINAL ENGLISH <<<\n{}", original.trim());
        println!("\n>>> YUMON TALK <<<\n{}", yumon.trim());
        println!("\n>>> RECONSTRUCTED ENGLISH (RANDOM SYNONYM SAMPLING) <<<\n{}", reconstructed.trim());
    }

    println!("\nWriting Yumon Talk cache to {} ...", args.output.display());
    let metadata = sample_cache::write_cache_with_objective(
        &args.output,
        &target_tokenizer,
        input_meta.stage,
        args.max_seq_len,
        args.seed,
        &yumon_samples,
        input_meta.objective,
    )?;

    let out_size_mb = std::fs::metadata(&args.output)?.len() as f64 / 1048576.0;
    println!("Cached {} Yumon Talk samples to {} ({:.2} MiB)", metadata.samples, args.output.display(), out_size_mb);

    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("Yumon Talk caching requires a native build.");
}
