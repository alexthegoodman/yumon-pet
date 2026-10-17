//! Convert raw English training samples into the Yumon Talk language and cache to `yumon_talk_samples.bin`.

#[cfg(not(target_arch = "wasm32"))]
use {
    anyhow::{Context, Result, ensure},
    clap::Parser,
    rand::SeedableRng,
    rayon::prelude::*,
    std::path::PathBuf,
    std::sync::atomic::{AtomicUsize, Ordering},
    yumon_pet::brain::{
        BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
        bpe::{BpeTokenizer, TokenizerKind},
        sample_cache,
        samples::Sample,
        yumon_talk::{self, YumonTalkVocab},
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
type PreviewItem = (String, String, String, String, String, String, usize, usize);

#[cfg(not(target_arch = "wasm32"))]
enum ConversionResult {
    Converted {
        sample: Sample,
        preview: Option<PreviewItem>,
    },
    SkippedUnk,
    SkippedRoundtripFailed,
    SkippedOversized,
    SkippedEmpty,
}

#[cfg(not(target_arch = "wasm32"))]
fn convert_single_sample(
    idx: usize,
    sample: &Sample,
    vocab: &YumonTalkVocab,
    target_tokenizer: &TokenizerKind,
    max_seq_len: usize,
    seed: u64,
    capture_preview: bool,
) -> ConversionResult {
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
            TokenizerKind::Bpe(bpe) => bpe.decode(&p_ids).unwrap_or_default(),
            _ => String::new(),
        };
        let r = match target_tokenizer {
            TokenizerKind::Bpe(bpe) => bpe.decode(&r_ids).unwrap_or_default(),
            _ => String::new(),
        };
        (p, r)
    };

    let mut ctx = yumon_talk::TranslationContext::new();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(idx as u64));

    let (prompt_tokens, reply_tokens, prompt_yumon, reply_yumon, prompt_recon, reply_recon) =
        if !prompt_raw.trim().is_empty() && !reply_raw.trim().is_empty() {
            let p_tokens = match vocab.english_to_yumon_tokens(&prompt_raw, &mut ctx) {
                Ok(t) => t,
                Err(_) => return ConversionResult::SkippedUnk,
            };
            let r_tokens = match vocab.english_to_yumon_tokens(&reply_raw, &mut ctx) {
                Ok(t) => t,
                Err(_) => return ConversionResult::SkippedUnk,
            };

            let p_yumon = vocab.detokenize_yumon(&p_tokens);
            let r_yumon = vocab.detokenize_yumon(&r_tokens);

            let p_recon = match vocab.yumon_to_english_with_context(&p_yumon, &ctx, &mut rng) {
                Ok(eng) => eng,
                Err(_) => return ConversionResult::SkippedRoundtripFailed,
            };
            let r_recon = match vocab.yumon_to_english_with_context(&r_recon_source(&r_yumon), &ctx, &mut rng) {
                Ok(eng) => eng,
                Err(_) => return ConversionResult::SkippedRoundtripFailed,
            };

            (p_tokens, r_tokens, p_yumon, r_yumon, p_recon, r_recon)
        } else {
            let text = if !reply_raw.trim().is_empty() {
                reply_raw.trim()
            } else {
                prompt_raw.trim()
            };
            if text.is_empty() {
                return ConversionResult::SkippedEmpty;
            }
            let r_tokens = match vocab.english_to_yumon_tokens(text, &mut ctx) {
                Ok(t) => t,
                Err(_) => return ConversionResult::SkippedUnk,
            };
            let r_yumon = vocab.detokenize_yumon(&r_tokens);
            let r_recon = match vocab.yumon_to_english_with_context(&r_yumon, &ctx, &mut rng) {
                Ok(eng) => eng,
                Err(_) => return ConversionResult::SkippedRoundtripFailed,
            };
            (Vec::new(), r_tokens, String::new(), r_yumon, String::new(), r_recon)
        };

    if prompt_tokens.iter().any(|&id| id == yumon_talk::UNK_ID)
        || reply_tokens.iter().any(|&id| id == yumon_talk::UNK_ID)
    {
        return ConversionResult::SkippedUnk;
    }

    // Budget check: BOS (1) + prompt + sep (1) + reply + EOS (1) <= max_seq_len
    let total_active_len = if prompt_tokens.is_empty() {
        1 + reply_tokens.len() + 1
    } else {
        1 + prompt_tokens.len() + 1 + reply_tokens.len() + 1
    };

    if total_active_len > max_seq_len {
        return ConversionResult::SkippedOversized;
    }

    let preview = if capture_preview {
        Some((
            prompt_raw,
            prompt_yumon.clone(),
            prompt_recon,
            reply_raw,
            reply_yumon.clone(),
            reply_recon,
            prompt_tokens.len(),
            reply_tokens.len(),
        ))
    } else {
        None
    };

    let mut input_ids = vec![BOS_TOKEN];
    input_ids.extend(prompt_tokens.iter().copied());
    let mut target_labels: Vec<usize> = reply_tokens.iter().copied().collect();
    target_labels.push(EOS_TOKEN);

    input_ids.resize(max_seq_len, PAD_TOKEN);
    target_labels.resize(max_seq_len, PAD_TOKEN);

    ConversionResult::Converted {
        sample: Sample {
            input_ids,
            target_labels,
            action: sample.action,
            motion_dir: sample.motion_dir,
            world: sample.world,
            target_json: sample.target_json.clone(),
            pair: (prompt_yumon, reply_yumon),
        },
        preview,
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn r_recon_source(s: &str) -> &str {
    s
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

    println!("Initializing thesaurus dictionary (static feature)...");
    thesaurus::init();
    yumon_talk::get_thesaurus_dict();

    if args.vocab.exists() {
        println!("Loading Yumon Talk dynamic vocabulary from {} ...", args.vocab.display());
        yumon_talk::load_global_vocab_file(&args.vocab)?;
    } else {
        println!("Notice: Vocab file {} not found; using built-in default vocabulary.", args.vocab.display());
    }

    println!("Loading input samples from {} ...", args.input_cache.display());
    let (input_meta, input_samples) = sample_cache::load_cache_lenient_limit(
        &args.input_cache,
        args.max_seq_len,
        args.limit,
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

    let total_available = input_meta.samples;
    let samples_to_process = input_samples.len();

    println!("Processing {} / {} samples into Yumon Talk language with rayon...", samples_to_process, total_available);

    let started = std::time::Instant::now();
    let processed_counter = AtomicUsize::new(0);
    let converted_counter = AtomicUsize::new(0);

    let vocab_guard = yumon_talk::get_global_vocab();
    let vocab_ref = &*vocab_guard;

    let preview_capture_limit = args.preview_samples.max(10) * 5;

    let results: Vec<ConversionResult> = input_samples
        .par_iter()
        .enumerate()
        .map(|(idx, sample)| {
            let capture_preview = idx < preview_capture_limit;
            let res = convert_single_sample(
                idx,
                sample,
                vocab_ref,
                &target_tokenizer,
                args.max_seq_len,
                args.seed,
                capture_preview,
            );

            let is_conv = matches!(res, ConversionResult::Converted { .. });
            let done = processed_counter.fetch_add(1, Ordering::Relaxed) + 1;
            let conv = if is_conv {
                converted_counter.fetch_add(1, Ordering::Relaxed) + 1
            } else {
                converted_counter.load(Ordering::Relaxed)
            };

            if done % 5000 == 0 || done == samples_to_process {
                let pct = (conv as f64 / done as f64) * 100.0;
                println!(
                    "  [{:>6} / {:>6}] Converted: {:>6} (Coverage: {:.2}%)",
                    done, samples_to_process, conv, pct
                );
            }

            res
        })
        .collect();

    let mut yumon_samples = Vec::with_capacity(results.len());
    let mut preview_list = Vec::new();

    let mut unk_token_count = 0;
    let mut oversized_count = 0;
    let mut empty_count = 0;
    let mut roundtrip_failed_count = 0;

    for res in results {
        match res {
            ConversionResult::Converted { sample, preview } => {
                if let Some(prev) = preview {
                    if preview_list.len() < args.preview_samples {
                        preview_list.push(prev);
                    }
                }
                yumon_samples.push(sample);
            }
            ConversionResult::SkippedUnk => unk_token_count += 1,
            ConversionResult::SkippedRoundtripFailed => roundtrip_failed_count += 1,
            ConversionResult::SkippedOversized => oversized_count += 1,
            ConversionResult::SkippedEmpty => empty_count += 1,
        }
    }

    let elapsed = started.elapsed().as_secs_f64();
    let coverage_pct = (yumon_samples.len() as f64 / samples_to_process as f64) * 100.0;

    println!("\n=== Yumon Talk Conversion Results ===");
    println!("Total samples considered: {}", samples_to_process);
    println!("Successfully converted to Yumon Talk: {} ({:.2}% coverage)", yumon_samples.len(), coverage_pct);
    println!("Skipped due to UNK tokens: {}", unk_token_count);
    println!("Skipped due to roundtrip failure: {}", roundtrip_failed_count);
    println!("Skipped due to sequence length > {}: {}", args.max_seq_len, oversized_count);
    println!("Skipped due to empty text: {}", empty_count);
    println!("Total processing time: {:.2}s ({:.1} samples/s)", elapsed, samples_to_process as f64 / elapsed.max(0.001));

    ensure!(
        coverage_pct >= args.min_coverage || samples_to_process < 10,
        "Coverage was {:.2}%, which is below the {:.1}% requirement! Please refine Yumon Talk vocabulary and mappings.",
        coverage_pct,
        args.min_coverage
    );

    println!("\n=== Converted Sample Previews ===");
    for (i, (p_orig, p_yum, p_rec, r_orig, r_yum, r_rec, n_p, n_r)) in preview_list.iter().enumerate() {
        println!("\n--- Sample {} (Prompt: {} toks, Reply: {} toks) ---", i + 1, n_p, n_r);
        if !p_orig.is_empty() {
            println!(">>> ORIGINAL PROMPT <<<\n{}", p_orig.trim());
            println!(">>> YUMON TALK PROMPT <<<\n{}", p_yum.trim());
            println!(">>> RECONSTRUCTED PROMPT <<<\n{}", p_rec.trim());
        }
        println!(">>> ORIGINAL REPLY <<<\n{}", r_orig.trim());
        println!(">>> YUMON TALK REPLY <<<\n{}", r_yum.trim());
        println!(">>> RECONSTRUCTED REPLY (RANDOM SYNONYM SAMPLING) <<<\n{}", r_rec.trim());
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
