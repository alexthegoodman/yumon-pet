//! Train and benchmark the contextual N-Gram statistical disambiguation decompressor.
//!
//! Scans raw English training samples, builds a compact N-gram language model,
//! and evaluates decompression on sample sentences.

#[cfg(not(target_arch = "wasm32"))]
use {
    anyhow::{Result, ensure},
    clap::Parser,
    std::path::PathBuf,
    std::time::Instant,
    yumon_pet::brain::{
        decompressor::NGramDecompressor,
        kilo_vocab::KiloVocab,
        yumon_talk::{TranslationContext, YumonTalkVocab},
    },
};

#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser, Debug)]
#[command(about = "Build and evaluate a contextual N-Gram statistical disambiguation decompressor for Yumon Talk / Kilo Talk")]
struct Args {
    /// Path to input training cache (default: training-cache/samples.bin)
    #[arg(long, default_value = "training-cache/samples.bin")]
    input_cache: PathBuf,

    /// Output binary path for the trained decompressor
    #[arg(long, default_value = "training-cache/ngram_decompressor.bin")]
    output: PathBuf,

    /// Maximum N-gram order (default: 3 for tri-gram, 4 for 4-gram)
    #[arg(long, default_value_t = 3)]
    order: usize,

    /// Limit number of input samples to scan (None = all samples)
    #[arg(long)]
    limit: Option<usize>,

    /// Path to vocabulary JSON (default: training-cache/kilo_talk_vocab.json)
    #[arg(long, default_value = "training-cache/kilo_talk_vocab.json")]
    vocab: PathBuf,

    /// Number of preview comparisons to print
    #[arg(long, default_value_t = 5)]
    previews: usize,
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<()> {
    let args = Args::parse();

    println!("=== Building Contextual N-Gram Decompressor Model ===");
    println!("Input cache:  {}", args.input_cache.display());
    println!("Output model: {}", args.output.display());
    println!("N-Gram order: {}-gram", args.order);

    ensure!(
        args.input_cache.exists(),
        "Input cache not found: {}",
        args.input_cache.display()
    );

    let started = Instant::now();
    let mut decompressor = NGramDecompressor::new(args.order);

    println!("\n[1/3] Scanning {} and accumulating N-gram statistics...", args.input_cache.display());
    let samples_scanned = decompressor.train_on_cache(&args.input_cache, args.limit)?;
    let train_elapsed = started.elapsed().as_secs_f64();

    println!(
        "Training complete in {:.2}s: {} samples scanned ({:.0} samples/s)",
        train_elapsed, samples_scanned, samples_scanned as f64 / train_elapsed.max(0.001)
    );
    println!("Vocabulary:           {} unique words", decompressor.vocab_size());
    println!("Total word tokens:    {}", decompressor.total_unigrams);
    println!("Unique bigrams:       {}", decompressor.bigrams.len());
    println!("Unique trigrams:      {}", decompressor.trigrams.len());
    if args.order >= 4 {
        println!("Unique fourgrams:     {}", decompressor.fourgrams.len());
    }

    println!("\n[2/3] Saving decompressor model to {} ...", args.output.display());
    decompressor.save_to_file(&args.output)?;
    let file_size_kb = std::fs::metadata(&args.output)?.len() as f64 / 1024.0;
    println!("Saved decompressor model ({:.1} KB)", file_size_kb);

    // [3/3] Preview decompression comparison
    let vocab_path = if args.vocab.exists() {
        args.vocab
    } else if std::path::Path::new("training-cache/yumon_talk_vocab.json").exists() {
        PathBuf::from("training-cache/yumon_talk_vocab.json")
    } else {
        args.vocab
    };

    if vocab_path.exists() && args.previews > 0 {
        println!("\n[3/3] Decompression Preview & Comparison (Top {} samples):", args.previews);

        let test_sentences = [
            "This release made a significant impact on performance.",
            "I am very happy to see my good companion today.",
            "They were having a great conversation about the new technology.",
            "The weather outside was cold and freezing.",
            "Nintendo announced an exciting handheld console.",
        ];

        if let Ok(kilo) = KiloVocab::load_from_json(&vocab_path) {
            println!("Loaded Kilo Tokenizer / Vocab Mapping Model from {} ({} tokens)", vocab_path.display(), kilo.vocab_size);
            for (i, &sentence) in test_sentences.iter().take(args.previews).enumerate() {
                let mut ctx = TranslationContext::new();
                let tokens = match kilo.encode(sentence, &mut ctx) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                let compressed_text = kilo.decode_tokens(&tokens);

                let t0 = Instant::now();
                let ngram_decomp = decompressor.decompress_kilo(&tokens, &kilo, &ctx, 4)?;
                let latency_us = t0.elapsed().as_micros();

                println!("\n--- Sample #{} ---", i + 1);
                println!("  Original:       {}", sentence);
                println!("  Compressed:     {}", compressed_text);
                println!("  N-Gram Model:   {} ({:>4} µs)", ngram_decomp, latency_us);
            }
        } else if let Ok(vocab) = YumonTalkVocab::load_from_file(&vocab_path) {
            println!("Loaded Yumon Talk vocabulary from {}", vocab_path.display());
            let mut rng = rand::thread_rng();
            for (i, &sentence) in test_sentences.iter().take(args.previews).enumerate() {
                let mut ctx = TranslationContext::new();
                let tokens = match vocab.english_to_yumon_tokens(sentence, &mut ctx) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                let yumon_talk = vocab.detokenize_yumon(&tokens);
                let random_decomp = vocab.yumon_to_english_with_context(&yumon_talk, &ctx, &mut rng).unwrap_or_default();

                let t0 = Instant::now();
                let ngram_decomp = decompressor.decompress_with_context(&yumon_talk, &vocab, &ctx)?;
                let latency_us = t0.elapsed().as_micros();

                println!("\n--- Sample #{} ---", i + 1);
                println!("  Original:       {}", sentence);
                println!("  Compressed:     {}", yumon_talk);
                println!("  Random Synonym: {}", random_decomp);
                println!("  N-Gram Model:   {} ({:>4} µs)", ngram_decomp, latency_us);
            }
        }
    }

    println!("\nSuccessfully built and verified decompressor model!");
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("Decompressor generation requires a native build.");
}
