//! Build a data-driven Kilo Tokenizer / Vocab Mapping Model (4096 tokens) from samples.bin.
//!
//! Scans English training samples, computes unigram and multi-word phrase frequencies,
//! and learns a strictly bounded 4,096-token vocabulary that preserves word distinctions
//! and mines recurring multi-word collocations without thesaurus-based semantic collapse.

#[cfg(not(target_arch = "wasm32"))]
use {
    anyhow::{Result, ensure},
    clap::Parser,
    std::collections::HashMap,
    std::path::PathBuf,
    yumon_pet::brain::{
        kilo_vocab::{self, ContextFrameHash, KiloVocab},
        sample_cache,
        yumon_talk::TranslationContext,
    },
};

#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser, Debug)]
#[command(about = "Build a data-driven 4,096-token Kilo Vocab Mapping Model preserving word distinctions")]
struct Args {
    /// Path to input training cache (default: training-cache/samples.bin)
    #[arg(long, default_value = "training-cache/samples.bin")]
    input_cache: PathBuf,

    /// Path to save generated vocabulary JSON (default: training-cache/kilo_talk_vocab.json)
    #[arg(long, default_value = "training-cache/kilo_talk_vocab.json")]
    output_vocab: PathBuf,

    /// Total target vocabulary size including special, punctuation, placeholders and words (default: 4096)
    #[arg(long, default_value_t = 4096)]
    vocab_size: usize,

    /// Maximum number of recurring multi-word phrases to mine and include (default: 300)
    #[arg(long, default_value_t = 300)]
    max_phrases: usize,

    /// Minimum confidence score to map an open-class word into a canonical anchor (default: 0.60).
    /// Words scoring below this threshold are strictly dropped.
    #[arg(long, default_value_t = 0.60)]
    min_mapping_score: f64,

    /// Maximum number of mapped words per canonical anchor cluster (default: 25)
    #[arg(long, default_value_t = 25)]
    max_cluster_size: usize,

    /// Minimum word occurrences required to perform multi-gram distributional matching (default: 10)
    #[arg(long, default_value_t = 10)]
    min_context_freq: usize,

    /// Minimum word frequency in corpus to be considered (default: 2)
    #[arg(long, default_value_t = 2)]
    min_word_freq: usize,

    /// Minimum multi-word phrase frequency to be considered (default: 5)
    #[arg(long, default_value_t = 5)]
    min_phrase_freq: usize,

    /// Optional limit on the number of samples to scan (None = scan all)
    #[arg(long)]
    sample_limit: Option<usize>,

    /// Number of sample previews to encode and display (default: 5)
    #[arg(long, default_value_t = 5)]
    preview_samples: usize,

    /// Optional test sentence to encode and evaluate
    #[arg(long)]
    test_sentence: Option<String>,
}

#[cfg(not(target_arch = "wasm32"))]
fn extract_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let normalized = text
        .replace("n't", " not")
        .replace("'re", " are")
        .replace("'ve", " have")
        .replace("'ll", " will")
        .replace("'d", " would")
        .replace("'m", " am")
        .replace("can't", "can not")
        .replace("won't", "will not")
        .replace("let's", "let us");

    for raw in normalized.split(|c: char| !c.is_alphabetic() && c != '\'') {
        let clean = raw.trim_matches('\'').to_lowercase();
        if clean.len() >= 2 || clean == "i" || clean == "a" {
            let base = clean.strip_suffix("'s").unwrap_or(&clean);
            if base.chars().all(|c| c.is_ascii_lowercase()) && !base.is_empty() {
                words.push(base.to_string());
            }
        }
    }
    words
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<()> {
    let args = Args::parse();

    println!("============================================================");
    println!("=== Building Data-Driven Kilo Vocab Mapping Model (4096) ===");
    println!("============================================================");
    println!("Input cache:          {}", args.input_cache.display());
    println!("Output vocab:         {}", args.output_vocab.display());
    println!("Target vocab size:    {}", args.vocab_size);
    println!("Max phrase tokens:    {}", args.max_phrases);
    println!("Min mapping score:    {:.2} (words below this are dropped)", args.min_mapping_score);
    println!("Max cluster size:     {}", args.max_cluster_size);
    println!("Min context freq:     {} (minimum occurrences for distributional matching)", args.min_context_freq);
    println!("Min word frequency:   {}", args.min_word_freq);
    println!("Min phrase frequency: {}", args.min_phrase_freq);

    ensure!(
        args.input_cache.exists(),
        "Input cache not found: {}",
        args.input_cache.display()
    );

    // 1. Scan corpus and count unigrams, recurring multi-word collocations, and N-gram context frames
    println!("\n[1/4] Scanning {} for unigrams, phrases, and N-gram context frames...", args.input_cache.display());
    let mut word_counts: HashMap<String, usize> = HashMap::new();
    let mut phrase_counts: HashMap<String, usize> = HashMap::new();
    let mut word_contexts: HashMap<String, Vec<ContextFrameHash>> = HashMap::new();
    let mut total_tokens = 0usize;
    let mut sample_count = 0usize;
    let started = std::time::Instant::now();

    sample_cache::stream_pairs_lenient(&args.input_cache, args.sample_limit, |pair| {
        sample_count += 1;
        let combined = format!("{} {}", pair.0, pair.1);
        let words = extract_words(&combined);
        let len = words.len();

        // Count unigrams and collect multi-gram context frames (left 4-gram & right 4-gram)
        for i in 0..len {
            total_tokens += 1;
            let w = &words[i];
            *word_counts.entry(w.clone()).or_insert(0) += 1;

            let w_minus2 = if i >= 2 { &words[i - 2] } else { "<s>" };
            let w_minus1 = if i >= 1 { &words[i - 1] } else { "<s>" };
            let w_plus1 = if i + 1 < len { &words[i + 1] } else { "</s>" };
            let w_plus2 = if i + 2 < len { &words[i + 2] } else { "</s>" };

            let h_left = kilo_vocab::hash_multigram_left(w_minus2, w_minus1, w_plus1);
            let h_right = kilo_vocab::hash_multigram_right(w_minus1, w_plus1, w_plus2);

            let frames = word_contexts.entry(w.clone()).or_default();
            // Cap at 150 context frames per word to maintain strict memory bounds
            if frames.len() < 150 && !frames.contains(&h_left) {
                frames.push(h_left);
            }
            if frames.len() < 150 && !frames.contains(&h_right) {
                frames.push(h_right);
            }
        }

        // Count 2-word and 3-word n-gram phrases
        for i in 0..len {
            if i + 1 < len {
                let bigram = format!("{} {}", words[i], words[i + 1]);
                *phrase_counts.entry(bigram).or_insert(0) += 1;
            }
            if i + 2 < len {
                let trigram = format!("{} {} {}", words[i], words[i + 1], words[i + 2]);
                *phrase_counts.entry(trigram).or_insert(0) += 1;
            }
        }

        if sample_count % 50_000 == 0 {
            println!(
                "  Scanned {:>7} samples ({:>8} words, {:>6} unique words, {:>7} phrases)...",
                sample_count, total_tokens, word_counts.len(), phrase_counts.len()
            );
        }
        Ok(())
    })?;

    let scan_elapsed = started.elapsed().as_secs_f64();
    println!(
        "Scan complete: {} samples, {} word tokens ({} unique words, {} unique phrases) in {:.2}s ({:.0} samples/s)",
        sample_count, total_tokens, word_counts.len(), phrase_counts.len(), scan_elapsed, sample_count as f64 / scan_elapsed.max(0.001)
    );

    // Filter by minimum frequency
    word_counts.retain(|_, count| *count >= args.min_word_freq);
    phrase_counts.retain(|_, count| *count >= args.min_phrase_freq);

    println!(
        "Filtered candidate pool: {} words (freq >= {}), {} phrases (freq >= {})",
        word_counts.len(), args.min_word_freq, phrase_counts.len(), args.min_phrase_freq
    );

    // 2. Build KiloVocab mapping model using empirical N-gram context distributions
    println!("\n[2/4] Inducing 4,096-token vocabulary using multi-gram context distributions (Zero WordNet)...");
    let model = KiloVocab::learn_from_corpus(
        &word_counts,
        &phrase_counts,
        args.vocab_size,
        args.max_phrases,
        args.min_mapping_score,
        args.max_cluster_size,
        args.min_context_freq,
        Some(&word_contexts),
    );

    let phrase_token_count = model.learned_vocab.iter().filter(|w| w.contains(' ')).count();
    let unigram_token_count = model.learned_vocab.len() - phrase_token_count;
    let mapped_anchors = model.allowable_clusters.len();
    let total_cluster_words: usize = model.allowable_clusters.values().map(|c| c.len()).sum();
    let total_candidate_words = word_counts.len();
    let unmapped_dropped = total_candidate_words.saturating_sub(unigram_token_count + total_cluster_words);

    println!("Vocabulary induction complete:");
    println!("  Total vocabulary size:      {}", model.vocab_size);
    println!("  Special tokens:             {}", model.special_tokens.len());
    println!("  Punctuation:                {}", model.punctuation.len());
    println!("  Placeholders:               {}", model.placeholders.len());
    println!("  Learned entries:            {}", model.learned_vocab.len());
    println!("    ↳ Single words:           {}", unigram_token_count);
    println!("    ↳ Multi-word phrases:     {}", phrase_token_count);
    println!("  Allowable clusters:         {} anchors with mapped words", mapped_anchors);
    println!("  Dynamically mapped words:   {} words (score >= {:.2})", total_cluster_words, args.min_mapping_score);
    println!("  Strictly dropped words:     {} words (score < {:.2})", unmapped_dropped, args.min_mapping_score);

    // 3. Save to output JSON
    println!("\n[3/4] Saving model to {}...", args.output_vocab.display());
    model.save_to_json(&args.output_vocab)?;
    println!("Successfully saved {} (version {})!", args.output_vocab.display(), model.version);

    let tokenizer_dir = args.output_vocab.parent().unwrap_or(std::path::Path::new(".")).join("kilo_talk_tokenizer");
    println!("Saving HuggingFace WordLevel tokenizer to {} ...", tokenizer_dir.display());
    model.save_tokenizer_json(&tokenizer_dir)?;
    println!("Successfully saved HuggingFace tokenizer.json at {}!", tokenizer_dir.display());

    let legacy_tok_dir = args.output_vocab.parent().unwrap_or(std::path::Path::new(".")).join("yumon_talk_tokenizer");
    let _ = model.save_tokenizer_json(&legacy_tok_dir);

    // 4. Previews and Analytics
    println!("\n[4/4] Running encoding previews and quality analytics...");
    let mut preview_count = 0usize;
    sample_cache::stream_pairs_lenient(&args.input_cache, Some(args.preview_samples), |(prompt, reply)| {
        preview_count += 1;
        println!("\n--- Preview Sample #{} ---", preview_count);
        let sample_text = if !reply.trim().is_empty() {
            format!("User: {}\nYumon: {}", prompt.trim(), reply.trim())
        } else {
            prompt.trim().to_string()
        };

        let mut ctx = TranslationContext::new();
        match model.encode(&sample_text, &mut ctx) {
            Ok(tokens) => {
                let stats = model.analyze_encoding(&sample_text)?;
                let decoded = model.decode_with_context(&tokens, &ctx);
                println!("Original Text:\n{}", sample_text);
                println!("Tokens ({} tokens, raw words: {}, ratio: {:.2}x):", tokens.len(), stats.raw_word_count, stats.compression_ratio());
                println!("  Phrases matched: {}, Entities: {}, Numbers: {}, Slots: {}",
                    stats.phrase_tokens_used, ctx.entities.len(), ctx.numbers.len(), ctx.slots.len());
                println!("Decoded with context:\n{}", decoded);
            }
            Err(e) => {
                println!("Encoding failed: {}", e);
            }
        }
        Ok(())
    })?;

    // Optional custom test sentence
    if let Some(custom) = &args.test_sentence {
        println!("\n--- Custom Test Sentence ---");
        println!("Input: \"{}\"", custom);
        let mut ctx = TranslationContext::new();
        let tokens = model.encode(custom, &mut ctx)?;
        let stats = model.analyze_encoding(custom)?;
        let decoded = model.decode_with_context(&tokens, &ctx);
        println!("Tokens: {:?}", tokens);
        println!("Stats:  {} raw words -> {} tokens ({:.2}x compression), {} phrases used",
            stats.raw_word_count, tokens.len(), stats.compression_ratio(), stats.phrase_tokens_used);
        println!("Entities: {:?}", ctx.entities);
        println!("Numbers:  {:?}", ctx.numbers);
        println!("Slots:    {:?}", ctx.slots);
        println!("Decoded: \"{}\"", decoded);
    }

    println!("\n=== Done! Kilo Vocab Mapping Model ready for training and decompression. ===");
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}
