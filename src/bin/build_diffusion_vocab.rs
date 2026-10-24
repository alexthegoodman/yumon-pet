//! Build a data-driven 4,096-token Kilo Vocab Model specifically from art prompts.
//!
//! Scans all prompt texts across `data/animals_images/images-64/part-*.json` (or DiffusionDb),
//! learns unigram frequencies and recurring multi-word collocations (e.g. "digital painting",
//! "concept art", "octane render", "sharp focus"), ensuring artistic styles, subjects,
//! and descriptors have dedicated canonical tokens rather than collapsing into generic <SLOT_x> tags.

#[cfg(not(target_arch = "wasm32"))]
use {
    anyhow::{Result, ensure},
    clap::Parser,
    serde::Deserialize,
    std::collections::{HashMap, HashSet},
    std::fs::File,
    std::io::BufReader,
    std::path::PathBuf,
    std::time::Instant,
    yumon_pet::brain::{
        kilo_vocab::{self, KiloVocab, WordContextProfile},
        yumon_talk::{SEALED_WORDS, TranslationContext},
    },
};

#[derive(Deserialize, Debug)]
struct ManifestItem {
    p: String,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser, Debug)]
#[command(about = "Build a data-driven 4,096-token vocabulary specifically for art prompts")]
struct Args {
    /// Directory or directories containing part-*.json manifests (comma-separated or repeated).
    #[arg(long = "images-dir", alias = "images-dirs", value_delimiter = ',', num_args = 1..)]
    images_dir: Vec<PathBuf>,

    /// Path to save generated vocabulary JSON
    #[arg(long, default_value = "training-cache/diffusion_talk_vocab.json")]
    output_vocab: PathBuf,

    /// Total target vocabulary size (default: 4096)
    #[arg(long, default_value_t = 4096)]
    vocab_size: usize,

    /// Maximum number of recurring multi-word phrases (default: 400)
    #[arg(long, default_value_t = 400)]
    max_phrases: usize,

    /// Minimum confidence score to map a word into a canonical anchor (default: 0.60)
    #[arg(long, default_value_t = 0.60)]
    min_mapping_score: f64,

    /// Maximum number of mapped words per canonical anchor cluster (default: 25)
    #[arg(long, default_value_t = 25)]
    max_cluster_size: usize,

    /// Minimum word occurrences required to perform multi-gram distributional matching (default: 5)
    #[arg(long, default_value_t = 5)]
    min_context_freq: usize,

    /// Minimum word frequency in corpus to be considered (default: 2)
    #[arg(long, default_value_t = 2)]
    min_word_freq: usize,

    /// Minimum multi-word phrase frequency to be considered (default: 3)
    #[arg(long, default_value_t = 3)]
    min_phrase_freq: usize,
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
        if clean.len() >= 2 || clean == "i" || clean == "a" || clean == "k" || clean == "d" {
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
    let images_dirs = if args.images_dir.is_empty() {
        vec![PathBuf::from("data/animals_images/images-64")]
    } else {
        args.images_dir
    };

    println!("============================================================");
    println!("=== Building Diffusion Art Vocab Model (4096 tokens)    ===");
    println!("============================================================");
    for dir in &images_dirs {
        println!("Images directory:     {}", dir.display());
    }
    println!("Output vocabulary:    {}", args.output_vocab.display());
    println!("Target vocab size:    {}", args.vocab_size);
    println!("Max phrase tokens:    {}", args.max_phrases);
    println!("Min word freq:        {}", args.min_word_freq);
    println!("Min phrase freq:      {}", args.min_phrase_freq);

    // 1. Scan all manifest files across all directories
    let mut manifest_paths = Vec::new();
    for dir in &images_dirs {
        ensure!(
            dir.exists(),
            "Images directory not found: {}",
            dir.display()
        );

        let mut dir_manifests = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with("part-") && name.ends_with(".json") {
                        dir_manifests.push(path);
                    }
                }
            }
        }
        dir_manifests.sort();
        println!("Found {} manifest JSON files in {}", dir_manifests.len(), dir.display());
        manifest_paths.extend(dir_manifests);
    }
    println!("Total manifest JSON files found: {}", manifest_paths.len());
    ensure!(!manifest_paths.is_empty(), "No part-*.json manifests found in any specified directory");

    // 2. Extract prompts and count n-grams
    println!("\n[1/4] Scanning prompts for unigrams, phrases, and N-gram context frames...");
    let t0 = Instant::now();

    let mut word_counts: HashMap<String, usize> = HashMap::new();
    let mut phrase_counts: HashMap<String, usize> = HashMap::new();
    let mut word_contexts: HashMap<String, WordContextProfile> = HashMap::new();
    let mut sample_prompts: Vec<String> = Vec::new();
    let mut total_prompts = 0usize;
    let mut total_tokens = 0usize;

    let mut sealed_set: HashSet<&'static str> = HashSet::with_capacity(256);
    for &(canon, syns) in SEALED_WORDS {
        sealed_set.insert(canon);
        for &s in syns {
            sealed_set.insert(s);
        }
    }

    for manifest_path in &manifest_paths {
        let file = File::open(manifest_path)?;
        let reader = BufReader::new(file);
        let items: HashMap<String, ManifestItem> = serde_json::from_reader(reader)?;

        for (_key, item) in items {
            let prompt = item.p.trim();
            if prompt.is_empty() {
                continue;
            }
            if sample_prompts.len() < 10 {
                sample_prompts.push(prompt.to_string());
            }

            total_prompts += 1;
            let words = extract_words(prompt);
            let len = words.len();

            for i in 0..len {
                total_tokens += 1;
                let w = &words[i];
                *word_counts.entry(w.clone()).or_insert(0) += 1;

                // 4-gram context frames
                let w_minus2 = if i >= 2 { &words[i - 2] } else { "<s>" };
                let w_minus1 = if i >= 1 { &words[i - 1] } else { "<s>" };
                let w_plus1 = if i + 1 < len { &words[i + 1] } else { "</s>" };
                let w_plus2 = if i + 2 < len { &words[i + 2] } else { "</s>" };

                let h_left = kilo_vocab::hash_multigram_left(w_minus2, w_minus1, w_plus1);
                let h_right = kilo_vocab::hash_multigram_right(w_minus1, w_plus1, w_plus2);

                let prof = word_contexts.entry(w.clone()).or_default();
                prof.add_multigram(h_left);
                prof.add_multigram(h_right);

                // Look back up to 3 words for nearest content word
                for offset in 1..=3 {
                    if i >= offset {
                        let prev = &words[i - offset];
                        if !sealed_set.contains(prev.as_str()) {
                            prof.add_left(kilo_vocab::fast_word_hash(prev));
                            break;
                        }
                    }
                }

                // Look forward up to 3 words for nearest content word
                for offset in 1..=3 {
                    if i + offset < len {
                        let next = &words[i + offset];
                        if !sealed_set.contains(next.as_str()) {
                            prof.add_right(kilo_vocab::fast_word_hash(next));
                            break;
                        }
                    }
                }
            }

            // Bigrams and Trigrams
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
        }
    }

    println!(
        "Scan complete: {} prompts, {} tokens ({} unique words, {} unique phrases) in {:.2}s",
        total_prompts, total_tokens, word_counts.len(), phrase_counts.len(), t0.elapsed().as_secs_f64()
    );

    // Filter candidate pool
    word_counts.retain(|_, count| *count >= args.min_word_freq);
    phrase_counts.retain(|_, count| *count >= args.min_phrase_freq);

    println!(
        "Filtered candidate pool: {} words (freq >= {}), {} phrases (freq >= {})",
        word_counts.len(), args.min_word_freq, phrase_counts.len(), args.min_phrase_freq
    );

    // 3. Induce vocabulary model
    println!("\n[2/4] Inducing 4,096-token vocabulary from DiffusionDb corpus...");
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

    println!("Vocabulary induction complete:");
    println!("  Total vocabulary size:      {}", model.vocab_size);
    println!("  Special tokens:             {}", model.special_tokens.len());
    println!("  Punctuation:                {}", model.punctuation.len());
    println!("  Placeholders:               {}", model.placeholders.len());
    println!("  Learned entries:            {}", model.learned_vocab.len());
    println!("    ↳ Single words:           {}", unigram_token_count);
    println!("    ↳ Multi-word art phrases: {}", phrase_token_count);
    println!("  Allowable clusters:         {} anchors", model.allowable_clusters.len());

    // Print top phrases mined
    let top_phrases: Vec<_> = model.learned_vocab.iter().filter(|w| w.contains(' ')).take(15).collect();
    println!("  Top mined art phrases:      {:?}", top_phrases);

    // 4. Save to JSON
    println!("\n[3/4] Saving vocabulary to {}...", args.output_vocab.display());
    if let Some(parent) = args.output_vocab.parent() {
        std::fs::create_dir_all(parent)?;
    }
    model.save_to_json(&args.output_vocab)?;
    println!("Successfully saved {}!", args.output_vocab.display());

    let tokenizer_dir = args.output_vocab.parent().unwrap_or(std::path::Path::new(".")).join("diffusion_tokenizer");
    model.save_tokenizer_json(&tokenizer_dir)?;
    println!("Successfully saved matching tokenizer.json at {}!", tokenizer_dir.display());

    // 5. Analytics & Tokenization Previews on real prompts
    println!("\n[4/4] Evaluating tokenization quality on real prompts...");
    for (i, p) in sample_prompts.iter().take(5).enumerate() {
        println!("\n--- Prompt #{} ---", i + 1);
        println!("Raw: \"{}\"", p);
        let mut ctx = TranslationContext::new();
        let tokens = model.encode(p, &mut ctx)?;
        let stats = model.analyze_encoding(p)?;
        let decoded = model.decode_with_context(&tokens, &ctx);
        println!("Tokens ({} tokens, raw words: {}, ratio: {:.2}x):", tokens.len(), stats.raw_word_count, stats.compression_ratio());
        println!("  Phrases: {}, Entities: {}, Numbers: {}, Slots: {}",
            stats.phrase_tokens_used, ctx.entities.len(), ctx.numbers.len(), ctx.slots.len());
        println!("Decoded: \"{}\"", decoded);
    }

    println!("\n=== Done! diffusion_talk_vocab.json is ready for training. ===");
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}
