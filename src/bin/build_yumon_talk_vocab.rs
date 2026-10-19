//! Build a dynamic, reproducible Yumon Talk vocabulary (4096 tokens) from training-cache/samples.bin.
//!
//! 1. Scans training-cache/samples.bin and computes word frequencies across real training data.
//! 2. Programmatically uses the `thesaurus` crate and morphological stemming to deduplicate synonyms against higher-frequency words.
//! 3. Selects the top canonical words needed to hit the exact 4096-token budget.
//! 4. Builds forward and reverse synonym mapping clusters.
//! 5. Saves the generated vocabulary to training-cache/yumon_talk_vocab.json.

#[cfg(not(target_arch = "wasm32"))]
use {
    anyhow::{Result, ensure},
    clap::Parser,
    std::collections::{HashMap, HashSet},
    std::path::PathBuf,
    yumon_pet::brain::{
        sample_cache,
        yumon_talk::YumonTalkVocab,
    },
};

#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser, Debug)]
#[command(about = "Build a reproducible 4096-token Yumon Talk vocabulary from samples.bin using thesaurus deduplication")]
struct Args {
    /// Path to input training cache (default: training-cache/samples.bin)
    #[arg(long, default_value = "training-cache/samples.bin")]
    input_cache: PathBuf,

    /// Path to save generated vocabulary JSON (default: training-cache/yumon_talk_vocab.json)
    #[arg(long, default_value = "training-cache/yumon_talk_vocab.json")]
    output_vocab: PathBuf,

    /// Number of canonical words to choose (default: calculated to meet vocab_size)
    #[arg(long)]
    target_words: Option<usize>,

    /// Total target vocabulary size including special, punctuation and placeholders (default: 4096)
    #[arg(long, default_value_t = 4096)]
    vocab_size: usize,

    /// Minimum word frequency to consider in the corpus (filters out 1-off typos)
    #[arg(long, default_value_t = 2)]
    min_word_freq: usize,

    /// Limit number of input samples to scan (None = scan all samples)
    #[arg(long)]
    sample_limit: Option<usize>,

    /// Size of top candidate word pool to deduplicate (default: 35000)
    #[arg(long, default_value_t = 35000)]
    candidate_pool: usize,

    /// Number of sample clusters to preview
    #[arg(long, default_value_t = 15)]
    preview_clusters: usize,

    /// Dump sorted word frequencies to JSON and exit
    #[arg(long)]
    dump_word_counts: Option<PathBuf>,

    /// Path to curated synonyms JSON (defaults to built-in curated dictionary)
    #[arg(long)]
    curated_synonyms: Option<PathBuf>,

    /// Enable legacy WordNet thesaurus expansion (default: false)
    #[arg(long, default_value_t = false)]
    enable_thesaurus: bool,
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

    println!("=== Building Dynamic Yumon Talk Vocabulary ===");

    let non_word_count = 4 + 24 + (48 + 20 + 16); // 4 special + 24 punct + 84 placeholders = 112
    let target_words = args.target_words.unwrap_or(args.vocab_size.saturating_sub(non_word_count));

    println!("Input cache:        {}", args.input_cache.display());
    println!("Output vocab:       {}", args.output_vocab.display());
    println!("Target words:       {}", target_words);
    println!("Target vocab size:  {}", args.vocab_size);
    println!("Min word frequency: {}", args.min_word_freq);

    ensure!(
        args.input_cache.exists(),
        "Input cache not found: {}",
        args.input_cache.display()
    );

    // 1. Scan samples.bin and count word frequencies
    println!("\n[1/4] Scanning {} for word frequencies...", args.input_cache.display());
    let mut word_counts: HashMap<String, usize> = HashMap::new();
    let mut total_tokens = 0usize;
    let mut sample_count = 0usize;
    let started = std::time::Instant::now();

    sample_cache::stream_pairs_lenient(&args.input_cache, args.sample_limit, |pair| {
        sample_count += 1;
        let combined = format!("{} {}", pair.0, pair.1);
        for word in extract_words(&combined) {
            total_tokens += 1;
            *word_counts.entry(word).or_insert(0) += 1;
        }
        if sample_count % 50000 == 0 {
            println!("  Scanned {:>7} samples ({:>8} tokens, {:>6} unique words)...", sample_count, total_tokens, word_counts.len());
        }
        Ok(())
    })?;

    let scan_elapsed = started.elapsed().as_secs_f64();
    println!(
        "Scan complete: {} samples, {} total word tokens, {} unique words in {:.2}s ({:.0} samples/s)",
        sample_count, total_tokens, word_counts.len(), scan_elapsed, sample_count as f64 / scan_elapsed.max(0.001)
    );

    // 2. Sort words by frequency descending and keep top candidate pool
    println!("\n[2/4] Sorting words and keeping top {} most frequent candidate words...", args.candidate_pool);
    let mut sorted_words: Vec<(String, usize)> = word_counts
        .iter()
        .filter(|(_, count)| **count >= args.min_word_freq)
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    sorted_words.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    sorted_words.truncate(args.candidate_pool);

    println!("Candidate pool truncated to top {} most frequent words", sorted_words.len());
    if let Some(dump_path) = &args.dump_word_counts {
        println!("Dumping {} word counts to {}...", sorted_words.len(), dump_path.display());
        let file = std::fs::File::create(dump_path)?;
        serde_json::to_writer_pretty(file, &sorted_words)?;
        println!("Successfully dumped word counts to {}!", dump_path.display());
        return Ok(());
    }

    // 3. Deduplicate using curated synonyms and morphological variants to select canonical words
    println!("\n[3/4] Selecting {} canonical words using curated synonyms & morphological variants...", target_words);

    let mut canonical_words: Vec<String> = Vec::with_capacity(target_words);
    let mut canonical_set: HashSet<String> = HashSet::with_capacity(target_words);
    let mut word_to_canonical: HashMap<String, String> = HashMap::new();
    let mut canonical_synonyms: HashMap<String, HashSet<String>> = HashMap::new();

    // Foundational closed-class seeds (pronouns, basic auxiliaries, core prepositions/conjunctions)
    let mut sealed_set: HashSet<String> = HashSet::new();

    for &(canon, syns) in yumon_pet::brain::yumon_talk::SEALED_WORDS.iter() {
        if canonical_words.len() < target_words {
            canonical_words.push(canon.to_string());
            canonical_set.insert(canon.to_string());
            sealed_set.insert(canon.to_string());
            word_to_canonical.insert(canon.to_string(), canon.to_string());
            let entry = canonical_synonyms.entry(canon.to_string()).or_default();
            entry.insert(canon.to_string());
            for &s in syns {
                sealed_set.insert(s.to_string());
                word_to_canonical.insert(s.to_string(), canon.to_string());
                entry.insert(s.to_string());
            }
        }
    }

    // Load curated synonyms (from args or built-in)
    let curated_map: HashMap<String, Vec<String>> = if let Some(path) = &args.curated_synonyms {
        println!("Loading custom curated synonyms from {} ...", path.display());
        let text = std::fs::read_to_string(path)?;
        serde_json::from_str(&text)?
    } else {
        println!("Using built-in curated synonyms ({} clusters)...", yumon_pet::brain::yumon_talk::get_curated_synonyms().len());
        yumon_pet::brain::yumon_talk::get_curated_synonyms().clone()
    };

    // Seed curated canonical words and their vetted synonyms
    for (canon, syns) in curated_map.iter() {
        if canonical_words.len() >= target_words {
            break;
        }
        if sealed_set.contains(canon) || canonical_set.contains(canon) {
            continue;
        }
        canonical_words.push(canon.clone());
        canonical_set.insert(canon.clone());
        word_to_canonical.insert(canon.clone(), canon.clone());
        let entry = canonical_synonyms.entry(canon.clone()).or_default();
        entry.insert(canon.clone());
        for s in syns {
            let clean = s.trim().to_lowercase();
            if !sealed_set.contains(&clean) {
                word_to_canonical.insert(clean.clone(), canon.clone());
                entry.insert(clean);
            }
        }
    }

    if args.enable_thesaurus {
        println!("Initializing thesaurus dictionary (static feature)...");
        thesaurus::init();
    }

    let mut deduplicated_count = 0usize;

    // Fill remaining canonical budget with highest-frequency open-class corpus words
    for (word, _freq) in sorted_words.iter() {
        if canonical_words.len() >= target_words {
            break;
        }

        // Already covered by sealed or curated cluster?
        if word_to_canonical.contains_key(word) {
            deduplicated_count += 1;
            continue;
        }

        // Check full morphological variants (handles -ied -> -y, irregular verbs, -s, -es, -ed, -ing, etc.)
        let mut matched_canon = None;
        for variant in yumon_pet::brain::yumon_talk::morphological_variants(word) {
            if variant != *word && !variant.is_empty() {
                if let Some(canon) = word_to_canonical.get(&variant) {
                    if !sealed_set.contains(canon) {
                        matched_canon = Some(canon.clone());
                        break;
                    }
                }
            }
        }

        if let Some(canon) = matched_canon {
            word_to_canonical.insert(word.clone(), canon.clone());
            canonical_synonyms.entry(canon).or_default().insert(word.clone());
            deduplicated_count += 1;
            continue;
        }

        // Optional legacy thesaurus deduplication
        if args.enable_thesaurus {
            let syns = thesaurus::synonyms(word);
            let mut matched = None;
            for syn in syns.iter() {
                let clean = syn.trim().to_lowercase();
                if clean.len() >= 2 && !sealed_set.contains(&clean) {
                    if let Some(canon) = word_to_canonical.get(&clean) {
                        if !sealed_set.contains(canon) && canonical_set.contains(canon) {
                            if thesaurus::synonyms(canon).iter().any(|s| s.trim().to_lowercase() == *word) {
                                matched = Some(canon.clone());
                                break;
                            }
                        }
                    }
                }
            }
            if let Some(canon) = matched {
                word_to_canonical.insert(word.clone(), canon.clone());
                canonical_synonyms.entry(canon).or_default().insert(word.clone());
                deduplicated_count += 1;
                continue;
            }
        }

        // Word becomes a NEW CANONICAL WORD!
        canonical_words.push(word.clone());
        canonical_set.insert(word.clone());
        word_to_canonical.insert(word.clone(), word.clone());
        let cluster = canonical_synonyms.entry(word.clone()).or_default();
        cluster.insert(word.clone());
    }

    println!("Selected {} canonical words (deduplicated {} words during selection)", canonical_words.len(), deduplicated_count);

    // 4. Enrich synonym clusters with remaining corpus words via morphological lemmatization
    println!("\n[4/4] Mapping remaining corpus words to canonical clusters...");
    let mut mapped_inflections = 0usize;

    for (word, _freq) in sorted_words.iter() {
        if word_to_canonical.contains_key(word) {
            continue;
        }
        for variant in yumon_pet::brain::yumon_talk::morphological_variants(word) {
            if variant != *word && !variant.is_empty() {
                if let Some(canon) = word_to_canonical.get(&variant).cloned() {
                    if !sealed_set.contains(&canon) {
                        word_to_canonical.insert(word.clone(), canon.clone());
                        canonical_synonyms.entry(canon).or_default().insert(word.clone());
                        mapped_inflections += 1;
                        break;
                    }
                }
            }
        }
    }

    println!("Enriched clusters: mapped {} inflections across the corpus", mapped_inflections);

    // Convert synonym sets to sorted, capped Vecs
    let mut final_synonym_clusters: HashMap<String, Vec<String>> = HashMap::new();
    let mut total_synonym_mappings = 0usize;

    for (canon, set) in canonical_synonyms.into_iter() {
        let list: Vec<String> = if sealed_set.contains(&canon) {
            let mut l: Vec<String> = set.into_iter().collect();
            l.sort();
            l
        } else {
            // Rank synonyms by frequency in the corpus descending, keeping top 35 most common
            let mut ranked: Vec<String> = set.into_iter().collect();
            ranked.sort_by(|a, b| {
                let fa = if a == &canon { usize::MAX } else { word_counts.get(a).copied().unwrap_or(0) };
                let fb = if b == &canon { usize::MAX } else { word_counts.get(b).copied().unwrap_or(0) };
                fb.cmp(&fa).then_with(|| a.cmp(b))
            });
            ranked.truncate(35);
            ranked.sort();
            ranked
        };
        total_synonym_mappings += list.len();
        final_synonym_clusters.insert(canon, list);
    }

    // 5. Construct YumonTalkVocab
    let special_tokens = vec![
        "<PAD>".to_string(),
        "<BOS>".to_string(),
        "<EOS>".to_string(),
        "<UNK>".to_string(),
    ];
    let punctuation = vec![
        ".".to_string(),
        ",".to_string(),
        "!".to_string(),
        "?".to_string(),
        ";".to_string(),
        ":".to_string(),
        "-".to_string(),
        "'".to_string(),
        "\"".to_string(),
        "(".to_string(),
        ")".to_string(),
        "[".to_string(),
        "]".to_string(),
        "/".to_string(),
        "\n".to_string(),
        "*".to_string(),
        "&".to_string(),
        "_".to_string(),
        "+".to_string(),
        "=".to_string(),
        "%".to_string(),
        "$".to_string(),
        "#".to_string(),
        "@".to_string(),
    ];
    let mut placeholders = Vec::with_capacity(84);
    for i in 0..48 {
        placeholders.push(format!("<NAME_{i}>"));
    }
    for i in 0..20 {
        placeholders.push(format!("<NUM_{i}>"));
    }
    for i in 0..16 {
        placeholders.push(format!("<SLOT_{i}>"));
    }

    let vocab = YumonTalkVocab::new(
        special_tokens,
        punctuation,
        placeholders,
        canonical_words,
        final_synonym_clusters,
    );

    ensure!(
        vocab.vocab_size == args.vocab_size,
        "Constructed vocab size {} does not match target {}",
        vocab.vocab_size,
        args.vocab_size
    );

    println!("\n=== Yumon Talk Vocabulary Summary ===");
    println!("Special tokens:           {}", vocab.special_tokens.len());
    println!("Punctuation tokens:       {}", vocab.punctuation.len());
    println!("Placeholders:             {}", vocab.placeholders.len());
    println!("Canonical words:          {}", vocab.canonical_words.len());
    println!("Total tokens:             {}", vocab.vocab_size);
    println!("Total synonym mappings:   {}", total_synonym_mappings);
    println!("Avg synonyms / word:      {:.1}", total_synonym_mappings as f64 / vocab.canonical_words.len().max(1) as f64);

    println!("\n=== Preview of Top {} Clusters ===", args.preview_clusters);
    for (i, canon) in vocab.canonical_words.iter().take(args.preview_clusters).enumerate() {
        let syns = vocab.synonym_clusters.get(canon).cloned().unwrap_or_default();
        let preview = if syns.len() > 8 {
            format!("{} (total {})", syns[..8].join(", "), syns.len())
        } else {
            syns.join(", ")
        };
        println!("  #{:>2} {:<15} -> [{}]", i + 1, canon, preview);
    }

    // 6. Save to disk
    println!("\nSaving dynamic vocabulary to {} ...", args.output_vocab.display());
    vocab.save_to_file(&args.output_vocab)?;

    let tokenizer_dir = args.output_vocab.parent().unwrap_or(std::path::Path::new(".")).join("yumon_talk_tokenizer");
    println!("Saving HuggingFace WordLevel tokenizer to {} ...", tokenizer_dir.display());
    vocab.save_tokenizer_json(&tokenizer_dir)?;

    let file_size_kb = std::fs::metadata(&args.output_vocab)?.len() as f64 / 1024.0;
    println!("Successfully saved dynamic Yumon Talk vocabulary ({:.1} KB) and tokenizer!\n", file_size_kb);

    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("Vocabulary generation requires a native build.");
}
