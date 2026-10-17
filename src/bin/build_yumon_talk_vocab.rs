//! Build a dynamic, reproducible Yumon Talk vocabulary (1024 tokens) from training-cache/samples.bin.
//!
//! 1. Scans training-cache/samples.bin and computes word frequencies across real training data.
//! 2. Programmatically uses the `thesaurus` crate and morphological stemming to deduplicate synonyms against higher-frequency words.
//! 3. Selects the top canonical words needed to hit the exact 1024-token budget.
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
#[command(about = "Build a reproducible 1024-token Yumon Talk vocabulary from samples.bin using thesaurus deduplication")]
struct Args {
    /// Path to input training cache (default: training-cache/samples.bin)
    #[arg(long, default_value = "training-cache/samples.bin")]
    input_cache: PathBuf,

    /// Path to save generated vocabulary JSON (default: training-cache/yumon_talk_vocab.json)
    #[arg(long, default_value = "training-cache/yumon_talk_vocab.json")]
    output_vocab: PathBuf,

    /// Number of canonical words to choose (default: 965, matching 1024 total tokens with 59 non-word tokens)
    #[arg(long, default_value_t = 965)]
    target_words: usize,

    /// Total target vocabulary size including special, punctuation and placeholders (default: 1024)
    #[arg(long, default_value_t = 1024)]
    vocab_size: usize,

    /// Minimum word frequency to consider in the corpus (filters out 1-off typos)
    #[arg(long, default_value_t = 2)]
    min_word_freq: usize,

    /// Limit number of input samples to scan (None = scan all samples)
    #[arg(long)]
    sample_limit: Option<usize>,

    /// Size of top candidate word pool to deduplicate (default: 4096)
    #[arg(long, default_value_t = 4096)]
    candidate_pool: usize,

    /// Number of sample clusters to preview
    #[arg(long, default_value_t = 15)]
    preview_clusters: usize,
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
    println!("Initializing thesaurus dictionary (static feature)...");
    thesaurus::init();

    println!("Input cache:        {}", args.input_cache.display());
    println!("Output vocab:       {}", args.output_vocab.display());
    println!("Target words:       {}", args.target_words);
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
        .into_iter()
        .filter(|(_, count)| *count >= args.min_word_freq)
        .collect();
    sorted_words.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    sorted_words.truncate(args.candidate_pool);

    println!("Candidate pool truncated to top {} most frequent words", sorted_words.len());

    // 3. Deduplicate using thesaurus and select canonical words
    println!("\n[3/4] Deduplicating synonyms via thesaurus to select {} canonical words...", args.target_words);

    let mut canonical_words: Vec<String> = Vec::with_capacity(args.target_words);
    let mut canonical_set: HashSet<String> = HashSet::with_capacity(args.target_words);
    let mut word_to_canonical: HashMap<String, String> = HashMap::new();
    let mut canonical_synonyms: HashMap<String, HashSet<String>> = HashMap::new();

    // Foundational closed-class seeds (pronouns, basic auxiliaries, core prepositions/conjunctions)
    let core_seeds = [
        ("i", &["i", "me", "myself"][..]),
        ("you", &["you", "thee", "thou", "yourself", "yourselves"][..]),
        ("he", &["he", "him", "himself"][..]),
        ("she", &["she", "her", "herself"][..]),
        ("it", &["it", "itself"][..]),
        ("we", &["we", "us", "ourselves"][..]),
        ("they", &["they", "them", "themselves"][..]),
        ("my", &["my", "mine"][..]),
        ("your", &["your", "yours", "thy", "thine"][..]),
        ("his", &["his"][..]),
        ("her", &["her", "hers"][..]),
        ("its", &["its"][..]),
        ("our", &["our", "ours"][..]),
        ("their", &["their", "theirs"][..]),
        ("this", &["this"][..]),
        ("that", &["that"][..]),
        ("these", &["these"][..]),
        ("those", &["those"][..]),
        ("who", &["who", "whom", "whose", "whoever"][..]),
        ("which", &["which", "whichever"][..]),
        ("what", &["what", "whatever"][..]),
        ("where", &["where", "wherever"][..]),
        ("when", &["when", "whenever"][..]),
        ("why", &["why"][..]),
        ("how", &["how"][..]),
        ("all", &["all", "everybody", "everyone", "everything", "whole"][..]),
        ("every", &["every", "each"][..]),
        ("any", &["any", "anybody", "anyone", "anything"][..]),
        ("some", &["some", "somebody", "someone", "something"][..]),
        ("none", &["none", "nobody", "nothing", "neither"][..]),
        ("both", &["both"][..]),
        ("the", &["the"][..]),
        ("a", &["a", "an"][..]),
        ("and", &["and", "also", "furthermore", "moreover"][..]),
        ("or", &["or", "either"][..]),
        ("but", &["but", "however", "although", "though", "yet"][..]),
        ("so", &["so", "therefore", "thus", "hence"][..]),
        ("because", &["because", "since", "as"][..]),
        ("if", &["if", "whether"][..]),
        ("then", &["then"][..]),
        ("in", &["in", "inside", "within"][..]),
        ("on", &["on", "upon"][..]),
        ("at", &["at"][..]),
        ("by", &["by"][..]),
        ("for", &["for"][..]),
        ("with", &["with"][..]),
        ("without", &["without"][..]),
        ("about", &["about"][..]),
        ("from", &["from"][..]),
        ("to", &["to"][..]),
        ("of", &["of"][..]),
        ("be", &["be", "am", "is", "are", "was", "were", "been", "being"][..]),
        ("have", &["have", "has", "had", "having"][..]),
        ("do", &["do", "does", "did", "done", "doing"][..]),
        ("will", &["will", "shall", "would"][..]),
        ("can", &["can", "could", "able"][..]),
        ("may", &["may", "might"][..]),
        ("must", &["must", "should", "ought"][..]),
        ("not", &["not", "never"][..]),
        ("no", &["no", "nay"][..]),
        ("yes", &["yes", "yeah", "aye"][..]),
    ];

    for &(canon, syns) in core_seeds.iter() {
        if canonical_words.len() < args.target_words {
            canonical_words.push(canon.to_string());
            canonical_set.insert(canon.to_string());
            word_to_canonical.insert(canon.to_string(), canon.to_string());
            let entry = canonical_synonyms.entry(canon.to_string()).or_default();
            entry.insert(canon.to_string());
            for &s in syns {
                word_to_canonical.insert(s.to_string(), canon.to_string());
                entry.insert(s.to_string());
            }
        }
    }

    let mut deduplicated_count = 0usize;

    for (word, _freq) in sorted_words.iter() {
        if canonical_words.len() >= args.target_words {
            break;
        }

        // Already covered?
        if word_to_canonical.contains_key(word) {
            deduplicated_count += 1;
            continue;
        }

        // Check morphological stem
        let stems = [
            word.strip_suffix("ing").unwrap_or(word),
            word.strip_suffix("ed").unwrap_or(word),
            word.strip_suffix("es").unwrap_or(word),
            word.strip_suffix("s").unwrap_or(word),
            word.strip_suffix("ly").unwrap_or(word),
            word.strip_suffix("er").unwrap_or(word),
            word.strip_suffix("est").unwrap_or(word),
        ];

        let mut matched_stem = None;
        for &stem in stems.iter() {
            if stem != word && !stem.is_empty() {
                if let Some(canon) = word_to_canonical.get(stem) {
                    matched_stem = Some(canon.clone());
                    break;
                }
            }
        }

        if let Some(canon) = matched_stem {
            word_to_canonical.insert(word.clone(), canon.clone());
            canonical_synonyms.entry(canon).or_default().insert(word.clone());
            deduplicated_count += 1;
            continue;
        }

        // Query thesaurus for synonyms
        let syns = thesaurus::synonyms(word);
        let mut matched_syn_canon = None;

        for syn in syns.iter() {
            let clean = syn.trim().to_lowercase();
            if let Some(canon) = word_to_canonical.get(&clean) {
                matched_syn_canon = Some(canon.clone());
                break;
            }
        }

        if let Some(canon) = matched_syn_canon {
            // Word is a synonym of an existing canonical word!
            word_to_canonical.insert(word.clone(), canon.clone());
            canonical_synonyms.entry(canon.clone()).or_default().insert(word.clone());
            // Also map remaining synonyms to this canonical word
            for syn in syns.iter() {
                let clean = syn.trim().to_lowercase();
                if clean.chars().all(|c| c.is_ascii_lowercase()) && !word_to_canonical.contains_key(&clean) {
                    word_to_canonical.insert(clean.clone(), canon.clone());
                    canonical_synonyms.entry(canon.clone()).or_default().insert(clean);
                }
            }
            deduplicated_count += 1;
            continue;
        }

        // New canonical word!
        canonical_words.push(word.clone());
        canonical_set.insert(word.clone());
        word_to_canonical.insert(word.clone(), word.clone());
        let cluster = canonical_synonyms.entry(word.clone()).or_default();
        cluster.insert(word.clone());

        for syn in syns.iter() {
            let clean = syn.trim().to_lowercase();
            if clean.chars().all(|c| c.is_ascii_lowercase()) && !clean.is_empty() && !word_to_canonical.contains_key(&clean) {
                word_to_canonical.insert(clean.clone(), word.clone());
                cluster.insert(clean);
            }
        }
    }

    println!("Selected {} canonical words (deduplicated {} words during selection)", canonical_words.len(), deduplicated_count);

    // 4. Enrich synonym clusters with any remaining corpus words
    println!("\n[4/4] Mapping remaining corpus words to canonical clusters...");
    let mut mapped_inflections = 0usize;
    let mut mapped_thesaurus = 0usize;

    // First pass: morphological stemming across sorted words (instant string operations)
    for (word, _freq) in sorted_words.iter() {
        if word_to_canonical.contains_key(word) {
            continue;
        }
        let stems = [
            word.strip_suffix("ing").unwrap_or(word),
            word.strip_suffix("ed").unwrap_or(word),
            word.strip_suffix("es").unwrap_or(word),
            word.strip_suffix("s").unwrap_or(word),
            word.strip_suffix("ly").unwrap_or(word),
            word.strip_suffix("er").unwrap_or(word),
            word.strip_suffix("est").unwrap_or(word),
        ];
        for &stem in stems.iter() {
            if stem != word && !stem.is_empty() {
                if let Some(canon) = word_to_canonical.get(stem).cloned() {
                    word_to_canonical.insert(word.clone(), canon.clone());
                    canonical_synonyms.entry(canon).or_default().insert(word.clone());
                    mapped_inflections += 1;
                    break;
                }
            }
        }
    }

    // Second pass: query thesaurus for top 10,000 words that remain unmapped
    for (word, _freq) in sorted_words.iter().take(10000) {
        if word_to_canonical.contains_key(word) {
            continue;
        }
        let syns = thesaurus::synonyms(word);
        for syn in syns.iter() {
            let clean = syn.trim().to_lowercase();
            if let Some(canon) = word_to_canonical.get(&clean).cloned() {
                word_to_canonical.insert(word.clone(), canon.clone());
                canonical_synonyms.entry(canon).or_default().insert(word.clone());
                mapped_thesaurus += 1;
                break;
            }
        }
    }

    println!("Enriched clusters: mapped {} inflections and {} high-frequency synonyms", mapped_inflections, mapped_thesaurus);

    // Convert synonym sets to sorted Vecs
    let mut final_synonym_clusters: HashMap<String, Vec<String>> = HashMap::new();
    let mut total_synonym_mappings = 0usize;

    for (canon, set) in canonical_synonyms.into_iter() {
        let mut list: Vec<String> = set.into_iter().collect();
        list.sort();
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
    ];
    let mut placeholders = Vec::with_capacity(40);
    for i in 0..32 {
        placeholders.push(format!("<NAME_{i}>"));
    }
    for i in 0..8 {
        placeholders.push(format!("<NUM_{i}>"));
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

    let file_size_kb = std::fs::metadata(&args.output_vocab)?.len() as f64 / 1024.0;
    println!("Successfully saved dynamic Yumon Talk vocabulary ({:.1} KB)!\n", file_size_kb);

    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("Vocabulary generation requires a native build.");
}
