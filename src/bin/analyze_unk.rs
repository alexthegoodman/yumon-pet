//! Analyze UNK tokens in samples.bin against Yumon Talk vocabulary.

#[cfg(not(target_arch = "wasm32"))]
use {
    anyhow::{Context, Result, ensure},
    clap::Parser,
    rayon::prelude::*,
    std::collections::HashMap,
    std::path::PathBuf,
    std::sync::Mutex,
    yumon_pet::brain::{
        sample_cache,
        yumon_talk::{self, YumonTalkVocab, UNK_ID},
    },
};

#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser)]
#[command(about = "Analyze UNK tokens and placeholder strategies for Yumon Talk conversion")]
struct Args {
    #[arg(long, default_value = "training-cache/samples.bin")]
    input_cache: PathBuf,

    #[arg(long, default_value = "training-cache/yumon_talk_vocab.json")]
    vocab: PathBuf,

    #[arg(long, default_value_t = 1024)]
    max_seq_len: usize,

    /// Number of samples to analyze (None = all samples).
    #[arg(long)]
    limit: Option<usize>,
}

#[cfg(not(target_arch = "wasm32"))]
struct SampleUnkStats {
    prompt_unks: Vec<String>,
    reply_unks: Vec<String>,
    prompt_names: usize,
    reply_names: usize,
    prompt_tokens: usize,
    reply_tokens: usize,
}

#[cfg(not(target_arch = "wasm32"))]
fn extract_words_and_analyze(
    text: &str,
    vocab: &YumonTalkVocab,
) -> (Vec<usize>, Vec<String>, usize) {
    let mut tokens = Vec::new();
    let mut unk_words = Vec::new();
    let mut name_count = 0;

    let mut ctx = yumon_talk::TranslationContext::new();
    let mut entity_map: HashMap<String, usize> = HashMap::new();

    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        if chars[i].is_whitespace() {
            if chars[i] == '\n' {
                if let Some(id) = vocab.token_to_id("\n") {
                    tokens.push(id);
                }
            }
            i += 1;
            continue;
        }

        if chars[i].is_alphabetic() || chars[i] == '_' {
            let start = i;
            while i < len && (chars[i].is_alphanumeric() || chars[i] == '_' || (chars[i] == '\'' && i + 1 < len && chars[i + 1].is_alphabetic())) {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();

            if let Some(id) = vocab.resolve_english_word(&word) {
                tokens.push(id);
                continue;
            }

            let is_entity = chars[start].is_uppercase()
                && (i - start > 1 || (chars[start] != 'A' && chars[start] != 'I'));

            if is_entity {
                let ent_idx = entity_map.entry(word.clone()).or_insert_with(|| {
                    let curr = ctx.entities.len();
                    ctx.entities.push(word);
                    curr % 48
                });
                let slot = format!("<NAME_{ent_idx}>");
                tokens.push(vocab.token_to_id(&slot).unwrap_or(UNK_ID));
                name_count += 1;
            } else {
                tokens.push(UNK_ID);
                unk_words.push(word.to_lowercase());
            }
            continue;
        }

        if chars[i].is_ascii_digit() {
            let start = i;
            while i < len && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            let num_str: String = chars[start..i].iter().collect();
            let idx = ctx.numbers.len() % 20;
            ctx.numbers.push(num_str);
            let slot = format!("<NUM_{idx}>");
            tokens.push(vocab.token_to_id(&slot).unwrap_or(UNK_ID));
            continue;
        }

        let ch = chars[i];
        let punct_str = ch.to_string();
        if let Some(id) = vocab.token_to_id(&punct_str) {
            tokens.push(id);
        }
        i += 1;
    }

    (tokens, unk_words, name_count)
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<()> {
    let args = Args::parse();

    println!("Initializing thesaurus dictionary...");
    thesaurus::init();
    yumon_talk::get_thesaurus_dict();

    println!("Loading Yumon Talk vocabulary from {}...", args.vocab.display());
    let vocab = YumonTalkVocab::load_from_file(&args.vocab)?;

    println!("Loading samples from {} (limit: {:?})...", args.input_cache.display(), args.limit);
    let (meta, samples) = sample_cache::load_cache_lenient_limit(&args.input_cache, args.max_seq_len, args.limit)?;
    println!("Loaded {} samples (cache total: {}).", samples.len(), meta.samples);

    let started = std::time::Instant::now();
    println!("Analyzing UNK tokens across {} samples...", samples.len());

    let results: Vec<SampleUnkStats> = samples
        .par_iter()
        .map(|sample| {
            let (p_toks, p_unks, p_names) = extract_words_and_analyze(&sample.pair.0, &vocab);
            let (r_toks, r_unks, r_names) = extract_words_and_analyze(&sample.pair.1, &vocab);
            SampleUnkStats {
                prompt_unks: p_unks,
                reply_unks: r_unks,
                prompt_names: p_names,
                reply_names: r_names,
                prompt_tokens: p_toks.len(),
                reply_tokens: r_toks.len(),
            }
        })
        .collect();

    let elapsed = started.elapsed().as_secs_f64();
    println!("Analysis completed in {:.2}s ({:.1} samples/s)\n", elapsed, samples.len() as f64 / elapsed);

    let n = results.len();
    let mut unk_freq: HashMap<String, usize> = HashMap::new();
    let mut prompt_unk_freq: HashMap<String, usize> = HashMap::new();
    let mut reply_unk_freq: HashMap<String, usize> = HashMap::new();

    let mut total_unk_dist = vec![0usize; 16]; // 0, 1, 2, ..., 14, 15+
    let mut reply_unk_dist = vec![0usize; 16];
    let mut prompt_unk_dist = vec![0usize; 16];

    let mut current_passed = 0usize;
    let mut passed_if_relax_prompt_unk = 0usize;
    let mut passed_if_1_reply_unk = 0usize;
    let mut passed_if_2_reply_unk = 0usize;
    let mut passed_if_3_reply_unk = 0usize;

    for r in &results {
        let p_u = r.prompt_unks.len();
        let r_u = r.reply_unks.len();
        let total_u = p_u + r_u;

        for w in &r.prompt_unks {
            *unk_freq.entry(w.clone()).or_insert(0) += 1;
            *prompt_unk_freq.entry(w.clone()).or_insert(0) += 1;
        }
        for w in &r.reply_unks {
            *unk_freq.entry(w.clone()).or_insert(0) += 1;
            *reply_unk_freq.entry(w.clone()).or_insert(0) += 1;
        }

        total_unk_dist[total_u.min(15)] += 1;
        reply_unk_dist[r_u.min(15)] += 1;
        prompt_unk_dist[p_u.min(15)] += 1;

        let names_ok = r.reply_names <= 2;
        if total_u == 0 && names_ok {
            current_passed += 1;
        }
        if r_u == 0 && names_ok {
            passed_if_relax_prompt_unk += 1;
        }
        if r_u <= 1 && names_ok {
            passed_if_1_reply_unk += 1;
        }
        if r_u <= 2 && names_ok {
            passed_if_2_reply_unk += 1;
        }
        if r_u <= 3 && names_ok {
            passed_if_3_reply_unk += 1;
        }
    }

    println!("=== CURRENT YUMON TALK RETENTION ===");
    println!("Total samples analyzed: {}", n);
    println!("Currently passing (0 UNK, <= 2 reply names): {} ({:.2}%)\n", current_passed, 100.0 * current_passed as f64 / n as f64);

    println!("=== TOTAL UNK COUNT PER SAMPLE (PROMPT + REPLY) ===");
    let mut cum = 0;
    for (k, &cnt) in total_unk_dist.iter().enumerate() {
        cum += cnt;
        let label = if k == 15 { "15+".to_string() } else { format!("{k}") };
        println!("  {:>3} UNKs: {:>7} samples ({:>5.2}%) | Cumulative: {:>7} ({:>5.2}%)", label, cnt, 100.0 * cnt as f64 / n as f64, cum, 100.0 * cum as f64 / n as f64);
    }

    println!("\n=== REPLY UNK COUNT PER SAMPLE (TARGET REPLY ONLY) ===");
    let mut cum_r = 0;
    for (k, &cnt) in reply_unk_dist.iter().enumerate() {
        cum_r += cnt;
        let label = if k == 15 { "15+".to_string() } else { format!("{k}") };
        println!("  {:>3} UNKs: {:>7} samples ({:>5.2}%) | Cumulative: {:>7} ({:>5.2}%)", label, cnt, 100.0 * cnt as f64 / n as f64, cum_r, 100.0 * cum_r as f64 / n as f64);
    }

    println!("\n=== PROMPT UNK COUNT PER SAMPLE (USER INPUT ONLY) ===");
    let mut cum_p = 0;
    for (k, &cnt) in prompt_unk_dist.iter().enumerate() {
        cum_p += cnt;
        let label = if k == 15 { "15+".to_string() } else { format!("{k}") };
        println!("  {:>3} UNKs: {:>7} samples ({:>5.2}%) | Cumulative: {:>7} ({:>5.2}%)", label, cnt, 100.0 * cnt as f64 / n as f64, cum_p, 100.0 * cum_p as f64 / n as f64);
    }

    println!("\n=== SIMULATED PLACEHOLDER WORKAROUNDS ===");
    println!("1. Current baseline (0 UNK in prompt & reply):        {:>7} ({:.2}%)", current_passed, 100.0 * current_passed as f64 / n as f64);
    println!("2. Allow placeholders in Prompt (0 UNK in Reply):     {:>7} ({:.2}%)", passed_if_relax_prompt_unk, 100.0 * passed_if_relax_prompt_unk as f64 / n as f64);
    println!("3. Allow <= 1 Placeholder token in Reply:             {:>7} ({:.2}%)", passed_if_1_reply_unk, 100.0 * passed_if_1_reply_unk as f64 / n as f64);
    println!("4. Allow <= 2 Placeholder tokens in Reply:            {:>7} ({:.2}%)", passed_if_2_reply_unk, 100.0 * passed_if_2_reply_unk as f64 / n as f64);
    println!("5. Allow <= 3 Placeholder tokens in Reply:            {:>7} ({:.2}%)", passed_if_3_reply_unk, 100.0 * passed_if_3_reply_unk as f64 / n as f64);

    let mut sorted_unks: Vec<(String, usize)> = unk_freq.into_iter().collect();
    sorted_unks.sort_by(|a, b| b.1.cmp(&a.1));

    println!("\n=== UNIQUE UNK WORDS ===");
    println!("Total distinct words that became UNK: {}", sorted_unks.len());
    let total_unk_tokens: usize = sorted_unks.iter().map(|(_, c)| c).sum();
    println!("Total UNK token instances across corpus: {}", total_unk_tokens);

    let top_50_count: usize = sorted_unks.iter().take(50).map(|(_, c)| c).sum();
    let top_200_count: usize = sorted_unks.iter().take(200).map(|(_, c)| c).sum();
    let top_500_count: usize = sorted_unks.iter().take(500).map(|(_, c)| c).sum();
    let top_1000_count: usize = sorted_unks.iter().take(1000).map(|(_, c)| c).sum();

    println!("Top 50 UNK words account for:   {:>7} ({:.1}%) of all UNK tokens", top_50_count, 100.0 * top_50_count as f64 / total_unk_tokens as f64);
    println!("Top 200 UNK words account for:  {:>7} ({:.1}%) of all UNK tokens", top_200_count, 100.0 * top_200_count as f64 / total_unk_tokens as f64);
    println!("Top 500 UNK words account for:  {:>7} ({:.1}%) of all UNK tokens", top_500_count, 100.0 * top_500_count as f64 / total_unk_tokens as f64);
    println!("Top 1000 UNK words account for: {:>7} ({:.1}%) of all UNK tokens", top_1000_count, 100.0 * top_1000_count as f64 / total_unk_tokens as f64);

    println!("\n=== TOP 60 MOST FREQUENT UNK WORDS ===");
    for (rank, (word, count)) in sorted_unks.iter().take(60).enumerate() {
        let p_count = prompt_unk_freq.get(word).copied().unwrap_or(0);
        let r_count = reply_unk_freq.get(word).copied().unwrap_or(0);
        println!("{:>3}. {:<20} total: {:>6} (prompt: {:>5}, reply: {:>5})", rank + 1, word, count, p_count, r_count);
    }

    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}
