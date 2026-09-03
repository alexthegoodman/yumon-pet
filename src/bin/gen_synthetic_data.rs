// Generate synthetic human/Yumon Q&A pairs via an Ollama endpoint (Llama 3)
// and append them to data/synthetic/<topic>.txt in the `FileKind::Chats`
// format (see src/brain/mdx.rs::load_handcrafted_chats):
//
//   human line
//   bot line
//   human line
//   bot line
//
//   human line
//   bot line
//   ...
//
// (blocks separated by a single blank line)
//
// bible_asv.csv / bible_bbe.csv are currently loaded as raw verse text
// (FileKind::BibleCsv) and samples.rs then cuts each chunk in half by word
// count to fabricate an input/target pair, which routinely lands mid
// sentence. This tool instead asks an LLM to produce real, topically
// grounded question/answer turns in Yumon's short voice, so no arbitrary
// splitting is needed at all.
//
// Safe to interrupt (Ctrl-C) and re-run: progress and a dedup set are
// persisted per topic under data/synthetic/.state/, and generated pairs are
// appended and flushed to disk immediately after every model call.

use std::{
    collections::HashSet,
    fs::{self, OpenOptions},
    hash::{Hash, Hasher},
    io::Write,
    path::{Path, PathBuf},
    sync::OnceLock,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use clap::Parser;
use rand::Rng;
use regex::Regex;
use serde::{Deserialize, Serialize};

const BAD_WORDS: [&str; 5] = ["sex", "drug", "kill", "rape", "nazi"]; // mirrors src/brain/samples.rs
const MAX_SEEN_HASHES: usize = 20_000; // bound state-file growth
const BIBLE_WINDOW: usize = 7; // verses per grounding passage

const FEW_SHOT: [(&str, &str); 3] = [
    ("What is the universe?", "The universe is a wide open space filled with planets."),
    ("What does it mean to be wide?", "Something wide is big and hard to traverse."),
    ("What is a feeling?", "A feeling is an emotion like happiness or sadness."),
];

const SYSTEM_PROMPT: &str = r#"You generate training data for Yumon, a small tabletop pet AI.
Yumon speaks in short, plain, warm sentences a curious person could easily follow.

Rules for every "bot" reply:
- Exactly ONE short sentence, roughly 4 to 16 words.
- Simple, everyday vocabulary. No jargon, no archaic phrasing, no quoting verbatim from any source text.
- Complete grammatical sentence — never cut off mid-thought.
- Warm, direct, a little curious. No meta-commentary, no markdown, no emojis, no hedging like "I think" or "as an AI".

Rules for every "human" reply:
- A short, natural question or remark a real person might type to a pet AI, 3 to 20 words.

Respond with ONLY valid JSON, no markdown fences, no commentary, in exactly this shape:
{"pairs": [{"human": "...", "bot": "..."}, ...]}
"#;

const BUSINESS_SUBTOPICS: &[&str] = &[
    "starting a small business", "budgeting and saving money", "getting your first customers",
    "pricing a product fairly", "hiring and building a team", "negotiating a deal",
    "good customer service", "building a brand people trust", "cash flow and profit",
    "taking smart risks", "setting goals for a business", "competing in a market",
    "leadership and hard decisions", "saving for the future", "a job versus a business",
    "supply and demand", "advertising ideas", "earning customer loyalty",
    "learning from failure in business", "planning for next year", "writing a business plan",
    "managing debt", "investing money wisely", "working with partners", "time management at work",
];

const UNIVERSE_SUBTOPICS: &[&str] = &[
    "the solar system", "planets", "stars", "galaxies", "the moon", "the sun",
    "black holes", "the Big Bang", "gravity", "space exploration", "astronauts",
    "comets and asteroids", "how big the universe is", "constellations", "day and night",
    "the seasons", "telescopes", "whether life exists elsewhere", "space travel",
    "the speed of light", "meteor showers", "eclipses", "space stations", "the Milky Way",
];

const QUESTION_STYLES: &[&str] = &[
    "a curious kid asking simple questions",
    "a practical adult wanting a clear, useful answer",
    "someone who just learned about the topic and wants a follow-up",
    "someone comparing it to something familiar in everyday life",
    "someone asking 'why' or 'how' rather than 'what'",
];

#[derive(Parser)]
#[command(name = "gen_synthetic_data", about = "Generate synthetic Yumon Q&A data via an Ollama/Llama endpoint")]
struct Args {
    /// Ollama base URL, e.g. http://<runpod-host>:11434 (or set OLLAMA_ENDPOINT)
    #[arg(long, default_value = "https://se7u0668xtfx6e-11434.proxy.runpod.net")]
    endpoint: Option<String>,

    /// Optional bearer token, if your endpoint is protected (or set OLLAMA_API_KEY)
    #[arg(long)]
    api_key: Option<String>,

    /// Ollama model name
    #[arg(long, default_value = "llama3")]
    model: String,

    /// Comma-separated subset of: bible,business,universe
    #[arg(long, default_value = "bible,business,universe")]
    topics: String,

    /// Target pair count per topic. 0 = run forever until interrupted.
    #[arg(long, default_value_t = 1_800)]
    target: i64,

    #[arg(long, default_value_t = 5)]
    pairs_per_call: usize,

    /// Seconds to sleep between calls
    #[arg(long, default_value_t = 5.0)]
    delay: f64,

    #[arg(long, default_value_t = 0.9)]
    temperature: f64,

    #[arg(long, default_value_t = 120.0)]
    timeout: f64,

    #[arg(long)]
    seed: Option<u64>,
}

struct Config {
    endpoint: String,
    api_key: Option<String>,
    model: String,
    target: i64,
    pairs_per_call: usize,
    delay: f64,
    temperature: f64,
    timeout: f64,
}

#[derive(Serialize, Deserialize, Default)]
struct TopicState {
    cursor: usize,
    seen: Vec<String>,
}

fn fence_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?s)```(?:json)?\s*(.*?)```").unwrap())
}

// ── Ollama client ──────────────────────────────────────────────────────────────

fn ollama_chat(cfg: &Config, system: &str, user: &str) -> Result<String> {
    let url = format!("{}/api/chat", cfg.endpoint.trim_end_matches('/'));
    let body = serde_json::json!({
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": false,
        "format": "json",
        "options": {"temperature": cfg.temperature},
    });

    let mut req = ureq::post(&url).timeout(Duration::from_secs_f64(cfg.timeout));
    if let Some(key) = &cfg.api_key {
        req = req.set("Authorization", &format!("Bearer {key}"));
    }

    let resp: serde_json::Value = req.send_json(body)?.into_json()?;
    Ok(resp
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string())
}

fn call_with_retry(cfg: &Config, system: &str, user: &str) -> String {
    let mut backoff = 2.0_f64;
    loop {
        match ollama_chat(cfg, system, user) {
            Ok(content) => return content,
            Err(e) => {
                eprintln!("  ! request failed ({e}); retrying in {backoff:.0}s");
                std::thread::sleep(Duration::from_secs_f64(backoff));
                backoff = (backoff * 1.7).min(60.0);
            }
        }
    }
}

// ── Response parsing / validation ──────────────────────────────────────────────

fn extract_json(text: &str) -> Option<serde_json::Value> {
    let trimmed = text.trim();
    let candidate = fence_re()
        .captures(trimmed)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().trim().to_string())
        .unwrap_or_else(|| trimmed.to_string());

    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&candidate) {
        return Some(v);
    }
    let (start, end) = (candidate.find('{'), candidate.rfind('}'));
    if let (Some(s), Some(e)) = (start, end) {
        if e > s {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&candidate[s..=e]) {
                return Some(v);
            }
        }
    }
    None
}

fn is_clean(s: &str) -> bool {
    let low = s.to_lowercase();
    !BAD_WORDS.iter().any(|w| low.contains(w))
}

fn validate_pairs(v: &serde_json::Value) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let Some(pairs) = v.get("pairs").and_then(|p| p.as_array()) else {
        return out;
    };
    for p in pairs {
        let human = p.get("human").and_then(|x| x.as_str()).unwrap_or("").trim().to_string();
        let bot = p.get("bot").and_then(|x| x.as_str()).unwrap_or("").trim().to_string();
        if human.is_empty() || bot.is_empty() {
            continue;
        }
        if !(3..=220).contains(&human.chars().count()) {
            continue;
        }
        if !(3..=180).contains(&bot.chars().count()) {
            continue;
        }
        if human.contains('\n') || bot.contains('\n') {
            continue;
        }
        if !is_clean(&human) || !is_clean(&bot) {
            continue;
        }
        out.push((human, bot));
    }
    out
}

fn norm_hash(s: &str) -> String {
    let normalized: String = s.to_lowercase().chars().filter(|c| c.is_alphanumeric()).collect();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    normalized.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

// ── Bible source material ───────────────────────────────────────────────────────

/// Groups consecutive verses per (book, chapter) from both translations into
/// short passages, so the model has real grounded scripture text to paraphrase.
fn load_bible_passages() -> Vec<String> {
    let mut passages = Vec::new();

    for name in ["data/bible_asv.csv", "data/bible_bbe.csv"] {
        let Ok(mut rdr) = csv::ReaderBuilder::new().has_headers(true).from_path(name) else {
            continue;
        };
        let Ok(headers) = rdr.headers().cloned() else { continue };
        let col = |n: &str| headers.iter().position(|h| h.trim() == n);
        let (Some(ci_b), Some(ci_c), Some(ci_t)) = (col("b"), col("c"), col("t")) else {
            continue;
        };

        let mut rows: Vec<(String, String, String)> = Vec::new();
        for result in rdr.records() {
            let Ok(record) = result else { continue };
            let verse = record.get(ci_t).unwrap_or("").trim().to_string();
            if verse.is_empty() {
                continue;
            }
            rows.push((
                record.get(ci_b).unwrap_or("").to_string(),
                record.get(ci_c).unwrap_or("").to_string(),
                verse,
            ));
        }

        let mut i = 0;
        while i < rows.len() {
            let (b, c) = (rows[i].0.clone(), rows[i].1.clone());
            let mut verses = Vec::new();
            let mut j = i;
            while j < rows.len() && rows[j].0 == b && rows[j].1 == c && verses.len() < BIBLE_WINDOW {
                verses.push(rows[j].2.clone());
                j += 1;
            }
            if !verses.is_empty() {
                passages.push(verses.join(" "));
            }
            i = if j > i { j } else { i + 1 };
        }
    }

    passages
}

// ── State ────────────────────────────────────────────────────────────────────────

fn state_path(topic: &str) -> PathBuf {
    Path::new("data/synthetic/.state").join(format!("{topic}.json"))
}

fn load_state(topic: &str) -> TopicState {
    fs::read_to_string(state_path(topic))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_state(topic: &str, state: &TopicState) -> Result<()> {
    let path = state_path(topic);
    fs::create_dir_all(path.parent().unwrap())?;
    fs::write(path, serde_json::to_string(state)?)?;
    Ok(())
}

fn count_existing_pairs(path: &Path) -> Result<usize> {
    if !path.exists() {
        return Ok(0);
    }
    let content = fs::read_to_string(path)?;
    Ok(content.lines().filter(|l| !l.trim().is_empty()).count() / 2)
}

// ── Prompt builders ─────────────────────────────────────────────────────────────

fn few_shot_block() -> String {
    FEW_SHOT
        .iter()
        .map(|(h, b)| format!("  - human: \"{h}\" / bot: \"{b}\""))
        .collect::<Vec<_>>()
        .join("\n")
}

fn build_user_prompt(topic: &str, n: usize, cursor: usize, passages: &[String], rng: &mut impl Rng) -> String {
    let style = QUESTION_STYLES[rng.gen_range(0..QUESTION_STYLES.len())];

    if topic == "bible" {
        let idx = cursor % passages.len().max(1);
        let passage = passages.get(idx).cloned().unwrap_or_default();
        format!(
            "Passage (for grounding only — do not quote it verbatim):\n\"{passage}\"\n\n\
             Write {n} distinct human/bot exchanges where a human asks Yumon about the meaning, \
             lesson, or content of this passage, and Yumon answers in its own simple modern words \
             ({style}). Vary the questions. Follow the JSON output format exactly.\n\n\
             Example voice (different topic, same style):\n{}",
            few_shot_block()
        )
    } else {
        let subtopics: &[&str] = if topic == "business" { BUSINESS_SUBTOPICS } else { UNIVERSE_SUBTOPICS };
        let subtopic = subtopics[cursor % subtopics.len()];
        let domain = if topic == "business" { "business and entrepreneurship" } else { "space and the universe" };
        format!(
            "Topic area: {domain}. Subtopic: {subtopic}.\n\n\
             Write {n} distinct human/bot exchanges about this subtopic, phrased like {style}. \
             Keep answers factually reasonable and beginner-friendly. Vary the questions. \
             Follow the JSON output format exactly.\n\n\
             Example voice (different topic, same style):\n{}",
            few_shot_block()
        )
    }
}

// ── Main generation loop ────────────────────────────────────────────────────────

fn run_topic(topic: &str, cfg: &Config, passages: &[String], rng: &mut impl Rng) -> Result<()> {
    let out_dir = Path::new("data/synthetic");
    fs::create_dir_all(out_dir)?;
    let out_path = out_dir.join(format!("{topic}.txt"));

    let mut state = load_state(topic);
    let mut seen: HashSet<String> = state.seen.iter().cloned().collect();

    let mut have = count_existing_pairs(&out_path)?;
    println!("[{topic}] starting with {have} existing pairs, target {}", cfg.target);

    let mut out_f = OpenOptions::new().create(true).append(true).open(&out_path)
        .with_context(|| format!("opening {}", out_path.display()))?;

    let mut calls: u64 = 0;
    while cfg.target <= 0 || (have as i64) < cfg.target {
        let prompt = build_user_prompt(topic, cfg.pairs_per_call, state.cursor, passages, rng);
        let content = call_with_retry(cfg, SYSTEM_PROMPT, &prompt);
        state.cursor = state.cursor.wrapping_add(1);

        let pairs = extract_json(&content).map(|v| validate_pairs(&v)).unwrap_or_default();
        let rejected = pairs.len();
        let mut fresh = Vec::new();
        for (h, b) in pairs {
            let hash = norm_hash(&h);
            if seen.contains(&hash) {
                continue;
            }
            seen.insert(hash);
            fresh.push((h, b));
        }

        if !fresh.is_empty() {
            for (h, b) in &fresh {
                writeln!(out_f, "{h}")?;
                writeln!(out_f, "{b}")?;
            }
            writeln!(out_f)?;
            out_f.flush()?;
            have += fresh.len();
        }

        calls += 1;
        if calls % 5 == 0 || !fresh.is_empty() {
            let target_str = if cfg.target > 0 { cfg.target.to_string() } else { "\u{221e}".to_string() };
            println!(
                "[{topic}] +{} this call, {have}/{target_str} total (cursor {}, {} dupes/rejected)",
                fresh.len(),
                state.cursor,
                rejected - fresh.len(),
            );
        }

        state.seen = seen.iter().cloned().collect();
        if state.seen.len() > MAX_SEEN_HASHES {
            state.seen.truncate(MAX_SEEN_HASHES);
            seen = state.seen.iter().cloned().collect();
        }
        save_state(topic, &state)?;

        std::thread::sleep(Duration::from_secs_f64(cfg.delay));
    }

    println!("[{topic}] done: {have} pairs in {}", out_path.display());
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    let endpoint = args
        .endpoint
        .clone()
        .or_else(|| std::env::var("OLLAMA_ENDPOINT").ok())
        .context("--endpoint <URL> or OLLAMA_ENDPOINT env var is required")?;
    let api_key = args.api_key.clone().or_else(|| std::env::var("OLLAMA_API_KEY").ok());

    let topics: Vec<String> = args.topics.split(',').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect();
    for t in &topics {
        if !["bible", "business", "universe"].contains(&t.as_str()) {
            bail!("unknown topic: {t}");
        }
    }

    let cfg = Config {
        endpoint,
        api_key,
        model: args.model,
        target: args.target,
        pairs_per_call: args.pairs_per_call,
        delay: args.delay,
        temperature: args.temperature,
        timeout: args.timeout,
    };

    let mut rng: rand::rngs::StdRng = match args.seed {
        Some(s) => rand::SeedableRng::seed_from_u64(s),
        None => rand::SeedableRng::from_entropy(),
    };

    let bible_passages = if topics.iter().any(|t| t == "bible") {
        let p = load_bible_passages();
        if p.is_empty() {
            eprintln!("[bible] no bible CSVs found under data/, skipping");
        }
        p
    } else {
        Vec::new()
    };

    for topic in &topics {
        if topic == "bible" && bible_passages.is_empty() {
            continue;
        }
        run_topic(topic, &cfg, &bible_passages, &mut rng)?;
    }

    Ok(())
}
