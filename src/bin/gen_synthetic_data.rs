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
// bible_asv.csv / bible_bbe.csv are loaded as consecutive verse pairs
// (FileKind::BibleCsv, see load_csv_bible_pairs) — verse N as input, verse
// N+1 as target. This tool instead asks an LLM to produce real, topically
// grounded question/answer turns in Yumon's short voice, for pairs with
// actual conversational structure rather than adjacent scripture.
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
const DEFAULT_ENDPOINT: &str = "https://w5od4nwjt1exl7-11434.proxy.runpod.net";
const SUPPORTED_TOPICS: &[&str] = &[
    "bible",
    "business",
    "universe",
    "world_basics",
    "daily_life",
    "social_life",
    "puzzles",
    "commands",
    "memory",
];

const TASK_SYSTEM_PROMPT: &str = r#"You generate training data for Yumon, a small tabletop pet AI.
Generate actual solvable tasks and correct answers, not explanations of task types.
Use plain everyday vocabulary. Each human and bot field must be one line.
Human prompts may contain several sentences of facts, constraints, and a question.
Bot replies should be concise. Exact-output instructions override conversational style:
single words, numbers, comma-separated lists, and multiple sentences are allowed.
For world commands, state an intention or ask for clarification; never claim an action
has already happened unless the supplied world state explicitly says so.
Solve each task privately and check the answer against every supplied fact and constraint.
Include all necessary facts; do not rely on unstated assumptions or outside knowledge.
No markdown fences, commentary, emojis, or reasoning traces. Return only the requested JSON.
"#;

const PUZZLE_SUBTOPICS: &[&str] = &[
    // Original entries
    "track an object's location through two to four explicit moves",
    "infer first or last place from an unambiguous ordering of three to five people",
    "count objects after small additions and removals",
    "follow left/right or above/below relations with an explicit orientation",
    "choose the only object satisfying two or three stated properties",
    "apply a short invented rule to a new example, with no outside knowledge",

    // Spatial & tracking additions
    "determine an object's final orientation after two to three 90-degree rotations",
    "identify whether an item is inside, outside, or adjacent in a simple nested container setup",
    "trace a 2D grid path given a start coordinate and three to five directional steps",
    "find which seat is empty after three to five people swap places in a fixed row",
    "determine relative depth or height after stacking and unstacking two to four labeled blocks",
    "identify the facing direction after a sequence of turn-left and turn-right commands",

    // Logic, deduction & elimination additions
    "deduce an item's owner through elimination from a three-by-three matching clue set",
    "identify the single liar or truth-teller from three short conflicting statements",
    "determine set membership from an explicit description of overlapping categories",
    "find the missing element in a simple alternating or repeating pattern of three to five items",
    "select the only assignment consistent with three pairwise constraints",
    "infer which of three doors is safe from two explicit true-or-false labels",
    "complete a two-attribute grid where each row and column has unique values",

    // State & resource tracking additions
    "balance an exchange where three items trade at explicit fixed ratios",
    "track capacity or volume across two to three containers after basic pour steps",
    "calculate the time elapsed or final time across two sequential scheduled events",
    "track a running score after three to five explicit gains and losses",
    "determine remaining inventory after two purchases and one return at stated prices",
    "compute the final count when items are grouped, split, or combined by a stated rule",

    // Comparison & quantity additions
    "identify the heavier or lighter item from two direct comparison outcomes",
    "choose the largest or smallest value after applying one stated transformation to each option",
    "determine whether a total meets, exceeds, or falls short of an explicit threshold",
];

const COMMAND_SUBTOPICS: &[&str] = &[
    // Original entries
    "copy or select supplied words in an exact requested format",
    "sort a supplied list or reverse its order, returning only the requested list",
    "follow a conditional instruction using an explicitly supplied state",
    "plan two or three actions in order while obeying a stated constraint",
    "ask which object is intended when a command matches multiple objects",
    "explain a blocked action briefly and request the missing item or information",

    // Extraction & formatting additions
    "extract only items matching a specific criteria and format them as a delimited list",
    "replace designated placeholder tokens in a template using a provided key-value map",
    "strip unwanted characters or prefixes while strictly preserving all other text verbatim",
    "return only the nth field from each line of a supplied table",
    "reformat supplied records into a fixed template with unchanged field values",
    "deduplicate a supplied list while preserving the first occurrence of each item",

    // Control flow & state manipulation additions
    "execute a repeated action a fixed number of times until a target count is reached",
    "switch to an alternative fallback action when the primary condition evaluates to false",
    "update an in-memory key-value state by applying an add, modify, or delete command",
    "apply a sequence of named operations to a starting value and return only the final result",
    "skip an item when it matches an exclusion rule and continue with the rest",
    "merge two supplied maps, with an explicit rule for conflicting keys",

    // Safety, error handling & clarification additions
    "refuse an action that explicitly violates a stated safety rule and state the rule",
    "halt execution early when a required precondition is explicitly unmet",
    "confirm execution parameters before running an action marked as irreversible",
    "report which required field is missing instead of inventing a value",
    "ask a single clarifying question when two interpretations of the command are equally licensed",
];

const FEW_SHOT: [(&str, &str); 3] = [
    (
        "What is the universe?",
        "The universe is a wide open space filled with planets.",
    ),
    (
        "What does it mean to be wide?",
        "Something wide is big and hard to traverse.",
    ),
    (
        "What is a feeling?",
        "A feeling is an emotion like happiness or sadness.",
    ),
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
    "starting a small business",
    "budgeting and saving money",
    "getting your first customers",
    "pricing a product fairly",
    "hiring and building a team",
    "negotiating a deal",
    "good customer service",
    "building a brand people trust",
    "cash flow and profit",
    "taking smart risks",
    "setting goals for a business",
    "competing in a market",
    "leadership and hard decisions",
    "saving for the future",
    "a job versus a business",
    "supply and demand",
    "advertising ideas",
    "earning customer loyalty",
    "learning from failure in business",
    "planning for next year",
    "writing a business plan",
    "managing debt",
    "investing money wisely",
    "working with partners",
    "time management at work",
];

const UNIVERSE_SUBTOPICS: &[&str] = &[
    "the solar system",
    "planets",
    "stars",
    "galaxies",
    "the moon",
    "the sun",
    "black holes",
    "the Big Bang",
    "gravity",
    "space exploration",
    "astronauts",
    "comets and asteroids",
    "how big the universe is",
    "constellations",
    "day and night",
    "the seasons",
    "telescopes",
    "whether life exists elsewhere",
    "space travel",
    "the speed of light",
    "meteor showers",
    "eclipses",
    "space stations",
    "the Milky Way",
];

// Concrete concepts Yumon can use when reasoning about its simulated world.
// These remain ordinary short Q&A pairs: they teach vocabulary and basic
// relationships without changing the conversational training format.
const WORLD_BASICS_SUBTOPICS: &[&str] = &[
    "left, right, above, and below",
    "near and far",
    "inside, outside, and between",
    "containers and what fits inside them",
    "open and closed things",
    "moving objects",
    "where an object is after it moves",
    "finding a lost object",
    "giving and taking objects",
    "full and empty containers",
    "big and small objects",
    "heavy and light objects",
    "stacking and arranging objects",
    "doors, paths, and obstacles",
    "rooms and places",
    "simple maps and directions",
    "what can be seen from a place",
    "cause and effect in everyday actions",
    "objects staying where they were left",
    "using an object for its purpose",
];

const DAILY_LIFE_SUBTOPICS: &[&str] = &[
    "morning and bedtime routines",
    "being hungry and eating",
    "being thirsty and drinking",
    "resting when tired",
    "staying warm or cool",
    "keeping a home tidy",
    "caring for belongings",
    "choosing what to do next",
    "waiting patiently",
    "planning a simple day",
    "sharing a meal",
    "getting ready to go outside",
    "weather changing a daily plan",
    "making a cozy space",
    "noticing time passing",
    "finishing a small task",
    "asking for help with a task",
    "taking breaks",
    "doing something again tomorrow",
    "small habits that help someone feel well",
];

const SOCIAL_LIFE_SUBTOPICS: &[&str] = &[
    "greeting someone",
    "making a new friend",
    "taking turns",
    "sharing something fairly",
    "helping someone",
    "asking for help",
    "saying thank you",
    "apologizing after a mistake",
    "keeping a promise",
    "working together",
    "listening before answering",
    "respecting personal space",
    "asking before borrowing something",
    "handling a disagreement calmly",
    "including someone in an activity",
    "cheering someone up",
    "being honest kindly",
    "saying no politely",
    "trust growing over time",
    "celebrating another person's success",
];

const QUESTION_STYLES: &[&str] = &[
    "a curious kid asking simple questions",
    "a practical adult wanting a clear, useful answer",
    "someone who just learned about the topic and wants a follow-up",
    "someone comparing it to something familiar in everyday life",
    "someone asking 'why' or 'how' rather than 'what'",
];

#[derive(Parser)]
#[command(
    name = "gen_synthetic_data",
    about = "Generate synthetic Yumon Q&A data via an Ollama/Llama endpoint"
)]
struct Args {
    /// Ollama base URL, e.g. http://<runpod-host>:11434 (or set OLLAMA_ENDPOINT)
    #[arg(long)]
    endpoint: Option<String>,

    /// Optional bearer token, if your endpoint is protected (or set OLLAMA_API_KEY)
    #[arg(long)]
    api_key: Option<String>,

    /// Ollama model name
    #[arg(long, default_value = "llama3")]
    model: String,

    /// Comma-separated subset of: bible,business,universe,world_basics,daily_life,social_life,puzzles,commands,memory
    #[arg(long, default_value = "puzzles,commands,memory")]
    topics: String,

    /// Target pair count per topic. Memory keeps complete conversations and may exceed this by up to 4 turns. 0 = run forever.
    #[arg(long, default_value_t = 1_800)]
    target: i64,

    #[arg(long, default_value_t = 5)]
    pairs_per_call: usize,

    /// Seconds to sleep between calls
    #[arg(long, default_value_t = 1.0)]
    delay: f64,

    #[arg(long, default_value_t = 0.9)]
    temperature: f64,

    #[arg(long, default_value_t = 120.0)]
    timeout: f64,

    #[arg(long)]
    seed: Option<u64>,

    /// Print one generation prompt per topic without contacting the endpoint or writing data.
    #[arg(long)]
    preview: bool,
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

fn output_schema(topic: &str) -> serde_json::Value {
    let pair = serde_json::json!({
        "type": "object", "required": ["human", "bot"], "additionalProperties": false,
        "properties": {
            "human": {"type": "string"}, "bot": {"type": "string"}
        }
    });
    if topic == "memory" {
        serde_json::json!({
            "type": "object", "required": ["conversations"], "additionalProperties": false,
            "properties": {"conversations": {"type": "array", "items": {
                "type": "array", "minItems": 3, "maxItems": 5, "items": pair
            }}}
        })
    } else {
        serde_json::json!({
            "type": "object", "required": ["pairs"], "additionalProperties": false,
            "properties": {"pairs": {"type": "array", "items": pair}}
        })
    }
}

fn ollama_chat(cfg: &Config, topic: &str, system: &str, user: &str) -> Result<String> {
    let url = format!("{}/api/chat", cfg.endpoint.trim_end_matches('/'));
    let body = serde_json::json!({
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": false,
        "format": output_schema(topic),
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

fn call_with_retry(cfg: &Config, topic: &str, system: &str, user: &str) -> Result<String> {
    let mut backoff = 2.0_f64;
    for attempt in 0..5 {
        match ollama_chat(cfg, topic, system, user) {
            Ok(content) => return Ok(content),
            Err(e) => {
                if attempt == 4 {
                    return Err(e).context("Ollama request failed after five attempts");
                }
                eprintln!("  ! request failed ({e}); retrying in {backoff:.0}s");
                std::thread::sleep(Duration::from_secs_f64(backoff));
                backoff = (backoff * 1.7).min(60.0);
            }
        }
    }
    unreachable!()
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

fn validate_pairs(v: &serde_json::Value, task: bool) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let Some(pairs) = v.get("pairs").and_then(|p| p.as_array()) else {
        return out;
    };
    for p in pairs {
        let human = p
            .get("human")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let bot = p
            .get("bot")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if human.is_empty() || bot.is_empty() {
            continue;
        }
        if !(3..=if task { 1400 } else { 220 }).contains(&human.chars().count()) {
            continue;
        }
        if !(if task { 1 } else { 3 }..=if task { 600 } else { 180 }).contains(&bot.chars().count())
        {
            continue;
        }
        if human.contains(['\n', '\r']) || bot.contains(['\n', '\r']) {
            continue;
        }
        if !is_clean(&human) || !is_clean(&bot) {
            continue;
        }
        out.push((human, bot));
    }
    out
}

type Chat = Vec<(String, String)>;

fn validate_chats(v: &serde_json::Value, topic: &str) -> Vec<Chat> {
    if topic != "memory" {
        return validate_pairs(v, matches!(topic, "puzzles" | "commands"))
            .into_iter()
            .map(|pair| vec![pair])
            .collect();
    }
    let Some(conversations) = v.get("conversations").and_then(|c| c.as_array()) else {
        return Vec::new();
    };
    conversations
        .iter()
        .filter_map(|c| {
            let turns = c.as_array()?;
            if !(3..=5).contains(&turns.len()) {
                return None;
            }
            let chat = validate_pairs(&serde_json::json!({"pairs": turns}), true);
            // Reject the whole conversation rather than removing an invalid fact-bearing turn.
            (chat.len() == turns.len()).then_some(chat)
        })
        .collect()
}

fn write_chats(out: &mut impl Write, chats: &[Chat]) -> Result<()> {
    for chat in chats {
        for (human, bot) in chat {
            writeln!(out, "{human}\n{bot}")?;
        }
        writeln!(out)?;
    }
    out.flush()?;
    Ok(())
}

fn norm_hash(s: &str) -> String {
    let normalized: String = s
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect();
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
        let Ok(headers) = rdr.headers().cloned() else {
            continue;
        };
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
            while j < rows.len() && rows[j].0 == b && rows[j].1 == c && verses.len() < BIBLE_WINDOW
            {
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

fn build_user_prompt(
    topic: &str,
    n: usize,
    cursor: usize,
    passages: &[String],
    rng: &mut impl Rng,
) -> String {
    if topic == "memory" {
        return format!(
            "Generate {n} independent conversations of 3 to 5 human/bot exchanges each. \
             Establish a named object's location, a preference, or an explicit plan in an early turn. \
             Include a related follow-up or a distraction, then an explicit correction or state update. \
             The final human question must require earlier conversation history, and the final bot \
             answer must use the latest applicable fact, not the superseded one. Vary names, objects, \
             locations, and wording. Keep each conversation logically consistent and isolated. \
             Return {{\"conversations\": [[{{\"human\": \"...\", \"bot\": \"...\"}}, ...], ...]}}."
        );
    }
    let task_subtopics = match topic {
        "puzzles" => Some(PUZZLE_SUBTOPICS),
        "commands" => Some(COMMAND_SUBTOPICS),
        _ => None,
    };
    if let Some(subtopics) = task_subtopics {
        let subtopic = subtopics[cursor % subtopics.len()];
        return format!(
            "Generate {n} independent {topic} human/bot exchanges. Task: {subtopic}. \
             Each human field must present the actual task and all needed facts. \
             Each bot field must directly solve it or follow its instructions exactly. \
             Vary the concrete facts, names, quantities, and phrasing. Use one to three reasoning \
             steps, include distracting facts in some examples, and keep answers unambiguous. \
             Do not ask what a puzzle or command means. Check every answer privately. \
             Return {{\"pairs\": [{{\"human\": \"...\", \"bot\": \"...\"}}, ...]}}."
        );
    }
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
        let (domain, subtopics): (&str, &[&str]) = match topic {
            "business" => ("business and entrepreneurship", BUSINESS_SUBTOPICS),
            "universe" => ("space and the universe", UNIVERSE_SUBTOPICS),
            "world_basics" => (
                "everyday spatial reasoning and physical relationships",
                WORLD_BASICS_SUBTOPICS,
            ),
            "daily_life" => (
                "daily life, routines, and practical self-care",
                DAILY_LIFE_SUBTOPICS,
            ),
            "social_life" => (
                "friendship, cooperation, and respectful social behavior",
                SOCIAL_LIFE_SUBTOPICS,
            ),
            _ => unreachable!("topics are validated before prompts are built"),
        };
        let subtopic = subtopics[cursor % subtopics.len()];
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
    println!(
        "[{topic}] starting with {have} existing pairs, target {}",
        cfg.target
    );

    let mut out_f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out_path)
        .with_context(|| format!("opening {}", out_path.display()))?;

    let mut calls: u64 = 0;
    let mut empty_calls = 0;
    while cfg.target <= 0 || (have as i64) < cfg.target {
        let n = if cfg.target > 0 {
            cfg.pairs_per_call
                .min((cfg.target as usize).saturating_sub(have))
        } else {
            cfg.pairs_per_call
        };
        let prompt = build_user_prompt(topic, n, state.cursor, passages, rng);
        let system = if matches!(topic, "puzzles" | "commands" | "memory") {
            TASK_SYSTEM_PROMPT
        } else {
            SYSTEM_PROMPT
        };
        let content = call_with_retry(cfg, topic, system, &prompt)?;
        state.cursor = state.cursor.wrapping_add(1);

        let chats = extract_json(&content)
            .map(|v| validate_chats(&v, topic))
            .unwrap_or_default();
        let accepted = chats.len();
        let mut fresh = Vec::new();
        let mut fresh_pairs = 0;
        for chat in chats.into_iter().take(n) {
            let hash = if chat.len() == 1 {
                norm_hash(&chat[0].0)
            } else {
                norm_hash(&serde_json::to_string(&chat)?)
            };
            if seen.contains(&hash) {
                continue;
            }
            seen.insert(hash);
            fresh_pairs += chat.len();
            fresh.push(chat);
            if cfg.target > 0 && have + fresh_pairs >= cfg.target as usize {
                break;
            }
        }

        if !fresh.is_empty() {
            write_chats(&mut out_f, &fresh)?;
            have += fresh_pairs;
            empty_calls = 0;
        } else {
            empty_calls += 1;
            anyhow::ensure!(
                empty_calls < 10,
                "[{topic}] ten calls produced no fresh valid data; check model output and dedup state"
            );
        }

        calls += 1;
        if calls % 5 == 0 || !fresh.is_empty() {
            let target_str = if cfg.target > 0 {
                cfg.target.to_string()
            } else {
                "\u{221e}".to_string()
            };
            println!(
                "[{topic}] +{} pairs this call, {have}/{target_str} total (cursor {}, {} accepted chats, {} written)",
                fresh_pairs,
                state.cursor,
                accepted,
                fresh.len(),
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
    anyhow::ensure!(args.target >= 0, "--target must be nonnegative");
    anyhow::ensure!(args.pairs_per_call > 0, "--pairs-per-call must be positive");
    anyhow::ensure!(
        args.delay.is_finite() && args.delay >= 0.0,
        "--delay must be finite and nonnegative"
    );
    anyhow::ensure!(
        args.timeout.is_finite() && args.timeout > 0.0,
        "--timeout must be finite and positive"
    );
    anyhow::ensure!(
        args.temperature.is_finite() && args.temperature >= 0.0,
        "--temperature must be finite and nonnegative"
    );

    let endpoint = args
        .endpoint
        .clone()
        .or_else(|| std::env::var("OLLAMA_ENDPOINT").ok())
        .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string());
    let api_key = args
        .api_key
        .clone()
        .or_else(|| std::env::var("OLLAMA_API_KEY").ok());

    let topics: Vec<String> = args
        .topics
        .split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();
    for t in &topics {
        if !SUPPORTED_TOPICS.contains(&t.as_str()) {
            bail!("unknown topic: {t}");
        }
    }
    anyhow::ensure!(!topics.is_empty(), "select at least one topic");

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
        if args.preview {
            let prompt = build_user_prompt(topic, cfg.pairs_per_call, 0, &bible_passages, &mut rng);
            println!("[{topic}]\n{prompt}");
            continue;
        }
        run_topic(topic, &cfg, &bible_passages, &mut rng)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;

    #[test]
    fn task_validation_preserves_exact_short_answers_and_long_premises() {
        let v = serde_json::json!({"pairs": [
            {"human": format!("{} How many?", "A marble stays in the box. ".repeat(10)), "bot": "2"},
            {"human": "Say only yes.", "bot": "yes\rhidden turn"}
        ]});
        let chats = validate_chats(&v, "puzzles");
        assert_eq!(chats.len(), 1);
        assert_eq!(chats[0][0].1, "2");
        assert!(validate_chats(&v, "world_basics").is_empty());
    }

    #[test]
    fn independent_pairs_have_separate_blocks_and_memory_keeps_related_turns() {
        let turns = serde_json::json!([
            {"human": "The key is on the shelf.", "bot": "I will remember that."},
            {"human": "I moved the key to the box.", "bot": "The key is now in the box."},
            {"human": "Where is the key?", "bot": "The key is in the box."}
        ]);
        let independent = validate_chats(&serde_json::json!({"pairs": turns}), "puzzles");
        let memory = validate_chats(&serde_json::json!({"conversations": [turns]}), "memory");
        let mut output = Vec::new();
        write_chats(&mut output, &independent).unwrap();
        let independent_text = String::from_utf8(output).unwrap();
        assert_eq!(independent_text.trim().split("\n\n").count(), 3);
        let mut output = Vec::new();
        write_chats(&mut output, &memory).unwrap();
        let memory_text = String::from_utf8(output).unwrap();
        assert_eq!(memory_text.trim().split("\n\n").count(), 1);
        assert_eq!(memory_text.lines().filter(|l| !l.is_empty()).count(), 6);

        // Exercise the real training parser: independent examples must not become memories.
        let path =
            std::env::temp_dir().join(format!("yumon-synthetic-{}.txt", uuid::Uuid::new_v4()));
        std::fs::write(&path, &independent_text).unwrap();
        let parsed = yumon_pet::brain::mdx::load_handcrafted_chats(path.to_str().unwrap()).unwrap();
        assert_eq!(parsed.blocks.len(), 3);
        assert!(parsed.blocks.iter().all(|b| b.memories.len() == 1));
        std::fs::write(&path, &memory_text).unwrap();
        let parsed = yumon_pet::brain::mdx::load_handcrafted_chats(path.to_str().unwrap()).unwrap();
        assert_eq!(parsed.blocks.len(), 1);
        assert_eq!(parsed.blocks[0].memories.len(), 3);
        assert_eq!(parsed.blocks[0].memories[2].bot, "The key is in the box.");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn invalid_memory_turn_rejects_entire_conversation() {
        let v = serde_json::json!({"conversations": [[
            {"human": "The key is on the shelf.", "bot": "Okay."},
            {"human": "I moved the key to the box.", "bot": ""},
            {"human": "Where is the key?", "bot": "The box."}
        ]]});
        assert!(validate_chats(&v, "memory").is_empty());
    }

    #[test]
    fn task_prompts_rotate_and_memory_requires_updated_history() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for topic in ["puzzles", "commands"] {
            assert!(SUPPORTED_TOPICS.contains(&topic));
            assert_ne!(
                build_user_prompt(topic, 5, 0, &[], &mut rng),
                build_user_prompt(topic, 5, 1, &[], &mut rng)
            );
            assert!(output_schema(topic)["properties"]["pairs"].is_object());
        }
        let prompt = build_user_prompt("memory", 5, 0, &[], &mut rng);
        assert!(prompt.contains("latest applicable fact"));
        assert!(output_schema("memory")["properties"]["conversations"].is_object());
    }

    #[test]
    fn simulation_topics_are_supported_and_have_distinct_domains() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let cases = [
            ("world_basics", "everyday spatial reasoning"),
            ("daily_life", "daily life, routines"),
            ("social_life", "friendship, cooperation"),
        ];

        for (topic, domain) in cases {
            assert!(SUPPORTED_TOPICS.contains(&topic));
            let prompt = build_user_prompt(topic, 5, 0, &[], &mut rng);
            assert!(prompt.contains(domain));
            assert!(prompt.contains("Write 5 distinct human/bot exchanges"));
        }
    }
}
