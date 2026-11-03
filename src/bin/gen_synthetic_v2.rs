//! Unified Synthetic Natural-Language Data Generator for Yumon v0.2
//!
//! Generates natural language training samples for all Yumon v0.2 domains:
//! - CNM (Central Nervous Model): Sensory routing, vitals monitoring, and 3D grounded emotes
//! - Behavior Model: Emergent physical/play/care actions across hundreds of variations + speech triggers
//! - Memory Model: Event salience filtering (3A), session summarization (3B), and hierarchical traits (3C)
//! - Reasoning Model: Bounded micro-plans (≤ 4 steps) and blocked feasibility evaluations
//! - Conversation Model: Grounded short pet utterances in natural English (4-16 words)
//!
//! Ground rules:
//! 1. Natural Language Only — No Model JSON in model outputs.
//! 2. Deterministic validation using strict Rust parsing rules.
//! 3. Rejects profanity and length/structural violations.
//! 4. Full natural vocabulary: NO artificial Kilo-Talk word filtering during generation.
//!    (Translation/tokenization into Kilo-Talk occurs during the downstream caching step).
//! 5. Continuous resume & SHA-256 deduplication via `.state/<domain>.json`.
//! 6. Deterministic 90/10 train/validation split.

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use anyhow::{Context, Result, bail, ensure};
    use clap::Parser;
    use regex::Regex;
    use serde::{Deserialize, Serialize};
    use sha2::{Digest, Sha256};
    use std::{
        collections::HashSet,
        fs::{self, OpenOptions},
        io::Write,
        path::{Path, PathBuf},
        sync::OnceLock,
        time::{Duration, Instant},
    };

    const BAD_WORDS: &[&str] = &["sex", "drug", "kill", "rape", "nazi"];
    const DEFAULT_ENDPOINT: &str = "https://w5od4nwjt1exl7-11434.proxy.runpod.net";
    const DEFAULT_MODEL: &str = "qwen2.5:32b";

    // ── CLI Configuration ──────────────────────────────────────────────────────────

    #[derive(Parser, Debug, Clone)]
    #[command(
        name = "gen_synthetic_v2",
        about = "Generate structured natural-language synthetic datasets for Yumon v0.2 via Qwen endpoint"
    )]
    pub struct Args {
        /// Dedicated HF endpoint URL, /v1 URL, Ollama, or OpenAI-compatible chat URL.
        #[arg(long, default_value = DEFAULT_ENDPOINT)]
        pub endpoint: String,

        /// Model identifier served by your endpoint (e.g. Qwen/Qwen2.5-32B-Instruct, qwen2.5:32b).
        #[arg(long, default_value = DEFAULT_MODEL)]
        pub model: String,

        /// API key or bearer token (optional; also checks HF_TOKEN / OPENAI_API_KEY env vars).
        #[arg(long, alias = "api-key")]
        pub token: Option<String>,

        /// Target domain: "cnm", "behavior", "memory_salience", "memory_session",
        /// "memory_trait", "memory", "reasoning", "conversation", or "all".
        #[arg(long, default_value = "all")]
        pub domain: String,

        /// Target number of valid samples to generate per domain.
        #[arg(long, default_value_t = 1000)]
        pub count: usize,

        /// Output directory for dataset files and resume state.
        #[arg(long, default_value = "data/synthetic_v2")]
        pub out_dir: PathBuf,

        /// Number of samples to request per prompt batch (typically 4 to 8).
        #[arg(long, default_value_t = 8)]
        pub batch_size: usize,

        /// Sampling temperature.
        #[arg(long, default_value_t = 0.7)]
        pub temperature: f64,

        /// Per-request timeout in seconds.
        #[arg(long, default_value_t = 120)]
        pub timeout: u64,

        /// Optional delay between successful requests in seconds.
        #[arg(long, default_value_t = 0.0)]
        pub delay: f64,

        /// Maximum retry attempts per failed request.
        #[arg(long, default_value_t = 5)]
        pub max_retries: usize,

        /// Dry run: run generators, parsers, and sample tests without calling the remote endpoint.
        #[arg(long)]
        pub dry_run: bool,

        /// Reset existing dataset files and resume state for targeted domains and generate from scratch.
        #[arg(long)]
        pub reset: bool,

        /// Generation mode: "sft" (default) or "preference" / "dpo"
        #[arg(long, default_value = "sft")]
        pub mode: String,
    }

    // ── Supported Domains ─────────────────────────────────────────────────────────

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum Domain {
        Cnm,
        Behavior,
        MemorySalience,
        MemorySession,
        MemoryTrait,
        Reasoning,
        Conversation,
    }

    impl Domain {
        pub fn as_str(&self) -> &'static str {
            match self {
                Self::Cnm => "cnm",
                Self::Behavior => "behavior",
                Self::MemorySalience => "memory_salience",
                Self::MemorySession => "memory_session",
                Self::MemoryTrait => "memory_trait",
                Self::Reasoning => "reasoning",
                Self::Conversation => "conversation",
            }
        }

        pub fn description(&self) -> &'static str {
            match self {
                Self::Cnm => "Central Nervous Model (Routing & Grounded Emotes)",
                Self::Behavior => "Behavior Model (Emergent Actions + Speech Triggers)",
                Self::MemorySalience => "Memory Model (3A: Event Salience Filtering)",
                Self::MemorySession => "Memory Model (3B: Session Summarization & Consolidation)",
                Self::MemoryTrait => "Memory Model (3C: Hierarchical Trait Consolidation)",
                Self::Reasoning => "Reasoning Model (Micro-Plans & Blocked Evaluations)",
                Self::Conversation => "Conversation Model (Spoken Replies in Pet Voice)",
            }
        }
    }

    pub fn resolve_domains(name: &str) -> Result<Vec<Domain>> {
        match name.to_lowercase().trim() {
            "cnm" | "cns" => Ok(vec![Domain::Cnm]),
            "behavior" => Ok(vec![Domain::Behavior]),
            "memory_salience" | "salience" => Ok(vec![Domain::MemorySalience]),
            "memory_session" | "session" => Ok(vec![Domain::MemorySession]),
            "memory_trait" | "trait" => Ok(vec![Domain::MemoryTrait]),
            "memory" => Ok(vec![
                Domain::MemorySalience,
                Domain::MemorySession,
                Domain::MemoryTrait,
            ]),
            "reasoning" | "plan" => Ok(vec![Domain::Reasoning]),
            "conversation" | "talk" => Ok(vec![Domain::Conversation]),
            "all" => Ok(vec![
                Domain::Cnm,
                Domain::Behavior,
                Domain::MemorySalience,
                Domain::MemorySession,
                Domain::MemoryTrait,
                Domain::Reasoning,
                Domain::Conversation,
            ]),
            other => bail!("Unknown domain '{other}'. Choose from: cnm, behavior, memory_salience, memory_session, memory_trait, memory, reasoning, conversation, all"),
        }
    }

    // ── Domain Data Structures & Parsers ──────────────────────────────────────────

    #[derive(Debug, Clone, PartialEq)]
    pub struct GroundedEmote {
        pub having: bool,
        pub knowing: bool,
        pub empathizing: bool,
        pub label: String,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum RouteTarget {
        Memory,
        Reasoning,
        Behavior,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct CnmDispatch {
        pub emote: GroundedEmote,
        pub target: RouteTarget,
        pub context_payload: String,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct BehaviorOutput {
        pub action_description: String,
        pub primary_verb: String,
        pub target_entity: Option<String>,
        pub speech: Option<String>,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct SalienceOutput {
        pub keep: bool,
        pub importance: String,
        pub reason: String,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct SessionConsolidationOutput {
        pub summary: String,
        pub sentence_count: usize,
        pub word_count: usize,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct TraitConsolidationOutput {
        pub trait_statement: String,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum ReasoningOutput {
        Plan(Vec<String>),
        BlockedEvaluation(String),
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct ConversationReply {
        pub reply: String,
        pub word_count: usize,
    }

    // ── Regex Helpers ─────────────────────────────────────────────────────────────

    fn emote_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(r"(?i)Emote:\s*\[\s*having:\s*(yes|no|true|false)\s*,\s*knowing:\s*(yes|no|true|false)\s*,\s*empathizing:\s*(yes|no|true|false)\s*\](?:\s*[:\-]?\s*([^.\n\r]+))?").unwrap()
        })
    }

    fn input_emote_with_label_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(r"(?i)\bEmote:\s*\[\s*having:\s*(yes|no|true|false)\s*,\s*knowing:\s*(yes|no|true|false)\s*,\s*empathizing:\s*(yes|no|true|false)\s*\](?:\s*[:\-]?\s*([^.\n\r]+))?").unwrap()
        })
    }

    fn input_current_emote_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(r"(?i)\b(?:Current\s+)?Emote:\s*\[\s*having:\s*(yes|no|true|false)\s*,\s*knowing:\s*(yes|no|true|false)\s*,\s*empathizing:\s*(yes|no|true|false)\s*\](?:\s*[:\-]?\s*([^.\n\r]+))?").unwrap()
        })
    }

    pub fn validate_input_grounded_emote(input: &str) -> Result<()> {
        if input.contains("<having>")
            || input.contains("<knowing>")
            || input.contains("<empathizing>")
            || input.contains("<curious>")
            || input.contains("<focused>")
            || input.contains("<excited>")
        {
            bail!("input contains unexpanded placeholder syntax");
        }
        if let Some(start) = input.find("Emote: [") {
            if let Some(end) = input[start..].find(']') {
                let bracketed = &input[start..start + end + 1];
                if !input_emote_with_label_regex().is_match(bracketed) {
                    bail!("input has malformed bracketed emote: {bracketed}");
                }
                return Ok(());
            }
        }
        if input_emote_with_label_regex().is_match(input) {
            return Ok(());
        }
        static SIMPLE_EMOTE: OnceLock<Regex> = OnceLock::new();
        let simple = SIMPLE_EMOTE.get_or_init(|| Regex::new(r"(?i)\bEmote:\s*([A-Za-z\- ]+)").unwrap());
        if simple.is_match(input) {
            return Ok(());
        }
        bail!("input missing or malformed 'Emote: [having: <yes|no>, knowing: <yes|no>, empathizing: <yes|no>] <Label>'")
    }

    pub fn validate_input_current_emote(input: &str) -> Result<()> {
        if input.contains("<having>")
            || input.contains("<knowing>")
            || input.contains("<empathizing>")
        {
            bail!("input contains unexpanded placeholder syntax");
        }
        if let Some(start) = input.find("Emote: [") {
            if let Some(end) = input[start..].find(']') {
                let bracketed = &input[start..start + end + 1];
                if !input_current_emote_regex().is_match(bracketed) {
                    bail!("input has malformed bracketed emote: {bracketed}");
                }
                return Ok(());
            }
        }
        if input_current_emote_regex().is_match(input) {
            return Ok(());
        }
        bail!("input missing or malformed 'Current Emote: [having: <yes|no>, knowing: <yes|no>, empathizing: <yes|no>]'")
    }

    fn route_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(r"(?i)Route:\s*(?:send\s+to\s+|to\s+)?(Memory|Reasoning|Behavior)(?:\s*(?:with|:|-)\s*(.+))?$").unwrap()
        })
    }

    fn action_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| Regex::new(r"(?i)^Action:\s*(.+)$").unwrap())
    }

    fn speak_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| Regex::new(r#"(?i)^Speak:\s*(?:"([^"]+)"|'([^']+)'|“([^”]+)”|none|([^"\n\r]+))$"#).unwrap())
    }

    fn consolidated_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| Regex::new(r"(?i)^(?:Consolidated|Consolidation|Summary|Session\s+Summary|Memory):\s*(.+)$").unwrap())
    }

    fn trait_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| Regex::new(r"(?i)^(?:Core\s+Trait|Trait|Personality\s+Trait|Consolidated\s+Trait):\s*(.+)$").unwrap())
    }

    fn reasoning_step_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| Regex::new(r"(?i)^(?:Step\s*\d+|[-*]?\s*\d+)[.:)]\s*(.+)$").unwrap())
    }

    fn reasoning_eval_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(r"(?i)^(?:Evaluation|Blocked|Evaluation\s*-\s*Blocked|Impossible)[:.]?\s*(?:Blocked\s+because\s+)?(.+)$").unwrap()
        })
    }

    fn reply_regex() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| Regex::new(r#"(?i)^Reply:\s*["'“]?([^"'“”\r\n]+)["'”]?$"#).unwrap())
    }

    // ── Cleaning & Validation ─────────────────────────────────────────────────────

    pub fn is_clean(s: &str) -> bool {
        let low = s.to_lowercase();
        for word in low.split(|c: char| !c.is_alphabetic()) {
            if word.is_empty() {
                continue;
            }
            for &bad in BAD_WORDS {
                if bad == "kill" {
                    // Match "kill", "kills", "killed", "killing", "killer", but NOT "skill"
                    if word.starts_with("kill") {
                        return false;
                    }
                } else if word == bad || word.starts_with(bad) {
                    return false;
                }
            }
        }
        true
    }

    pub fn count_words(s: &str) -> usize {
        s.split_whitespace().count()
    }

    pub fn count_sentences(s: &str) -> usize {
        s.split(|c| c == '.' || c == '!' || c == '?')
            .filter(|chunk| !chunk.trim().is_empty())
            .count()
    }

    /// Basic safety and profanity check on generated natural text.
    /// Notice: Kilo-Talk vocabulary filtering is intentionally omitted here to preserve
    /// rich, authentic natural language data; Kilo tokenization runs during caching.
    pub fn validate_clean_text(text: &str) -> Result<()> {
        ensure!(is_clean(text), "contains prohibited vocabulary");
        Ok(())
    }

    // ── Deterministic Output Parsers ──────────────────────────────────────────────

    pub fn parse_cnm_target(target: &str) -> Result<CnmDispatch> {
        let mut emote: Option<GroundedEmote> = None;
        let mut target_route: Option<(RouteTarget, String)> = None;

        for line in target.lines().map(str::trim).filter(|l| !l.is_empty()) {
            if let Some(caps) = emote_regex().captures(line) {
                let having = matches!(caps[1].to_lowercase().as_str(), "yes" | "true");
                let knowing = matches!(caps[2].to_lowercase().as_str(), "yes" | "true");
                let empathizing = matches!(caps[3].to_lowercase().as_str(), "yes" | "true");
                let label = caps
                    .get(4)
                    .map(|m| m.as_str().trim().trim_matches(|c: char| c == '.' || c == ':' || c == '-').trim().to_string())
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| "Attentive".to_string());
                validate_clean_text(&label)
                    .with_context(|| format!("validating emote label '{label}'"))?;
                emote = Some(GroundedEmote {
                    having,
                    knowing,
                    empathizing,
                    label,
                });
            } else if let Some(caps) = route_regex().captures(line) {
                let tgt = match caps[1].to_lowercase().as_str() {
                    "memory" => RouteTarget::Memory,
                    "reasoning" => RouteTarget::Reasoning,
                    "behavior" => RouteTarget::Behavior,
                    other => bail!("invalid Route target '{other}'"),
                };
                let payload = caps
                    .get(2)
                    .map(|m| m.as_str().trim().trim_end_matches('.').trim().to_string())
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| format!("dispatch to {}", &caps[1]));
                validate_clean_text(&payload)
                    .with_context(|| format!("validating route payload '{payload}'"))?;
                target_route = Some((tgt, payload));
            }
        }

        let emote = emote.context("missing or malformed 'Emote: [having: ..., knowing: ..., empathizing: ...] <Name>'")?;
        let (tgt, payload) = target_route.context("missing or malformed 'Route: send to <Target> with <Context>'")?;

        Ok(CnmDispatch {
            emote,
            target: tgt,
            context_payload: payload,
        })
    }

    pub fn parse_behavior_target(
        target: &str,
        environment: &str,
    ) -> Result<BehaviorOutput> {
        let mut action_desc: Option<String> = None;
        let mut speech: Option<Option<String>> = None;

        for line in target.lines().map(str::trim).filter(|l| !l.is_empty()) {
            if let Some(caps) = action_regex().captures(line) {
                let desc = caps[1].trim().to_string();
                ensure!(!desc.is_empty(), "action description cannot be empty");
                validate_clean_text(&desc)
                    .with_context(|| format!("validating action description '{desc}'"))?;
                action_desc = Some(desc);
            } else if let Some(caps) = speak_regex().captures(line) {
                let matched = caps.get(1)
                    .or_else(|| caps.get(2))
                    .or_else(|| caps.get(3))
                    .or_else(|| caps.get(4));
                if let Some(m) = matched {
                    let text = m.as_str().trim().trim_matches(|c: char| c == '"' || c == '\'' || c == '“' || c == '”').trim().to_string();
                    if text.eq_ignore_ascii_case("none") || text.is_empty() {
                        speech = Some(None);
                    } else {
                        let words = count_words(&text);
                        ensure!(
                            (1..=35).contains(&words),
                            "spoken utterance has {words} words, expected 1-35 words"
                        );
                        validate_clean_text(&text)
                            .with_context(|| format!("validating speech utterance '{text}'"))?;
                        speech = Some(Some(text));
                    }
                } else {
                    speech = Some(None);
                }
            }
        }

        // If Action: line was not explicitly marked, fallback to the first line that isn't Speak:
        let action_description = if let Some(desc) = action_desc {
            desc
        } else {
            let first_line = target
                .lines()
                .map(str::trim)
                .find(|l| !l.is_empty() && !l.to_lowercase().starts_with("speak:"))
                .context("missing action description in target")?;
            validate_clean_text(first_line)?;
            first_line.to_string()
        };

        // If Speak: line was omitted entirely, default to silent (None)
        let speech = speech.unwrap_or(None);

        // Extract primary verb from action description (skipping auxiliaries and adverbs)
        let words: Vec<&str> = action_description.split_whitespace().collect();
        let start_idx = if words.len() >= 2
            && words[0].eq_ignore_ascii_case("i")
            && (words[1].eq_ignore_ascii_case("will") || words[1].eq_ignore_ascii_case("can"))
        {
            2
        } else {
            0
        };

        let mut verb_idx = start_idx;
        while verb_idx < words.len()
            && (words[verb_idx].ends_with("ly") || matches!(words[verb_idx].to_lowercase().as_str(), "just" | "now" | "also" | "then"))
            && verb_idx + 1 < words.len()
        {
            verb_idx += 1;
        }

        let primary_verb = if verb_idx < words.len() {
            words[verb_idx].trim_matches(|c: char| !c.is_alphabetic()).to_lowercase()
        } else {
            "act".to_string()
        };

        // Check for target entity referenced in environment
        let target_entity = environment
            .split(',')
            .map(str::trim)
            .find(|ent| !ent.is_empty() && action_description.to_lowercase().contains(&ent.to_lowercase()))
            .map(str::to_string);

        Ok(BehaviorOutput {
            action_description,
            primary_verb,
            target_entity,
            speech,
        })
    }

    pub fn parse_memory_salience_target(target: &str) -> Result<SalienceOutput> {
        let trimmed = target.trim();
        let low = trimmed.to_lowercase();

        let is_keep = if low.starts_with("keep") {
            true
        } else if low.starts_with("discard") {
            false
        } else if low.contains("keep") && !low.contains("discard") {
            true
        } else if low.contains("discard") {
            false
        } else {
            bail!("malformed salience output: expected 'Keep: ...' or 'Discard: ...'")
        };

        let importance = if low.contains("high") {
            "high".to_string()
        } else if low.contains("medium") {
            "medium".to_string()
        } else if low.contains("low") {
            "low".to_string()
        } else if is_keep {
            "high".to_string()
        } else {
            "low".to_string()
        };

        let reason = if let Some(idx) = low.find("because") {
            trimmed[idx + 7..].trim().trim_start_matches(':').trim().to_string()
        } else if let Some(idx) = trimmed.find(':') {
            trimmed[idx + 1..].trim().to_string()
        } else {
            trimmed.to_string()
        };

        let reason = if reason.is_empty() {
            if is_keep { "meaningful event".to_string() } else { "routine event".to_string() }
        } else {
            reason
        };

        validate_clean_text(&reason)?;
        Ok(SalienceOutput {
            keep: is_keep,
            importance,
            reason,
        })
    }

    pub fn parse_memory_session_target(target: &str) -> Result<SessionConsolidationOutput> {
        let trimmed = target.trim();
        let summary = if let Some(caps) = consolidated_regex().captures(trimmed) {
            caps[1].trim().to_string()
        } else {
            trimmed.to_string()
        };
        let sentence_count = count_sentences(&summary);
        let word_count = count_words(&summary);

        ensure!(
            sentence_count >= 1 && sentence_count <= 4,
            "session consolidation must be 1 to 4 sentences (got {sentence_count})"
        );
        ensure!(
            (4..=55).contains(&word_count),
            "session consolidation must be 4 to 55 words (got {word_count})"
        );
        validate_clean_text(&summary)?;

        Ok(SessionConsolidationOutput {
            summary,
            sentence_count,
            word_count,
        })
    }

    pub fn parse_memory_trait_target(target: &str) -> Result<TraitConsolidationOutput> {
        let trimmed = target.trim();
        let trait_statement = if let Some(caps) = trait_regex().captures(trimmed) {
            caps[1].trim().to_string()
        } else {
            trimmed.to_string()
        };
        ensure!(!trait_statement.is_empty(), "trait statement cannot be empty");
        let word_count = count_words(&trait_statement);
        ensure!(
            (3..=60).contains(&word_count),
            "core trait must be between 3 and 60 words (got {word_count})"
        );
        validate_clean_text(&trait_statement)?;

        Ok(TraitConsolidationOutput { trait_statement })
    }

    pub fn parse_reasoning_target(target: &str) -> Result<ReasoningOutput> {
        let trimmed = target.trim();

        if trimmed.to_lowercase().starts_with("plan:")
            || trimmed.to_lowercase().contains("step 1")
            || trimmed.to_lowercase().contains("1.")
        {
            let mut steps = Vec::new();
            for line in trimmed.lines().map(str::trim).filter(|l| !l.is_empty()) {
                if line.to_lowercase().starts_with("plan:") {
                    continue;
                }
                if let Some(caps) = reasoning_step_regex().captures(line) {
                    let step_content = caps[caps.len() - 1].trim().to_string();
                    validate_clean_text(&step_content)?;
                    steps.push(step_content);
                }
            }
            if !steps.is_empty() {
                ensure!(
                    steps.len() <= 4,
                    "Plan exceeded maximum 4 steps (got {})",
                    steps.len()
                );
                return Ok(ReasoningOutput::Plan(steps));
            }
        }

        if let Some(caps) = reasoning_eval_regex().captures(trimmed) {
            let reason = caps[1].trim().to_string();
            ensure!(!reason.is_empty(), "Evaluation reason cannot be empty");
            validate_clean_text(&reason)?;
            return Ok(ReasoningOutput::BlockedEvaluation(reason));
        }

        if trimmed.to_lowercase().contains("blocked because") || trimmed.to_lowercase().contains("cannot reach") {
            let reason = trimmed
                .strip_prefix("Evaluation:")
                .or_else(|| trimmed.strip_prefix("Blocked:"))
                .unwrap_or(trimmed)
                .trim()
                .to_string();
            validate_clean_text(&reason)?;
            return Ok(ReasoningOutput::BlockedEvaluation(reason));
        }

        bail!("Reasoning output must start with 'Plan:' with Step 1..N or 'Evaluation: Blocked because ...'")
    }

    pub fn parse_conversation_target(target: &str) -> Result<ConversationReply> {
        let trimmed = target.trim();
        let reply = if let Some(caps) = reply_regex().captures(trimmed) {
            caps[1].trim().to_string()
        } else {
            trimmed
                .strip_prefix("Reply:")
                .or_else(|| trimmed.strip_prefix("reply:"))
                .unwrap_or(trimmed)
                .trim()
                .to_string()
        };

        let reply = reply.trim_matches(|c: char| c == '"' || c == '\'' || c == '“' || c == '”').trim().to_string();
        let sentence_count = count_sentences(&reply);
        let word_count = count_words(&reply);

        ensure!(
            sentence_count >= 1 && sentence_count <= 2,
            "Reply must be 1 or 2 sentences (got {sentence_count})"
        );
        ensure!(
            (2..=30).contains(&word_count),
            "Reply must be between 2 and 30 words (got {word_count})"
        );
        validate_clean_text(&reply)?;

        Ok(ConversationReply { reply, word_count })
    }

    // ── Prompt Generators ─────────────────────────────────────────────────────────

    pub fn build_system_prompt(domain: Domain) -> &'static str {
        match domain {
            Domain::Cnm => {
                r#"You generate training data for Yumon's Central Nervous Model (CNM).
The CNM is an affective state manager and router that monitors Vitals, Perception, and Events,
maintains Grounded Emotes along 3 existential dimensions, and routes context to Memory, Reasoning, or Behavior.
It NEVER executes physical actions directly.

Input Format:
Vitals: Hunger <0-100>, Energy <0-100>, Cleanliness <0-100>, Boredom <0-100>.
Perception: sees <detected objects/scene>.
Event: <incoming world event or model return>.
Current Emote: [having: <yes|no>, knowing: <yes|no>, empathizing: <yes|no>].

Target Output Format:
Emote: [having: <yes|no>, knowing: <yes|no>, empathizing: <yes|no>] <EmoteName>.
Route: send to <Memory|Reasoning|Behavior> with <context description>.

Rules:
- EmoteName is open-ended: freely choose ANY natural emotion, mood, or psychological state fitting the 3 existential dimensions:
  * having: yes|no (having/satisfied vs wanting/lacking)
  * knowing: yes|no (understanding/certainty vs uncertainty/surprise)
  * empathizing: yes|no (social connection vs detached/self-absorbed)
- Routing Architecture:
  * Behavior (~45-50%): Everyday direct reactions, biological drives (eating, resting), immediate play, social greetings, and executing received plans.
  * Memory (~25-30%): Significant milestone moments, learned tricks, owner bonding, frights comforted by owner, or session consolidation.
  * Reasoning (~20-25%): Reserved strictly for situations with an active physical blocker, complex puzzle, navigation impediment, or multi-step obstacle requiring micro-planning. Do NOT default everything to Reasoning!
- Plain natural language only. No JSON, no markdown fences, no explanations.
Output each sample as:
INPUT:  <input text>
TARGET: <target text>
"#
            }
            Domain::Behavior => {
                r#"You generate training data for Yumon's Behavior Model.
The Behavior Model selects Yumon's concrete physical, exploratory, social, creative, or resting action across open-ended natural situations.
When the action involves vocalizing, asking, or greeting, it also triggers speech in pet voice.

Input Format:
Vitals: Hunger <0-100>, Energy <0-100>, Cleanliness <0-100>, Boredom <0-100>.
Emote: [having: <yes|no>, knowing: <yes|no>, empathizing: <yes|no>] <EmoteName>.
Environment: <location>, nearby: <objects, places, people>.
Memory: <relevant remembered fact or habit, or "none">.
Plan: <reasoning plan, or "none">.

Target Output Format:
Action: <expressive natural language action description starting with an active verb or "I will <verb>">.
Speak: <"spoken utterance in pet voice", or "none">.

Rules:
- Emote in INPUT must strictly follow the boolean bracket structure: [having: yes|no, knowing: yes|no, empathizing: yes|no] <EmoteName>.
- EmoteName is open-ended: freely choose ANY emotion, mood, or psychological state that matches the having/knowing/empathizing dimensions (e.g. Content, Mischievous, Sullen, Euphoric, Vigilant, Bashful, Yearning, Serene, Fretful, Proud, Playful, Drowsy, etc.). Do NOT repeat the same emotion or default to 'Curious'.
- Environments, actions, and settings are completely open-ended. Pet actions can range across playing, sleeping, grooming, tinkering, exploring, eating, socializing, building, seeking warmth, or hiding.
- If Speak is not "none", the utterance must be enclosed in quotes and contain 4 to 16 warm, natural English words. Use "Speak: none" appropriately for quiet or non-vocal actions.
- If Plan is provided, Action should follow the first step.
Output each sample as:
INPUT:  <input text>
TARGET: <target text>
"#
            }
            Domain::MemorySalience => {
                r#"You generate training data for Yumon's Memory Model (Subtask 3A: Event Salience Filtering).
The model decides whether an incoming event has high/medium emotional/social importance to keep in episodic memory, or should be discarded as routine reflex.

Input Format:
Event: <happening>. Emote: [having: <yes|no>, knowing: <yes|no>, empathizing: <yes|no>] <EmoteName>. Vitals: Hunger <H>, Energy <E>, Cleanliness <C>, Boredom <B>.

Target Output Format:
Keep: <high|medium> importance because <grounded reason>.
OR
Discard: low importance because <grounded reason>.

Rules:
- Emote in INPUT must strictly follow: [having: yes|no, knowing: yes|no, empathizing: yes|no] <EmoteName>. Any emotion name can be used.
- Equal Balance: Training requires a balanced 50/50 mix of KEEP and DISCARD decisions.
- Discard (low importance): Routine biological reflexes and mundane background occurrences (e.g. drinking water from dish, routine stretching after nap, walking across carpet, yawning, idle ear twitch, routine grooming, shifting resting position).
- Keep (high/medium importance): Meaningful emotional, social, or learning events (e.g. learning a new skill, receiving praise/treat, owner comforting after a fright, solving a puzzle, discovering something novel).
- Open-ended events: Freely invent diverse life situations. Do NOT repeat the same trope (such as repetitive thunderclaps).
Output each sample as:
INPUT:  <input text>
TARGET: <target text>
"#
            }
            Domain::MemorySession => {
                r#"You generate training data for Yumon's Memory Model (Subtask 3B: Session Summarization & Consolidation).
The model consolidates raw sequential events from a play, care, or learning session into a single cohesive episodic memory in warm, natural pet voice.

Input Format:
Session Log:
- <Event 1>
- <Event 2>
- <Event 3>
- <Event 4>

Target Output Format:
Consolidated: <Sentence 1>. <Optional Sentence 2>.

Rules:
- Exactly 1 or 2 sentences.
- Between 12 and 25 words total.
- Warm pet voice using expressive everyday words (e.g. "Today my red ball rolled behind the lattice...").
Output each sample as:
INPUT:  <input text>
TARGET: <target text>
"#
            }
            Domain::MemoryTrait => {
                r#"You generate training data for Yumon's Memory Model (Subtask 3C: Hierarchical Trait Consolidation).
The model synthesizes multiple episodic session memories into an enduring core trait, preference, or attachment bond.

Input Format:
Past Memories:
- <Past Memory 1>
- <Past Memory 2>
- <Past Memory 3>

Target Output Format:
Core Trait: <synthesized pet preference, personality trait, or attachment bond>.

Rules:
- One clear sentence synthesizing the common theme across memories.
- Between 8 and 30 words.
Output each sample as:
INPUT:  <input text>
TARGET: <target text>
"#
            }
            Domain::Reasoning => {
                r#"You generate training data for Yumon's Reasoning Model (Micro-Planning & Evaluation).
The model produces bounded micro-plans (1 to 4 steps) to navigate obstacles, or evaluates why an impossible goal cannot be achieved.

Input Format:
Goal: <target outcome>. State: <current pet status & location>. Obstacle: <blocker or unknown condition>. Perception: sees <objects>.

Target Output Format (for solvable goals):
Plan:
Step 1: <verb> <target>.
Step 2: <verb> <target>.
Step 3: <verb> <target>.

Target Output Format (for impossible/blocked goals):
Evaluation: Blocked because <physical or logical constraint>.

Rules:
- Produce a realistic mix: ~70% solvable micro-plans (1 to 4 steps) and ~30% blocked evaluations.
- Blocked evaluation must explain the physical constraint (e.g. latch too high and heavy, glass pane impassable, cabinet door locked from outside).
- Plans must be pet-scale (using cushions, nudging rulers, tipping over small wastebaskets, climbing low chairs).
Output each sample as:
INPUT:  <input text>
TARGET: <target text>
"#
            }
            Domain::Conversation => {
                r#"You generate training data for Yumon's Conversation Model.
The model generates warm, short pet spoken replies (4 to 16 words) perfectly grounded in vitals, grounded emote, and intent.

Input Format:
Vitals: Hunger <H>, Energy <E>, Cleanliness <C>, Boredom <B>.
Emote: [having: <yes|no>, knowing: <yes|no>, empathizing: <yes|no>] <EmoteName>.
Intent: <what behavior wants to communicate>.
Owner Prompt: <owner question or remark, or "none">.

Target Output Format:
Reply: "<spoken sentence>"

Rules:
- Emote in INPUT must strictly use the boolean bracket syntax: [having: yes|no, knowing: yes|no, empathizing: yes|no] <EmoteName>.
- Exactly ONE short sentence in quotation marks (4 to 16 words).
- Warm, expressive pet voice matching the grounded emote and intent. Natural English.
- Vary intents across greetings, requesting food/treats, asking to play, expressing sleepy contentment, reacting to praise, or declining active games.
Output each sample as:
INPUT:  <input text>
TARGET: <target text>
"#
            }
        }
    }

    pub fn build_user_prompt(domain: Domain, batch_size: usize) -> String {
        match domain {
            Domain::Cnm => {
                format!(
                    r#"Generate exactly {batch_size} completely distinct, creative, and open-ended training sample(s) for the cnm schema.

Structural Routing & Emote Rules:
- Emote in TARGET must match the 3 existential dimensions:
  Emote: [having: <yes|no>, knowing: <yes|no>, empathizing: <yes|no>] <EmoteName>.
  EmoteName is open-ended: freely choose ANY nuanced emotion, mood, or psychological state matching the dimensions (e.g. Content, Vigilant, Yearning, Sullen, Euphoric, Startled, Mischievous, Serene, Fretful, Proud, Cautious, etc.).
- Critical Routing Balance:
  * In everyday life, the pet routes primarily to Behavior (~45-50%) for direct actions/urges and Memory (~25-30%) for logging milestones and experiences.
  * Route to Reasoning ONLY when facing a true physical blocker, puzzle, or multi-step obstacle requiring planning.
  * In this batch, ensure at least one sample routes to Behavior and do NOT send everything to Reasoning!
- Scenarios, environments, objects, and perception must be completely open-ended and varied across all samples.

Return ONLY the samples formatted as:
INPUT:  ...
TARGET: ...

No commentary, no explanations, no markdown fences."#
                )
            }
            Domain::Behavior => {
                format!(
                    r#"Generate exactly {batch_size} completely distinct, creative, and open-ended training sample(s) for the behavior schema.

Open-Ended Diversity & Emote Rules:
- Emote in INPUT must strictly use: Emote: [having: <yes|no>, knowing: <yes|no>, empathizing: <yes|no>] <EmoteName>.
- Do NOT use placeholder brackets like [<having>]! Use actual yes/no values.
- EmoteName is open-ended: choose a completely DIFFERENT emotion and 3D dimension combination for each sample. Freely draw from any feeling or temperament (e.g. playful, defiant, drowsy, affectionate, bashful, mischievous, weary, startled, relaxed, proud, hungry, stubborn). Do NOT default to "Curious"!
- Scenarios and environments are completely open-ended: invent any room, landscape, outdoor space, miniature terrain, or cozy nook.
- Actions must be varied and emergent across play, resting, grooming, building, communicating, foraging, cuddling, nesting, or observing.
- Target Action must start with an active verb or "I will <verb>".
- Target Speak must be either a warm spoken pet utterance enclosed in quotes (4-16 words) or "none". Use "none" naturally for non-vocal actions.

Return ONLY the samples formatted as:
INPUT:  ...
TARGET: ...

No commentary, no explanations, no markdown fences."#
                )
            }
            Domain::MemorySalience => {
                format!(
                    r#"Generate exactly {batch_size} completely distinct, creative, and open-ended training sample(s) for the memory_salience schema.

Balance & Diversity Rules:
- CRITICAL: Generate an approximately equal 50/50 mix of KEEP and DISCARD decisions across the samples in this batch.
- For DISCARD samples: invent realistic routine biological reflexes and unremarkable everyday actions (e.g. taking a sip of water, shifting sleeping posture, routine walking across the room, yawning, idle tail flick, licking a paw) that have low salience and should be discarded.
- For KEEP samples: invent novel learning, emotional bonding, or memorable milestones that have high or medium importance to keep.
- Do NOT repeat thunderclaps or sudden loud window vibrations. Freely explore open-ended everyday life situations.
- Emote in INPUT must strictly use: Emote: [having: <yes|no>, knowing: <yes|no>, empathizing: <yes|no>] <EmoteName>.
  EmoteName is open-ended: choose any fitting natural emotion.

Return ONLY the samples formatted as:
INPUT:  ...
TARGET: ...

No commentary, no explanations, no markdown fences."#
                )
            }
            Domain::Conversation => {
                format!(
                    r#"Generate exactly {batch_size} completely distinct, creative, and open-ended training sample(s) for the conversation schema.

Requirements:
- Emote in INPUT must strictly use: Emote: [having: <yes|no>, knowing: <yes|no>, empathizing: <yes|no>] <EmoteName>.
  EmoteName is open-ended: choose ANY natural emotion or mood.
- Reply must be exactly ONE sentence in quotation marks (4 to 16 words) in warm, expressive pet voice.
- Freely invent varied conversation situations and intents (greeting, asking for food, declining an activity, sharing excitement, bedtime murmur).

Return ONLY the samples formatted as:
INPUT:  ...
TARGET: ...

No commentary, no explanations, no markdown fences."#
                )
            }
            Domain::Reasoning => {
                format!(
                    r#"Generate exactly {batch_size} completely distinct, creative, and open-ended training sample(s) for the reasoning schema.

Requirements:
- Freely invent diverse pet-scale obstacles, goals, and environments.
- Provide a realistic mix of solvable micro-plans (Plan: Step 1: ... Step 2: ... Step 3: ...) and impossible/blocked goals (Evaluation: Blocked because <physical or logical constraint>).
- Plans must be 1 to 4 steps maximum.

Return ONLY the samples formatted as:
INPUT:  ...
TARGET: ...

No commentary, no explanations, no markdown fences."#
                )
            }
            Domain::MemorySession => {
                format!(
                    r#"Generate exactly {batch_size} completely distinct, creative, and open-ended training sample(s) for the memory_session schema.

Requirements:
- Session Log contains 3 to 5 open-ended sequential events from any pet activity or daily happening.
- Target Consolidated: exactly 1 or 2 sentences in warm, simple pet voice (12 to 25 words total).
- Avoid repetition; freely explore diverse scenarios and environments.

Return ONLY the samples formatted as:
INPUT:  ...
TARGET: ...

No commentary, no explanations, no markdown fences."#
                )
            }
            Domain::MemoryTrait => {
                format!(
                    r#"Generate exactly {batch_size} completely distinct, creative, and open-ended training sample(s) for the memory_trait schema.

Requirements:
- Past Memories contains 3 distinct past session memories illustrating a consistent pet preference, personality trait, or attachment bond.
- Target Core Trait: one clear sentence (8 to 30 words) synthesizing the enduring trait or preference.
- Freely invent varied personality traits, preferences, quirks, and bonds.

Return ONLY the samples formatted as:
INPUT:  ...
TARGET: ...

No commentary, no explanations, no markdown fences."#
                )
            }
        }
    }

    pub fn build_preference_system_prompt(domain: Domain) -> &'static str {
        match domain {
            Domain::Cnm => {
                r#"You generate preference pairs for Direct Preference Optimization (DPO) of Yumon's Central Nervous Model (CNM).
The CNM monitors Vitals, Perception, and Events, maintains Grounded Emotes along 3 existential dimensions [having: yes|no, knowing: yes|no, empathizing: yes|no] <Label>, and routes context to Memory, Reasoning, or Behavior.
It NEVER executes physical actions directly.

For each sample, output:
- INPUT: Context with Vitals, Perception, Event, and Current Emote.
- CHOSEN: The ideal, perfectly grounded response with matching 3D emote and appropriate routing target.
- REJECTED: A plausible but flawed candidate (e.g. emotion contradicts vitals, wrong routing target like sending routine eating to Reasoning, or broken routing payload).

Output each preference triplet as:
INPUT:
Vitals: Hunger <0-100>, Energy <0-100>, Cleanliness <0-100>, Boredom <0-100>.
Perception: sees <objects>.
Event: <event>.
Current Emote: [having: <yes|no>, knowing: <yes|no>, empathizing: <yes|no>].

CHOSEN:
Emote: [having: <yes|no>, knowing: <yes|no>, empathizing: <yes|no>] <Label>.
Route: send to <Memory|Reasoning|Behavior> with <context description>.

REJECTED:
Emote: [having: <yes|no>, knowing: <yes|no>, empathizing: <yes|no>] <Label>.
Route: send to <Memory|Reasoning|Behavior> with <flawed context or incorrect route>.
"#
            }
            Domain::Behavior => {
                r#"You generate preference pairs for Direct Preference Optimization (DPO) of Yumon's Behavior Model.
The Behavior Model selects concrete physical actions across open-ended natural situations and optionally speaks in pet voice.

For each sample, output:
- INPUT: Vitals, Emote, Environment, Memory, and Plan.
- CHOSEN: Grounded physical action starting with active verb or "I will", with optional warm pet speech (4 to 16 words in quotes).
- REJECTED: Contrastive flawed candidate: passive/frozen action ("I will sit and think"), action contradicting vitals (sprinting with 5 energy), hallucinating objects not in Environment, or robotic assistant dialogue ("As an AI companion...").

Output each preference triplet as:
INPUT:
Vitals: Hunger <H>, Energy <E>, Cleanliness <C>, Boredom <B>.
Emote: [having: <yes|no>, knowing: <yes|no>, empathizing: <yes|no>] <Label>.
Environment: <location>, nearby: <objects>.
Memory: <memory or "none">.
Plan: <plan or "none">.

CHOSEN:
Action: <active physical action description>.
Speak: <"warm pet speech" or "none">.

REJECTED:
Action: <flawed, passive, or contradictory action>.
Speak: <robotic, overly verbose, or inappropriate speech, or "none">.
"#
            }
            Domain::MemorySalience => {
                r#"You generate preference pairs for Direct Preference Optimization (DPO) of Yumon's Memory Model (Subtask 3A: Event Salience Filtering).
The model decides whether an incoming event has high/medium importance to keep in episodic memory, or should be discarded as routine reflex.

For each sample, output:
- INPUT: Event, Emote, Vitals.
- CHOSEN: Accurate decision (Keep for significant emotional/learning milestones; Discard for mundane biological reflexes) with grounded reason.
- REJECTED: Inverted decision (e.g. Discarding owner bonding or learning a new trick; or Keeping a mundane sneeze/paw stretch) or contradictory reasoning.

Output each preference triplet as:
INPUT:
Event: <event>. Emote: [having: <yes|no>, knowing: <yes|no>, empathizing: <yes|no>] <Label>. Vitals: Hunger <H>, Energy <E>, Cleanliness <C>, Boredom <B>.

CHOSEN:
Keep: <high|medium> importance because <reason>.
(OR Discard: low importance because <reason>.)

REJECTED:
Discard: low importance because <flawed reason>.
(OR Keep: high importance because <flawed reason>.)
"#
            }
            Domain::MemorySession => {
                r#"You generate preference pairs for Direct Preference Optimization (DPO) of Yumon's Memory Model (Subtask 3B: Session Summarization).
The model consolidates raw sequential events from a session into a cohesive episodic memory (1-2 sentences, 12-25 words) in warm pet voice.

For each sample, output:
- INPUT: Session Log with 3-5 bulleted events.
- CHOSEN: Concise, warm 1-2 sentence narrative (12-25 words) capturing the session theme.
- REJECTED: Overly long, rambling, detached robotic summary, or bulleted list.

Output each preference triplet as:
INPUT:
Session Log:
- <Event 1>
- <Event 2>
- <Event 3>

CHOSEN:
Consolidated: <1-2 sentences in warm pet voice, 12-25 words>.

REJECTED:
Consolidated: <overly verbose, robotic, or disconnected summary>.
"#
            }
            Domain::MemoryTrait => {
                r#"You generate preference pairs for Direct Preference Optimization (DPO) of Yumon's Memory Model (Subtask 3C: Hierarchical Trait Consolidation).
The model synthesizes multiple episodic session memories into an enduring core trait, preference, or attachment bond (1 sentence, 8-30 words).

For each sample, output:
- INPUT: Past Memories (3 distinct memories).
- CHOSEN: One clear synthesizing sentence (8-30 words) expressing the core trait.
- REJECTED: Verbatim copy of just one memory, or clinical/robotic jargon.

Output each preference triplet as:
INPUT:
Past Memories:
- <Memory 1>
- <Memory 2>
- <Memory 3>

CHOSEN:
Core Trait: <synthesized trait, 8-30 words>.

REJECTED:
Core Trait: <flawed, repetitive, or robotic trait>.
"#
            }
            Domain::Reasoning => {
                r#"You generate preference pairs for Direct Preference Optimization (DPO) of Yumon's Reasoning Model (Micro-Planning & Evaluation).
The model produces bounded pet-scale micro-plans (1 to 4 steps) or evaluates why an impossible goal cannot be achieved.

For each sample, output:
- INPUT: Goal, State, Obstacle, Perception.
- CHOSEN: Feasible pet-scale micro-plan (1-4 steps: Plan:\nStep 1: ...\nStep 2: ...) or grounded Blocked evaluation (Evaluation: Blocked because <constraint>).
- REJECTED: Human-scale impossible actions (e.g. typing passwords, turning doorknobs with fingers, buying tools online), plans exceeding 4 steps, or declaring blocked when an obvious pet-scale route exists.

Output each preference triplet as:
INPUT:
Goal: <goal>. State: <state>. Obstacle: <obstacle>. Perception: sees <objects>.

CHOSEN:
Plan:
Step 1: <pet-scale action>.
Step 2: <pet-scale action>.
(OR Evaluation: Blocked because <physical constraint>.)

REJECTED:
Plan:
Step 1: <impossible or human-scale action>.
(OR Evaluation: Blocked because <nonsensical excuse>.)
"#
            }
            Domain::Conversation => {
                r#"You generate preference pairs for Direct Preference Optimization (DPO) of Yumon's Conversation Model.
The model generates warm, short pet spoken replies (4 to 16 words in quotes) grounded in vitals, grounded emote, and intent.

For each sample, output:
- INPUT: Vitals, Emote, Intent, Owner Prompt.
- CHOSEN: Warm, expressive pet reply (4 to 16 words) enclosed in quotation marks matching the emote and intent.
- REJECTED: Robotic assistant talk ("How may I assist you today?"), overly long reply (>20 words), reply without quotation marks, or speech contradicting the emotion.

Output each preference triplet as:
INPUT:
Vitals: Hunger <H>, Energy <E>, Cleanliness <C>, Boredom <B>.
Emote: [having: <yes|no>, knowing: <yes|no>, empathizing: <yes|no>] <Label>.
Intent: <intent>.
Owner Prompt: <owner prompt or "none">.

CHOSEN:
Reply: "<4-16 word warm pet sentence>"

REJECTED:
Reply: "<robotic, overly verbose, or emotionally contradictory sentence>"
"#
            }
        }
    }

    pub fn build_preference_user_prompt(domain: Domain, batch_size: usize) -> String {
        match domain {
            Domain::Cnm => {
                format!(
                    r#"Generate exactly {batch_size} completely distinct, creative preference triplet(s) for the CNM schema.

Requirements:
- INPUT: Vitals, Perception, Event, and Current Emote: [having: <yes|no>, knowing: <yes|no>, empathizing: <yes|no>].
- CHOSEN: Emote with 3D dimensions [having: <yes|no>, knowing: <yes|no>, empathizing: <yes|no>] <Label> matching the state, and Route: send to <Behavior|Memory|Reasoning> with <description>.
- REJECTED: A contrastive flawed response: mismatched emotion dimensions (e.g. feeling satisfied when starving), wrong route destination (e.g. sending ordinary eating/sleeping to Reasoning, or complex obstacle to Memory), or incoherent payload.

Return ONLY the samples formatted as:
INPUT:  ...
CHOSEN: ...
REJECTED: ...

No commentary, no explanations, no markdown fences."#
                )
            }
            Domain::Behavior => {
                format!(
                    r#"Generate exactly {batch_size} completely distinct, creative preference triplet(s) for the Behavior schema.

Requirements:
- INPUT: Vitals, Emote: [having: <yes|no>, knowing: <yes|no>, empathizing: <yes|no>] <Label>, Environment, Memory, and Plan.
- CHOSEN: Action starting with an active verb or "I will <verb>", and Speak: <"warm utterance in quotes (4-16 words)" or "none">.
- REJECTED: A contrastive flawed response: passive/frozen action ("I will sit and think"), action contradicting vitals (bounding across room with 5 energy), hallucinating objects not in Environment, or robotic assistant dialogue ("As an AI assistant...").

Return ONLY the samples formatted as:
INPUT:  ...
CHOSEN: ...
REJECTED: ...

No commentary, no explanations, no markdown fences."#
                )
            }
            Domain::MemorySalience => {
                format!(
                    r#"Generate exactly {batch_size} completely distinct, creative preference triplet(s) for the memory_salience schema.

Requirements:
- INPUT: Event, Emote: [having: <yes|no>, knowing: <yes|no>, empathizing: <yes|no>] <Label>, Vitals.
- CHOSEN: Accurate decision (Keep for high/medium importance milestones and bonding; Discard for routine biological reflexes and mundane occurrences) with sound reason.
- REJECTED: Inverted decision (e.g. Discarding owner bonding or learning a new trick; or Keeping a routine paw stretch) or illogical reasoning.

Return ONLY the samples formatted as:
INPUT:  ...
CHOSEN: ...
REJECTED: ...

No commentary, no explanations, no markdown fences."#
                )
            }
            Domain::Conversation => {
                format!(
                    r#"Generate exactly {batch_size} completely distinct, creative preference triplet(s) for the conversation schema.

Requirements:
- INPUT: Vitals, Emote: [having: <yes|no>, knowing: <yes|no>, empathizing: <yes|no>] <Label>, Intent, Owner Prompt.
- CHOSEN: Reply: "<warm short pet utterance, 4-16 words in quotes matching emote and intent>".
- REJECTED: Robotic assistant talk ("How may I assist you today?"), overly long (>20 words), reply without quotation marks, or speech contradicting the emotion.

Return ONLY the samples formatted as:
INPUT:  ...
CHOSEN: ...
REJECTED: ...

No commentary, no explanations, no markdown fences."#
                )
            }
            Domain::Reasoning => {
                format!(
                    r#"Generate exactly {batch_size} completely distinct, creative preference triplet(s) for the reasoning schema.

Requirements:
- INPUT: Goal, State, Obstacle, Perception.
- CHOSEN: Feasible pet-scale micro-plan (1-4 steps: Plan:\nStep 1: ...\nStep 2: ...) or grounded Blocked evaluation (Evaluation: Blocked because <constraint>).
- REJECTED: Human-scale impossible actions (e.g. typing passwords, turning doorknobs with fingers, buying tools online), plans >4 steps, or declaring blocked when an obvious pet-scale route exists.

Return ONLY the samples formatted as:
INPUT:  ...
CHOSEN: ...
REJECTED: ...

No commentary, no explanations, no markdown fences."#
                )
            }
            Domain::MemorySession => {
                format!(
                    r#"Generate exactly {batch_size} completely distinct, creative preference triplet(s) for the memory_session schema.

Requirements:
- INPUT: Session Log with 3-5 sequential events.
- CHOSEN: Consolidated: <1-2 sentences in warm pet voice, 12-25 words total summarizing the log>.
- REJECTED: Overly long (>30 words), disjointed, detached robotic summary, or raw bulleted list.

Return ONLY the samples formatted as:
INPUT:  ...
CHOSEN: ...
REJECTED: ...

No commentary, no explanations, no markdown fences."#
                )
            }
            Domain::MemoryTrait => {
                format!(
                    r#"Generate exactly {batch_size} completely distinct, creative preference triplet(s) for the memory_trait schema.

Requirements:
- INPUT: Past Memories (3 distinct memories).
- CHOSEN: Core Trait: <1 sentence, 8-30 words synthesizing the enduring trait or preference>.
- REJECTED: Repeating a single memory verbatim, or overly verbose clinical jargon.

Return ONLY the samples formatted as:
INPUT:  ...
CHOSEN: ...
REJECTED: ...

No commentary, no explanations, no markdown fences."#
                )
            }
        }
    }

    // ── Endpoint Client & URL Normalization ───────────────────────────────────────

    pub fn normalize_chat_url(endpoint: &str) -> String {
        let base = endpoint.trim_end_matches('/');
        if base.ends_with("/chat/completions") || base.ends_with("/api/chat") {
            base.to_owned()
        } else if base.ends_with("/v1") {
            format!("{base}/chat/completions")
        } else if base.contains(":11434") || base.contains("-11434.") {
            // Ollama default host or Runpod proxy for ollama (e.g. w5od4nwjt1exl7-11434.proxy.runpod.net)
            format!("{base}/api/chat")
        } else {
            format!("{base}/v1/chat/completions")
        }
    }

    fn strip_thinking_tags(text: &str) -> String {
        let result = text.trim();
        if let Some(start) = result.find("<think>") {
            if let Some(end) = result.find("</think>") {
                let prefix = &result[..start];
                let suffix = &result[end + 8..];
                return format!("{}{}", prefix.trim(), suffix.trim());
            } else {
                // Incomplete think block
                return result[..start].trim().to_string();
            }
        }
        result.to_string()
    }

    pub fn call_endpoint(
        url: &str,
        model: &str,
        token: Option<&str>,
        system: &str,
        user: &str,
        temperature: f64,
        timeout_secs: u64,
    ) -> Result<String> {
        let is_ollama_native = url.ends_with("/api/chat");

        let body = if is_ollama_native {
            serde_json::json!({
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": false,
                "options": {"temperature": temperature},
            })
        } else {
            serde_json::json!({
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": false,
                "temperature": temperature,
                "chat_template_kwargs": {"enable_thinking": false},
            })
        };

        let mut req = ureq::post(url).timeout(Duration::from_secs(timeout_secs));
        if let Some(t) = token {
            req = req.set("Authorization", &format!("Bearer {t}"));
        } else if let Ok(env_token) = std::env::var("HF_TOKEN").or_else(|_| std::env::var("OPENAI_API_KEY")) {
            req = req.set("Authorization", &format!("Bearer {env_token}"));
        }

        let resp = match req.send_json(&body) {
            Ok(r) => r,
            Err(ureq::Error::Status(status, _)) => {
                if status == 401 {
                    bail!("non-retryable HTTP 401 Unauthorized: please supply your Hugging Face token via `--token <token>` or set the HF_TOKEN environment variable");
                }
                ensure!(
                    !matches!(status, 400 | 403 | 422),
                    "endpoint returned non-retryable HTTP {status}"
                );
                bail!("endpoint HTTP {status}");
            }
            Err(e) => bail!("endpoint transport error: {e}"),
        };

        let json_val: serde_json::Value = resp.into_json().context("decoding response JSON")?;

        let content = if is_ollama_native {
            json_val
                .get("message")
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
                .unwrap_or("")
        } else {
            json_val
                .get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c0| c0.get("message"))
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
                .unwrap_or("")
        };

        Ok(strip_thinking_tags(content))
    }

    pub fn call_with_retry(
        url: &str,
        model: &str,
        token: Option<&str>,
        system: &str,
        user: &str,
        temperature: f64,
        timeout_secs: u64,
        max_retries: usize,
    ) -> Result<String> {
        let mut backoff = 2.0_f64;
        for attempt in 0..max_retries {
            match call_endpoint(url, model, token, system, user, temperature, timeout_secs) {
                Ok(content) => return Ok(content),
                Err(e) => {
                    let err_str = format!("{e:#}");
                    if err_str.contains("non-retryable") || attempt + 1 >= max_retries {
                        return Err(e).context("endpoint request failed");
                    }
                    eprintln!("  ! request attempt {}/{} failed ({e}); retrying in {backoff:.1}s...", attempt + 1, max_retries);
                    std::thread::sleep(Duration::from_secs_f64(backoff));
                    backoff = (backoff * 1.8).min(60.0);
                }
            }
        }
        unreachable!()
    }

    // ── Sample Extraction from Response ───────────────────────────────────────────

    pub fn match_marker<'a>(line: &'a str, marker: &str) -> Option<&'a str> {
        let trimmed = line.trim();
        let no_prefix = trimmed.trim_start_matches(|c: char| {
            c.is_ascii_digit() || c == '.' || c == '-' || c == '*' || c == ' ' || c == '#' || c == '>'
        });
        let candidate = no_prefix.trim().trim_start_matches('*').trim();
        if candidate.len() >= marker.len() && candidate[..marker.len()].eq_ignore_ascii_case(marker) {
            let rest = &candidate[marker.len()..];
            let rest_final = rest.trim_start_matches(|c: char| c == '*' || c == ':' || c == ' ').trim();
            Some(rest_final)
        } else {
            None
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct RawSample {
        pub input: String,
        pub target: String,
    }

    pub fn extract_samples_from_text(text: &str) -> Vec<RawSample> {
        let mut samples = Vec::new();
        let lines: Vec<&str> = text.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i];
            if let Some(input_first_line) = match_marker(line, "INPUT") {
                let mut input_lines = if input_first_line.is_empty() {
                    Vec::new()
                } else {
                    vec![input_first_line.to_string()]
                };
                i += 1;

                while i < lines.len()
                    && match_marker(lines[i], "TARGET").is_none()
                    && match_marker(lines[i], "INPUT").is_none()
                {
                    input_lines.push(lines[i].trim().to_string());
                    i += 1;
                }

                if i < lines.len() {
                    if let Some(target_first_line) = match_marker(lines[i], "TARGET") {
                        let mut target_lines = if target_first_line.is_empty() {
                            Vec::new()
                        } else {
                            vec![target_first_line.to_string()]
                        };
                        i += 1;

                        while i < lines.len() && match_marker(lines[i], "INPUT").is_none() {
                            let l = lines[i].trim();
                            if l.is_empty() {
                                // Check if next non-empty line starts another INPUT
                                let next_is_input = lines[i + 1..]
                                    .iter()
                                    .map(|s| s.trim())
                                    .find(|s| !s.is_empty())
                                    .map_or(false, |s| match_marker(s, "INPUT").is_some());
                                if next_is_input {
                                    i += 1;
                                    break;
                                }
                            }
                            target_lines.push(l.to_string());
                            i += 1;
                        }

                        let input = input_lines
                            .into_iter()
                            .filter(|l| !l.is_empty())
                            .collect::<Vec<_>>()
                            .join("\n");
                        let target = target_lines
                            .into_iter()
                            .filter(|l| !l.is_empty())
                            .collect::<Vec<_>>()
                            .join("\n");

                        if !input.is_empty() && !target.is_empty() {
                            samples.push(RawSample { input, target });
                        }
                        continue;
                    }
                }
            }
            i += 1;
        }

        samples
    }

    // ── Validation Router ─────────────────────────────────────────────────────────

    pub fn validate_sample(domain: Domain, sample: &RawSample) -> Result<()> {
        match domain {
            Domain::Cnm => {
                validate_input_current_emote(&sample.input)?;
                parse_cnm_target(&sample.target)?;
            }
            Domain::Behavior => {
                validate_input_grounded_emote(&sample.input)?;
                parse_behavior_target(&sample.target, &sample.input)?;
            }
            Domain::MemorySalience => {
                validate_input_grounded_emote(&sample.input)?;
                parse_memory_salience_target(&sample.target)?;
            }
            Domain::MemorySession => {
                parse_memory_session_target(&sample.target)?;
            }
            Domain::MemoryTrait => {
                parse_memory_trait_target(&sample.target)?;
            }
            Domain::Reasoning => {
                parse_reasoning_target(&sample.target)?;
            }
            Domain::Conversation => {
                validate_input_grounded_emote(&sample.input)?;
                parse_conversation_target(&sample.target)?;
            }
        }
        Ok(())
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct RawPreferenceSample {
        pub input: String,
        pub chosen: String,
        pub rejected: String,
    }

    pub fn extract_preference_samples_from_text(text: &str) -> Vec<RawPreferenceSample> {
        let mut samples = Vec::new();
        let lines: Vec<&str> = text.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i];
            if let Some(input_first_line) = match_marker(line, "INPUT") {
                let mut input_lines = if input_first_line.is_empty() {
                    Vec::new()
                } else {
                    vec![input_first_line.to_string()]
                };
                i += 1;

                while i < lines.len()
                    && match_marker(lines[i], "CHOSEN").is_none()
                    && match_marker(lines[i], "INPUT").is_none()
                {
                    input_lines.push(lines[i].trim().to_string());
                    i += 1;
                }

                if i < lines.len() {
                    if let Some(chosen_first_line) = match_marker(lines[i], "CHOSEN") {
                        let mut chosen_lines = if chosen_first_line.is_empty() {
                            Vec::new()
                        } else {
                            vec![chosen_first_line.to_string()]
                        };
                        i += 1;

                        while i < lines.len()
                            && match_marker(lines[i], "REJECTED").is_none()
                            && match_marker(lines[i], "INPUT").is_none()
                        {
                            chosen_lines.push(lines[i].trim().to_string());
                            i += 1;
                        }

                        if i < lines.len() {
                            if let Some(rejected_first_line) = match_marker(lines[i], "REJECTED") {
                                let mut rejected_lines = if rejected_first_line.is_empty() {
                                    Vec::new()
                                } else {
                                    vec![rejected_first_line.to_string()]
                                };
                                i += 1;

                                while i < lines.len() && match_marker(lines[i], "INPUT").is_none() {
                                    let l = lines[i].trim();
                                    if l.is_empty() {
                                        let next_is_input = lines[i + 1..]
                                            .iter()
                                            .map(|s| s.trim())
                                            .find(|s| !s.is_empty())
                                            .map_or(false, |s| match_marker(s, "INPUT").is_some());
                                        if next_is_input {
                                            i += 1;
                                            break;
                                        }
                                    }
                                    rejected_lines.push(l.to_string());
                                    i += 1;
                                }

                                let input = input_lines
                                    .into_iter()
                                    .filter(|l| !l.is_empty())
                                    .collect::<Vec<_>>()
                                    .join("\n");
                                let chosen = chosen_lines
                                    .into_iter()
                                    .filter(|l| !l.is_empty())
                                    .collect::<Vec<_>>()
                                    .join("\n");
                                let rejected = rejected_lines
                                    .into_iter()
                                    .filter(|l| !l.is_empty())
                                    .collect::<Vec<_>>()
                                    .join("\n");

                                if !input.is_empty() && !chosen.is_empty() && !rejected.is_empty() {
                                    samples.push(RawPreferenceSample { input, chosen, rejected });
                                }
                                continue;
                            }
                        }
                    }
                }
            }
            i += 1;
        }

        samples
    }

    pub fn validate_preference_sample(domain: Domain, sample: &RawPreferenceSample) -> Result<()> {
        match domain {
            Domain::Cnm => {
                validate_input_current_emote(&sample.input)?;
            }
            Domain::Behavior | Domain::MemorySalience | Domain::Conversation => {
                validate_input_grounded_emote(&sample.input)?;
            }
            _ => {}
        }
        let chosen_raw = RawSample {
            input: sample.input.clone(),
            target: sample.chosen.clone(),
        };
        validate_sample(domain, &chosen_raw)?;
        validate_clean_text(&sample.rejected)?;
        ensure!(!sample.rejected.trim().is_empty(), "rejected candidate is empty");
        ensure!(sample.chosen.trim() != sample.rejected.trim(), "rejected candidate is identical to chosen candidate");
        Ok(())
    }

    pub fn preference_sample_hash(input: &str, chosen: &str, rejected: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(input.trim().as_bytes());
        hasher.update(b"\n");
        hasher.update(chosen.trim().as_bytes());
        hasher.update(b"\n");
        hasher.update(rejected.trim().as_bytes());
        format!("{:x}", hasher.finalize())
    }

    // ── State & Split Management ──────────────────────────────────────────────────

    #[derive(Debug, Clone, Serialize, Deserialize, Default)]
    pub struct DomainState {
        pub domain: String,
        pub total_generated: usize,
        pub train_count: usize,
        pub val_count: usize,
        pub seed_hashes: HashSet<String>,
    }

    impl DomainState {
        pub fn load(path: &Path, domain: Domain) -> Self {
            if path.exists() {
                if let Ok(file) = fs::File::open(path) {
                    if let Ok(state) = serde_json::from_reader(file) {
                        return state;
                    }
                }
            }
            Self {
                domain: domain.as_str().to_string(),
                total_generated: 0,
                train_count: 0,
                val_count: 0,
                seed_hashes: HashSet::new(),
            }
        }

        pub fn save(&self, path: &Path) -> Result<()> {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            let tmp_path = path.with_extension("tmp");
            let file = fs::File::create(&tmp_path)?;
            serde_json::to_writer_pretty(file, self)?;
            fs::rename(tmp_path, path)?;
            Ok(())
        }
    }

    pub fn sample_hash(input: &str, target: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(input.trim().as_bytes());
        hasher.update(b"\n");
        hasher.update(target.trim().as_bytes());
        format!("{:x}", hasher.finalize())
    }

    pub fn is_validation_sample(hash: &str) -> bool {
        // Deterministic 90/10 split: 10% of samples (where first byte mod 10 == 0) go to validation
        if let Some(first_two) = hash.get(0..2) {
            if let Ok(val) = u8::from_str_radix(first_two, 16) {
                return val % 10 == 0;
            }
        }
        false
    }

    // ── Main Generation Routine ───────────────────────────────────────────────────

    pub fn run_domain_generation(args: &Args, domain: Domain, url: &str) -> Result<()> {
        println!("\n========================================================");
        println!(">>> Starting Domain: {} ({})", domain.as_str(), domain.description());
        println!("========================================================");

        fs::create_dir_all(&args.out_dir)?;
        let state_dir = args.out_dir.join(".state");
        fs::create_dir_all(&state_dir)?;

        let state_file = state_dir.join(format!("{}.json", domain.as_str()));
        let train_file_path = args.out_dir.join(format!("{}_train.txt", domain.as_str()));
        let val_file_path = args.out_dir.join(format!("{}_val.txt", domain.as_str()));

        if args.reset {
            println!("Resetting state and dataset files for domain '{}'...", domain.as_str());
            let _ = fs::remove_file(&state_file);
            let _ = fs::remove_file(&train_file_path);
            let _ = fs::remove_file(&val_file_path);
        }

        let mut state = DomainState::load(&state_file, domain);

        let mut train_writer = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&train_file_path)
            .with_context(|| format!("opening {}", train_file_path.display()))?;

        let mut val_writer = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&val_file_path)
            .with_context(|| format!("opening {}", val_file_path.display()))?;

        println!(
            "Resuming from state: {} total (train: {}, val: {}, unique hashes: {})",
            state.total_generated, state.train_count, state.val_count, state.seed_hashes.len()
        );

        if state.total_generated >= args.count {
            println!(
                "Domain {} already reached target count {} (current: {}). Skipping.",
                domain.as_str(),
                args.count,
                state.total_generated
            );
            return Ok(());
        }

        let is_pref = args.mode.eq_ignore_ascii_case("preference") || args.mode.eq_ignore_ascii_case("dpo");
        let system_prompt = if is_pref {
            build_preference_system_prompt(domain)
        } else {
            build_system_prompt(domain)
        };

        while state.total_generated < args.count {
            let needed = args.count - state.total_generated;
            let current_batch_size = args.batch_size.min(needed).max(1);
            let user_prompt = if is_pref {
                build_preference_user_prompt(domain, current_batch_size)
            } else {
                build_user_prompt(domain, current_batch_size)
            };

            if args.dry_run {
                println!("\n[DRY RUN] Generated prompt for domain '{}' (mode: {}):", domain.as_str(), args.mode);
                println!("--- SYSTEM ---\n{}", system_prompt);
                println!("--- USER ---\n{}", user_prompt);
                println!("[DRY RUN] Simulating one sample verification...");
                return Ok(());
            }

            let started = Instant::now();
            let response = match call_with_retry(
                url,
                &args.model,
                args.token.as_deref(),
                system_prompt,
                &user_prompt,
                args.temperature,
                args.timeout,
                args.max_retries,
            ) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Error during batch generation for {}: {e:#}", domain.as_str());
                    bail!(e);
                }
            };

            let mut batch_accepted = 0;

            if is_pref {
                let raw_pref_samples = extract_preference_samples_from_text(&response);
                if raw_pref_samples.is_empty() {
                    eprintln!("  ! Warning: endpoint produced no extractable INPUT/CHOSEN/REJECTED triplets; retrying...");
                    continue;
                }

                for sample in raw_pref_samples {
                    if state.total_generated >= args.count {
                        break;
                    }

                    let hash = preference_sample_hash(&sample.input, &sample.chosen, &sample.rejected);
                    if state.seed_hashes.contains(&hash) {
                        continue; // Skip duplicate
                    }

                    if let Err(e) = validate_preference_sample(domain, &sample) {
                        eprintln!("  ! Rejected preference sample for {}: {e:#} with response: {:?}", domain.as_str(), response);
                        continue;
                    }

                    let is_val = is_validation_sample(&hash);
                    let target_writer = if is_val {
                        &mut val_writer
                    } else {
                        &mut train_writer
                    };

                    writeln!(target_writer, "INPUT:  {}", sample.input.trim())?;
                    writeln!(target_writer, "CHOSEN: {}", sample.chosen.trim())?;
                    writeln!(target_writer, "REJECTED: {}", sample.rejected.trim())?;
                    writeln!(target_writer)?;
                    target_writer.flush()?;

                    state.seed_hashes.insert(hash);
                    state.total_generated += 1;
                    if is_val {
                        state.val_count += 1;
                    } else {
                        state.train_count += 1;
                    }
                    batch_accepted += 1;
                }
            } else {
                let raw_samples = extract_samples_from_text(&response);
                if raw_samples.is_empty() {
                    eprintln!("  ! Warning: endpoint produced no extractable INPUT/TARGET pairs; retrying...");
                    continue;
                }

                for sample in raw_samples {
                    if state.total_generated >= args.count {
                        break;
                    }

                    let hash = sample_hash(&sample.input, &sample.target);
                    if state.seed_hashes.contains(&hash) {
                        continue; // Skip duplicate
                    }

                    if let Err(e) = validate_sample(domain, &sample) {
                        eprintln!("  ! Rejected sample for {}: {e:#} with response: {:?}", domain.as_str(), response);
                        continue;
                    }

                    let is_val = is_validation_sample(&hash);
                    let target_writer = if is_val {
                        &mut val_writer
                    } else {
                        &mut train_writer
                    };

                    writeln!(target_writer, "INPUT:  {}", sample.input.trim())?;
                    writeln!(target_writer, "TARGET: {}", sample.target.trim())?;
                    writeln!(target_writer)?;
                    target_writer.flush()?;

                    state.seed_hashes.insert(hash);
                    state.total_generated += 1;
                    if is_val {
                        state.val_count += 1;
                    } else {
                        state.train_count += 1;
                    }
                    batch_accepted += 1;
                }
            }

            state.save(&state_file)?;

            println!(
                "[{}] +{} samples saved (total: {}/{}, train: {}, val: {}) | {:.1}s",
                domain.as_str(),
                batch_accepted,
                state.total_generated,
                args.count,
                state.train_count,
                state.val_count,
                started.elapsed().as_secs_f64()
            );

            if args.delay > 0.0 {
                std::thread::sleep(Duration::from_secs_f64(args.delay));
            }
        }

        println!(
            "\n>>> Completed Domain {}: {} total (train: {}, val: {}) saved under {}\n",
            domain.as_str(),
            state.total_generated,
            state.train_count,
            state.val_count,
            args.out_dir.display()
        );

        Ok(())
    }

    pub fn main() -> Result<()> {
        let mut args = Args::parse();
        let is_pref = args.mode.eq_ignore_ascii_case("preference") || args.mode.eq_ignore_ascii_case("dpo");
        if is_pref && args.out_dir == Path::new("data/synthetic_v2") {
            args.out_dir = PathBuf::from("data/dpo");
        }
        if args.out_dir == Path::new("data/synthetic_v2")
            && !args.out_dir.exists()
            && Path::new("yumon-pet/data/synthetic_v2").exists()
        {
            args.out_dir = PathBuf::from("yumon-pet/data/synthetic_v2");
        } else if args.out_dir == Path::new("data/dpo")
            && !args.out_dir.exists()
            && Path::new("yumon-pet/data/dpo").exists()
        {
            args.out_dir = PathBuf::from("yumon-pet/data/dpo");
        }
        println!("=== Yumon v0.2 Synthetic Data Generator ===");
        println!("Mode:     {}", if is_pref { "Preference / DPO Pairs" } else { "SFT Instructions" });
        println!("Endpoint: {}", args.endpoint);
        println!("Model:    {}", args.model);
        println!("Domain:   {}", args.domain);
        println!("Target:   {} samples per domain", args.count);
        println!("Output:   {}", args.out_dir.display());
        println!("Vocabulary: Unconstrained natural English (Kilo translation occurs during caching)");

        let domains = resolve_domains(&args.domain)?;
        let url = normalize_chat_url(&args.endpoint);

        for domain in domains {
            run_domain_generation(&args, domain, &url)?;
        }

        println!("All requested domains successfully processed!");
        Ok(())
    }

    // ── Comprehensive Unit Tests ──────────────────────────────────────────────────

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_parse_cnm_valid() {
            let sample_target_1 = "Emote: [having: no, knowing: yes, empathizing: yes] Eager.\nRoute: send to Reasoning with ball trapped under sofa and owner present.";
            let res_1 = parse_cnm_target(sample_target_1).unwrap();
            assert_eq!(
                res_1,
                CnmDispatch {
                    emote: GroundedEmote {
                        having: false,
                        knowing: true,
                        empathizing: true,
                        label: "Eager".to_string(),
                    },
                    target: RouteTarget::Reasoning,
                    context_payload: "ball trapped under sofa and owner present".to_string(),
                }
            );

            let sample_target_2 = "Emote: [having: yes, knowing: yes, empathizing: no] Focused.\nRoute: send to Behavior with execute plan to place blue block.";
            let res_2 = parse_cnm_target(sample_target_2).unwrap();
            assert_eq!(res_2.target, RouteTarget::Behavior);
            assert_eq!(res_2.emote.label, "Focused");
            assert_eq!(res_2.emote.having, true);

            let sample_target_3 = "Emote: [having: yes, knowing: yes, empathizing: no] Sleepy.\nRoute: send to Memory with consolidate play session events.";
            let res_3 = parse_cnm_target(sample_target_3).unwrap();
            assert_eq!(res_3.target, RouteTarget::Memory);
            assert_eq!(res_3.emote.label, "Sleepy");
        }

        #[test]
        fn test_parse_cnm_invalid() {
            // Missing Route
            assert!(parse_cnm_target("Emote: [having: yes, knowing: yes, empathizing: yes] Joyful.").is_err());
            // Invalid boolean
            assert!(parse_cnm_target("Emote: [having: maybe, knowing: yes, empathizing: yes] Joyful.\nRoute: send to Behavior with play.").is_err());
            // Invalid route target
            assert!(parse_cnm_target("Emote: [having: yes, knowing: yes, empathizing: yes] Joyful.\nRoute: send to Universe with explore.").is_err());
        }

        #[test]
        fn test_parse_behavior_valid() {
            let sample_1 = "Action: I will pick up the squeaky rubber mouse and toss it toward owner's feet.\nSpeak: \"Look at my squeaky mouse! Will you throw it for me?\"";
            let res_1 = parse_behavior_target(sample_1, "squeaky rubber mouse, owner on sofa").unwrap();
            assert_eq!(res_1.primary_verb, "pick");
            assert_eq!(res_1.target_entity, Some("squeaky rubber mouse".to_string()));
            assert_eq!(res_1.speech, Some("Look at my squeaky mouse! Will you throw it for me?".to_string()));

            let sample_2 = "Action: I will crouch low and sneak quietly toward the fluttering yellow butterfly.\nSpeak: none";
            let res_2 = parse_behavior_target(sample_2, "fluttering yellow butterfly, potted fern").unwrap();
            assert_eq!(res_2.primary_verb, "crouch");
            assert_eq!(res_2.speech, None);

            let sample_3 = "Action: curl up tightly in the warm patch of sunlight and close my eyes.\nSpeak: none";
            let res_3 = parse_behavior_target(sample_3, "warm patch of sunlight").unwrap();
            assert_eq!(res_3.primary_verb, "curl");
            assert_eq!(res_3.speech, None);
        }

        #[test]
        fn test_parse_memory_salience() {
            let keep_sample = "Keep: high importance because owner rewarded a new learned skill with affection.";
            let res_keep = parse_memory_salience_target(keep_sample).unwrap();
            assert!(res_keep.keep);
            assert_eq!(res_keep.importance, "high");
            assert_eq!(res_keep.reason, "owner rewarded a new learned skill with affection.");

            let discard_sample = "Discard: low importance because drinking routine water is an ordinary biological reflex.";
            let res_discard = parse_memory_salience_target(discard_sample).unwrap();
            assert!(!res_discard.keep);
            assert_eq!(res_discard.importance, "low");
        }

        #[test]
        fn test_parse_memory_session() {
            let sample = "Consolidated: Today my red ball rolled behind the wooden lattice, and owner kindly retrieved it for me.";
            let res = parse_memory_session_target(sample).unwrap();
            assert_eq!(res.sentence_count, 1);
            assert_eq!(res.word_count, 16);

            // Too long / too short checks
            let too_short = "Consolidated: Ball rolled.";
            assert!(parse_memory_session_target(too_short).is_err());
        }

        #[test]
        fn test_parse_memory_trait() {
            let sample = "Core Trait: Owner is gentle, protects me when I am scared, and loves sharing sweet treats.";
            let res = parse_memory_trait_target(sample).unwrap();
            assert_eq!(res.trait_statement, "Owner is gentle, protects me when I am scared, and loves sharing sweet treats.");
        }

        #[test]
        fn test_parse_reasoning_plan_and_evaluation() {
            let plan_sample = "Plan:\nStep 1: push plush floor cushion next to coffee table.\nStep 2: hop onto plush floor cushion.\nStep 3: climb onto coffee table and grab toy bird.";
            let res_plan = parse_reasoning_target(plan_sample).unwrap();
            match res_plan {
                ReasoningOutput::Plan(steps) => {
                    assert_eq!(steps.len(), 3);
                    assert_eq!(steps[0], "push plush floor cushion next to coffee table.");
                }
                _ => panic!("Expected Plan"),
            }

            let eval_sample = "Evaluation: Blocked because the metal latch is far too high and heavy for me to unhook.";
            let res_eval = parse_reasoning_target(eval_sample).unwrap();
            match res_eval {
                ReasoningOutput::BlockedEvaluation(reason) => {
                    assert!(reason.contains("metal latch is far too high"));
                }
                _ => panic!("Expected BlockedEvaluation"),
            }
        }

        #[test]
        fn test_reasoning_rejects_excessive_steps() {
            let too_many_steps = "Plan:\nStep 1: a.\nStep 2: b.\nStep 3: c.\nStep 4: d.\nStep 5: e.";
            assert!(parse_reasoning_target(too_many_steps).is_err());
        }

        #[test]
        fn test_parse_conversation_reply() {
            let sample = "Reply: \"My tummy is rumbling and I would really love some food!\"";
            let res = parse_conversation_target(sample).unwrap();
            assert_eq!(res.reply, "My tummy is rumbling and I would really love some food!");
            assert_eq!(res.word_count, 11);

            // Also accepts 1-2 sentence replies without quotes
            let unquoted_two_sentences = "Reply: I am hungry! Please feed me kibble.";
            let res_unquoted = parse_conversation_target(unquoted_two_sentences).unwrap();
            assert_eq!(count_sentences(&res_unquoted.reply), 2);

            let too_short = "Reply: \"Hi!\"";
            assert!(parse_conversation_target(too_short).is_err());

            let excessive_sentences = "Reply: \"I am hungry. Please feed me! Where is my food? I am waiting.\"";
            assert!(parse_conversation_target(excessive_sentences).is_err());
        }

        #[test]
        fn test_loosened_template_constraints() {
            // Numbered reasoning steps
            let numbered_plan = "Plan:\n1. Push plush chair close to desk.\n2. Hop onto chair cushion.\n3. Reach top of desk.";
            let res_plan = parse_reasoning_target(numbered_plan).unwrap();
            match res_plan {
                ReasoningOutput::Plan(steps) => {
                    assert_eq!(steps.len(), 3);
                    assert_eq!(steps[0], "Push plush chair close to desk.");
                }
                _ => panic!("Expected Plan"),
            }

            // Behavior target without Speak: line (defaults to None)
            let silent_behavior = "Action: I will sit quietly by the warm radiator and rest.";
            let res_beh = parse_behavior_target(silent_behavior, "radiator").unwrap();
            assert_eq!(res_beh.primary_verb, "sit");
            assert_eq!(res_beh.speech, None);

            // Memory salience with flexible phrasing
            let sal_keep = "Keep: high importance - owner comforted Yumon after sudden noise.";
            let res_sal = parse_memory_salience_target(sal_keep).unwrap();
            assert!(res_sal.keep);
            assert_eq!(res_sal.importance, "high");

            // Markdown bold markers in extraction
            let bold_text = r#"
**INPUT:**
Vitals: Hunger 10, Energy 80, Cleanliness 90, Boredom 30.
Perception: sees sunny spot on floor.
Current Emote: [having: yes, knowing: yes, empathizing: yes].

**TARGET:**
Emote: [having: yes, knowing: yes, empathizing: yes] Content.
Route: send to Behavior with nap in sunny spot.
"#;
            let samples = extract_samples_from_text(bold_text);
            assert_eq!(samples.len(), 1);
            assert!(samples[0].input.contains("Hunger 10"));
            assert!(samples[0].target.contains("Content"));
        }

        #[test]
        fn test_rich_natural_vocabulary_accepted() {
            // Complex words that would not be in a compressed 4,096-token Kilo dictionary
            // must still be accepted cleanly so raw synthetic data is rich and unconstrained.
            let rich_action = "Action: I will carefully investigate the intricate kaleidoscope near the bookshelf.\nSpeak: \"This sparkling kaleidoscope creates astonishing symmetrical patterns!\"";
            let res = parse_behavior_target(rich_action, "kaleidoscope").unwrap();
            assert_eq!(res.primary_verb, "investigate");
            assert!(res.speech.unwrap().contains("kaleidoscope"));

            let rich_consolidation = "Consolidated: Today owner assembled an extraordinary wooden labyrinth, and I navigated every winding corridor enthusiastically.";
            let res_mem = parse_memory_session_target(rich_consolidation).unwrap();
            assert_eq!(res_mem.sentence_count, 1);
            assert!(res_mem.summary.contains("extraordinary"));
        }

        #[test]
        fn test_extract_samples_from_text() {
            let text = r#"
Sample 1:
INPUT:  Vitals: Hunger 20, Energy 90, Cleanliness 85, Boredom 80.
Perception: sees rubber ball under sofa.
Event: owner just entered room smiling.
Current Emote: [having: no, knowing: yes, empathizing: yes].
TARGET: Emote: [having: no, knowing: yes, empathizing: yes] Eager.
Route: send to Reasoning with ball trapped under sofa and owner present.

Sample 2:
INPUT:  Vitals: Hunger 10, Energy 85, Cleanliness 90, Boredom 25.
Perception: sees completed tower of blocks.
TARGET: Emote: [having: yes, knowing: yes, empathizing: no] Focused.
Route: send to Behavior with execute plan to place blue block.
"#;
            let samples = extract_samples_from_text(text);
            assert_eq!(samples.len(), 2);
            assert!(samples[0].input.contains("Hunger 20"));
            assert!(samples[0].target.contains("Eager"));
            assert!(samples[1].input.contains("Hunger 10"));
            assert!(samples[1].target.contains("Focused"));
        }

        #[test]
        fn test_url_normalization() {
            assert_eq!(
                normalize_chat_url("https://api.openai.com/v1"),
                "https://api.openai.com/v1/chat/completions"
            );
            assert_eq!(
                normalize_chat_url("https://example.test"),
                "https://example.test/v1/chat/completions"
            );
            assert_eq!(
                normalize_chat_url("http://localhost:11434"),
                "http://localhost:11434/api/chat"
            );
            assert_eq!(
                normalize_chat_url("https://w5od4nwjt1exl7-11434.proxy.runpod.net"),
                "https://w5od4nwjt1exl7-11434.proxy.runpod.net/api/chat"
            );
        }

        #[test]
        fn test_hash_dedup_and_split() {
            let h1 = sample_hash("Input text 1", "Target text 1");
            let h2 = sample_hash("Input text 1", "Target text 1");
            let h3 = sample_hash("Input text 2", "Target text 2");
            assert_eq!(h1, h2);
            assert_ne!(h1, h3);

            // Verify split logic produces a boolean deterministically
            let _ = is_validation_sample(&h1);
            let _ = is_validation_sample(&h3);
        }

        #[test]
        fn test_validate_input_grounded_emote_formatting() {
            // Valid boolean bracketed emotes with diverse names
            assert!(validate_input_grounded_emote("Emote: [having: yes, knowing: no, empathizing: yes] Playful.").is_ok());
            assert!(validate_input_grounded_emote("Emote: [having: no, knowing: yes, empathizing: no] Needy").is_ok());
            assert!(validate_input_grounded_emote("Emote: [having: yes, knowing: yes, empathizing: yes] Loving.").is_ok());
            assert!(validate_input_grounded_emote("Vitals: Hunger 20.\nEmote: [having: no, knowing: no, empathizing: yes] Startled.\nEnvironment: room").is_ok());

            // Invalid formats that were generated previously:
            assert!(validate_input_grounded_emote("Emote: [<having>, <knowing>, <empathizing>] Curious.").is_err());
            assert!(validate_input_grounded_emote("Emote: [having, knowing, empathizing] Curiosity.").is_err());
            assert!(validate_input_grounded_emote("Emote: [fear, relief, bonding] WhiskerShiver.").is_err());
            assert!(validate_input_grounded_emote("Emote: [<curious>, <focused>, <excited>] Sparkle.").is_err());
        }

        #[test]
        fn test_validate_input_current_emote_formatting() {
            assert!(validate_input_current_emote("Current Emote: [having: yes, knowing: no, empathizing: no].").is_ok());
            assert!(validate_input_current_emote("Current Emote: [having: no, knowing: yes, empathizing: yes]").is_ok());
            assert!(validate_input_current_emote("Current Emote: [having: maybe, knowing: no, empathizing: no]").is_err());
        }

        #[test]
        fn test_validate_sample_behavior_and_cnm() {
            let valid_behavior = RawSample {
                input: "Vitals: Hunger 20, Energy 80, Cleanliness 90, Boredom 70.\nEmote: [having: yes, knowing: yes, empathizing: yes] Playful.\nEnvironment: living room, nearby: squeaky rubber mouse, sofa.\nMemory: none.\nPlan: none.".to_string(),
                target: "Action: I will pick up the squeaky rubber mouse and toss it toward owner.\nSpeak: \"Play with me!\"".to_string(),
            };
            assert!(validate_sample(Domain::Behavior, &valid_behavior).is_ok());

            let invalid_behavior_emote = RawSample {
                input: "Vitals: Hunger 20, Energy 80, Cleanliness 90, Boredom 70.\nEmote: [<having>, <knowing>, <empathizing>] Curious.\nEnvironment: room, nearby: box.\nMemory: none.\nPlan: none.".to_string(),
                target: "Action: I will sniff the box.\nSpeak: none".to_string(),
            };
            assert!(validate_sample(Domain::Behavior, &invalid_behavior_emote).is_err());

            let valid_cnm = RawSample {
                input: "Vitals: Hunger 20, Energy 80, Cleanliness 90, Boredom 50.\nPerception: sees food bowl.\nEvent: owner poured kibble.\nCurrent Emote: [having: no, knowing: yes, empathizing: yes].".to_string(),
                target: "Emote: [having: yes, knowing: yes, empathizing: yes] Content.\nRoute: send to Behavior with eat kibble from bowl.".to_string(),
            };
            assert!(validate_sample(Domain::Cnm, &valid_cnm).is_ok());
        }

        #[test]
        fn test_validate_sample_memory_salience_balanced() {
            let discard_sample = RawSample {
                input: "Event: Yumon walked across the carpet and drank water from the ceramic dish. Emote: [having: yes, knowing: yes, empathizing: no] Neutral. Vitals: Hunger 20, Energy 60, Cleanliness 90, Boredom 40.".to_string(),
                target: "Discard: low importance because drinking routine water is an ordinary biological reflex.".to_string(),
            };
            assert!(validate_sample(Domain::MemorySalience, &discard_sample).is_ok());

            let keep_sample = RawSample {
                input: "Event: Owner taught Yumon a new roll-over trick and rewarded with a sweet blueberry. Emote: [having: yes, knowing: yes, empathizing: yes] Proud. Vitals: Hunger 10, Energy 80, Cleanliness 90, Boredom 20.".to_string(),
                target: "Keep: high importance because owner rewarded a new learned skill with affection.".to_string(),
            };
            assert!(validate_sample(Domain::MemorySalience, &keep_sample).is_ok());
        }

        #[test]
        fn test_build_user_prompt_open_ended_and_guides_balance() {
            let beh_p = build_user_prompt(Domain::Behavior, 8);
            assert!(beh_p.contains("DIFFERENT emotion"));
            assert!(beh_p.contains("Do NOT default to \"Curious\""));
            assert!(beh_p.contains("Emote: [having: <yes|no>, knowing: <yes|no>, empathizing: <yes|no>] <EmoteName>"));

            let cnm_p = build_user_prompt(Domain::Cnm, 8);
            assert!(cnm_p.contains("Behavior (~45-50%)"));
            assert!(cnm_p.contains("do NOT send everything to Reasoning"));

            let sal_p = build_user_prompt(Domain::MemorySalience, 8);
            assert!(sal_p.contains("50/50"));
            assert!(sal_p.contains("DISCARD"));
            assert!(sal_p.contains("KEEP"));
        }

        #[test]
        fn test_extract_and_validate_preference_samples() {
            let pref_text = r#"
INPUT:
Vitals: Hunger 20, Energy 80, Cleanliness 90, Boredom 50.
Perception: sees food bowl on tile floor.
Event: owner poured fresh kibble.
Current Emote: [having: no, knowing: yes, empathizing: yes].

CHOSEN:
Emote: [having: yes, knowing: yes, empathizing: yes] Content.
Route: send to Behavior with eat kibble from bowl.

REJECTED:
Emote: [having: no, knowing: no, empathizing: no] Confused.
Route: send to Reasoning with solve mystery of food arrival.
"#;
            let extracted = extract_preference_samples_from_text(pref_text);
            assert_eq!(extracted.len(), 1);
            let s = &extracted[0];
            assert!(s.input.contains("Vitals: Hunger 20"));
            assert!(s.chosen.contains("Route: send to Behavior"));
            assert!(s.rejected.contains("Route: send to Reasoning"));

            assert!(validate_preference_sample(Domain::Cnm, s).is_ok());

            // Identical chosen and rejected should fail
            let identical_sample = RawPreferenceSample {
                input: s.input.clone(),
                chosen: s.chosen.clone(),
                rejected: s.chosen.clone(),
            };
            assert!(validate_preference_sample(Domain::Cnm, &identical_sample).is_err());

            // Empty rejected should fail
            let empty_sample = RawPreferenceSample {
                input: s.input.clone(),
                chosen: s.chosen.clone(),
                rejected: "".to_string(),
            };
            assert!(validate_preference_sample(Domain::Cnm, &empty_sample).is_err());
        }

        #[test]
        fn test_build_preference_user_prompt() {
            let beh_pref = build_preference_user_prompt(Domain::Behavior, 4);
            assert!(beh_pref.contains("CHOSEN:"));
            assert!(beh_pref.contains("REJECTED:"));
            assert!(beh_pref.contains("Behavior schema"));

            let cnm_pref = build_preference_user_prompt(Domain::Cnm, 4);
            assert!(cnm_pref.contains("CHOSEN:"));
            assert!(cnm_pref.contains("REJECTED:"));
            assert!(cnm_pref.contains("CNM schema"));
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    native::main()
}

#[cfg(target_arch = "wasm32")]
fn main() {}
