//! A desktop Product Hunt companion with cached products and local commentary.
#![recursion_limit = "256"]

#[cfg(not(target_arch = "wasm32"))]
#[path = "rss/mod.rs"]
mod rss;

#[cfg(not(target_arch = "wasm32"))]
mod desktop {
    use super::rss::{self, Entry, HISTORY_LIMIT, POLL_INTERVAL, product_hunt};
    use anyhow::{Context, Result};
    use burn::backend::Wgpu;
    use chrono::{DateTime, Local, Utc};
    use clap::{Parser, ValueEnum};
    use serde::{Deserialize, Serialize};
    use std::{
        path::PathBuf,
        sync::mpsc,
        thread,
        time::{Duration, Instant},
    };
    use tao::{
        dpi::{LogicalSize, PhysicalPosition},
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy},
        window::WindowBuilder,
    };
    use wry::WebViewBuilder;
    use yumon_pet::brain::{
        bpe::TokenizerKind,
        decompressor::NGramDecompressor,
        decoder_model::YumonDecBrain,
        model::YumonBrain,
        moe_model::YumonMoeBrain,
        samples::{TrainingStage, language_prompt},
        xlstm_model::YumonXLstmBrain,
        yumon_talk,
    };
    use yumon_pet::vision::{
        paint_studio::{GalleryItem, PaintConfig, PaintPaths, PassProgress},
        PaintStudio,
    };

    type BackendType = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;

    #[derive(Clone, Copy, ValueEnum)]
    enum Architecture {
        DecoderOnly,
        Moe,
        Xlstm,
        EncoderDecoder,
    }

    const EMBEDDED_MODEL_BYTES: &[u8] = include_bytes!("../../assets/portal_model/model.bin");
    const EMBEDDED_TOKENIZER_BYTES: &[u8] = include_bytes!("../../assets/portal_model/tokenizer.json");
    const EMBEDDED_METADATA_STR: &str = include_str!("../../assets/portal_model/metadata.json");
    const EMBEDDED_VOCAB_STR: &str = include_str!("../../assets/portal_model/kilo_talk_vocab.json");

    #[derive(Parser)]
    #[command(about = "Yumon Portal: daily Product Hunt ideas with local commentary")]
    struct Args {
        /// Optional path to custom checkpoint directory (defaults to embedded model if omitted).
        #[arg(long)]
        checkpoint: Option<String>,
        #[arg(long, value_enum, default_value = "decoder-only")]
        architecture: Architecture,
        /// JSON history location (defaults to YumonRSS's local application data).
        #[arg(long)]
        history: Option<PathBuf>,
        /// Daily product cache (defaults to producthunt-cache.json beside history).
        #[arg(long)]
        cache: Option<PathBuf>,
        /// Print today's cached Product Hunt descriptions without a model or window.
        #[arg(long)]
        check_feeds: bool,
        /// Enable Yumon Talk bidirectional translation (English -> Yumon Talk prompt, Yumon Talk -> English reply).
        #[arg(long)]
        yumon_talk: bool,
        /// Optional path to Yumon Talk vocabulary JSON (defaults to embedded vocabulary if omitted).
        #[arg(long)]
        yumon_talk_vocab: Option<PathBuf>,
        /// Optional path to N-gram statistical decompressor model (falls back to embedded vocabulary).
        #[arg(long)]
        decompressor: Option<PathBuf>,
        /// Disable decompressor and output raw Yumon Talk tokens.
        #[arg(long)]
        no_decompressor: bool,
        /// Pass 1 checkpoint for art studio
        #[arg(long)]
        paint_pass1: Option<PathBuf>,
        /// Pass 2 checkpoint for art studio
        #[arg(long)]
        paint_pass2: Option<PathBuf>,
        /// Pass 3 checkpoint for art studio
        #[arg(long)]
        paint_pass3: Option<PathBuf>,
        /// Pass 4 checkpoint for art studio (64x64 details)
        #[arg(long)]
        paint_pass4: Option<PathBuf>,
        /// Pass 5 Neural Upscaler checkpoint for art studio
        #[arg(long)]
        paint_upscaler: Option<PathBuf>,
        /// Pass 6 Colorizer checkpoint for art studio
        #[arg(long)]
        paint_colorizer: Option<PathBuf>,
        /// Vocabulary JSON for art studio
        #[arg(long)]
        paint_vocab: Option<PathBuf>,
    }

    #[derive(Clone, Serialize)]
    struct ViewState {
        entries: Vec<Entry>,
        chat: Vec<ChatMessage>,
        chat_busy: bool,
        chat_status: String,
        status: String,
        busy: bool,
        can_refresh: bool,
        next_check: Option<DateTime<Utc>>,
        paint_status: String,
        paint_busy: bool,
    }

    #[derive(Clone, Serialize, Deserialize)]
    struct ChatMessage {
        role: &'static str,
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        citizen_name: Option<String>,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct VitalsData {
        pub hunger: u32,
        pub energy: u32,
        pub cleanliness: u32,
        pub boredom: u32,
    }

    impl Default for VitalsData {
        fn default() -> Self {
            Self {
                hunger: 20,
                energy: 85,
                cleanliness: 90,
                boredom: 25,
            }
        }
    }

    impl VitalsData {
        pub fn compute_grounded_emote(&self) -> EmoteData {
            let having = self.hunger < 60 && self.energy > 20;
            let knowing = self.boredom < 70;
            let empathizing = self.cleanliness > 30;

            let name = match (having, knowing, empathizing) {
                (true, true, true) => {
                    if self.energy > 75 && self.boredom > 40 {
                        "Playful"
                    } else if self.energy < 40 {
                        "Sleepy"
                    } else {
                        "Content"
                    }
                }
                (true, true, false) => {
                    if self.energy < 30 {
                        "Fatigue"
                    } else {
                        "Focused"
                    }
                }
                (true, false, true) => "Curious",
                (true, false, false) => "Intrigued",
                (false, true, true) => {
                    if self.hunger >= 70 {
                        "Hungry"
                    } else {
                        "Needy"
                    }
                }
                (false, true, false) => {
                    if self.hunger >= 70 {
                        "Hungry"
                    } else if self.cleanliness <= 30 {
                        "Uncomfortable"
                    } else {
                        "Distressed"
                    }
                }
                (false, false, true) => "Startled",
                (false, false, false) => {
                    if self.hunger >= 80 {
                        "Starving"
                    } else if self.energy <= 20 {
                        "Exhausted"
                    } else {
                        "Perplexed"
                    }
                }
            };

            EmoteData {
                having,
                knowing,
                empathizing,
                name: name.to_string(),
            }
        }
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct EmoteData {
        pub having: bool,
        pub knowing: bool,
        pub empathizing: bool,
        pub name: String,
    }

    impl Default for EmoteData {
        fn default() -> Self {
            Self {
                having: true,
                knowing: true,
                empathizing: true,
                name: "Content".to_string(),
            }
        }
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct BehaviorAction {
        pub action: String,
        pub primary_verb: String,
        pub target: Option<String>,
        pub speech: Option<String>,
    }

    enum WorkerRequest {
        Refresh,
        Chat {
            message: String,
            citizen_id: usize,
            citizen_name: String,
            vitals: Option<VitalsData>,
            emote: Option<EmoteData>,
        },
        ClearChat,
        SimDecide {
            citizen_id: usize,
            citizen_name: String,
            vitals: VitalsData,
            emote: EmoteData,
            environment: String,
            memory: Option<String>,
            plan: Option<String>,
        },
    }

    #[derive(Debug, Deserialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    enum Ipc {
        Ready,
        Refresh,
        Drag,
        Close,
        Open { link: String },
        Chat {
            message: String,
            citizen_id: Option<usize>,
            citizen_name: Option<String>,
            vitals: Option<VitalsData>,
            emote: Option<EmoteData>,
        },
        ClearChat,
        SimDecide {
            citizen_id: usize,
            citizen_name: Option<String>,
            vitals: VitalsData,
            emote: EmoteData,
            environment: String,
            memory: Option<String>,
            plan: Option<String>,
        },
        Paint {
            prompt: String,
            passes: Option<usize>,
            temperature: Option<f32>,
        },
        LoadGallery,
        GetFullImage { path: String },
        OpenArtFile { path: String },
    }

    enum RssEvent {
        State(ViewState),
        Ui(Ipc),
        SimAction {
            citizen_id: usize,
            action: String,
            primary_verb: String,
            target: Option<String>,
            speech: Option<String>,
        },
        ChatReply {
            citizen_id: usize,
            citizen_name: String,
            text: String,
        },
        PaintProgress(PassProgress),
        PaintComplete {
            item: GalleryItem,
            final_data_url: String,
        },
        PaintError(String),
        GalleryLoaded(Vec<GalleryItem>),
        FullImageLoaded {
            path: String,
            data_url: String,
        },
    }

    enum PaintJob {
        Generate(PaintConfig),
        RefreshGallery,
        GetFullImage { path: String },
    }

    #[derive(Clone, Debug)]
    enum PromptKind {
        Product,
        Chat {
            vitals: Option<VitalsData>,
            emote: Option<EmoteData>,
            intent: Option<String>,
        },
        Simulation {
            vitals: VitalsData,
            emote: EmoteData,
            environment: String,
            memory: Option<String>,
            plan: Option<String>,
        },
    }

    type Responder = Box<dyn Fn(&str, PromptKind, &[ChatMessage]) -> Result<String>>;

    /// Keep room for BOS, the Language separator, and reply tokens.
    /// Trim at UTF-8 boundaries and re-encode, since BPE merges can change at a cut.
    fn comment_prompt(title: &str, context: usize, tokenizer: &TokenizerKind) -> Result<String> {
        let reply_tokens = (context / 4).clamp(8, 256);
        let sep_len = tokenizer.encode(yumon_pet::brain::samples::language_separator_for(tokenizer)).len();
        let overhead = 1 + sep_len + reply_tokens;
        let budget = context
            .checked_sub(overhead)
            .context("Checkpoint context is too short")?;
        let human_prefix = yumon_pet::brain::samples::human_prefix_for(tokenizer);
        let human_prefix_len = tokenizer.encode(human_prefix).len();
        let title_budget = budget.checked_sub(human_prefix_len).context("Checkpoint context is too short")?;
        let prompt = fit_prompt(title, title_budget, |text| tokenizer.encode(text).len())?;
        Ok(format!("{human_prefix}{prompt}"))
    }

    fn fit_prompt(
        title: &str,
        budget: usize,
        token_count: impl Fn(&str) -> usize,
    ) -> Result<String> {
        let mut headline = title.to_owned();
        loop {
            anyhow::ensure!(
                !headline.trim().is_empty(),
                "Checkpoint context cannot fit a product description"
            );
            let prompt = format!("Your thoughts on {headline}?");
            if token_count(&prompt) <= budget {
                return Ok(prompt);
            }
            headline.pop();
        }
    }

    fn fit_chat_prompt(
        message: &str,
        budget: usize,
        token_count: impl Fn(&str) -> usize,
    ) -> Result<String> {
        let mut prompt = message.trim().to_owned();
        while token_count(&prompt) > budget {
            prompt.pop();
        }
        anyhow::ensure!(
            !prompt.trim().is_empty(),
            "Checkpoint context cannot fit your message"
        );
        Ok(prompt)
    }

    pub fn strip_outer_quotes(s: &str) -> &str {
        let trimmed = s.trim();
        let trimmed = trimmed.strip_prefix('"').unwrap_or(trimmed);
        let trimmed = trimmed.strip_suffix('"').unwrap_or(trimmed);
        let trimmed = trimmed.strip_prefix('“').unwrap_or(trimmed);
        let trimmed = trimmed.strip_suffix('”').unwrap_or(trimmed);
        let trimmed = trimmed.strip_prefix('\'').unwrap_or(trimmed);
        let trimmed = trimmed.strip_suffix('\'').unwrap_or(trimmed);
        trimmed.trim()
    }

    pub fn parse_chat_response_loose(raw: &str) -> String {
        let trimmed = raw.trim();
        let stripped = if let Some(idx) = trimmed.to_lowercase().find("reply:") {
            &trimmed[idx + 6..]
        } else if let Some(idx) = trimmed.to_lowercase().find("speak:") {
            &trimmed[idx + 6..]
        } else if let Some(idx) = trimmed.to_lowercase().find("yumon:") {
            &trimmed[idx + 6..]
        } else {
            trimmed
        };
        let stripped = stripped.trim();
        let first_turn = stripped.split("\nHuman:")
            .next().unwrap_or(stripped)
            .split("\nperson:")
            .next().unwrap_or(stripped)
            .split("\nINPUT:")
            .next().unwrap_or(stripped)
            .split("\nREJECTED:")
            .next().unwrap_or(stripped)
            .split("\nCHOSEN:")
            .next().unwrap_or(stripped)
            .trim();

        let unquoted = strip_outer_quotes(first_turn);
        if unquoted.is_empty() {
            first_turn.to_string()
        } else {
            unquoted.to_string()
        }
    }

    const ACTION_VERBS: &[&str] = &[
        "eat", "feed", "graze", "snack", "drink", "taste",
        "play", "chase", "bounce", "catch", "toss", "bat", "roll", "tumble",
        "pick", "grab", "carry",
        "swim", "glide", "dive", "paddle", "dash", "sprint", "run", "hop", "jump",
        "rest", "sleep", "nap", "curl", "relax", "doze", "sprawl", "sit", "lie",
        "explore", "inspect", "sniff", "look", "peer", "watch", "track", "peek",
        "tend", "groom", "clean", "wash", "brush", "shake", "wipe",
        "build", "stack", "arrange", "gather", "dig",
        "visit", "nudge", "approach", "crouch", "rub",
    ];

    const TARGET_KEYWORDS: &[&str] = &[
        "flake", "flakes", "food", "feed", "treat", "snack",
        "ball", "toy", "mouse", "blocks", "block", "pebble",
        "castle", "caves", "cave", "stone", "stones", "rock", "rocks", "kelp", "plant", "plants", "coral", "driftwood", "water", "current",
        "tree", "bench", "fountain", "garden", "flower", "flowers", "house", "cafe", "bakery", "shop", "telescope", "sofa", "rug", "bed", "door", "porch", "window", "cushion",
    ];

    pub fn extract_action_verb(action_text: &str) -> String {
        let lower = action_text.to_lowercase();
        let words: Vec<&str> = lower.split(|c: char| !c.is_alphabetic()).filter(|s| !s.is_empty()).collect();
        for word in words {
            if ACTION_VERBS.contains(&word) {
                return word.to_string();
            }
        }
        if lower.contains("swim") { return "swim".into(); }
        if lower.contains("eat") { return "eat".into(); }
        if lower.contains("play") { return "play".into(); }
        if lower.contains("rest") || lower.contains("sleep") { return "rest".into(); }
        if lower.contains("build") || lower.contains("stack") { return "build".into(); }
        if lower.contains("clean") || lower.contains("groom") { return "tend".into(); }
        "explore".into()
    }

    pub fn extract_target_entity(action_text: &str) -> Option<String> {
        let lower = action_text.to_lowercase();
        let words: Vec<&str> = lower.split(|c: char| !c.is_alphanumeric()).filter(|s| !s.is_empty()).collect();
        for word in words {
            if TARGET_KEYWORDS.contains(&word) {
                return Some(word.to_string());
            }
        }
        None
    }

    pub fn parse_behavior_response_loose(raw: &str) -> BehaviorAction {
        let mut action_desc = String::new();
        let mut speech_str = None;

        for line in raw.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let lower = trimmed.to_lowercase();
            if lower.starts_with("action:") {
                action_desc = trimmed["action:".len()..].trim().to_string();
            } else if lower.starts_with("speak:") {
                let s = trimmed["speak:".len()..].trim();
                let s_clean = strip_outer_quotes(s).trim();
                if !s_clean.eq_ignore_ascii_case("none") && !s_clean.is_empty() {
                    speech_str = Some(s_clean.to_string());
                }
            } else if action_desc.is_empty()
                && !lower.starts_with("vitals:")
                && !lower.starts_with("emote:")
                && !lower.starts_with("environment:")
                && !lower.starts_with("memory:")
                && !lower.starts_with("plan:")
            {
                action_desc = trimmed.to_string();
            }
        }

        if action_desc.is_empty() {
            action_desc = raw.trim().lines().next().unwrap_or("I will explore the world.").trim().to_string();
        }
        action_desc = strip_outer_quotes(&action_desc).to_string();
        if action_desc.is_empty() {
            action_desc = "I will explore the surroundings.".to_string();
        }

        let verb = extract_action_verb(&action_desc);
        let target = extract_target_entity(&action_desc);

        BehaviorAction {
            action: action_desc,
            primary_verb: verb,
            target,
            speech: speech_str,
        }
    }

    pub fn resolve_default_checkpoint(explicit: Option<&str>) -> Option<String> {
        if let Some(path) = explicit {
            return Some(path.to_string());
        }
        let candidates = [
            "checkpoints/yumon-dpo-mamba2-medium"
            // "checkpoints/yumon-v2-dpo/code_256h_16l_8a_512ff_256len_b2_Moe_e8_k2_f32_tied",
            // "checkpoints/yumon-v2-dpo",
            // "checkpoints/yumon-v2-sft/code_256h_16l_8a_512ff_256len_b2_Moe_e8_k2_f32_tied",
            // "checkpoints/yumon-v2-sft",
        ];
        for candidate in candidates {
            let p = PathBuf::from(candidate);
            if p.join("model.bin").exists() && p.join("metadata.json").exists() {
                return Some(candidate.to_string());
            }
            if p.is_dir() {
                if let Ok(entries) = std::fs::read_dir(&p) {
                    for entry in entries.flatten() {
                        let sub = entry.path();
                        if sub.is_dir() && sub.join("model.bin").exists() && sub.join("metadata.json").exists() {
                            return Some(sub.to_str().unwrap().to_string());
                        }
                    }
                }
            }
        }
        None
    }

    fn answer_chat(
        state: &mut ViewState,
        message: String,
        _citizen_id: usize,
        citizen_name: String,
        vitals: Option<VitalsData>,
        emote: Option<EmoteData>,
        responder: &Responder,
    ) -> Option<String> {
        let kind = PromptKind::Chat {
            vitals,
            emote,
            intent: Some("Chat".to_string()),
        };
        let res = match responder(&message, kind, &state.chat) {
            Ok(reply) => {
                let text = reply.clone();
                state.chat.push(ChatMessage {
                    role: "yumon",
                    text: reply,
                    citizen_name: Some(citizen_name),
                });
                state.chat_status = "Say hello or share an idea.".into();
                Some(text)
            }
            Err(error) => {
                state.chat.push(ChatMessage {
                    role: "error",
                    text: format!("Could not reply: {error:#}"),
                    citizen_name: None,
                });
                state.chat_status = "Try sending your message again.".into();
                None
            }
        };
        if state.chat.len() > HISTORY_LIMIT {
            state.chat.drain(..state.chat.len() - HISTORY_LIMIT);
        }
        state.chat_busy = false;
        res
    }

    fn chat_prompt(
        message: &str,
        history: &[ChatMessage],
        context: usize,
        tokenizer: &TokenizerKind,
        use_yumon_talk: bool,
    ) -> Result<String> {
        // Reserve up to 256 reply tokens (including EOS), with room for older checkpoints.
        let reply_tokens = (context / 4).clamp(8, 256);
        let sep_tokens = tokenizer.encode(yumon_pet::brain::samples::language_separator_for(tokenizer)).len();
        let budget = context
            .checked_sub(1 + sep_tokens + reply_tokens)
            .context("Checkpoint context is too short")?;
        let human_prefix = yumon_pet::brain::samples::human_prefix_for(tokenizer);
        let human_prefix_len = tokenizer.encode(human_prefix).len();
        let message_budget = budget.checked_sub(human_prefix_len).context("Checkpoint context is too short")?;
        let message = fit_chat_prompt(message, message_budget, |s| tokenizer.encode(s).len())?;

        let translated_memories: Vec<(String, String)> = if use_yumon_talk {
            history.windows(2).filter_map(|pair| {
                if pair[0].role == "user" && pair[1].role == "yumon" {
                    let u = yumon_pet::brain::yumon_talk::english_to_yumon(&pair[0].text).unwrap_or_else(|_| pair[0].text.clone());
                    let b = yumon_pet::brain::yumon_talk::english_to_yumon(&pair[1].text).unwrap_or_else(|_| pair[1].text.clone());
                    Some((u, b))
                } else {
                    None
                }
            }).collect()
        } else {
            history.windows(2).filter_map(|pair| {
                if pair[0].role == "user" && pair[1].role == "yumon" {
                    Some((pair[0].text.clone(), pair[1].text.clone()))
                } else {
                    None
                }
            }).collect()
        };

        let memories = translated_memories.iter().map(|(u, b)| (u.as_str(), b.as_str()));
        language_prompt(&message, memories, tokenizer, budget)
            .context("Checkpoint context cannot fit your message")
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn tokenizer() -> TokenizerKind {
            TokenizerKind::Bpe(yumon_pet::brain::bpe::BpeTokenizer::load("yumon_bpe").unwrap())
        }

        fn chat(role: &'static str, text: &str) -> ChatMessage {
            ChatMessage {
                role,
                text: text.into(),
                citizen_name: None,
            }
        }

        #[test]
        fn chat_memories_match_training_and_exclude_failures_and_pending_messages() {
            let tokenizer = tokenizer();
            let history = vec![
                chat("user", "My favorite color is blue."),
                chat("yumon", "I like blue too."),
                chat("user", "A failed question."),
                chat("error", "Could not reply: GPU unavailable"),
                chat("user", "Remember it?"),
                chat("yumon", "Yes, blue."),
                chat("user", "What color?"),
            ];
            let prompt = chat_prompt("What color?", &history, 256, &tokenizer, false).unwrap();
            assert_eq!(
                prompt,
                "Human: My favorite color is blue.\nYumon: I like blue too.\nHuman: Remember it?\nYumon: Yes, blue.\nHuman: What color?"
            );
            assert_eq!(
                chat_prompt("What color?", &[], 256, &tokenizer, false).unwrap(),
                "Human: What color?"
            );
            assert_eq!(
                chat_prompt("What color?", &history[2..4], 256, &tokenizer, false).unwrap(),
                "Human: What color?"
            );
            assert!(1 + tokenizer.encode(&prompt).len() + tokenizer.encode(yumon_pet::brain::samples::language_separator_for(&tokenizer)).len() + 64 <= 256);
        }

        #[test]
        fn chat_budget_keeps_recent_complete_turns_and_reserves_reply_space() {
            let tokenizer = tokenizer();
            let history = vec![
                chat("user", &"An old memory with many words. ".repeat(100)),
                chat("yumon", "An old answer."),
                chat("user", "My favorite color is blue."),
                chat("yumon", "I like blue too."),
            ];
            let prompt = chat_prompt("What color?", &history, 256, &tokenizer, false).unwrap();
            assert_eq!(
                prompt,
                "Human: My favorite color is blue.\nYumon: I like blue too.\nHuman: What color?"
            );
            let long = "Hello 🐾 friend! ".repeat(200);
            for context in [32, 256] {
                let prompt = chat_prompt(&long, &history, context, &tokenizer, false).unwrap();
                assert!(long.starts_with(prompt.strip_prefix("Human: ").unwrap_or(&prompt)));
                assert!(
                    1 + tokenizer.encode(&prompt).len()
                        + tokenizer.encode(yumon_pet::brain::samples::language_separator_for(&tokenizer)).len()
                        + (context / 4).clamp(8, 64)
                        <= context
                );
            }
            assert!(chat_prompt("Hello", &history, 8, &tokenizer, false).is_err());
        }

        #[test]
        fn long_unicode_headlines_fit_without_splitting_characters() {
            let prompt = fit_prompt("月面ニュース: a very long discovery", 22, |s| {
                s.chars().count()
            })
            .unwrap();
            assert_eq!(prompt, "Your thoughts on 月面ニュ?");
            assert!(fit_prompt("Headline", 17, |s| s.chars().count()).is_err());
            assert_eq!(
                fit_prompt("Short", 100, |s| s.chars().count()).unwrap(),
                "Your thoughts on Short?"
            );
        }

        #[test]
        fn chat_prompt_preserves_direct_messages_and_trims_unicode() {
            assert_eq!(
                fit_chat_prompt("  Hello Yumon!  ", 32, |s| s.chars().count()).unwrap(),
                "Hello Yumon!"
            );
            assert_eq!(
                fit_chat_prompt("Hello 🐾 friend", 7, |s| s.chars().count()).unwrap(),
                "Hello 🐾"
            );
            assert!(fit_chat_prompt("Hello", 0, |s| s.chars().count()).is_err());
            assert!(fit_chat_prompt(" \n ", 20, |s| s.chars().count()).is_err());
        }

        #[test]
        fn chat_replies_and_failures_leave_the_product_schedule_untouched() {
            let next_check = Some(Utc::now() + chrono::Duration::minutes(5));
            let mut state = ViewState {
                entries: Vec::new(),
                chat: vec![chat("user", "Hello Yumon!")],
                chat_busy: true,
                chat_status: "Thinking".into(),
                status: "Products waiting".into(),
                busy: false,
                can_refresh: true,
                next_check,
                paint_status: "Ready".into(),
                paint_busy: false,
            };
            let responder: Responder = Box::new(|text, kind, history| {
                assert!(matches!(kind, PromptKind::Chat { .. }));
                assert_eq!(text, "Hello Yumon!");
                assert_eq!(history[0].text, "Hello Yumon!");
                Ok("Hello friend!".into())
            });
            answer_chat(&mut state, "Hello Yumon!".into(), 0, "Yumon".into(), None, None, &responder);
            assert_eq!(state.chat[1].text, "Hello friend!");
            assert_eq!(state.chat[1].role, "yumon");
            assert_eq!(state.chat[1].citizen_name.as_deref(), Some("Yumon"));
            assert!(!state.chat_busy);
            let failed: Responder = Box::new(|_, _, _| anyhow::bail!("GPU unavailable"));
            state.chat_busy = true;
            answer_chat(&mut state, "Try again".into(), 0, "Yumon".into(), None, None, &failed);
            assert_eq!(state.chat[2].role, "error");
            assert!(state.chat[2].text.contains("GPU unavailable"));
            assert!(!state.chat_busy);
            assert_eq!(state.next_check, next_check);
            assert_eq!(state.status, "Products waiting");
            assert!(state.entries.is_empty());
        }

        #[test]
        fn test_loose_response_parsing_chat() {
            // Standard format with prefix
            assert_eq!(
                parse_chat_response_loose("Reply: \"My tummy rumbles loud! Please feed me now.\""),
                "My tummy rumbles loud! Please feed me now."
            );
            // Lowercase prefix
            assert_eq!(
                parse_chat_response_loose("reply: \"Rest your head, I'll stay right here with you.\""),
                "Rest your head, I'll stay right here with you."
            );
            // Without quotes
            assert_eq!(
                parse_chat_response_loose("Reply: Yes! I am ready to sprint right now!"),
                "Yes! I am ready to sprint right now!"
            );
            // Bare sentence without prefix
            assert_eq!(
                parse_chat_response_loose("\"I would love a delicious snack!\""),
                "I would love a delicious snack!"
            );
            // Truncates hallucinated follow-up turns
            let multi_turn = "Reply: \"I am ready to play!\"\nperson: what game?\nyumon: fetch!";
            assert_eq!(
                parse_chat_response_loose(multi_turn),
                "I am ready to play!"
            );
        }

        #[test]
        fn test_loose_response_parsing_behavior() {
            // Standard two-line format
            let standard = "Action: I will pick up the squeaky rubber mouse and toss it toward owner's feet.\nSpeak: \"Look at my squeaky mouse!\"";
            let parsed = parse_behavior_response_loose(standard);
            assert_eq!(parsed.action, "I will pick up the squeaky rubber mouse and toss it toward owner's feet.");
            assert_eq!(parsed.primary_verb, "pick");
            assert_eq!(parsed.target.as_deref(), Some("mouse"));
            assert_eq!(parsed.speech.as_deref(), Some("Look at my squeaky mouse!"));

            // Speak: none
            let no_speak = "Action: I will curl up tightly in the warm patch of sunlight and close my eyes.\nSpeak: none";
            let parsed2 = parse_behavior_response_loose(no_speak);
            assert_eq!(parsed2.primary_verb, "curl");
            assert_eq!(parsed2.speech, None);

            // Missing Action: prefix (loose parsing)
            let no_prefix = "I will swim over to the glowing kelp.\nSpeak: \"So pretty!\"";
            let parsed3 = parse_behavior_response_loose(no_prefix);
            assert_eq!(parsed3.action, "I will swim over to the glowing kelp.");
            assert_eq!(parsed3.primary_verb, "swim");
            assert_eq!(parsed3.target.as_deref(), Some("kelp"));
            assert_eq!(parsed3.speech.as_deref(), Some("So pretty!"));
        }

        #[test]
        fn test_dpo_vitals_grounded_emote_and_format() {
            let v = VitalsData {
                hunger: 80,
                energy: 40,
                cleanliness: 90,
                boredom: 20,
            };
            let e = v.compute_grounded_emote();
            assert_eq!(e.name, "Hungry");
            assert_eq!(e.having, false);
            assert_eq!(e.knowing, true);
            assert_eq!(e.empathizing, true);

            let header = format!(
                "Vitals: Hunger {}, Energy {}, Cleanliness {}, Boredom {}.\nEmote: [having: {}, knowing: {}, empathizing: {}] {}.",
                v.hunger, v.energy, v.cleanliness, v.boredom,
                if e.having { "yes" } else { "no" },
                if e.knowing { "yes" } else { "no" },
                if e.empathizing { "yes" } else { "no" },
                e.name
            );
            assert_eq!(
                header,
                "Vitals: Hunger 80, Energy 40, Cleanliness 90, Boredom 20.\nEmote: [having: no, knowing: yes, empathizing: yes] Hungry."
            );
        }

        #[test]
        fn comment_prompt_reserves_reply_space_and_scales_with_context() {
            let tokenizer = tokenizer();
            let prompt = comment_prompt("Product Hunt launch today", 1024, &tokenizer).unwrap();
            let prompt_len = tokenizer.encode(&prompt).len();
            // On a 1024 context, at least 256 reply tokens are reserved (instead of just 8).
            assert!(1 + prompt_len + tokenizer.encode(yumon_pet::brain::samples::language_separator_for(&tokenizer)).len() + 256 <= 1024);

            let prompt_256 = comment_prompt("Product Hunt launch today", 256, &tokenizer).unwrap();
            let prompt_256_len = tokenizer.encode(&prompt_256).len();
            assert!(1 + prompt_256_len + tokenizer.encode(yumon_pet::brain::samples::language_separator_for(&tokenizer)).len() + 64 <= 256);
        }

        #[test]
        fn embedded_assets_are_valid() {
            let meta: serde_json::Value = serde_json::from_str(EMBEDDED_METADATA_STR).unwrap();
            assert_eq!(meta["training_stage"], "Language");
            assert_eq!(meta["vocab_size"], 4096);
            assert!(EMBEDDED_MODEL_BYTES.len() > 1_000_000);
            assert!(EMBEDDED_TOKENIZER_BYTES.len() > 1_000);
            let parsed_vocab = yumon_pet::brain::yumon_talk::YumonTalkVocab::load_from_str(EMBEDDED_VOCAB_STR);
            assert!(parsed_vocab.is_ok());
            assert_eq!(parsed_vocab.unwrap().vocab_size, 4096);
        }

        #[test]
        fn test_embedded_model_inference_smoke() {
            let args = Args {
                checkpoint: None,
                architecture: Architecture::DecoderOnly,
                history: None,
                cache: None,
                check_feeds: false,
                yumon_talk: true,
                yumon_talk_vocab: None,
                decompressor: Some(PathBuf::from("training-cache/ngram_decompressor.bin")),
                no_decompressor: false,
                paint_pass1: None,
                paint_pass2: None,
                paint_pass3: None,
                paint_pass4: None,
                paint_upscaler: None,
                paint_colorizer: None,
                paint_vocab: None,
            };
            let responder = load_responder(&args).expect("load_responder with embedded model should succeed");
            let reply = responder("Hello", PromptKind::Chat { vitals: None, emote: None, intent: None }, &[]).expect("inference with embedded model should succeed");
            assert!(!reply.trim().is_empty());
        }

        #[test]
        fn test_dpo_chat_and_simulation_inference() {
            let dpo_path = resolve_default_checkpoint(None);
            if let Some(ckpt) = dpo_path {
                let args = Args {
                    checkpoint: Some(ckpt),
                    architecture: Architecture::Moe,
                    history: None,
                    cache: None,
                    check_feeds: false,
                    yumon_talk: true,
                    yumon_talk_vocab: None,
                    decompressor: Some(PathBuf::from("training-cache/ngram_decompressor.bin")),
                    no_decompressor: false,
                    paint_pass1: None,
                    paint_pass2: None,
                    paint_pass3: None,
                    paint_pass4: None,
                    paint_upscaler: None,
                    paint_colorizer: None,
                    paint_vocab: None,
                };
                let responder = load_responder(&args).expect("load DPO responder should succeed");

                // 1. Chat with vitals
                let vitals = VitalsData {
                    hunger: 80,
                    energy: 40,
                    cleanliness: 90,
                    boredom: 20,
                };
                let emote = vitals.compute_grounded_emote();
                let chat_kind = PromptKind::Chat {
                    vitals: Some(vitals.clone()),
                    emote: Some(emote.clone()),
                    intent: Some("Chat".into()),
                };
                let reply = responder("Is it dinner time yet?", chat_kind, &[]).expect("DPO chat inference");
                println!("[DPO Chat Output] {}", reply);
                assert!(!reply.trim().is_empty());

                // 2. Simulation behavior
                let sim_kind = PromptKind::Simulation {
                    vitals,
                    emote,
                    environment: "Suburban Park, nearby: Oak Tree, Park Bench, Flower Bed".into(),
                    memory: Some("Owner fed a crunchy snack earlier".into()),
                    plan: None,
                };
                let action_json = responder("", sim_kind, &[]).expect("DPO simulation inference");
                println!("[DPO Simulation Output JSON] {}", action_json);
                let action: BehaviorAction = serde_json::from_str(&action_json).expect("valid action json");
                assert!(!action.action.is_empty());
                assert!(!action.primary_verb.is_empty());
            }
        }

        #[test]
        fn chat_prompt_scales_reply_budget_for_large_context() {
            let tokenizer = tokenizer();
            let history = vec![
                chat("user", "Hello Yumon!"),
                chat("yumon", "Hello! How can I help you?"),
            ];
            let prompt = chat_prompt("Tell me a story", &history, 1024, &tokenizer, false).unwrap();
            let prompt_len = tokenizer.encode(&prompt).len();
            // 256 tokens reserved on 1024 context rather than clamped to 64.
            assert!(1 + prompt_len + tokenizer.encode(yumon_pet::brain::samples::language_separator_for(&tokenizer)).len() + 256 <= 1024);
        }
    }

    fn load_responder(args: &Args) -> Result<Responder> {
        let resolved_ckpt = resolve_default_checkpoint(args.checkpoint.as_deref());
        let metadata: serde_json::Value = if let Some(ckpt) = &resolved_ckpt {
            println!("[Yumon Portal] Found checkpoint at {ckpt}");
            serde_json::from_str(&std::fs::read_to_string(
                PathBuf::from(ckpt).join("metadata.json"),
            )?)?
        } else {
            println!("[Yumon Portal] No custom checkpoint found; using embedded model.");
            serde_json::from_str(EMBEDDED_METADATA_STR)?
        };
        anyhow::ensure!(
            metadata["training_stage"] == "Language",
            "Portal requires a Language-stage checkpoint"
        );
        let vocab_size_is_4k = metadata["vocab_size"].as_u64() == Some(4096);
        let use_yumon_talk = args.yumon_talk || vocab_size_is_4k;
        if use_yumon_talk {
            if let Some(vocab_path) = &args.yumon_talk_vocab {
                if vocab_path.exists() {
                    println!("[Yumon Portal] Loading Yumon Talk vocabulary from {} ...", vocab_path.display());
                    yumon_talk::load_global_vocab_file(vocab_path)?;
                } else {
                    println!("[Yumon Portal] Specified vocab file {} not found; using embedded vocabulary.", vocab_path.display());
                    yumon_talk::load_global_vocab_str(EMBEDDED_VOCAB_STR)?;
                }
            } else {
                println!("[Yumon Portal] Using embedded Yumon Talk vocabulary.");
                yumon_talk::load_global_vocab_str(EMBEDDED_VOCAB_STR)?;
            }
        }
        let decompressor: Option<std::sync::Arc<NGramDecompressor>> = if use_yumon_talk && !args.no_decompressor {
            let candidate_path = args.decompressor.clone().or_else(|| {
                // 1. Next to the executable (standard release package layout)
                if let Ok(exe_path) = std::env::current_exe() {
                    let beside = exe_path.with_file_name("ngram_decompressor.bin");
                    if beside.exists() {
                        return Some(beside);
                    }
                }
                // 2. Local workspace cache (for development)
                let local = PathBuf::from("training-cache/ngram_decompressor.bin");
                if local.exists() {
                    return Some(local);
                }
                // 3. Local app data directory
                if let Some(dirs) = directories::ProjectDirs::from("", "Yumon", "YumonRSS") {
                    let app_data = dirs.data_local_dir().join("ngram_decompressor.bin");
                    if app_data.exists() {
                        return Some(app_data);
                    }
                }
                None
            });

            if let Some(path) = candidate_path {
                println!("[Yumon Portal] Loading N-gram statistical decompressor from {} ...", path.display());
                let d = NGramDecompressor::load_from_file(&path)
                    .with_context(|| format!("loading decompressor from {}", path.display()))?;
                Some(std::sync::Arc::new(d))
            } else {
                anyhow::bail!(
                    "Decompressor model 'ngram_decompressor.bin' not found.
                     Please ensure 'ngram_decompressor.bin' is placed in the same folder as yumon_portal.exe,
                     or specify its path via --decompressor <path>."
                );
            }
        } else {
            None
        };
        let device = Default::default();

        macro_rules! build_responder {
            ($brain:expr, $tokenizer:expr, $config:expr, $model_ty:ty, $generate:expr) => {{
                let brain = $brain;
                let tokenizer = $tokenizer;
                let config = $config;
                anyhow::ensure!(
                    config.training_stage == TrainingStage::Language,
                    "Expected Language stage"
                );
                // Check before fetching so an unusable context produces one visible error.
                comment_prompt("idea", config.max_seq_len, &tokenizer)?;
                let dec_clone = decompressor.clone();
                Ok(Box::new(
                    move |text: &str, kind: PromptKind, _history: &[ChatMessage]| {
                        let is_structured = yumon_pet::brain::train::is_structured_or_instruction_prompt(text);
                        let (prompt_to_encode, ctx) = match &kind {
                            PromptKind::Product => {
                                let (prompt_text, ctx) = if use_yumon_talk {
                                    match yumon_talk::english_to_yumon_with_map(text) {
                                        Ok((talk, context)) => (talk, context),
                                        Err(e) => {
                                            eprintln!("[Yumon Portal] English -> Yumon Talk translation fallback: {e}");
                                            (text.to_string(), yumon_talk::TranslationContext::new())
                                        }
                                    }
                                } else {
                                    (text.to_string(), yumon_talk::TranslationContext::new())
                                };
                                let p = comment_prompt(&prompt_text, config.max_seq_len, &tokenizer)?;
                                (p, ctx)
                            }
                            PromptKind::Chat { vitals, emote, intent } => {
                                let formatted_prompt = if is_structured {
                                    text.to_string()
                                } else if let (Some(v), Some(e)) = (vitals, emote) {
                                    format!(
                                        "Vitals: Hunger {}, Energy {}, Cleanliness {}, Boredom {}.\nEmote: [having: {}, knowing: {}, empathizing: {}] {}.\nIntent: {}.\nOwner Prompt: \"{}\"",
                                        v.hunger, v.energy, v.cleanliness, v.boredom,
                                        if e.having { "yes" } else { "no" },
                                        if e.knowing { "yes" } else { "no" },
                                        if e.empathizing { "yes" } else { "no" },
                                        e.name,
                                        intent.as_deref().unwrap_or("Chat"),
                                        text
                                    )
                                } else {
                                    let def_v = VitalsData::default();
                                    let def_e = def_v.compute_grounded_emote();
                                    format!(
                                        "Vitals: Hunger {}, Energy {}, Cleanliness {}, Boredom {}.\nEmote: [having: {}, knowing: {}, empathizing: {}] {}.\nIntent: Chat.\nOwner Prompt: \"{}\"",
                                        def_v.hunger, def_v.energy, def_v.cleanliness, def_v.boredom,
                                        if def_e.having { "yes" } else { "no" },
                                        if def_e.knowing { "yes" } else { "no" },
                                        if def_e.empathizing { "yes" } else { "no" },
                                        def_e.name,
                                        text
                                    )
                                };

                                if use_yumon_talk {
                                    match yumon_talk::english_to_yumon_with_map(&formatted_prompt) {
                                        Ok((talk, context)) => (talk, context),
                                        Err(e) => {
                                            eprintln!("[Yumon Portal] English -> Yumon Talk translation fallback: {e}");
                                            (formatted_prompt, yumon_talk::TranslationContext::new())
                                        }
                                    }
                                } else {
                                    (formatted_prompt, yumon_talk::TranslationContext::new())
                                }
                            }
                            PromptKind::Simulation { vitals, emote, environment, memory, plan } => {
                                let formatted_prompt = if is_structured {
                                    text.to_string()
                                } else {
                                    format!(
                                        "Vitals: Hunger {}, Energy {}, Cleanliness {}, Boredom {}.\nEmote: [having: {}, knowing: {}, empathizing: {}] {}.\nEnvironment: {}.\nMemory: {}.\nPlan: {}.",
                                        vitals.hunger, vitals.energy, vitals.cleanliness, vitals.boredom,
                                        if emote.having { "yes" } else { "no" },
                                        if emote.knowing { "yes" } else { "no" },
                                        if emote.empathizing { "yes" } else { "no" },
                                        emote.name,
                                        environment,
                                        memory.as_deref().unwrap_or("none"),
                                        plan.as_deref().unwrap_or("none")
                                    )
                                };

                                if use_yumon_talk {
                                    match yumon_talk::english_to_yumon_with_map(&formatted_prompt) {
                                        Ok((talk, context)) => (talk, context),
                                        Err(e) => {
                                            eprintln!("[Yumon Portal] English -> Yumon Talk translation fallback: {e}");
                                            (formatted_prompt, yumon_talk::TranslationContext::new())
                                        }
                                    }
                                } else {
                                    (formatted_prompt, yumon_talk::TranslationContext::new())
                                }
                            }
                        };

                        let generate: fn(&$model_ty, &TokenizerKind, &str, usize, &_) -> String =
                            $generate;
                        let reply = generate(
                            &brain,
                            &tokenizer,
                            &prompt_to_encode,
                            config.max_seq_len.min(256),
                            &device,
                        );
                        let reply = match reply.split_once("\nHuman:")
                            .or_else(|| reply.split_once("\nperson:"))
                            .or_else(|| reply.split_once("\n person:"))
                        {
                            Some((clean, _)) => clean.trim().to_owned(),
                            None => reply.trim().to_owned(),
                        };
                        anyhow::ensure!(!reply.is_empty(), "Yumon returned an empty reply");

                        let decompressed_reply = if use_yumon_talk {
                            if let Some(dec) = &dec_clone {
                                let vocab = yumon_talk::get_global_vocab();
                                match dec.decompress_with_context(&reply, &vocab, &ctx) {
                                    Ok(decompressed) => decompressed,
                                    Err(e) => {
                                        eprintln!("[Yumon Portal] Decompressor error: {e}");
                                        reply
                                    }
                                }
                            } else {
                                reply
                            }
                        } else {
                            reply
                        };

                        println!("[DEBUG decompressed_reply] {:?}", decompressed_reply);

                        let final_reply = match &kind {
                            PromptKind::Product => decompressed_reply,
                            PromptKind::Chat { .. } => parse_chat_response_loose(&decompressed_reply),
                            PromptKind::Simulation { .. } => {
                                let action = parse_behavior_response_loose(&decompressed_reply);
                                serde_json::to_string(&action)?
                            }
                        };

                        Ok(final_reply)
                    },
                ) as Responder)
            }};
        }

        if let Some(ckpt) = &resolved_ckpt {
            let arch = if metadata.get("num_experts").is_some() && matches!(args.architecture, Architecture::DecoderOnly) {
                println!("[Yumon Portal] Auto-detected MoE checkpoint metadata; using MoE architecture.");
                Architecture::Moe
            } else {
                args.architecture
            };
            println!("[Yumon Portal] Loading checkpoint from directory: {ckpt}");
            match arch {
                Architecture::DecoderOnly => {
                    let (brain, tokenizer, config) = YumonDecBrain::<burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>>::load(ckpt, &device)?;
                    build_responder!(brain, tokenizer, config, YumonDecBrain<burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>>, |b, t, p, n, d| b.generate_unmasked_parsed(t, p, n, d).raw_output)
                }
                Architecture::Moe => {
                    let (brain, tokenizer, config) = YumonMoeBrain::<burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>>::load(ckpt, &device)?;
                    build_responder!(brain, tokenizer, config, YumonMoeBrain<burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>>, |b, t, p, n, d| b.generate_unmasked_parsed(t, p, n, d).raw_output)
                }
                Architecture::Xlstm => {
                    let (brain, tokenizer, config) = YumonXLstmBrain::<Wgpu>::load(ckpt, &device)?;
                    build_responder!(brain, tokenizer, config, YumonXLstmBrain<Wgpu>, |b, t, p, n, d| b.generate_unmasked_parsed(t, p, n, d).raw_output)
                }
                Architecture::EncoderDecoder => {
                    let (brain, tokenizer, config) = YumonBrain::<Wgpu>::load(ckpt, &device)?;
                    build_responder!(brain, tokenizer, config, YumonBrain<Wgpu>, |b, t, p, n, d| b.generate_unmasked_parsed::<cubecl::wgpu::WgpuRuntime>(t, p, n, d).raw_output)
                }
            }
        } else {
            println!("[Yumon Portal] Loading self-contained embedded MoE checkpoint...");
            let (brain, tokenizer, config) = YumonMoeBrain::<burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>>::load_from_bytes(
                EMBEDDED_MODEL_BYTES,
                EMBEDDED_TOKENIZER_BYTES,
                EMBEDDED_METADATA_STR,
                &device,
            )?;
            build_responder!(brain, tokenizer, config, YumonMoeBrain<burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>>, |b, t, p, n, d| b.generate_unmasked_parsed(t, p, n, d).raw_output)
        }
    }

    fn agent() -> ureq::Agent {
        ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(20))
            .build()
    }

    fn worker(
        args: Args,
        history_path: PathBuf,
        cache_path: PathBuf,
        mut cache: Option<product_hunt::DailyCache>,
        mut state: ViewState,
        rx: mpsc::Receiver<WorkerRequest>,
        proxy: EventLoopProxy<RssEvent>,
    ) {
        let publish = |state: &ViewState| proxy.send_event(RssEvent::State(state.clone())).is_ok();
        let responder = match load_responder(&args) {
            Ok(responder) => responder,
            Err(error) => {
                state.status = format!("Could not load Yumon: {error:#}");
                state.busy = false;
                state.chat_status = "Yumon could not load. See the status below.".into();
                publish(&state);
                return;
            }
        };
        state.can_refresh = true;
        state.chat_status = "Say hello or share an idea.".into();
        let drain_pending = |state: &mut ViewState, refresh: &mut bool| {
            while let Ok(req) = rx.try_recv() {
                match req {
                    WorkerRequest::ClearChat => {
                        state.chat.clear();
                        state.chat_status = "Say hello or share an idea.".into();
                    }
                    WorkerRequest::Refresh => *refresh = true,
                    WorkerRequest::Chat { .. } => {}
                    WorkerRequest::SimDecide { .. } => {}
                }
            }
        };
        let agent = agent();
        let mut deadline = Instant::now();
        let mut refresh = false;
        loop {
            state.busy = true;
            state.next_check = None;
            let mut errors = Vec::new();
            let day = Local::now().date_naive();
            if cache.as_ref().is_some_and(|c| c.day != day) {
                cache = None;
            }
            if refresh || cache.is_none() {
                state.status = "Loading today's Product Hunt products...".into();
                drain_pending(&mut state, &mut refresh);
                if !publish(&state) {
                    return;
                }
                match product_hunt::fetch(&agent, day) {
                    Ok(mut fetched) => {
                        if let Some(previous) = &cache {
                            fetched.commented = previous.commented.clone();
                        }
                        if let Err(error) = product_hunt::save(&cache_path, &fetched) {
                            errors.push(format!("Cache: {error:#}"));
                        }
                        cache = Some(fetched);
                        refresh = false;
                    }
                    Err(error) => {
                        errors.push(format!("Product Hunt: {error:#}"));
                        refresh = true;
                    }
                }
            }
            // Refresh updates the snapshot without accelerating the comment timer.
            if Instant::now() >= deadline {
                if let Some(daily) = &mut cache {
                    if let Some(product) = daily.next_product().cloned() {
                        state.status = format!("Yumon is reading {}...", product.article.title);
                        drain_pending(&mut state, &mut refresh);
                        if !publish(&state) {
                            return;
                        }
                        // Put the description first: small checkpoints should spend
                        // their limited context on the idea rather than its name.
                        match responder(&product.article.description, PromptKind::Product, &[]) {
                            Ok(comment) => {
                                state.entries.push(Entry {
                                    source: "Product Hunt".into(),
                                    article: product.article,
                                    comment,
                                    received: Utc::now(),
                                });
                                if state.entries.len() > HISTORY_LIMIT {
                                    state.entries.remove(0);
                                }
                                daily.commented.insert(product.id);
                                if let Err(error) = rss::save_history(&history_path, &state.entries)
                                {
                                    errors.push(format!("History: {error:#}"));
                                }
                            }
                            Err(error) => errors.push(format!("Comment: {error:#}")),
                        }
                    }
                }
                deadline = Instant::now() + POLL_INTERVAL;
            }
            if let Some(daily) = &cache {
                if let Err(error) = product_hunt::save(&cache_path, daily) {
                    errors.push(format!("Cache: {error:#}"));
                }
            }
            state.busy = false;
            state.next_check = Some(
                Utc::now()
                    + chrono::Duration::from_std(
                        deadline.saturating_duration_since(Instant::now()),
                    )
                    .unwrap(),
            );
            state.status = if errors.is_empty() {
                let remaining = cache.as_ref().map_or(0, |c| c.remaining());
                if remaining == 0 {
                    "All caught up with today's products. Refresh for new arrivals.".into()
                } else {
                    format!("{remaining} products waiting. Another thought in five minutes.")
                }
            } else {
                format!("{} Will retry on the next check.", errors.join(" · "))
            };
            drain_pending(&mut state, &mut refresh);
            if !publish(&state) {
                return;
            }
            loop {
                match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                    Ok(WorkerRequest::Refresh) => {
                        refresh = true;
                        break;
                    }
                    Ok(WorkerRequest::Chat {
                        message,
                        citizen_id,
                        citizen_name,
                        vitals,
                        emote,
                    }) => {
                        state.chat_busy = true;
                        state.chat_status = format!("{citizen_name} is thinking...");
                        state.chat.push(ChatMessage {
                            role: "user",
                            text: message.clone(),
                            citizen_name: None,
                        });
                        if !publish(&state) {
                            return;
                        }
                        if let Some(reply_text) = answer_chat(
                            &mut state,
                            message,
                            citizen_id,
                            citizen_name.clone(),
                            vitals,
                            emote,
                            &responder,
                        ) {
                            let _ = proxy.send_event(RssEvent::ChatReply {
                                citizen_id,
                                citizen_name,
                                text: reply_text,
                            });
                        }
                        drain_pending(&mut state, &mut refresh);
                        if !publish(&state) {
                            return;
                        }
                        if Instant::now() >= deadline {
                            break;
                        }
                    }
                    Ok(WorkerRequest::SimDecide {
                        citizen_id,
                        citizen_name,
                        vitals,
                        emote,
                        environment,
                        memory,
                        plan,
                    }) => {
                        let kind = PromptKind::Simulation {
                            vitals,
                            emote,
                            environment,
                            memory,
                            plan,
                        };
                        match responder("", kind, &[]) {
                            Ok(action_json) => {
                                if let Ok(action) = serde_json::from_str::<BehaviorAction>(&action_json) {
                                    let _ = proxy.send_event(RssEvent::SimAction {
                                        citizen_id,
                                        action: action.action,
                                        primary_verb: action.primary_verb,
                                        target: action.target,
                                        speech: action.speech,
                                    });
                                }
                            }
                            Err(e) => {
                                eprintln!("[Yumon Portal] Simulation decision error for citizen {citizen_id} ({citizen_name}): {e:#}");
                            }
                        }
                    }
                    Ok(WorkerRequest::ClearChat) => {
                        state.chat.clear();
                        state.chat_status = "Say hello or share an idea.".into();
                        if !publish(&state) {
                            return;
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => break,
                    Err(mpsc::RecvTimeoutError::Disconnected) => return,
                }
            }
        }
    }

    fn open_article(link: &str) -> Result<()> {
        anyhow::ensure!(rss::is_web_link(link), "Invalid article URL");
        #[cfg(target_os = "windows")]
        let mut command = {
            let mut command = std::process::Command::new("rundll32.exe");
            command.arg("url.dll,FileProtocolHandler");
            command
        };
        #[cfg(target_os = "macos")]
        let mut command = std::process::Command::new("open");
        #[cfg(not(any(target_os = "windows", target_os = "macos")))]
        let mut command = std::process::Command::new("xdg-open");
        command
            .arg(link)
            .spawn()
            .context("Could not open article in your browser")?;
        Ok(())
    }

    fn open_art_file(path_str: &str) -> Result<()> {
        let path = PathBuf::from(path_str);
        anyhow::ensure!(path.exists(), "Artwork file not found: {path_str}");
        #[cfg(target_os = "windows")]
        {
            let mut command = std::process::Command::new("explorer.exe");
            command.arg(format!("/select,\"{}\"", path.display()));
            command.spawn().context("Opening Explorer")?;
        }
        #[cfg(target_os = "macos")]
        {
            let mut command = std::process::Command::new("open");
            command.arg("-R").arg(&path);
            command.spawn().context("Opening Finder")?;
        }
        #[cfg(not(any(target_os = "windows", target_os = "macos")))]
        {
            let mut command = std::process::Command::new("xdg-open");
            command.arg(path.parent().unwrap_or(&path));
            command.spawn().context("Opening File Manager")?;
        }
        Ok(())
    }

    fn paint_worker(
        paths: PaintPaths,
        rx: mpsc::Receiver<PaintJob>,
        proxy: EventLoopProxy<RssEvent>,
    ) {
        let initial_items = yumon_pet::vision::paint_studio::scan_gallery(std::path::Path::new("inference-cache"), 60);
        let _ = proxy.send_event(RssEvent::GalleryLoaded(initial_items));

        let device: <BackendType as burn::prelude::Backend>::Device = Default::default();
        let mut studio: Option<PaintStudio<BackendType>> = None;

        while let Ok(job) = rx.recv() {
            match job {
                PaintJob::RefreshGallery => {
                    let items = yumon_pet::vision::paint_studio::scan_gallery(std::path::Path::new("inference-cache"), 60);
                    let _ = proxy.send_event(RssEvent::GalleryLoaded(items));
                }
                PaintJob::GetFullImage { path } => {
                    let p = PathBuf::from(&path);
                    if p.exists() {
                        if let Ok(bytes) = std::fs::read(&p) {
                            use base64::Engine;
                            let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
                            let url = format!("data:image/png;base64,{b64}");
                            let _ = proxy.send_event(RssEvent::FullImageLoaded { path, data_url: url });
                        }
                    }
                }
                PaintJob::Generate(config) => {
                    if studio.is_none() {
                        let _ = proxy.send_event(RssEvent::PaintProgress(PassProgress {
                            pass: 0,
                            total_passes: config.passes,
                            stage_name: "Waking Yumon Paint".into(),
                            description: "Loading neural weights into GPU...".into(),
                            width: 256,
                            height: 256,
                            data_url: String::new(),
                            elapsed_secs: 0.0,
                        }));
                        match PaintStudio::<BackendType>::load(&paths, &device) {
                            Ok(s) => studio = Some(s),
                            Err(e) => {
                                let _ = proxy.send_event(RssEvent::PaintError(format!("Failed to load paint models: {e:#}")));
                                continue;
                            }
                        }
                    }

                    if let Some(s) = &studio {
                        let proxy_clone = proxy.clone();
                        match s.generate(&config, move |progress| {
                            let _ = proxy_clone.send_event(RssEvent::PaintProgress(progress));
                        }) {
                            Ok(res) => {
                                let item = GalleryItem {
                                    id: res.final_path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "artwork".into()),
                                    path: res.final_path.to_string_lossy().to_string(),
                                    raw_path: Some(res.raw_path.to_string_lossy().to_string()),
                                    prompt: res.prompt,
                                    timestamp: chrono::Local::now().to_rfc3339(),
                                    width: res.width,
                                    height: res.height,
                                    thumbnail_url: res.thumbnail_url,
                                };
                                let _ = proxy.send_event(RssEvent::PaintComplete {
                                    item,
                                    final_data_url: res.final_data_url,
                                });
                            }
                            Err(e) => {
                                let _ = proxy.send_event(RssEvent::PaintError(format!("Generation failed: {e:#}")));
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn run() -> Result<()> {
        let args = Args::parse();
        let history_path = args.history.clone().unwrap_or_else(|| {
            directories::ProjectDirs::from("", "Yumon", "YumonRSS")
                .map(|dirs| dirs.data_local_dir().join("history.json"))
                .unwrap_or_else(|| PathBuf::from("yumon-rss-history.json"))
        });
        let cache_path = args
            .cache
            .clone()
            .unwrap_or_else(|| history_path.with_file_name("producthunt-cache.json"));
        anyhow::ensure!(
            cache_path != history_path,
            "Cache and history must use different files"
        );
        let day = Local::now().date_naive();
        let cache = product_hunt::load(&cache_path, day)?;
        if args.check_feeds {
            let cache = match cache {
                Some(cache) => cache,
                None => {
                    let cache = product_hunt::fetch(&agent(), day)?;
                    product_hunt::save(&cache_path, &cache)?;
                    cache
                }
            };
            for product in cache.products {
                println!(
                    "{}\n{}\n{}\n",
                    product.article.title, product.article.description, product.article.link
                );
            }
            return Ok(());
        }
        // Never overwrite an unreadable history with an empty one.
        let entries = rss::load_history(&history_path)?;

        let mut state = ViewState {
            entries,
            chat: Vec::new(),
            chat_busy: false,
            chat_status: "Waking Yumon up...".into(),
            status: "Waking Yumon up...".into(),
            busy: true,
            can_refresh: false,
            next_check: None,
            paint_status: "Ready".into(),
            paint_busy: false,
        };
        let mut gallery: Vec<GalleryItem> = Vec::new();
        let event_loop = EventLoopBuilder::<RssEvent>::with_user_event().build();
        let proxy = event_loop.create_proxy();
        let mut builder = WindowBuilder::new()
            .with_title("Yumon Portal")
            .with_inner_size(LogicalSize::new(960, 580))
            .with_decorations(false)
            .with_transparent(true)
            .with_always_on_top(true)
            .with_resizable(true);
        if let Some(monitor) = event_loop
            .primary_monitor()
            .or_else(|| event_loop.available_monitors().next())
        {
            let scale = monitor.scale_factor();
            let origin = monitor.position();
            let size = monitor.size();
            builder = builder.with_position(PhysicalPosition::new(
                origin.x + (16.0 * scale) as i32,
                origin.y + (size.height as i32 - (640.0 * scale) as i32).max(0),
            ));
        }
        #[cfg(target_os = "windows")]
        {
            use tao::platform::windows::WindowBuilderExtWindows;
            builder = builder.with_undecorated_shadow(false);
        }
        let window = builder.build(&event_loop)?;
        let ipc_proxy = proxy.clone();
        let ui_html = include_str!("rss_ui.html")
            .replace("/* __VENDOR_THREE__ */", include_str!("vendor/three.min.js"))
            .replace("/* __VENDOR_ORBIT_CONTROLS__ */", include_str!("vendor/OrbitControls.js"))
            .replace("/* __UNIVERSE_VIEW__ */", include_str!("universe_view.js"));
        let wv_builder = WebViewBuilder::new()
            .with_transparent(true)
            .with_html(&ui_html)
            .with_new_window_req_handler(|_, _| wry::NewWindowResponse::Deny)
            .with_ipc_handler(move |request| {
                if let Ok(ipc) = serde_json::from_str::<Ipc>(request.body()) {
                    let _ = ipc_proxy.send_event(RssEvent::Ui(ipc));
                }
            });
        #[cfg(any(target_os = "windows", target_os = "macos"))]
        let webview = wv_builder.build(&window)?;
        #[cfg(not(any(target_os = "windows", target_os = "macos")))]
        let webview = {
            use tao::platform::unix::WindowExtUnix;
            use wry::WebViewBuilderExtUnix;
            wv_builder.build_gtk(window.default_vbox().context("No GTK window container")?)?
        };
        let (tx, rx) = mpsc::channel();
        let worker_state = state.clone();
        let worker_args = Args {
            checkpoint: args.checkpoint.clone(),
            architecture: args.architecture,
            history: args.history.clone(),
            cache: args.cache.clone(),
            check_feeds: args.check_feeds,
            yumon_talk: args.yumon_talk,
            yumon_talk_vocab: args.yumon_talk_vocab.clone(),
            decompressor: args.decompressor.clone(),
            no_decompressor: args.no_decompressor,
            paint_pass1: args.paint_pass1.clone(),
            paint_pass2: args.paint_pass2.clone(),
            paint_pass3: args.paint_pass3.clone(),
            paint_pass4: args.paint_pass4.clone(),
            paint_upscaler: args.paint_upscaler.clone(),
            paint_colorizer: args.paint_colorizer.clone(),
            paint_vocab: args.paint_vocab.clone(),
        };
        let worker_proxy = proxy.clone();
        thread::spawn(move || {
            worker(
                worker_args,
                history_path,
                cache_path,
                cache,
                worker_state,
                rx,
                worker_proxy,
            )
        });

        // Initialize Paint Background Worker
        let (paint_tx, paint_rx) = mpsc::channel();
        let paint_proxy = proxy.clone();
        let paint_paths = PaintPaths {
            vocab_path: args.paint_vocab.clone().unwrap_or_else(|| PathBuf::from("training-cache/diffusion_talk_vocab.json")),
            checkpoint_pass1: args.paint_pass1.clone().or_else(|| {
                ["checkpoints/yumon-art-pass1-db", "checkpoints/yumon-art-pass1"]
                    .iter()
                    .map(PathBuf::from)
                    .find(|p| p.exists())
            }),
            checkpoint_pass2: args.paint_pass2.clone().or_else(|| {
                ["checkpoints/yumon-art-pass2-db", "checkpoints/yumon-art-pass2"]
                    .iter()
                    .map(PathBuf::from)
                    .find(|p| p.exists())
            }),
            checkpoint_pass3: args.paint_pass3.clone().or_else(|| {
                ["checkpoints/yumon-art-pass3-db", "checkpoints/yumon-art-pass3"]
                    .iter()
                    .map(PathBuf::from)
                    .find(|p| p.exists())
            }),
            checkpoint_pass4: args.paint_pass4.clone().or_else(|| {
                ["checkpoints/yumon-art-pass4-db", "checkpoints/yumon-art-pass4"]
                    .iter()
                    .map(PathBuf::from)
                    .find(|p| p.exists())
            }),
            upscaler_checkpoint: args.paint_upscaler.clone().or_else(|| {
                ["checkpoints/yumon-upscaler-64", "checkpoints/yumon-upscaler-256", "checkpoints/yumon-upscaler-64c"]
                    .iter()
                    .map(PathBuf::from)
                    .find(|p| p.exists())
            }),
            colorizer_checkpoint: args.paint_colorizer.clone().or_else(|| {
                ["checkpoints/yumon-colorizer-256"]
                    .iter()
                    .map(PathBuf::from)
                    .find(|p| p.exists())
            }),
        };
        thread::spawn(move || {
            paint_worker(paint_paths, paint_rx, paint_proxy);
        });

        let mut ready = false;
        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Wait;
            let mut state_changed = false;
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                }
                | Event::UserEvent(RssEvent::Ui(Ipc::Close)) => {
                    *control_flow = ControlFlow::Exit;
                    return;
                }
                Event::UserEvent(RssEvent::Ui(Ipc::Ready)) => {
                    ready = true;
                    if let Ok(json) = serde_json::to_string(&gallery) {
                        let _ = webview.evaluate_script(&format!("window.__yumon_gallery_loaded({json})"));
                    }
                    state_changed = true;
                }
                Event::UserEvent(RssEvent::State(update)) => {
                    state = update;
                    state_changed = true;
                }
                Event::UserEvent(RssEvent::PaintProgress(progress)) => {
                    state.paint_status = format!("{}: {}", progress.stage_name, progress.description);
                    if ready {
                        if let Ok(json) = serde_json::to_string(&progress) {
                            let _ = webview.evaluate_script(&format!("window.__yumon_paint_progress({json})"));
                        }
                    }
                    return;
                }
                Event::UserEvent(RssEvent::PaintComplete { item, final_data_url }) => {
                    state.paint_busy = false;
                    state.paint_status = "Artwork complete!".into();
                    gallery.insert(0, item.clone());
                    if ready {
                        let payload = serde_json::json!({
                            "item": item,
                            "final_data_url": final_data_url,
                        });
                        if let Ok(json) = serde_json::to_string(&payload) {
                            let _ = webview.evaluate_script(&format!("window.__yumon_paint_complete({json})"));
                        }
                    }
                    state_changed = true;
                }
                Event::UserEvent(RssEvent::PaintError(err)) => {
                    state.paint_busy = false;
                    state.paint_status = format!("Paint error: {err}");
                    if ready {
                        let escaped = serde_json::to_string(&err).unwrap_or_default();
                        let _ = webview.evaluate_script(&format!("window.__yumon_paint_error({escaped})"));
                    }
                    state_changed = true;
                }
                Event::UserEvent(RssEvent::GalleryLoaded(items)) => {
                    gallery = items;
                    if ready {
                        if let Ok(json) = serde_json::to_string(&gallery) {
                            let _ = webview.evaluate_script(&format!("window.__yumon_gallery_loaded({json})"));
                        }
                    }
                    return;
                }
                Event::UserEvent(RssEvent::FullImageLoaded { path, data_url }) => {
                    if ready {
                        let payload = serde_json::json!({ "path": path, "data_url": data_url });
                        if let Ok(json) = serde_json::to_string(&payload) {
                            let _ = webview.evaluate_script(&format!("window.__yumon_full_image_loaded({json})"));
                        }
                    }
                    return;
                }
                Event::UserEvent(RssEvent::Ui(Ipc::Paint { prompt, passes, temperature })) => {
                    let prompt = prompt.trim();
                    if !prompt.is_empty() && !state.paint_busy {
                        state.paint_busy = true;
                        state.paint_status = "Composing initial blocks...".into();
                        let config = PaintConfig {
                            prompt: prompt.to_string(),
                            passes: passes.unwrap_or(4),
                            temperature: temperature.unwrap_or(1.1),
                            ..Default::default()
                        };
                        let _ = paint_tx.send(PaintJob::Generate(config));
                        state_changed = true;
                    }
                }
                Event::UserEvent(RssEvent::Ui(Ipc::LoadGallery)) => {
                    let _ = paint_tx.send(PaintJob::RefreshGallery);
                    return;
                }
                Event::UserEvent(RssEvent::Ui(Ipc::GetFullImage { path })) => {
                    let _ = paint_tx.send(PaintJob::GetFullImage { path });
                    return;
                }
                Event::UserEvent(RssEvent::Ui(Ipc::OpenArtFile { path })) => {
                    if let Err(e) = open_art_file(&path) {
                        eprintln!("[Yumon Portal] Could not open artwork file {path}: {e:#}");
                    }
                    return;
                }
                Event::UserEvent(RssEvent::SimAction { citizen_id, action, primary_verb, target, speech }) => {
                    if ready {
                        let payload = serde_json::json!({
                            "citizen_id": citizen_id,
                            "action": action,
                            "primary_verb": primary_verb,
                            "target": target,
                            "speech": speech,
                        });
                        if let Ok(json) = serde_json::to_string(&payload) {
                            let _ = webview.evaluate_script(&format!("window.__yumon_sim_action({json})"));
                        }
                    }
                    return;
                }
                Event::UserEvent(RssEvent::ChatReply { citizen_id, citizen_name, text }) => {
                    if ready {
                        let payload = serde_json::json!({
                            "citizen_id": citizen_id,
                            "citizen_name": citizen_name,
                            "text": text,
                        });
                        if let Ok(json) = serde_json::to_string(&payload) {
                            let _ = webview.evaluate_script(&format!("window.__yumon_chat_reply({json})"));
                        }
                    }
                    return;
                }
                Event::UserEvent(RssEvent::Ui(Ipc::SimDecide {
                    citizen_id,
                    citizen_name,
                    vitals,
                    emote,
                    environment,
                    memory,
                    plan,
                })) => {
                    let name = citizen_name.unwrap_or_else(|| "Yumon".into());
                    let _ = tx.send(WorkerRequest::SimDecide {
                        citizen_id,
                        citizen_name: name,
                        vitals,
                        emote,
                        environment,
                        memory,
                        plan,
                    });
                    return;
                }
                Event::UserEvent(RssEvent::Ui(Ipc::Refresh))
                    if state.can_refresh && !state.busy && !state.chat_busy =>
                {
                    if tx.send(WorkerRequest::Refresh).is_ok() {
                        state.busy = true;
                        state.status = "Refreshing Product Hunt...".into();
                        state_changed = true;
                    }
                }
                Event::UserEvent(RssEvent::Ui(Ipc::Chat { message, citizen_id, citizen_name, vitals, emote }))
                    if state.can_refresh && !state.busy && !state.chat_busy =>
                {
                    let message = message.trim();
                    if !message.is_empty() && message.chars().count() <= 4096 {
                        let cid = citizen_id.unwrap_or(0);
                        let cname = citizen_name.unwrap_or_else(|| "Yumon".into());
                        if tx.send(WorkerRequest::Chat {
                            message: message.to_owned(),
                            citizen_id: cid,
                            citizen_name: cname,
                            vitals,
                            emote,
                        }).is_ok() {
                            state.chat_busy = true;
                            state.chat_status = "Yumon is thinking...".into();
                            state_changed = true;
                        }
                    }
                }
                Event::UserEvent(RssEvent::Ui(Ipc::ClearChat)) => {
                    state.chat.clear();
                    state.chat_status = "Say hello or share an idea.".into();
                    let _ = tx.send(WorkerRequest::ClearChat);
                    state_changed = true;
                }
                Event::UserEvent(RssEvent::Ui(Ipc::Drag)) => {
                    let _ = window.drag_window();
                    return;
                }
                Event::UserEvent(RssEvent::Ui(Ipc::Open { link })) => {
                    if state.entries.iter().any(|e| e.article.link == link) {
                        if let Err(error) = open_article(&link) {
                            state.status = format!("{error:#}");
                            state_changed = true;
                        }
                    }
                }
                _ => return,
            }
            if ready && state_changed {
                if let Ok(json) = serde_json::to_string(&state) {
                    let _ = webview.evaluate_script(&format!("window.__yumon_portal({json})"));
                }
            }
        });
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    desktop::run()
}

#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("Yumon RSS requires a native desktop build.");
}
