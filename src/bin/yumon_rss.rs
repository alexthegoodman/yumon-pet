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
        model::YumonBrain,
        moe_model::YumonMoeBrain,
        samples::{TrainingStage, language_prompt},
        xlstm_model::YumonXLstmBrain,
    };

    #[derive(Clone, Copy, ValueEnum)]
    enum Architecture {
        Moe,
        Xlstm,
        EncoderDecoder,
    }

    #[derive(Parser)]
    #[command(about = "Yumon RSS: daily Product Hunt ideas with local commentary")]
    struct Args {
        #[arg(
            long,
            default_value = "D:/models/runpod/large1/512h_16l_8a_256len_b32_Moe_e4_k1_Language_1m"
        )]
        checkpoint: String,
        #[arg(long, value_enum, default_value = "moe")]
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
    }

    #[derive(Clone, Serialize)]
    struct ChatMessage {
        role: &'static str,
        text: String,
    }

    enum WorkerRequest {
        Refresh,
        Chat(String),
        ClearChat,
    }

    #[derive(Debug, Deserialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    enum Ipc {
        Ready,
        Refresh,
        Drag,
        Close,
        Open { link: String },
        Chat { message: String },
        ClearChat,
    }

    enum RssEvent {
        State(ViewState),
        Ui(Ipc),
    }

    enum PromptKind {
        Product,
        Chat,
    }

    type Responder = Box<dyn Fn(&str, PromptKind, &[ChatMessage]) -> Result<String>>;

    /// Keep room for BOS, the Language separator, and at least eight reply tokens.
    /// Trim at UTF-8 boundaries and re-encode, since BPE merges can change at a cut.
    fn comment_prompt(title: &str, context: usize, tokenizer: &TokenizerKind) -> Result<String> {
        let overhead = 1 + tokenizer.encode(" ").len() + 8;
        let budget = context
            .checked_sub(overhead)
            .context("Checkpoint context is too short")?;
        fit_prompt(title, budget, |text| tokenizer.encode(text).len())
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

    fn answer_chat(state: &mut ViewState, message: String, responder: &Responder) {
        match responder(&message, PromptKind::Chat, &state.chat) {
            Ok(reply) => {
                state.chat.push(ChatMessage {
                    role: "yumon",
                    text: reply,
                });
                state.chat_status = "Say hello or share an idea.".into();
            }
            Err(error) => {
                state.chat.push(ChatMessage {
                    role: "error",
                    text: format!("Could not reply: {error:#}"),
                });
                state.chat_status = "Try sending your message again.".into();
            }
        }
        if state.chat.len() > HISTORY_LIMIT {
            state.chat.drain(..state.chat.len() - HISTORY_LIMIT);
        }
        state.chat_busy = false;
    }

    fn chat_prompt(
        message: &str,
        history: &[ChatMessage],
        context: usize,
        tokenizer: &TokenizerKind,
    ) -> Result<String> {
        // Reserve up to 64 reply tokens (including EOS), with room for older checkpoints.
        let reply_tokens = (context / 4).clamp(8, 64);
        let budget = context
            .checked_sub(1 + tokenizer.encode(" ").len() + reply_tokens)
            .context("Checkpoint context is too short")?;
        let message = fit_chat_prompt(message, budget, |s| tokenizer.encode(s).len())?;
        // Only adjacent, successful exchanges form memories. The pending current
        // user message and failed replies never form a pair.
        let memories = history.windows(2).filter_map(|pair| {
            (pair[0].role == "user" && pair[1].role == "yumon")
                .then_some((pair[0].text.as_str(), pair[1].text.as_str()))
        });
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
            let prompt = chat_prompt("What color?", &history, 256, &tokenizer).unwrap();
            assert_eq!(
                prompt,
                "Human: My favorite color is blue.\nYumon: I like blue too.\nHuman: Remember it?\nYumon: Yes, blue.\nHuman: What color?"
            );
            assert_eq!(
                chat_prompt("What color?", &[], 256, &tokenizer).unwrap(),
                "What color?"
            );
            assert_eq!(
                chat_prompt("What color?", &history[2..4], 256, &tokenizer).unwrap(),
                "What color?"
            );
            assert!(1 + tokenizer.encode(&prompt).len() + tokenizer.encode(" ").len() + 64 <= 256);
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
            let prompt = chat_prompt("What color?", &history, 256, &tokenizer).unwrap();
            assert_eq!(
                prompt,
                "Human: My favorite color is blue.\nYumon: I like blue too.\nHuman: What color?"
            );
            let long = "Hello 🐾 friend! ".repeat(200);
            for context in [32, 256] {
                let prompt = chat_prompt(&long, &history, context, &tokenizer).unwrap();
                assert!(long.starts_with(&prompt));
                assert!(
                    1 + tokenizer.encode(&prompt).len()
                        + tokenizer.encode(" ").len()
                        + (context / 4).clamp(8, 64)
                        <= context
                );
            }
            assert!(chat_prompt("Hello", &history, 8, &tokenizer).is_err());
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
            };
            let responder: Responder = Box::new(|text, kind, history| {
                assert!(matches!(kind, PromptKind::Chat));
                assert_eq!(text, "Hello Yumon!");
                assert_eq!(history[0].text, "Hello Yumon!");
                Ok("Hello friend!".into())
            });
            answer_chat(&mut state, "Hello Yumon!".into(), &responder);
            assert_eq!(state.chat[1].text, "Hello friend!");
            assert_eq!(state.chat[1].role, "yumon");
            assert!(!state.chat_busy);
            let failed: Responder = Box::new(|_, _, _| anyhow::bail!("GPU unavailable"));
            state.chat_busy = true;
            answer_chat(&mut state, "Try again".into(), &failed);
            assert_eq!(state.chat[2].role, "error");
            assert!(state.chat[2].text.contains("GPU unavailable"));
            assert!(!state.chat_busy);
            assert_eq!(state.next_check, next_check);
            assert_eq!(state.status, "Products waiting");
            assert!(state.entries.is_empty());
        }
    }

    fn load_responder(args: &Args) -> Result<Responder> {
        let metadata: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
            PathBuf::from(&args.checkpoint).join("metadata.json"),
        )?)?;
        anyhow::ensure!(
            metadata["training_stage"] == "Language",
            "RSS requires a Language-stage checkpoint"
        );
        let device = Default::default();
        macro_rules! load {
            ($model:ty, $generate:expr) => {{
                let (brain, tokenizer, config) = <$model>::load(&args.checkpoint, &device)?;
                anyhow::ensure!(
                    config.training_stage == TrainingStage::Language,
                    "Expected Language stage"
                );
                // Check before fetching so an unusable context produces one visible error.
                comment_prompt("idea", config.max_seq_len, &tokenizer)?;
                Ok(Box::new(
                    move |text: &str, kind: PromptKind, history: &[ChatMessage]| {
                        let prompt = match kind {
                            PromptKind::Product => {
                                comment_prompt(text, config.max_seq_len, &tokenizer)?
                            }
                            PromptKind::Chat => {
                                chat_prompt(text, history, config.max_seq_len, &tokenizer)?
                            }
                        };
                        let generate: fn(&$model, &TokenizerKind, &str, usize, &_) -> String =
                            $generate;
                        let reply = generate(
                            &brain,
                            &tokenizer,
                            &prompt,
                            config.max_seq_len.min(64),
                            &device,
                        );
                        let reply = reply.trim().to_owned();
                        anyhow::ensure!(!reply.is_empty(), "Yumon returned an empty reply");
                        Ok(reply)
                    },
                ) as Responder)
            }};
        }
        match args.architecture {
            Architecture::Moe => load!(YumonMoeBrain<Wgpu>, |b, t, p, n, d| b
                .generate_unmasked_parsed(t, p, n, d)
                .raw_output),
            Architecture::Xlstm => load!(YumonXLstmBrain<Wgpu>, |b, t, p, n, d| b
                .generate_unmasked_parsed(t, p, n, d)
                .raw_output),
            Architecture::EncoderDecoder => load!(YumonBrain<Wgpu>, |b, t, p, n, d| b
                .generate_unmasked_parsed::<cubecl::wgpu::WgpuRuntime>(t, p, n, d)
                .raw_output),
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
            if !publish(&state) {
                return;
            }
            loop {
                match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                    Ok(WorkerRequest::Refresh) => {
                        refresh = true;
                        break;
                    }
                    Ok(WorkerRequest::Chat(message)) => {
                        state.chat_busy = true;
                        state.chat_status = "Yumon is thinking...".into();
                        state.chat.push(ChatMessage {
                            role: "user",
                            text: message.clone(),
                        });
                        if !publish(&state) {
                            return;
                        }
                        answer_chat(&mut state, message, &responder);
                        if !publish(&state) {
                            return;
                        }
                        // Chat uses the same model but leaves the product deadline intact.
                        if Instant::now() >= deadline {
                            break;
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
        };
        let event_loop = EventLoopBuilder::<RssEvent>::with_user_event().build();
        let proxy = event_loop.create_proxy();
        let mut builder = WindowBuilder::new()
            .with_title("Yumon RSS")
            .with_inner_size(LogicalSize::new(360, 520))
            .with_decorations(false)
            .with_transparent(true)
            .with_always_on_top(true)
            .with_resizable(false);
        if let Some(monitor) = event_loop
            .primary_monitor()
            .or_else(|| event_loop.available_monitors().next())
        {
            let scale = monitor.scale_factor();
            let origin = monitor.position();
            let size = monitor.size();
            builder = builder.with_position(PhysicalPosition::new(
                origin.x + (16.0 * scale) as i32,
                origin.y + (size.height as i32 - (584.0 * scale) as i32).max(0),
            ));
        }
        #[cfg(target_os = "windows")]
        {
            use tao::platform::windows::WindowBuilderExtWindows;
            builder = builder.with_undecorated_shadow(false);
        }
        let window = builder.build(&event_loop)?;
        let ipc_proxy = proxy.clone();
        let wv_builder = WebViewBuilder::new()
            .with_transparent(true)
            .with_html(include_str!("rss_ui.html"))
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
        let (tx, rx) = mpsc::sync_channel(1);
        let worker_state = state.clone();
        thread::spawn(move || {
            worker(
                args,
                history_path,
                cache_path,
                cache,
                worker_state,
                rx,
                proxy,
            )
        });
        let mut ready = false;
        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Wait;
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                }
                | Event::UserEvent(RssEvent::Ui(Ipc::Close)) => {
                    *control_flow = ControlFlow::Exit;
                    return;
                }
                Event::UserEvent(RssEvent::Ui(Ipc::Ready)) => ready = true,
                Event::UserEvent(RssEvent::State(update)) => state = update,
                Event::UserEvent(RssEvent::Ui(Ipc::Refresh))
                    if state.can_refresh && !state.busy && !state.chat_busy =>
                {
                    if tx.try_send(WorkerRequest::Refresh).is_ok() {
                        state.busy = true;
                        state.status = "Refreshing Product Hunt...".into();
                    }
                }
                Event::UserEvent(RssEvent::Ui(Ipc::Chat { message }))
                    if state.can_refresh && !state.busy && !state.chat_busy =>
                {
                    let message = message.trim();
                    if !message.is_empty() && message.chars().count() <= 4096 {
                        if tx.try_send(WorkerRequest::Chat(message.to_owned())).is_ok() {
                            state.chat_busy = true;
                            state.chat_status = "Yumon is thinking...".into();
                        }
                    }
                }
                Event::UserEvent(RssEvent::Ui(Ipc::ClearChat))
                    if !state.busy && !state.chat_busy =>
                {
                    let _ = tx.try_send(WorkerRequest::ClearChat);
                }
                Event::UserEvent(RssEvent::Ui(Ipc::Drag)) => {
                    let _ = window.drag_window();
                    return;
                }
                Event::UserEvent(RssEvent::Ui(Ipc::Open { link })) => {
                    if state.entries.iter().any(|e| e.article.link == link) {
                        if let Err(error) = open_article(&link) {
                            state.status = format!("{error:#}");
                        }
                    }
                }
                _ => return,
            }
            if ready {
                if let Ok(json) = serde_json::to_string(&state) {
                    let _ = webview.evaluate_script(&format!("window.__yumon_rss({json})"));
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
