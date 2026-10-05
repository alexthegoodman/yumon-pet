//! A quiet desktop news companion: RSS headlines and local Yumon commentary.
#![recursion_limit = "256"]

#[cfg(not(target_arch = "wasm32"))]
#[path = "rss/mod.rs"]
mod rss;

#[cfg(not(target_arch = "wasm32"))]
mod desktop {
    use super::rss::{self, Entry, HISTORY_LIMIT, POLL_INTERVAL, SOURCES};
    use anyhow::{Context, Result};
    use burn::backend::Wgpu;
    use chrono::{DateTime, Utc};
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
        bpe::TokenizerKind, model::YumonBrain, moe_model::YumonMoeBrain, samples::TrainingStage,
        xlstm_model::YumonXLstmBrain,
    };

    #[derive(Clone, Copy, ValueEnum)]
    enum Architecture {
        Moe,
        Xlstm,
        EncoderDecoder,
    }

    #[derive(Parser)]
    #[command(about = "Yumon RSS: desktop headlines with a little local commentary")]
    struct Args {
        #[arg(
            long,
            default_value = "D:/models/runpod/256h_32l_4a_32len_b32_Moe_e4_k1_Language_600k"
        )]
        checkpoint: String,
        #[arg(long, value_enum, default_value = "moe")]
        architecture: Architecture,
        /// JSON history location (defaults to YumonRSS's local application data).
        #[arg(long)]
        history: Option<PathBuf>,
        /// Fetch and print the latest headlines without loading a model or window.
        #[arg(long)]
        check_feeds: bool,
    }

    #[derive(Clone, Serialize)]
    struct ViewState {
        entries: Vec<Entry>,
        status: String,
        busy: bool,
        can_refresh: bool,
        next_check: Option<DateTime<Utc>>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    enum Ipc {
        Ready,
        Refresh,
        Drag,
        Close,
        Open { link: String },
    }

    enum RssEvent {
        State(ViewState),
        Ui(Ipc),
    }

    type Commenter = Box<dyn Fn(&str) -> Result<String>>;

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
                "Checkpoint context cannot fit a news headline"
            );
            let prompt = format!("Your thoughts on {headline}?");
            if token_count(&prompt) <= budget {
                return Ok(prompt);
            }
            headline.pop();
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

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
    }

    fn load_commenter(args: &Args) -> Result<Commenter> {
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
                comment_prompt("news", config.max_seq_len, &tokenizer)?;
                Ok(Box::new(move |title: &str| {
                    let prompt = comment_prompt(title, config.max_seq_len, &tokenizer)?;
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
                    anyhow::ensure!(!reply.is_empty(), "Yumon returned an empty comment");
                    Ok(reply)
                }) as Commenter)
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
        mut state: ViewState,
        rx: mpsc::Receiver<()>,
        proxy: EventLoopProxy<RssEvent>,
    ) {
        let publish = |state: &ViewState| proxy.send_event(RssEvent::State(state.clone())).is_ok();
        let commenter = match load_commenter(&args) {
            Ok(commenter) => commenter,
            Err(error) => {
                state.status = format!("Could not load Yumon: {error:#}");
                state.busy = false;
                publish(&state);
                return;
            }
        };
        state.can_refresh = true;
        let agent = agent();
        loop {
            state.busy = true;
            state.next_check = None;
            let mut added = 0;
            let mut errors = Vec::new();
            for source in SOURCES {
                state.status = format!("Checking {}...", source.name);
                if !publish(&state) {
                    return;
                }
                let article = match rss::fetch_latest(&agent, source) {
                    Ok(Some(article)) => article,
                    Ok(None) => {
                        errors.push(format!("{}: no articles", source.name));
                        continue;
                    }
                    Err(error) => {
                        errors.push(format!("{}: {error:#}", source.name));
                        continue;
                    }
                };
                if rss::already_seen(&state.entries, &article) {
                    continue;
                }
                state.status = format!("Yumon is reading {}...", source.name);
                if !publish(&state) {
                    return;
                }
                let comment = match commenter(&article.title) {
                    Ok(comment) => comment,
                    Err(error) => {
                        errors.push(format!("{}: {error:#}", source.name));
                        continue;
                    }
                };
                state.entries.push(Entry {
                    source: source.name.into(),
                    article,
                    comment,
                    received: Utc::now(),
                });
                if state.entries.len() > HISTORY_LIMIT {
                    state.entries.remove(0);
                }
                added += 1;
                if let Err(error) = rss::save_history(&history_path, &state.entries) {
                    errors.push(format!("History: {error:#}"));
                }
                if !publish(&state) {
                    return;
                }
            }
            state.busy = false;
            let deadline = Instant::now() + POLL_INTERVAL;
            state.next_check = Some(Utc::now() + chrono::Duration::minutes(15));
            state.status = if errors.is_empty() {
                if added == 0 {
                    "All caught up. No new headlines.".into()
                } else {
                    format!(
                        "{added} new headline{}. All caught up.",
                        if added == 1 { "" } else { "s" }
                    )
                }
            } else {
                format!("{} Will retry on the next check.", errors.join(" · "))
            };
            if !publish(&state) {
                return;
            }
            match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                Ok(()) | Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
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
        if args.check_feeds {
            let agent = agent();
            let mut failed = false;
            for source in SOURCES {
                match rss::fetch_latest(&agent, source) {
                    Ok(Some(article)) => {
                        println!("{}: {}\n{}", source.name, article.title, article.link)
                    }
                    Ok(None) => {
                        eprintln!("{}: no articles", source.name);
                        failed = true;
                    }
                    Err(error) => {
                        eprintln!("{}: {error:#}", source.name);
                        failed = true;
                    }
                }
            }
            anyhow::ensure!(!failed, "Some feeds could not be read");
            return Ok(());
        }
        let history_path = args.history.clone().unwrap_or_else(|| {
            directories::ProjectDirs::from("", "Yumon", "YumonRSS")
                .map(|dirs| dirs.data_local_dir().join("history.json"))
                .unwrap_or_else(|| PathBuf::from("yumon-rss-history.json"))
        });
        
        // Never overwrite an unreadable history with an empty one.
        // let entries: Vec<Entry> = rss::load_history(&history_path)?;
        let entries: Vec<Entry> = Vec::new(); // dont want to load old history yet

        let mut state = ViewState {
            entries,
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
        thread::spawn(move || worker(args, history_path, worker_state, rx, proxy));
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
                    if state.can_refresh && !state.busy =>
                {
                    if tx.try_send(()).is_ok() {
                        state.busy = true;
                        state.status = "Checking the news...".into();
                    }
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
