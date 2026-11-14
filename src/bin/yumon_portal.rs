//! Yumon Portal: the WIR Ant and Chess simulations on the left, chat with Yumon on the right.
#![recursion_limit = "256"]

#[cfg(not(target_arch = "wasm32"))]
mod desktop {
    use anyhow::{Context, Result};
    use clap::Parser;
    use serde::{Deserialize, Serialize};
    use std::{sync::mpsc, thread};
    use tao::{
        dpi::{LogicalSize, PhysicalPosition},
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoopBuilder},
        window::WindowBuilder,
    };
    use wry::WebViewBuilder;
    use yumon_pet::brain::{
        bpe::TokenizerKind,
        decoder_model::YumonDecBrain,
        samples::{language_prompt, language_separator_for, human_prefix_for},
    };

    type BackendType = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;

    const EMBEDDED_MODEL_BYTES: &[u8] = include_bytes!("../../assets/portal_model/model.bin");
    const EMBEDDED_TOKENIZER_BYTES: &[u8] = include_bytes!("../../assets/portal_model/tokenizer.json");
    const EMBEDDED_METADATA_STR: &str = include_str!("../../assets/portal_model/metadata.json");

    const HISTORY_LIMIT: usize = 200;
    const READY_STATUS: &str = "Say hello to Yumon.";

    #[derive(Parser)]
    #[command(about = "Yumon Portal: chat with Yumon and watch the WIR Ant and Chess simulations")]
    struct Args {
        /// Optional path to a decoder-only checkpoint directory (defaults to the embedded
        /// yumon-dec-ngram-white-dwarf model).
        #[arg(long)]
        checkpoint: Option<String>,
    }

    #[derive(Clone, Serialize)]
    struct ViewState {
        chat: Vec<ChatMessage>,
        chat_busy: bool,
        chat_status: String,
        can_chat: bool,
    }

    #[derive(Clone, Serialize, Deserialize)]
    struct ChatMessage {
        role: &'static str,
        text: String,
    }

    enum WorkerRequest {
        Chat { message: String },
        ClearChat,
    }

    #[derive(Debug, Deserialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    enum Ipc {
        Ready,
        Drag,
        Close,
        Chat { message: String },
        ClearChat,
    }

    enum PortalEvent {
        State(ViewState),
        Ui(Ipc),
    }

    type Responder = Box<dyn Fn(&str, &[ChatMessage]) -> Result<String>>;

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
        let first_turn = stripped
            .split("\nHuman:")
            .next()
            .unwrap_or(stripped)
            .split("\nperson:")
            .next()
            .unwrap_or(stripped)
            .split("\nINPUT:")
            .next()
            .unwrap_or(stripped)
            .split("\nREJECTED:")
            .next()
            .unwrap_or(stripped)
            .split("\nCHOSEN:")
            .next()
            .unwrap_or(stripped)
            .trim();

        let unquoted = strip_outer_quotes(first_turn);
        if unquoted.is_empty() {
            first_turn.to_string()
        } else {
            unquoted.to_string()
        }
    }

    fn chat_prompt(
        message: &str,
        history: &[ChatMessage],
        context: usize,
        tokenizer: &TokenizerKind,
    ) -> Result<String> {
        // Reserve up to 256 reply tokens (including EOS), with room for older checkpoints.
        let reply_tokens = (context / 4).clamp(8, 256);
        let sep_tokens = tokenizer.encode(language_separator_for(tokenizer)).len();
        let budget = context
            .checked_sub(1 + sep_tokens + reply_tokens)
            .context("Checkpoint context is too short")?;
        let human_prefix_len = tokenizer.encode(human_prefix_for(tokenizer)).len();
        let message_budget = budget
            .checked_sub(human_prefix_len)
            .context("Checkpoint context is too short")?;
        let message = fit_chat_prompt(message, message_budget, |s| tokenizer.encode(s).len())?;

        let memories: Vec<(&str, &str)> = history
            .windows(2)
            .filter_map(|pair| {
                (pair[0].role == "user" && pair[1].role == "yumon")
                    .then(|| (pair[0].text.as_str(), pair[1].text.as_str()))
            })
            .collect();
        language_prompt(&message, memories.into_iter(), tokenizer, budget)
            .context("Checkpoint context cannot fit your message")
    }

    fn answer_chat(state: &mut ViewState, message: String, responder: &Responder) {
        match responder(&message, &state.chat) {
            Ok(reply) => {
                state.chat.push(ChatMessage { role: "yumon", text: reply });
                state.chat_status = READY_STATUS.into();
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

    fn load_responder(args: &Args) -> Result<Responder> {
        let device = Default::default();
        let (brain, tokenizer, config) = match &args.checkpoint {
            Some(path) => {
                println!("[Yumon Portal] Loading checkpoint from {path}");
                YumonDecBrain::<BackendType>::load(path, &device)?
            }
            None => {
                println!("[Yumon Portal] Loading embedded yumon-dec-ngram-white-dwarf checkpoint.");
                YumonDecBrain::<BackendType>::load_from_bytes(
                    EMBEDDED_MODEL_BYTES,
                    EMBEDDED_TOKENIZER_BYTES,
                    EMBEDDED_METADATA_STR,
                    &device,
                )?
            }
        };
        // Fail once, visibly, when the context cannot hold even a short message.
        chat_prompt("hi", &[], config.max_seq_len, &tokenizer)?;
        Ok(Box::new(move |text: &str, history: &[ChatMessage]| {
            let prompt = chat_prompt(text, history, config.max_seq_len, &tokenizer)?;
            let reply = brain
                .generate_unmasked_parsed(&tokenizer, &prompt, config.max_seq_len.min(256), &device)
                .raw_output;
            let reply = match reply
                .split_once("\nHuman:")
                .or_else(|| reply.split_once("\nperson:"))
            {
                Some((clean, _)) => clean.trim().to_owned(),
                None => reply.trim().to_owned(),
            };
            anyhow::ensure!(!reply.is_empty(), "Yumon returned an empty reply");
            Ok(parse_chat_response_loose(&reply))
        }))
    }

    fn worker(
        args: Args,
        mut state: ViewState,
        rx: mpsc::Receiver<WorkerRequest>,
        proxy: tao::event_loop::EventLoopProxy<PortalEvent>,
    ) {
        let publish = |state: &ViewState| proxy.send_event(PortalEvent::State(state.clone())).is_ok();
        let responder = match load_responder(&args) {
            Ok(responder) => responder,
            Err(error) => {
                state.chat_status = format!("Could not load Yumon: {error:#}");
                publish(&state);
                return;
            }
        };
        state.can_chat = true;
        state.chat_status = READY_STATUS.into();
        if !publish(&state) {
            return;
        }
        while let Ok(request) = rx.recv() {
            match request {
                WorkerRequest::ClearChat => {
                    state.chat.clear();
                    state.chat_status = READY_STATUS.into();
                }
                WorkerRequest::Chat { message } => {
                    state.chat_busy = true;
                    state.chat_status = "Yumon is thinking...".into();
                    state.chat.push(ChatMessage { role: "user", text: message.clone() });
                    if !publish(&state) {
                        return;
                    }
                    answer_chat(&mut state, message, &responder);
                }
            }
            if !publish(&state) {
                return;
            }
        }
    }

    /// The two simulation pages, as JSON string literals that are safe inside a <script>.
    fn sim_pages_script() -> String {
        // Escaping every '<' also rules out `<!--` / `<script` double-escape states.
        let literal = |html: &str| serde_json::to_string(html).unwrap().replace('<', "\\u003c");
        format!(
            "window.__simPages = {{ ant: {}, chess: {} }};",
            literal(include_str!("sims/yumon-ant.html")),
            literal(include_str!("sims/wir-chess.html")),
        )
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn tokenizer() -> TokenizerKind {
            TokenizerKind::Bpe(
                yumon_pet::brain::bpe::BpeTokenizer::load_from_bytes(EMBEDDED_TOKENIZER_BYTES).unwrap(),
            )
        }

        fn chat(role: &'static str, text: &str) -> ChatMessage {
            ChatMessage { role, text: text.into() }
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
                "Human: What color?"
            );
            assert_eq!(
                chat_prompt("What color?", &history[2..4], 256, &tokenizer).unwrap(),
                "Human: What color?"
            );
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
            let sep = tokenizer.encode(language_separator_for(&tokenizer)).len();
            let long = "Hello 🐾 friend! ".repeat(200);
            for context in [32, 256] {
                let prompt = chat_prompt(&long, &history, context, &tokenizer).unwrap();
                assert!(1 + tokenizer.encode(&prompt).len() + sep + (context / 4).clamp(8, 256) <= context);
            }
            assert!(chat_prompt("Hello", &history, 8, &tokenizer).is_err());
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
        fn chat_replies_and_failures_are_recorded() {
            let mut state = ViewState {
                chat: vec![chat("user", "Hello Yumon!")],
                chat_busy: true,
                chat_status: "Thinking".into(),
                can_chat: true,
            };
            let responder: Responder = Box::new(|text, history| {
                assert_eq!(text, "Hello Yumon!");
                assert_eq!(history[0].text, "Hello Yumon!");
                Ok("Hello friend!".into())
            });
            answer_chat(&mut state, "Hello Yumon!".into(), &responder);
            assert_eq!(state.chat[1].text, "Hello friend!");
            assert_eq!(state.chat[1].role, "yumon");
            assert!(!state.chat_busy);
            let failed: Responder = Box::new(|_, _| anyhow::bail!("GPU unavailable"));
            state.chat_busy = true;
            answer_chat(&mut state, "Try again".into(), &failed);
            assert_eq!(state.chat[2].role, "error");
            assert!(state.chat[2].text.contains("GPU unavailable"));
            assert!(!state.chat_busy);
        }

        #[test]
        fn loose_response_parsing_chat() {
            assert_eq!(
                parse_chat_response_loose("Reply: \"My tummy rumbles loud! Please feed me now.\""),
                "My tummy rumbles loud! Please feed me now."
            );
            assert_eq!(
                parse_chat_response_loose("Reply: Yes! I am ready to sprint right now!"),
                "Yes! I am ready to sprint right now!"
            );
            assert_eq!(
                parse_chat_response_loose("\"I would love a delicious snack!\""),
                "I would love a delicious snack!"
            );
            assert_eq!(
                parse_chat_response_loose("Reply: \"I am ready to play!\"\nperson: what game?\nyumon: fetch!"),
                "I am ready to play!"
            );
        }

        #[test]
        fn embedded_assets_are_valid() {
            let meta: serde_json::Value = serde_json::from_str(EMBEDDED_METADATA_STR).unwrap();
            assert_eq!(meta["training_stage"], "Language");
            assert_eq!(meta["vocab_size"], 8192);
            assert!(EMBEDDED_MODEL_BYTES.len() > 1_000_000);
            assert!(EMBEDDED_TOKENIZER_BYTES.len() > 1_000);
        }

        #[test]
        fn simulation_pages_embed_as_script_safe_literals() {
            let script = sim_pages_script();
            assert!(script.starts_with("window.__simPages = { ant: \""));
            assert!(!script.contains('<'), "a literal would close the host <script>");
            assert!(script.len() > 800_000);
        }

        #[test]
        fn embedded_model_inference_smoke() {
            let args = Args { checkpoint: None };
            let responder = load_responder(&args).expect("embedded model should load");
            let reply = responder("Hello", &[]).expect("inference with embedded model should succeed");
            assert!(!reply.trim().is_empty());
        }
    }

    pub fn run() -> Result<()> {
        let args = Args::parse();
        let state = ViewState {
            chat: Vec::new(),
            chat_busy: false,
            chat_status: "Waking Yumon up...".into(),
            can_chat: false,
        };
        let event_loop = EventLoopBuilder::<PortalEvent>::with_user_event().build();
        let proxy = event_loop.create_proxy();
        let mut builder = WindowBuilder::new()
            .with_title("Yumon Portal")
            .with_inner_size(LogicalSize::new(1100, 760))
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
                origin.y + (size.height as i32 - (820.0 * scale) as i32).max(0),
            ));
        }
        #[cfg(target_os = "windows")]
        {
            use tao::platform::windows::WindowBuilderExtWindows;
            builder = builder.with_undecorated_shadow(false);
        }
        let window = builder.build(&event_loop)?;
        let ipc_proxy = proxy.clone();
        let ui_html = include_str!("portal_ui.html").replace("/* __SIM_PAGES__ */", &sim_pages_script());
        let wv_builder = WebViewBuilder::new()
            .with_transparent(true)
            .with_html(&ui_html)
            .with_new_window_req_handler(|_, _| wry::NewWindowResponse::Deny)
            .with_ipc_handler(move |request| {
                if let Ok(ipc) = serde_json::from_str::<Ipc>(request.body()) {
                    let _ = ipc_proxy.send_event(PortalEvent::Ui(ipc));
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
        let worker_proxy = proxy.clone();
        thread::spawn(move || worker(args, worker_state, rx, worker_proxy));

        let mut state = state;
        let mut ready = false;
        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Wait;
            let mut state_changed = false;
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                }
                | Event::UserEvent(PortalEvent::Ui(Ipc::Close)) => {
                    *control_flow = ControlFlow::Exit;
                    return;
                }
                Event::UserEvent(PortalEvent::Ui(Ipc::Ready)) => {
                    ready = true;
                    state_changed = true;
                }
                Event::UserEvent(PortalEvent::State(update)) => {
                    state = update;
                    state_changed = true;
                }
                Event::UserEvent(PortalEvent::Ui(Ipc::Chat { message })) if state.can_chat && !state.chat_busy => {
                    let message = message.trim();
                    if !message.is_empty() && message.chars().count() <= 4096 {
                        if tx.send(WorkerRequest::Chat { message: message.to_owned() }).is_ok() {
                            state.chat_busy = true;
                            state.chat_status = "Yumon is thinking...".into();
                            state_changed = true;
                        }
                    }
                }
                Event::UserEvent(PortalEvent::Ui(Ipc::ClearChat)) => {
                    state.chat.clear();
                    state.chat_status = READY_STATUS.into();
                    let _ = tx.send(WorkerRequest::ClearChat);
                    state_changed = true;
                }
                Event::UserEvent(PortalEvent::Ui(Ipc::Drag)) => {
                    let _ = window.drag_window();
                    return;
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
    eprintln!("Yumon Portal requires a native desktop build.");
}
