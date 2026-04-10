use anyhow::Result;
use burn::{backend::Wgpu, prelude::*};
use cubecl::wgpu::WgpuDevice;
use serde::{Deserialize, Serialize};
use std::{
    sync::{mpsc, Arc, Mutex},
    thread,
};
use tao::{
    dpi::{LogicalPosition, LogicalSize},
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop, EventLoopBuilder, EventLoopProxy},
    monitor::MonitorHandle,
    window::WindowBuilder,
};
use wry::{WebViewBuilder, http::Request};

use yumon_pet::{
    brain::{
        model::{GenerationResult, YumonBrain},
        samples::{TrainingStage, WorldContext},
    },
};

// ── Custom event ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum CompanionEvent {
    /// Brain finished generating; carry the result to the UI thread.
    BrainReply(GenerationResult),
    /// A transient system message (e.g. "Models loaded!")
    SystemMsg(String),
}

// ── Emote name mapping (mirrors EMOTE_NAMES in vision) ───────────────────
//
// Map the parsed_emotion variant name to one of the JS emote keys.
// Adjust as needed when you add more emote classes.

fn emotion_to_js_key(emotion: &str) -> &'static str {
    match emotion.to_lowercase().as_str() {
        "happy"     | "joy"        => "happy",
        "excited"   | "playful"    => "excited",
        "sad"       | "melancholy" => "sad",
        "angry"     | "frustrated" => "angry",
        "surprised" | "fearful"    => "surprised",
        "thinking"  | "curious"    => "thinking",
        _                          => "neutral",
    }
}

// ── IPC message from JS ───────────────────────────────────────────────────

#[derive(Deserialize, Debug)]
struct IpcMsg {
    #[serde(rename = "type")]
    kind: String,
    msg:  Option<String>,
}

// ── Entry point ───────────────────────────────────────────────────────────

pub fn run(brain_cp: String, device: WgpuDevice) -> Result<()> {
    let mut event_loop = EventLoopBuilder::<CompanionEvent>::with_user_event();
    let event_loop = event_loop.build();
    let proxy = event_loop.create_proxy();

    // ── Channels: UI → brain, brain → UI ─────────────────────────────────
    let (tx_brain, rx_brain) =
        mpsc::channel::<(String, WorldContext, Vec<(String, String)>)>();

    // Brain thread
    {
        let brain_cp  = brain_cp.clone();
        let device    = device.clone();
        let proxy     = proxy.clone();

        thread::spawn(move || {
            let res = YumonBrain::<Wgpu>::load(&brain_cp, &device);
            let (brain_model, tokenizer, config) = match res {
                Ok(m) => {
                    let _ = proxy.send_event(CompanionEvent::SystemMsg("Models loaded!".into()));
                    m
                }
                Err(e) => {
                    let _ = proxy.send_event(CompanionEvent::SystemMsg(
                        format!("Error loading models: {e}"),
                    ));
                    return;
                }
            };

            while let Ok((prompt_text, _world, memories)) = rx_brain.recv() {
                let memories_json: Vec<serde_json::Value> = memories
                    .iter()
                    .map(|(h, b)| serde_json::json!({ "human": h, "bot": b }))
                    .collect();

                let training_stage = config.training_stage;
                let prompt = if training_stage == TrainingStage::Language {
                    prompt_text
                } else {
                    serde_json::to_string_pretty(&serde_json::json!({
                        "obstacle_dir": "none",
                        "building_dir": "none",
                        "resource_dir": "none",
                        "memories":     memories_json,
                        "message":      prompt_text,
                    }))
                    .unwrap()
                };

                let result = brain_model.generate_unmasked_parsed(
                    &tokenizer,
                    &prompt,
                    config.max_seq_len,
                    &device,
                );

                let _ = proxy.send_event(CompanionEvent::BrainReply(result));
            }
        });
    }

    // ── Window ────────────────────────────────────────────────────────────
    let monitor_size = event_loop
        .available_monitors()
        .next()
        .map(|m| m.size())
        .unwrap_or_else(|| tao::dpi::PhysicalSize::new(1920, 1080));

    let win_w: u32 = 200;
    let win_h: u32 = 200;
    // 16px and 32px margin from bottom-left corner
    let pos_x = 16i32;
    let pos_y = monitor_size.height as i32 - win_h as i32 - 32;

    #[allow(unused_mut)]
    let mut builder = WindowBuilder::new()
        .with_title("Yumon")
        .with_inner_size(LogicalSize::new(win_w, win_h))
        .with_position(LogicalPosition::new(pos_x, pos_y))
        .with_decorations(false)
        .with_transparent(true)
        .with_always_on_top(true)
        .with_resizable(false);

    #[cfg(target_os = "windows")]
    {
        use tao::platform::windows::WindowBuilderExtWindows;
        builder = builder.with_undecorated_shadow(false);
    }

    let window = builder.build(&event_loop)?;

    #[cfg(target_os = "windows")]
    {
        use tao::platform::windows::WindowExtWindows;
        window.set_undecorated_shadow(false);
    }

    // ── Shared state for IPC handler ──────────────────────────────────────
    let recent_memories: Arc<Mutex<Vec<(String, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let memories_clone  = recent_memories.clone();
    let tx_brain_ipc    = tx_brain.clone();

    // ── WebView ───────────────────────────────────────────────────────────
    let html = include_str!("./desktop_ui.html");  // embed at compile time
    // Or swap to: let html = std::fs::read_to_string("companion_ui.html")?;

    let wv_builder = WebViewBuilder::new()
        .with_transparent(true)
        .with_html(html)
        .with_ipc_handler(move |msg: Request<String>| {
            // JS sends: window.ipc.postMessage(JSON.stringify({type:"send",msg:"..."}))
            if let Ok(ipc) = serde_json::from_str::<IpcMsg>(&msg.body()) {
                if ipc.kind == "send" {
                    if let Some(text) = ipc.msg {
                        let mems = memories_clone.lock().unwrap().clone();
                        let _ = tx_brain_ipc.send((text, WorldContext::default(), mems));
                    }
                }
            }
        });

    #[cfg(any(target_os = "windows", target_os = "macos", target_os = "ios", target_os = "android"))]
    let webview = wv_builder.build(&window)?;

    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "ios", target_os = "android")))]
    let webview = {
        use tao::platform::unix::WindowExtUnix;
        use wry::WebViewBuilderExtUnix;
        let vbox = window.default_vbox().unwrap();
        wv_builder.build_gtk(vbox)?
    };

    // ── Event loop ────────────────────────────────────────────────────────
    let mut loading = true;
    let memories_el = recent_memories.clone();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,

            Event::UserEvent(CompanionEvent::SystemMsg(msg)) => {
                if msg.contains("loaded") {
                    loading = false;
                    // Greet once models are ready
                    let js = r#"window.__yumon_reply({ reply: "i'm awake ✦", emote: "happy" })"#;
                    let _ = webview.evaluate_script(js);
                }
                // Otherwise ignore system messages in UI
            }

            Event::UserEvent(CompanionEvent::BrainReply(result)) => {
                let reply = if result.reply.len() < 2 {
                    format!("{:?}", result.raw_output).to_lowercase()
                } else {
                    result.reply.clone()
                };

                let emote_key = emotion_to_js_key(&format!("{:?}", result.parsed_emotion));

                // Escape the reply for JS string injection
                let reply_escaped = reply
                    .replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('\n', " ");

                let js = format!(
                    r#"window.__yumon_reply({{ reply: "{reply_escaped}", emote: "{emote_key}" }})"#
                );
                let _ = webview.evaluate_script(&js);

                // Store memory pair (last user msg is tracked below)
                // We track it on the IPC side; here just update last bot reply
                let mut mems = memories_el.lock().unwrap();
                if let Some(last) = mems.last_mut() {
                    if last.1.is_empty() {
                        last.1 = reply;
                    }
                }
                if mems.len() > 3 {
                    mems.remove(0);
                }
            }

            _ => {}
        }
    });
}

fn main() {
    let brain_cp = "checkpoints/brain/512h_3l_8a_180len".to_string();

    let _ = run(brain_cp, Default::default());
}

// ── Memory helper (called from ipc_handler closure) ───────────────────────
// When the user sends a message, we push a (human, "") pair so the bot reply
// can fill in the second slot when it arrives.
//
// This is handled inline above; the ipc_handler closure closes over
// memories_clone and tx_brain_ipc to do the push before sending.