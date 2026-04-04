#![recursion_limit = "256"]

use anyhow::Result;
use burn::{backend::Wgpu, prelude::*};
use cubecl::wgpu::WgpuDevice;
use ratzilla::{DomBackend, event::{KeyCode, KeyEvent}, WebRenderer};
use ratzilla::ratatui::{
    layout::{Constraint, Direction, Layout}, style::{Color, Modifier, Style}, text::{Line, Span, Text}, widgets::{Block, Borders, List, ListItem, Paragraph},
    Terminal,
};
use textwrap::wrap;
use std::{io, sync::{mpsc, Arc, Mutex}, time::{Duration, Instant}};
use rand::Rng;
use wasm_bindgen_futures::spawn_local;

use yumon_pet::{
    brain::{
        model::{GenerationResult, YUMON_SCHEMA, YumonBrain}, samples::{TrainingStage, WorldContext}, train::MAX_SEQ_LEN,
    },
    vision::{self, CIFAR_CLASSES, EMOTE_CLASSES, EMOTE_NAMES},
};

#[derive(Clone, Debug)]
enum Message {
    User(String),
    Yumon(GenerationResult),
    System(String),
}

#[derive(Default, Clone)]
struct AgentState {
    action:     String,
    motion_dir: String,
    emote:      String,
    raw_output: String,
    fsm_state:       u32,
    allowed_count:   Option<usize>,
}

struct AppState {
    input:               String,
    messages:            Vec<Message>,
    loading:             bool,
    brain_cp:            String,
    device:              WgpuDevice,
    last_yumon_speak:    Instant,
    next_speak_interval: Duration,
    agent:               AgentState,
    recent_memories:     Vec<(String, String)>,
}

impl AppState {
    fn new(brain_cp: String) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            input:               String::new(),
            messages:            vec![Message::System("Loading models...".into())],
            loading:             true,
            brain_cp,
            device:              Default::default(),
            last_yumon_speak:    Instant::now(),
            next_speak_interval: Duration::from_secs(rng.gen_range(30..120)),
            agent:               AgentState::default(),
            recent_memories: Vec::new(),
        }
    }
}

fn action_display(action: &str) -> &'static str {
    match action {
        "speak"  => "[ speak  ]",
        "build"  => "[ build  ]",
        "travel" => "[ travel ]",
        "use"    => "[ use    ]",
        _        => "[ idle   ]",
    }
}

fn dir_arrow(dir: &str) -> &'static str {
    match dir {
        "north" => "↑ north",
        "south" => "↓ south",
        "east"  => "→ east",
        "west"  => "← west",
        _       => "· none",
    }
}

fn main() -> Result<()> {
    // Ratzilla setup
    let backend = DomBackend::new().map_err(|e| anyhow::anyhow!("Backend: {e:?}"))?;
    let terminal = Terminal::new(backend).map_err(|e| anyhow::anyhow!("Terminal: {e:?}"))?;

    let brain_cp = "checkpoints/brain/384h_4l_6a_160len".to_string();
    let app = Arc::new(Mutex::new(AppState::new(brain_cp.clone())));

    let (tx_user, rx_user) = mpsc::channel::<(String, usize, WorldContext, Vec<(String, String)>)>();
    let (tx_model, rx_model) = mpsc::channel::<Message>();

    // WASM "model thread" (using spawn_local)
    let device = app.lock().unwrap().device.clone();
    let brain_cp_cl = brain_cp.clone();
    let tx_model_cl = tx_model.clone();
    
    spawn_local(async move {
        // In WASM, std::fs::read_to_string won't work. 
        // We'd need to use fetch() here, but for now we'll just send an error message
        // to show where the loading should happen.
        // real implementation would use: fetch_model_bytes().await
        
        tx_model_cl.send(Message::System("Web model loading requires fetch(). Adapting load() to web...".into())).unwrap();
        
        // Let's at least simulate it for now if we don't have a web-ready load()
        /*
        let res = YumonBrain::<Wgpu>::load(&brain_cp_cl, &device);
        ...
        */
        
        // Wait, for now we'll just stay in loading mode or show error
        tx_model_cl.send(Message::System("Note: Model files must be available via fetch for full web functionality.".into())).unwrap();
    });

    // Handle key events
    let app_cl = app.clone();
    let tx_user_cl = tx_user.clone();
    terminal.on_key_event(move |key| {
        let mut app = app_cl.lock().unwrap();
        match key.code {
            KeyCode::Enter => {
                if !app.input.is_empty() && !app.loading {
                    let input = std::mem::take(&mut app.input);
                    app.messages.push(Message::User(input.clone()));
                    app.loading = true;
                    tx_user_cl.send((input, 4, WorldContext::default(), app.recent_memories.clone())).ok();
                }
            }
            KeyCode::Char(c) => app.input.push(c),
            KeyCode::Backspace => { app.input.pop(); }
            _ => {}
        }
    });

    // Render loop
    let app_cl = app.clone();
    terminal.draw_web(move |f| {
        let mut app = app_cl.lock().unwrap();
        
        // Poll model messages
        while let Ok(msg) = rx_model.try_recv() {
            let mut msg = msg.clone();
            match &mut msg {
                Message::System(s) if s == "Models loaded!" => app.loading = false,
                Message::Yumon(r) => {
                    app.loading = false;
                    app.agent.action     = format!("{:?}", r.action).to_lowercase();
                    app.agent.motion_dir = format!("{:?}", r.motion_dir).to_lowercase();
                    app.agent.emote      = EMOTE_NAMES[r.yumon_emote_idx].to_string();
                    app.agent.raw_output = format!("{:?}", r.raw_output).to_lowercase();
                    app.agent.fsm_state  = r.fsm_state;
                    app.agent.allowed_count = r.allowed_count;

                    if r.reply.len() < 4 {
                        r.reply = format!("{:?}", r.raw_output).to_lowercase();
                    }

                    if let Some(Message::User(human)) = app.messages.iter().rev().find(|m| matches!(m, Message::User(_))) {
                        let human_text = human.clone();
                        let bot   = r.reply.clone();
                        app.recent_memories.push((human_text, bot));
                        if app.recent_memories.len() > 3 {
                            app.recent_memories.remove(0);
                        }
                    }
                }
                _ => {}
            }
            app.messages.push(msg);
        }

        ui(f, &app);
    });

    Ok(())
}

fn ui(f: &mut ratatui::Frame, app: &AppState) {
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),
            Constraint::Length(3),
        ])
        .split(f.area());

    let main = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Min(1),
            Constraint::Length(82),
        ])
        .split(outer[0]);

    // ── Chat history ──────────────────────────────────────────────────────────

    let items: Vec<ListItem> = app.messages.iter().rev().take(15).rev().map(|m| {
        match m {
            Message::User(s) => {
                let raw = format!("you  {s}");
                let wrapped = wrap(&raw, 100); 
                let lines: Vec<Line> = wrapped.iter().map(|l| Line::from(l.to_string())).collect();
                ListItem::new(Text::from(lines))
                    .style(Style::default().fg(Color::Cyan))
            }

            Message::Yumon(r) => {
                let action_str = format!("{:?}", r.action).to_lowercase();
                let raw = if r.reply.is_empty() {
                    format!("yumon  {}", action_display(&action_str))
                } else {
                    format!("yumon  {}", r.reply)
                };
                let wrapped = wrap(&raw, 100);
                let lines: Vec<Line> = wrapped.iter().map(|l| Line::from(l.to_string())).collect();
                ListItem::new(Text::from(lines))
                    .style(Style::default().fg(Color::Green))
            }

            Message::System(s) => {
                let raw = format!("sys  {s}");
                let wrapped = wrap(&raw, 100);
                let lines: Vec<Line> = wrapped.iter().map(|l| Line::from(l.to_string())).collect();
                ListItem::new(Text::from(lines))
                    .style(Style::default().fg(Color::Yellow))
            }
        }
    }).collect();

    let chat = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(" chat "));
    f.render_widget(chat, main[0]);

    // ── Sidebar ───────────────────────────────────────────────────────────────
    let status_color = if app.loading { Color::Yellow } else { Color::Green };
    let status_str   = if app.loading { "thinking" } else { "idle" };

    let memory_str: String = app.recent_memories
        .iter()
        .map(|(h, b)| format!("human: {h} | yumon: {b}"))
        .collect::<Vec<_>>()
        .join("\n ");

    let sidebar_text = format!(
        " status\n {}\n\n action\n {}\n\n heading\n {}\n\n emote\n {}\n\n raw\n {}\n\n memory\n {}\n\n",
        status_str,
        action_display(&app.agent.action),
        dir_arrow(&app.agent.motion_dir),
        app.agent.emote,
        wrap_str(&app.agent.raw_output, 40),
        wrap_str(&memory_str, 40)
    );

    let sidebar = Paragraph::new(sidebar_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" yumon ")
                .border_style(Style::default().fg(status_color)),
        );
    f.render_widget(sidebar, main[1]);

    // ── Input ─────────────────────────────────────────────────────────────────
    let input_style = Style::default().fg(Color::DarkGray);

    let input = Paragraph::new(app.input.as_str())
        .style(input_style)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(if app.loading { " input (waiting) " } else { " input " })
                .border_style(input_style),
        );
    f.render_widget(input, outer[1]);
}

fn wrap_str(s: &str, width: usize) -> String {
    s.chars()
        .collect::<Vec<_>>()
        .chunks(width)
        .map(|c| c.iter().collect::<String>())
        .collect::<Vec<_>>()
        .join("\n ")
}
