#![recursion_limit = "256"]

use anyhow::Result;
use burn::{backend::Wgpu, prelude::*};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use cubecl::wgpu::WgpuDevice;
use ratatui::{
    Terminal, backend::CrosstermBackend, layout::{Constraint, Direction, Layout}, style::{Color, Modifier, Style}, text::{Line, Span, Text}, widgets::{Block, Borders, List, ListItem, Paragraph}
};
use textwrap::wrap;
use std::{io, sync::mpsc, thread, time::{Duration, Instant}};
use rand::Rng;

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

// Tracks the last known agent state for the sidebar
struct AgentState {
    action:     String,
    motion_dir: String,
    emote:      String,
    raw_output: String,
    
    pub fsm_state:       u32,      // ← final FSM state
    pub allowed_count:   Option<usize>, // ← None if masking never fired
}

impl Default for AgentState {
    fn default() -> Self {
        Self {
            action:     "idle".into(),
            motion_dir: "none".into(),
            emote:      "neutral".into(),
            raw_output: String::new(),

            fsm_state:           0,
            allowed_count:       None
        }
    }
}

struct AppState {
    input:               String,
    messages:            Vec<Message>,
    loading:             bool,
    brain_cp:            String,
    // device:              Device<Wgpu>,
    device:              WgpuDevice,
    last_yumon_speak:    Instant,
    next_speak_interval: Duration,
    agent:               AgentState,
    recent_memories:     Vec<(String, String)>,  // (human, bot) pairs
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

// Action → display string with a simple symbol prefix
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
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let brain_cp = "checkpoints/brain/384h_4l_6a_160len".to_string();
    // let brain_cp = "checkpoints/brain/128h_2l_2a_200len".to_string();
    let mut app = AppState::new(brain_cp.clone());

    // let (tx_user, rx_user) = mpsc::channel::<(String, usize, WorldContext)>();
    let (tx_user, rx_user) = mpsc::channel::<(String, usize, WorldContext, Vec<(String, String)>)>();
    let (tx_model, rx_model) = mpsc::channel::<Message>();

    let device = app.device.clone();
    thread::spawn(move || {
        let res = YumonBrain::<Wgpu>::load(&brain_cp, &device);
        let (brain_model, tokenizer, config) = match res {
            Ok(m) => {
                tx_model.send(Message::System("Models loaded!".into())).unwrap();
                m
            }
            Err(e) => {
                tx_model.send(Message::System(format!("Error loading models: {e}"))).unwrap();
                return;
            }
        };

        // Build outlines index once — cheap at 4096 vocab
        let bpe = match &tokenizer {
            yumon_pet::brain::bpe::TokenizerKind::Bpe(b) => b,
            _ => {
                tx_model.send(Message::System("BPE tokenizer required".into())).unwrap();
                return;
            }
        };
        let index = match YumonBrain::<Wgpu>::build_outlines_index(bpe, YUMON_SCHEMA) {
            Ok(idx) => idx,
            Err(e) => {
                tx_model.send(Message::System(format!("Index build failed: {e}"))).unwrap();
                return;
            }
        };

        // let class_probs = vec![1.0 / CIFAR_CLASSES as f32; CIFAR_CLASSES];
        // let emote_probs = vec![1.0 / EMOTE_CLASSES as f32; EMOTE_CLASSES];

        while let Ok((prompt_text, user_emote_idx, world, memories)) = rx_user.recv() {

            // let prompt = serde_json::to_string_pretty(&serde_json::json!({
            //     "obstacle_dir": "none",
            //     "building_dir": "none",
            //     "resource_dir": "none",
            //     "message":      prompt_text,
            // }))
            // .unwrap();

            let memories_json: Vec<serde_json::Value> = memories
                .iter()
                .map(|(h, b)| serde_json::json!({ "human": h, "bot": b }))
                .collect();

            // let training_stage = TrainingStage::Language;
            let training_stage = TrainingStage::Structured;

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

            tx_model.send(Message::Yumon(result)).unwrap();
        }
    });

    let mut last_tick = Instant::now();
    let tick_rate = Duration::from_millis(100);

    loop {
        terminal.draw(|f| ui(f, &app))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_default();

        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == event::KeyEventKind::Press {
                    match key.code {
                        KeyCode::Enter => {
                            if !app.input.is_empty() && !app.loading {
                                let input = std::mem::take(&mut app.input);
                                app.messages.push(Message::User(input.clone()));
                                app.loading = true;
                                // No spatial context from text input — world is empty
                                // tx_user.send((input, 4, WorldContext::default())).unwrap();
                                tx_user.send((input, 4, WorldContext::default(), app.recent_memories.clone())).unwrap();
                            }
                        }
                        KeyCode::Char(c) => app.input.push(c),
                        KeyCode::Backspace => { app.input.pop(); }
                        KeyCode::Esc => break,
                        _ => {}
                    }
                }
            }
        }

        while let Ok(msg) = rx_model.try_recv() {

            let mut msg = msg.clone();

            match &mut msg {
                Message::System(s) if s == "Models loaded!" => app.loading = false,
                // Message::Yumon(r) => {
                //     app.loading = false;
                //     // Update sidebar agent state
                //     app.agent.action     = format!("{:?}", r.action).to_lowercase();
                //     app.agent.motion_dir = format!("{:?}", r.motion_dir).to_lowercase();
                //     app.agent.emote      = EMOTE_NAMES[r.yumon_emote_idx].to_string();
                //     app.agent.raw_output = format!("{:?}", r.raw_output).to_lowercase();
                //     app.agent.fsm_state =  r.fsm_state;
                //     app.agent.allowed_count =  r.allowed_count;

                //     if r.reply.len() < 4 {
                //         r.reply = format!("{:?}", r.raw_output).to_lowercase();
                //     }
                // }
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

                    // Store the last human message + this reply as a memory
                    if let Some(Message::User(human)) = app.messages.last() {
                        let human = human.clone();
                        let bot   = r.reply.clone();
                        app.recent_memories.push((human, bot));
                        if app.recent_memories.len() > 3 {
                            app.recent_memories.remove(0);
                        }
                    }
                }
                _ => {}
            }

            app.messages.push(msg);
        }

        // Yumon autonomous thought timer — inject a random world context
        // if !app.loading && app.last_yumon_speak.elapsed() >= app.next_speak_interval {
        //     app.loading = true;
        //     app.last_yumon_speak = Instant::now();
        //     let mut rng = rand::thread_rng();
        //     app.next_speak_interval = Duration::from_secs(rng.gen_range(30..120));
        //     let world = WorldContext::random(&mut rng);
        //     tx_user.send(("".to_string(), 4, world)).unwrap();
        // }

        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }
    }

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

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
                let wrapped = wrap(&raw, 100); // or your terminal width
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
        " status\n {}\n\n action\n {}\n\n heading\n {}\n\n emote\n {}\n\n raw\n {}\n\n memory\n {}\n\n esc to quit",
        status_str,
        action_display(&app.agent.action),
        dir_arrow(&app.agent.motion_dir),
        app.agent.emote,
        wrap_str(&app.agent.raw_output, 40),  // keep sidebar from overflowing
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