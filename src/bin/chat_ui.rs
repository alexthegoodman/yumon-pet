// use anyhow::Result;
// use burn::{backend::Wgpu, prelude::*};
// use crossterm::{
//     event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
//     execute,
//     terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
// };
// use ratatui::{
//     backend::CrosstermBackend,
//     layout::{Constraint, Direction, Layout},
//     style::{Color, Modifier, Style},
//     text::{Line, Span},
//     widgets::{Block, Borders, List, ListItem, Paragraph},
//     Terminal,
// };
// use std::{io, sync::mpsc, thread, time::{Duration, Instant}};
// use rand::Rng;

// // Import from our crate
// // We need to be careful with imports since this is a bin file in the same crate.
// // Using crate:: is for lib.rs, but this is a separate binary.
// // However, in a Cargo project, we can usually refer to the library crate by its name.
// use yumon_pet::{
//     brain::model::{YumonBrain, GenerationResult},
//     vision::{self, EMOTE_NAMES, CIFAR_CLASSES, EMOTE_CLASSES},
// };

// enum Message {
//     User(String),
//     Yumon(GenerationResult),
//     System(String),
// }

// struct AppState {
//     input: String,
//     messages: Vec<Message>,
//     loading: bool,
//     vision_cp: String,
//     brain_cp: String,
//     device: Device<Wgpu>,
//     // We'll store the models in an Option and move them to the background thread or keep them if small enough
//     // For now, let's keep it simple and load them once.
//     last_yumon_speak: Instant,
//     next_speak_interval: Duration,
// }

// impl AppState {
//     fn new(vision_cp: String, brain_cp: String) -> Self {
//         let mut rng = rand::thread_rng();

//         Self {
//             input: String::new(),
//             messages: vec![Message::System("Loading models...".into())],
//             loading: true,
//             vision_cp,
//             brain_cp,
//             device: Default::default(),
//             last_yumon_speak: Instant::now(),
//             next_speak_interval: Duration::from_secs(rng.gen_range(30..120)),
//         }
//     }
// }

// fn main() -> Result<()> {
//     // Setup terminal
//     enable_raw_mode()?;
//     let mut stdout = io::stdout();
//     execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
//     let backend = CrosstermBackend::new(stdout);
//     let mut terminal = Terminal::new(backend)?;

//     // App state
//     let vision_cp = "checkpoints/vision".to_string();
//     let brain_cp = "checkpoints/brain-128-8b-8k-mix-5".to_string();
//     let mut app = AppState::new(vision_cp.clone(), brain_cp.clone());

//     // Channels for communication with the model thread
//     let (tx_user, rx_user) = mpsc::channel::<(String, usize)>(); // (prompt, user_emote_idx)
//     let (tx_model, rx_model) = mpsc::channel::<Message>();

//     // Model thread
//     let device = app.device.clone();
//     thread::spawn(move || {
//         // Load brain model
//         let res = YumonBrain::<Wgpu>::load(&brain_cp, &device);
//         let (brain_model, tokenizer) = match res {
//             Ok(m) => {
//                 tx_model.send(Message::System("Models loaded!".into())).unwrap();
//                 m
//             }
//             Err(e) => {
//                 tx_model.send(Message::System(format!("Error loading models: {}", e))).unwrap();
//                 return;
//             }
//         };

//         // Neutral context for now
//         let class_probs = vec![1.0 / CIFAR_CLASSES as f32; CIFAR_CLASSES];
//         let emote_probs = vec![1.0 / EMOTE_CLASSES as f32; EMOTE_CLASSES];

//         while let Ok((prompt, user_emote_idx)) = rx_user.recv() {
//             let result = brain_model.generate(
//                 &tokenizer,
//                 &class_probs,
//                 &emote_probs,
//                 user_emote_idx,
//                 &prompt,
//                 80,
//                 &device,
//             );
//             tx_model.send(Message::Yumon(result)).unwrap();
//         }
//     });

//     let mut last_tick = Instant::now();
//     let tick_rate = Duration::from_millis(100);

//     loop {
//         terminal.draw(|f| ui(f, &app))?;

//         let timeout = tick_rate
//             .checked_sub(last_tick.elapsed())
//             .unwrap_or_else(|| Duration::from_secs(0));

//         if event::poll(timeout)? {
//             if let Event::Key(key) = event::read()? {
//                 if key.kind == event::KeyEventKind::Press {
//                     match key.code {
//                         KeyCode::Enter => {
//                             if !app.input.is_empty() && !app.loading {
//                                 let input = std::mem::take(&mut app.input);
//                                 app.messages.push(Message::User(input.clone()));
//                                 app.loading = true;
//                                 tx_user.send((input, 4)).unwrap(); // 4 is neutral
//                             }
//                         }
//                         KeyCode::Char(c) => {
//                             app.input.push(c);
//                         }
//                         KeyCode::Backspace => {
//                             app.input.pop();
//                         }
//                         KeyCode::Esc => break,
//                         _ => {}
//                     }
//                 }
//             }
//         }

//         // Check for model responses
//         while let Ok(msg) = rx_model.try_recv() {
//             match &msg {
//                 Message::System(s) if s == "Models loaded!" => app.loading = false,
//                 Message::Yumon(_) => app.loading = false,
//                 _ => {}
//             }
//             app.messages.push(msg);
//         }

//         // Yumon random thought timer
//         if !app.loading && app.last_yumon_speak.elapsed() >= app.next_speak_interval {
//             app.loading = true;
//             app.last_yumon_speak = Instant::now();
            
//             // Pick a new random interval for next time
//             let mut rng = rand::thread_rng();
//             app.next_speak_interval = Duration::from_secs(rng.gen_range(30..120));
            
//             tx_user.send(("".to_string(), 4)).unwrap(); // empty prompt, neutral emote
//         }

//         if last_tick.elapsed() >= tick_rate {
//             last_tick = Instant::now();
//         }
//     }

//     // Restore terminal
//     disable_raw_mode()?;
//     execute!(
//         terminal.backend_mut(),
//         LeaveAlternateScreen,
//         DisableMouseCapture
//     )?;
//     terminal.show_cursor()?;

//     Ok(())
// }

// fn ui(f: &mut ratatui::Frame, app: &AppState) {
//     let chunks = Layout::default()
//         .direction(Direction::Vertical)
//         .constraints([
//             Constraint::Min(1),
//             Constraint::Length(3),
//         ])
//         .split(f.area());

//     let main_chunks = Layout::default()
//         .direction(Direction::Horizontal)
//         .constraints([
//             Constraint::Percentage(80),
//             Constraint::Percentage(20),
//         ])
//         .split(chunks[0]);

//     // Chat history
//     let messages: Vec<ListItem> = app
//         .messages
//         .iter()
//         .map(|m| {
//             let (content, color) = match m {
//                 Message::User(s) => (format!("You: {}", s), Color::Blue),
//                 Message::Yumon(r) => (format!("Yumon [{}]: {}", EMOTE_NAMES[r.yumon_emote_idx], r.reply), Color::Green),
//                 Message::System(s) => (format!("*** {}", s), Color::Yellow),
//             };
//             ListItem::new(content).style(Style::default().fg(color))
//         })
//         .collect();

//     let chat = List::new(messages)
//         .block(Block::default().borders(Borders::ALL).title("Yumon Chat"));
//     f.render_widget(chat, main_chunks[0]);

//     // Sidebar
//     let status = if app.loading { "Thinking..." } else { "Idle" };
//     let sidebar = Paragraph::new(format!("Status: {}\n\nPress ESC to quit", status))
//         .block(Block::default().borders(Borders::ALL).title("Yumon"));
//     f.render_widget(sidebar, main_chunks[1]);

//     // Input field
//     let input = Paragraph::new(app.input.as_str())
//         // .style(match app.loading {
//         //     true => Style::default().fg(Color::DarkGray),
//         //     false => Style::default(),
//         // })
//         .block(Block::default().borders(Borders::ALL).title("Input"));
//     f.render_widget(input, chunks[1]);
// }


use anyhow::Result;
use burn::{backend::Wgpu, prelude::*};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Terminal,
};
use std::{io, sync::mpsc, thread, time::{Duration, Instant}};
use rand::Rng;

use yumon_pet::{
    brain::{
        model::{GenerationResult, YUMON_SCHEMA, YumonBrain}, samples::WorldContext,
    },
    vision::{self, CIFAR_CLASSES, EMOTE_CLASSES, EMOTE_NAMES},
};

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
    device:              Device<Wgpu>,
    last_yumon_speak:    Instant,
    next_speak_interval: Duration,
    agent:               AgentState,
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

    let brain_cp = "checkpoints/brain".to_string();
    let mut app = AppState::new(brain_cp.clone());

    let (tx_user, rx_user) = mpsc::channel::<(String, usize, WorldContext)>();
    let (tx_model, rx_model) = mpsc::channel::<Message>();

    let device = app.device.clone();
    thread::spawn(move || {
        let res = YumonBrain::<Wgpu>::load(&brain_cp, &device);
        let (brain_model, tokenizer) = match res {
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

        let class_probs = vec![1.0 / CIFAR_CLASSES as f32; CIFAR_CLASSES];
        let emote_probs = vec![1.0 / EMOTE_CLASSES as f32; EMOTE_CLASSES];

        while let Ok((prompt, user_emote_idx, world)) = rx_user.recv() {
            // let result = brain_model.generate_unmasked(
            let result = brain_model.generate_structured(
                &tokenizer,
                &index,
                &world,
                &class_probs,
                &emote_probs,
                user_emote_idx,
                &prompt,
                80,
                &device,
            );
            let initial = index.initial_state();
            let allowed = index.allowed_tokens(&initial);
            tx_model.send(Message::System(format!(
                "index initial={:?} allowed={:?}",
                initial,
                allowed.as_ref().map(|a| a.len())
            ))).unwrap();
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
                                tx_user.send((input, 4, WorldContext::default())).unwrap();
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
            match &msg {
                Message::System(s) if s == "Models loaded!" => app.loading = false,
                Message::Yumon(r) => {
                    app.loading = false;
                    // Update sidebar agent state
                    app.agent.action     = format!("{:?}", r.action).to_lowercase();
                    app.agent.motion_dir = format!("{:?}", r.motion_dir).to_lowercase();
                    app.agent.emote      = EMOTE_NAMES[r.yumon_emote_idx].to_string();
                    app.agent.raw_output = format!("{:?}", r.raw_output).to_lowercase();
                    app.agent.fsm_state =  r.fsm_state;
                    app.agent.allowed_count =  r.allowed_count;
                }
                _ => {}
            }
            app.messages.push(msg);
        }

        // Yumon autonomous thought timer — inject a random world context
        if !app.loading && app.last_yumon_speak.elapsed() >= app.next_speak_interval {
            app.loading = true;
            app.last_yumon_speak = Instant::now();
            let mut rng = rand::thread_rng();
            app.next_speak_interval = Duration::from_secs(rng.gen_range(30..120));
            let world = WorldContext::random(&mut rng);
            tx_user.send(("".to_string(), 4, world)).unwrap();
        }

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
    let items: Vec<ListItem> = app.messages.iter().map(|m| {
        match m {
            Message::User(s) => ListItem::new(format!("you  {s}"))
                .style(Style::default().fg(Color::Cyan)),

            Message::Yumon(r) => {
                // Only show reply line if action is speak/idle — otherwise
                // show the action prominently and reply as a sub-line if non-empty
                let action_str = format!("{:?}", r.action).to_lowercase();
                let main_line = if r.reply.is_empty() {
                    format!("yumon  {}", action_display(&action_str))
                } else {
                    format!("yumon  {}", r.reply)
                };
                ListItem::new(main_line)
                    .style(Style::default().fg(Color::Green))
            }

            Message::System(s) => ListItem::new(format!("sys  {s}"))
                .style(Style::default().fg(Color::Yellow)),
        }
    }).collect();

    let chat = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(" chat "));
    f.render_widget(chat, main[0]);

    // ── Sidebar ───────────────────────────────────────────────────────────────
    let status_color = if app.loading { Color::Yellow } else { Color::Green };
    let status_str   = if app.loading { "thinking" } else { "idle" };

    // let sidebar_text = format!(
    //     " status\n {}\n\n action\n {}\n\n heading\n {}\n\n emote\n {}\n\n raw\n {}\n\n\n esc to quit",
    //     status_str,
    //     action_display(&app.agent.action),
    //     dir_arrow(&app.agent.motion_dir),
    //     app.agent.emote,
    //     app.agent.raw_output
    // );

    let sidebar_text = format!(
        " status\n {}\n\n action\n {}\n\n heading\n {}\n\n emote\n {}\n\n raw\n {}\n\n fsm\n state={} allowed={}\n\n esc to quit",
        status_str,
        action_display(&app.agent.action),
        dir_arrow(&app.agent.motion_dir),
        app.agent.emote,
        wrap_str(&app.agent.raw_output, 20),  // keep sidebar from overflowing
        app.agent.fsm_state,
        app.agent.allowed_count
            .map(|n| if n == 0 { "NONE - mask not firing!".to_string() } 
                    else { format!("{n} tokens") })
            .unwrap_or("pending".into()),
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