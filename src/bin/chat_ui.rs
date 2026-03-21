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
//                 // &class_probs,
//                 // &emote_probs,
//                 // user_emote_idx,
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
    brain::model::{YumonBrain, GenerationResult},
    vision::{self, EMOTE_NAMES, CIFAR_CLASSES, EMOTE_CLASSES},
};

// ── Map constants ─────────────────────────────────────────────────────────────

const MAP_W: usize = 20;
const MAP_H: usize = 20;
const NUM_YUMON: usize = 4;
const NUM_OBSTACLES: usize = 8;
const NUM_RESOURCES: usize = 6;

// Motion index → (dx, dy):  0=N,1=NE,2=E,3=SE,4=S,5=SW,6=W,7=NW
const MOTION_DELTAS: [(i32, i32); 8] = [
    (0, -1),  // N
    (1, -1),  // NE
    (1,  0),  // E
    (1,  1),  // SE
    (0,  1),  // S
    (-1, 1),  // SW
    (-1, 0),  // W
    (-1,-1),  // NW
];

// ── Map cell ──────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum Cell { Empty, Obstacle, Resource, Building }

// ── World map ─────────────────────────────────────────────────────────────────

struct WorldMap {
    cells:  [[Cell; MAP_W]; MAP_H],
    yumon:  [(usize, usize); NUM_YUMON],  // (x, y) per Yumon
}

impl WorldMap {
    fn new(rng: &mut impl Rng) -> Self {
        let mut cells = [[Cell::Empty; MAP_W]; MAP_H];

        // Place obstacles
        for _ in 0..NUM_OBSTACLES {
            let x = rng.gen_range(0..MAP_W);
            let y = rng.gen_range(0..MAP_H);
            cells[y][x] = Cell::Obstacle;
        }

        // Place resources (don't overwrite obstacles)
        let mut placed = 0;
        while placed < NUM_RESOURCES {
            let x = rng.gen_range(0..MAP_W);
            let y = rng.gen_range(0..MAP_H);
            if cells[y][x] == Cell::Empty {
                cells[y][x] = Cell::Resource;
                placed += 1;
            }
        }

        // Place Yumon on empty cells
        let mut yumon = [(0usize, 0usize); NUM_YUMON];
        for slot in yumon.iter_mut() {
            loop {
                let x = rng.gen_range(0..MAP_W);
                let y = rng.gen_range(0..MAP_H);
                if cells[y][x] == Cell::Empty {
                    *slot = (x, y);
                    break;
                }
            }
        }

        WorldMap { cells, yumon }
    }

    /// Apply a GenerationResult for one Yumon (round-robin by index).
    fn apply_result(&mut self, yumon_idx: usize, result: &GenerationResult) {
        let (x, y) = self.yumon[yumon_idx];

        println!("apply result {:?}", result.yumon_action_idx);

        match result.yumon_action_idx {
            // Travel (0) — move in motion direction if cell is passable
            0 => {
                let (dx, dy) = MOTION_DELTAS[result.yumon_motion_idx % 8];
                let nx = (x as i32 + dx).clamp(0, MAP_W as i32 - 1) as usize;
                let ny = (y as i32 + dy).clamp(0, MAP_H as i32 - 1) as usize;
                if self.cells[ny][nx] != Cell::Obstacle {
                    self.yumon[yumon_idx] = (nx, ny);
                }
            }
            // Build (1) — place a building at current position, consume resource if present
            1 => {
                if self.cells[y][x] != Cell::Obstacle {
                    self.cells[y][x] = Cell::Building;
                }
            }
            // Collect (2) — consume a resource at current position
            2 => {
                if self.cells[y][x] == Cell::Resource {
                    self.cells[y][x] = Cell::Empty;
                }
            }
            // All other actions (Speak, Emote, Idle, Wait) — no map change
            _ => {}
        }
    }

    /// Render the map as a vec of ratatui Lines.
    fn render_lines(&self) -> Vec<Line<'static>> {
        let mut lines = Vec::with_capacity(MAP_H);

        for y in 0..MAP_H {
            let mut spans = Vec::with_capacity(MAP_W);

            for x in 0..MAP_W {
                // Yumon overrides cell rendering
                let yumon_here = self.yumon.iter().position(|&pos| pos == (x, y));

                let (ch, color) = if let Some(_) = yumon_here {
                    ('@', Color::Cyan)
                } else {
                    match self.cells[y][x] {
                        Cell::Empty    => ('.', Color::DarkGray),
                        Cell::Obstacle => ('#', Color::Red),
                        Cell::Resource => ('*', Color::Yellow),
                        Cell::Building => ('+', Color::Green),
                    }
                };

                spans.push(Span::styled(
                    ch.to_string(),
                    Style::default().fg(color),
                ));
            }

            lines.push(Line::from(spans));
        }

        lines
    }
}

// ── Messages ──────────────────────────────────────────────────────────────────

enum Message {
    User(String),
    Yumon(GenerationResult, usize),  // result + which Yumon index responded
    System(String),
}

// ── App state ─────────────────────────────────────────────────────────────────

struct AppState {
    input:               String,
    messages:            Vec<Message>,
    loading:             bool,
    brain_cp:            String,
    device:              Device<Wgpu>,
    last_yumon_speak:    Instant,
    next_speak_interval: Duration,
    world:               WorldMap,
    next_yumon_idx:      usize,   // round-robin Yumon turn tracker
    rng:                 rand::rngs::ThreadRng,
}

impl AppState {
    fn new(brain_cp: String) -> Self {
        let mut rng = rand::thread_rng();
        let world   = WorldMap::new(&mut rng);

        Self {
            input:               String::new(),
            messages:            vec![Message::System("Loading models...".into())],
            loading:             true,
            brain_cp,
            device:              Default::default(),
            last_yumon_speak:    Instant::now(),
            next_speak_interval: Duration::from_secs(rng.gen_range(30..120)),
            world,
            next_yumon_idx:      0,
            rng,
        }
    }

    fn advance_yumon(&mut self) -> usize {
        let idx = self.next_yumon_idx;
        self.next_yumon_idx = (self.next_yumon_idx + 1) % NUM_YUMON;
        idx
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend  = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let brain_cp = "checkpoints/brain".to_string();
    let mut app  = AppState::new(brain_cp.clone());

    let (tx_user, rx_user) = mpsc::channel::<(String, usize)>(); // (prompt, yumon_idx)
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
                tx_model.send(Message::System(format!("Error loading models: {}", e))).unwrap();
                return;
            }
        };

        while let Ok((prompt, yumon_idx)) = rx_user.recv() {
            let result = brain_model.generate(&tokenizer, &prompt, 80, &device);
            tx_model.send(Message::Yumon(result, yumon_idx)).unwrap();
        }
    });

    let tick_rate = Duration::from_millis(100);
    let mut last_tick = Instant::now();

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
                                let input     = std::mem::take(&mut app.input);
                                let yumon_idx = app.advance_yumon();
                                app.messages.push(Message::User(input.clone()));
                                app.loading = true;
                                tx_user.send((input, yumon_idx)).unwrap();
                            }
                        }
                        KeyCode::Char(c) => { app.input.push(c); }
                        KeyCode::Backspace => { app.input.pop(); }
                        KeyCode::Esc => break,
                        _ => {}
                    }
                }
            }
        }

        // Receive model responses
        while let Ok(msg) = rx_model.try_recv() {
            match &msg {
                Message::System(s) if s == "Models loaded!" => app.loading = false,
                Message::Yumon(result, yumon_idx) => {
                    app.world.apply_result(*yumon_idx, result);
                    app.loading = false;
                }
                _ => {}
            }
            app.messages.push(msg);
        }

        // Autonomous Yumon tick
        if !app.loading && app.last_yumon_speak.elapsed() >= app.next_speak_interval {
            app.loading = true;
            app.last_yumon_speak = Instant::now();
            app.next_speak_interval = Duration::from_secs(app.rng.gen_range(10..40));
            let yumon_idx = app.advance_yumon();
            tx_user.send(("".to_string(), yumon_idx)).unwrap();
        }

        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;

    Ok(())
}

// ── UI ────────────────────────────────────────────────────────────────────────

fn ui(f: &mut ratatui::Frame, app: &AppState) {
    // Top: chat | map sidebar
    // Bottom: input bar
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(3)])
        .split(f.area());

    let main = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(1), Constraint::Length(MAP_W as u16 + 2)])
        .split(root[0]);

    // ── Chat ──────────────────────────────────────────────────────────────────
    let items: Vec<ListItem> = app.messages.iter().map(|m| {
        let (content, color) = match m {
            Message::User(s) =>
                (format!("You: {}", s), Color::Blue),
            Message::Yumon(r, idx) =>
                (format!("Yumon#{} [{}]: {}", idx, EMOTE_NAMES[r.yumon_emote_idx], r.reply), Color::Green),
            Message::System(s) =>
                (format!("*** {}", s), Color::Yellow),
        };
        ListItem::new(content).style(Style::default().fg(color))
    }).collect();

    let chat = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("Yumon Room"));
    f.render_widget(chat, main[0]);

    // ── Map sidebar ───────────────────────────────────────────────────────────
    let status = if app.loading { "thinking…" } else { "idle" };
    let mut map_lines = vec![
        Line::from(Span::styled(
            format!(" [{status}]"),
            Style::default().fg(if app.loading { Color::Yellow } else { Color::DarkGray }),
        )),
        Line::from(""),
    ];
    map_lines.extend(app.world.render_lines());
    map_lines.push(Line::from(""));
    map_lines.push(Line::from(vec![
        Span::styled("@", Style::default().fg(Color::Cyan)),
        Span::raw(" Yumon  "),
        Span::styled("#", Style::default().fg(Color::Red)),
        Span::raw(" obstacle"),
    ]));
    map_lines.push(Line::from(vec![
        Span::styled("*", Style::default().fg(Color::Yellow)),
        Span::raw(" resource "),
        Span::styled("+", Style::default().fg(Color::Green)),
        Span::raw(" built"),
    ]));

    let map_widget = Paragraph::new(map_lines)
        .block(Block::default().borders(Borders::ALL).title("World"));
    f.render_widget(map_widget, main[1]);

    // ── Input ─────────────────────────────────────────────────────────────────
    let input = Paragraph::new(app.input.as_str())
        .block(Block::default().borders(Borders::ALL).title("Input (ESC to quit)"));
    f.render_widget(input, root[1]);
}
