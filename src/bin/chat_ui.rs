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

// Import from our crate
// We need to be careful with imports since this is a bin file in the same crate.
// Using crate:: is for lib.rs, but this is a separate binary.
// However, in a Cargo project, we can usually refer to the library crate by its name.
use yumon_pet::{
    brain::model::{YumonBrain, GenerationResult},
    vision::{self, EMOTE_NAMES, CIFAR_CLASSES, EMOTE_CLASSES},
};

enum Message {
    User(String),
    Yumon(GenerationResult),
    System(String),
}

struct AppState {
    input: String,
    messages: Vec<Message>,
    loading: bool,
    vision_cp: String,
    brain_cp: String,
    device: Device<Wgpu>,
    // We'll store the models in an Option and move them to the background thread or keep them if small enough
    // For now, let's keep it simple and load them once.
    last_yumon_speak: Instant,
    next_speak_interval: Duration,
}

impl AppState {
    fn new(vision_cp: String, brain_cp: String) -> Self {
        let mut rng = rand::thread_rng();

        Self {
            input: String::new(),
            messages: vec![Message::System("Loading models...".into())],
            loading: true,
            vision_cp,
            brain_cp,
            device: Default::default(),
            last_yumon_speak: Instant::now(),
            next_speak_interval: Duration::from_secs(rng.gen_range(30..120)),
        }
    }
}

fn main() -> Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // App state
    let vision_cp = "checkpoints/vision".to_string();
    let brain_cp = "checkpoints/brain".to_string();
    let mut app = AppState::new(vision_cp.clone(), brain_cp.clone());

    // Channels for communication with the model thread
    let (tx_user, rx_user) = mpsc::channel::<(String, usize)>(); // (prompt, user_emote_idx)
    let (tx_model, rx_model) = mpsc::channel::<Message>();

    // Model thread
    let device = app.device.clone();
    thread::spawn(move || {
        // Load brain model
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

        // Neutral context for now
        let class_probs = vec![1.0 / CIFAR_CLASSES as f32; CIFAR_CLASSES];
        let emote_probs = vec![1.0 / EMOTE_CLASSES as f32; EMOTE_CLASSES];

        while let Ok((prompt, user_emote_idx)) = rx_user.recv() {
            let result = brain_model.generate(
                &tokenizer,
                &class_probs,
                &emote_probs,
                user_emote_idx,
                &prompt,
                80,
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
            .unwrap_or_else(|| Duration::from_secs(0));

        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == event::KeyEventKind::Press {
                    match key.code {
                        KeyCode::Enter => {
                            if !app.input.is_empty() && !app.loading {
                                let input = std::mem::take(&mut app.input);
                                let input = "Prompt: ".to_owned() + &input.clone() + " / ";
                                app.messages.push(Message::User(input.clone()));
                                app.loading = true;
                                tx_user.send((input, 4)).unwrap(); // 4 is neutral
                            }
                        }
                        KeyCode::Char(c) => {
                            app.input.push(c);
                        }
                        KeyCode::Backspace => {
                            app.input.pop();
                        }
                        KeyCode::Esc => break,
                        _ => {}
                    }
                }
            }
        }

        // Check for model responses
        while let Ok(msg) = rx_model.try_recv() {
            match &msg {
                Message::System(s) if s == "Models loaded!" => app.loading = false,
                Message::Yumon(_) => app.loading = false,
                _ => {}
            }
            app.messages.push(msg);
        }

        // Yumon random thought timer
        // if !app.loading && app.last_yumon_speak.elapsed() >= app.next_speak_interval {
        //     app.loading = true;
        //     app.last_yumon_speak = Instant::now();
            
        //     // Pick a new random interval for next time
        //     let mut rng = rand::thread_rng();
        //     app.next_speak_interval = Duration::from_secs(rng.gen_range(30..120));
            
        //     tx_user.send(("".to_string(), 4)).unwrap(); // empty prompt, neutral emote
        // }

        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }
    }

    // Restore terminal
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
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),
            Constraint::Length(3),
        ])
        .split(f.area());

    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(80),
            Constraint::Percentage(20),
        ])
        .split(chunks[0]);

    // Chat history
    let messages: Vec<ListItem> = app
        .messages
        .iter()
        .map(|m| {
            let (content, color) = match m {
                Message::User(s) => (format!("You: {}", s), Color::Blue),
                Message::Yumon(r) => (format!("Yumon [{}]: {}", EMOTE_NAMES[r.yumon_emote_idx], r.reply), Color::Green),
                Message::System(s) => (format!("*** {}", s), Color::Yellow),
            };
            ListItem::new(content).style(Style::default().fg(color))
        })
        .collect();

    let chat = List::new(messages)
        .block(Block::default().borders(Borders::ALL).title("Yumon Chat"));
    f.render_widget(chat, main_chunks[0]);

    // Sidebar
    let status = if app.loading { "Thinking..." } else { "Idle" };
    let sidebar = Paragraph::new(format!("Status: {}\n\nPress ESC to quit", status))
        .block(Block::default().borders(Borders::ALL).title("Yumon"));
    f.render_widget(sidebar, main_chunks[1]);

    // Input field
    let input = Paragraph::new(app.input.as_str())
        // .style(match app.loading {
        //     true => Style::default().fg(Color::DarkGray),
        //     false => Style::default(),
        // })
        .block(Block::default().borders(Borders::ALL).title("Input"));
    f.render_widget(input, chunks[1]);
}
