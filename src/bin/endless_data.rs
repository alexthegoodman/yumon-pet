// Procedurally generate questions to answer, save qa pairs to file

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph, Wrap},
    Frame, Terminal,
};
use serde::Serialize;
use std::{
    fs::OpenOptions,
    io::{self, Write},
    time::{SystemTime, UNIX_EPOCH},
};

// ── Data ─────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct Entry {
    topic: String,
    question: String,
    answer: String,
    timestamp: u64,
}

// ── Procedural question engine ────────────────────────────────────────────────
//
// Each Category owns several independent slot-lists. generate() picks one item
// from each slot and renders a template string.  With the counts below:
//
//   ~10 categories × ~8 templates × ~6 subjects × ~6 framings × ~5 angles
//   × ~5 time_ctx  ≈  72,000+ distinct surface forms before any duplicates.
//
// Adding more slots or items scales the space multiplicatively for free.

fn pick<'a>(list: &'a [&'static str]) -> &'a str {
    list[rand::random::<usize>() % list.len()]
}

struct Category {
    topic: &'static str,

    // Each template uses named placeholders: {subject}, {framing}, {angle}, {time}
    // A template may omit any placeholder it doesn't need.
    templates: &'static [&'static str],

    subjects:  &'static [&'static str], // the concrete noun/domain
    framings:  &'static [&'static str], // verb phrase / lens ("think about", "approach")
    angles:    &'static [&'static str], // what dimension to explore
    time_ctx:  &'static [&'static str], // temporal framing
}

impl Category {
    fn generate(&self) -> String {
        let tmpl    = pick(self.templates);
        let subject = pick(self.subjects);
        let framing = pick(self.framings);
        let angle   = pick(self.angles);
        let time    = pick(self.time_ctx);

        tmpl.replace("{subject}", subject)
            .replace("{framing}", framing)
            .replace("{angle}",   angle)
            .replace("{time}",    time)
    }
}

struct QuestionEngine {
    categories: Vec<Category>,
}

impl QuestionEngine {
    fn new() -> Self {
        Self { categories: vec![

            // ── Hobbies ───────────────────────────────────────────────────
            Category {
                topic: "Hobbies",
                templates: &[
                    "When it comes to {subject}, how do you {framing} the {angle} side of it?",
                    "What draws you to {subject} — especially the {angle} aspect?",
                    "How has your relationship with {subject} changed {time}?",
                    "If you had to {framing} {subject} to a stranger, what would you emphasise about the {angle}?",
                    "What's something about {subject} that surprised you {time}?",
                    "Do you find the {angle} of {subject} energising or draining, and why?",
                    "What would you want to explore more deeply about {subject} {time}?",
                    "Has the {angle} of {subject} ever clashed with other parts of your life?",
                ],
                subjects: &[
                    "reading", "gaming", "cooking", "hiking", "music-making",
                    "photography", "drawing or painting", "gardening", "crafting",
                    "writing", "collecting things", "sport or fitness", "film-watching",
                    "learning languages", "woodworking", "board games",
                ],
                framings: &[
                    "think about", "approach", "explain", "justify", "get lost in",
                    "make time for", "push yourself in",
                ],
                angles: &[
                    "social", "creative", "competitive", "meditative", "skill-building",
                    "escapist", "productive",
                ],
                time_ctx: &[
                    "over the last year", "since you started", "recently",
                    "compared to when you began", "as you've gotten older",
                ],
            },

            // ── Work & Career ─────────────────────────────────────────────
            Category {
                topic: "Work",
                templates: &[
                    "How do you {framing} the {angle} demands of {subject}?",
                    "What part of {subject} has shaped your {angle} the most {time}?",
                    "When {subject} gets hard, how does the {angle} side affect how you cope?",
                    "What would an ideal version of {subject} look like from a {angle} perspective?",
                    "How has your view of {subject} shifted {time}, especially around {angle}?",
                    "What's a {angle} lesson {subject} has taught you that you didn't expect?",
                    "Do you feel the {angle} aspects of {subject} are undervalued or overvalued?",
                    "If you could redesign {subject}, what {angle} changes would matter most?",
                ],
                subjects: &[
                    "your current role", "your industry", "remote or hybrid work",
                    "career progression", "feedback and performance review",
                    "collaboration with colleagues", "managing your own time",
                    "learning on the job", "switching jobs or fields",
                ],
                framings: &[
                    "handle", "navigate", "think about", "communicate about",
                    "push back against", "embrace", "reframe",
                ],
                angles: &[
                    "emotional", "financial", "social", "creative", "strategic",
                    "ethical", "personal-growth",
                ],
                time_ctx: &[
                    "over the past few years", "since you changed roles",
                    "as your career has progressed", "recently", "early in your career versus now",
                ],
            },

            // ── Beliefs & Values ──────────────────────────────────────────
            Category {
                topic: "Beliefs",
                templates: &[
                    "How do you {framing} the idea that {subject}?",
                    "Has your view that {subject} been tested {time}?",
                    "What {angle} evidence would make you reconsider the belief that {subject}?",
                    "How does believing that {subject} actually show up in your daily choices?",
                    "Where did your conviction that {subject} first come from?",
                    "Do you find it easy or hard to defend the position that {subject} to others?",
                    "Has the {angle} dimension of '{subject}' ever conflicted with something else you value?",
                    "How certain are you that {subject} — and what would shift that certainty?",
                ],
                subjects: &[
                    "people are mostly trying their best",
                    "hard work reliably leads to good outcomes",
                    "institutions can be trusted to self-correct",
                    "individual choices matter more than systems",
                    "most disagreements are really about values, not facts",
                    "meaning has to be made rather than found",
                    "technology is broadly making life better",
                    "human nature is fundamentally cooperative",
                    "tradition deserves more respect than it gets",
                    "radical honesty is usually the right policy",
                ],
                framings: &[
                    "hold onto", "challenge", "articulate", "live out",
                    "reconcile contradictions in", "arrive at", "pass on",
                ],
                angles: &[
                    "empirical", "emotional", "social", "historical",
                    "personal", "philosophical", "ethical",
                ],
                time_ctx: &[
                    "in the last few years", "as you've gotten older",
                    "after a major life event", "recently", "over time",
                ],
            },

            // ── Interests & Curiosity ─────────────────────────────────────
            Category {
                topic: "Interests",
                templates: &[
                    "What first pulled you toward {subject}, and what keeps you interested {time}?",
                    "How do you {framing} your interest in {subject} when talking to people who don't share it?",
                    "What's the most {angle} thing you've learned from going deep into {subject}?",
                    "Has {subject} ever changed how you see something unrelated?",
                    "What would you most want to understand about {subject} that still eludes you?",
                    "How does {subject} connect to other things you care about?",
                    "Is your interest in {subject} more {angle} or something else — and does the distinction matter?",
                    "What's an entry point into {subject} you'd recommend to a curious newcomer?",
                ],
                subjects: &[
                    "history", "science", "philosophy", "economics", "psychology",
                    "a particular art form", "a niche subculture", "linguistics",
                    "politics", "mathematics", "anthropology", "a specific genre of fiction",
                    "architecture", "ecology", "a sport or game at a deep level",
                ],
                framings: &[
                    "explain", "justify", "share", "defend", "downplay",
                    "introduce", "get others into",
                ],
                angles: &[
                    "surprising", "counterintuitive", "practical", "beautiful",
                    "humbling", "unsettling", "inspiring",
                ],
                time_ctx: &[
                    "now versus when you started", "over the years", "recently",
                    "after going deeper", "as it's evolved",
                ],
            },

            // ── Relationships ─────────────────────────────────────────────
            Category {
                topic: "Relationships",
                templates: &[
                    "How has your approach to {subject} shifted {time}?",
                    "What {angle} pattern do you notice in yourself around {subject}?",
                    "When {subject} gets difficult, what's your instinct — and does it serve you?",
                    "What does {subject} at its best actually look like for you?",
                    "How do you {framing} the {angle} side of {subject}?",
                    "What's something you wish you understood earlier about {subject}?",
                    "Where does the {angle} dimension of {subject} show up most clearly for you?",
                    "How do you decide when to invest more in {subject} versus step back?",
                ],
                subjects: &[
                    "close friendship", "family dynamics", "romantic partnership",
                    "making new connections", "maintaining long-distance relationships",
                    "navigating conflict", "setting limits with people you care about",
                    "supporting someone through difficulty", "being supported",
                    "trust and vulnerability",
                ],
                framings: &[
                    "handle", "navigate", "communicate about", "think about",
                    "grow through", "show up for", "protect yourself in",
                ],
                angles: &[
                    "emotional", "practical", "social", "recurring",
                    "unconscious", "productive", "avoidant",
                ],
                time_ctx: &[
                    "over the past few years", "since a relationship ended or changed",
                    "as you've gotten older", "recently", "across different life stages",
                ],
            },

            // ── Personal Growth ───────────────────────────────────────────
            Category {
                topic: "Growth",
                templates: &[
                    "What does genuinely improving at {subject} look like for you {time}?",
                    "How do you {framing} setbacks related to {subject}?",
                    "What {angle} shift has helped you most with {subject}?",
                    "When it comes to {subject}, what's the gap between who you are and who you want to be?",
                    "What has {subject} revealed about your {angle} tendencies?",
                    "How do you sustain motivation around {subject} when it gets hard?",
                    "What feedback about {subject} was hardest to accept — and were they right?",
                    "Has working on {subject} ever changed how you see yourself?",
                ],
                subjects: &[
                    "self-discipline", "emotional regulation", "communication",
                    "patience", "confidence", "consistency", "learning new skills",
                    "managing your attention", "facing fear", "breaking old habits",
                    "asking for help", "receiving criticism", "staying motivated",
                ],
                framings: &[
                    "handle", "reframe", "learn from", "talk to yourself about",
                    "keep perspective on", "recover from", "use productively",
                ],
                angles: &[
                    "mindset", "behavioural", "emotional", "social",
                    "habitual", "unconscious", "identity-level",
                ],
                time_ctx: &[
                    "over the past year", "recently", "since you started focusing on it",
                    "compared to five years ago", "across different chapters of your life",
                ],
            },

            // ── Philosophy ────────────────────────────────────────────────
            Category {
                topic: "Philosophy",
                templates: &[
                    "How do you {framing} the question of {subject}?",
                    "Has your view on {subject} changed {time}?",
                    "What {angle} argument around {subject} do you find most compelling?",
                    "Where does {subject} actually show up in real decisions you make?",
                    "What would you need to believe to hold a completely different view on {subject}?",
                    "Is {subject} something you've resolved for yourself, or does it stay open?",
                    "What's the most {angle} implication of taking {subject} seriously?",
                    "How does your position on {subject} affect how you live day to day?",
                ],
                subjects: &[
                    "free will and responsibility", "what makes a life meaningful",
                    "whether morality is objective", "the nature of identity over time",
                    "how certain we can be about anything",
                    "whether progress is real", "the self and consciousness",
                    "obligation to others versus to yourself",
                    "whether suffering can be justified",
                    "what we owe to future generations",
                ],
                framings: &[
                    "live with", "think through", "articulate", "apply",
                    "sit with uncertainty about", "resolve", "keep revisiting",
                ],
                angles: &[
                    "practical", "uncomfortable", "liberating", "counterintuitive",
                    "underexplored", "ancient", "modern",
                ],
                time_ctx: &[
                    "over the years", "since you encountered a challenge to it",
                    "as you've gotten older", "recently", "after a significant experience",
                ],
            },

            // ── Creativity ────────────────────────────────────────────────
            Category {
                topic: "Creativity",
                templates: &[
                    "How do you {framing} the {angle} side of {subject}?",
                    "What has working on {subject} taught you about your creative process {time}?",
                    "When {subject} isn't flowing, what's usually behind it?",
                    "How do constraints affect your {subject}?",
                    "What {angle} feedback about your {subject} has stuck with you?",
                    "Does your approach to {subject} come more from instinct or deliberate method?",
                    "How do you know when a piece of {subject} is finished?",
                    "What's the gap between the {subject} you make and the {subject} you imagine?",
                ],
                subjects: &[
                    "writing", "visual art", "music composition or performance",
                    "design work", "cooking as a creative act", "photography",
                    "storytelling", "problem-solving at work", "building things",
                    "any creative project you're working on",
                ],
                framings: &[
                    "protect", "develop", "share", "criticise", "sustain",
                    "break out of patterns in", "push further in",
                ],
                angles: &[
                    "self-critical", "experimental", "playful", "vulnerable",
                    "technical", "intuitive", "commercial",
                ],
                time_ctx: &[
                    "over the past year", "as you've developed your craft",
                    "recently", "since you started taking it seriously",
                    "across different projects",
                ],
            },

            // ── Technology ────────────────────────────────────────────────
            Category {
                topic: "Technology",
                templates: &[
                    "How has {subject} changed the {angle} side of your life {time}?",
                    "What do you find most {angle} about where {subject} is heading?",
                    "How do you {framing} your relationship with {subject}?",
                    "What would you want people to understand about the {angle} effects of {subject}?",
                    "Has {subject} ever created a problem you didn't anticipate?",
                    "Do you think the {angle} benefits of {subject} outweigh its costs?",
                    "What's something about {subject} that you actively push back against?",
                    "How do you draw a line around {subject} in your own life?",
                ],
                subjects: &[
                    "social media", "AI tools", "smartphones", "remote collaboration software",
                    "surveillance and data collection", "algorithmic recommendation",
                    "electric vehicles and green tech", "automation in your field",
                    "the pace of technological change generally",
                ],
                framings: &[
                    "manage", "think critically about", "set limits around",
                    "take advantage of", "resist", "stay informed about",
                ],
                angles: &[
                    "social", "psychological", "economic", "ethical",
                    "exciting", "alarming", "ambiguous",
                ],
                time_ctx: &[
                    "over the past decade", "recently", "as it's matured",
                    "since it became mainstream", "in the last couple of years",
                ],
            },

            // ── Memory & Identity ─────────────────────────────────────────
            Category {
                topic: "Memory",
                templates: &[
                    "How does {subject} still shape who you are {time}?",
                    "What {angle} thing about {subject} do you find yourself returning to?",
                    "If you could revisit {subject} with fresh eyes, what would you notice differently?",
                    "What did {subject} teach you that you only understood {time}?",
                    "Is there something about {subject} you've made peace with — or haven't?",
                    "How has your interpretation of {subject} changed {time}?",
                    "What would you want to preserve from {subject} that's easy to forget?",
                    "Does {subject} feel like a foundation or something you've moved on from?",
                ],
                subjects: &[
                    "your childhood home or town", "a formative friendship",
                    "a time you failed at something important", "an early creative or academic success",
                    "a period of significant change", "a mentor or teacher who mattered",
                    "a trip or experience that stands out", "a belief you held much earlier",
                    "a version of yourself you've grown past",
                ],
                framings: &[
                    "hold onto", "revisit", "reinterpret", "let go of",
                    "talk about", "draw from", "be honest about",
                ],
                angles: &[
                    "unexpected", "recurring", "formative", "complicated",
                    "vivid", "bittersweet", "clarifying",
                ],
                time_ctx: &[
                    "in retrospect", "looking back now", "with more distance",
                    "as you've gotten older", "compared to how you saw it at the time",
                ],
            },

        ]}
    }

    fn random_question(&self) -> (String, String) {
        let cat = &self.categories[rand::random::<usize>() % self.categories.len()];
        (cat.topic.to_string(), cat.generate())
    }
}

// ── App state ─────────────────────────────────────────────────────────────────

struct App {
    input: String,
    current_topic: String,
    current_q: String,
    history_count: usize,
    bank: QuestionEngine,
    status_msg: String, // Feedback line shown after save
}

impl App {
    fn new() -> Self {
        let bank = QuestionEngine::new();
        let (topic, q) = bank.random_question();
        Self {
            input: String::new(),
            current_topic: topic,
            current_q: q,
            history_count: 0,
            bank,
            status_msg: String::from("Type your answer and press Enter to save."),
        }
    }

    fn generate_new_question(&mut self) {
        let (topic, q) = self.bank.random_question();
        self.current_topic = topic;
        self.current_q = q;
        self.input.clear();
    }

    fn save_to_file(&mut self) -> io::Result<()> {
        let answer = self.input.trim().to_string();
        if answer.is_empty() {
            self.status_msg = "⚠  Answer is empty — nothing saved.".to_string();
            return Ok(());
        }

        let entry = Entry {
            topic: self.current_topic.clone(),
            question: self.current_q.clone(),
            answer,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("qa_journal.jsonl")?;

        let line = serde_json::to_string(&entry)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        writeln!(file, "{line}")?;

        self.history_count += 1;
        self.status_msg = format!("✓  Saved! ({} answered so far)", self.history_count);
        Ok(())
    }
}

// ── UI ────────────────────────────────────────────────────────────────────────

fn ui(f: &mut Frame, app: &App) {
    let area = f.size();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(3), // Header / status bar
            Constraint::Min(6),    // Question area
            Constraint::Length(3), // Input box
            Constraint::Length(3), // Status / feedback
        ])
        .split(area);

    // ── Header ────────────────────────────────────────────────────────────────
    let header = Paragraph::new(Line::from(vec![
        Span::styled("  QA Journal  ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw("│ "),
        Span::styled(
            format!("{} answered", app.history_count),
            Style::default().fg(Color::Green),
        ),
        Span::raw("  │  Ctrl-Q to quit"),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::DarkGray)),
    );
    f.render_widget(header, chunks[0]);

    // ── Question ──────────────────────────────────────────────────────────────
    let question = Paragraph::new(vec![
        Line::from(Span::styled(
            format!("  Topic: {}", app.current_topic),
            Style::default().fg(Color::Yellow).add_modifier(Modifier::ITALIC),
        )),
        Line::from(""),
        Line::from(Span::styled(
            format!("  {}", app.current_q),
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        )),
    ])
    .wrap(Wrap { trim: false })
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(" Question ")
            .title_style(Style::default().fg(Color::Cyan))
            .border_style(Style::default().fg(Color::Cyan)),
    );
    f.render_widget(question, chunks[1]);

    // ── Input ─────────────────────────────────────────────────────────────────
    let input = Paragraph::new(app.input.as_str())
        .style(Style::default().fg(Color::White))
        .wrap(Wrap { trim: false })
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Your Answer  (Enter = save & next  |  Backspace = delete) ")
                .title_style(Style::default().fg(Color::Green))
                .border_style(Style::default().fg(Color::Green)),
        );
    f.render_widget(input, chunks[2]);

    // ── Status / feedback ────────────────────────────────────────────────────
    let status = Paragraph::new(app.status_msg.as_str())
        .style(Style::default().fg(Color::DarkGray))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
    f.render_widget(status, chunks[3]);
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> io::Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new();

    loop {
        terminal.draw(|f| ui(f, &app))?;

        if let Event::Key(key) = event::read()? {
            match (key.modifiers, key.code) {
                // Quit
                (KeyModifiers::CONTROL, KeyCode::Char('q')) | (_, KeyCode::Esc) => break,

                // Submit answer → save + next question
                (_, KeyCode::Enter) => {
                    if let Err(e) = app.save_to_file() {
                        app.status_msg = format!("✗  Error saving: {e}");
                    }
                    app.generate_new_question();
                }

                // Typing
                (_, KeyCode::Char(c)) => {
                    app.input.push(c);
                }

                // Backspace — delete last char
                (_, KeyCode::Backspace) => {
                    app.input.pop();
                }

                // Skip this question without saving
                (KeyModifiers::CONTROL, KeyCode::Char('s')) => {
                    app.status_msg = "Skipped.".to_string();
                    app.generate_new_question();
                }

                _ => {}
            }
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

    println!("Journal saved to qa_journal.jsonl");
    Ok(())
}