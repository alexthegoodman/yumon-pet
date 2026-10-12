//! Small terminal editor with local, debounced Yumon Code completions.
#![recursion_limit = "256"]

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use anyhow::{Context, Result, bail, ensure};
    use burn::tensor::{Int, Tensor, TensorData};
    use clap::{Parser, ValueEnum};
    use crossterm::{
        event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
        execute,
        terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
    };
    use rand::{
        distributions::{Distribution, WeightedIndex},
        thread_rng,
    };
    use ratatui::{
        Terminal,
        backend::CrosstermBackend,
        layout::{Constraint, Layout},
        style::{Color, Style},
        text::{Line, Span},
        widgets::Paragraph,
    };
    use std::{
        fs,
        io::{self, IsTerminal, Write},
        path::PathBuf,
        sync::{
            Arc,
            atomic::{AtomicU64, Ordering},
            mpsc,
        },
        thread,
        time::{Duration, Instant},
    };
    use unicode_width::UnicodeWidthStr;
    use yumon_pet::brain::{
        bpe::{BpeTokenizer, TokenizerKind},
        decoder_model::YumonDecBrain,
        moe_model::YumonMoeBrain,
        samples::TrainingStage,
        tokenizer::{BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN},
    };

    type Backend = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;
    type Device = cubecl::wgpu::WgpuDevice;

    #[derive(Clone, Copy, ValueEnum)]
    enum Architecture {
        DecoderOnly,
        Moe,
    }

    #[derive(Clone, Parser)]
    #[command(about = "Nano-style editor with local Yumon Code ghost completions")]
    struct Args {
        /// File to open or create. UTF-8 text only.
        file: PathBuf,
        /// Exact checkpoint directory containing metadata.json, model.bin and tokenizer.json.
        #[arg(long)]
        checkpoint: String,
        #[arg(long, value_enum, default_value = "decoder-only")]
        architecture: Architecture,
        /// Idle time before requesting a completion.
        #[arg(long, default_value_t = 750)]
        debounce_ms: u64,
        /// Maximum number of nearby preceding lines (current line is always included).
        #[arg(long, default_value_t = 10)]
        context_lines: usize,
        #[arg(long, default_value_t = 48, value_parser = clap::value_parser!(u32).range(1..=512))]
        max_tokens: u32,
    }

    enum Model {
        Dense(YumonDecBrain<Backend>),
        Moe(YumonMoeBrain<Backend>),
    }
    impl Model {
        fn load(args: &Args, device: &Device) -> Result<(Self, BpeTokenizer, usize)> {
            let (model, tokenizer, capacity, stage) = match args.architecture {
                Architecture::DecoderOnly => {
                    let (m, t, c) = YumonDecBrain::load(&args.checkpoint, device)?;
                    (Self::Dense(m), t, c.max_seq_len, c.training_stage)
                }
                Architecture::Moe => {
                    let (m, t, c) = YumonMoeBrain::load(&args.checkpoint, device)?;
                    (Self::Moe(m), t, c.max_seq_len, c.training_stage)
                }
            };
            ensure!(
                stage == TrainingStage::Language,
                "use a raw-code Language-stage checkpoint"
            );
            let TokenizerKind::Bpe(tokenizer) = tokenizer else {
                bail!("Code requires BPE")
            };
            ensure!(capacity >= 3, "checkpoint context is too small");
            Ok((model, tokenizer, capacity))
        }

        fn forward(&self, input: Tensor<Backend, 2, Int>) -> Tensor<Backend, 3> {
            match self {
                Self::Dense(m) => m.forward(input),
                Self::Moe(m) => m.forward(input),
            }
        }
    }

    struct Request {
        revision: u64,
        lines: Vec<String>,
    }
    struct Reply {
        revision: u64,
        text: Result<String>,
    }

    // Keep whole, contiguous nearby lines; never drop or truncate the current prefix.
    fn context_tokens(
        lines: &[String],
        budget: usize,
        encode: impl Fn(&str) -> Result<Vec<u32>>,
    ) -> Result<Vec<u32>> {
        let mut prompt = lines.last().cloned().unwrap_or_default();
        let mut tokens = encode(&prompt)?;
        ensure!(
            tokens.len() <= budget,
            "current line exceeds model context; shorten it or use a larger checkpoint"
        );
        for previous in lines.iter().rev().skip(1) {
            let candidate = format!("{previous}\n{prompt}");
            let candidate_tokens = encode(&candidate)?;
            if candidate_tokens.len() > budget {
                break;
            }
            prompt = candidate;
            tokens = candidate_tokens;
        }
        Ok(tokens)
    }

    fn sample(logits: &[f32]) -> Result<usize> {
        let mut ranked: Vec<_> = logits
            .iter()
            .copied()
            .enumerate()
            .filter(|(id, score)| *id != BOS_TOKEN && *id != UNK_TOKEN && score.is_finite())
            .collect();
        ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        ranked.truncate(40);
        ensure!(!ranked.is_empty(), "model produced no finite logits");
        let best = ranked[0].1;
        let weights: Vec<_> = ranked
            .iter()
            .map(|(_, score)| ((*score - best) / 0.8).exp())
            .collect();
        Ok(ranked[WeightedIndex::new(weights)?.sample(&mut thread_rng())].0)
    }

    fn complete(
        model: &Model,
        tokenizer: &BpeTokenizer,
        capacity: usize,
        max_tokens: usize,
        device: &Device,
        request: &Request,
        revision: &AtomicU64,
    ) -> Result<String> {
        // Reserve one BOS and output space. No conversational separator or JSON repair:
        // Code training uses BOS + exact source bytes with shifted next-token targets.
        let current = tokenizer.encode_raw(request.lines.last().map_or("", String::as_str))?;
        ensure!(
            current.len() <= capacity - 2,
            "current line exceeds model context; shorten it or use a larger checkpoint"
        );
        let output_budget = max_tokens.min(capacity - current.len() - 1);
        let mut ids = vec![BOS_TOKEN as u32];
        ids.extend(context_tokens(
            &request.lines,
            capacity - output_budget - 1,
            |text| tokenizer.encode_raw(text),
        )?);
        let prompt_len = ids.len();
        for _ in 0..output_budget {
            if revision.load(Ordering::Relaxed) != request.revision {
                return Ok(String::new());
            }
            let len = ids.len();
            let input = Tensor::<Backend, 2, Int>::from_ints(
                TensorData::new(
                    ids.iter().map(|&id| id as i32).collect::<Vec<_>>(),
                    [1, len],
                ),
                device,
            );
            let scores = model
                .forward(input)
                .slice([0..1, len - 1..len, 0..tokenizer.vocab_size])
                .reshape([tokenizer.vocab_size])
                .to_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .map_err(|e| anyhow::anyhow!("reading logits: {e:?}"))?;
            let next = sample(&scores)?;
            if next == EOS_TOKEN || next == PAD_TOKEN {
                break;
            }
            ids.push(next as u32);
        }
        tokenizer.decode(&ids[prompt_len..])
    }

    struct Editor {
        lines: Vec<String>,
        row: usize,
        col: usize, // col is a UTF-8 byte boundary
        saved: String,
        crlf: bool,
        path: PathBuf,
        ghost: String,
        revision: Arc<AtomicU64>,
        due: Option<Instant>,
        status: String,
        confirm_quit: bool,
        top: usize,
        left: usize,
    }
    impl Editor {
        fn open(path: PathBuf) -> Result<Self> {
            let original = match fs::read_to_string(&path) {
                Ok(text) => text,
                Err(e) if e.kind() == io::ErrorKind::NotFound => String::new(),
                Err(e) => return Err(e).with_context(|| format!("opening {}", path.display())),
            };
            let crlf = original.contains("\r\n");
            let text = original.replace("\r\n", "\n");
            Ok(Self {
                lines: text.split('\n').map(str::to_owned).collect(),
                row: 0,
                col: 0,
                saved: text,
                crlf,
                path,
                ghost: String::new(),
                revision: Arc::new(AtomicU64::new(0)),
                due: None,
                status: "Ready".into(),
                confirm_quit: false,
                top: 0,
                left: 0,
            })
        }
        fn text(&self) -> String {
            self.lines.join("\n")
        }
        fn dirty(&self) -> bool {
            self.text() != self.saved
        }
        fn invalidate(&mut self, delay: Option<Duration>) {
            self.revision.fetch_add(1, Ordering::Relaxed);
            self.ghost.clear();
            self.due = delay.map(|delay| Instant::now() + delay);
        }
        fn insert(&mut self, text: &str) {
            let text = text.replace("\r\n", "\n").replace('\r', "\n");
            let suffix = self.lines[self.row].split_off(self.col);
            let mut pieces = text.split('\n');
            self.lines[self.row].push_str(pieces.next().unwrap());
            for piece in pieces {
                self.row += 1;
                self.lines.insert(self.row, piece.to_owned());
            }
            self.col = self.lines[self.row].len();
            self.lines[self.row].push_str(&suffix);
        }
        fn backspace(&mut self) {
            if self.col > 0 {
                let prev = self.lines[self.row][..self.col]
                    .char_indices()
                    .last()
                    .unwrap()
                    .0;
                self.lines[self.row].drain(prev..self.col);
                self.col = prev;
            } else if self.row > 0 {
                let line = self.lines.remove(self.row);
                self.row -= 1;
                self.col = self.lines[self.row].len();
                self.lines[self.row].push_str(&line);
            }
        }
        fn delete(&mut self) {
            if self.col < self.lines[self.row].len() {
                let len = self.lines[self.row][self.col..]
                    .chars()
                    .next()
                    .unwrap()
                    .len_utf8();
                self.lines[self.row].drain(self.col..self.col + len);
            } else if self.row + 1 < self.lines.len() {
                let next = self.lines.remove(self.row + 1);
                self.lines[self.row].push_str(&next);
            }
        }
        fn vertical(&mut self, row: usize) {
            let chars = self.lines[self.row][..self.col].chars().count();
            self.row = row.min(self.lines.len() - 1);
            self.col = self.lines[self.row]
                .char_indices()
                .nth(chars)
                .map_or(self.lines[self.row].len(), |(byte, _)| byte);
        }
        fn preceding(&self, count: usize) -> Vec<String> {
            let mut lines = self.lines[self.row.saturating_sub(count)..self.row].to_vec();
            lines.push(self.lines[self.row][..self.col].to_owned());
            lines
        }
        fn save(&mut self) -> Result<()> {
            let text = self.text();
            let bytes = if self.crlf {
                text.replace('\n', "\r\n")
            } else {
                text.clone()
            };
            // Do not silently truncate an externally edited file.
            match fs::read_to_string(&self.path) {
                Ok(current) => ensure!(
                    current.replace("\r\n", "\n") == self.saved,
                    "file changed on disk; save cancelled"
                ),
                Err(e) if e.kind() == io::ErrorKind::NotFound && self.saved.is_empty() => {}
                Err(e) => return Err(e).context("checking file before save"),
            }
            // Replace only after a complete write, so failed writes leave the original intact.
            // Resolve existing symlinks so saving does not replace the link itself.
            let destination = if self.path.exists() {
                fs::canonicalize(&self.path)?
            } else {
                self.path.clone()
            };
            let temporary =
                destination.with_file_name(format!(".yumon-nano-{}.tmp", uuid::Uuid::new_v4()));
            let result = (|| -> Result<()> {
                let mut file = fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&temporary)?;
                file.write_all(bytes.as_bytes())?;
                file.sync_all()?;
                drop(file);
                if let Ok(metadata) = fs::metadata(&destination) {
                    fs::set_permissions(&temporary, metadata.permissions())?;
                }
                fs::rename(&temporary, &destination)?;
                Ok(())
            })();
            if result.is_err() {
                let _ = fs::remove_file(&temporary);
            }
            result.context("saving file")?;
            self.saved = text;
            self.status = format!("Saved {}", self.path.display());
            Ok(())
        }
    }

    fn display(text: &str) -> String {
        let mut output = String::new();
        let mut col = 0;
        for c in text.chars() {
            if c == '\t' {
                let spaces = 4 - col % 4;
                output.push_str(&" ".repeat(spaces));
                col += spaces;
            } else {
                let c = if c.is_control() { '?' } else { c };
                output.push(c);
                col += unicode_width::UnicodeWidthChar::width(c).unwrap_or(0);
            }
        }
        output
    }

    // Clip styled text by terminal cells, preserving the ghost style across scrolling.
    fn clipped(parts: Vec<(&str, Color)>, left: usize, width: usize) -> Line<'static> {
        let mut spans = Vec::new();
        let mut col = 0;
        for (text, color) in parts {
            for c in text.chars() {
                let rendered = if c == '\t' {
                    " ".repeat(4 - col % 4)
                } else if c.is_control() {
                    "?".into()
                } else {
                    c.to_string()
                };
                let size = rendered.width();
                if col >= left && col + size <= left + width {
                    spans.push(Span::styled(rendered, Style::default().fg(color)));
                } else if col < left + width && col + size > left {
                    let cells = (col + size).min(left + width) - col.max(left);
                    spans.push(Span::styled(" ".repeat(cells), Style::default().fg(color)));
                }
                col += size;
            }
        }
        Line::from(spans)
    }

    fn draw(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>, app: &mut Editor) -> Result<()> {
        terminal.draw(|frame| {
            let areas = Layout::vertical([
                Constraint::Length(1),
                Constraint::Min(1),
                Constraint::Length(1),
                Constraint::Length(1),
            ])
            .split(frame.area());
            let body = areas[1];
            let height = usize::from(body.height).max(1);
            let width = usize::from(body.width).max(1);
            if app.row < app.top {
                app.top = app.row;
            }
            if app.row >= app.top + height {
                app.top = app.row + 1 - height;
            }
            let cursor_col = display(&app.lines[app.row][..app.col]).width();
            if cursor_col < app.left {
                app.left = cursor_col;
            }
            if cursor_col >= app.left + width {
                app.left = cursor_col + 1 - width;
            }
            let title = format!(
                " Yumon Nano | {}{}",
                app.path.display(),
                if app.dirty() { " *" } else { "" }
            );
            frame.render_widget(
                Paragraph::new(title).style(Style::default().bg(Color::Blue)),
                areas[0],
            );
            let mut rendered = Vec::new();
            for (row, line) in app.lines.iter().enumerate().skip(app.top) {
                if rendered.len() >= height {
                    break;
                }
                if row == app.row && !app.ghost.is_empty() {
                    let ghost: Vec<_> = app.ghost.split('\n').collect();
                    let mut parts = vec![
                        (&line[..app.col], Color::Reset),
                        (ghost[0], Color::DarkGray),
                    ];
                    if ghost.len() == 1 {
                        parts.push((&line[app.col..], Color::Reset));
                    }
                    rendered.push(clipped(parts, app.left, width));
                    for (i, piece) in ghost.iter().enumerate().skip(1) {
                        if rendered.len() >= height {
                            break;
                        }
                        let mut parts = vec![(*piece, Color::DarkGray)];
                        if i + 1 == ghost.len() {
                            parts.push((&line[app.col..], Color::Reset));
                        }
                        rendered.push(clipped(parts, app.left, width));
                    }
                } else {
                    rendered.push(clipped(vec![(line, Color::Reset)], app.left, width));
                }
            }
            frame.render_widget(Paragraph::new(rendered), body);
            let status = if app.confirm_quit {
                "Save before exit? Y: save  N: discard  Esc: cancel".into()
            } else {
                format!(
                    "{} | Ln {}, Col {}",
                    app.status,
                    app.row + 1,
                    app.lines[app.row][..app.col].chars().count() + 1
                )
            };
            frame.render_widget(Paragraph::new(status), areas[2]);
            frame.render_widget(
                Paragraph::new("^O Save  ^X Exit  Tab Accept  ^R Regenerate  Esc Dismiss"),
                areas[3],
            );
            if body.width > 0 && body.height > 0 {
                frame.set_cursor_position((
                    body.x + (cursor_col - app.left) as u16,
                    body.y + (app.row - app.top) as u16,
                ));
            }
        })?;
        Ok(())
    }

    struct TerminalGuard;
    impl Drop for TerminalGuard {
        fn drop(&mut self) {
            let _ = disable_raw_mode();
            let _ = execute!(
                io::stdout(),
                event::DisableBracketedPaste,
                LeaveAlternateScreen
            );
        }
    }

    fn run_editor(
        args: &Args,
        mut app: Editor,
        requests: mpsc::SyncSender<Request>,
        replies: mpsc::Receiver<Reply>,
    ) -> Result<()> {
        enable_raw_mode()?;
        let _guard = TerminalGuard;
        execute!(
            io::stdout(),
            EnterAlternateScreen,
            event::EnableBracketedPaste
        )?;
        let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
        let debounce = Duration::from_millis(args.debounce_ms);
        app.due = Some(Instant::now() + debounce);
        let mut busy = false;
        let mut completion_online = true;
        loop {
            while completion_online {
                let reply = match replies.try_recv() {
                    Ok(reply) => reply,
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        completion_online = false;
                        busy = false;
                        app.invalidate(None);
                        app.status =
                            "Completion worker stopped; editing and saving remain available".into();
                        break;
                    }
                };
                busy = false;
                if reply.revision == app.revision.load(Ordering::Relaxed) {
                    match reply.text {
                        Ok(text) => {
                            app.status = if text.is_empty() {
                                "No completion"
                            } else {
                                "Tab to accept"
                            }
                            .into();
                            app.ghost = text;
                        }
                        Err(e) => app.status = format!("Completion: {e}"),
                    }
                }
            }
            if completion_online
                && !busy
                && !app.confirm_quit
                && app.due.is_some_and(|at| Instant::now() >= at)
            {
                let request = Request {
                    revision: app.revision.load(Ordering::Relaxed),
                    lines: app.preceding(args.context_lines),
                };
                completion_online = requests.send(request).is_ok();
                busy = completion_online;
                app.due = None;
                app.status = if completion_online {
                    "Completing..."
                } else {
                    "Completion worker stopped; editing and saving remain available"
                }
                .into();
            }
            draw(&mut terminal, &mut app)?;
            if !event::poll(Duration::from_millis(25))? {
                continue;
            }
            let key = match event::read()? {
                Event::Key(key) if key.kind != KeyEventKind::Release => key,
                Event::Paste(text) if !app.confirm_quit => {
                    app.insert(&text);
                    app.invalidate(Some(debounce));
                    continue;
                }
                _ => continue,
            };
            if app.confirm_quit {
                match key.code {
                    KeyCode::Char('y' | 'Y') => match app.save() {
                        Ok(()) => break,
                        Err(e) => {
                            app.status = e.to_string();
                            app.confirm_quit = false;
                        }
                    },
                    KeyCode::Char('n' | 'N') => break,
                    KeyCode::Esc => {
                        app.confirm_quit = false;
                        app.status = "Exit cancelled".into();
                    }
                    _ => {}
                }
                continue;
            }
            if key.modifiers.contains(KeyModifiers::CONTROL) {
                match key.code {
                    KeyCode::Char('x' | 'q') => {
                        if !app.dirty() {
                            break;
                        }
                        app.invalidate(None);
                        app.confirm_quit = true;
                    }
                    KeyCode::Char('s' | 'o') => {
                        if let Err(e) = app.save() {
                            app.status = e.to_string();
                        }
                    }
                    KeyCode::Char('r') => {
                        app.invalidate(Some(Duration::ZERO));
                        app.status = "Regenerating...".into();
                    }
                    _ => {}
                }
                continue;
            }
            let mut changed = true;
            match key.code {
                KeyCode::Esc => {
                    app.invalidate(None);
                    app.status = "Dismissed".into();
                    continue;
                }
                KeyCode::Tab => {
                    let text = if app.ghost.is_empty() {
                        "    ".into()
                    } else {
                        std::mem::take(&mut app.ghost)
                    };
                    app.insert(&text);
                }
                KeyCode::Enter => app.insert("\n"),
                KeyCode::Backspace => app.backspace(),
                KeyCode::Delete => app.delete(),
                KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::ALT) => {
                    app.insert(&c.to_string())
                }
                KeyCode::Left => {
                    if app.col > 0 {
                        app.col = app.lines[app.row][..app.col]
                            .char_indices()
                            .last()
                            .unwrap()
                            .0;
                    } else if app.row > 0 {
                        app.row -= 1;
                        app.col = app.lines[app.row].len();
                    }
                }
                KeyCode::Right => {
                    if app.col < app.lines[app.row].len() {
                        app.col += app.lines[app.row][app.col..]
                            .chars()
                            .next()
                            .unwrap()
                            .len_utf8();
                    } else if app.row + 1 < app.lines.len() {
                        app.row += 1;
                        app.col = 0;
                    }
                }
                KeyCode::Up => app.vertical(app.row.saturating_sub(1)),
                KeyCode::Down => app.vertical(app.row + 1),
                KeyCode::Home => app.col = 0,
                KeyCode::End => app.col = app.lines[app.row].len(),
                KeyCode::PageUp => app.vertical(
                    app.row
                        .saturating_sub(terminal.size()?.height.saturating_sub(3) as usize),
                ),
                KeyCode::PageDown => {
                    app.vertical(app.row + terminal.size()?.height.saturating_sub(3) as usize)
                }
                _ => changed = false,
            }
            if changed {
                app.invalidate(Some(debounce));
                app.status = "Ready".into();
            }
        }
        app.invalidate(None);
        Ok(())
    }

    pub fn main() -> Result<()> {
        let args = Args::parse();
        ensure!(
            io::stdin().is_terminal() && io::stdout().is_terminal(),
            "run yumon_nano in an interactive terminal"
        );
        let app = Editor::open(args.file.clone())?;
        let revision = Arc::clone(&app.revision);
        let (request_tx, request_rx) = mpsc::sync_channel::<Request>(1);
        let (reply_tx, reply_rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::channel();
        eprintln!("Loading Yumon Code from {}...", args.checkpoint);
        let worker_args = args.clone();
        thread::spawn(move || {
            let device = Device::default();
            let loaded = Model::load(&worker_args, &device);
            let (model, tokenizer, capacity) = match loaded {
                Ok(loaded) => loaded,
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("{e:#}")));
                    return;
                }
            };
            if ready_tx.send(Ok(())).is_err() {
                return;
            }
            while let Ok(request) = request_rx.recv() {
                let text = complete(
                    &model,
                    &tokenizer,
                    capacity,
                    worker_args.max_tokens as usize,
                    &device,
                    &request,
                    &revision,
                );
                if reply_tx
                    .send(Reply {
                        revision: request.revision,
                        text,
                    })
                    .is_err()
                {
                    break;
                }
            }
        });
        ready_rx
            .recv()
            .context("model worker stopped during loading")?
            .map_err(anyhow::Error::msg)?;
        run_editor(&args, app, request_tx, reply_rx)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn editor(text: &str) -> Editor {
            let mut app = Editor::open(PathBuf::from("__yumon_nano_test_not_saved__")).unwrap();
            app.lines = text.split('\n').map(str::to_owned).collect();
            app.saved = text.into();
            app
        }
        #[test]
        fn context_keeps_current_prefix_and_nearest_whole_lines() {
            let lines = vec!["old".into(), "near".into(), "now".into()];
            let encode = |s: &str| Ok(s.bytes().map(u32::from).collect());
            assert_eq!(
                context_tokens(&lines, 8, encode).unwrap(),
                encode("near\nnow").unwrap()
            );
            assert_eq!(
                context_tokens(&lines, 3, encode).unwrap(),
                encode("now").unwrap()
            );
            assert!(context_tokens(&lines, 2, encode).is_err());
            assert!(
                context_tokens(&[String::new()], 0, encode)
                    .unwrap()
                    .is_empty()
            );
        }
        #[test]
        fn cursor_context_excludes_suffix_and_later_lines() {
            let mut app = editor("fn f() {\n    let x = 1;\n}");
            app.row = 1;
            app.col = 8;
            assert_eq!(app.preceding(40), vec!["fn f() {", "    let "]);
            assert_eq!(app.preceding(0), vec!["    let "]);
        }
        #[test]
        fn unicode_editing_and_multiline_accept_preserve_suffix() {
            let mut app = editor("aé尾");
            app.col = "aé".len();
            app.insert("λ\nnext");
            assert_eq!(app.text(), "aéλ\nnext尾");
            assert_eq!((app.row, app.col), (1, 4));
            app.delete();
            assert_eq!(app.text(), "aéλ\nnext");
            app.vertical(0);
            app.backspace();
            assert_eq!(app.text(), "aé\nnext");
            app.col = 0;
            app.vertical(1);
            app.backspace();
            assert_eq!(app.text(), "aénext");
        }
        #[test]
        fn edits_and_dismissals_invalidate_inflight_completions() {
            let mut app = editor("abc");
            let old = app.revision.load(Ordering::Relaxed);
            app.ghost = "def".into();
            app.invalidate(Some(Duration::ZERO));
            assert_ne!(old, app.revision.load(Ordering::Relaxed));
            assert!(app.ghost.is_empty());
            assert!(app.due.is_some());
            app.invalidate(None);
            assert!(app.due.is_none());
        }
        #[test]
        fn ghost_rendering_handles_tabs_unicode_and_horizontal_scroll() {
            let line = clipped(vec![("\t尾", Color::Reset), ("λ", Color::DarkGray)], 4, 3);
            assert_eq!(line.to_string(), "尾λ");
            assert_eq!(line.spans.last().unwrap().style.fg, Some(Color::DarkGray));
        }

        #[test]
        fn save_preserves_crlf_and_refuses_external_edits() {
            let path = std::env::temp_dir().join(format!("yumon-nano-{}.rs", uuid::Uuid::new_v4()));
            fs::write(&path, "a\r\nb\r\n").unwrap();
            let mut app = Editor::open(path.clone()).unwrap();
            app.col = 1;
            app.insert("é");
            app.save().unwrap();
            assert_eq!(fs::read_to_string(&path).unwrap(), "aé\r\nb\r\n");
            assert!(!app.dirty());
            fs::write(&path, "external").unwrap();
            app.insert("more");
            assert!(app.save().is_err());
            assert_eq!(fs::read_to_string(&path).unwrap(), "external");
            fs::remove_file(path).unwrap();
        }

        #[test]
        fn new_file_is_created_only_on_save() {
            let path = std::env::temp_dir().join(format!("yumon-nano-{}.rs", uuid::Uuid::new_v4()));
            let mut app = Editor::open(path.clone()).unwrap();
            app.insert("fn main() {}\n");
            assert!(!path.exists());
            app.save().unwrap();
            assert_eq!(fs::read_to_string(&path).unwrap(), "fn main() {}\n");
            fs::remove_file(path).unwrap();
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    native::main()
}

#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("Yumon Nano requires a native terminal.");
}
