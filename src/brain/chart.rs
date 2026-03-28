pub struct TrainingState {
    pub loss_history: Vec<(f64, f64)>,   // (batch_global, loss)
    pub avg_loss_history: Vec<(f64, f64)>,
    pub current_loss: f32,
    pub avg_loss: f32,
    pub epoch: usize,
    pub total_epochs: usize,
    pub batch: usize,
    pub total_batches: usize,
    pub current_lr: f64,
    pub global_step: usize,
    pub entropy: f32,
    pub entropy_history: Vec<(f64, f64)>,
    pub last_reply: String,
}

use ratatui::{
    Terminal, TerminalOptions, Viewport,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    symbols,
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Gauge, Paragraph},
};

pub fn render(frame: &mut ratatui::Frame, state: &TrainingState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),   // stats bar
            Constraint::Min(10),     // chart
        ])
        .split(frame.area());

    // ── Top bar: epoch / batch / lr / loss ──────────────────────────────────
    let stats = format!(
        " Epoch {}/{} │ Batch {}/{} │ LR {:.2e} │ Loss {:.4} │ Avg {:.4} | Entropy: {:.4} | Last Reply: {:?}",
        state.epoch, state.total_epochs,
        state.batch, state.total_batches,
        state.current_lr,
        state.current_loss,
        state.avg_loss,
        state.entropy,
        state.last_reply
    );
    let para = Paragraph::new(stats)
        .block(Block::default().borders(Borders::ALL).title("Training"));
    frame.render_widget(para, chunks[0]);

    // ── Chart ───────────────────────────────────────────────────────────────
    let max_loss = state.loss_history.iter()
        .map(|&(_, l)| l)
        .fold(0.0_f64, f64::max)
        .max(0.01);

    let entropy_ds = Dataset::default()
        .name("entropy")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Magenta))
        .data(&state.entropy_history);

    let loss_ds = Dataset::default()
        .name("loss")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Cyan))
        .data(&state.loss_history);

    let avg_ds = Dataset::default()
        .name("avg")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Yellow))
        .data(&state.avg_loss_history);

    let n = state.global_step.max(1) as f64;
    let chart = Chart::new(vec![loss_ds, avg_ds, entropy_ds])
        .block(Block::default().title("Loss").borders(Borders::ALL))
        .x_axis(
            Axis::default()
                .bounds([0.0, n])
                .labels(["0".to_string(), format!("{}", state.global_step)]),
        )
        .y_axis(
            Axis::default()
                .bounds([0.0, max_loss * 1.1])
                .labels(["0".to_string(), format!("{:.3}", max_loss)]),
        );
    frame.render_widget(chart, chunks[1]);
}