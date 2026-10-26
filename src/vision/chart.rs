//! Live Terminal TUI Dashboard and Chart for Pass 4 Upscaler Training.
//!
//! Provides an interactive in-terminal dashboard via Ratatui, featuring:
//! - Top statistics panel (epochs, batches, learning rate, G/D/Adv losses, val loss, ETA)
//! - Side-by-side live loss charts (Generator loss & EMA vs. GAN equilibrium D/Adv dynamics)
//! - Overall progress gauge
//! - PNG export of training curves

use std::time::Instant;

#[cfg(not(target_arch = "wasm32"))]
use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    symbols,
    widgets::{Axis, Block, Borders, Chart, Dataset, Gauge, GraphType, Paragraph},
};

pub const EMA_ALPHA: f64 = 0.05;

/// Exponential moving average of a (step, value) series.
pub fn ema(data: &[(f64, f64)], alpha: f64) -> Vec<(f64, f64)> {
    let mut previous: Option<f64> = None;
    data.iter()
        .map(|&(x, y)| {
            let v = previous.map_or(y, |p| p + alpha * (y - p));
            previous = Some(v);
            (x, v)
        })
        .collect()
}

pub struct UpscalerTrainingState {
    pub epoch: usize,
    pub total_epochs: usize,
    pub batch: usize,
    pub total_batches: usize,
    pub global_step: usize,

    pub current_lr: f64,
    pub current_g_loss: f32,
    pub current_d_loss: f32,
    pub current_adv_loss: f32,
    pub current_val_loss: Option<f32>,

    pub g_loss_history: Vec<(f64, f64)>,
    pub d_loss_history: Vec<(f64, f64)>,
    pub adv_loss_history: Vec<(f64, f64)>,
    pub val_loss_history: Vec<(f64, f64)>,
    pub lr_history: Vec<(f64, f64)>,

    pub is_gan: bool,
    pub start_time: Instant,
    pub last_preview_path: Option<String>,
}

impl UpscalerTrainingState {
    pub fn new(total_epochs: usize, total_batches: usize, is_gan: bool) -> Self {
        Self {
            epoch: 1,
            total_epochs,
            batch: 0,
            total_batches,
            global_step: 0,
            current_lr: 0.0,
            current_g_loss: 0.0,
            current_d_loss: 0.0,
            current_adv_loss: 0.0,
            current_val_loss: None,
            g_loss_history: Vec::new(),
            d_loss_history: Vec::new(),
            adv_loss_history: Vec::new(),
            val_loss_history: Vec::new(),
            lr_history: Vec::new(),
            is_gan,
            start_time: Instant::now(),
            last_preview_path: None,
        }
    }

    /// Save training metrics to a CSV file.
    pub fn save_csv(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);

        if self.is_gan {
            writeln!(writer, "step,g_loss,d_loss,adv_loss,lr")?;
            for (i, &(step, g)) in self.g_loss_history.iter().enumerate() {
                let d = self.d_loss_history.get(i).map_or(0.0, |p| p.1);
                let adv = self.adv_loss_history.get(i).map_or(0.0, |p| p.1);
                let lr = self.lr_history.get(i).map_or(0.0, |p| p.1);
                writeln!(writer, "{},{},{},{},{}", step, g, d, adv, lr)?;
            }
        } else {
            writeln!(writer, "step,loss,lr")?;
            for (i, &(step, g)) in self.g_loss_history.iter().enumerate() {
                let lr = self.lr_history.get(i).map_or(0.0, |p| p.1);
                writeln!(writer, "{},{},{}", step, g, lr)?;
            }
        }

        writer.flush()?;
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn render(frame: &mut ratatui::Frame, state: &UpscalerTrainingState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header stats bar
            Constraint::Min(12),   // Charts area (split into columns)
            Constraint::Length(3), // Progress gauge
        ])
        .split(frame.area());

    // 1. Header Stats Bar
    let elapsed = state.start_time.elapsed();
    let elapsed_secs = elapsed.as_secs();
    let elapsed_str = format!("{:02}:{:02}", elapsed_secs / 60, elapsed_secs % 60);

    let val_str = state
        .current_val_loss
        .map_or("N/A".to_string(), |v| format!("{:.4}", v));

    let stats_text = if state.is_gan {
        format!(
            " Epoch {}/{} │ Batch {}/{} │ LR {:.2e} │ G: {:.4} │ D: {:.4} │ Adv: {:.4} │ Val: {} │ ⏱ {}",
            state.epoch,
            state.total_epochs,
            state.batch,
            state.total_batches,
            state.current_lr,
            state.current_g_loss,
            state.current_d_loss,
            state.current_adv_loss,
            val_str,
            elapsed_str,
        )
    } else {
        format!(
            " Epoch {}/{} │ Batch {}/{} │ LR {:.2e} │ Loss: {:.4} │ Val: {} │ ⏱ {}",
            state.epoch,
            state.total_epochs,
            state.batch,
            state.total_batches,
            state.current_lr,
            state.current_g_loss,
            val_str,
            elapsed_str,
        )
    };

    let title = if state.is_gan {
        " Yumon Pass 4 Super-Resolution GAN (32x32 -> 256x256) "
    } else {
        " Yumon Pass 4 Super-Resolution Upscaler (32x32 -> 256x256) "
    };

    let header = Paragraph::new(stats_text)
        .style(Style::default().fg(Color::White))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(title)
                .title_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        );
    frame.render_widget(header, chunks[0]);

    // 2. Charts Area
    if state.is_gan {
        let chart_cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(55), // Generator & Val Loss
                Constraint::Percentage(45), // Discriminator & Adversarial Dynamics
            ])
            .split(chunks[1]);

        render_g_chart(frame, chart_cols[0], state);
        render_gan_chart(frame, chart_cols[1], state);
    } else {
        let chart_cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(60), // Loss
                Constraint::Percentage(40), // Learning Rate
            ])
            .split(chunks[1]);

        render_g_chart(frame, chart_cols[0], state);
        render_lr_chart(frame, chart_cols[1], state);
    }

    // 3. Progress Gauge
    let total_batches_all = state.total_epochs * state.total_batches;
    let completed_batches =
        (state.epoch.saturating_sub(1)) * state.total_batches + state.batch;
    let ratio = if total_batches_all > 0 {
        (completed_batches as f64 / total_batches_all as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let label = format!(
        "Progress: {:.1}% (Epoch {}/{} - Batch {}/{})",
        ratio * 100.0,
        state.epoch,
        state.total_epochs,
        state.batch,
        state.total_batches,
    );

    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title("Total Progress"))
        .gauge_style(Style::default().fg(Color::Cyan))
        .ratio(ratio)
        .label(label);
    frame.render_widget(gauge, chunks[2]);
}

#[cfg(not(target_arch = "wasm32"))]
fn to_log10_points(data: &[(f64, f64)]) -> Vec<(f64, f64)> {
    data.iter()
        .filter(|p| p.1 > 0.0 && p.1.is_finite())
        .map(|&(x, y)| (x, y.log10()))
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
fn fmt_loss(v: f64) -> String {
    if v >= 1.0 {
        format!("{:.2}", v)
    } else if v >= 0.1 {
        format!("{:.3}", v)
    } else if v >= 0.01 {
        format!("{:.4}", v)
    } else {
        format!("{:.1e}", v)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn render_g_chart(
    frame: &mut ratatui::Frame,
    area: ratatui::layout::Rect,
    state: &UpscalerTrainingState,
) {
    let raw = &state.g_loss_history;
    let smooth = ema(raw, EMA_ALPHA);

    let raw_log = to_log10_points(raw);
    let smooth_log = to_log10_points(&smooth);
    let val_log = to_log10_points(&state.val_loss_history);

    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for &(_, y) in raw_log.iter().chain(&val_log) {
        if y.is_finite() {
            y_min = y_min.min(y);
            y_max = y_max.max(y);
        }
    }

    if !y_min.is_finite() {
        y_min = -2.0; // 0.01
    }
    if !y_max.is_finite() || y_max <= y_min {
        y_max = y_min + 1.0;
    }

    // Add 5% headroom and footroom on log scale
    let y_span = (y_max - y_min).max(0.2);
    let y_low = y_min - y_span * 0.05;
    let y_high = y_max + y_span * 0.05;
    let y_mid = (y_low + y_high) * 0.5;

    let x_max = (state.global_step as f64).max(10.0);

    let mut datasets = vec![
        Dataset::default()
            .name("G Raw")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::DarkGray))
            .data(&raw_log),
        Dataset::default()
            .name("G EMA")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Cyan))
            .data(&smooth_log),
    ];

    if !val_log.is_empty() {
        datasets.push(
            Dataset::default()
                .name("Val Huber")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Yellow))
                .data(&val_log),
        );
    }

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .title(" Generator Loss (Log10 scale; Cyan=EMA, Yellow=Val) ")
                .borders(Borders::ALL),
        )
        .x_axis(
            Axis::default()
                .bounds([0.0, x_max])
                .labels(vec!["0".to_string(), format!("{}", state.global_step)]),
        )
        .y_axis(
            Axis::default()
                .bounds([y_low, y_high])
                .labels(vec![
                    fmt_loss(10.0f64.powf(y_low)),
                    fmt_loss(10.0f64.powf(y_mid)),
                    fmt_loss(10.0f64.powf(y_high)),
                ]),
        );

    frame.render_widget(chart, area);
}

#[cfg(not(target_arch = "wasm32"))]
fn render_gan_chart(
    frame: &mut ratatui::Frame,
    area: ratatui::layout::Rect,
    state: &UpscalerTrainingState,
) {
    let d_data = &state.d_loss_history;
    let adv_data = &state.adv_loss_history;

    let mut y_max = 1.2f64;
    for &(_, y) in d_data.iter().chain(adv_data) {
        if y.is_finite() {
            y_max = y_max.max(y * 1.1);
        }
    }

    let x_max = (state.global_step as f64).max(10.0);

    let datasets = vec![
        Dataset::default()
            .name("D Loss")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::LightRed))
            .data(d_data),
        Dataset::default()
            .name("Adv Loss")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::LightMagenta))
            .data(adv_data),
    ];

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .title(" GAN Dynamics (Red=D, Magenta=Adv) ")
                .borders(Borders::ALL),
        )
        .x_axis(
            Axis::default()
                .bounds([0.0, x_max])
                .labels(vec!["0".to_string(), format!("{}", state.global_step)]),
        )
        .y_axis(
            Axis::default()
                .bounds([0.0, y_max])
                .labels(vec!["0.0".to_string(), format!("{:.2}", y_max)]),
        );

    frame.render_widget(chart, area);
}

#[cfg(not(target_arch = "wasm32"))]
fn render_lr_chart(
    frame: &mut ratatui::Frame,
    area: ratatui::layout::Rect,
    state: &UpscalerTrainingState,
) {
    let lr_data = &state.lr_history;
    let max_lr = lr_data.iter().map(|p| p.1).fold(0.0_f64, f64::max).max(1e-6);
    let x_max = (state.global_step as f64).max(10.0);

    let datasets = vec![
        Dataset::default()
            .name("LR")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Green))
            .data(lr_data),
    ];

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .title(" Learning Rate ")
                .borders(Borders::ALL),
        )
        .x_axis(
            Axis::default()
                .bounds([0.0, x_max])
                .labels(vec!["0".to_string(), format!("{}", state.global_step)]),
        )
        .y_axis(
            Axis::default()
                .bounds([0.0, max_lr * 1.1])
                .labels(vec!["0".to_string(), format!("{:.1e}", max_lr)]),
        );

    frame.render_widget(chart, area);
}

// ─── Colorizer Training Dashboard ──────────────────────────────────────────

pub struct ColorizerTrainingState {
    pub epoch: usize,
    pub total_epochs: usize,
    pub batch: usize,
    pub total_batches: usize,
    pub global_step: usize,

    pub current_lr: f64,
    pub current_loss: f32,
    pub current_acc: f32,
    pub current_val_loss: Option<f32>,
    pub current_val_acc: Option<f32>,

    pub loss_history: Vec<(f64, f64)>,
    pub val_loss_history: Vec<(f64, f64)>,
    pub acc_history: Vec<(f64, f64)>,
    pub lr_history: Vec<(f64, f64)>,

    pub start_time: Instant,
    pub last_preview_path: Option<String>,
}

impl ColorizerTrainingState {
    pub fn new(total_epochs: usize, total_batches: usize) -> Self {
        Self {
            epoch: 1,
            total_epochs,
            batch: 0,
            total_batches,
            global_step: 0,
            current_lr: 0.0,
            current_loss: 0.0,
            current_acc: 0.0,
            current_val_loss: None,
            current_val_acc: None,
            loss_history: Vec::new(),
            val_loss_history: Vec::new(),
            acc_history: Vec::new(),
            lr_history: Vec::new(),
            start_time: Instant::now(),
            last_preview_path: None,
        }
    }

    pub fn save_csv(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);

        writeln!(writer, "step,loss,acc,lr")?;
        for (i, &(step, loss)) in self.loss_history.iter().enumerate() {
            let acc = self.acc_history.get(i).map_or(0.0, |p| p.1);
            let lr = self.lr_history.get(i).map_or(0.0, |p| p.1);
            writeln!(writer, "{},{},{},{}", step, loss, acc, lr)?;
        }
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn render_colorizer_chart(frame: &mut ratatui::Frame, state: &ColorizerTrainingState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header stats
            Constraint::Min(10),   // Charts
            Constraint::Length(3), // Progress gauge
        ])
        .split(frame.area());

    // 1. Header
    let elapsed = state.start_time.elapsed();
    let elapsed_str = format!("{:02}:{:02}", elapsed.as_secs() / 60, elapsed.as_secs() % 60);

    let val_str = match (state.current_val_loss, state.current_val_acc) {
        (Some(vl), Some(va)) => format!("{:.4} (Acc: {:.1}%)", vl, va),
        (Some(vl), None) => format!("{:.4}", vl),
        _ => "N/A".to_string(),
    };

    let stats_text = format!(
        " Epoch {}/{} │ Batch {}/{} │ LR {:.2e} │ Loss: {:.4} │ Top-1 Acc: {:.1}% │ Val: {} │ ⏱ {}",
        state.epoch,
        state.total_epochs,
        state.batch,
        state.total_batches,
        state.current_lr,
        state.current_loss,
        state.current_acc,
        val_str,
        elapsed_str,
    );

    let header = Paragraph::new(stats_text)
        .style(Style::default().fg(Color::White))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Yumon Pass 6 Full Zhang 2016 Colorizer (256x256 L -> 313 Color Bins) ")
                .title_style(Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        );
    frame.render_widget(header, chunks[0]);

    // 2. Charts
    let chart_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(55), // Loss
            Constraint::Percentage(45), // Top-1 Accuracy & LR
        ])
        .split(chunks[1]);

    // Loss Chart
    let loss_data = &state.loss_history;
    let ema_data = ema(loss_data, EMA_ALPHA);
    let val_data = &state.val_loss_history;

    let max_loss = loss_data
        .iter()
        .chain(val_data.iter())
        .map(|p| p.1)
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let min_loss = loss_data
        .iter()
        .chain(val_data.iter())
        .map(|p| p.1)
        .fold(max_loss, f64::min)
        .min(max_loss * 0.9);

    let x_max = (state.global_step as f64).max(10.0);

    let mut datasets = vec![
        Dataset::default()
            .name("Loss")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::DarkGray))
            .data(loss_data),
        Dataset::default()
            .name("EMA")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Magenta))
            .data(&ema_data),
    ];

    if !val_data.is_empty() {
        datasets.push(
            Dataset::default()
                .name("Val")
                .marker(symbols::Marker::Dot)
                .graph_type(GraphType::Scatter)
                .style(Style::default().fg(Color::Yellow))
                .data(val_data),
        );
    }

    let loss_chart = Chart::new(datasets)
        .block(
            Block::default()
                .title(" Rebalanced Multinomial CE Loss (Magenta=EMA, Yellow=Val) ")
                .borders(Borders::ALL),
        )
        .x_axis(
            Axis::default()
                .bounds([0.0, x_max])
                .labels(vec!["0".to_string(), format!("{}", state.global_step)]),
        )
        .y_axis(
            Axis::default()
                .bounds([min_loss * 0.95, max_loss * 1.05])
                .labels(vec![format!("{:.2}", min_loss), format!("{:.2}", max_loss)]),
        );
    frame.render_widget(loss_chart, chart_cols[0]);

    // Acc Chart
    let acc_data = &state.acc_history;
    let acc_datasets = vec![
        Dataset::default()
            .name("Top-1 Acc %")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Cyan))
            .data(acc_data),
    ];

    let acc_chart = Chart::new(acc_datasets)
        .block(
            Block::default()
                .title(" Color Bin Classification Accuracy ")
                .borders(Borders::ALL),
        )
        .x_axis(
            Axis::default()
                .bounds([0.0, x_max])
                .labels(vec!["0".to_string(), format!("{}", state.global_step)]),
        )
        .y_axis(
            Axis::default()
                .bounds([0.0, 100.0])
                .labels(vec!["0%".to_string(), "50%".to_string(), "100%".to_string()]),
        );
    frame.render_widget(acc_chart, chart_cols[1]);

    // 3. Progress Gauge
    let total_batches_all = state.total_epochs * state.total_batches;
    let completed_batches =
        (state.epoch.saturating_sub(1)) * state.total_batches + state.batch;
    let progress_ratio = if total_batches_all > 0 {
        (completed_batches as f64 / total_batches_all as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let gauge_label = if let Some(ref path) = state.last_preview_path {
        format!("{:.1}% │ Last preview: {}", progress_ratio * 100.0, path)
    } else {
        format!("{:.1}%", progress_ratio * 100.0)
    };

    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" Overall Progress "))
        .gauge_style(
            Style::default()
                .fg(Color::Magenta)
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .ratio(progress_ratio)
        .label(gauge_label);
    frame.render_widget(gauge, chunks[2]);
}
