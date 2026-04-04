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
    pub lr_history: Vec<(f64, f64)>,
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
        state.last_reply,
    );
    let para = Paragraph::new(stats)
        .block(Block::default().borders(Borders::ALL).title("Training"));
    frame.render_widget(para, chunks[0]);

    // ── Chart ───────────────────────────────────────────────────────────────
    let max_loss = state.loss_history.iter()
        .map(|&(_, l)| l)
        .fold(0.0_f64, f64::max)
        .max(0.01);

    let lr_ds = Dataset::default()
        .name("lr")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Green))
        .data(&state.lr_history);

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
    let chart = Chart::new(vec![loss_ds, avg_ds, entropy_ds, lr_ds])
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

use image::{Rgb, RgbImage};

impl TrainingState {
    pub fn save_chart_image(&self, path: &str) -> anyhow::Result<()> {
        let width = 1200;
        let height = 800;
        let mut img = RgbImage::new(width, height);

        // Background
        for pixel in img.pixels_mut() {
            *pixel = Rgb([10, 10, 15]);
        }

        let margin = 60;
        let graph_width = width - 2 * margin;
        let graph_height = height - 2 * margin;

        let max_loss = self.loss_history.iter()
            .map(|&(_, l)| l)
            .fold(0.0_f64, f64::max)
            .max(0.01);

        let max_steps = self.global_step.max(1) as f64;

        let to_x = |step: f64| (margin as f64 + (step / max_steps) * graph_width as f64) as u32;
        let to_y = |val: f64| (height as f64 - margin as f64 - (val / (max_loss * 1.1)) * graph_height as f64) as u32;

        // Draw axes
        let white = Rgb([200, 200, 200]);
        for x in margin..(width - margin) {
            img.put_pixel(x, height - margin, white);
        }
        for y in margin..(height - margin) {
            img.put_pixel(margin, y, white);
        }

        // Draw loss (Cyan)
        let cyan = Rgb([0, 255, 255]);
        self.draw_line(&mut img, &self.loss_history, cyan, to_x, to_y);

        // Draw avg loss (Yellow)
        let yellow = Rgb([255, 255, 0]);
        self.draw_line(&mut img, &self.avg_loss_history, yellow, to_x, to_y);

        // Draw entropy (Magenta)
        let magenta = Rgb([255, 0, 255]);
        self.draw_line(&mut img, &self.entropy_history, magenta, to_x, to_y);

        // Draw Learning Rate (Green)
        // Scale LR to match max_loss for visibility
        let max_lr = self.lr_history.iter()
            .map(|&(_, lr)| lr)
            .fold(0.0_f64, f64::max)
            .max(1e-10);

        let lr_scale = (max_loss * 0.8) / max_lr;
        let green = Rgb([0, 255, 0]);
        let to_y_lr = |lr: f64| to_y(lr * lr_scale);
        self.draw_line(&mut img, &self.lr_history, green, to_x, to_y_lr);

        img.save(path)?;
        Ok(())
    }
    fn draw_line<FX, FY>(&self, img: &mut RgbImage, data: &[(f64, f64)], color: Rgb<u8>, to_x: FX, to_y: FY)
    where
        FX: Fn(f64) -> u32,
        FY: Fn(f64) -> u32,
    {
        if data.len() < 2 { return; }
        for window in data.windows(2) {
            let (x1, y1) = (to_x(window[0].0), to_y(window[0].1));
            let (x2, y2) = (to_x(window[1].0), to_y(window[1].1));
            self.draw_pixel_line(img, x1, y1, x2, y2, color);
        }
    }

    fn draw_pixel_line(&self, img: &mut RgbImage, x1: u32, y1: u32, x2: u32, y2: u32, color: Rgb<u8>) {
        let dx = (x2 as i32 - x1 as i32).abs();
        let dy = (y2 as i32 - y1 as i32).abs();
        let sx = if x1 < x2 { 1 } else { -1 };
        let sy = if y1 < y2 { 1 } else { -1 };
        let mut err = dx - dy;

        let mut x = x1 as i32;
        let mut y = y1 as i32;

        loop {
            if x >= 0 && x < img.width() as i32 && y >= 0 && y < img.height() as i32 {
                img.put_pixel(x as u32, y as u32, color);
            }
            if x == x2 as i32 && y == y2 as i32 { break; }
            let e2 = 2 * err;
            if e2 > -dy {
                err -= dy;
                x += sx;
            }
            if e2 < dx {
                err += dx;
                y += sy;
            }
        }
    }
    }