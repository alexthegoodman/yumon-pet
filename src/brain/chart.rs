pub struct TrainingState {
    pub loss_history: Vec<(f64, f64)>,   // (batch_global, loss)
    pub avg_loss_history: Vec<(f64, f64)>,
    pub val_loss_history: Vec<(f64, f64)>, // (batch_global, val_loss)
    pub val_top3_history: Vec<(f64, f64)>, // (batch_global, val_top3)
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
    /// Fractions in [0, 1]; unavailable for the legacy training paths.
    pub top3_accuracy: Option<f64>,
    pub avg_top3_accuracy: Option<f64>,
    pub top3_history: Vec<(f64, f64)>,
    pub last_reply: String,
}

/// Smoothing factor for the loss EMA drawn over the raw per-batch points.
pub const EMA_ALPHA: f64 = 0.05;

/// Exponential moving average of a (step, value) series.
pub fn ema(data: &[(f64, f64)], alpha: f64) -> Vec<(f64, f64)> {
    let mut previous: Option<f64> = None;
    data.iter().map(|&(x, y)| {
        let v = previous.map_or(y, |p| p + alpha * (y - p));
        previous = Some(v);
        (x, v)
    }).collect()
}

#[cfg(not(target_arch = "wasm32"))]
use ratatui::{
    Terminal, TerminalOptions, Viewport,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    symbols,
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Gauge, Paragraph},
};

/// (ln step, log10 value) for the terminal chart, which has no log axes of its own.
#[cfg(not(target_arch = "wasm32"))]
fn log_xy(data: &[(f64, f64)]) -> Vec<(f64, f64)> {
    data.iter().filter(|p| p.1 > 0.0 && p.1.is_finite())
        .map(|&(x, y)| (x.max(1.0).ln(), y.log10())).collect()
}

#[cfg(not(target_arch = "wasm32"))]
pub fn render(frame: &mut ratatui::Frame, state: &TrainingState) {
    use self::plot::fmt_num;
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),   // stats bar
            Constraint::Min(10),     // loss chart (log-log)
            Constraint::Length(7),   // learning rate
            Constraint::Length(8),   // top-3 accuracy
        ])
        .split(frame.area());

    // ── Top bar: epoch / batch / lr / loss ──────────────────────────────────
    let top3 = match (state.top3_accuracy, state.avg_top3_accuracy) {
        (Some(batch), Some(avg)) => format!(" | Top-3: {:.2}% (epoch {:.2}%)", batch * 100.0, avg * 100.0),
        _ => String::new(),
    };
    let val_str = state.val_loss_history.last().map_or(String::new(), |&(_, v)| format!(" | Val: {:.4}", v));
    let stats = format!(
        " Epoch {}/{} │ Batch {}/{} │ LR {:.2e} │ Loss {:.4} │ Avg {:.4}{} | Entropy: {:.4} {} | Last Reply: {:?}",
        state.epoch, state.total_epochs,
        state.batch, state.total_batches,
        state.current_lr,
        state.current_loss,
        state.avg_loss,
        val_str,
        state.entropy,
        top3,
        state.last_reply,
    );
    let para = Paragraph::new(stats)
        .block(Block::default().borders(Borders::ALL).title("Training"));
    frame.render_widget(para, chunks[0]);

    // ── Loss: raw (dim), EMA, epoch average, entropy, and validation on log10 axes ──────
    let raw = log_xy(&state.loss_history);
    let smooth = log_xy(&ema(&state.loss_history, EMA_ALPHA));
    let avg = log_xy(&state.avg_loss_history);
    let entropy = log_xy(&state.entropy_history);
    let val_loss = log_xy(&state.val_loss_history);
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &(_, y) in raw.iter().chain(&entropy).chain(&avg).chain(&val_loss) {
        lo = lo.min(y);
        hi = hi.max(y);
    }
    if !lo.is_finite() { lo = -1.0; hi = 1.0; }
    let (lo, hi) = (lo.floor(), (hi.ceil()).max(lo.floor() + 1.0));

    fn datasets<'a>(name: &'a str, color: Color, data: &'a [(f64, f64)]) -> Dataset<'a> {
        Dataset::default().name(name).marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line).style(Style::default().fg(color)).data(data)
    }
    let n = (state.global_step.max(2) as f64).ln();
    let mut loss_datasets = vec![
        datasets("raw", Color::DarkGray, &raw),
        datasets("entropy", Color::Magenta, &entropy),
        datasets("avg", Color::Yellow, &avg),
        datasets("ema", Color::Cyan, &smooth),
    ];
    if !val_loss.is_empty() {
        loss_datasets.push(datasets("val", Color::Red, &val_loss));
    }
    let chart = Chart::new(loss_datasets)
        .block(Block::default().title("Loss (log-log; ema over raw; red=val)").borders(Borders::ALL))
        .x_axis(
            Axis::default()
                .bounds([0.0, n])
                .labels(["1".to_string(), format!("{}", state.global_step)]),
        )
        .y_axis(
            Axis::default()
                .bounds([lo, hi])
                .labels([fmt_num(10f64.powf(lo)), fmt_num(10f64.powf(hi))]),
        );
    frame.render_widget(chart, chunks[1]);

    // ── LR on its own axis ──────────────────────────────────────────────────
    let lr: Vec<(f64, f64)> = state.lr_history.iter().map(|&(x, y)| (x.max(1.0).ln(), y)).collect();
    let max_lr = state.lr_history.iter().map(|p| p.1).fold(0.0_f64, f64::max).max(1e-12);
    let lr_chart = Chart::new(vec![datasets("lr", Color::Green, &lr)])
        .block(Block::default().title("Learning rate").borders(Borders::ALL))
        .x_axis(Axis::default().bounds([0.0, n]))
        .y_axis(Axis::default().bounds([0.0, max_lr * 1.1])
            .labels(["0".to_string(), format!("{:.1e}", max_lr)]));
    frame.render_widget(lr_chart, chunks[2]);

    let top3_data: Vec<(f64, f64)> = state.top3_history.iter().map(|&(x, y)| (x.max(1.0).ln(), y)).collect();
    let val_top3_data: Vec<(f64, f64)> = state.val_top3_history.iter().map(|&(x, y)| (x.max(1.0).ln(), y)).collect();
    let mut top3_datasets = vec![datasets("top-3", Color::Blue, &top3_data)];
    if !val_top3_data.is_empty() {
        top3_datasets.push(datasets("val-top3", Color::Red, &val_top3_data));
    }
    let chart = Chart::new(top3_datasets)
        .block(Block::default().title("Top-3 token accuracy (red=val)").borders(Borders::ALL))
        .x_axis(Axis::default().bounds([0.0, n]))
        .y_axis(Axis::default().bounds([0.0, 1.0]).labels(["0%", "50%", "100%"]));
    frame.render_widget(chart, chunks[3]);
}

/// Software-rendered PNG plotting: log or linear axes, decade gridlines and
/// numeric tick labels from a tiny built-in digit font (no font dependency).
#[cfg(not(target_arch = "wasm32"))]
pub mod plot {
    use image::{Rgb, RgbImage};

    pub type Series<'a> = (&'a [(f64, f64)], Rgb<u8>);

    pub const BACKGROUND: Rgb<u8> = Rgb([10, 10, 15]);
    const PANEL: Rgb<u8> = Rgb([16, 16, 24]);
    const GRID: Rgb<u8> = Rgb([42, 42, 54]);
    const BORDER: Rgb<u8> = Rgb([120, 120, 130]);
    const LABEL: Rgb<u8> = Rgb([175, 175, 185]);

    fn glyph(c: char) -> [u8; 5] {
        match c {
            '0' => [0b111, 0b101, 0b101, 0b101, 0b111],
            '1' => [0b010, 0b110, 0b010, 0b010, 0b111],
            '2' => [0b111, 0b001, 0b111, 0b100, 0b111],
            '3' => [0b111, 0b001, 0b111, 0b001, 0b111],
            '4' => [0b101, 0b101, 0b111, 0b001, 0b001],
            '5' => [0b111, 0b100, 0b111, 0b001, 0b111],
            '6' => [0b111, 0b100, 0b111, 0b101, 0b111],
            '7' => [0b111, 0b001, 0b001, 0b001, 0b001],
            '8' => [0b111, 0b101, 0b111, 0b101, 0b111],
            '9' => [0b111, 0b101, 0b111, 0b001, 0b111],
            '.' => [0, 0, 0, 0, 0b010],
            '-' => [0, 0, 0b111, 0, 0],
            _ => [0; 5],
        }
    }

    /// Plain decimal without exponent or trailing zeros: 0.0003, 1.5, 128.
    pub fn fmt_num(v: f64) -> String {
        if v == 0.0 { return "0".into(); }
        let a = v.abs();
        let s = if a >= 100.0 { format!("{v:.0}") }
            else if a >= 1.0 { format!("{v:.2}") }
            else { format!("{:.*}", ((-a.log10().floor()) as usize + 1).min(10), v) };
        if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
    }

    pub fn put(img: &mut RgbImage, x: i32, y: i32, c: Rgb<u8>) {
        if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
            img.put_pixel(x as u32, y as u32, c);
        }
    }

    pub fn text_width(s: &str) -> i32 { s.chars().count() as i32 * 8 }

    pub fn text(img: &mut RgbImage, s: &str, x: i32, y: i32, color: Rgb<u8>) {
        let mut cx = x;
        for c in s.chars() {
            for (r, row) in glyph(c).iter().enumerate() {
                for col in 0..3 {
                    if (row >> (2 - col)) & 1 == 1 {
                        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                            put(img, cx + col * 2 + dx, y + r as i32 * 2 + dy, color);
                        }
                    }
                }
            }
            cx += 8;
        }
    }

    pub fn fill(img: &mut RgbImage, x: i32, y: i32, w: i32, h: i32, c: Rgb<u8>) {
        for yy in y..y + h { for xx in x..x + w { put(img, xx, yy, c); } }
    }

    pub fn line(img: &mut RgbImage, (x1, y1): (i32, i32), (x2, y2): (i32, i32), color: Rgb<u8>) {
        let (dx, dy) = ((x2 - x1).abs(), (y2 - y1).abs());
        let (sx, sy) = (if x1 < x2 { 1 } else { -1 }, if y1 < y2 { 1 } else { -1 });
        let (mut x, mut y, mut err) = (x1, y1, dx - dy);
        loop {
            put(img, x, y, color);
            if x == x2 && y == y2 { break; }
            let e2 = 2 * err;
            if e2 > -dy { err -= dy; x += sx; }
            if e2 < dx { err += dx; y += sy; }
        }
    }

    /// One axes rectangle. Ranges are in data units; `*log` axes map through ln.
    pub struct Panel {
        pub x: i32, pub y: i32, pub w: i32, pub h: i32,
        pub xr: (f64, f64), pub yr: (f64, f64),
        pub xlog: bool, pub ylog: bool,
    }

    impl Panel {
        pub fn new(x: i32, y: i32, w: i32, h: i32, xlog: bool, ylog: bool) -> Self {
            Self { x, y, w, h, xr: (1.0, 2.0), yr: (0.0, 1.0), xlog, ylog }
        }

        /// Fit both ranges to the data. Log axes snap to whole decades.
        pub fn fit(mut self, series: &[Series], zero_based: bool) -> Self {
            let (mut x0, mut x1, mut y0, mut y1) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
            for (data, _) in series {
                for &(x, y) in data.iter() {
                    if !x.is_finite() || !y.is_finite() || (self.xlog && x <= 0.0) || (self.ylog && y <= 0.0) { continue; }
                    x0 = x0.min(x); x1 = x1.max(x); y0 = y0.min(y); y1 = y1.max(y);
                }
            }
            if x0 > x1 { x0 = 1.0; x1 = 2.0; y0 = 0.1; y1 = 1.0; }
            self.xr = if self.xlog { (1.0, x1.max(2.0)) } else { (x0, if x1 > x0 { x1 } else { x0 + 1.0 }) };
            self.yr = if self.ylog {
                let (lo, mut hi) = (10f64.powf(y0.log10().floor()), 10f64.powf(y1.log10().ceil()));
                if hi <= lo * 1.0001 { hi = lo * 10.0; }
                (lo, hi)
            } else {
                let lo = if zero_based { y0.min(0.0) } else { y0 };
                let hi = y1 + 0.05 * (y1 - lo);
                (lo, if hi > lo { hi } else { lo + 1.0 })
            };
            self
        }

        pub fn y_range(mut self, lo: f64, hi: f64) -> Self { self.yr = (lo, hi); self }

        fn tx(&self, v: f64) -> f64 { if self.xlog { v.max(1e-300).ln() } else { v } }
        fn ty(&self, v: f64) -> f64 { if self.ylog { v.max(1e-300).ln() } else { v } }

        pub fn map(&self, x: f64, y: f64) -> (i32, i32) {
            let (a, b) = (self.tx(self.xr.0), self.tx(self.xr.1));
            let (c, d) = (self.ty(self.yr.0), self.ty(self.yr.1));
            let fx = ((self.tx(x) - a) / (b - a)).clamp(0.0, 1.0);
            let fy = ((self.ty(y) - c) / (d - c)).clamp(0.0, 1.0);
            (self.x + (fx * self.w as f64).round() as i32, self.y + self.h - (fy * self.h as f64).round() as i32)
        }

        fn ticks(range: (f64, f64), log: bool) -> Vec<f64> {
            if log {
                let (lo, hi) = (range.0.log10().floor() as i32, range.1.log10().ceil() as i32);
                (lo..=hi).map(|k| 10f64.powi(k))
                    .filter(|v| *v >= range.0 * 0.9999 && *v <= range.1 * 1.0001).collect()
            } else {
                (0..=4).map(|i| range.0 + (range.1 - range.0) * i as f64 / 4.0).collect()
            }
        }

        pub fn draw(&self, img: &mut RgbImage, series: &[Series]) {
            fill(img, self.x, self.y, self.w + 1, self.h + 1, PANEL);
            for v in Self::ticks(self.yr, self.ylog) {
                let (_, py) = self.map(self.xr.0, v);
                line(img, (self.x, py), (self.x + self.w, py), GRID);
                let label = fmt_num(v);
                text(img, &label, self.x - 6 - text_width(&label), py - 5, LABEL);
            }
            for v in Self::ticks(self.xr, self.xlog) {
                let (px, _) = self.map(v, self.yr.0);
                line(img, (px, self.y), (px, self.y + self.h), GRID);
                let label = fmt_num(v);
                text(img, &label, px - text_width(&label) / 2, self.y + self.h + 6, LABEL);
            }
            for (data, color) in series {
                let mut prev: Option<(i32, i32)> = None;
                for &(x, y) in data.iter() {
                    if !x.is_finite() || !y.is_finite() { prev = None; continue; }
                    let p = self.map(x, y);
                    if let Some(q) = prev { line(img, q, p, *color); } else { put(img, p.0, p.1, *color); }
                    prev = Some(p);
                }
            }
            line(img, (self.x, self.y), (self.x, self.y + self.h), BORDER);
            line(img, (self.x, self.y + self.h), (self.x + self.w, self.y + self.h), BORDER);
        }
    }

    /// Fraction of a color kept for de-emphasised "raw" series.
    pub fn dim(c: Rgb<u8>, f: f32) -> Rgb<u8> { Rgb(c.0.map(|v| (v as f32 * f) as u8)) }

    /// Draw a filled circular marker disk.
    pub fn draw_marker(img: &mut RgbImage, x: i32, y: i32, radius: i32, color: Rgb<u8>) {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= radius * radius + 1 {
                    put(img, x + dx, y + dy, color);
                }
            }
        }
    }

    /// Single-panel plot saved to `path`.
    pub fn save_plot(path: &std::path::Path, width: u32, height: u32, panel: Panel, series: &[Series]) -> anyhow::Result<()> {
        let mut img = RgbImage::from_pixel(width, height, BACKGROUND);
        panel.draw(&mut img, series);
        img.save(path)?;
        Ok(())
    }

    /// Inferno-like colormap on [0, 1].
    pub fn colormap(t: f32) -> [u8; 3] {
        const STOPS: [(f32, [f32; 3]); 5] = [
            (0.0, [0.0, 0.0, 4.0]), (0.25, [87.0, 16.0, 110.0]), (0.5, [188.0, 55.0, 84.0]),
            (0.75, [249.0, 142.0, 9.0]), (1.0, [252.0, 255.0, 164.0]),
        ];
        let t = t.clamp(0.0, 1.0);
        let i = STOPS.windows(2).position(|w| t <= w[1].0).unwrap_or(3);
        let (a, b) = (STOPS[i], STOPS[i + 1]);
        let f = (t - a.0) / (b.0 - a.0);
        [0, 1, 2].map(|k| (a.1[k] + (b.1[k] - a.1[k]) * f) as u8)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl TrainingState {
    /// Writes the training dashboard PNG (log-log loss with EMA over raw points,
    /// entropy, epoch average, validation runs; LR and top-3 on their own panels) plus CSVs.
    pub fn save_chart_image(&self, path: &str) -> anyhow::Result<()> {
        use self::plot::{Panel, dim, Series};
        use image::Rgb;
        let (width, height) = (1400u32, 1000u32);
        let mut img = image::RgbImage::from_pixel(width, height, plot::BACKGROUND);
        let (left, panel_w) = (100, width as i32 - 100 - 24);

        let smooth = ema(&self.loss_history, EMA_ALPHA);
        let cyan = Rgb([0, 255, 255]);
        let val_color = Rgb([255, 80, 80]);
        let mut loss_series: Vec<Series> = vec![
            (&self.loss_history, dim(cyan, 0.35)),
            (&self.entropy_history, Rgb([150, 0, 150])),
            (&self.avg_loss_history, Rgb([255, 220, 0])),
            (&smooth, cyan),
        ];
        if !self.val_loss_history.is_empty() {
            loss_series.push((&self.val_loss_history, val_color));
        }
        let loss_panel = Panel::new(left, 20, panel_w, 560, true, true).fit(&loss_series, false);
        loss_panel.draw(&mut img, &loss_series);
        for &(x, y) in &self.val_loss_history {
            if x.is_finite() && y.is_finite() && x > 0.0 && y > 0.0 {
                let p = loss_panel.map(x, y);
                plot::draw_marker(&mut img, p.0, p.1, 4, val_color);
            }
        }

        let lr_series: [Series; 1] = [(&self.lr_history, Rgb([0, 255, 0]))];
        Panel::new(left, 625, panel_w, 170, true, false).fit(&lr_series, true).draw(&mut img, &lr_series);

        let mut top3_series: Vec<Series> = vec![(&self.top3_history, Rgb([80, 140, 255]))];
        let val_top3_color = Rgb([255, 100, 70]);
        if !self.val_top3_history.is_empty() {
            top3_series.push((&self.val_top3_history, val_top3_color));
        }
        let top3_panel = Panel::new(left, 840, panel_w, 120, true, false).fit(&top3_series, true)
            .y_range(0.0, 1.0);
        top3_panel.draw(&mut img, &top3_series);
        for &(x, y) in &self.val_top3_history {
            if x.is_finite() && y.is_finite() && x > 0.0 {
                let p = top3_panel.map(x, y);
                plot::draw_marker(&mut img, p.0, p.1, 4, val_top3_color);
            }
        }

        img.save(path)?;

        use std::io::Write;
        let csv_path = std::path::Path::new(path).with_extension("csv");
        let mut csv = std::io::BufWriter::new(std::fs::File::create(csv_path)?);
        writeln!(csv, "step,loss,loss_ema,epoch_avg,lr,entropy,top3_accuracy")?;
        let get = |h: &[(f64, f64)], i: usize| h.get(i).map_or(String::new(), |p| p.1.to_string());
        for (i, (step, loss)) in self.loss_history.iter().enumerate() {
            writeln!(csv, "{step},{loss},{},{},{},{},{}", smooth[i].1, get(&self.avg_loss_history, i),
                get(&self.lr_history, i), get(&self.entropy_history, i), get(&self.top3_history, i))?;
        }
        csv.flush()?;
        if !self.val_loss_history.is_empty() {
            let csv_path = std::path::Path::new(path).with_extension("val.csv");
            let mut csv = std::io::BufWriter::new(std::fs::File::create(csv_path)?);
            writeln!(csv, "step,val_loss,val_top3")?;
            for (i, &(step, val_loss)) in self.val_loss_history.iter().enumerate() {
                let val_top3 = self.val_top3_history.get(i).map_or(String::new(), |p| p.1.to_string());
                writeln!(csv, "{step},{val_loss},{val_top3}")?;
            }
            csv.flush()?;
        }
        if !self.top3_history.is_empty() {
            let csv_path = std::path::Path::new(path).with_extension("top3.csv");
            let mut csv = std::io::BufWriter::new(std::fs::File::create(csv_path)?);
            writeln!(csv, "step,top3_accuracy")?;
            for (step, accuracy) in &self.top3_history {
                writeln!(csv, "{step},{accuracy}")?;
            }
            csv.flush()?;
        }
        Ok(())
    }
}
