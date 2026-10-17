//! Per-token loss diagnostics for training: where the loss sits, not just how big it is.
//!
//! - batch heatmap: sample (row) x position (column), colored by token loss
//! - loss by position in the sequence, averaged over a window of batches
//! - loss by token id, sorted by frequency, with cumulative share of total loss
//! - an HTML view that colors real held-out text token by token
use super::{
    PAD_TOKEN,
    chart::plot::{self, Panel, colormap, dim, fmt_num},
};
use anyhow::Result;
use image::{Rgb, RgbImage};
use std::{collections::HashMap, fmt::Write as _, io::Write as _, path::Path};

/// Loss accumulators over a window of batches (cleared by `reset` after each snapshot).
pub struct TokenLossStats {
    seq_len: usize,
    pos_sum: Vec<f64>,
    pos_count: Vec<u64>,
    tok_sum: Vec<f64>,
    tok_count: Vec<u64>,
}

impl TokenLossStats {
    pub fn new(seq_len: usize, vocab: usize) -> Self {
        Self {
            seq_len,
            pos_sum: vec![0.0; seq_len],
            pos_count: vec![0; seq_len],
            tok_sum: vec![0.0; vocab],
            tok_count: vec![0; vocab],
        }
    }

    pub fn reset(&mut self) {
        self.pos_sum.fill(0.0);
        self.pos_count.fill(0);
        self.tok_sum.fill(0.0);
        self.tok_count.fill(0);
    }

    /// `targets` and `losses` are flat [batch * seq_len] for a batch trimmed to
    /// `seq_len` (at most the context); PAD targets are skipped.
    pub fn add(&mut self, targets: &[i32], losses: &[f32], seq_len: usize) {
        assert!(seq_len <= self.seq_len, "batch length exceeds the context");
        for (i, (&target, &loss)) in targets.iter().zip(losses).enumerate() {
            if target == PAD_TOKEN as i32 || !loss.is_finite() { continue; }
            let pos = i % seq_len;
            self.pos_sum[pos] += loss as f64;
            self.pos_count[pos] += 1;
            if let Some(sum) = self.tok_sum.get_mut(target as usize) {
                *sum += loss as f64;
                self.tok_count[target as usize] += 1;
            }
        }
    }

    /// (position + 1, mean loss) for positions that were supervised.
    pub fn position_curve(&self) -> Vec<(f64, f64)> {
        (0..self.seq_len).filter(|&p| self.pos_count[p] > 0)
            .map(|p| (p as f64 + 1.0, self.pos_sum[p] / self.pos_count[p] as f64)).collect()
    }

    /// Token ids by descending frequency: (id, count, mean loss, share of total loss).
    pub fn token_table(&self) -> Vec<(usize, u64, f64, f64)> {
        let total: f64 = self.tok_sum.iter().sum::<f64>().max(1e-12);
        let mut ids: Vec<usize> = (0..self.tok_count.len()).filter(|&t| self.tok_count[t] > 0).collect();
        ids.sort_by(|&a, &b| self.tok_count[b].cmp(&self.tok_count[a]).then(a.cmp(&b)));
        ids.into_iter().map(|t| (t, self.tok_count[t], self.tok_sum[t] / self.tok_count[t] as f64, self.tok_sum[t] / total)).collect()
    }

    /// Writes position_loss_<step>.{png,csv} and token_loss_<step>.{png,csv} into `dir`.
    pub fn save(&self, dir: &Path, step: usize) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let position = self.position_curve();
        let cyan = Rgb([0, 255, 255]);
        if !position.is_empty() {
            let series = [(position.as_slice(), cyan)];
            let panel = Panel::new(100, 20, 880, 440, true, true).fit(&series, false);
            plot::save_plot(&dir.join(format!("position_loss_{step:06}.png")), 1000, 500, panel, &series)?;
            let mut csv = std::io::BufWriter::new(std::fs::File::create(dir.join(format!("position_loss_{step:06}.csv")))?);
            writeln!(csv, "position,mean_loss,tokens")?;
            for p in 0..self.seq_len {
                if self.pos_count[p] > 0 {
                    writeln!(csv, "{p},{},{}", self.pos_sum[p] / self.pos_count[p] as f64, self.pos_count[p])?;
                }
            }
            csv.flush()?;
        }

        let table = self.token_table();
        if !table.is_empty() {
            let mean: Vec<(f64, f64)> = table.iter().enumerate().map(|(r, t)| (r as f64 + 1.0, t.2)).collect();
            let mut running = 0.0;
            let share: Vec<(f64, f64)> = table.iter().enumerate().map(|(r, t)| { running += t.3; (r as f64 + 1.0, running) }).collect();
            let mut img = RgbImage::from_pixel(1000, 820, plot::BACKGROUND);
            let mean_series = [(mean.as_slice(), dim(cyan, 0.8))];
            Panel::new(100, 20, 880, 380, true, true).fit(&mean_series, false).draw(&mut img, &mean_series);
            let share_series = [(share.as_slice(), Rgb([255, 220, 0]))];
            Panel::new(100, 450, 880, 330, true, false).fit(&share_series, true).y_range(0.0, 1.0).draw(&mut img, &share_series);
            img.save(dir.join(format!("token_loss_{step:06}.png")))?;
            let mut csv = std::io::BufWriter::new(std::fs::File::create(dir.join(format!("token_loss_{step:06}.csv")))?);
            writeln!(csv, "rank,token_id,count,mean_loss,share_of_total_loss")?;
            for (rank, (id, count, mean, share)) in table.iter().enumerate() {
                writeln!(csv, "{},{id},{count},{mean},{share}", rank + 1)?;
            }
            csv.flush()?;
        }
        Ok(())
    }
}

/// Per-token loss of one batch as an image: one row per sample, one column per
/// position, color = sqrt(loss / ceiling) on an inferno-like map (PAD = dark gray).
/// Pass ln(vocab) as the ceiling: that is the loss of a uniform guess.
pub fn save_batch_heatmap(path: &Path, targets: &[i32], losses: &[f32], rows: usize, seq_len: usize, ceiling: f32) -> Result<()> {
    let cell_w = (1024 / seq_len).max(1) as i32;
    let cell_h = (480 / rows.max(1)).clamp(4, 24) as i32;
    let (left, top, bar_gap, bar_w, right) = (10, 10, 16, 14, 70);
    let (mw, mh) = (seq_len as i32 * cell_w, rows as i32 * cell_h);
    let mut img = RgbImage::from_pixel((left + mw + bar_gap + bar_w + right) as u32, (top + mh + 26) as u32, plot::BACKGROUND);
    let color = |loss: f32| Rgb(colormap((loss / ceiling).max(0.0).sqrt()));
    for r in 0..rows {
        for p in 0..seq_len {
            let i = r * seq_len + p;
            let c = if targets[i] == PAD_TOKEN as i32 { Rgb([28, 28, 34]) } else { color(losses[i]) };
            plot::fill(&mut img, left + p as i32 * cell_w, top + r as i32 * cell_h, cell_w, cell_h, c);
        }
    }
    let mut p = 0;
    while p < seq_len {
        let x = left + p as i32 * cell_w;
        plot::line(&mut img, (x, top + mh), (x, top + mh + 4), Rgb([175, 175, 185]));
        plot::text(&mut img, &p.to_string(), x, top + mh + 8, Rgb([175, 175, 185]));
        p += 64;
    }
    let bar_x = left + mw + bar_gap;
    for y in 0..mh {
        let t = 1.0 - y as f32 / (mh - 1).max(1) as f32;
        plot::fill(&mut img, bar_x, top + y, bar_w, 1, Rgb(colormap(t)));
    }
    for v in 0..=(ceiling.floor() as i32) {
        let y = top + mh - 1 - ((v as f32 / ceiling).sqrt() * (mh - 1) as f32) as i32;
        plot::line(&mut img, (bar_x + bar_w, y), (bar_x + bar_w + 4, y), Rgb([175, 175, 185]));
        plot::text(&mut img, &v.to_string(), bar_x + bar_w + 8, y - 5, Rgb([175, 175, 185]));
    }
    img.save(path)?;
    Ok(())
}

fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '\n' => out.push_str("↵\n"),
            c => out.push(c),
        }
    }
    out
}

/// One predicted token in the held-out view: its text, the loss of predicting
/// it, and (with adaptive depth) the deepest tier index spent on it.
pub struct TokenView {
    pub text: String,
    pub loss: f32,
    pub tier: Option<u8>,
}

/// Tier background colors: green (shallowest) through amber to red (deepest).
fn tier_color(tier: usize, tiers: usize) -> [u8; 3] {
    const PALETTE: [[u8; 3]; 5] = [[46, 139, 87], [212, 160, 23], [192, 57, 43], [52, 101, 164], [142, 68, 173]];
    let slot = if tiers <= 3 { tier * 2 / (tiers - 1).max(1) } else { tier % PALETTE.len() };
    PALETTE[slot.min(PALETTE.len() - 1)]
}

fn span(html: &mut String, [r, g, b]: [u8; 3], tip: &str, text: &str) {
    let fg = if (r as u32 * 3 + g as u32 * 6 + b as u32) > 1400 { "#000" } else { "#fff" };
    let _ = write!(html, "<span style=\"background:rgb({r},{g},{b});color:{fg}\" title=\"{tip}\">{}</span>", escape(text));
}

/// Colors held-out text token by token with the loss of predicting that token.
/// With `tier_names` (adaptive depth), each sample also gets a view colored by
/// the deepest tier reached, plus per-tier summaries. Hover for exact values.
pub fn write_token_html(path: &Path, title: &str, rows: &[Vec<TokenView>], tier_names: Option<&[String]>, ceiling: f32) -> Result<()> {
    let mut html = String::new();
    let _ = write!(html, "<!doctype html><meta charset=utf-8><title>{}</title><body style=\"background:#0a0a0f;color:#ddd;font:13px/1.5 monospace\">\
        <h3>{}</h3><p>Color = sqrt(loss / {:.2}); black is ~0 loss, yellow is a uniform-guess loss. Hover for exact values.</p>",
        escape(title), escape(title), ceiling);
    let tiers = tier_names.map_or(0, |n| n.len());
    let tier_name = |t: Option<u8>| t.and_then(|t| tier_names?.get(t as usize)).map_or("", |s| s.as_str());
    let tip = |t: &TokenView| match t.tier {
        Some(_) => format!("loss {:.4}, tier {}", t.loss, escape(tier_name(t.tier))),
        None => format!("loss {:.4}", t.loss),
    };
    // Per token text: loss sum, count, and count per deepest tier.
    let mut by_token: HashMap<&str, (f64, u32, Vec<u32>)> = HashMap::new();
    let mut by_tier = vec![(0.0f64, 0usize); tiers];
    for t in rows.iter().flatten() {
        let key = if t.text.trim().is_empty() { t.text.as_str() } else { t.text.trim() };
        let e = by_token.entry(key).or_insert_with(|| (0.0, 0, vec![0; tiers]));
        e.0 += t.loss as f64;
        e.1 += 1;
        if let Some(slot) = t.tier.and_then(|tier| e.2.get_mut(tier as usize)) { *slot += 1; }
        if let Some(s) = t.tier.and_then(|tier| by_tier.get_mut(tier as usize)) { s.0 += t.loss as f64; s.1 += 1; }
    }
    let (sum, n) = rows.iter().flatten().fold((0.0f64, 0usize), |a, t| (a.0 + t.loss as f64, a.1 + 1));
    let _ = write!(html, "<p>{n} tokens, mean loss {:.4}</p>", sum / n.max(1) as f64);
    if let Some(names) = tier_names {
        html.push_str("<h4>adaptive depth: deepest tier reached per predicted token</h4><table><tr><td>tier</td><td>tokens</td><td>share</td><td>mean loss</td></tr>");
        for (i, (name, (s, c))) in names.iter().zip(&by_tier).enumerate() {
            html.push_str("<tr><td>");
            span(&mut html, tier_color(i, tiers), name, name);
            let _ = write!(html, "</td><td>{c}</td><td>{:.1}%</td><td>{}</td></tr>",
                100.0 * *c as f64 / n.max(1) as f64, if *c > 0 { fmt_num(s / *c as f64) } else { "-".into() });
        }
        html.push_str("</table>");
    }
    for (i, row) in rows.iter().enumerate() {
        let mean = row.iter().map(|t| t.loss as f64).sum::<f64>() / row.len().max(1) as f64;
        let _ = write!(html, "<h4>sample {i} (mean loss {mean:.4})</h4><pre style=\"white-space:pre-wrap;word-break:break-all\">");
        for t in row {
            span(&mut html, colormap((t.loss / ceiling).max(0.0).sqrt()), &tip(t), &t.text);
        }
        html.push_str("</pre>");
        if tiers > 0 {
            let _ = write!(html, "<h4>sample {i} depth</h4><pre style=\"white-space:pre-wrap;word-break:break-all\">");
            for t in row {
                span(&mut html, tier_color(t.tier.unwrap_or(0) as usize, tiers), &tip(t), &t.text);
            }
            html.push_str("</pre>");
        }
    }
    let tier_mix = |counts: &[u32]| -> String {
        tier_names.unwrap_or(&[]).iter().zip(counts).filter(|(_, c)| **c > 0)
            .map(|(name, c)| format!("{} {c}", escape(name))).collect::<Vec<_>>().join(" / ")
    };
    let mut worst: Vec<(&str, f64, u32, &[u32])> = by_token.iter().map(|(t, (s, c, k))| (*t, s / *c as f64, *c, k.as_slice())).collect();
    worst.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(b.0)));
    html.push_str("<h4>highest mean loss tokens</h4><table>");
    for (text, mean, count, k) in worst.iter().take(20) {
        let _ = write!(html, "<tr><td><pre style=\"margin:0\">{}</pre></td><td>{}</td><td>x{count}</td><td>{}</td></tr>", escape(text), fmt_num(*mean), tier_mix(k));
    }
    html.push_str("</table>");
    if let Some(deepest) = tier_names.and_then(|n| n.last()).filter(|_| tiers > 1) {
        // Ranked by how often the token went all the way down.
        let mut deep: Vec<_> = worst.iter().filter(|w| w.3[tiers - 1] > 0).collect();
        deep.sort_by(|a, b| b.3[tiers - 1].cmp(&a.3[tiers - 1]).then(a.0.cmp(b.0)));
        let _ = write!(html, "<h4>tokens most often sent to {}</h4><table>", escape(deepest));
        for (text, mean, count, k) in deep.iter().take(20) {
            let _ = write!(html, "<tr><td><pre style=\"margin:0\">{}</pre></td><td>{}/{count}</td><td>loss {}</td><td>{}</td></tr>",
                escape(text), k[tiers - 1], fmt_num(*mean), tier_mix(k));
        }
        html.push_str("</table>");
    }
    std::fs::write(path, html)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_by_position_and_token_and_skips_pad() {
        let mut stats = TokenLossStats::new(3, 8);
        // two rows of three positions; the last position of row 2 is PAD
        stats.add(&[4, 5, 4, 4, 5, 0], &[2.0, 1.0, 3.0, 4.0, 1.0, 9.0], 3);
        assert_eq!(stats.position_curve(), vec![(1.0, 3.0), (2.0, 1.0), (3.0, 3.0)]);
        let table = stats.token_table();
        assert_eq!((table[0].0, table[0].1), (4, 3));
        assert!((table[0].2 - 3.0).abs() < 1e-9);
        assert!((table.iter().map(|t| t.3).sum::<f64>() - 1.0).abs() < 1e-9);
        stats.reset();
        assert!(stats.position_curve().is_empty());
    }

    #[test]
    fn writes_heatmap_plots_and_html() {
        let dir = std::env::temp_dir().join("yumon-loss-stats-test");
        std::fs::create_dir_all(&dir).unwrap();
        let (targets, losses): (Vec<i32>, Vec<f32>) = (0..2 * 130).map(|i| ((i % 7) as i32, (i % 5) as f32)).unzip();
        save_batch_heatmap(&dir.join("h.png"), &targets, &losses, 2, 130, 9.0).unwrap();
        let mut stats = TokenLossStats::new(130, 16);
        stats.add(&targets, &losses, 130);
        stats.save(&dir, 7).unwrap();
        for f in ["h.png", "position_loss_000007.png", "token_loss_000007.csv"] {
            assert!(dir.join(f).exists(), "{f}");
        }
        let view = |text: &str, loss, tier| TokenView { text: text.into(), loss, tier };
        write_token_html(&dir.join("t.html"), "t", &[vec![view("a<b", 1.0, None), view("\n", 0.0, None)]], None, 9.0).unwrap();
        let html = std::fs::read_to_string(dir.join("t.html")).unwrap();
        assert!(html.contains("a&lt;b") && !html.contains("depth"));
        let names = ["easy".to_string(), "medium".into(), "hard".into()];
        let rows = [vec![view("fn", 0.5, Some(0)), view("foo", 6.0, Some(2)), view("foo", 4.0, Some(1))]];
        write_token_html(&dir.join("d.html"), "d", &rows, Some(&names), 9.0).unwrap();
        let html = std::fs::read_to_string(dir.join("d.html")).unwrap();
        for s in ["sample 0 depth", "loss 6.0000, tier hard", "tokens most often sent to hard", "medium 1 / hard 1"] {
            assert!(html.contains(s), "{s}");
        }
    }
}
