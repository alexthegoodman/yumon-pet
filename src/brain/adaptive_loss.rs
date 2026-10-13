//! Adaptive learning-value token weighting for causal-LM training
//! (docs/ADAPTIVE_LEARNING.md). Training only: inference never touches it.
//!
//! Statistics are kept per key. A key is the target token id, or in `context`
//! mode the pair (occurrence category, token id). Per key there is a difficulty
//! EMA of detached cross-entropy, decayed occurrence counts and a short history
//! of difficulty snapshots for learning progress. Sparse keys shrink toward
//! their category's mean. Statistics accumulate on the CPU from the per-token
//! NLL the trainer already reads back, and update once per window of optimizer
//! steps. Weights used at a step always come from windows completed before it.
#[cfg(test)]
#[path = "../../tests/adaptive_loss/mod.rs"]
mod tests;
use std::{collections::{HashSet, VecDeque}, path::Path};
use anyhow::{Context as _, Result, ensure};
use serde::{Deserialize, Serialize};
use super::{PAD_TOKEN, bpe::TokenizerKind};

pub const STATE_FILE: &str = "adaptive_loss.bin";
pub const LOG_FILE: &str = "adaptive_loss.jsonl";
/// Key for PAD and out-of-range targets; never observed or weighted.
pub const NO_KEY: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdaptiveMode {
    /// Plain cross-entropy. Statistics are still collected and logged, so this is
    /// the baseline with the same diagnostics as the weighted runs.
    Disabled,
    Difficulty,
    #[default]
    DifficultyProgress,
    /// Inverse smoothed-frequency control: score 1 / (C_v + kappa).
    Frequency,
    /// difficulty_progress keyed by (occurrence context, token): the same token
    /// copied from earlier in the sequence, newly introduced, or inside a
    /// comment gets separate statistics.
    Context,
}

impl AdaptiveMode {
    pub fn tag(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Difficulty => "difficulty",
            Self::DifficultyProgress => "difficulty_progress",
            Self::Frequency => "frequency",
            Self::Context => "context",
        }
    }

    fn categories(self) -> usize {
        if self == Self::Context { CONTEXT_LABELS.len() } else { 1 }
    }
}

/// Optional `adaptive_loss` section of a code config. Defaults are the
/// experimental starting values from the design doc, not tuned hyperparameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AdaptiveLossConfig {
    pub enabled: bool,
    pub mode: AdaptiveMode,
    pub lambda: f64,
    pub difficulty_beta: f64,
    pub difficulty_alpha: f64,
    pub progress_eta: f64,
    pub weight_gamma: f64,
    pub min_weight: f64,
    pub max_weight: f64,
    pub confidence_kappa: f64,
    pub statistics_interval_steps: usize,
    pub progress_history_windows: usize,
    pub warmup_fraction: f64,
    pub ramp_fraction: f64,
    pub epsilon: f64,
    pub progress_floor: f64,
}

impl Default for AdaptiveLossConfig {
    fn default() -> Self {
        Self {
            enabled: true, mode: AdaptiveMode::DifficultyProgress, lambda: 0.3,
            difficulty_beta: 0.99, difficulty_alpha: 1.0, progress_eta: 0.5, weight_gamma: 0.5,
            min_weight: 0.5, max_weight: 2.0, confidence_kappa: 100.0,
            statistics_interval_steps: 100, progress_history_windows: 5,
            warmup_fraction: 0.10, ramp_fraction: 0.10, epsilon: 1e-6, progress_floor: 0.01,
        }
    }
}

impl AdaptiveLossConfig {
    pub fn validate(&self) -> Result<()> {
        let finite = [self.lambda, self.difficulty_beta, self.difficulty_alpha, self.progress_eta,
            self.weight_gamma, self.min_weight, self.max_weight, self.confidence_kappa,
            self.warmup_fraction, self.ramp_fraction, self.epsilon, self.progress_floor];
        ensure!(finite.iter().all(|v| v.is_finite()), "adaptive_loss values must be finite");
        ensure!((0.0..=1.0).contains(&self.lambda), "adaptive_loss.lambda must be in [0, 1]");
        ensure!(self.difficulty_beta > 0.0 && self.difficulty_beta < 1.0, "adaptive_loss.difficulty_beta must be in (0, 1)");
        ensure!(self.difficulty_alpha >= 0.0 && self.progress_eta >= 0.0 && self.weight_gamma >= 0.0,
            "adaptive_loss exponents must be non-negative");
        ensure!(self.min_weight > 0.0 && self.min_weight <= 1.0 && self.max_weight >= 1.0,
            "adaptive_loss requires 0 < min_weight <= 1 <= max_weight");
        ensure!(self.confidence_kappa >= 0.0, "adaptive_loss.confidence_kappa must be non-negative");
        ensure!(self.statistics_interval_steps > 0 && self.progress_history_windows > 0,
            "adaptive_loss interval and history must be positive");
        ensure!(self.warmup_fraction >= 0.0 && self.ramp_fraction >= 0.0
            && self.warmup_fraction + self.ramp_fraction <= 1.0, "adaptive_loss warm-up + ramp must be within [0, 1]");
        ensure!(self.epsilon > 0.0 && self.progress_floor > 0.0, "adaptive_loss epsilon and progress_floor must be positive");
        Ok(())
    }

    pub fn effective_mode(&self) -> AdaptiveMode {
        if self.enabled { self.mode } else { AdaptiveMode::Disabled }
    }

    /// lambda_effective(s): 0 during warm-up, then a linear ramp to `lambda`.
    pub fn lambda_at(&self, step: usize, total_steps: usize) -> f64 {
        if self.effective_mode() == AdaptiveMode::Disabled { return 0.0; }
        let warmup = self.warmup_fraction * total_steps as f64;
        let ramp = self.ramp_fraction * total_steps as f64;
        let s = step as f64;
        if s < warmup { 0.0 } else if ramp <= 0.0 { self.lambda } else { self.lambda * ((s - warmup) / ramp).min(1.0) }
    }
}

/// Occurrence categories for `context` mode and the per-context diagnostics.
/// Symbols are code tokens without letters or digits; a copy is a token id that
/// already occurred earlier in the same sequence.
pub const CONTEXT_LABELS: [&str; 5] = ["code_symbol", "code_novel", "code_copy", "comment_novel", "comment_copy"];
pub const NO_LABEL: u8 = u8::MAX;

/// Labels target positions by scanning decoded token text. Rust-aware enough for
/// line/block comments and string literals; raw strings and odd char literals
/// can mislabel a few tokens, which only shifts statistics, never targets.
pub struct ContextLabeler {
    text: Vec<String>,
    alphanumeric: Vec<bool>,
}

impl ContextLabeler {
    pub fn new(text: Vec<String>) -> Self {
        let alphanumeric = text.iter().map(|t| t.chars().any(char::is_alphanumeric)).collect();
        Self { text, alphanumeric }
    }

    pub fn from_tokenizer(tokenizer: &TokenizerKind, vocab: usize) -> Self {
        Self::new((0..vocab).map(|id| tokenizer.decode(&[id])).collect())
    }

    /// One label per target, rows of `seq_len`; `NO_LABEL` at PAD/invalid targets.
    pub fn label(&self, targets: &[i32], seq_len: usize) -> Vec<u8> {
        #[derive(Clone, Copy, PartialEq)]
        enum Scan { Code, Line, Block, Str { escaped: bool } }
        let mut labels = Vec::with_capacity(targets.len());
        for row in targets.chunks(seq_len) {
            let (mut scan, mut prev) = (Scan::Code, '\0');
            let mut seen = HashSet::new();
            for &t in row {
                if t == PAD_TOKEN as i32 || t < 0 || t as usize >= self.text.len() {
                    labels.push(NO_LABEL);
                    continue;
                }
                let copy = !seen.insert(t) as u8;
                let comment = matches!(scan, Scan::Line | Scan::Block);
                labels.push(if comment { 3 + copy } else if !self.alphanumeric[t as usize] { 0 } else { 1 + copy });
                for c in self.text[t as usize].chars() {
                    let next = match scan {
                        Scan::Code if prev == '/' && c == '/' => Scan::Line,
                        Scan::Code if prev == '/' && c == '*' => Scan::Block,
                        // `'"'` is a char literal, not the start of a string.
                        Scan::Code if c == '"' && prev != '\'' => Scan::Str { escaped: false },
                        Scan::Line if c == '\n' => Scan::Code,
                        Scan::Block if prev == '*' && c == '/' => Scan::Code,
                        Scan::Str { escaped: true } => Scan::Str { escaped: false },
                        Scan::Str { .. } if c == '\\' => Scan::Str { escaped: true },
                        Scan::Str { .. } if c == '"' => Scan::Code,
                        other => other,
                    };
                    // A char that opens or closes a construct never pairs with the next one
                    // (`/*/` stays open, `*//` closes once).
                    prev = if next != scan { '\0' }
                        else if next == Scan::Code && (c == '/' || c == '\'') || next == Scan::Block { c }
                        else { '\0' };
                    scan = next;
                }
            }
        }
        labels
    }
}

/// Training-frequency buckets for weight-vs-frequency and validation CE reports.
pub struct FrequencyBuckets {
    bucket: Vec<u8>,
}

impl FrequencyBuckets {
    pub const LABELS: [&'static str; 5] = ["<10", "10-99", "100-999", "1k-9.9k", ">=10k"];

    pub fn from_counts(counts: &[u64]) -> Self {
        Self { bucket: counts.iter().map(|&c| match c { 0..=9 => 0, 10..=99 => 1, 100..=999 => 2, 1000..=9999 => 3, _ => 4 }).collect() }
    }

    /// Supervised target counts over a sample set.
    pub fn count_targets(samples: &[super::samples::Sample], vocab: usize) -> Vec<u64> {
        let mut counts = vec![0u64; vocab];
        for s in samples {
            for &t in &s.target_labels {
                if t != PAD_TOKEN && t < vocab { counts[t] += 1; }
            }
        }
        counts
    }

    pub fn of(&self, token: usize) -> usize { self.bucket.get(token).copied().unwrap_or(0) as usize }
}

/// Unweighted loss sum, applied-weight sum and token count per group.
#[derive(Debug, Clone, Default)]
pub struct GroupLoss(pub [(f64, f64, u64); 5]);

impl GroupLoss {
    pub fn add(&mut self, group: usize, loss: f32, weight: f32) {
        if !loss.is_finite() { return; }
        let g = &mut self.0[group];
        g.0 += loss as f64;
        g.1 += weight as f64;
        g.2 += 1;
    }

    pub fn add_by_frequency(&mut self, buckets: &FrequencyBuckets, targets: &[i32], nll: &[f32]) {
        for (&t, &l) in targets.iter().zip(nll) {
            if t != PAD_TOKEN as i32 && t >= 0 { self.add(buckets.of(t as usize), l, 1.0); }
        }
    }

    pub fn add_by_label(&mut self, labels: &[u8], nll: &[f32], weights: Option<&[f32]>) {
        for (i, (&label, &l)) in labels.iter().zip(nll).enumerate() {
            if label != NO_LABEL { self.add(label as usize, l, weights.map_or(1.0, |w| w[i])); }
        }
    }

    /// `{label: {ce, tokens[, mean_weight]}}`.
    pub fn to_json(&self, labels: &[&str; 5], with_weight: bool) -> serde_json::Value {
        serde_json::Value::Object(labels.iter().zip(&self.0).map(|(label, &(sum, weight, n))| {
            let mean = |v: f64| (n > 0).then(|| v / n as f64);
            let mut entry = serde_json::json!({"ce": mean(sum), "tokens": n});
            if with_weight { entry["mean_weight"] = serde_json::json!(mean(weight)); }
            (label.to_string(), entry)
        }).collect())
    }
}

/// Registry state; everything needed to resume weighting is serialized.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLoss {
    pub config: AdaptiveLossConfig,
    vocab: usize,
    /// 1, or `CONTEXT_LABELS.len()` in context mode; key = category * vocab + token.
    categories: usize,
    /// Completed statistics windows.
    pub windows: u64,
    window_steps: usize,
    pending_sum: Vec<f64>,
    pending_count: Vec<u32>,
    /// D per key; NaN until first observed.
    difficulty: Vec<f32>,
    /// C per key, decayed by `difficulty_beta` each window (bounded history).
    counts: Vec<f32>,
    /// Per-category EMA of the mean window loss (D_global per category); NaN before data.
    category_difficulty: Vec<f32>,
    /// Difficulty snapshots, oldest first; at most `progress_history_windows + 1`.
    history: VecDeque<Vec<f32>>,
    weights: Vec<f32>,
}

/// Distribution summary written once per completed window.
#[derive(Debug, Clone, Serialize)]
pub struct WindowReport {
    pub window: u64,
    pub observed_keys: usize,
    pub category_difficulty: Vec<f32>,
    pub difficulty_p50: f32,
    pub difficulty_p95: f32,
    /// Count-weighted mean of max(P, 0) and the fraction of observed keys with P > 0.
    pub progress_mean: f32,
    pub progress_positive_fraction: f32,
    /// Raw weight w over observed keys (before lambda mixing).
    pub weight_p05: f32,
    pub weight_p50: f32,
    pub weight_p95: f32,
    pub weight_at_min_fraction: f32,
    pub weight_at_max_fraction: f32,
    /// Occurrence-weighted mean w per training-frequency bucket.
    pub weight_by_frequency: serde_json::Value,
}

impl AdaptiveLoss {
    pub fn new(config: AdaptiveLossConfig, vocab: usize) -> Self {
        let categories = config.effective_mode().categories();
        let keys = vocab * categories;
        Self {
            config, vocab, categories, windows: 0, window_steps: 0,
            pending_sum: vec![0.0; keys], pending_count: vec![0; keys],
            difficulty: vec![f32::NAN; keys], counts: vec![0.0; keys],
            category_difficulty: vec![f32::NAN; categories], history: VecDeque::new(), weights: vec![1.0; keys],
        }
    }

    /// Restores `adaptive_loss.bin` from a run directory, or starts fresh (also
    /// for checkpoints written before adaptive loss or this state format existed).
    pub fn load_or_new(dir: &Path, config: &AdaptiveLossConfig, vocab: usize) -> Result<Self> {
        let path = dir.join(STATE_FILE);
        if !path.exists() { return Ok(Self::new(config.clone(), vocab)); }
        let state: Self = match bincode::deserialize(&std::fs::read(&path)?) {
            Ok(state) => state,
            Err(e) => {
                eprintln!("⚠️  {} uses an older format ({e}); starting adaptive statistics fresh.", path.display());
                return Ok(Self::new(config.clone(), vocab));
            }
        };
        ensure!(state.vocab == vocab, "adaptive_loss state vocabulary differs from the tokenizer");
        ensure!(&state.config == config, "adaptive_loss config differs from the checkpoint's; use a new out_dir");
        Ok(state)
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        let tmp = dir.join(format!("{STATE_FILE}.tmp"));
        std::fs::write(&tmp, bincode::serialize(self)?).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, dir.join(STATE_FILE))?;
        Ok(())
    }

    pub fn weights(&self) -> &[f32] { &self.weights }
    pub fn uses_context(&self) -> bool { self.categories > 1 }

    /// Statistics key per target. Context mode needs the labeler's labels.
    pub fn keys(&self, targets: &[i32], labels: Option<&[u8]>) -> Vec<u32> {
        assert!(!self.uses_context() || labels.is_some(), "context mode requires context labels");
        targets.iter().enumerate().map(|(i, &t)| {
            if t == PAD_TOKEN as i32 || t < 0 || t as usize >= self.vocab { return NO_KEY; }
            if !self.uses_context() { return t as u32; }
            match labels.unwrap()[i] {
                NO_LABEL => NO_KEY,
                label => (label as usize * self.vocab + t as usize) as u32,
            }
        }).collect()
    }

    /// True when the next `observe` closes a window (the trainer measures the
    /// gradient norm on that step only, to avoid a readback every step).
    pub fn closes_window_next_step(&self) -> bool {
        self.window_steps + 1 >= self.config.statistics_interval_steps
    }

    /// Per-row a = (1 - lambda) + lambda * w[key], and 0 for `NO_KEY`.
    pub fn row_weights(&self, keys: &[u32], lambda: f64) -> Vec<f32> {
        keys.iter().map(|&k| {
            if k == NO_KEY { return 0.0; }
            ((1.0 - lambda) + lambda * self.weights[k as usize] as f64) as f32
        }).collect()
    }

    /// Adds one step's detached per-token losses. Returns true when this step
    /// closed a statistics window and the weights were recomputed.
    pub fn observe(&mut self, keys: &[u32], nll: &[f32]) -> bool {
        for (&k, &l) in keys.iter().zip(nll) {
            if k == NO_KEY || !l.is_finite() { continue; }
            self.pending_sum[k as usize] += l as f64;
            self.pending_count[k as usize] += 1;
        }
        self.window_steps += 1;
        if self.window_steps < self.config.statistics_interval_steps { return false; }
        self.close_window();
        true
    }

    fn category(&self, key: usize) -> usize { key / self.vocab }

    fn close_window(&mut self) {
        let beta = self.config.difficulty_beta as f32;
        let mut by_category = vec![(0.0f64, 0u64); self.categories];
        for k in 0..self.difficulty.len() {
            let count = self.pending_count[k];
            self.counts[k] = beta * self.counts[k] + count as f32;
            if count == 0 { continue; }
            let mean = (self.pending_sum[k] / count as f64) as f32;
            let d = &mut self.difficulty[k];
            *d = if d.is_nan() { mean } else { beta * *d + (1.0 - beta) * mean };
            let c = &mut by_category[k / self.vocab];
            c.0 += self.pending_sum[k];
            c.1 += count as u64;
        }
        for (g, (sum, n)) in self.category_difficulty.iter_mut().zip(by_category) {
            if n == 0 { continue; }
            let mean = (sum / n as f64) as f32;
            *g = if g.is_nan() { mean } else { beta * *g + (1.0 - beta) * mean };
        }
        self.history.push_back(self.difficulty.clone());
        while self.history.len() > self.config.progress_history_windows + 1 { self.history.pop_front(); }
        self.pending_sum.fill(0.0);
        self.pending_count.fill(0);
        self.window_steps = 0;
        self.windows += 1;
        self.recompute_weights();
    }

    /// R = C / (C + kappa).
    fn confidence(&self, k: usize) -> f32 {
        let c = self.counts[k];
        let denom = c + self.config.confidence_kappa as f32;
        if denom > 0.0 { c / denom } else { 0.0 }
    }

    /// P = D(k - h) - D(k) against the oldest retained snapshot; 0 when either is unset.
    pub fn progress(&self, k: usize) -> f32 {
        let old = self.history.front().map_or(f32::NAN, |h| h[k]);
        let p = old - self.difficulty[k];
        if p.is_finite() { p } else { 0.0 }
    }

    /// D~ = R D + (1 - R) D_category.
    pub fn smoothed_difficulty(&self, k: usize) -> f32 {
        let prior = self.category_difficulty[self.category(k)];
        let prior = if prior.is_nan() { 0.0 } else { prior };
        let d = self.difficulty[k];
        if d.is_nan() { return prior; }
        let r = self.confidence(k);
        r * d + (1.0 - r) * prior
    }

    fn recompute_weights(&mut self) {
        let cfg = &self.config;
        let mode = match cfg.effective_mode() {
            // The disabled baseline logs what difficulty_progress would have applied.
            AdaptiveMode::Disabled => AdaptiveMode::DifficultyProgress,
            m => m,
        };
        let total_count: f64 = self.counts.iter().map(|&c| c as f64).sum();
        if total_count <= 0.0 { self.weights.fill(1.0); return; }
        // Unreliable progress shrinks toward its category's count-weighted positive
        // progress, the same way difficulty shrinks toward the category mean.
        let mut category_progress = vec![(0.0f64, 0.0f64); self.categories];
        for k in 0..self.counts.len() {
            let c = &mut category_progress[k / self.vocab];
            c.0 += self.counts[k] as f64 * self.progress(k).max(0.0) as f64;
            c.1 += self.counts[k] as f64;
        }
        let category_progress: Vec<f64> = category_progress.iter().map(|&(p, c)| if c > 0.0 { p / c } else { 0.0 }).collect();
        let (eps, kappa) = (cfg.epsilon, cfg.confidence_kappa);
        let scores: Vec<f64> = (0..self.counts.len()).map(|k| {
            let difficulty = (self.smoothed_difficulty(k) as f64 + eps).powf(cfg.difficulty_alpha);
            match mode {
                AdaptiveMode::Frequency => 1.0 / (self.counts[k] as f64 + kappa + eps),
                AdaptiveMode::Difficulty => difficulty,
                _ => {
                    let r = self.confidence(k) as f64;
                    let progress = r * self.progress(k).max(0.0) as f64 + (1.0 - r) * category_progress[self.category(k)];
                    difficulty * (progress + cfg.progress_floor).powf(cfg.progress_eta)
                }
            }
        }).collect();
        // S_ref: occurrence-weighted mean score over all keys, so typical training
        // tokens sit near w = 1 and categories are compared on one scale.
        let reference = (0..scores.len()).map(|k| self.counts[k] as f64 * scores[k]).sum::<f64>() / total_count;
        for k in 0..scores.len() {
            let w = (scores[k] / (reference + eps)).powf(cfg.weight_gamma).clamp(cfg.min_weight, cfg.max_weight);
            self.weights[k] = if w.is_finite() { w as f32 } else { 1.0 };
        }
    }

    pub fn report(&self, buckets: &FrequencyBuckets) -> WindowReport {
        let observed: Vec<usize> = (0..self.counts.len()).filter(|&k| self.counts[k] > 0.0).collect();
        let quantile = |mut values: Vec<f32>, q: f32| -> f32 {
            if values.is_empty() { return f32::NAN; }
            values.sort_by(|a, b| a.total_cmp(b));
            values[((values.len() - 1) as f32 * q).round() as usize]
        };
        let difficulty: Vec<f32> = observed.iter().map(|&k| self.difficulty[k]).filter(|d| !d.is_nan()).collect();
        let weights: Vec<f32> = observed.iter().map(|&k| self.weights[k]).collect();
        let total_count: f64 = observed.iter().map(|&k| self.counts[k] as f64).sum();
        let progress_mean = observed.iter().map(|&k| self.counts[k] as f64 * self.progress(k).max(0.0) as f64)
            .sum::<f64>() / total_count.max(f64::MIN_POSITIVE);
        let n = observed.len().max(1) as f32;
        let (lo, hi) = (self.config.min_weight as f32, self.config.max_weight as f32);
        let mut by_bucket = [(0.0f64, 0.0f64); 5];
        for &k in &observed {
            let b = &mut by_bucket[buckets.of(k % self.vocab)];
            b.0 += self.counts[k] as f64 * self.weights[k] as f64;
            b.1 += self.counts[k] as f64;
        }
        WindowReport {
            window: self.windows,
            observed_keys: observed.len(),
            category_difficulty: self.category_difficulty.clone(),
            difficulty_p50: quantile(difficulty.clone(), 0.5),
            difficulty_p95: quantile(difficulty, 0.95),
            progress_mean: progress_mean as f32,
            progress_positive_fraction: observed.iter().filter(|&&k| self.progress(k) > 0.0).count() as f32 / n,
            weight_p05: quantile(weights.clone(), 0.05),
            weight_p50: quantile(weights.clone(), 0.5),
            weight_p95: quantile(weights.clone(), 0.95),
            weight_at_min_fraction: weights.iter().filter(|&&w| w <= lo).count() as f32 / n,
            weight_at_max_fraction: weights.iter().filter(|&&w| w >= hi).count() as f32 / n,
            weight_by_frequency: serde_json::Value::Object(FrequencyBuckets::LABELS.iter().zip(by_bucket)
                .map(|(label, (sum, c))| (label.to_string(), serde_json::json!((c > 0.0).then(|| sum / c)))).collect()),
        }
    }
}

/// Appends one JSON line to the run's adaptive diagnostics log.
pub fn append_log(dir: &Path, line: &serde_json::Value) -> Result<()> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(dir.join(LOG_FILE))?;
    writeln!(file, "{line}")?;
    Ok(())
}
