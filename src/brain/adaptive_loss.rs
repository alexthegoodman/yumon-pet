//! Adaptive learning-value token weighting for causal-LM training
//! (docs/ADAPTIVE_LEARNING.md). Training only: inference never touches it.
//!
//! Per vocabulary token it keeps a difficulty EMA of detached cross-entropy,
//! decayed occurrence counts and a short history of difficulty snapshots for
//! learning progress. Statistics accumulate on the CPU from the per-token NLL the
//! trainer already reads back, and update once per window of optimizer steps.
//! Weights used at a step always come from windows completed before it.
#[cfg(test)]
#[path = "../../tests/adaptive_loss/mod.rs"]
mod tests;
use std::{collections::VecDeque, path::Path};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use super::PAD_TOKEN;

pub const STATE_FILE: &str = "adaptive_loss.bin";
pub const LOG_FILE: &str = "adaptive_loss.jsonl";

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
}

impl AdaptiveMode {
    pub fn tag(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Difficulty => "difficulty",
            Self::DifficultyProgress => "difficulty_progress",
            Self::Frequency => "frequency",
        }
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

/// (loss sum, token count) per frequency bucket.
#[derive(Debug, Clone, Default)]
pub struct BucketLoss(pub [(f64, u64); 5]);

impl BucketLoss {
    pub fn add(&mut self, buckets: &FrequencyBuckets, targets: &[i32], nll: &[f32]) {
        for (&t, &l) in targets.iter().zip(nll) {
            if t == PAD_TOKEN as i32 || !l.is_finite() { continue; }
            let b = &mut self.0[buckets.of(t as usize)];
            b.0 += l as f64;
            b.1 += 1;
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::Value::Object(FrequencyBuckets::LABELS.iter().zip(&self.0).map(|(label, &(sum, n))| {
            (label.to_string(), serde_json::json!({"ce": (n > 0).then(|| sum / n as f64), "tokens": n}))
        }).collect())
    }
}

/// Registry state; everything needed to resume weighting is serialized.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLoss {
    pub config: AdaptiveLossConfig,
    vocab: usize,
    /// Completed statistics windows.
    pub windows: u64,
    window_steps: usize,
    pending_sum: Vec<f64>,
    pending_count: Vec<u32>,
    /// D_v; NaN until the token is first observed.
    difficulty: Vec<f32>,
    /// C_v, decayed by `difficulty_beta` each window (bounded history).
    counts: Vec<f32>,
    /// D_global, an EMA of the mean window loss; NaN before the first window.
    global_difficulty: f32,
    /// Difficulty snapshots, oldest first; at most `progress_history_windows + 1`.
    history: VecDeque<Vec<f32>>,
    weights: Vec<f32>,
}

/// Distribution summary written once per completed window.
#[derive(Debug, Clone, Serialize)]
pub struct WindowReport {
    pub window: u64,
    pub observed_tokens: usize,
    pub global_difficulty: f32,
    pub difficulty_p50: f32,
    pub difficulty_p95: f32,
    /// Count-weighted mean of max(P_v, 0) and the fraction of observed ids with P_v > 0.
    pub progress_mean: f32,
    pub progress_positive_fraction: f32,
    /// Raw weight w_v over observed ids (before lambda mixing).
    pub weight_p05: f32,
    pub weight_p50: f32,
    pub weight_p95: f32,
    pub weight_at_min_fraction: f32,
    pub weight_at_max_fraction: f32,
    /// Occurrence-weighted mean w_v per training-frequency bucket.
    pub weight_by_frequency: serde_json::Value,
}

impl AdaptiveLoss {
    pub fn new(config: AdaptiveLossConfig, vocab: usize) -> Self {
        Self {
            config, vocab, windows: 0, window_steps: 0,
            pending_sum: vec![0.0; vocab], pending_count: vec![0; vocab],
            difficulty: vec![f32::NAN; vocab], counts: vec![0.0; vocab],
            global_difficulty: f32::NAN, history: VecDeque::new(), weights: vec![1.0; vocab],
        }
    }

    /// Restores `adaptive_loss.bin` from a run directory, or starts fresh (also
    /// for checkpoints written before adaptive loss existed).
    pub fn load_or_new(dir: &Path, config: &AdaptiveLossConfig, vocab: usize) -> Result<Self> {
        let path = dir.join(STATE_FILE);
        if !path.exists() { return Ok(Self::new(config.clone(), vocab)); }
        let state: Self = bincode::deserialize(&std::fs::read(&path)?)
            .with_context(|| format!("reading {}", path.display()))?;
        ensure!(state.vocab == vocab, "adaptive_loss state vocabulary differs from the tokenizer");
        ensure!(&state.config == config, "adaptive_loss config differs from the checkpoint's; use a new out_dir");
        Ok(state)
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        let tmp = dir.join(format!("{STATE_FILE}.tmp"));
        std::fs::write(&tmp, bincode::serialize(self)?)?;
        std::fs::rename(&tmp, dir.join(STATE_FILE))?;
        Ok(())
    }

    pub fn weights(&self) -> &[f32] { &self.weights }

    /// True when the next `observe` closes a window (the trainer measures the
    /// gradient norm on that step only, to avoid a readback every step).
    pub fn closes_window_next_step(&self) -> bool {
        self.window_steps + 1 >= self.config.statistics_interval_steps
    }

    /// Per-row a_v = (1 - lambda) + lambda * w_v, and 0 for PAD/invalid targets.
    pub fn row_weights(&self, targets: &[i32], lambda: f64) -> Vec<f32> {
        targets.iter().map(|&t| {
            if t == PAD_TOKEN as i32 || t < 0 || t as usize >= self.vocab { return 0.0; }
            ((1.0 - lambda) + lambda * self.weights[t as usize] as f64) as f32
        }).collect()
    }

    /// Adds one step's detached per-token losses. Returns true when this step
    /// closed a statistics window and the weights were recomputed.
    pub fn observe(&mut self, targets: &[i32], nll: &[f32]) -> bool {
        for (&t, &l) in targets.iter().zip(nll) {
            if t == PAD_TOKEN as i32 || t < 0 || t as usize >= self.vocab || !l.is_finite() { continue; }
            self.pending_sum[t as usize] += l as f64;
            self.pending_count[t as usize] += 1;
        }
        self.window_steps += 1;
        if self.window_steps < self.config.statistics_interval_steps { return false; }
        self.close_window();
        true
    }

    fn close_window(&mut self) {
        let beta = self.config.difficulty_beta as f32;
        let (mut sum, mut n) = (0.0f64, 0u64);
        for v in 0..self.vocab {
            let count = self.pending_count[v];
            self.counts[v] = beta * self.counts[v] + count as f32;
            if count == 0 { continue; }
            let mean = (self.pending_sum[v] / count as f64) as f32;
            let d = &mut self.difficulty[v];
            *d = if d.is_nan() { mean } else { beta * *d + (1.0 - beta) * mean };
            sum += self.pending_sum[v];
            n += count as u64;
        }
        if n > 0 {
            let mean = (sum / n as f64) as f32;
            let g = &mut self.global_difficulty;
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

    /// R_v = C_v / (C_v + kappa).
    fn confidence(&self, v: usize) -> f32 {
        let c = self.counts[v];
        let denom = c + self.config.confidence_kappa as f32;
        if denom > 0.0 { c / denom } else { 0.0 }
    }

    /// P_v = D_v(k - h) - D_v(k) against the oldest retained snapshot; 0 when either is unset.
    pub fn progress(&self, v: usize) -> f32 {
        let old = self.history.front().map_or(f32::NAN, |h| h[v]);
        let p = old - self.difficulty[v];
        if p.is_finite() { p } else { 0.0 }
    }

    /// D~_v = R_v D_v + (1 - R_v) D_global.
    pub fn smoothed_difficulty(&self, v: usize) -> f32 {
        let global = if self.global_difficulty.is_nan() { 0.0 } else { self.global_difficulty };
        let d = self.difficulty[v];
        if d.is_nan() { return global; }
        let r = self.confidence(v);
        r * d + (1.0 - r) * global
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
        // Unreliable progress shrinks toward the count-weighted global positive progress,
        // the same way difficulty shrinks toward D_global, so unseen ids score near neutral.
        let global_progress = (0..self.vocab).map(|v| self.counts[v] as f64 * self.progress(v).max(0.0) as f64)
            .sum::<f64>() / total_count;
        let (eps, kappa) = (cfg.epsilon, cfg.confidence_kappa);
        let scores: Vec<f64> = (0..self.vocab).map(|v| {
            let difficulty = (self.smoothed_difficulty(v) as f64 + eps).powf(cfg.difficulty_alpha);
            match mode {
                AdaptiveMode::Frequency => 1.0 / (self.counts[v] as f64 + kappa + eps),
                AdaptiveMode::Difficulty => difficulty,
                _ => {
                    let r = self.confidence(v) as f64;
                    let progress = r * self.progress(v).max(0.0) as f64 + (1.0 - r) * global_progress;
                    difficulty * (progress + cfg.progress_floor).powf(cfg.progress_eta)
                }
            }
        }).collect();
        // S_ref: occurrence-weighted mean score, so typical training tokens sit near w = 1.
        let reference = (0..self.vocab).map(|v| self.counts[v] as f64 * scores[v]).sum::<f64>() / total_count;
        for v in 0..self.vocab {
            let w = (scores[v] / (reference + eps)).powf(cfg.weight_gamma).clamp(cfg.min_weight, cfg.max_weight);
            self.weights[v] = if w.is_finite() { w as f32 } else { 1.0 };
        }
    }

    pub fn report(&self, buckets: &FrequencyBuckets) -> WindowReport {
        let observed: Vec<usize> = (0..self.vocab).filter(|&v| self.counts[v] > 0.0).collect();
        let quantile = |mut values: Vec<f32>, q: f32| -> f32 {
            if values.is_empty() { return f32::NAN; }
            values.sort_by(|a, b| a.total_cmp(b));
            values[((values.len() - 1) as f32 * q).round() as usize]
        };
        let difficulty: Vec<f32> = observed.iter().map(|&v| self.difficulty[v]).filter(|d| !d.is_nan()).collect();
        let weights: Vec<f32> = observed.iter().map(|&v| self.weights[v]).collect();
        let total_count: f64 = observed.iter().map(|&v| self.counts[v] as f64).sum();
        let progress_mean = observed.iter().map(|&v| self.counts[v] as f64 * self.progress(v).max(0.0) as f64)
            .sum::<f64>() / total_count.max(f64::MIN_POSITIVE);
        let n = observed.len().max(1) as f32;
        let (lo, hi) = (self.config.min_weight as f32, self.config.max_weight as f32);
        let mut by_bucket = [(0.0f64, 0.0f64); 5];
        for &v in &observed {
            let b = &mut by_bucket[buckets.of(v)];
            b.0 += self.counts[v] as f64 * self.weights[v] as f64;
            b.1 += self.counts[v] as f64;
        }
        WindowReport {
            window: self.windows,
            observed_tokens: observed.len(),
            global_difficulty: self.global_difficulty,
            difficulty_p50: quantile(difficulty.clone(), 0.5),
            difficulty_p95: quantile(difficulty, 0.95),
            progress_mean: progress_mean as f32,
            progress_positive_fraction: observed.iter().filter(|&&v| self.progress(v) > 0.0).count() as f32 / n,
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
