//! Experimental MoE optimizers selected by the config's `optimizer` section
//! (docs/OPTIMIZERS.md). Both are Burn `SimpleOptimizer`s, so every parameter
//! tensor is its own block: curvature is estimated per tensor and never mixed
//! across tensors. Everything stays on device; scalars are `[1]` tensors and the
//! only readbacks are the periodic stats.
//!
//! - `slbfgs`: block-diagonal stochastic L-BFGS for small batches. Curvature
//!   pairs come from re-evaluating the same batch after a step (`probe`, needs the
//!   training loop) or from EMA-averaged iterates and momentum (`ema`, free).
//!   H0 = gamma * I from the newest pair, pairs are Powell-damped, and the step is
//!   the quasi-Newton step itself: the trainer's lr only contributes its schedule
//!   shape (lr / first_lr), and a per-tensor RMS cap acts as the trust region.
//! - `fsecant`: lightweight diagonal quasi-Newton. Secant curvature from the
//!   momentum change over the step just taken, least-squares fitted per row and
//!   per column (Adafactor-style factors), with a Sophia-style clipped Newton step.
//!   One full-size state tensor (momentum), half of Adam.

use std::sync::{Arc, Mutex, atomic::{AtomicU8, Ordering}};

use burn::{
    module::AutodiffModule,
    optim::{AdamW, AdamWConfig, GradientsParams, Optimizer, SimpleOptimizer, adaptor::OptimizerAdaptor},
    grad_clipping::GradientClippingConfig,
    prelude::*,
    record::Record,
    tensor::backend::AutodiffBackend,
};
use serde::{Deserialize, Serialize};

/// Guards divisions by sums of squares; far below any real f32 dot product here.
const TINY: f32 = 1e-30;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OptimizerConfig {
    /// The default: Burn AdamW (beta 0.9/0.999, stage epsilon and weight decay).
    Adamw,
    Slbfgs(SlbfgsConfig),
    Fsecant(FsecantConfig),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CurvatureSource {
    /// y = g_B(x + s) - g_B(x) on the same batch and dropout mask: one extra
    /// forward/backward every `interval` steps.
    Probe,
    /// s and y from EMA-averaged parameters and momentum every `interval` steps.
    Ema,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SlbfgsConfig {
    /// Curvature pairs kept per tensor. Memory is 2 * history parameter copies.
    pub history: usize,
    pub curvature: CurvatureSource,
    /// Optimizer steps between curvature pairs.
    pub interval: usize,
    /// Momentum of the gradient fed to the two-loop recursion (and the EMA pairs).
    pub beta_1: f32,
    /// Multiplier on the quasi-Newton step H m (1 = full step), scaled by the
    /// schedule shape lr / first_lr.
    pub step_size: f32,
    /// Trust region: per-tensor cap on the RMS of the applied step, in parameter
    /// units. Also bounds the steps before the first curvature pair (H = I). It is
    /// the only step safeguard: gradients are not clipped, since rescaling them
    /// would rescale the Newton step too.
    pub max_step: f32,
    /// Powell damping: keep s'y >= damping * |s| |y| (B0 = I scaled to |y| / |s|).
    pub damping: f32,
    /// Steps between stats readbacks (optimizer.jsonl); 0 disables.
    pub stats_every: usize,
}

impl Default for SlbfgsConfig {
    fn default() -> Self {
        Self { history: 4, curvature: CurvatureSource::Probe, interval: 10, beta_1: 0.9, step_size: 1.0,
            max_step: 1e-3, damping: 0.2, stats_every: 200 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FsecantConfig {
    pub beta_1: f32,
    /// EMA decay of the row/column curvature estimates.
    pub beta_2: f32,
    /// Newton steps shorter than this (parameter units) are scaled by lr / radius;
    /// longer ones are clipped to lr per coordinate.
    pub radius: f32,
    /// Curvature floor as a fraction of the tensor's mean, so rows without a
    /// recent gradient take shrinking steps rather than sign steps.
    pub floor: f32,
    pub grad_clip: f32,
    pub stats_every: usize,
}

impl Default for FsecantConfig {
    fn default() -> Self {
        Self { beta_1: 0.9, beta_2: 0.99, radius: 0.001, floor: 0.1, grad_clip: 1.0, stats_every: 200 }
    }
}

impl OptimizerConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        let beta = |b: f32| b >= 0.0 && b < 1.0;
        match self {
            Self::Adamw => {}
            Self::Slbfgs(c) => {
                anyhow::ensure!(c.history >= 1 && c.interval >= 1, "slbfgs needs history >= 1 and interval >= 1");
                anyhow::ensure!(beta(c.beta_1), "slbfgs beta_1 must be in [0, 1)");
                anyhow::ensure!(c.damping > 0.0 && c.damping < 1.0, "slbfgs damping must be in (0, 1)");
                anyhow::ensure!(c.step_size > 0.0 && c.max_step > 0.0, "slbfgs step_size and max_step must be positive");
            }
            Self::Fsecant(c) => {
                anyhow::ensure!(beta(c.beta_1) && beta(c.beta_2), "fsecant betas must be in [0, 1)");
                anyhow::ensure!(c.radius > 0.0 && c.floor >= 0.0 && c.grad_clip >= 0.0,
                    "fsecant radius must be positive, floor and grad_clip >= 0");
            }
        }
        Ok(())
    }

    /// Run-name suffix, so arms never resume each other's checkpoints.
    pub fn tag(&self) -> String {
        match self {
            Self::Adamw => String::new(),
            Self::Slbfgs(c) => format!("_opt_slbfgs_{}_h{}_i{}_s{}_m{}",
                match c.curvature { CurvatureSource::Probe => "probe", CurvatureSource::Ema => "ema" }, c.history, c.interval,
                c.step_size, c.max_step),
            Self::Fsecant(c) => format!("_opt_fsecant_r{}", c.radius),
        }
    }

    pub fn needs_probe(&self) -> bool {
        matches!(self, Self::Slbfgs(c) if c.curvature == CurvatureSource::Probe)
    }
}

// ─── Shared control between the training loop and the per-tensor steps ──────

const PHASE_NORMAL: u8 = 0;
/// This step also stores (g, applied step) for a probe pair.
const PHASE_ARM: u8 = 1;
/// No update: turn the probe gradient into a curvature pair.
const PHASE_PROBE: u8 = 2;

/// Numel-weighted sums from one stats step, read back once per tensor.
#[derive(Debug, Default, Clone)]
pub struct OptimizerStats {
    pub tensors: usize,
    pub numel: f64,
    /// slbfgs: cosine between the L-BFGS direction and the momentum.
    pub cos_grad: f64,
    /// slbfgs: tensors whose step hit the `max_step` cap.
    pub capped: usize,
    /// slbfgs: mean curvature pairs held per tensor.
    pub history: f64,
    /// slbfgs: tensors whose newest pair needed Powell damping.
    pub damped: usize,
    /// fsecant: fraction of coordinates whose Newton step was not clipped.
    pub unclipped: f64,
    /// RMS of the applied update: divided by lr for fsecant, in parameter units
    /// for slbfgs.
    pub update_rms: f64,
}

impl OptimizerStats {
    pub fn to_json(&self, kind: &str) -> serde_json::Value {
        let n = self.numel.max(1.0);
        match kind {
            "slbfgs" => serde_json::json!({"tensors": self.tensors, "cos_grad": self.cos_grad / n,
                "mean_history": self.history / self.tensors.max(1) as f64, "damped_tensors": self.damped,
                "capped_tensors": self.capped, "update_rms": self.update_rms / n}),
            _ => serde_json::json!({"tensors": self.tensors, "unclipped": self.unclipped / n,
                "update_rms_over_lr": self.update_rms / n}),
        }
    }
}

#[derive(Clone, Default)]
struct Control {
    phase: Arc<AtomicU8>,
    collect: Arc<AtomicU8>,
    stats: Arc<Mutex<OptimizerStats>>,
}

impl Control {
    fn phase(&self) -> u8 { self.phase.load(Ordering::Relaxed) }
    fn collecting(&self) -> bool { self.collect.load(Ordering::Relaxed) != 0 }
    fn record(&self, f: impl FnOnce(&mut OptimizerStats)) {
        if let Ok(mut s) = self.stats.lock() { f(&mut s); }
    }
}

fn scalar<B: Backend>(t: Tensor<B, 1>) -> f64 {
    t.into_scalar().elem::<f32>() as f64
}

fn dot<B: Backend, const D: usize>(a: &Tensor<B, D>, b: &Tensor<B, D>) -> Tensor<B, 1> {
    (a.clone() * b.clone()).sum()
}

/// A `[1]` scalar shaped to broadcast against a rank-D tensor.
fn bc<B: Backend, const D: usize>(s: Tensor<B, 1>) -> Tensor<B, D> {
    s.reshape([1usize; D])
}

fn clip_norm<B: Backend, const D: usize>(g: Tensor<B, D>, max: f32) -> Tensor<B, D> {
    if max <= 0.0 { return g; }
    let norm = g.clone().square().sum().sqrt();
    let coef = norm.add_scalar(1e-6).recip().mul_scalar(max).clamp_max(1.0);
    g * bc(coef)
}

// ─── Stochastic block L-BFGS ─────────────────────────────────────────────────

#[derive(Clone)]
pub struct StochasticLbfgs {
    cfg: SlbfgsConfig,
    weight_decay: f32,
    /// The schedule's first lr; lr / peak_lr is the step's schedule factor.
    peak_lr: f64,
    control: Control,
}

#[derive(Record, Clone)]
pub struct SlbfgsState<B: Backend, const D: usize> {
    pub time: usize,
    pub m: Tensor<B, D>,
    pub s: Vec<Tensor<B, D>>,
    pub y: Vec<Tensor<B, D>>,
    pub rho: Vec<Tensor<B, 1>>,
    /// H0 = gamma * I, gamma = s'y / y'y from the newest usable pair.
    pub gamma: Tensor<B, 1>,
    /// Damping factor of the newest pair (1 = undamped), for stats.
    pub theta: Tensor<B, 1>,
    /// Probe: raw gradient and applied step from the armed step.
    pub pending_g: Option<Tensor<B, D>>,
    pub pending_s: Option<Tensor<B, D>>,
    /// EMA pairs: averaged parameters and the snapshots at the last pair.
    pub x_avg: Option<Tensor<B, D>>,
    pub x_snap: Option<Tensor<B, D>>,
    pub m_snap: Option<Tensor<B, D>>,
}

impl StochasticLbfgs {
    /// Powell-damped pair against B0 = (|y| / |s|) I, i.e. B0 scaled to match the
    /// magnitude of y. Pushes the pair and updates gamma when s'y_hat is usable; a
    /// zero pair gets rho = 0 and drops out.
    fn push_pair<B: Backend, const D: usize>(&self, state: &mut SlbfgsState<B, D>, s: Tensor<B, D>, y: Tensor<B, D>) {
        let s_norm = s.clone().square().sum().sqrt();
        let y_norm = y.clone().square().sum().sqrt();
        let scale = y_norm / s_norm.clone().add_scalar(TINY);
        let s_b_s = s_norm.square() * scale.clone();
        let s_y = dot(&s, &y);
        let damp = s_y.clone().lower(s_b_s.clone().mul_scalar(self.cfg.damping));
        let theta_damped = s_b_s.clone().mul_scalar(1.0 - self.cfg.damping)
            / (s_b_s - s_y).add_scalar(TINY);
        let theta = Tensor::ones_like(&theta_damped).mask_where(damp, theta_damped);
        let b0_s = s.clone() * bc(scale);
        let y_hat = y * bc(theta.clone()) + b0_s * bc(theta.clone().neg().add_scalar(1.0));
        let s_y_hat = dot(&s, &y_hat);
        // Safe reciprocal: tends to 0 instead of inf as s'y_hat -> 0.
        let rho = s_y_hat.clone() / (s_y_hat.clone().square().add_scalar(TINY));
        let gamma_new = s_y_hat.clone() / y_hat.clone().square().sum().add_scalar(TINY);
        let usable = s_y_hat.greater_elem(TINY.sqrt());
        state.gamma = state.gamma.clone().mask_where(usable, gamma_new);
        state.theta = theta;
        if state.s.len() == self.cfg.history {
            state.s.remove(0);
            state.y.remove(0);
            state.rho.remove(0);
        }
        state.s.push(s);
        state.y.push(y_hat);
        state.rho.push(rho);
    }

    /// Two-loop recursion H q with H0 = gamma * I.
    fn direction<B: Backend, const D: usize>(state: &SlbfgsState<B, D>, q: Tensor<B, D>) -> Tensor<B, D> {
        let k = state.s.len();
        let mut q = q;
        let mut alphas = Vec::with_capacity(k);
        for i in (0..k).rev() {
            let alpha = state.rho[i].clone() * dot(&state.s[i], &q);
            q = q - state.y[i].clone() * bc(alpha.clone());
            alphas.push(alpha);
        }
        let mut r = q * bc(state.gamma.clone());
        for i in 0..k {
            let beta = state.rho[i].clone() * dot(&state.y[i], &r);
            r = r + state.s[i].clone() * bc(alphas[k - 1 - i].clone() - beta);
        }
        r
    }
}

impl<B: Backend> SimpleOptimizer<B> for StochasticLbfgs {
    type State<const D: usize> = SlbfgsState<B, D>;

    fn step<const D: usize>(&self, lr: f64, tensor: Tensor<B, D>, grad: Tensor<B, D>, state: Option<Self::State<D>>)
        -> (Tensor<B, D>, Option<Self::State<D>>)
    {
        let phase = self.control.phase();
        if phase == PHASE_PROBE {
            let Some(mut state) = state else { return (tensor, None) };
            if let (Some(g0), Some(s)) = (state.pending_g.take(), state.pending_s.take()) {
                self.push_pair(&mut state, s, grad - g0);
            }
            return (tensor, Some(state));
        }

        let g = grad;
        let b1 = self.cfg.beta_1;
        let ema = self.cfg.curvature == CurvatureSource::Ema;
        // First step: momentum starts at g, so no bias correction is needed.
        let mut state = match state {
            Some(mut st) => {
                st.m = st.m.mul_scalar(b1) + g.clone().mul_scalar(1.0 - b1);
                st.time += 1;
                st
            }
            None => {
                let device = tensor.device();
                SlbfgsState {
                    time: 1, m: g.clone(), s: Vec::new(), y: Vec::new(), rho: Vec::new(),
                    gamma: Tensor::ones([1], &device), theta: Tensor::ones([1], &device),
                    pending_g: None, pending_s: None,
                    x_avg: ema.then(|| tensor.clone()), x_snap: ema.then(|| tensor.clone()), m_snap: ema.then(|| g.clone()),
                }
            }
        };

        if ema {
            let x_avg = state.x_avg.take().unwrap_or_else(|| tensor.clone());
            let x_avg = x_avg.mul_scalar(b1) + tensor.clone().mul_scalar(1.0 - b1);
            if state.time % self.cfg.interval == 0 {
                let x_snap = state.x_snap.take().unwrap_or_else(|| x_avg.clone());
                let m_snap = state.m_snap.take().unwrap_or_else(|| state.m.clone());
                let y = state.m.clone() - m_snap;
                self.push_pair(&mut state, x_avg.clone() - x_snap, y);
                state.x_snap = Some(x_avg.clone());
                state.m_snap = Some(state.m.clone());
            }
            state.x_avg = Some(x_avg);
        }

        // The quasi-Newton step H m sets direction and length; the schedule only
        // contributes its shape, and the RMS cap bounds the step (trust region).
        let d = Self::direction(&state, state.m.clone());
        let factor = self.cfg.step_size as f64 * if self.peak_lr > 0.0 { lr / self.peak_lr } else { 1.0 };
        let step = d.clone().mul_scalar(factor);
        let step_rms = step.clone().square().mean().sqrt();
        let cap = step_rms.add_scalar(TINY).recip().mul_scalar(self.cfg.max_step).clamp_max(1.0);
        let update = step * bc(cap.clone());

        if self.control.collecting() {
            let numel = tensor.shape().num_elements() as f64;
            let m_norm = state.m.clone().square().sum().sqrt();
            let d_norm = d.clone().square().sum().sqrt();
            let cos = scalar(dot(&d, &state.m) / (d_norm * m_norm).add_scalar(TINY));
            let rms = scalar(update.clone().square().mean().sqrt());
            let capped = scalar(cap) < 1.0;
            let damped = !state.s.is_empty() && scalar(state.theta.clone()) < 1.0;
            let held = state.s.len() as f64;
            self.control.record(|s| {
                s.tensors += 1;
                s.numel += numel;
                s.cos_grad += cos * numel;
                s.history += held;
                s.damped += damped as usize;
                s.capped += capped as usize;
                s.update_rms += rms * numel;
            });
        }

        let decay = 1.0 - lr * self.weight_decay as f64;
        let new = tensor.clone().mul_scalar(decay) - update;
        if phase == PHASE_ARM {
            state.pending_s = Some(new.clone() - tensor);
            state.pending_g = Some(g);
        }
        (new, Some(state))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &B::Device) -> Self::State<D> {
        let mv = |t: Tensor<B, D>| t.to_device(device);
        state.m = mv(state.m);
        state.s = state.s.into_iter().map(mv).collect();
        state.y = state.y.into_iter().map(mv).collect();
        state.rho = state.rho.into_iter().map(|t| t.to_device(device)).collect();
        state.gamma = state.gamma.to_device(device);
        state.theta = state.theta.to_device(device);
        state.pending_g = state.pending_g.map(mv);
        state.pending_s = state.pending_s.map(mv);
        state.x_avg = state.x_avg.map(mv);
        state.x_snap = state.x_snap.map(mv);
        state.m_snap = state.m_snap.map(mv);
        state
    }
}

// ─── Factored secant (lightweight) ───────────────────────────────────────────

#[derive(Clone)]
pub struct FactoredSecant {
    cfg: FsecantConfig,
    weight_decay: f32,
    control: Control,
}

#[derive(Record, Clone)]
pub struct FsecantState<B: Backend, const D: usize> {
    pub time: usize,
    pub m: Tensor<B, D>,
    /// Signed EMA row and column curvature (Rayleigh quotients of dm against dx).
    pub rows: Tensor<B, 2>,
    pub cols: Tensor<B, 2>,
    /// Learning rate of the previous step, to rebuild the step it applied.
    pub prev_lr: f64,
}

/// Rows x cols view: all leading dims are rows; a vector is one row.
fn matrix_dims(dims: &[usize]) -> [usize; 2] {
    match dims {
        [n] => [1, *n],
        _ => [dims[..dims.len() - 1].iter().product(), dims[dims.len() - 1]],
    }
}

impl FactoredSecant {
    /// h_ij = |r_i| |c_j| / mean|r|, exact for rank-one curvature, floored at
    /// `floor` times its mean.
    fn curvature<B: Backend>(&self, rows: &Tensor<B, 2>, cols: &Tensor<B, 2>) -> Tensor<B, 2> {
        let r = rows.clone().abs();
        let mean_r = r.clone().mean();
        let h = r * cols.clone().abs() / bc(mean_r.add_scalar(TINY));
        let floor = bc::<B, 2>(h.clone().mean().mul_scalar(self.cfg.floor));
        h.clone() + (floor - h).clamp_min(0.0)
    }

    /// Clipped Newton step in units of lr.
    fn unit_step<B: Backend>(&self, m: Tensor<B, 2>, h: Tensor<B, 2>) -> Tensor<B, 2> {
        (m / h.mul_scalar(self.cfg.radius).clamp_min(1e-12)).clamp(-1.0, 1.0)
    }
}

impl<B: Backend> SimpleOptimizer<B> for FactoredSecant {
    type State<const D: usize> = FsecantState<B, D>;

    fn step<const D: usize>(&self, lr: f64, tensor: Tensor<B, D>, grad: Tensor<B, D>, state: Option<Self::State<D>>)
        -> (Tensor<B, D>, Option<Self::State<D>>)
    {
        let dims = tensor.dims();
        let [n_rows, n_cols] = matrix_dims(&dims);
        let g = clip_norm(grad, self.cfg.grad_clip).reshape([n_rows, n_cols]);
        let x = tensor.clone().reshape([n_rows, n_cols]);
        let (b1, b2) = (self.cfg.beta_1, self.cfg.beta_2);
        let decay = lr * self.weight_decay as f64;

        let (time, m, rows, cols) = match state {
            Some(st) => {
                let m_old = st.m.reshape([n_rows, n_cols]);
                // The previous update, rebuilt from the factors before they change:
                // x = x_prev (1 - lr wd) - lr u  =>  x_prev = (x + lr u) / (1 - lr wd).
                let u_prev = self.unit_step(m_old.clone(), self.curvature(&st.rows, &st.cols));
                let prev_decay = 1.0 - st.prev_lr * self.weight_decay as f64;
                let x_prev = (x.clone() + u_prev.mul_scalar(st.prev_lr)).div_scalar(prev_decay);
                let dx = x.clone() - x_prev;
                // m_t - m_{t-1}; tracks H dx for a steadily moving iterate.
                let dm = (g.clone() - m_old.clone()).mul_scalar(1.0 - b1);
                let fit = |dim: usize, old: Tensor<B, 2>| {
                    let num = (dm.clone() * dx.clone()).sum_dim(dim);
                    let den = dx.clone().square().sum_dim(dim);
                    let q = num / den.clone().add_scalar(TINY);
                    let moved = den.greater_elem(TINY);
                    old.clone().mask_where(moved, old.mul_scalar(b2) + q.mul_scalar(1.0 - b2))
                };
                let rows = fit(1, st.rows);
                let cols = fit(0, st.cols);
                let m = m_old.mul_scalar(b1) + g.mul_scalar(1.0 - b1);
                (st.time + 1, m, rows, cols)
            }
            None => {
                let device = tensor.device();
                (1, g, Tensor::zeros([n_rows, 1], &device), Tensor::zeros([1, n_cols], &device))
            }
        };

        let h = self.curvature(&rows, &cols);
        let u = self.unit_step(m.clone(), h);
        if self.control.collecting() {
            let numel = (n_rows * n_cols) as f64;
            let unclipped = scalar(u.clone().abs().lower_elem(1.0).float().mean());
            let rms = scalar(u.clone().square().mean().sqrt());
            self.control.record(|s| {
                s.tensors += 1;
                s.numel += numel;
                s.unclipped += unclipped * numel;
                s.update_rms += rms * numel;
            });
        }
        let new = (x.mul_scalar(1.0 - decay) - u.mul_scalar(lr)).reshape(dims);
        (new, Some(FsecantState { time, m: m.reshape(dims), rows, cols, prev_lr: lr }))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &B::Device) -> Self::State<D> {
        state.m = state.m.to_device(device);
        state.rows = state.rows.to_device(device);
        state.cols = state.cols.to_device(device);
        state
    }
}

// ─── The optimizer the training loop holds ───────────────────────────────────

pub enum TrainOptimizer<M: AutodiffModule<B>, B: AutodiffBackend> {
    AdamW(OptimizerAdaptor<AdamW, M, B>),
    Slbfgs(OptimizerAdaptor<StochasticLbfgs, M, B>, Schedule),
    Fsecant(OptimizerAdaptor<FactoredSecant, M, B>, Schedule),
}

/// CPU-side step counter deciding which steps arm a probe or collect stats.
pub struct Schedule {
    control: Control,
    steps: usize,
    probe_interval: Option<usize>,
    stats_every: usize,
    probe_pending: bool,
}

impl Schedule {
    fn new(control: Control, probe_interval: Option<usize>, stats_every: usize) -> Self {
        Self { control, steps: 0, probe_interval, stats_every, probe_pending: false }
    }

    fn before_step(&mut self) {
        self.steps += 1;
        let arm = self.probe_interval.is_some_and(|i| self.steps % i == 0);
        self.control.phase.store(if arm { PHASE_ARM } else { PHASE_NORMAL }, Ordering::Relaxed);
        let collect = self.stats_every > 0 && self.steps % self.stats_every == 0;
        self.control.collect.store(collect as u8, Ordering::Relaxed);
        if collect {
            if let Ok(mut s) = self.control.stats.lock() { *s = OptimizerStats::default(); }
        }
        self.probe_pending = arm;
    }

    fn after_step(&mut self) {
        self.control.phase.store(PHASE_NORMAL, Ordering::Relaxed);
    }
}

impl<M: AutodiffModule<B>, B: AutodiffBackend> TrainOptimizer<M, B> {
    /// `peak_lr` is the schedule's first lr; slbfgs scales its step by lr / peak_lr.
    pub fn new(config: Option<&OptimizerConfig>, weight_decay: f32, epsilon: f32, peak_lr: f64) -> Self {
        let clip = Some(GradientClippingConfig::Norm(1.0));
        match config.unwrap_or(&OptimizerConfig::Adamw) {
            OptimizerConfig::Adamw => Self::AdamW(AdamWConfig::new().with_epsilon(epsilon)
                .with_grad_clipping(clip).with_weight_decay(weight_decay).init()),
            OptimizerConfig::Slbfgs(cfg) => {
                let control = Control::default();
                let probe = (cfg.curvature == CurvatureSource::Probe).then_some(cfg.interval);
                let schedule = Schedule::new(control.clone(), probe, cfg.stats_every);
                let opt = StochasticLbfgs { cfg: cfg.clone(), weight_decay, peak_lr, control };
                Self::Slbfgs(OptimizerAdaptor::from(opt), schedule)
            }
            OptimizerConfig::Fsecant(cfg) => {
                let control = Control::default();
                let schedule = Schedule::new(control.clone(), None, cfg.stats_every);
                Self::Fsecant(OptimizerAdaptor::from(FactoredSecant { cfg: cfg.clone(), weight_decay, control }), schedule)
            }
        }
    }

    pub fn kind(&self) -> &'static str {
        match self { Self::AdamW(_) => "adamw", Self::Slbfgs(..) => "slbfgs", Self::Fsecant(..) => "fsecant" }
    }

    /// Whether the next `step` will arm a probe (seed dropout before its forward).
    pub fn arms_next_step(&self) -> bool {
        match self {
            Self::Slbfgs(_, s) => s.probe_interval.is_some_and(|i| (s.steps + 1) % i == 0),
            _ => false,
        }
    }

    pub fn step(&mut self, lr: f64, module: M, grads: GradientsParams) -> M {
        match self {
            Self::AdamW(o) => o.step(lr, module, grads),
            Self::Slbfgs(o, s) => { s.before_step(); let m = o.step(lr, module, grads); s.after_step(); m }
            Self::Fsecant(o, s) => { s.before_step(); let m = o.step(lr, module, grads); s.after_step(); m }
        }
    }

    /// True after an armed step until `probe` is called with the same batch's
    /// gradient at the updated parameters.
    pub fn probe_pending(&self) -> bool {
        matches!(self, Self::Slbfgs(_, s) if s.probe_pending)
    }

    /// Forms curvature pairs from the probe gradient; parameters are unchanged.
    pub fn probe(&mut self, module: M, grads: GradientsParams) -> M {
        let Self::Slbfgs(o, s) = self else { return module };
        s.control.phase.store(PHASE_PROBE, Ordering::Relaxed);
        // Keep this step's stats (if any) for take_stats; probes add none.
        let collect = s.control.collect.swap(0, Ordering::Relaxed);
        let module = o.step(0.0, module, grads);
        s.control.collect.store(collect, Ordering::Relaxed);
        s.control.phase.store(PHASE_NORMAL, Ordering::Relaxed);
        s.probe_pending = false;
        module
    }

    /// Stats from the last step if it was a stats step.
    pub fn take_stats(&mut self) -> Option<serde_json::Value> {
        let kind = self.kind();
        let (Self::Slbfgs(_, s) | Self::Fsecant(_, s)) = self else { return None };
        if s.control.collect.swap(0, Ordering::Relaxed) == 0 { return None; }
        let stats = s.control.stats.lock().ok()?.clone();
        Some(stats.to_json(kind))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{module::Param, nn::{Linear, LinearConfig}};

    type TB = burn::backend::Autodiff<burn::backend::Wgpu>;

    #[derive(Module, Debug)]
    struct Quad<B: Backend> {
        w: Param<Tensor<B, 2>>,
        lin: Linear<B>,
    }

    /// 0.5 * w' A w with eigenvalues 1..1000 (A diagonal, or rotated by a
    /// Householder reflection so it is dense), plus a linear layer pulled toward
    /// zero output on a fixed input.
    fn curvature_matrix(rotated: bool) -> Vec<f32> {
        let n = 64;
        let eig: Vec<f32> = (0..n).map(|i| 1.0 + 999.0 * (i as f32 / 63.0).powi(2)).collect();
        let v: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 11) as f32 - 5.0).collect();
        let vv: f32 = v.iter().map(|x| x * x).sum();
        // Q = I - 2 v v' / v'v is symmetric orthogonal; A = Q D Q.
        let q = |i: usize, j: usize| (i == j) as u8 as f32 - if rotated { 2.0 * v[i] * v[j] / vv } else { 0.0 };
        (0..n * n).map(|ij| { let (i, j) = (ij / n, ij % n); (0..n).map(|k| q(i, k) * eig[k] * q(k, j)).sum() }).collect()
    }

    fn loss(model: &Quad<TB>, a: &Tensor<TB, 2>, input: &Tensor<TB, 2>) -> Tensor<TB, 1> {
        let w = model.w.val().reshape([1, 64]);
        let quad = (w.clone().matmul(a.clone()) * w).sum().mul_scalar(0.5);
        quad + model.lin.forward(input.clone()).square().sum()
    }

    /// Final clean loss after `steps`; each step's gradient carries N(0, noise)
    /// per coordinate (a fresh "batch"), reused by the probe like a real batch.
    fn run(config: Option<OptimizerConfig>, steps: usize, lr: f64, noise: f64, rotated: bool) -> f32 {
        let device = Default::default();
        <TB as Backend>::seed(&device, 7);
        let a = Tensor::<TB, 1>::from_floats(curvature_matrix(rotated).as_slice(), &device).reshape([64, 64]);
        let input = Tensor::<TB, 2>::ones([4, 8], &device);
        let mut model = Quad { w: Param::from_tensor(Tensor::ones([8, 8], &device)), lin: LinearConfig::new(8, 4).init(&device) };
        let mut opt = TrainOptimizer::<Quad<TB>, TB>::new(config.as_ref(), 0.0, 1e-8, lr);
        let batch_loss = |model: &Quad<TB>, n: &Tensor<TB, 2>| loss(model, &a, &input) + (model.w.val() * n.clone()).sum();
        let mut seen_stats = 0;
        for _ in 0..steps {
            let n = Tensor::<TB, 2>::random([8, 8], burn::tensor::Distribution::Normal(0.0, noise), &device);
            let grads = GradientsParams::from_grads::<TB, Quad<TB>>(batch_loss(&model, &n).backward(), &model);
            model = opt.step(lr, model, grads);
            if opt.probe_pending() {
                let grads = GradientsParams::from_grads::<TB, Quad<TB>>(batch_loss(&model, &n).backward(), &model);
                model = opt.probe(model, grads);
            }
            if let Some(stats) = opt.take_stats() { seen_stats += 1; assert!(stats["tensors"].as_u64() == Some(3), "{stats}"); }
        }
        assert!(config.is_none() || steps < 10 || seen_stats > 0, "{config:?} never reported stats");
        let l = loss(&model, &a, &input).into_scalar().elem::<f32>();
        assert!(l.is_finite(), "{config:?} diverged");
        l
    }

    #[test]
    fn all_optimizers_descend_on_ill_conditioned_quadratics() {
        for rotated in [false, true] {
            let start = run(None, 0, 0.0, 0.0, rotated);
            for noise in [0.0, 5.0] {
                let probe = Some(OptimizerConfig::Slbfgs(SlbfgsConfig { interval: 2, max_step: 0.05, stats_every: 10, ..Default::default() }));
                let ema = Some(OptimizerConfig::Slbfgs(SlbfgsConfig { curvature: CurvatureSource::Ema, interval: 5, max_step: 0.05,
                    stats_every: 10, ..Default::default() }));
                let secant = Some(OptimizerConfig::Fsecant(FsecantConfig { stats_every: 10, ..Default::default() }));
                let results = [("adamw", None), ("slbfgs-probe", probe), ("slbfgs-ema", ema), ("fsecant", secant)]
                    .map(|(name, cfg)| (name, run(cfg, 200, 0.02, noise, rotated)));
                println!("rotated {rotated} noise {noise}: start {start:.1} {results:?}");
                for (name, l) in results {
                    assert!(l < start * 0.5, "{name} did not descend (rotated {rotated}, noise {noise}): {l} vs {start}");
                }
            }
        }
    }

    #[test]
    fn config_parses_and_tags() {
        let c: OptimizerConfig = serde_json::from_str(r#"{"kind":"slbfgs","history":3,"curvature":"ema"}"#).unwrap();
        assert_eq!(c.tag(), "_opt_slbfgs_ema_h3_i10_s1_m0.001");
        c.validate().unwrap();
        let c: OptimizerConfig = serde_json::from_str(r#"{"kind":"fsecant"}"#).unwrap();
        assert!(c.tag().starts_with("_opt_fsecant"));
        assert!(serde_json::from_str::<OptimizerConfig>(r#"{"kind":"slbfgs","histroy":3}"#).is_err());
        assert!(serde_json::from_str::<OptimizerConfig>(r#"{"kind":"slbfgs","beta_2":0.999}"#).is_err(), "no Adam fields left");
        assert!(serde_json::from_str::<OptimizerConfig>(r#"{"kind":"slbfgs","grad_clip":1.0}"#).is_err(), "slbfgs does not clip");
        let c: OptimizerConfig = serde_json::from_str(r#"{"kind":"adamw"}"#).unwrap();
        assert_eq!(c.tag(), "");
    }

    /// With no history H = I: the step is the momentum (= g on the first step),
    /// scaled by step_size * lr / peak_lr, and shrunk to max_step RMS when longer.
    #[test]
    fn slbfgs_first_step_is_momentum_within_trust_region() {
        let device = Default::default();
        let first_step = |max_step: f32, lr: f64| {
            let cfg = SlbfgsConfig { max_step, step_size: 0.5, ..Default::default() };
            let opt = StochasticLbfgs { cfg, weight_decay: 0.0, peak_lr: 0.2, control: Control::default() };
            let x = Tensor::<burn::backend::Wgpu, 1>::from_floats([1.0, -2.0, 3.0], &device);
            let g = Tensor::<burn::backend::Wgpu, 1>::from_floats([0.5, 4.0, -1.0], &device);
            let (new, _) = SimpleOptimizer::step(&opt, lr, x.clone(), g, None);
            (x - new).into_data().to_vec::<f32>().unwrap()
        };
        // Uncapped: 0.5 * (0.1 / 0.2) * g.
        let step = first_step(10.0, 0.1);
        for (s, e) in step.iter().zip([0.125, 1.0, -0.25]) { assert!((s - e).abs() < 1e-5, "{step:?}"); }
        // Capped: same direction, RMS = 0.01.
        let step = first_step(0.01, 0.1);
        let rms = (step.iter().map(|s| s * s).sum::<f32>() / 3.0).sqrt();
        assert!((rms - 0.01).abs() < 1e-5 && step[1] > 0.0 && step[2] < 0.0, "{step:?}");
    }

    /// The two-loop recursion satisfies the newest secant condition H y = s.
    #[test]
    fn slbfgs_two_loop_satisfies_secant_condition() {
        type IB = burn::backend::Wgpu;
        let device = Default::default();
        let opt = StochasticLbfgs { cfg: SlbfgsConfig::default(), weight_decay: 0.0, peak_lr: 1.0, control: Control::default() };
        let t = |v: [f32; 4]| Tensor::<IB, 1>::from_floats(v, &device);
        let a = [[4.0, 1.0, 0.0, 0.5], [1.0, 3.0, 0.2, 0.0], [0.0, 0.2, 2.0, 0.3], [0.5, 0.0, 0.3, 1.0]];
        let mul = |s: [f32; 4]| t(std::array::from_fn(|i| (0..4).map(|j| a[i][j] * s[j]).sum()));
        let mut state = SlbfgsState::<IB, 1> {
            time: 1, m: t([1.0; 4]), s: vec![], y: vec![], rho: vec![],
            gamma: Tensor::ones([1], &device), theta: Tensor::ones([1], &device),
            pending_g: None, pending_s: None, x_avg: None, x_snap: None, m_snap: None,
        };
        for s in [[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, -0.3, 0.2], [0.3, -0.2, 1.0, 1.0]] {
            opt.push_pair(&mut state, t(s), mul(s));
            assert!(state.theta.clone().into_scalar() == 1.0, "a positive definite A needs no damping");
            let hy: Vec<f32> = StochasticLbfgs::direction(&state, mul(s)).into_data().to_vec().unwrap();
            for (h, e) in hy.iter().zip(s) { assert!((h - e).abs() < 1e-4, "H y = {hy:?}, s = {s:?}"); }
        }
        // A negative-curvature pair is damped and keeps H positive definite.
        opt.push_pair(&mut state, t([1.0, 0.0, 0.0, 0.0]), t([-1.0, 0.0, 0.0, 0.0]));
        assert!(state.theta.clone().into_scalar() < 1.0);
        let q = t([1.0, -1.0, 0.5, 2.0]);
        let d = StochasticLbfgs::direction(&state, q.clone());
        assert!((d * q).sum().into_scalar() > 0.0, "direction must stay a descent direction");
    }

    #[test]
    fn matrix_dims_flatten_leading() {
        assert_eq!(matrix_dims(&[5]), [1, 5]);
        assert_eq!(matrix_dims(&[3, 4]), [3, 4]);
        assert_eq!(matrix_dims(&[2, 3, 4]), [6, 4]);
    }
}
