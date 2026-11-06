# Experimental MoE optimizers

`src/brain/optimizers.rs`. An optional `optimizer` section in a MoE config replaces
AdamW in `train_causal_lm`. Absent (or `{"kind": "adamw"}`) keeps AdamW exactly as
before. Each kind adds a tag to the run name (`_opt_slbfgs_probe_h4_i10`,
`_opt_fsecant_r0.001`), so arms never resume each other. Optimizer state is not
checkpointed (neither was AdamW's), so a resumed run restarts its curvature history.

Both optimizers are Burn `SimpleOptimizer`s: every parameter tensor is its own block,
and curvature is never shared between tensors. All math stays on the device. Scalars
are `[1]` tensors and conditionals are masks. The only readbacks are the stats steps.

Every `stats_every` steps, a line goes to `optimizer.jsonl` in the run directory
(and to the console when it is not a TUI).

## `slbfgs`: block-diagonal stochastic L-BFGS

```json
"optimizer": {"kind": "slbfgs", "history": 4, "curvature": "probe", "interval": 10,
              "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-8, "damping": 0.2,
              "grad_clip": 1.0, "stats_every": 200}
```

Plain L-BFGS breaks at batch sizes of 2-32. Its gradient difference
y = g_{k+1} - g_k is taken on two different batches, so y is mostly sampling noise,
and the inverse-Hessian estimate fits that noise. Each piece below targets one part
of the problem:

- **Same-batch curvature (`curvature: "probe"`).** Every `interval` steps the loop
  re-runs the batch it just stepped on, at the new parameters and with the same
  dropout seed, and forms y = g_B(x + s) - g_B(x). Both gradients see the same
  samples, so the sampling noise cancels and y is a finite-difference Hessian-vector
  product H_B s (oLBFGS, Schraudolph et al. 2007; SQN, Byrd et al. 2016). Cost: one
  extra forward/backward per `interval` steps, about 10% at the default.
  MoE routing can still flip between the two evaluations, and damping absorbs that.
- **Averaged curvature (`curvature: "ema"`).** No extra pass. Every `interval`
  steps it takes s = x_avg - x_avg_prev and y = m - m_prev, where x_avg is an EMA of
  the parameters and m the gradient momentum, both with `beta_1`. For a steadily
  moving iterate, m is close to the gradient at x_avg, so the pair is consistent, and
  averaging over about 1/(1 - beta_1) batches cuts the noise. Use this one for DPO
  (the probe re-runs the CE loss only).
- **H0 = gamma * Adam diagonal.** The recursion starts from
  diag(1 / (sqrt(v) + eps)) instead of the identity, so directions the history does
  not cover still get Adam's per-coordinate scaling. gamma = s'y / y'P y comes from
  the newest usable pair. With empty history the direction is exactly Adam's.
- **Powell damping.** If s'y < damping * s'B0 s (negative or weak curvature, common
  in a nonconvex net), y is blended toward B0 s, so H stays positive definite and
  the step stays a descent direction. B0's scale is matched to |y|.
- **Safe reciprocal.** rho = s'y / ((s'y)^2 + tiny), so a degenerate pair drops out
  (rho -> 0) instead of blowing up. This keeps everything branch-free on the GPU.
- **Grafting.** The update takes the L-BFGS direction with the norm of the Adam
  step, per tensor (Agarwal et al., learning-rate grafting). L-BFGS supplies the
  shape and Adam the length, so the configs' learning rates and linear decay carry
  over unchanged. Without it the Newton-scaled step lengths would need a separate LR
  search.
- Momentum m (`beta_1`) is what the recursion transforms, not the raw gradient.
  Per-tensor norm clipping (`grad_clip`) applies to the update; probe pairs use raw
  gradients. Weight decay is decoupled, as in AdamW.

Memory per parameter: m, v, 2 x `history` pair vectors (default 4 -> 10 copies),
plus 2 transient copies at probe steps (or 3 persistent ones in `ema` mode). Compute:
about 4 x `history` dot products and axpys per tensor per step.

Stats: `cos_adam` (cosine between the L-BFGS and Adam directions; 1 means
curvature changed nothing), `mean_history`, `damped_tensors` (tensors whose newest
pair needed damping) and `update_rms_over_lr`.

## `fsecant`: factored secant (lightweight)

```json
"optimizer": {"kind": "fsecant", "beta_1": 0.9, "beta_2": 0.99, "radius": 0.001,
              "floor": 0.1, "grad_clip": 1.0, "stats_every": 200}
```

A diagonal quasi-Newton method that stores one full-size tensor, the momentum, so it
uses half of AdamW's optimizer memory.

- **Secant curvature for free.** The step just taken (dx) is rebuilt from state
  already in memory: the previous lr, the momentum and the curvature factors. The
  momentum change dm = (1 - beta_1)(g - m_old) tracks H dx for a steadily moving
  iterate (Apollo, Ma 2020, uses the same pair). No previous gradient or step is
  stored.
- **Factored least-squares fit.** One coordinate's dm/dx at batch size 2 is pure
  noise. Instead, each row and each column gets a Rayleigh quotient
  sum(dm * dx) / sum(dx^2) over its coordinates, which is the least-squares diagonal
  fit with one value per row (or column). Noise uncorrelated with dx cancels in the
  signed sum. Both are EMA'd with `beta_2`. The tensor is viewed as
  [all leading dims, last dim], so expert banks [E, d, f] are E*d rows by f columns.
- **Rank-one recombination.** h_ij = |r_i| |c_j| / mean|r|, exact when curvature
  factors as a_i b_j (the Adafactor identity). Taking the absolute value follows
  saddle-free Newton: negative curvature still gets a step along -m. Rows and columns
  with no movement keep their old estimate.
- **Sophia-style clipped Newton step.** The update is
  u = clip(m / (radius * h), -1, 1) and x -= lr * u. Coordinates whose Newton step
  is shorter than `radius` take lr * newton / radius; longer ones (flat or unknown
  curvature) are clipped to lr, which is sign descent. lr is therefore the
  per-coordinate maximum step, the same scale as Adam's typical step. Before any
  curvature exists, every step is a sign step.
- `floor` lifts h to at least `floor` times the tensor's mean, so embedding rows
  that stop receiving gradient take shrinking steps as their momentum decays, rather
  than sign steps.

Stats: `unclipped` (fraction of coordinates in the Newton regime; tune `radius`
with it) and `update_rms_over_lr`.

## Configs

| Config | Optimizer |
|---|---|
| `yumon-moe-base-pretrain.json` | AdamW (unchanged) |
| `yumon-moe-base-pretrain-slbfgs.json` | slbfgs, probe curvature |
| `yumon-moe-base-pretrain-slbfgs-ema.json` | slbfgs, EMA curvature (no extra pass) |
| `yumon-moe-base-pretrain-fsecant.json` | fsecant |

All four use the same `out_dir`. Run names differ by tag, so they sit side by side.

```sh
cargo run --release --bin train_code -- --config configs/yumon-moe-base-pretrain-slbfgs.json
```

## First local comparison (2026-11-06)

This is a smoke check, not a verdict. Setup: `yumon-moe-base-pretrain` (256h, 16
layers, 8 experts top-2, batch 2, lr 3e-4), Intel UHD 770 via WGPU, FP32, 420 s per
arm with the same seeded batch order, scratch `out_dir`. The loss is the epoch's
running mean of training CE (the console `avg`).

| Optimizer | Steps in 420 s | avg @ 300 | avg @ 350 | avg @ 400 | Notes |
|---|---|---|---|---|---|
| AdamW | 650 | 5.715 | 5.674 | 5.624 | |
| slbfgs probe | 350 | 5.817 | 5.780 | - | cos_adam 0.41-0.46; 76-80% of tensors' newest pair damped |
| slbfgs ema | 300 | 5.844 | - | - | cos_adam 0.48-0.55; 46-59% damped |
| fsecant r=0.01 | 350 | 5.789 | 5.746 | - | 94-99% unclipped, update RMS fell to 0.08 lr (effectively a lower LR) |
| fsecant r=0.001 | 400 | 5.748 | 5.709 | 5.660 | 62-71% unclipped, update RMS steady at about 0.63 lr |

What it shows so far:

- AdamW leads per step at this stage and runs about 1.6-2x more steps per second.
  The new optimizers launch many small kernels per tensor (Burn's per-tensor
  `SimpleOptimizer` path, about 450 tensors), which hurts on an iGPU. RunPod CUDA is
  untested.
- slbfgs: most curvature pairs are weak or negative at this stage and get damped,
  and the L-BFGS direction is far from Adam's (cosine about 0.45). The curvature is
  changing the steps, but over 350 steps that did not help. The first few hundred
  steps are the least informative part of training for a quasi-Newton method; the
  real comparison is full runs against the AdamW base run, especially late in
  training where the loss plateaus near 3.
- fsecant tracks AdamW closely with half its optimizer memory once `radius` is set
  so most coordinates sit in the Newton regime. 0.001 is the default after this
  run; use the `unclipped` stat to tune it.

## Tests

`cargo test --lib brain::optimizers` covers config parsing, the empty-history step
matching Adam, the secant condition H y_newest = s_newest after each undamped pair,
damping of a negative-curvature pair (the direction stays a descent direction), and
descent of every optimizer on diagonal and rotated quadratics (condition number
1000) with and without gradient noise.
