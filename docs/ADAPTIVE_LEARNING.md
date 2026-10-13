# User Story: Adaptive Learning-Value Loss for MoE Training

**Feature:** Adaptive Token Difficulty & Learning-Progress Weighting (ATDLP)

**Priority:** Research / Experimental Feature

**System:** Rust-based Mixture-of-Experts Language Model

**Objective:** Improve training efficiency by dynamically allocating gradient emphasis toward tokens that remain difficult but demonstrate measurable learning progress, while preserving standard cross-entropy as the foundation of language modeling.

---

## 1. User Story

**As a** language model researcher and engineer,

**I want** the MoE training pipeline to maintain historical estimates of token prediction difficulty and learning progress, and use those estimates to dynamically weight token-level cross-entropy loss,

**So that** training effort can be preferentially allocated toward learnable weaknesses rather than repeatedly emphasizing tokens the model already predicts confidently or tokens that remain unpredictable despite continued exposure.

The system should adapt these weights throughout training, reflecting the model's evolving capabilities.

This feature must be independently configurable, measurable, and reversible without modifying the underlying MoE architecture.

## 2. Vision and Research Hypothesis

Traditional autoregressive language modeling minimizes cross-entropy across all valid target positions.

Although individual token losses naturally differ, the objective does not explicitly account for the model's historical learning trajectory.

A frequent token may already be well understood. A rare token may still be difficult. However, rarity alone does not establish educational value.

Likewise, a token with consistently high loss may be inherently unpredictable rather than a productive learning opportunity.

We propose distinguishing three signals:

1. **Difficulty:** How poorly does the model currently predict this token?
2. **Learning progress:** Has the model recently improved at predicting this token?
3. **Reliability:** Do we have enough observations to trust these estimates?

The core hypothesis is:

> Training efficiency may improve when gradient emphasis is adaptively allocated toward tokens that are difficult, measurably learnable, and supported by reliable historical statistics.

Importantly, this is a hypothesis to evaluate experimentally, not an assumed property of the new objective.

### Long-term vision

Evolve from frequency-based token weighting toward an adaptive learning-allocation mechanism that estimates where additional optimization effort is most likely to improve generalization.

The initial implementation will operate at the vocabulary-token level. Future versions may consider context-dependent difficulty and learning progress.

---

## 3. Existing Architecture Context

The current model configuration includes:

| Parameter | Value |
|---|---|
| Architecture | Mixture of Experts |
| Experts | 8 |
| Active experts per token | Top 2 |
| Vocabulary size | 16,384 |
| Embedding dimension | 256 |
| Transformer layers | 16 |
| Attention heads | 8 |
| Feed-forward dimension | 1,024 |
| Maximum sequence length | 512 |
| Tokenizer | yumon_code_bpe |
| Batch size | 2 |
| Epochs | 15 |
| Training corpus | Rust source code |
| Learning rate | 0.0001 → 0.00002 |

The feature must integrate into the existing training loss computation.

It must not require changes to:

- Expert architecture or expert selection
- Router logits or top-k routing
- Attention mechanisms
- Tokenizer vocabulary
- Embedding dimensions
- Transformer layer structure
- Inference-time decoding

The adaptive weighting system will operate exclusively during training.

## 4. Mathematical Foundation

### 4.1 Baseline Cross-Entropy

For target token \(x_t\) and preceding context \(x_{<t}\), define:

\[
p_t=P_\theta(x_t\mid x_{<t})
\]

The standard token-level loss is:

\[
\ell_t=-\log(p_t)
\]

For a batch containing \(N\) valid target positions:

\[
\mathcal L_{\mathrm{CE}}
=\frac{1}{N}\sum_{t=1}^{N}\ell_t
\]

This remains the baseline objective.

### 4.2 Historical Token Difficulty

Maintain a difficulty estimate for each vocabulary token:

\[
D_v^{(k)}
\]

where \(v\) is a vocabulary token ID and \(k\) is the statistics update index.

For every statistics window, calculate the mean detached cross-entropy loss for occurrences of token \(v\):

\[
\bar{\ell}_v^{(k)}
=
\frac{
\sum_{t:x_t=v}\operatorname{stopgrad}(\ell_t)
}{
n_v^{(k)}
}
\]

where \(n_v^{(k)}\) is the number of valid occurrences in that window.

Update the difficulty estimate using an exponential moving average:

\[
D_v^{(k)}
=
\beta D_v^{(k-1)}
+
(1-\beta)\bar{\ell}_v^{(k)}
\]

If token \(v\) does not occur during a window, retain its previous difficulty estimate.

**Initial recommendation:**

\[
\beta=0.99
\]

Because the batch size is only 2 and the vocabulary contains 16,384 tokens, statistics should be accumulated across multiple batches before updating the EMA.

An update window of approximately 100 optimizer steps is a reasonable starting point, subject to memory and throughput measurements.

The EMA should update once per statistics window, not once per occurrence.

### 4.3 Learning-Progress Estimation

Difficulty alone cannot distinguish learnable tokens from inherently unpredictable tokens.

Define learning progress as the reduction in historical difficulty over a fixed interval:

\[
P_v^{(k)}
=
D_v^{(k-h)}-D_v^{(k)}
\]

where \(h\) is the historical comparison interval measured in statistics updates.

Interpretation:

- \(P_v>0\): Prediction loss has decreased.
- \(P_v\approx0\): Prediction loss is relatively stable.
- \(P_v<0\): Prediction loss has increased.

Positive progress is:

\[
P_v^+=\max(P_v,0)
\]

Learning progress must be estimated using completed historical windows, not the current differentiable loss.

**Important:** The initial implementation uses observed loss reduction as a proxy for learnability. This is not a causal estimate of how much benefit additional training on a token would provide.

### 4.4 Reliability and Sparse-Token Smoothing

Rare tokens may appear too infrequently to produce reliable difficulty or progress estimates.

Maintain a rolling observation count \(C_v\) for each token.

Define confidence:

\[
R_v=\frac{C_v}{C_v+\kappa}
\]

where \(\kappa\) controls the number of observations required before trusting the token-specific estimate.

For example:

\[
\kappa=100
\]

A token with very few observations has low confidence; a well-observed token approaches confidence 1.

Define a smoothed difficulty estimate:

\[
\widetilde D_v
=
R_vD_v+(1-R_v)D_{\mathrm{global}}
\]

where \(D_{\mathrm{global}}\) is a running global average of token-level cross-entropy.

This prevents newly encountered or sparsely observed tokens from receiving extreme weights.

The engineer should also suppress or shrink unreliable progress estimates. A simple first implementation may use:

\[
\widetilde P_v^+=R_v\max(P_v,0)
\]

Counts should be based on a bounded rolling history or decayed statistics rather than unbounded lifetime totals.

### 4.5 Adaptive Learning-Value Score

Define an experimental learning-value score:

\[
S_v=
(\widetilde D_v+\epsilon)^\alpha
(\widetilde P_v^++\delta)^\eta
\]

where:

- \(\alpha\) controls the importance of difficulty.
- \(\eta\) controls the importance of learning progress.
- \(\epsilon\) provides numerical stability.
- \(\delta>0\) maintains a nonzero score for tokens with no measured progress.

The intended behavior is:

| Difficulty | Progress | Desired behavior |
|---|---|---|
| Low | Low | Reduce adaptive emphasis |
| High | Positive | Increase adaptive emphasis |
| High | Near zero | Avoid automatically maximizing emphasis |
| Low | Positive | Moderate emphasis |
| Unreliable | Any | Fall back toward neutral weighting |

The formula is a starting hypothesis. In particular, high difficulty with zero progress may still deserve training, and measured progress may reflect changes in sampled contexts rather than genuine improvement.

### 4.6 Convert Scores into Weights

Normalize scores using a reference scale:

\[
\widehat S_v=
\frac{S_v}{S_{\mathrm{ref}}+\epsilon}
\]

For the initial implementation, \(S_{\mathrm{ref}}\) may be a smoothed mean score over the vocabulary, with appropriate handling for unobserved tokens.

Compute bounded adaptive weights:

\[
w_v=
\operatorname{clip}
\left(
\widehat S_v^\gamma,
w_{\min},
w_{\max}
\right)
\]

Recommended starting bounds:

\[
w_{\min}=0.5,\qquad w_{\max}=2.0
\]

The system must not assign zero weight to any valid target token.

### 4.7 Hybrid Training Objective

Preserve standard cross-entropy as the dominant objective.

Define the effective token weight:

\[
a_v=(1-\lambda)+\lambda w_v
\]

The adaptive language-modeling loss is:

\[
\boxed{
\mathcal L_{\mathrm{LM}}
=
\frac{1}{N}
\sum_{t=1}^{N}
a_{x_t}\ell_t
}
\]

Recommended initial value:

\[
\lambda=0.3
\]

At this setting, standard cross-entropy contributes a fixed 70% component, while the adaptive weight controls the remaining 30%.

For a complete MoE objective, retain existing auxiliary losses unchanged:

\[
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{LM}}
+
\mathcal L_{\mathrm{MoE,aux}}
\]

Any existing router load-balancing, regularization, or other auxiliary losses should preserve their current coefficients and behavior.

The adaptive mechanism must only modify the language-modeling token loss.

**Normalization decision:** The initial implementation should retain division by the number of valid tokens, \(N\), rather than the sum of adaptive weights. This makes the formula explicit and allows the engineer to measure changes in gradient scale. A separately controlled weight-mean normalization may be evaluated later.

---

## 5. Functional Requirements

### FR-1: Token Statistics Registry

Create a lightweight training-only component responsible for maintaining:

- Token difficulty EMA
- Historical difficulty snapshots
- Learning-progress estimates
- Rolling occurrence counts
- Reliability scores
- Adaptive weights
- Global reference statistics

All arrays should be indexed directly by token ID.

For the current vocabulary, each primary array contains 16,384 entries.

Float32 is sufficient for the initial implementation.

### FR-2: Statistics Collection

After computing unreduced token-level cross-entropy, the training loop must:

1. Exclude padding, ignored targets, and other invalid positions.
2. Collect detached token losses.
3. Aggregate loss sums and occurrence counts by target token ID.
4. Update statistics at the configured interval.
5. Compute new adaptive weights for subsequent training steps.

The statistics update must not participate in autograd.

### FR-3: Adaptive Loss Integration

Replace the final reduction of language-modeling cross-entropy with a weighted reduction.

Conceptual pseudocode:

```text
logits = model.forward(inputs)

token_losses = cross_entropy(
    logits,
    targets,
    reduction = NONE
)

valid = valid_target_mask(targets)

weights = adaptive_loss.lookup_weights(targets)

effective_weights =
    (1 - lambda) + lambda * weights

lm_loss = sum(
    token_losses * effective_weights * valid
) / count(valid)

total_loss = lm_loss + existing_moe_aux_losses

backward(total_loss)
optimizer.step()

adaptive_loss.observe(
    targets,
    detach(token_losses),
    valid
)
```

The exact observation timing may differ with gradient accumulation or the framework's optimizer-step ordering, but weights applied to a step must come from previously completed statistics windows.

### FR-4: Warm-Up Period

Adaptive weighting should not begin immediately.

During warm-up:

\[
\lambda_{\mathrm{effective}}=0
\]

The system collects statistics while training with ordinary cross-entropy.

After warm-up, ramp the adaptive coefficient gradually:

\[
\lambda_{\mathrm{effective}}(s)
=
\lambda_{\max}
\min\left(
1,\frac{s-s_{\mathrm{warmup}}}{s_{\mathrm{ramp}}}
\right)
\]

where \(s\) is the optimizer step.

Recommended initial experiment:

- Warm-up: first 10% of training optimizer steps
- Ramp: next 10%
- Full adaptive weighting: remaining training

These percentages should be configurable.

### FR-5: Configuration

Add an optional configuration section without changing the existing MoE architecture fields.

Suggested configuration:

```json
{
  "adaptive_loss": {
    "enabled": true,
    "mode": "difficulty_progress",
    "lambda": 0.3,
    "difficulty_beta": 0.99,
    "difficulty_alpha": 1.0,
    "progress_eta": 0.5,
    "weight_gamma": 0.5,
    "min_weight": 0.5,
    "max_weight": 2.0,
    "confidence_kappa": 100,
    "statistics_interval_steps": 100,
    "progress_history_windows": 5,
    "warmup_fraction": 0.10,
    "ramp_fraction": 0.10,
    "epsilon": 0.000001,
    "progress_floor": 0.01
  }
}
```

These values are experimental defaults, not validated hyperparameters.

The implementation should support at least three modes:

- `disabled`: Standard cross-entropy only.
- `difficulty`: Adaptive weighting based on historical difficulty.
- `difficulty_progress`: Adaptive weighting based on difficulty and learning progress.

An optional `frequency` mode may be added for benchmarking the original inverse-frequency hypothesis.

### FR-6: Checkpoint Persistence

Save the adaptive statistics and configuration alongside the model's training checkpoint.

The checkpoint should preserve sufficient state to resume weighting consistently, including pending statistics accumulators, EMA values, progress history, counts, reference statistics, and the current warm-up/ramp position.

The feature must support loading checkpoints created before adaptive loss existed, initializing statistics safely when needed.

### FR-7: Monitoring and Observability

Log the following metrics:

| Metric | Purpose |
|---|---|
| Standard CE loss | Comparable baseline training signal |
| Adaptive LM loss | Actual weighted training objective |
| Validation CE | Primary generalization measure |
| Mean effective weight | Detect gradient-scale drift |
| Weight distribution | Detect saturation or collapse |
| Difficulty distribution | Track learned vs. difficult tokens |
| Learning-progress distribution | Track historical improvement |
| Weight vs. token frequency | Evaluate the original frequency hypothesis |
| Rare-token validation CE | Detect rare-token benefits or regressions |
| Gradient norm | Detect optimization instability |
| Tokens/sec and step time | Measure computational overhead |

Validation must always report ordinary, unweighted cross-entropy, regardless of training mode.

Weighted validation loss may be logged as an additional diagnostic, but must not replace standard validation CE for model comparison or checkpoint selection.

---

## 6. Engineering Constraints

**No inference overhead:** Adaptive statistics and weight lookups must not be required during generation.

**Minimal memory overhead:** The registry should use fixed-size vocabulary-indexed arrays and bounded history buffers.

**Deterministic behavior:** With the existing seed, the mechanism should produce reproducible results under the same execution environment and training order.

**Numerical stability:** Weights and losses must remain finite, including for unseen tokens and sparse statistics.

**No gradient through statistics:** The model must not optimize the weight estimator through the current token loss.

**Existing MoE compatibility:** Router behavior, expert selection, and auxiliary losses remain unchanged.

**Controlled overhead:** Measure the extra time required for statistics aggregation and weight lookup. Avoid per-token CPU synchronization or other operations that substantially reduce training throughput.

---

## 7. Acceptance Criteria

The feature is considered functionally complete when:

- [ ] Adaptive weighting can be enabled or disabled through configuration.
- [ ] Disabled mode reproduces existing training-loss behavior within numerical tolerance.
- [ ] Token difficulty statistics update correctly and ignore invalid positions.
- [ ] Unobserved tokens retain stable estimates.
- [ ] Learning-progress calculations use historical, detached statistics.
- [ ] Rare-token estimates are confidence-smoothed.
- [ ] Adaptive weights respect configured bounds.
- [ ] Warm-up and ramp behavior work correctly.
- [ ] The weighted loss produces valid gradients.
- [ ] Existing MoE auxiliary losses remain unchanged.
- [ ] Checkpoint save/resume preserves adaptive state.
- [ ] Standard validation cross-entropy remains available and unweighted.
- [ ] Training logs expose adaptive weighting diagnostics.
- [ ] No adaptive computation is required during inference.

Functional completion does not imply that the method improves model quality.

Performance improvements must be demonstrated separately.

---

## 8. Experimental Validation Plan

Run controlled comparisons using the same:

- Architecture and expert configuration
- Training and validation split
- Tokenizer
- Initialization seed
- Batch size
- Learning-rate schedule
- Training-token budget
- Data order, where feasible

### Experiment A: Baseline

Standard cross-entropy, adaptive loss disabled.

### Experiment B: Difficulty Only

Adaptive weighting using historical token difficulty.

### Experiment C: Difficulty + Progress

Full proposed adaptive weighting.

### Experiment D: Frequency-Based Control

Optional inverse-frequency weighting baseline.

### Primary Success Metric

Compare unweighted validation cross-entropy at equal numbers of processed training tokens and optimizer steps.

Also report total training FLOPs or wall-clock time so that any extra overhead is visible.

### Secondary Metrics

- Validation CE on token-frequency buckets
- Performance on held-out Rust source code
- Training stability
- Gradient norms
- Weight distribution over time
- Validation improvement per unit compute
- Generalization across different initialization seeds

Because the current validation fraction is only 1%, ensure the validation set contains enough tokens for meaningful bucket-level analysis.

Ideally, repeat promising experiments across multiple seeds.

### Success Definition

The hypothesis is supported if adaptive weighting provides a repeatable improvement in unweighted validation CE, downstream code-modeling performance, or convergence efficiency without unacceptable regressions in calibration, stability, or compute overhead.

If the adaptive objective only improves its own weighted training loss, that does not count as success.

---

## 9. Known Risks and Tradeoffs

### Distribution Distortion

Weighted cross-entropy generally changes the optimal predicted token distribution.

Retaining a standard CE component reduces but does not eliminate this effect.

The implementation must evaluate whether improved rare-token performance causes degraded calibration or excessive rare-token prediction.

### Token-Level Statistics Lose Context

The same token can be easy in one context and difficult in another.

A single vocabulary-level score cannot represent that distinction.

This is an accepted limitation of version one.

### Progress Is Not Causal Learning Value

Observed loss reduction may result from changing context distributions, general model improvements, or statistical noise.

Positive learning progress is only a heuristic for allocating additional training emphasis.

### Sparse Observations

Low-frequency tokens may produce unreliable statistics. Confidence smoothing and bounded weights are required.

### Nonstationary Training

Model parameters and data distributions change during training. Historical statistics may become stale.

The EMA, history window, and update interval must be tunable.

### MoE Specialization

Different experts may specialize in different code patterns.

Global token statistics do not distinguish expert-specific learning behavior.

Expert-conditioned statistics may be explored in the future but are explicitly out of scope for this implementation.

---

## 10. Future Research Extensions

Potential follow-up versions include:

**Context-aware difficulty:** Track difficulty for token occurrences conditioned on contextual features rather than only vocabulary identity.

**Expert-aware learning statistics:** Measure whether tokens exhibit different learning trajectories under different expert-routing patterns.

**Gradient-aware learning value:** Explore gradient magnitude, gradient alignment, or estimated validation-gradient benefit as alternatives to loss-based difficulty.

**Adaptive training-data sampling:** Use learning-value estimates to adjust sampling of training sequences rather than only loss weights.

**Compute-aware optimization:** Optimize learning progress per unit of training compute, rather than treating all token occurrences as equally expensive.

These extensions are not required for the initial release.

---

## 11. Implementation Scope and Deliverables

The engineer should deliver:

1. A self-contained adaptive-loss statistics and weighting component.
2. Integration into the existing autoregressive cross-entropy calculation.
3. Backward-compatible configuration parsing.
4. Training checkpoint serialization and restoration.
5. Unit tests for EMA updates, progress estimation, smoothing, clipping, masking, and weighted reduction.
6. Training diagnostics for evaluating the feature.
7. A benchmark comparing standard CE, difficulty-only weighting, and difficulty-plus-progress weighting.

**Out of scope:** Rewriting the MoE, changing expert routing, changing tokenization, modifying model inference, or introducing a separate learned loss-prediction network.

---

## 12. Final Product Principle

The purpose of this feature is not simply to favor rare tokens or punish common tokens.

It is to test whether training can become more efficient by adapting gradient emphasis to the model's evolving learning state.

**Guiding principle:**

> Allocate more learning emphasis where the model is struggling, where improvement appears achievable, and where the evidence is sufficiently reliable—while preserving the ability to learn the full language distribution.

The implementation should remain simple enough to benchmark, transparent enough to diagnose, and modular enough to disable if the hypothesis does not hold.