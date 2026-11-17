# Gated attention

Config key `gated_attention` for the dense decoder (`architecture: "decoder-only"`), after Qiu et al.
2025, "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free"
(the SDPA-output gate, G1 in the paper). Implemented in `src/brain/decoder_model.rs` (`AttnGate`).

```json
"gated_attention": "elementwise"
```

| value | gate | weights per block |
|---|---|---|
| `none` (default) | no gate | 0 |
| `headwise` | one scalar per head | width x heads |
| `elementwise` | one value per head channel | width x heads x head_dim |

## Block

```
n    = rms_norm(x)
attn = flash(q(n), k(n), v(n))          [b, heads, s, head_dim]
g    = sigmoid(W_g n)                    [b, heads, s, 1 or head_dim]
x    = x + o(attn * g)
```

W_g is a bias-free linear on the same normed input as q/k/v, initialised like them, so a fresh
gate sits near 1/2. With an `attention` JL map the elementwise gate is heads x K wide (it gates the
projected head outputs `o` reads). The gate depends only on the query row: the KV cache, flash
attention, JL maps and N-Grammer layers are unchanged.

## Use

- `train_code --gate none|headwise|elementwise` overrides the config's arm.
- Run names gain `_gateh` / `_gatee`; resuming checks the arm matches.
- A gate arm seeds the backend like `jl`/`ngram` arms. Gates are created last in `init`, so every
  other weight starts identical to the ungated run with the same seed.
- Checkpoints: `attn_gates` is the last record field and `gated_attention` a metadata key (absent =
  none). Older checkpoints load through `NgramDecBrain` / `ProjDecBrain` / `JlDecBrain` /
  `LegacyDecBrain`.
- Weights: `YumonDecBrainConfig::gate_param_count()`, included in `param_count()`. White-dwarf shape
  (128 wide, 4 heads, 6 layers): elementwise 98,304, headwise 3,072.
- Config `configs/yumon-dec-base-pretrain-jl-ffn64-ngram-gated.json` is the white-dwarf config with
  `elementwise` and out_dir `checkpoints/yumon-dec-ngram-gated`.

## Status

Tests in decoder_model.rs: cached logits = full forward (headwise, elementwise, with attention JL,
with residual JL + N-Grammer + factorized tied table), zero gates = ungated with halved `o`,
BalancedCheckpointing gradients = plain, roundtrips, pre-gate N-Grammer records and the Portal
checkpoint load. Smoke training (16 samples, resume) passed. Real A/B runs are not done.

The white-dwarf config holds out 299 samples (`validation_fraction` 0.0001), so small val-loss
differences between arms may be noise.
