# Johnson-Lindenstrauss projections in the dense decoder

Experiment: put a fixed random projection R [d, k] (entries N(0, 1/k)) somewhere
in the decoder-only model (`decoder_model.rs`) and see what it buys in speed and
costs in loss. By the JL lemma, |xR| ~ |x| and (xR).(yR) ~ x.y, with relative
error around sqrt(2/k), so geometry in the high-dimensional space is roughly
kept in the k-dimensional one.

Code: `src/brain/jl.rs` (config, spec parser, `JlProjection`), wired through
`YumonDecBrainConfig.jl`, `CodeConfig.jl`, `RunConfig.jl` and `train_code --jl`.

## Sites

| site | where | what changes |
| --- | --- | --- |
| `residual:K@L` | the residual stream, d -> K, before block L | blocks L.. run at width K (same head count, head dim K/heads, MLP width ff_dim * K/d unless `ff_dim` is set), and so do the final norm and token head. L = n_layers projects only before the head; L = 0 projects right after the embedding. |
| `attention:K@L` | per head in blocks L.., after RoPE | q, k and v go head_dim -> K through one R per block. Scores keep the 1/sqrt(head_dim) scale (q is multiplied by sqrt(K/head_dim) because the flash kernel divides by sqrt(K)), so they estimate the unprojected q.k. The output projection reads heads * K, and the KV cache stores K per head. |
| `ffn:K@L` | MLP input in blocks L.. | the normed MLP input goes d -> K; w1/w3 read K, w2 still writes d. |
| `none` | nowhere | baseline arm. Same seeding and run-name layout as the other arms. |

R is frozen (no gradient, the optimizer never touches it) unless the spec ends
in `+learned`, which trains R with everything else. That is the control for
"random projection vs learned projection of the same shape". Block i uses
seed + i. R is saved in the checkpoint.

A projection that follows a learned linear layer can be folded into it at
inference (W R is just a narrower matrix). The residual site (after a residual
add) and the attention site (after RoPE) cannot. The FFN site can (W1' = R W1),
so in training it acts as a rank-K limit on w1/w3 with a fixed basis.

## Running

`configs/yumon-dec-base-pretrain-jl.json` is the sunrise config
(`yumon-dec-base-pretrain.json`) with `out_dir` `checkpoints/yumon-base-dec-jl`
and `"jl": {"site": "none", "seed": 1}`. `--jl` replaces site, dim, layer and
`+learned` and keeps the config's seed and `ff_dim`. With a `jl` section the
backend is seeded with the config seed, so all arms start from the same
initialization and dropout streams. Run names gain `_jl_<site><K>_at<L>`
(`_learned`, `_ff<n>`, `_s<seed>` when set), so arms never resume each other.

```sh
cargo run --release --bin train_code -- --config configs/yumon-dec-base-pretrain-jl.json --jl none
cargo run --release --bin train_code -- --config configs/yumon-dec-base-pretrain-jl.json --jl residual:64@6
cargo run --release --bin train_code -- --config configs/yumon-dec-base-pretrain-jl.json --jl residual:64@3
cargo run --release --bin train_code -- --config configs/yumon-dec-base-pretrain-jl.json --jl residual:64@0
cargo run --release --bin train_code -- --config configs/yumon-dec-base-pretrain-jl.json --jl attention:16
cargo run --release --bin train_code -- --config configs/yumon-dec-base-pretrain-jl.json --jl ffn:64
cargo run --release --bin train_code -- --config configs/yumon-dec-base-pretrain-jl.json --jl residual:64@3+learned
```

Each arm also has its own config, so no flag is needed (all share the out_dir
and seed; `--jl` still overrides):

| config (`configs/yumon-dec-base-pretrain-jl-*.json`) | arm |
| --- | --- |
| `head64` | residual:64@6, projection only before the head |
| `residual64-mid` | residual:64@3 |
| `residual64-embed` | residual:64@0 |
| `residual64-mid-learned` | residual:64@3+learned (control) |
| `attn16` | attention:16, all blocks |
| `attn8-late` | attention:8 from block 3 |
| `ffn64` | ffn:64, all blocks |
| `width64-baseline` | no projection, embed_dim 64, ff_dim 256 (fair baseline for residual64-embed) |

The training log already writes `tokens_per_sec` and `step_ms` per window, so
the speed side of each arm comes out of the same run as its loss.

Checkpoints from before this change still load: `jl` is the record's last field.
Without a JL map, the loader reads the old layout, which also reads new
checkpoints because bincode ignores the trailing empty list. Picking the layout
from `metadata.json` matters because `BinBytesRecorder` panics on a short record
instead of returning an error, so a try-then-fall-back load cannot work.

## Measured so far

Training step time (forward + backward, BalancedCheckpointing, FP32, full
256-token rows) at the sunrise shape: width 128, 6 layers, 4 heads, ff 512,
vocab 8192, batch 16. Intel UHD 770 (WGPU), mean of 10 steps after 2 warm-up
steps. Repeated runs agreed within 1%.

```sh
PROBE_JL=residual:64@3 YUMON_PROBE=dec-balanced PROBE_WIDTH=128 PROBE_LAYERS=6 PROBE_HEADS=4 \
  PROBE_SEQ=256 PROBE_BATCH=16 PROBE_VOCAB=8192 PROBE_FF=512 PROBE_STEPS=12 \
  cargo test --release --lib training_memory_probe -- --ignored --nocapture
```

| arm | params (incl. R) | step ms | vs none | forward ms | held after forward |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 3.680M | 679 | - | 225 | 515 MiB |
| residual:64@6 (head only) | 3.164M | 617 | -9% | 202 | 514 MiB |
| residual:64@3 | 2.574M | 482 | -29% | 162 | 451 MiB |
| residual:64@0 | 1.983M | 344 | -49% | 124 | 389 MiB |
| attention:16 | 3.634M | 615 | -9% | 214 | 527 MiB |
| attention:8 | 3.608M | 599 | -12% | 211 | 514 MiB |
| ffn:64 | 3.336M | 635 | -6% | 210 | 519 MiB |
| ffn:32 | 3.115M | 608 | -10% | 198 | 515 MiB |

The residual site is the only one that changes much, and mostly because the
blocks after it are narrower. `residual:64@0` is close to a plain 64-wide model
with a 128-wide embedding table, so a 64-wide baseline (`embed_dim: 64`,
`ff_dim: 256`, no `jl`) is the fair comparison for it. The attention site saves
time despite the extra matmuls; my guess is that the flash kernel's work scales
with head dim, but that was not profiled.

Loss: not measured yet. That needs the full runs above. Overfit smoke runs (64
fixed sequences, 50 epochs, constant LR 1e-3) only show that every arm trains,
saves and generates. Final val loss: none 0.038, attention:16 0.038, ffn:64
0.039, residual:64@3 0.107 (2.57M vs 3.68M parameters).

Tests: `brain::jl` (spec parsing, limits, inner products kept on average),
`decoder_model` `jl_param_count_and_cache` (parameter count and KV-cached
logits = full forward for 8 site/layer variants), `jl_frozen_unless_trainable`
(no gradient and no AdamW change on frozen R, both on `+learned`),
`decoder_loads_pre_jl_records`, and a JL residual arm in the GPU
all-architectures train/resume test.

## Not done

- Loss comparison runs (above).
- Inference (KV-cached decoding) timing per arm; only training steps were timed.
- Sparse or structured maps (Achlioptas +-1, fast JL with a Hadamard transform)
  instead of a dense Gaussian.
- JL in the MoE decoder.
