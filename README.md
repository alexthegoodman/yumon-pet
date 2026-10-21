# Yumon

Yumon Pet is a conversational desktop companion. Yumon Code trains on Rust code
with and without doc comments.

### Quick Start

If you are retraining Yumon Talk MoE:

> [!NOTE]
> Update your data sources in `src/brain/train.rs` first if needed.

#### 1. Regenerate base samples with turn 0 and "Human: " prefix
```sh
cargo run --release --bin cache_samples -- --max-seq-len 1024
```

#### 2. Build the data-driven 4k vocabulary & tokenizer

Note: still has some semantic redundancy and needs to be semantically deduped more effectively.
Once fully deduped, more tokens will be available for rich encoding.

```sh
cargo run --release --bin build_kilo_vocab
```

#### 3. Regenerate the compressed sample cache
```sh
cargo run --release --bin cache_yumon_talk -- --vocab training-cache/kilo_talk_vocab.json --output training-cache/kilo_talk_samples.bin --max-seq-len 1024 --force
```

#### 4. Refresh the n-gram decompressor
```sh
cargo run --release --bin build_decompressor
```

#### 5. Start training
```sh
cargo run --release --bin train_code -- --config configs/yumon-talk-moe.json
```

### For Distribution

Windows: Run the `scripts/package_release.ps1` script in PowerShell.

## Train and deploy Yumon Code

Run these from the repository root with the original buckets, tokenizer and
saved comment journals already prepared.

Check `docs/commands.md` for more details.

1. Train the tokenizer and prepare matching training caches:

   ```sh
   cargo run --release --no-default-features --bin train_bpe
   ```

2. Build the Rust books pretraining stage with the new tokenizer:

   ```sh
   cargo run --release --no-default-features --bin cache_samples -- --code-config training-cache/code_tokenizer_retrained/config.json --rust-docs
   ```

3. Build the Docker image:

   ```sh
   docker build -t alexthegoodman/yumon-code:latest .
   ```

4. Push the image:

   ```sh
   docker push alexthegoodman/yumon-code:latest
   ```

Docker automatically copies the new bundle from
`training-cache/code_tokenizer_retrained` and validates it during the build.
Launch the image on RunPod with persistent storage at `/workspace`; training
starts automatically.

To train locally with the same prepared bundle and regimen as RunPod, run this
from the repository root after step 1:

```sh
cargo run --release --no-default-features --bin train_code -- --config training-cache/code_tokenizer_retrained/config.json --require-curriculum
```

The bundle's `config.json` uses local tokenizer/cache paths and the same model,
training settings and balanced documented/undocumented curriculum as Docker's
`docker-config.json`. Set `out_dir` in the bundle's `config.json` to your local
checkpoint directory before launching; it inherits the preparation config's
output root. 

Optional `cuda-training` selects RunPod's CUDA/FP32 backend on Windows too;
omit it to use WGPU on Windows. Append `--check` to validate the bundle without
starting training.

Configure tokenizer and model settings in
[configs/yumon-code.json](configs/yumon-code.json). With `docs` set, epoch 1
trains only on the official Rust books (see
[Rust books stage](docs/commands.md#code-rust-books-pretraining-stage); rebuild
it after step 1). The code curriculum then uses low, adds mild, and adds
moderate from its third epoch onward. `pretrain_code` pretrains on all of
`code.bin` instead, or on `pretrain_cache` (`code_long.bin`, 2048 tokens in
`yumon-code-moe-lite.json`) for a longer context, with `pretrain_min_tokens` packing
short neighbouring items into blocks of at least that size; see
[code pretraining](docs/commands.md#code-code-pretraining-at-a-longer-context). Each tier
has equal documented and distinct undocumented samples. Tokenizer fitting
excludes validation files. Existing artifacts are preserved; use a new output
directory for subsequent tokenizer runs.

See the [command reference](docs/commands.md) for first-time data preparation,
validation, alternate configs, Pet apps and smoke tests. Details:
[comment generation](docs/synthetic-code-comments.md),
[complexity scoring](docs/code-complexity.md),
[RunPod sizing](configs/runpod-h100.md).

### Loss diagnostics, overfit check and reference model

The terminal/PNG chart plots loss log-log (raw points dim, EMA over them) with LR
and top-3 on their own panels, and writes a `.csv` beside the PNG. Every
`heatmap_every` steps (config, default 250, 0 = off) the run folder's `heatmaps/`
gets `batch_*.png` (sample x position, colored by token loss), `position_loss_*`,
`token_loss_*` (loss by token id sorted by frequency, plus cumulative loss share)
and `tokens_*.html` (held-out text colored token by token).

`overfit_samples: N` trains on N fixed low-tier sequences with dropout 0 and a
constant `first_lr`; it checks the label shift at startup and prints a verdict
(below 0.1 expected, above 1 suggests a shift, masking or padding bug):

```sh
cargo run --release --bin train_code -- --config configs/yumon-code-overfit.json
cargo run --release --bin train_code -- --config configs/yumon-code-overfit-reference.json
```

`"architecture": "reference"` selects `reference_model.rs`, a baseline built only
from stock `burn::nn` modules (`TransformerEncoder` with a causal mask, learned
positions, LayerNorm; no flash attention or RoPE). Same overfit config, so the two
curves are directly comparable.

### Adaptive learning-value loss (experimental)

An optional `adaptive_loss` config section ([design](docs/ADAPTIVE_LEARNING.md),
`src/brain/adaptive_loss.rs`) weights each target's CE by
`(1 - lambda) + lambda * w_v`. Here `w_v` comes from per-token difficulty, learning
progress and confidence, and is clipped to `[min_weight, max_weight]`. Weights
update every `statistics_interval_steps` from detached losses of completed windows.
`lambda` is 0 for `warmup_fraction` of the steps, then ramps over `ramp_fraction`.
The loss is still divided by the supervised token count, and MoE aux losses are
unchanged. Without the section, training is unchanged.

Modes: `disabled` (plain CE, same stats and logs: the baseline), `difficulty`,
`difficulty_progress`, `frequency` (inverse smoothed-count control) and `context`.
`context` keys the statistics by (occurrence context, token). Each target is
labelled `code_symbol`, `code_novel`, `code_copy`, `comment_novel` or
`comment_copy`, where a copy is a token id seen earlier in the same sequence.
Sparse keys shrink toward their category's mean. Each mode gets its own run folder
(`_alw_<mode>`). The backend is seeded from `seed` when the section is present.
State resumes from `adaptive_loss.bin`. Per window, `adaptive_loss.jsonl` logs
standard vs. weighted CE, CE and applied weight by context, mean effective weight,
weight/difficulty/progress quantiles, weight by training frequency, grad norm,
tokens/s and CPU overhead. Every validation pass logs unweighted CE by training
frequency and by context. Charts, metadata and `val_loss` always use unweighted CE.

`configs/yumon-code-moe-mini-adaptive.json` is set up for quick testing: `context`
mode with no warm-up or ramp (weights stay at 1 until the first 50-step window
closes), beta 0.9, kappa 10, gamma 1 and lambda 0.5.

```sh
for mode in disabled context difficulty_progress; do
  cargo run --release --bin train_code -- --config configs/yumon-code-moe-mini-adaptive.json --adaptive-mode $mode
done
```

### Adaptive depth with expert tiers (experimental)

An optional `adaptive_depth` section ([design](docs/ADAPTIVE_DEPTH.md),
`src/brain/adaptive_depth.rs`) splits the MoE layers into tiers such as easy,
medium and hard. Each tier has its own `layers`, `top_k` and `experts` groups
(`count` x `ff_dim`, mixed sizes allowed), plus a target `capacity`: the share of
tokens that should reach it. Every token runs the first tier. Before each later tier
a per-token gate decides whether that position goes on. Exited positions keep their
hidden state, stay visible to attention and skip the deeper experts. A budget loss
(`budget_loss_weight`, default 0.1) keeps the share reaching each tier near its
capacity. When the section is present it replaces `num_experts`/`top_k`/`ff_dim`,
and tier layers must add up to `n_layers`. Run names use `MoeDepth_<tiers>`.
`adaptive_depth.jsonl` and the console log the share of tokens reaching each tier.
Checkpoints from before this change still load.

```sh
cargo run --release --bin train_code -- --config configs/yumon-code-moe-mini-depth.json
```

### KV-cached inference and grouped-query attention

MoE and dense decoder generation keeps a KV cache: the prompt runs once through
flash attention, then each new token attends to cached keys/values. Exit gates
work as in training: exited positions still add their frozen keys/values. On
the local Intel UHD 770 at context ~1550, decoding was 7.8x faster for moe-lite
(196 to 25 ms/token) and 16x faster for moe-mini-depth (1.23 s to 75 ms/token).
`cargo test --release --lib -- --ignored moe_kv_cache_decode_timing --nocapture`
reruns the comparison.

Optional `kv_heads` (MoE only) shares each key/value head across
`attn_heads / kv_heads` query heads. The run name gains `_gqa<n>`, and checkpoints
without it still load. At these model sizes it made no measurable speed difference
(the cache is small), so the configs leave it unset.

## Yumon Nano / InstaCode Edit:

```
cargo run --release --no-default-features --bin yumon_nano -- tmp/example2.rs --checkpoint "D:/models/moe-mini-2/code_256h_24l_32a_1024ff_512len_b2_Moe_e8_k2_f32_curriculum_documented_balanced_v2" --architecture moe
```

## XLSTM Hyperparameter Mapping Breakdown

architecture                                                           │ "xlstm"                                                                │ Instantiates YumonXLstmBrain

embed_dim                                                              │ 256                                                                    │ Model dimension d_{model}

attn_heads                                                             │ 8                                                                      │ Number of independent mLSTM matrix-memory heads (256/8 = 32 head dim)

ff_dim                                                                 │ 1024                                                                   │ Hidden dimension of the post-mLSTM MLP block

n_layers                                                               │ 16                                                                     │ Number of stacked XLstmBlock units

vocab_size                                                             │ 4096                                                                   │ Matches Talk BPE tokenizer

batch_size                                                             │ 2                                                                      │ Batch size

epochs                                                                 │ 10                                                                     │ Total training epochs
