# Command reference

Yumon Pet is a conversational desktop companion. Yumon Code trains on complete
Rust functions, methods and structs. Run commands from this repository's root.
Native training uses FP32: WGPU on Windows, CUDA on headless Linux/RunPod.

## Code: tokenizer and training

The production config is [configs/yumon-code.json](../configs/yumon-code.json);
[configs/yumon-code-moe.json](../configs/yumon-code-moe.json) selects MoE.
Both use a 1,024-token context and the documented/undocumented curriculum:

| Epoch | Eligible tiers |
| --- | --- |
| 1 | Low |
| 2 | Low + mild |
| 3 onward | Low + mild + moderate |

Every saved documented record is retained across training and validation. Each
tier in each split gets equally many distinct undocumented items; matching
documented source items and existing doc attributes are excluded. High,
excluded and unscored buckets never enter training.

With the original buckets, tokenizer and saved comment journals already prepared:

```sh
# Fit BPE on exactly the balanced training split; validation files are excluded.
cargo run --release --no-default-features --bin train_bpe

# Validate the new bundle without model training or GPU initialization.
cargo run --release --no-default-features --bin train_code -- --config training-cache/code_tokenizer_retrained/config.json --check

# Train using that tokenizer and its matching caches.
cargo run --release --no-default-features --bin train_code -- --config training-cache/code_tokenizer_retrained/config.json
```

`train_bpe` feeds the selected source text into the existing
`code_corpus::train_tokenizer()` BPE trainer, which saves `BpeTokenizer` from
`bpe.rs`. It preserves case, source bytes and generated comments. It snapshots the
journals and writes a new bundle containing the tokenizer, retokenized buckets,
local/Docker configs, exact fitting corpus and provenance. Existing artifacts
are preserved. For another run, append a new output directory to the command.
Tier assignments and source bytes are preserved after retokenization. Records
exceeding `max_seq_len` are skipped with counts printed; records are never
truncated. Each tier and split keeps equal documented and undocumented counts.

## RunPod Docker

Docker defaults to the bundle produced by `train_bpe`:

```sh
docker build -t alexthegoodman/yumon-code:latest .
docker push alexthegoodman/yumon-code:latest
```

Attach persistent storage at `/workspace`. The existing image startup runs:

```sh
./train_code --config configs/yumon-code.json --require-curriculum
```

Docker copies eligible caches, journals, metadata and the tokenizer, then runs a
CPU-only preflight. Its CUDA runtime settings are unchanged. Plain `docker build`
uses `training-cache/code_tokenizer_retrained`. For alternate bundles, override
`CODE_TOKENIZER`, `CODE_BUCKETS`, `CODE_DOCUMENTED` and `CODE_CONFIG`.
The default bundle directory is allowed by `.dockerignore`; add exceptions for
other bundle paths beneath `training-cache`.

Checkpoints use the config's `out_dir`. Retrained bundles select a separate root.
Resume restores weights and completed epochs, but not optimizer state or the
within-epoch position. Changed tokenizer/data identities require a fresh run.
Freeze input journals for each run; later comments require a new bundle.

## Code: first-time data preparation

Skip this section if `training-cache/code_buckets`,
`training-cache/code_buckets_documented` and `yumon_code_bpe` already exist.
Set the Rust `source` folder in [configs/yumon-code-local.json](../configs/yumon-code-local.json)
(or pass `--source <folder>`). This bootstrap config uses a 512-token context,
leaving room for comments in the production 1,024-token context.

```sh
# Create the initial source-only tokenizer and raw cache.
cargo run --release --no-default-features --bin cache_samples -- --code-config configs/yumon-code-local.json --train-code-tokenizer

# Explicitly calibrate and sort the raw cache into complexity buckets.
cargo run --release --no-default-features --bin sort_code_buckets -- --code-config configs/yumon-code-local.json --reference training-cache/code-reference-v1.json --calibrate 1.0.0

# Generate resumable doc comments using your existing inference endpoint.
cargo run --release --no-default-features --bin gen_synthetic_captions -- --endpoint "https://YOUR-ENDPOINT.endpoints.huggingface.cloud" --model "YOUR-SERVED-MODEL-ID" --token "YOUR-HF-TOKEN"

# Inspect the original cache metadata.
cargo run --release --no-default-features --bin cache_samples -- --code-config configs/yumon-code-local.json --inspect
```

Preparation refuses to overwrite existing artifacts. Reuse a tokenizer by
omitting `--train-code-tokenizer`; choose new cache/reference/output paths for
new datasets. Raw preparation skips oversized whole items and prints 50 samples
by default (`--preview-samples 0` disables previews). Comment generation covers
four tiers, but training uses only low/mild/moderate committed journal records.
Then run `train_bpe` above to fit the final tokenizer on the actual training mix.

## Yumon Nano code editor

```sh
cargo run --release --no-default-features --bin yumon_nano -- src/example.rs --checkpoint <code-checkpoint-directory>
# For an MoE checkpoint:
cargo run --release --no-default-features --bin yumon_nano -- src/example.rs --checkpoint <code-checkpoint-directory> --architecture moe
```

Use the exact saved checkpoint directory containing `metadata.json`, `model.bin`
and `tokenizer.json`, rather than the training output root. Dense decoder and MoE
Yumon Code checkpoints are supported; inference runs locally on WGPU.

Completions appear in gray after 450 ms without typing. The prompt contains the
current line up to the cursor and up to 40 nearby preceding lines, limited to the
checkpoint's token budget. No text after the cursor is sent to the model. A long
current line reduces the completion budget; if it cannot fit at all, the editor
shows a status message. Override `--debounce-ms`, `--context-lines` (0 for current
line only), or `--max-tokens` (default 48) as needed.

Tab accepts a suggestion (otherwise inserts four spaces), Ctrl+R regenerates,
and Esc dismisses. Ctrl+O or Ctrl+S saves; Ctrl+X exits with an unsaved-change
prompt. Arrow keys, Home/End, PageUp/PageDown, Delete/Backspace, and terminal
bracketed paste are supported. UTF-8 text and CRLF files are supported.

## Pet and desktop apps

```sh
# Legacy Pet prose tokenizer; default output is yumon_bpe.
cargo run --release --no-default-features --bin train_bpe -- --pet

# Prepare/inspect the Pet cache (context must match the training grid).
cargo run --release --no-default-features --bin cache_samples -- --max-seq-len 256
cargo run --release --no-default-features --bin cache_samples -- --inspect

# Pet model training; --architecture decoder-only selects the dense model.
cargo run --release --bin yumon-pet -- train-brain --architecture moe --batch-size 8 --epochs 15 --out-dir checkpoints/moe --moe-experts 4 --moe-top-k 1

# Desktop apps.
cargo run --release --bin chat_ui
cargo run --release --bin yumon_world
cargo run --release --bin yumon_universe -- --checkpoint <language-checkpoint> --seed 42 --theme suburban
cargo run --release --bin yumon_rss -- --checkpoint <language-checkpoint> --architecture moe
cargo run --release --bin yumon_rss -- --check-feeds
cargo run --release --bin endless_data

# Web app (clang may be needed).
trunk serve --release
trunk build --release
```

Set `YUMON_SAMPLE_CACHE=training-cache/samples.bin` to train Pet from its prepared
cache; otherwise it prepares raw sources. In PowerShell:
`$env:YUMON_SAMPLE_CACHE = "training-cache/samples.bin"`.
The active Dockerfile builds Code only; Pet Docker instructions remain commented.
RSS caches Product Hunt's daily feed and supports chat; select an architecture
matching the Language-stage checkpoint.

Optional synthetic Pet data:

```sh
cargo run --no-default-features --bin gen_synthetic_data -- --preview
cargo run --release --no-default-features --bin gen_synthetic_data -- --topics puzzles,commands,memory --target 12 --seed 42
```

## Checks and smoke runs

```sh
# CPU data, curriculum, tokenizer and cache checks.
cargo test --no-default-features --lib brain::code_
cargo test --no-default-features --lib sample_cache
cargo test --no-default-features --bin gen_synthetic_captions
cargo test --bin yumon_rss

# Tiny local Code model (first command is preparation only, run once).
cargo run --no-default-features --bin cache_samples -- --code-config configs/yumon-code-smoke.json --train-code-tokenizer --preview-samples 0
cargo run --no-default-features --bin train_code -- --config configs/yumon-code-smoke.json --check
cargo run --no-default-features --bin train_code -- --config configs/yumon-code-smoke.json

# On a CUDA pod: tiny FP32 MoE training/checkpoint check, no corpus needed.
./train_code --smoke-test
```

Details: [complexity scoring](code-complexity.md),
[comment generation and resume](synthetic-code-comments.md),
[RunPod sizing notes](../configs/runpod-h100.md).
