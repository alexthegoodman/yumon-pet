# Command reference

Yumon Pet is a conversational desktop companion. Yumon Code trains on complete
Rust functions, methods and structs. Run commands from this repository's root.
Native training uses FP32: WGPU on Windows, CUDA on headless Linux/RunPod.

## Code: data, tokenizer and training

The training cache holds `code.bin` (every chunk of `rust-code`), the frozen
`code-reference-v1.json`, and two folders with one file per tier:

```
training-cache/documented/tier_{1_low,2_mild,3_moderate}.bin    code with its doc comment
training-cache/undocumented/tier_{1_low,2_mild,3_moderate}.bin  the same code without it
```

The files are append-only text records (file, location, code), so there are no
hashes or positions to keep in sync and nothing depends on the tokenizer. The two
folders hold different code and always the same count: comments are written for
code taken from the front of each tier in `code.bin`, and plain code (without doc
comments) is taken from the back. Training tokenizes at load time (samples over `max_seq_len` are skipped), shuffles the two kinds together
within each tier, and holds out whole source files for validation. Configs set
`curriculum` to `training-cache`.

| Epoch | Eligible tiers |
| --- | --- |
| 1 | Low |
| 2 | Low + mild |
| 3 onward | Low + mild + moderate |

```sh
# Caption more code. Resumable: code already stored is skipped. See synthetic-code-comments.md.
cargo run --release --no-default-features --bin gen_synthetic_captions -- --endpoint "https://YOUR-ENDPOINT.endpoints.huggingface.cloud" --model "YOUR-SERVED-MODEL-ID" --token "YOUR-HF-TOKEN"

# Optional: refit the BPE on the stored samples (held-out files excluded). Writes only tokenizer.json.
cargo run --release --no-default-features --bin train_bpe -- --replace

# Validate without model training or GPU initialization, then train.
cargo run --release --no-default-features --bin train_code -- --config configs/yumon-code.json --check
cargo run --release --no-default-features --bin train_code -- --config configs/yumon-code.json
```

After a new tokenizer, `code.bin` (and the Rust books stage) were tokenized with the
old one: delete `training-cache/code.bin` and rerun `cache_samples` (below) before
captioning more. The documented/undocumented files need no rebuild.

## Code: Rust books pretraining stage

Epoch 1 can train on the official books from doc.rust-lang.org/stable before the
code tiers. Sources come from the toolchain's `rust-docs` component
(`rustup component add rust-docs`), which ships each mdBook's `print.html` with
includes and rustdoc's hidden lines already resolved.

```sh
# Needs "docs" in the config; uses its tokenizer and max_seq_len. CPU only.
cargo run --release --no-default-features --bin cache_samples -- --code-config training-cache/code_tokenizer_retrained/config.json --rust-docs --preview-samples 5
```

Default books: book, rust-by-example, reference, nomicon, edition-guide,
error_codes, cargo, rustdoc, clippy, style-guide, embedded-book, rustc
(`--books a,b` overrides; `--docs-html` uses another mdBook HTML root). Chapters
start new samples, sections are packed to the context, fenced code keeps its
attributes (`rust,compile_fail,E0382`) and is never split; oversized blocks are
counted and skipped. The output directory holds `rust_docs.bin`, `report.json`
(toolchain, per-book digests and counts) and `samples.jsonl` for inspection.
Docs never enter validation. `docs_share` (default 0) optionally replays a
rotating docs window taking that share of tokens in later epochs. After
`train_bpe`, rebuild the stage: it is tied to the tokenizer and context.

## Code: code pretraining at a longer context

`pretrain_code: true` trains the first `pretrain_epochs` on every chunk in a raw
code cache (minus held-out files), then runs the tiers. A cache only loads at the
context it was chunked for, and items over that context were skipped, so a
longer context needs its own cache: set `pretrain_cache` (e.g.
`training-cache/code_long.bin`) and build it at the config's `max_seq_len`.
`code.bin` stays the 512 cache that captioning reads. The tiers tokenize at load,
so they work at any context.

```sh
# Chunks rust-code into complete items up to max_seq_len - 1 tokens. CPU only.
cargo run --release --no-default-features --bin cache_samples -- --code-config configs/yumon-code-moe-lite.json --pretrain-cache --preview-samples 0
# Header only:
cargo run --release --no-default-features --bin cache_samples -- --code-config configs/yumon-code-moe-lite.json --pretrain-cache --inspect
```

`configs/yumon-code-moe-lite.json` uses 2048. Training trims each batch to its
longest sample, so short samples do not pay for the full context.

`pretrain_min_tokens` (0 = off) packs short items with their neighbours from the
same file, in source order and joined by a blank line, until a block has that
many tokens. Items already that long stay alone. A file's short remainder joins its
last block, or is dropped if it doesn't fit. Packing happens at load, so no cache
rebuild is needed. The run name gains `_min<n>`. On `code_long.bin`: 146,388 items
(mean 195 tokens) became 89,508 blocks (mean 318) at 128, dropping 998 items
(0.2% of tokens), or 60,474 blocks (mean 470) at 256, dropping 2,461 items (0.7%).
moe-lite uses 128.

## RunPod Docker

```sh
docker build -t alexthegoodman/yumon-code:latest .
docker push alexthegoodman/yumon-code:latest
```

Attach persistent storage at `/workspace`. The image startup runs:

```sh
./train_code --config configs/yumon-code.json --require-curriculum
```

Docker copies `yumon_code_bpe/tokenizer.json`, the six tier files and the optional
Rust books stage, then runs a CPU-only preflight. Override `CODE_TOKENIZER`,
`CODE_DATA`, `CODE_DOCS` and `CODE_CONFIG` build arguments for other locations.

Checkpoints use the config's `out_dir`. Resume restores weights and completed
epochs, but not optimizer state or the within-epoch position. A changed
tokenizer or data identity (including new captions) needs a fresh run.

## Code: first-time data preparation

Set the Rust `source` folder in [configs/yumon-code-local.json](../configs/yumon-code-local.json)
(or pass `--source <folder>`). This config uses a 512-token context.
Adding code to `rust-code` later needs only the first command (delete the old
`code.bin` first) and then more captioning; existing samples keep their tiers
because `code-reference-v1.json` is frozen.

```sh
# Create the initial source-only tokenizer and the raw cache (code.bin).
cargo run --release --no-default-features --bin cache_samples -- --code-config configs/yumon-code-local.json --train-code-tokenizer

# Reuse an existing tokenizer instead (rebuilding code.bin after adding code):
cargo run --release --no-default-features --bin cache_samples -- --code-config configs/yumon-code-local.json

# Inspect the cache metadata.
cargo run --release --no-default-features --bin cache_samples -- --code-config configs/yumon-code-local.json --inspect
```

Preparation refuses to overwrite existing artifacts. Raw preparation skips
oversized whole items and prints 50 samples by default (`--preview-samples 0`
disables previews). The captioner assigns tiers with the frozen reference and
covers the low, mild and moderate tiers; high code is never stored.

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
cargo run --release --bin yumon_portal -- --checkpoint <language-checkpoint> --architecture moe
cargo run --release --bin yumon_portal -- --check-feeds
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
cargo test --bin yumon_portal

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
