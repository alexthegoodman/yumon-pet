# Yumon Pet

Yumon is a tabletop ePet that responds to various inputs with short text replies and emotes.

The vision portion is currently unused. The brain constains the transformer for structured text generation with flash attention.

The bin folder contains the front-ends for interacting with Yumon.

Yumon Code (2026-10-09): `code_corpus` prepares recursive Rust sources with a
case-preserving tokenizer and shifted full-sequence targets. `cache_samples
--code-config configs/yumon-code.json` writes one raw-code cache; `train_code`
uses the same config locally on WGPU and on RunPod CUDA, both FP32. `--check` is
CPU-only validation. See README for preparation and context changes. Docker
builds and launches only Code by default; Pet instructions are commented out.
Cache preparation prints 50 chunks by default (`--preview-samples` controls it). The chunker now uses syn to select complete structs/functions, with full
impl/trait wrappers for methods; oversized items are reported and skipped.
Old token-window caches must be rebuilt with the existing tokenizer. Five
code-corpus acceptance tests and five existing cache tests passed locally. Local smoke training is authorized using configs/yumon-code-smoke.json. Docker/CUDA runtime validation is pending.

Code data (2026-10-12): `training-cache` holds `code.bin`, the frozen
`code-reference-v1.json`, and `documented/` and `undocumented/` folders with one
append-only `.bin` per tier (low/mild/moderate; high is never stored). Records are
file, location and code text. `documented/` holds code from the front of each tier
in `code.bin` with a `///` comment; `undocumented/` holds different, doc-free code
from the back, one per documented sample, so counts match. `gen_synthetic_captions`
tiers `code.bin` chunks with the frozen reference and appends one record to each
folder per sample; it resumes by skipping code already stored in either. Training (`code_data.rs`, `code_curriculum.rs`) tokenizes at load,
skips samples over `max_seq_len`, mixes both kinds within each tier and holds out
whole files. Epochs: all low, then 20% low + 80% mild, then 10/20/70 low/mild/moderate.
`train_bpe` fits only on stored training-pool text and writes just `tokenizer.json`
(`--replace` to overwrite); `code.bin` must then be rebuilt with `cache_samples`.
Configs set `curriculum` to `training-cache` (`documented` was removed). The old
buckets, journals and retrained bundles were moved to `../yumon-pet-cache-backup`;
`sort_code_buckets` and `view_code_bucket` still exist but training no longer uses
their output. Checkpoint run names use `_curriculum_documented_tiers_v3`.

Adaptive loss (2026-10-12): optional `adaptive_loss` config section
(`adaptive_loss.rs`, docs/ADAPTIVE_LEARNING.md) weights token CE by per-token
difficulty/progress/confidence with warm-up and ramp. Stats come from the per-token
NLL the trainer already reads back. State is in `adaptive_loss.bin`, diagnostics in
`adaptive_loss.jsonl`. Run names gain `_alw_<mode>`, and the backend is seeded only
when the section is present. `train_code --adaptive-mode` picks the experiment arm
for `configs/yumon-code-moe-mini-adaptive.json`. Unit tests and local smoke runs
(including resume and seeded repeatability) passed. Real A/B/C/D runs are not done yet.
2026-10-13: `context` mode keys stats by (occurrence context, token). `ContextLabeler`
tags targets as code_symbol/code_novel/code_copy/comment_novel/comment_copy (copy = id
seen earlier in the sequence), with category priors. All modes log CE and weight by
context. The test config is context mode with no warm-up, beta 0.9, kappa 10, gamma 1,
lambda 0.5 and out_dir moe-mini-adaptive-2. Older `adaptive_loss.bin` files restart fresh.

Adaptive depth (2026-10-13): optional `adaptive_depth` config (`adaptive_depth.rs`,
docs/ADAPTIVE_DEPTH.md) splits MoE layers into tiers (easy/medium/hard), each with its
own experts (`count` x `ff_dim` groups, mixed sizes allowed) and top-k. All tokens run
tier 1. Per-token sigmoid exit gates (causal, s >= 0.5, permanent) pick who continues.
Exited rows are frozen but still serve as K/V, and they skip deeper MoE. A
straight-through mask gives the gate an LM gradient, and the budget loss pulls reach
toward `capacity`. `YumonMoeBrain.exit_gates` is new. `load` falls back to
`LegacyMoeBrainRecord` for older checkpoints. `adaptive_depth.jsonl` logs reach per tier.
Configs: `yumon-code-moe-mini-depth.json`, `yumon-code-smoke-depth.json`. Tests and a
local smoke run (with resume) passed. Real runs are not done yet.

Rust books stage (2026-10-11): `code_docs` converts the toolchain `rust-docs`
component's mdBook `print.html` (12 stable books, unstable-book excluded) into
packed Markdown-like chunks via `cache_samples --code-config <cfg> --rust-docs`
(output dir = config `docs`: rust_docs.bin, report.json, samples.jsonl). With
`docs`, epoch 1 is docs-only and code tiers follow; `docs_share` (default 0)
optionally replays a rotating docs window by token share. Docs never enter
validation; run names gain `_rust_docs_v1`. The stage is tied to tokenizer and
context: rebuild after train_bpe. Docker copies it via `CODE_DOCS`. Retrained
bundle at 512: 5,027 samples / 1.94M tokens (code split is ~0.39M). Smoke
training passed locally; Docker build/RunPod untested.

Long-context pretraining (2026-10-13): `pretrain_code` reads `pretrain_cache` when set
(else `cache`). A raw cache only loads at its own context, and items over it were
skipped, so `cache_samples --code-config <cfg> --pretrain-cache` builds it at the
config's max_seq_len. `yumon-code-moe-lite.json` is now 2048 with
`training-cache/code_long.bin` (148,018 items vs 136,591 in the 512 `code.bin`; 842
still over 2048). `code.bin` stays 512 for captioning. Training and validation trim
each batch to its longest row (`trim_batch`), so short samples skip unused positions.
Preparation now skips files syn cannot parse instead of aborting. Docker does not copy
any raw code cache yet, so `pretrain_code` is local-only.
