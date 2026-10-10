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

Code curriculum: production configs set `curriculum` to `training-cache/code_buckets`.
The shared code trainer uses low in epoch 1, low+mild in epoch 2, and
low+mild+moderate from epoch 3 onward; high is never loaded. Validation files
are split once across all eligible tiers. Dense, MoE, xLSTM and encoder-decoder
use this path. Docker copies only eligible buckets plus reference/report and
launches with `--require-curriculum`. Check with `train_code --check`.
Validation: 16 code CPU tests and the tiny WGPU curriculum/resume/causality
check passed for all four architectures. Docker image `yumon-code-curriculum:local`
built successfully; its Linux/CUDA CPU preflight validated the real three-bucket
schedule. Actual RunPod GPU training remains untested.
