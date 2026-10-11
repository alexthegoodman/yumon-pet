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

Balanced documented curriculum: production configs additionally set `documented`
to `training-cache/code_buckets_documented` and use context 1024. All committed
low/mild/moderate JSONL comments are reconstructed against verified raw buckets;
each tier and validation partition gets equally many distinct undocumented
items, excluding matching documented source locations/code and existing doc
attributes. Journals are read without mutation or endpoint calls. Docker adds
only generation metadata and the three eligible journals; its runtime is unchanged.
Balanced checkpoint directories use `_curriculum_documented_balanced_v1` and
reject changed journals on resume. Freeze journal inputs for each training run.

Code tokenizer retraining: `train_bpe` defaults to the shared balanced Code
training split, excludes held-out files from BPE fitting, and preserves source
case/bytes. Default output is `training-cache/code_tokenizer_retrained`, a new
bundle with matching tokenizer, retokenized low/mild/moderate buckets, frozen
journals and local/Docker configs. Selection/source equality is checked before
publication. `--pet` retains the old prose-tokenizer path. README gives Docker
build arguments; existing artifacts and runtime settings are preserved.

Docker COPY inputs now default to the `training-cache/code_tokenizer_retrained`
bundle. README presents tokenizer -> docker build -> docker push; additional
commands are in docs/commands.md. Runtime/CUDA/startup settings are unchanged.

Tokenizer fitting reuses code_corpus::train_tokenizer() and the existing
BpeTokenizer. There is no code_tokenizer module or separate BPE implementation.
Corpus preparation and compatible cache rebuilding live in code_corpus.
