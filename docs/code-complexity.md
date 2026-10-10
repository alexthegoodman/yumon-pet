# Rust complexity buckets

`sort_code_buckets` reads the existing raw-code cache, decodes each complete
item with its matching tokenizer, and writes training-compatible caches. It
does not read the original corpus or initialize a model. Rust is the initial
supported language. Functions and methods are the normal analysis unit;
existing struct samples use `analysis_unit: item`. A method retains its
impl/trait wrapper. Multiple functions in a supplied item are aggregated.

Create a frozen reference explicitly, then reuse it for future datasets:

```powershell
cargo run --release --no-default-features --bin sort_code_buckets -- --reference training-cache/code-reference-v1.json --calibrate 1.0.0 --preview-samples 12
cargo run --release --no-default-features --bin sort_code_buckets -- --reference training-cache/code-reference-v1.json --output training-cache/code_buckets-next
```

`--code-config` defaults to `configs/yumon-code.json`. Input cache dimensions,
stage and seed come from its header; tokenizer and cache paths come from the
config. The loader validates tokenizer identity, raw-code shifting, corpus
digest and complete-item preparation. References and output directories must
be new. Preserve the calibration JSON with your dataset; it contains the
reference distributions, source cache SHA-256, parser/analyzer versions,
weights, missing-metric policy and boundaries. Outputs also include a copy.
Nothing automatically recalibrates when a processing batch changes.

The default weights are CC .25, cognitive .30, depth .20, mutation .15,
Halstead .10. Override them only during explicit calibration using
`--weights 0.25,0.30,0.20,0.15,0.10`. They must sum to one. New methodology
requires a new version and reference path. Parser versions are pinned in Cargo.

## Metric definitions: rust-syntax-1.0.0

These are reproducible syntax metrics, without name resolution, type checking,
macro expansion, CFG construction or runtime interpretation. Cognitive
complexity is this analyzer's documented variant, rather than a claim of
compatibility with another analyzer.

| Metric | Counting rule |
| --- | --- |
| Cyclomatic | One per function/method body, plus each if, while, for, `&&`, `||`, `?`, and match arm beyond the first. Match guards contribute through their expressions. Unconditional loop adds no independent exit path. |
| Cognitive | Each if, while, for, loop or match adds 1 plus enclosing control depth; each else, logical operator, `?`, return, break and continue adds 1. Else-if follows the AST and counts as nested. |
| Control depth | Maximum nesting of if/while/for/loop/match expressions, excluding ordinary blocks and method wrappers. |
| Mutation | Each initialized let binding, assignment and compound assignment adds 1. An initialized destructuring binding counts once. Mutations through calls, aliases and interior mutability cannot be inferred. |
| Halstead volume | `(operator occurrences + operand occurrences) * log2(unique operators + unique operands)`. Empty vocabulary yields 0. Punctuation characters, group opening/closing delimiters and the keyword set in `lexical` are operators; other identifiers and literals are operands. Multi-character punctuation is counted by character. Comments/whitespace are ignored; doc attributes remain syntax. |
| Token count | Same lexical occurrence count as Halstead, separate from the composite. This is Rust lexical count, not BPE length. |
| AST node count | Diagnostic count of visited items, expressions, patterns, types and local bindings. This explicitly defined subset avoids dependence on every auxiliary syn node. |
| Expression depth | Maximum nested expression visits, including block/call expressions. |
| Dependency count | Number of syntactic call and method-call sites, including repeated calls. Does not claim external dependency resolution. |

Nested bodies and closures contribute to their containing cached unit. Structs
have no function baseline and therefore CC zero. These tiers measure relative
structural complexity, with lexical volume partly sensitive to size; they do
not establish LLM difficulty.

Unexpanded macros and verbatim expressions make CC, cognitive, control depth,
mutation and calls unavailable (`null`). Lexical and diagnostic counts remain
available. Such samples go to `unscored.bin`, never a fabricated zero or a
renormalized partial score. Parse errors/unsupported languages have null
metrics and go to `failed.bin`. Unsupported-language status is also available
through the analyzer API; the cache CLI accepts Rust only.

## Calibration and tie policy

Only successful, non-excluded, complete samples enter calibration. For each
primary metric, `P(x) = (count(reference < x) + count(reference <= x)) / (2N)`.
This midrank policy gives identical values identical ranks; out-of-range
values clamp to 0 or 1. The weighted score is in [0,1]. Reference composite
scores determine nearest-rank quartiles at `ceil(q*N)-1`. Scores equal to a
boundary go into the lower tier. Repeated boundaries can leave tiers empty;
ties always stay together. This first calibration is Rust specific.

Default preprocessing excludes path components `vendor`, `vendored`,
`generated`, samples containing `@generated` or `automatically generated`,
and samples with lines longer than 2000 characters (a minification heuristic).
These heuristics cannot detect every generated file, especially when its
header was removed during earlier chunking. Configure them during calibration
with `--preprocessing path.json`, for example:

```json
{
  "excluded_path_components": ["vendor", "generated"],
  "generated_markers": ["@generated"],
  "max_line_length": 2000
}
```

Use `null` to disable the line length rule. Rules are frozen into the reference
and reused during sorting. Excluded samples are preserved in `excluded.bin`.

## Output and curriculum use

Inspect actual bucket contents without rerunning the sorter:

```powershell
cargo run --no-default-features --bin view_code_bucket -- --bucket training-cache/code_buckets/tier_1_low.bin --limit 5
cargo run --no-default-features --bin view_code_bucket -- --bucket training-cache/code_buckets/tier_4_high.bin --offset 20 --limit 5
```

The viewer prints source code, source paths and locations. `--offset` counts
from zero; displayed sample numbers count from one. Use `--limit 0` for header
information only, or `--tokenizer path` for a different matching tokenizer.
Viewing source validates and loads the selected bucket into memory. Metadata
inspection alone does not load its samples or tokenizer.

The output contains `tier_1_low.bin`, `tier_2_mild.bin`, `tier_3_moderate.bin`,
`tier_4_high.bin`, and the separate excluded/unscored/failed caches when
nonempty. Empty caches are omitted because the training loader rejects empty
datasets; `report.json` still lists their zero counts. Original token tensors,
source paths, locations and order within each bucket are preserved. Output is
assembled in a temporary sibling directory and published only when complete.
A failed run can leave a hidden `.code-buckets-*` directory for diagnosis.

`records.jsonl` has one record per input sample, with input-cache hash + index
as ID, source hash/path/location, nullable metrics, status/errors, score, tier,
analyzer/parser versions and calibration identity. `report.json` contains cache
digests, counts, failure and missing-metric rates, scoring version usage,
Pearson correlations (null for constant features), and token/CC/cognitive
baseline comparisons. Baseline quartiles use the validation batch only and
never affect production boundaries. Weight sensitivity perturbs each weight
by +/-10%, renormalizes, and compares tiers at frozen boundaries. Whitespace
stability tests each scored sample with leading/trailing newlines. This is a
narrow stability probe; inspect other semantic edits on a representative
validation set before making broader stability claims. Downstream outcome
analysis is null until labeled outcomes are supplied; language comparison is
explicitly unavailable for this Rust-only implementation.

The production training configs set `curriculum` to `training-cache/code_buckets`.
`train_code --config configs/yumon-code.json --check` validates the three
eligible caches and prints the complete epoch plan without creating a GPU/model.
Epoch 1 uses low; epoch 2 low+mild; epoch 3 and later low+mild+moderate. High,
excluded, unscored and failed samples are never loaded, including for validation.
The original `cache` remains the preparation output and is ignored by curriculum
training. A missing/corrupt bucket fails explicitly without falling back to it.

One file-level validation split is computed across the eligible tier union.
All samples from those files stay held out in every epoch. Validation covers
all three allowed tiers from the start, so losses are comparable as training
expands. The optimizer persists between epochs, and linear learning-rate decay
uses the cumulative batch count of the varying epoch sizes. Resumes select the
curriculum from completed epochs, and completed epoch budgets do no further
updates. Optimizer state and within-epoch position are still not restored;
an interrupted epoch is replayed from its beginning using saved model weights.

Curriculum checkpoints have a `_curriculum_lmm_v1` suffix, keeping them separate
from earlier mixed-cache runs. Frozen reference/bucket identities, validation
fraction, seed and schedule must match on resume. `decoder-only`, `moe`, `xlstm`
and `encoder-decoder` all use the shared code trainer. The encoder-decoder
adapter uses a constant BOS encoder input and a causal source-token decoder.
Its qualitative completions use greedy decoding with the source prefix in the
decoder. The separate Pet training entry points retain their existing data flows.

Docker copies only the low, mild and moderate caches and their reference/report.
It requires curriculum mode at build-time validation and in its default command.
Set `CODE_BUCKETS`, `CODE_TOKENIZER` and `CODE_CONFIG` build arguments for alternate
local inputs while retaining the in-container paths. Mount `/workspace` on
RunPod for persistent checkpoints. A legacy local config with no `curriculum`
still uses a single cache; Docker's `--require-curriculum` rejects that mode.

Curriculum CPU checks and the explicit tiny GPU integration check:

```powershell
cargo test --no-default-features --lib brain::code_
cargo test --no-default-features --lib code_curriculum_gpu -- --ignored --nocapture --test-threads=1
```

The GPU check was run locally on WGPU for all four architectures. It covers
optimizer steps, checkpoint/tokenizer reload, resuming at epoch 3, completed
budget behavior and causal prefix predictions. On a CUDA host add
`--features cuda-training`. A full RunPod training run remains a separate
deployment check.

Calibration and sorting currently load one prepared cache into memory, like
the existing trainer, and write buckets without cloning sample tensors. Large
corpora should be profiled before establishing throughput or memory targets.
Reports and calibration artifacts live under ignored `training-cache`; archive
them alongside the training data when transferring to RunPod.

## Initial local calibration

Version `1.0.0` was calibrated from `training-cache/code.bin` with default
weights and preprocessing. The run processed 69,132 samples: 43,759 scored,
23,240 unscored due to opaque macros, 2,133 excluded, and zero parse failures.
Tier counts were 10,979 / 10,904 / 10,983 / 10,893. All 43,759 scored samples
retained their tiers in the whitespace probe. These are calibration-population
results; they do not establish generalization or downstream task difficulty.
The frozen artifact is `training-cache/code-reference-v1.json`; outputs and
the full report are in `training-cache/code_buckets`.
