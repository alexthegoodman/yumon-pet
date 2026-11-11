# Yumon Swarm: tiny specialist models writing Rust together

Experiment: write Rust with tens or hundreds of very small decoder models, each
doing one narrow job, instead of one model doing everything. Planners decide what
the code looks like but never write the contents; workers fill one kind of slot
each (loop headers, struct fields, match arms, function signatures, colors, names).
Written code (and maybe WIR) handles everything that has an exact answer. There is
no target for speed or quality. The point is to see what happens and to measure
each piece on its own.

Code lives in `swarm/` (see "Layout"). This doc is the canonical plan and log.

## The format: skeletons with typed holes

Models don't pass free text to each other. They pass Rust skeletons, where every
unfinished part is a typed hole:

```rust
/// Sums the visible widths of the given rows.
fn total_width(rows: &[Row], gap: u32) -> u32 {
    let mut total = hole!(expr);
    for hole!(pat) in hole!(iter) {
        hole!(block)
    }
    total
}
```

- A **planner** writes a skeleton. It decides structure and never fills a hole.
- A **worker** gets the skeleton with one hole marked as the target and writes
  only that hole's contents. A fill may contain new holes (a `block` fill can hold
  `hole!(expr)`s), and those go back to the queue. That is what gives the org its
  levels. A depth limit stops runaway recursion.
- **Written glue** routes holes to roles, splices fills back in, and checks every
  fill with `syn` (and the finished file with `cargo check`). A fill that doesn't
  parse is retried or sent back up as a rejected hole.

Hole markers are plain text that the 2,048-token code BPE already encodes, so the
base model needs no new embedding rows. `hole!(kind)` is a valid macro in
expression, statement, pattern, type and item position, so `syn` parses those
skeletons as they are. Field lists and match-arm lists don't allow macros there.
Phase 0 picks a marker for those (for example a placeholder field or arm the
assembler recognises) and measures how many tokens each marker costs.

### Why this format

Training data costs nothing. Every parsed chunk in `training-cache/code.bin` can be
cut into a skeleton plus its fills with `syn`. That gives every role exact
`(context, hole) -> fill` pairs, and gives planners `doc comment -> skeleton` pairs
from `training-cache/documented/`. No captioning or synthetic generation is needed.

## Input

The task is a `///` doc comment, as in the documented tiers. Kilo Code (canonical
registers) is not used: workers see and write real identifiers in the code BPE.

## The org

| Level | Role | Gets | Writes |
| --- | --- | --- | --- |
| L0 | architect | doc comment for a module or task | list of items with kinds (struct, enum, fn, impl) and one-line docs |
| L1 | item planner | an item's doc comment | signature + body skeleton |
| L2 | block planner | a `block` hole | statements with holes |
| worker | `sig`, `fields`, `variants`, `pat`, `iter`, `expr`, `call_args`, `arms`, `literal`, `color`, `name`, ... | skeleton with one target hole | that hole's contents |

How narrow each job should be is the main question, so the role set is a config
file (`swarm/roles/*.json`), not code. The same extraction produces a 5-role,
15-role or 40-role split, and they can be compared. "Hundreds of models" means many
instances of each role running on different holes at once, plus several instances
sampling the same hole while a verifier picks the winner.

## Base model: yumon-code-dec-sugar-maniac

Every specialist starts from the same pretrained decoder-only checkpoint:
`checkpoints/yumon-code-dec-sugar-maniac/code_32h_4l_2a_128ff_512len_b16_DecoderOnly_f32_curriculum_documented_tiers_v3_tok64_jl_ffn16_at0_tied/`

| | |
| --- | --- |
| shape | 4 layers, width 32, 2 heads, ff 128, context 512 |
| embedding | tied, 2,048 x 64 table, learned 64 -> 32 projection |
| JL | `ffn:16@0` (frozen map on every MLP input) |
| size | `model.bin` 747,754 bytes, about 187k f32 params |
| tokenizer | `yumon_code_bpe` (ByteLevel BPE, 2,048 tokens, specials PAD BOS EOS UNK) |
| training | 10 epochs on the documented tiers (10,353 samples), val loss 3.327, val top-3 49.6% |

The embedding table is about 131k of the 187k parameters (70%).

### 256-token roles

Each role keeps the full 2,048-token input vocab, so it can read any context,
but its output is limited to the role's own 256 tokens. Because the head is tied,
that's just the matching 256 rows of the embedding table. Logits are only computed
for those rows, and the pretrained weights are used unchanged. A role checkpoint
can then store only its fine-tuned weights plus the 256 ids, or the full table if
the role also tunes its input embeddings.

A role's 256 ids come from its training fills (most frequent tokens, plus EOS and
any hole markers). Phase 0 measures coverage: the share of the role's fills whose
BPE encoding uses only those ids. Fills outside it count as out of role, not as
errors. Roles with poor coverage (`name` is the obvious one) either get a larger
output vocab or a copy head that points at identifiers in the context. That's a
per-role decision based on the coverage numbers.

### Worker prompt

```
<BOS> /// doc comment
<skeleton with the target hole written as hole!(TARGET kind) and the rest as hole!(kind)>
<SEP-like marker> fill <EOS>
```

Loss counts only the fill tokens. The decoder trainer has no prompt loss mask
today (no mask in `src/brain/`), so phase 2 adds one. Prompts longer than 512
tokens are cropped around the target hole. The crop window is part of the role
config.

## Written vs trained

| Piece | How |
| --- | --- |
| dispatcher, queue, scheduler, assembler, depth limit | written (Rust) |
| `syn` check per fill, `cargo check` per file | written |
| "is this name already real?" | written |
| match arms from a known enum definition | written; WIR candidate |
| struct literal fields from a known struct definition | written; WIR candidate |
| color conversions (hex, rgb, hsl, named) | written; good WIR demo |
| bracket balancing, legal-next-token mask | written; WIR 0.3 attention candidate (Dyck, see `wir-attention-more-primitives`) |
| architect, item and block planners | trained |
| `expr`, `iter`, `call_args`, `block` bodies, `name` | trained |

Everything written goes in plain Rust first. Porting one or two pieces to WIR (the
token mask is the most interesting) comes later, as a "some members of the swarm
were never trained" variant. WIR is never a dependency.

## Phases

Each phase ends with numbers that can be checked on their own.

0. **Extraction, no training.** `syn` hole extractor over `code.bin`, skeleton
   format and markers, role config, per-role datasets with counts, per-role
   256-token coverage, prompt length distribution. Also measure how long the
   `swarm/` crate takes to build (see "Layout").
1. **Oracle swarm.** The full pipeline (dispatcher, splicing, `syn` and
   `cargo check`) runs with ground-truth fills, checking that held-out functions
   rebuild byte-for-byte. Then trained roles are swapped in one at a time, so each
   role's effect on parse rate and check rate is measured with everything else
   perfect.
2. **Workers.** Fine-tune sugar-maniac per role with the prompt mask and the 256-row
   output. Per role: fill exact-match, `syn` parse rate, top-3, plus parse rate when
   it alone is swapped into the oracle swarm.
3. **Planners.** L1 and L2 first (structure from the doc comment), then L0.
4. **Full swarm** on held-out documented functions. Metrics: share of holes that
   parse, share of files passing `cargo check`, fills and retries per finished
   file. Baseline: one decoder with about the same total parameters as the active
   roles (sugar-maniac fine-tuned on the whole task), on the same inputs.
5. **Granularity sweep.** 5 vs 15 vs 40 roles, same data and budget. Plus a swarm
   viewer page: the org chart with holes moving through it.
6. **RL against real builds.** Assembled candidates go into a scratch crate and run
   through `cargo check --message-format=json`. Each error has a code (E0308, E0425,
   ...) and a span, and the assembler knows which hole produced which bytes, so an
   error is charged to the role (and hole) that wrote it, not to the whole file.
   Rewards: parse, check, and later tests passing. Several candidates share one
   `cargo check` as separate modules to spread the cost. Error codes can also act as
   feedback text for a "repair" role.

## Layout

`swarm/` is a separate crate with its own `[workspace]` (like `wir/`). It depends
on the main crate by path (`yumon-pet = { path = "..", default-features = false }`),
so it uses `YumonDecBrain`, the tokenizer and the training loop directly.
`target-dir` points at `../target` to reuse built dependencies. The hope is that
editing swarm code only rebuilds the small swarm crate. Phase 0 measures this
against a bin inside yumon-pet. If it doesn't help, the code still lives in
`swarm/` and bins move into the main crate.

```
swarm/
  Cargo.toml
  roles/        role configs (job, hole kinds, crop window, output vocab size)
  src/
    skeleton.rs   hole markers, extraction, splicing
    roles.rs      role config, vocab selection, coverage
    dataset.rs    per-role samples from code.bin / documented tiers
    swarm.rs      dispatcher, queue, verifier, oracle mode
  src/bin/        extract, coverage, oracle, train_role, run_swarm
```

## Open questions

- Markers for field and arm lists (phase 0).
- Whether one shared fine-tuned trunk with per-role output rows (cheap) does as well
  as fully separate role models (the "many models" version). Worth testing in
  phase 2, since both start from the same checkpoint.
- Whether the 512-token context is enough for workers, given cropping.
- How the planner decides that a hole kind should be filled by a written piece
  rather than a trained one (probably the dispatcher's job, not a model's).

## Log

- 2026-11-11: Plan written. `swarm/` created (README only). Nothing built or trained.
