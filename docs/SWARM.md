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

## Phase 0 results (2026-11-11)

Run: `cd swarm && cargo run --release --bin phase0` (about 40 s after build; writes
`swarm/out/phase0/`: `report.json`, `examples.txt`, `vocab-<set>.json`). Unit tests:
`cargo test --release` (3 tests: byte-exact rebuild, skeleton shape, target contexts).

### What the extractor does

`skeleton.rs` cuts a chunk into a tree of `Node`s (`kind`, text segments, child holes).
Holes are the outermost eligible syntax at each level, and each fill is decomposed again
(depth limit 8):

| Parent | Holes |
| --- | --- |
| item (root) | fn/method: `name`, `sig` (parameters through return type; omitted when empty), body `block`; struct: `name`, `fields`; enum: `name`, `variants` |
| `block` | `let`: `pat` + init (`literal` if it is one, else `expr`); `for`: `pat`, `iter`, `block`; `while`: `cond`, `block`; `if`/`else if`/`else`: `cond`, `block`; `match`: scrutinee `expr`, `arms`; bare/unsafe block; other statements (calls, assignments, macros with `;`): `expr` |
| `arms` | per arm: `pat`, body (`block` if braced, else `expr`/`literal`); guards stay concrete |
| `expr`, `iter`, `cond` | `call_args` for calls and method calls; control flow as in `block` |

Layout survives: hole ranges are trimmed, so skeletons keep `{`, newline, `hole!(block)`,
newline, `}`. Brace macros without `;` (`table! { .. }`) stay concrete. A worker's target is
the fill with its own holes still open (`Node::skeleton`). Its context (`Node::context`) is
the whole chunk with the target's ancestors expanded, siblings open and the target written
`hole!(TARGET kind)`. That is what a top-down, breadth-first dispatcher would hold when it
reaches the hole. Worker prompt = `BOS`, context, `"\n// fill\n"` (5 tokens), fill, `EOS`.

### Numbers (130,414 chunks from `training-cache/code.bin`, 908 of 9,086 files held out)

- Extraction: 0 chunks fail to parse, 0 byte-for-byte rebuild mismatches, 0 skeleton parse
  failures, 0 fill parse failures, over all 1,095,900 nodes in training files. (The checkers
  found and fixed four extractor/wrapper bugs on the way: or-patterns in `let` wrappers,
  typed `let` patterns, brace macros as statements, trailing `//` comments swallowing the
  wrapper's closing text.)
- Instances per kind (training files): expr 320,102; block 145,136; pat 138,203; name
  117,813; call_args 111,681; sig 82,307; cond 23,653; fields 12,810; literal 9,081; arms
  8,936; iter 8,365; **variants 0**. By depth 1-8: 313,543 / 392,314 / 193,515 / 56,364 /
  16,418 / 4,605 / 1,125 / 203.
- **No `variants` data**: the chunker behind `code.bin` only keeps structs, functions and
  methods, so no enum is in the cache. The extractor handles enums (unit test); the data
  has to come from the raw sources (card `swarm-enum-source`).
- Prompt length (context + separator + fill, tokens): p50 60-176, p99 281-538 by kind.
  Share over 512: 2.0% literal, 0.3% fields, 0.2% pat, 0.1% expr/call_args/arms, 0% others.
  Cropping is nearly unnecessary at 512; only `literal` and `fields` sometimes need it.
- Marker cost (code BPE): `hole!(kind)` is 6-10 tokens, `hole!(TARGET kind)` 11-14. A
  skeleton with ten open holes spends about 70 of 512 tokens on markers. Syn placeholders
  for list positions are cheaper for `name` (4 vs 6), `variants` (4 vs 9), `arms` (5 vs 8)
  and dearer for `sig` (11 vs 7). Decision: models see the text marker everywhere; the
  assembler swaps in placeholders (`syn_placeholder`) only to parse skeletons. A shorter
  marker (`hole!(e)`) is untested.
- Fill length (tokens, p50/p90/p99): block 17/92/201, expr 18/44/146, call_args 10/42/168,
  pat 3/10/24, name 6/16/31, sig 15/41/83, fields 44/191/394, arms 50/108/233.

### 256 output rows: coverage

Rows = EOS plus the most frequent target tokens of the role's training fills (markers
included). Share of held-out fills whose every token is inside the rows, by output rows:

| kind | 256 | 512 | 1024 | 2048 |
| --- | --- | --- | --- | --- |
| block | 88.4% | 89.8% | 92.7% | 100% |
| arms | 92.1% | 94.1% | 96.8% | 98.7% |
| literal | 70.2% | 82.0% | 90.6% | 96.4% |
| pat | 42.2% | 65.0% | 91.6% | 100% |
| iter | 38.6% | 65.0% | 91.0% | 98.0% |
| name | 27.7% | 56.9% | 99.6% | 100% |
| sig | 25.0% | 39.8% | 77.8% | 99.9% |
| call_args | 19.7% | 37.2% | 69.7% | 100% |
| expr | 15.6% | 34.2% | 73.4% | 100% |
| cond | 12.4% | 36.1% | 78.9% | 96.8% |
| fields | 2.3% | 5.7% | 24.1% | 99.8% |

(The 2048 column is below 100% where a held-out token never appears in training fills.)
At the token level 256 rows cover 75-82% of target tokens for value roles and 96-99% for
`block` and `arms`. So the 256-row plan holds for flow roles (`block`, `arms`) and `literal`;
`name`, `pat`, `iter` need about 1024; `expr`, `call_args`, `cond`, `sig` need 1024-2048 or a
copy head; `fields` needs the full table. Phase 2 sets `out_vocab` per role from this table
(`roles/*.json` has the field; default 256). Two role sets exist: `kinds` (one role per
kind, 12) and `coarse` (5); 15- and 40-role sets need finer kinds (phase 5).

### Build time (Windows, release)

First build of the `swarm/` crate: 3m12s dev, about 3 minutes again for release (all
dependencies, including burn/cubecl/wgpu through the `yumon-pet` path dependency). After
that, editing swarm code rebuilds only swarm: 5.1 s for a bin, 8.8 s for the lib (dominated
by linking the yumon-pet rlib). A bin inside yumon-pet (`view_code_bucket`,
`--no-default-features`) rebuilt in 2.6 s, but its first build also took 3m17s. So the
separate crate does not beat a bin in yumon-pet on edit cycles (about 5-9 s vs 3 s); it
does keep swarm edits from touching yumon-pet. Decision: keep `swarm/`.

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
    skeleton.rs   hole markers, extraction (syn), splicing, parse checks
    roles.rs      role config (roles/*.json)
    dataset.rs    code.bin loader, per-role instances, held-out file split
    swarm.rs      dispatcher, queue, verifier, oracle mode (phase 1, not written)
  src/bin/        phase0, show_failures (later: oracle, train_role, run_swarm)
```

## Open questions

- Markers for field and arm lists: settled in phase 0 (text markers everywhere, syn
  placeholders only for parsing); shorter markers untested.
- Whether one shared fine-tuned trunk with per-role output rows (cheap) does as well
  as fully separate role models (the "many models" version). Worth testing in
  phase 2, since both start from the same checkpoint.
- Whether the 512-token context is enough for workers: yes for 98%+ of fills (phase 0).
- Output rows per role (see the coverage table), and whether a copy head beats more rows.
- Where `variants` data comes from (no enums in `code.bin`).
- How the planner decides that a hole kind should be filled by a written piece
  rather than a trained one (probably the dispatcher's job, not a model's).

## Log

- 2026-11-11: Plan written. `swarm/` created (README only). Nothing built or trained.
- 2026-11-11: Phase 0 done. `swarm/` crate (skeleton, roles, dataset, `phase0`,
  `show_failures`), `roles/kinds.json` and `roles/coarse.json`. 130k chunks extract with 0
  failures and byte-exact rebuilds; coverage, prompt lengths, marker costs and build times
  above. Gaps: no `variants` data, 256 rows too few for most value roles. Nothing trained.
