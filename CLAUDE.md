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

KV cache and GQA (2026-10-14): `decoder_model.rs` has `KvCache`, `cached_attention`
(prompt through flash, later steps grouped matmuls over the cache) and
`generate_causal_cached`; MoE and dense `generate_unmasked_parsed` use it
(`forward_cached`). MoE cached decoding keeps sparse routing and exit gates, and
exited rows still append frozen K/V. Optional `kv_heads` (CodeConfig, MoE only; run
name `_gqa<n>`, metadata `kv_heads`) shrinks k/v projections; training repeats K/V
for flash. Tests check cached logits = full forward (MHA, GQA, depth, MQA, dense).
Intel UHD 770, context ~1550: lite 196 -> 25 ms/token, mini-depth 1.23 s -> 75
ms/token. GQA gave no measurable speedup at these sizes. Masking all experts on
device to avoid routing reads was tried and was slower (more kernel launches).

Pretraining floor (2026-10-14): `pretrain_min_tokens` packs `pretrain_code` items
per file in source order (blank-line join) until a block reaches the floor (max
context - 1). A file's short remainder merges into its last block or is dropped.
Packing is done at load (`pack_min_tokens` in code_curriculum.rs). Run name
`_code_pretrain_v1_min<n>`. moe-lite: 128 -> 89,508 blocks (mean 318, 0.2% of tokens
dropped); 256 gives 60,474 blocks (mean 470, 0.7% dropped). Tiers are unaffected.


Optimizers (2026-11-06): optional `optimizer` config section (`optimizers.rs`,
docs/OPTIMIZERS.md), MoE-only; absent = AdamW. `slbfgs`: per-tensor stochastic
L-BFGS (history pairs, Powell damping, H0 = gamma I). No Adam and no gradient clip: the
step is step_size * (lr / first_lr) * H m, with a per-tensor RMS cap `max_step` as the
trust region (2026-11-06; the Adam-grafted version was removed);
`curvature: "probe"` re-runs the same batch with the same dropout seed every
`interval` steps (`TrainOptimizer::probe` in `train_causal_lm`; not for DPO), `"ema"`
uses EMA iterates/momentum. `fsecant`: momentum-only factored row/column secant
curvature with Sophia-style clipped steps. Run names gain `_opt_...`; stats go to
`optimizer.jsonl`. Optimizer state is not checkpointed. Configs:
`yumon-moe-base-pretrain-{slbfgs,slbfgs-ema,fsecant}.json`.

Parameter groups (2026-11-06): optional MoE `router` / `experts` sections
(`ParamGroupConfig`: first_lr, last_lr, weight_decay, init_std) each get their own
optimizer instance (`TrainOptimizer::add_group` / `step_groups`; gradients split by
`split_grads` via `router_modules()` / `expert_modules()`, shared experts count as
experts). `token_embed_dim` factorizes the embedding (`embed_projections`: in, plus out
when tied); `embed_dim` stays the model width. Record layout before this is
`LegacyMoeBrainV5`. Run names gain `_router_...`, `_experts_...`, `_tok<n>`. Unit tests
and the GPU all-architectures test (now with a grouped, factorized MoE) pass; no real
runs yet.

Inner-epoch curriculum (2026-11-06): `cache_yumon_talk` drops repetitive samples
(`--max-repetition`, default 0.5 = share of repeated token 4-grams), then sorts
easiest first (`--order difficulty`, the default; `original` keeps input order) by
the weighted mean of percentile ranks of length, word rarity (mean unigram
surprisal) and complexity (clause punctuation + slot/name/num tokens),
`--sort-weights length,rarity,complexity` (default 1,1,1). DPO pairs stay
together. `sample_order.rs` holds the scoring. The cache header gets
`order: "difficulty-v1"`, and full-cache configs then train in cache order every
epoch with no shuffle (`PreparedCode.shuffle`); config `"shuffle": true/false`
overrides. Rebuild caches to use it.

WIR v0 (2026-11-06): `wir/` is a standalone crate (own `[workspace]`, serde_json
only, not part of the yumon-pet build) with the reference compiler and decompiler
for docs/WIR.md: WIR source <-> one-hidden-layer ReLU weights (`wir compile`,
`decompile`, `canon`, `fmt`, `eval`). Weights use nn.Linear layout; .npz (W1 b1 W2
b2), .safetensors (nn.Sequential keys 0.weight..2.bias) and .json; `--keys` and
`--transpose` read slices of other checkpoints. Canonical form uses max|w_i| = 1
and axis-pinned points so printed text is a parse fixed point (see the
implementation notes at the end of WIR.md). 51 tests pass (`cargo test` in wir/,
~20 s) incl. the 1,000-network fuzz; `wir/scripts/interop.py` cross-checks NumPy
and PyTorch via the python-train uv env. Repo checkpoints (Burn .bin, SwiGLU) are
not yet decompilable.

WIR on real checkpoints (2026-11-07): see docs/WIR.md "Compact syntax..." and "Real
checkpoints". wir now has vector literals (`[..] / 990`, integers over a 2-sig-digit
scale), `crease: ramp along V above E`, SwiGLU `gate: silu along A, times along U`
(weights W3/b3), vector readouts `[..] . (c1..c64)`, shaped inputs `R64x3x3` with
`patch` directions (`--kernel-terms N` lossy), box domains `in [lo, hi]` and
`decompile --domain probe.json`, and `wir verify`. Compact text for D > 3 or gates.
Colorizer head (`export_wir_colorizer`, `ColorizerModel::features`/`head()`): 674k
-> 79k tokens, 2.6e-4 relative error vs the model's logits. Sunrise
(`wir_decoder record|eval`, `YumonDecBrain::forward_probe`/`mlp_weights`, `WirMlp`):
each SwiGLU layer is 512 gates (~407k tokens); all six WIR-compiled layers in the
model change loss by < 1e-4 with identical argmax tokens. Tokens stay ~2 per weight:
FFT/low-rank/pruning of trained weights tested poorly. Attention/norms are not WIR.

WIR codebook vectors (2026-11-07): Qwen splits numbers into single digits, so compact
text was ~4 Qwen tokens per weight (sunrise layer ~745k Qwen tokens). `codebook g4`
plus `{word ...} / q` vectors (wir/src/codebook.rs, wir/codebooks/g4.txt and g2.txt
from wir/scripts/make_codebook.py) write 4 (g4) or 2 (g2) weights per single-token word;
entries are Gaussian k-means centroids, shared by every model. `wir decompile --codebook
g4 --calibrate probe.json` rounds GPTQ style. A last word may overhang a vector;
`output y1..y128: R1` ranges. Sunrise: 65k Qwen tokens per layer with g4 (+0.019 loss,
98.7% top-1 with all six layers), 114k with g2 (+0.005, 99.4%); colorizer head 19.7k.
`wir_decoder eval --offset N` scores on samples other than the probe's. `canon` expands
coded text back to numbers. 68 wir tests.

WIR distillation (2026-11-07): see docs/WIR.md "Function-space distillation".
`wir/scripts/distill.py` fits an h-gate SwiGLU program to a layer's recorded
inputs/outputs (MSE; `--init top` starts from the original's strongest units) and
writes a wir weights JSON for `decompile --codebook g4`. Sunrise at 32 gates + g4:
8.5k Qwen tokens per layer, 92.1% top-1 with all six layers replaced (16 gates: 83%).
`wir_decoder record --mlp l=path` takes probe inputs from the model with those MLPs
replaced (outputs stay original); sequential fitting with it did not help top-1.

WIR 0.2 (2026-11-07): `wir 0.2` programs (`wir/src/deep.rs`, docs/WIR.md "WIR 0.2"):
named scalar inputs with intervals, `let`/`output` affine expressions, combinators
(relu abs max min clip step gt ge lt le eq and or not select) lowered exactly to
relu + affine nodes. Intervals give select's big-M and drop relus that cannot bend
(valid only inside the declared domains). The compiler picks depth and adds carry
units; deep weights are .json `layers` or nn.Sequential safetensors. Any ReLU MLP
decompiles to flat 0.2 (`let hK_U = relu(...)`, rationals up to /1000).
`wir/src/games.rs` + `wir pong [--variants N --out DIR]` generate Pong step programs
checked against a simulator; `examples/pong.wir` is the generator default (4 hidden
layers, 1,084 params, all 604,800 states exact, 1,160 Qwen tokens; 42 of 64 units
are carries). 81 wir tests.

WIR grid programs and pathfinding (2026-11-07): see docs/WIR.md "Grid programs and
pathfinding" and "Written vs learned planners". `input d: grid in [lo, hi] pad v` makes a
0.2 program one cell's rule; inputs are read at offsets (`d[-1, 0]`), each read is a tap
(scalar input of the cell network), and the cell network is a Conv2d stack
(`StencilWeights` in deep.rs: apply, conv_layers/from_conv, save/load_stencil,
decompile_stencil; safetensors carry wir.kind = stencil and per-channel pads). Relu units
now normalize to smallest coefficient 1, so integer programs give integer weights (exact
in f32). `wir path` (games.rs: path_program, path_cell/path_step, path_distances =
Dijkstra, path_check, path_all_mazes, path_flat_program) writes one value-iteration step:
default 4-move, ceiling 64 is [6, 5, 2], 218 conv weights (flat 6 x 6 MLP: 55,958), exact
on all 71,402,500 cell inputs and all 1,048,576 4x4 mazes; 8-move and terrain variants
too. `wir/scripts/path_planner.py` (check/train/eval/analyze) trains the same conv shape
from scratch: the 6+5+2 shape never learned it (0/6), 32+32 one-step training reaches
99.8% steps exact but only 71% exact cells on 32 x 32, end-to-end training learns a
48-step transient. Runs and summary.json in tmp/wir-path. Demo artifact "WIR Pathfinder":
https://claude.ai/artifact/RSPLyZ7kbA5coemgbPfwHQ (private). 92 wir tests.

WIR cellular water (2026-11-07): see docs/WIR.md "Cellular water and conservation proofs".
Grid programs can read a `let` at an offset (`fall[-1, 0]` relowers the let's text with
every read moved; Scope.shift / shifted_let in deep.rs), so two-pass rules stay one conv
stack. `conserve out = channel` (deep::prove_conservation, DeepProgram.conserves) proves
the grid sum is unchanged: node shapes up to translation must cancel, and cells near each
corner, rebuilt symbolically with pads, must add exactly 0; failure is a compile error.
`wir fluid` (src/fluid.rs: fluid_program, fluid_cell, fluid_step, fluid_check) writes
`examples/fluid.wir`: gravity, then alternating left/right pairs levelling by half the
difference; channels m (0..16), wall, pair. C 16: 17 taps, [15, 12, 70, 8], 2,108 integer
conv weights; exact on all 5.04M cell inputs at C 2, 20M sampled at C 16, 30k random grid
steps with mass unchanged. CPU bench 256x256: network 26 ms/step vs plain rule 1.8 ms; GPU
not measured. Demo artifact "WIR Water": https://claude.ai/artifact/W7gEhnMaKSc9P4WZ4B97Qs
(private). 97 wir tests.

WIR swarms and a shared world (2026-11-07): see docs/WIR.md "Swarms and a shared world".
`wir swarm` (src/swarm.rs) writes two programs per species into `examples/`: `*_sense.wir`, a
5 x 5 grid program run only at agents' cells (a/n/t/b/g senses: alignment by heading,
cohesion and the other species by pinwheel sector of 3 cells, blocked directions with no
corner cutting, goal gradient from `dist`), and `*_brain.wir`, a 0.2 utility-AI program
scoring 8 headings (one-hot out; blocked penalty = score span + 1, so an open heading always
wins). Keys are written `relu(lin) - relu(-lin)` and the max is a balanced tree, or the
compiler carries every input (9 layers, 34.7k params before, 4.2k after). Host (`SwarmWorld::step`):
per-heading counts capped at 3, sub-cell positions (16 per cell), at most `ROOM` = 3 of a
species per cell. `swarm_check`: 400k inputs and 384k lockstep decisions exact; local order
0.85 (fish) / 0.88 (birds) vs 0.43 / 0.61 with align 0. Demo artifact "WIR Tidepool"
(https://claude.ai/artifact/4pq3RpGc1GijfbYMbCfAyC, private) runs water, path (food goal,
dry = wall) and the four swarm networks in one 40 x 72 grid. 102 wir tests.

WIR tactical enemies (2026-11-07): see docs/WIR.md "Tactical enemies and response curves".
`a * b` in 0.2 now compiles when one factor has at most three integer values (`Graph::mul_small`,
4 relus, no gate bits). `wir tactics` (src/tactics.rs) writes `examples/tactics_sense.wir` and
`{grunt,rusher,sniper}_brain.wir`: response curves (fear from hp and empty ammo, approach
= clip(dp - range - fear), cover, peek) times field gradients graded against the best
neighbour, inertia on the current option, blocked headings lose to holding. Shared fields are
the 8-move path step (dp with ally cells costing 3, dc to cells the player cannot see); the
host traces sight and moves. Brains 13,888 / 7,644 params (6 layers; the first version was
93k / 10 layers of carries). 800k inputs and 4,800 lockstep ticks exact. Fear: 85-95% of
seen hurt/empty grunt and sniper decisions step toward cover (vs 16-18% without); inertia:
0.6% step-and-reverse vs 7.4%; spacing and ally cost did not produce pincers. `--export`
writes weights for a page; demo artifact "WIR Skirmish"
(https://claude.ai/artifact/CvH96ywotgdD56qR2gCtxD, private). 106 wir tests.

Yumon Ant (2026-11-07): see docs/WIR.md "Yumon Ant: a colony of networks". `wir ant`
(wir/src/ant.rs) writes `examples/ant_sense.wir` (grid program: 5 field grades over
dn/df/dg/dm/dh, blocked = no footing, fear grade, at nest/dump, smell bits, next-to
bits) and `examples/ant_brain.wir` (drives with role thresholds r0..r2 and age, task =
argmax drive, heading along that task's field, action bits pick/dig2/grip/drop/eat,
one-hot emote em0..em6). 4-move path ceiling 120. 200k inputs each exact; `--export`
adds 256 sense/brain test vectors for the page's JS rule port. The colony host lives
in `wir/demos/ant/` (ant_core.js, ant_ui.js, template, build.py, harness.js): side view,
32-column chunks, heavy items move only with weight-many grippers, queen/eggs/hunger/age,
sand kept out of planned tunnels. Headless 10k ticks: 846k decisions, 71M field cells,
0 differences; pop 70-130; 3.7 ms/tick (i5-12500, Node 24). Thinking every Nth tick
starved the colony, so the page thinks every tick. Demo artifact "Yumon Ant"
(https://claude.ai/artifact/9EEF76QG5QgoLB1NfoPhsD, private). `tests/ant.rs`: 3 tests.

Yumon Park (2026-11-08): see docs/WIR.md "Yumon Park: an ecosystem that learns". `wir park`
(wir/src/park.rs) writes `examples/park_brain.wir`, one 0.2 utility brain for every species
(20 inputs; keys for eat hunt flee hide mate sleep flock roam; hidden [34, 24, 21], 2,255
dyadic weights, exact on 200k inputs). The host is `wir/demos/park/` (park_core.js, park_ui.js,
template, build.py, harness.js `TICKS learn|nolearn|mixed SEED JSON`, env PARK_TUNE overrides
tuning): 17 capabilities on 10 limbs, 30 species from permutations (half plant eaters),
four bands, genes blended from both parents with one mutation, softmax(keys/4) sampling, per-Yumon
REINFORCE at each sleep on a homeostatic reward, anchored to birth weights; children copy the
mother's weights, death drops them. Untrained brains: 0 of 4.6M decisions off the rule. Learning
changes ~18% of trained decisions but did not help survival (mixed runs: learners ended
41/50/49%; at lr 0.01, 10-29%). Demo artifact "Yumon Park" (https://claude.ai/artifact/8dGTNMcvmx2XiVZFr735TN, private). `tests/park.rs`: 2 tests.

