# wir

Reference compiler and decompiler for **WIR v0** (Weight-Space Intermediate
Representation). The language spec and roadmap are in
[`../docs/WIR.md`](../docs/WIR.md).

A WIR program describes a one-hidden-layer ReLU network as geometry: each
hidden unit is a crease (a hyperplane, an orientation and a slope fixed by a
point value). The compiler turns that into weights; the decompiler turns any
one-hidden-layer ReLU network back into canonical WIR.

Standalone crate (its own `[workspace]`), so it builds in seconds without the
yumon-pet dependency tree. The only dependency is `serde_json`.

## CLI

```
cargo run -- compile examples/xor.wir -o xor.safetensors   # cost report, asserts, weights
cargo run -- decompile xor.safetensors                     # canonical WIR
cargo run -- decompile net.npz --exact --assert-at "0,0;1,1"
cargo run -- canon examples/xor_solve.wir                  # what decompile would print
cargo run -- fmt examples/tent.wir                         # resolved program, names kept
cargo run -- eval examples/tent.wir 0.25 "0.5;2"
cargo run -- verify net.wir probe.json --tol 1e-4         # vs reference inputs/outputs
```

`verify` takes a JSON file with `inputs` [N][D] and `outputs` [N][O] (for example
activations and logits recorded from the source model) and reports max and relative
error and argmax agreement. Real Yumon Pet checkpoints (the CNN colorizer head and
the sunrise decoder's SwiGLU MLPs) are the worked examples: see "Real checkpoints"
in `../docs/WIR.md`.

`decompile` options: `--keys W1,b1,W2,b2[,W3,b3]` for tensors with other names
(slices of larger checkpoints), `--transpose` for `[in, out]` storage (Burn
`Linear`), `--input`/`--outputs` to name things, `--exact` for lossless numbers,
`--digits N` for vector precision (default 3), `--domain probe.json` to drop
creases never active on the probe's inputs, `--shape 64x3x3` for patch directions,
`--kernel-terms N` to keep only the first N patch components (lossy), and
`--codebook g4|g2` to write vectors as codebook words (lossy; rounding is
error-aware on `--calibrate probe.json`, default the `--domain` probe).

## Compact syntax

D > 3 or gates switch the printer to the compact form (all of it is accepted in
any program):

```
input  f: R64x3x3 in [-4, 4]                                   # optional shape and box domain
crease c1: ramp along [990 -12 455 ...] / 990 above 0.0123     # relu(V . x - E)
gate   c2: silu along [...] / 2.1e3, times along [...] / 990   # SwiGLU unit
crease c3: ramp along patch level [...] / 1400 dx [...] / 990  # 3x3 basis per channel
y = [-115 -155 ...] / 1e4 . (c1..c64) + 0.0817                 # vector readout
```

Vectors are integers over a two-significant-digit scale, so the largest entry
lands at 900-999 (one token per entry). See `../docs/WIR.md` for semantics and the
canonical form of gates.

## Codebook vectors

```
wir 0.1
codebook g4
input  x: R128
output y1..y128: R1
gate c1: silu along {ts trend refresh ...} / 11, times along {...} / 2.5
y1 = {...} / 66 . (c1..c512)
```

Each word is a fixed group of numbers from `codebooks/g4.txt` (4,096 words x 4,
3 bits per weight) or `g2.txt` (256 x 2), divided by the scale. Words are single
Qwen2.5 tokens, so g4 text costs about a quarter of a token per weight, against
about 4 for digits under Qwen's digit-by-digit tokenizer. The entries are k-means
centroids of Gaussian samples (`scripts/make_codebook.py`), shared by every model.
A last word may overhang the vector. Sunrise: 745k -> 65k Qwen tokens per layer.

## Weight files

Layout is PyTorch `nn.Linear`: W1 `[H, D]`, b1 `[H]`, W2 `[O, H]`, b2 `[O]`.

| format | tensor names written | notes |
| --- | --- | --- |
| `.npz` | `W1 b1 W2 b2` + `wir_meta` | `np.load` works; read side needs `np.savez`, not `savez_compressed` |
| `.safetensors` | `0.weight 0.bias 2.weight 2.bias` | loads into `nn.Sequential(Linear, ReLU, Linear)` with `load_state_dict` |
| `.json` | `W1 b1 W2 b2` | human-readable |

SwiGLU weights add `W3 b3` (`w3.weight`/`w3.bias` in safetensors, which then uses
`w1.*`/`w2.*` keys). An input shape travels as `input_shape`.

All three accept any of these names on read. f64 by default, `--dtype f32` on
compile. F16/BF16 safetensors and `<f2` npy are readable.

## WIR 0.2: deep programs

A `wir 0.2` header switches to deep ReLU programs written with combinators
(`src/deep.rs`, spec in `../docs/WIR.md`):

```
wir 0.2
input bx: R1 in [1, 14]                       # named scalar inputs, optional interval
input pl, ul: R1 in [0, 9]
let hit = and(lt(bx - 1, 1), ge(bx, pl))      # relu abs max min clip step gt ge lt le eq and or not select
let sp = ul*bx                                # a * b when one factor has <= 3 integer values
output pl2 = clip(pl + ul, 0, 9)
output bx2 = select(hit, 2, bx - 1)
assert bx2(1, 0, 0) == 2                      # one value per input, in order
```

```
cargo run -- compile examples/pong.wir -o pong.safetensors   # per-layer cost report, asserts
cargo run -- eval examples/pong.wir "1,5,-1,1,4,0,0,0"
cargo run -- decompile pong.safetensors                      # flat 0.2: one `let hK_U = relu(...)` per unit
cargo run -- pong --width 20 --height 14 --paddle 4 --speed 2 -o p.wir   # generated + checked
cargo run -- pong --variants 50 --out ../tmp/wir-pong                 # batch + manifest.jsonl
```

Deep weights are `.json` (`{"format": "wir-deep-weights", inputs, outputs, layers:
[{W, b}]}`) or `.safetensors` with `nn.Sequential(Linear, ReLU, ..., Linear)` keys
`0.weight`, `2.weight`, ... and `wir.inputs` / `wir.outputs` metadata.

### Grid programs

Inputs typed `grid` make the program one cell's rule, applied at every cell. Grid
inputs are read at integer offsets (`d[-1, 0]` is the cell above) and take a `pad`
value for reads outside the grid. The compiled cell network reads every tap in its
first layer, so it is a Conv2d stack: a (2r+1) x (2r+1) kernel, then 1 x 1 kernels.

```
wir 0.2
input d: grid in [0, 64] pad 64
input wall: grid in [0, 1] pad 1
input goal: grid in [0, 1] pad 0
let best = min(min(d[-1, 0], d[1, 0]), min(d[0, -1], d[0, 1]))
output d2 = clip(1 + best + 64*wall - 129*goal, 0, 64)
```

```
cargo run --release -- path -o examples/path.wir          # one value-iteration step, checked
cargo run --release -- path --moves 8 --terrain 4 --ceiling 200 --all-mazes 4x4
cargo run --release -- path --flat 6x6                     # the same rule as a plain 0.2 MLP
cargo run --release -- path --variants 20 --out ../tmp/wir-path-variants
cargo run -- compile examples/path.wir -o path.safetensors  # Conv2d weights
cargo run -- decompile path.safetensors                     # grid text back
uv run --project ../python-train python scripts/path_planner.py check path.safetensors
```

Stencil weights are `.json` (`{"format": "wir-stencil-weights", channels, pads,
taps, outputs, layers}`, layers over the taps) or `.safetensors` with
`nn.Sequential(Conv2d, ReLU, ..., Conv2d)` keys and `wir.kind = stencil`,
`wir.inputs`, `wir.pads`, `wir.outputs` metadata. Pad each input channel with its
own value before the first convolution (`scripts/path_planner.py` does `F.pad(x - p) + p`).
`scripts/path_planner.py train` fits the same conv shape to the rule from scratch
(one step at a time, or end to end through K unrolled steps) for comparison.

Relu units are scaled so their smallest coefficient is 1, so programs with integer
coefficients compile to integer weights, exact in f32.

A grid program's `let` can also be read at an offset: `fall[-1, 0]` is the whole
`fall` expression one row up (lowered again with every read moved). Two-pass rules
stay one cell network. `conserve m2 = m` proves that the output summed over any grid
equals the input: each node shape's coefficients must cancel, and every cell near a
corner, rebuilt with the pads, must add exactly 0. A failed proof is a compile error.

```
cargo run --release -- fluid -o examples/fluid.wir --bench 256x256   # cellular water, checked
cargo run --release -- fluid --capacity 2 --local 6000000            # every cell input
```

### Swarms

`wir swarm` writes two programs per species (fish and birds): a **sense** grid program
(5 x 5 window: neighbours by heading, crowding and the other species by direction, blocked
directions, the goal gradient), run only at agents' cells, and a **brain** 0.2 program that
scores the 8 headings and outputs the best one-hot. The host counts agents into per-heading
channels and moves them; there is no neighbour search.

```
cargo run --release -- swarm --out examples        # writes {fish,bird}_{sense,brain}.wir, checked
cargo run --release -- swarm --no-align            # control: no alignment term
```

The check compares each network with the rule as plain code on random inputs, then runs
worlds with networks and rule in lockstep, agent for agent, and reports how flocked each
species ends up.

### Tactical enemies

`wir tactics` writes a **sense** grid program (field gradients graded against the best
neighbour, exposure, allies by sector, blocked headings) and one **brain** per archetype
(grunt, rusher, sniper): response curves for fear, approach, cover and peek, a score for 8
headings and holding, the best one-hot, and fire / reload bits. The host traces the
player's sight and runs two shared `wir path` fields (to the player, to cover).

```
cargo run --release -- tactics --out examples --export demo.json   # checked; --export: weights for a page
cargo run --release -- tactics --rule-only --no-fear               # controls: --no-fear --no-inertia
                                                                    # --no-spacing --no-ally-cost
```

The check compares the networks with the rule on random inputs, then runs arenas with
networks and rule in lockstep, and reports behaviour (retreat to cover when hurt or
empty, step-and-reverse moves, spread around the player).

The WIR Skirmish page lives in `demos/tactics/` (`skirmish_template.html`, the page with a
`__DATA__` slot): `python demos/tactics/build.py demo.json skirmish.html`. A built copy,
`demos/tactics/wir-skirmish.html`, opens directly in a browser.

### Yumon Ant

`wir ant` writes an ant's **sense** grid program (5 field grades, blocked headings, fear
grade, at-nest / at-dump, which fields it smells, what is next to it) and its **brain**:
drives over hunger, age, role and the colony store, the strongest drive as the task, a
heading along that task's field, five action bits (pick, dig, grip, drop, eat) and a
one-hot emote. `demos/ant/` holds the page: the colony host in plain JavaScript
(`ant_core.js`), the UI, and `build.py`, which inlines the exported weights.

```
cargo run --release -- ant --out examples --export ant_demo.json   # 200k inputs each, exact
python demos/ant/build.py ant_demo.json yumon-ant.html
node demos/ant/harness.js 8000 1 check ant_demo.json               # headless colony, network = rule
```

### Yumon Park

`wir park` writes one **brain** for every species (`examples/park_brain.wir`): response
curves over hunger, fatigue, hurt, fertility and the nearest threat / food / prey / mate /
kin, plus whether an attacker struck and the odds against it, give a key per option (eat
hunt flee hide mate sleep flock roam fight brace). `demos/park/` holds the ecosystem: 30
species from capability permutations, per-Yumon capabilities that mutate and split off new
species, genes on limbs, four bands, drawable walls, and per-Yumon learning (REINFORCE on
its own copy of the weights at each sleep, with kill and escape rewards; children copy
their mother's weights).

```
cargo run --release -- park --out examples --export park_demo.json  # 200k inputs, exact keys
python demos/park/build.py park_demo.json yumon-park.html
node demos/park/harness.js 10000 learn|nolearn|mixed 1 park_demo.json
```

### Superposition: `wir superpose-choice-layer`

For **choice programs** (features, then key groups affine in them, then the largest key of
each group wins or is sampled): squeezes the listed features through **one layer** of N
shared neurons with no training. The encoder is weighted PCA (closed form), and the decoder
is solved by one linear program (`microlp`) whose constraints are the choices: every
group's winner leads by `--margin` (default 1, which keeps compiled `gt`/`ge` bits exact),
or with `--preserve values` every key's value (for keys a host samples from). The tool
reads each key's coefficient on each feature from the program itself, writes the
bottleneck after the last feature, rewrites the keys to read the decoded features, adds
renamed copies of lets in between (other readers keep exact values), and prints a
certificate checked on every state. It fits the ant, fish/bird, grunt/rusher/sniper and
park brains. Grid programs, Pong, path and fluid are exact-value programs and are rejected
or out of scope. Background and colony results: `../docs/POLYSEMANTIC.md`, `../docs/WIR.md`
"Superposition without training".

```
node scripts/superpose/colony.js ant_demo.json - 10000 7 ant_probe.json   # record states
cargo run --release -- superpose-choice-layer examples/ant_brain.wir \
  --features home,forage,dig,dump,help,lost,hungry,workem,carrying,grip \
  --keys tkey0..tkey6 --keys ekey0..ekey6 --neurons 6 --states ant_probe.json \
  -o examples/ant_brain_choice6.wir
cargo run --release -- ant --export ant_demo.json --variant "Superposed, 6 neurons=examples/ant_brain_choice6.wir"
```

States come from `--states probe.json` (`inputs` [N][D], optional `weights` [N]) or
`--sample N` (uniform over the declared input intervals, reachable or not). `--weight
name=f` counts states where a let is on f times more, `--fidelity` (default 1e-4) keeps
keys near their values so the decoded features keep their meaning, `--digits` sets the
written precision. `wir ant` and `wir park` take `--variant label=brain.wir` (repeatable)
to add brains to the page's brain selector.

## Library

```rust
let c = wir::compile(src, Default::default())?;            // Program + Weights + cost report
let p = wir::decompile(&c.weights, &Default::default())?;  // canonical Program
let text = wir::print(&p, wir::NumFormat::CANONICAL);      // or NumFormat::EXACT
```

## Language additions and decisions beyond the spec

- `zero on plane normal V through P` plane form (besides `offset`); the printer
  uses it for D != 2.
- `p/q` number literals (`1/3`); canonical text prints snapped rationals.
- Several scalar outputs may share creases (`output u: R1`, `output v: R1`).
- **Canonical scale is the max norm**: every canonical unit has max|w_i| = 1,
  not |w| = 1. Unit normals are rarely exact decimals, so 6-digit text could
  not hold "slope 1" and the error leaked into coefficients. With max-norm,
  every canonical crease reads `equals 1`, and XOR decompiles to `y = c1 - 2*c2`.
- Points are printed on the coordinate axis where the normal is +-1, not at the
  foot of the perpendicular, so every printed number survives a parse
  unchanged (`print(parse(print(p))) == print(p)` by construction).
- Tolerances are relative with a floor of 1 (`Tolerances`): dead weight/coeff
  1e-9, merge 1e-7, sort grid 1e-6; asserts and `solve` residuals 1e-6.
- Rational snapping: within 1e-9 relative of p/q, q <= 16, |x| <= 1e6.
- `unit(x)` (0.2): always one hidden unit holding x, `relu(x - lo) + lo`, even where interval
  analysis would fold a relu away. Later layers then carry that one neuron instead of the
  values it reads (packed carries, superposition bottlenecks).

## Tests

```
cargo test                      # 118 tests, ~60 s debug (fuzz and fluid are the slow ones)
uv run --project ../python-train python scripts/interop.py   # NumPy 2.5 / PyTorch 2.14 cross-check
```

`tests/fuzz.rs` is the weights round-trip acceptance test: 1,000 random networks
(D 1-3, 1-16 units, including constant, dead, duplicated and opposite units)
through decompile, exact print and compile, each compared on 10,000 points.
`tests/decompile.rs` checks that scaling and permuting hidden units leaves the
text unchanged; `tests/roundtrip.rs` checks `decompile(compile(src)) ==
canonicalize(src)` on 600 generated programs. `tests/compact.rs` covers the compact
syntax, gates, patches and domains, including 300 random D 4-9 ReLU and SwiGLU
networks through exact and canonical compact text. `tests/codebook.rs` covers coded
vectors: parsing, bit-exact round trips, overhang and calibrated rounding.
`tests/deep.rs` covers 0.2: combinators against their definitions, layering against
the program's own graph off the integer grid, `examples/pong.wir` against an
independent simulator on 181,440 states, generated variants, and 500 random deep
networks through exact 0.2 text. `tests/path.rs` covers grid programs: the 4-neighbour
step on every cell input (ceiling 12), 4/8-neighbour and terrain variants against
Dijkstra, all 3 x 3 mazes, self-healing after the goal moves, the flat expansion, and
Conv2d / JSON files plus 100 random conv stacks through exact grid text.
`tests/fluid.rs` covers lets read at offsets, `conserve` proofs and refusals (a leak at
the edge, a missing inflow, a constant), the water step on every cell input at
capacity 2, random grids against the simulator, and a basin settling level.
`tests/swarm.rs` covers the swarm networks against the rule and in lockstep worlds, the
blocked-heading bound (a penalty one smaller fails), the pinwheel sense sectors, and
flocking with and without the alignment term. `tests/tactics.rs` covers the tactical
networks against the rule and in lockstep arenas, holding beating every blocked heading,
small-integer products (and the refusal of two wide factors), and the fear and inertia
curves changing behaviour against their controls.
`tests/ant.rs` covers the ant networks against the rule, holding beating every blocked
heading, and roles choosing the task (a digger follows the dig field, a forager food).
`tests/superpose.rs` covers `superpose-choice-layer`: a full-width layer keeping every key
value, renamed copies for lets between features and keys, the certificate, and the refusals
(a key reading a feature through a relu, a key before its feature).
