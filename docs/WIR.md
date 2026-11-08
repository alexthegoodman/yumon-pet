# WIR v0: Compiler & Decompiler Handoff

Nov 6, 2026 · @Alex Goodman

We are building a hypernetwork for Weight Space Programming. First, we need to build a compiler that transforms WIR into raw weights. Then, we need a decompiler, so we can transform raw weights into WIR. Finally, we will fine-tune Qwen on WIR.

We are doing all of this here in the Rust-based Yumon Pet repo so that we can easily test and experiment with the real variety of checkpoints we have here. I also think Rust is a good choice for creating new programming languages.

## Purpose

Build a reference compiler (WIR source → ReLU MLP weights) and a decompiler (weights → canonical WIR) for WIR v0, with round-trip tests as the acceptance bar.

WIR (Weight-Space Intermediate Representation) lets an author describe a network as geometry instead of numbers. Each first-layer ReLU unit is one hyperplane (a "crease") in input space; a WIR statement fixes its position, orientation and slope. The bet is that LLMs place points and planes reliably but cannot write raw weight matrices, so a compiler should own the arithmetic.

v0 targets tiny networks: 1–3 inputs, one hidden ReLU layer, one linear output, roughly 10–100 parameters. Depth and combinators come after the round-trip works.

## Build order

Write the compiler first, but define the canonical form before writing either tool. The decompiler is the more important deliverable; it just needs a fixed target to decompile into, and the compiler is the cheapest way to pin that target down and generate test cases.

1. **Core data model.** A `Program` holds inputs, creases, and output terms in canonical form (unit normal, offset, slope, readout coefficient). Both tools read and write this structure; the text syntax is just a view of it.
2. **Parser + compiler.** WIR text → `Program` → weight tensors (NumPy first, PyTorch export second). Asserts are checked at compile time.
3. **Printer.** `Program` → WIR text in canonical form, so `print(parse(src))` is stable.
4. **Decompiler.** Weight tensors → `Program` → WIR text, using the canonicalization rules below.
5. **Round-trip tests.** `decompile(compile(src))` must equal the canonical form of `src`, and `compile(decompile(W))` must match `W` as a function (sampled), even when the tensors differ by symmetry.
6. **Fuzzing.** Random small networks → decompile → recompile → functional equality.

## Core semantics

A v0 program denotes a function y(x) = Σ cᵢ · relu(wᵢ·x + bᵢ) + c₀. Every WIR statement describes one term geometrically; nothing in the source is a raw weight.

**Crease.** One hidden unit, read as a hyperplane plus a slope:

- Normal: n̂ = w / |w| (points toward the rising side)
- Offset: the plane is {x : n̂·x = d}, with d = −b / |w|
- Slope: s = |w|, output gained per unit distance from the plane on the rising side

**Anchoring by values at points.** A crease's slope is never written directly. The author pins the plane, then gives one point where the unit has a known value ("equals 1 at (1,0)"). The compiler derives s. LLMs place points far more reliably than they compute normalized gradients, so this is the default way to specify scale.

**Hinge (1D sugar).** In one input dimension a crease is a knot. `hinge at k turn t` means t · relu(x − k): the slope of y changes by t at x = k. `hinge left at k turn t` means t · relu(k − x). Any continuous piecewise-linear 1D function is a sum of hinges plus a constant.

**Readout.** Output terms are linear combinations of named creases (`y = a - 2*b`), or fitted by `solve` (least squares over named creases from point→value pairs).

**Assert.** `assert y(p) == v` is checked at compile time with a tolerance (default 1e-6). The decompiler never invents asserts; it can emit them on request for sample points the user supplies.

## v0 syntax

Line-oriented, `#` comments, one statement per line (a readout may continue on indented lines starting with `+` or `-`). Proposed grammar, to be refined in implementation:

```
program   := header decl* stmt*
header    := "wir" VERSION
decl      := "input" NAME ":" "R" INT | "output" NAME ":" "R" INT
stmt      := crease | readout | solve | assert

crease    := "crease" NAME ":" plane "," anchor
plane     := "zero on line" POINT "->" POINT        # 2D only
           | "zero on plane normal" VEC "offset" NUM  # any dim
           | "between" POINT "and" POINT            # bisector, rises toward 2nd
anchor    := "equals" NUM "at" POINT

hinge     := "hinge" ["left"] "at" NUM "turn" NUM    # 1D, usable inside readout
readout   := NAME "=" term (("+"|"-") term)*
term      := [NUM "*"] (NAME | hinge) | NUM
solve     := NAME "=" "solve over" "{" NAME ("," NAME)* "}" ":" pair ("," pair)*
pair      := POINT "->" NUM
assert    := "assert" NAME "(" coords ")" "==" NUM ("," ...)*
```

### Example 1: 1D tent (3 hidden units, 10 parameters)

```
wir 0.1
input  x: R1
output y: R1

y = hinge at 0.0 turn +1    # slope 0 -> +1
  + hinge at 0.5 turn -2    # slope +1 -> -1
  + hinge at 1.0 turn +1    # slope -1 -> 0

assert y(0.5) == 0.5
assert y(2.0) == 0
```

Expected weights: W1 = \[1, 1, 1\]ᵀ, b1 = \[0, −0.5, −1\], W2 = \[1, −2, 1\], b2 = 0.

### Example 2: 2D XOR (2 hidden units, 9 parameters)

```
wir 0.1
input  x: R2
output y: R1

crease a: zero on line (0,0)->(1,-1), equals 1 at (1,0)   # x1+x2
crease b: zero on line (1,0)->(0,1),  equals 1 at (1,1)   # x1+x2-1
y = a - 2*b

assert y(0,0)==0, y(1,0)==1, y(0,1)==1, y(1,1)==0
```

Expected weights: W1 = \[\[1,1\],\[1,1\]\], b1 = \[0, −1\], W2 = \[1, −2\], b2 = 0.

### Example 3: same XOR with a solved readout

```
crease a: zero on line (0,0)->(1,-1), equals 1 at (1,0)
crease b: zero on line (1,0)->(0,1),  equals 1 at (1,1)
y = solve over {a, b}: (0,0)->0, (1,0)->1, (0,1)->1, (1,1)->0
```

The author chooses geometry; the compiler fits readout coefficients (and c₀) by least squares. If the residual exceeds tolerance, compilation fails and reports it: the creases cannot express the target.

## Compilation rules

Every plane form reduces to (n̂, d) with |n̂| = 1; the anchor then fixes the sign of n̂ and the slope s. Weights follow from those three values.

**Plane forms → (n̂, d)**

- `zero on line P -> Q` (2D): direction u = Q − P, n = (−u₂, u₁), n̂ = n / |n|, d = n̂·P.
- `zero on plane normal v offset d`: n̂ = v / |v|, offset used as given (in unit-normal terms).
- `between P and Q`: n̂ = (Q − P) / |Q − P|, d = n̂·(P + Q)/2. Rising side is toward Q.

**Anchor `equals v at A` → orientation and slope**

Let δ = n̂·A − d (signed distance of A from the plane). Then:

- δ = 0 → compile error: anchor lies on the plane.
- v ≤ 0 → compile error: a unit's output is never negative and a zero anchor cannot set scale. Negative contributions belong in the readout coefficient.
- δ < 0 → flip n̂ → −n̂ and d → −d, so the anchor sits on the rising side.
- s = v / |δ|.

```latex
w = s\,\hat n, \qquad b = -s\,d
```

**Hinges (1D):** `hinge at k turn t` → hidden unit w = 1, b = −k, readout coefficient t. `hinge left at k turn t` → w = −1, b = k, coefficient t.

**Readout:** each named crease or hinge becomes one hidden unit (in declaration order) and its coefficient goes in W2. Bare constants sum into b2. A crease declared but never used in a readout is a warning and is dropped.

**Solve:** build matrix H with H\[j, i\] = relu(wᵢ·pⱼ + bᵢ) and a column of ones; solve min ‖H c − v‖² with `numpy.linalg.lstsq`. Fail if the max residual > tolerance; warn if H is rank-deficient (coefficients not unique).

**Cost report:** the compiler prints per-statement hidden units and parameters, plus totals. For an input of dimension D, each crease costs D + 2 parameters (D weights, 1 bias, 1 readout coefficient); the output adds 1 bias.

## Canonical form and decompiler

One function has many weight settings; the canonical form picks exactly one, so weights → WIR is deterministic. Two symmetries must be removed: positive scaling (relu(c·z) = c·relu(z) for c > 0, so only the product of slope and readout coefficient matters) and permutation of hidden units.

**Canonical form (v0)**

- Every crease has a unit normal and slope 1. Its scale moves into the readout: c′ = c · |w|.
- No two creases share the same oriented plane (same n̂ and d within tolerance).
- No dead units: none with |w| ≈ 0, none with c′ ≈ 0.
- Creases are sorted by n̂ (lexicographic after rounding), then by d.
- Numbers are printed at a fixed precision (proposal: 6 significant digits), after an optional snap to simple rationals.

**Decompiler steps**

1. Load W1, b1, W2, b2 and check shapes (one hidden layer, scalar or vector output).
2. Fold constant units: if |wᵢ| < ε, the unit outputs relu(bᵢ), a constant; add cᵢ · relu(bᵢ) to b2 and drop it.
3. Drop units with |cᵢ| < ε.
4. Normalize: s = |wᵢ|, n̂ = wᵢ / s, d = −bᵢ / s, c′ = cᵢ · s.
5. Merge units with matching (n̂, d) by summing c′; drop the result if it cancels to ≈ 0.
6. Sort into canonical order and name units `c1, c2, …`.
7. Emit text by input dimension:
   - 1D: w > 0 → `hinge at d turn c′`; w < 0 → `hinge left at −d turn c′` (since n̂ = −1).
   - 2D: `zero on line P -> Q` with P = d·n̂ (foot of the perpendicular from the origin) and Q = P + (−n̂₂, n̂₁), anchored `equals 1 at P + n̂`.
   - ≥3D: `zero on plane normal n̂ offset d, equals 1 at d·n̂ + n̂`.
8. Emit the readout `y = c′₁*c1 + … + b2`.

**Known gaps (record, don't solve in v0):** an opposite pair relu(z) and −relu(−z) is a hidden linear term and could print as `linear` sugar. Units that never activate on the input domain are invisible without a declared domain. Parallel creases with different offsets are distinct units and should stay distinct.

## Scope and open questions

**In v0:** ReLU only; one hidden layer; inputs of dimension 1–3 in tests (the math works for any D); scalar output (vector output is a stretch goal: one readout statement per output component, sharing creases).

**Not in v0:** depth, smooth activations, convolutions or attention, training, a hypernetwork. Smooth activations would make "crease" an approximate statement, so they wait until the exact ReLU semantics are solid.

**Planned for v0.2, depth via combinators** (built Nov 7, see "WIR 0.2" at the end): `max(f,g)`, `min(f,g)`, `clip(f,lo,hi)`, `and(...)` of half-spaces, compiled to known ReLU constructions (for example max(f,g) = g + relu(f − g), with an identity passthrough for g). The author stays in input space and the compiler chooses depth. A raw feature-space escape hatch (`in space(a,b,c): crease ...`) comes alongside.

**Open questions**

- Should inputs declare a domain (`input x: R2 in [0,1]^2`)? It would let the decompiler drop units that never activate there, and let the compiler warn about creases that miss the domain.
- How aggressive should number snapping be when decompiling (simple rationals only, or also √2-style constants)?
- Should the decompiler emit a recognizable macro (`tent`, `xor`) when a known pattern appears, or always the flat canonical form?
- Tolerances: one global ε, or relative to weight magnitudes?

## Implementation plan

Python 3.11+, NumPy as the only required dependency, PyTorch optional for export, pytest for tests. A hand-written recursive-descent parser is enough for v0; reach for Lark only if the grammar grows.

```
wir/
  model.py        # Program, Crease, Readout dataclasses (canonical data model)
  parser.py       # WIR text -> Program
  compiler.py     # Program -> (W1, b1, W2, b2); anchors, solve, asserts, cost report
  printer.py      # Program -> canonical WIR text
  decompiler.py   # weights -> Program (canonicalize)
  export.py       # .npz and optional torch.nn.Sequential
  cli.py          # wir compile file.wir | wir decompile weights.npz
examples/
  tent.wir  xor.wir  xor_solve.wir
tests/
  test_compile.py  test_decompile.py  test_roundtrip.py  test_fuzz.py
```

**Acceptance criteria**

- [x] `tent.wir` and `xor.wir` compile to exactly the expected weights in the syntax section, and all asserts pass.
- [x] `xor_solve.wir` compiles with residual < 1e-9 and reproduces the XOR truth table.
- [x] Each compile error case (anchor on plane, non-positive anchor, unsolvable `solve`, failed assert) has a test with a clear message naming the statement and line.
- [x] `print(parse(src))` is idempotent: printing twice gives identical text.
- [x] Round-trip source: `decompile(compile(src))` equals `canonicalize(src)` as text.
- [x] Round-trip weights: for 1,000 random networks (D ∈ {1,2,3}, 1–16 hidden units), `compile(decompile(W))` matches W as a function on 10,000 sampled inputs within 1e-6.
- [x] Scaling and permuting hidden units of any network leaves its decompiled text unchanged.
- [x] The CLI prints a per-statement cost report on compile.

**Milestones**

1. Data model, parser and compiler pass the tent and XOR examples.
2. Printer and canonicalization pass the idempotence test.
3. Decompiler passes both round-trip tests and the symmetry test.
4. Fuzzing and CLI polish; then open v0.2 (combinators and depth).

## Implementation notes (v0 reference, Nov 6, 2026)

The reference implementation is the Rust crate `yumon-pet/wir` (standalone, `serde_json` only), not the Python layout sketched above, to match the repo's language and keep checkpoints close. Usage and test commands are in `wir/README.md`. All eight acceptance criteria pass (51 tests); the weights round-trip worst case is 9.1e-13 over 1,000 networks x 10,000 points. NumPy 2.5.4 and PyTorch 2.14.1 read the exported `.npz`/`.safetensors` (`wir/scripts/interop.py`).

Decisions that refine the spec:

- **Canonical scale is the max norm** (max|wᵢ| = 1), not |w| = 1. Unit normals are rarely exact decimals, so 6-digit text could not hold "slope 1" and the error leaked into readout coefficients, which broke re-canonicalization. With the max norm every canonical crease reads `equals 1` and XOR decompiles to `y = c1 - 2*c2`. In 1D the two agree.
- **Printed points sit on a coordinate axis.** P is where the plane crosses the axis on which the normal is ±1, Q = P + (−m₂, m₁), and the anchor is P one step along that axis. A foot of the perpendicular drifts by a rounding error on every parse; this choice makes `print(parse(print(p))) == print(p)` hold by construction.
- **New plane form** `zero on plane normal V through P`, used by the printer for D ≠ 2 (`offset` is still accepted).
- **Number literals** may be `p/q`; canonical text snaps values within 1e-9 (relative) of p/q with q ≤ 16 and |x| ≤ 1e6.
- **Several scalar outputs** may share creases (`output u: R1`, `output v: R1`), giving W2 [O, H]. `R2` outputs are rejected with a hint.
- **Weights** use PyTorch `nn.Linear` layout. `.safetensors` uses `nn.Sequential` keys (`0.weight` ... `2.bias`), `.npz` uses `W1 b1 W2 b2`, and either set of names is accepted on read.

Open questions, as answered for v0:

- Tolerances are relative with a floor of 1: zero when |x| ≤ ε·max(1, largest |x| of its kind). Dead weight/coefficient 1e-9, merge 1e-7, sort grid 1e-6, asserts and `solve` residuals 1e-6 (`--tol`).
- Snapping covers simple rationals only (see above), not √2-style constants.
- The decompiler always emits the flat canonical form, with no macros.
- Input domains are not implemented yet; the known gaps above (hidden linear terms, units that never activate on a domain) remain.

## Compact syntax, gates, patches and domains (Nov 7, 2026)

Real checkpoints made two things clear: point-anchored text does not scale past a few dimensions (a point in R576 is 576 numbers, and the first colorizer decompile was 674k tokens by `uvx count-tokens`), and the repo's language models are SwiGLU, not ReLU. Additions, all accepted by the parser in any program:

- **Vector literals** `[-881 -415 107] / 990`: entries separated by spaces, divided by an optional scale. The printer writes integers over a two-significant-digit scale (`/ 990`, `/ 4.1e4`) chosen so the largest entry lands at 900-999: three digits of resolution relative to the vector's largest entry, and one token per entry for common tokenizers (`--digits` changes it; `--exact` lists exact floats).
- **`crease c: ramp along V [above E]`** is relu(V . x - E). It writes the slope directly, unlike the anchored forms, and is what the printer uses for D > 3.
- **`gate g: silu along A [above E], times along U [above F]`** is a SwiGLU unit silu(A . x - E) * (U . x - F). A program is all ReLU or all gates (the compiler rejects mixtures). Weights add `W3`, `b3`. Canonical form keeps A as is (silu has no scaling symmetry), scales U so its largest-magnitude entry is +1 and moves that factor and sign into the readout; gates are sorted like creases and never merged.
- **Vector readouts** `y = [..] / 1e4 . (c1..c64) + 0.1`: one coefficient per unit, with `a..b` name ranges. Used by the printer whenever the compact form is.
- **Shaped inputs** `input f: R64x3x3` (product = D). Directions over such inputs may be written as `patch level [..] dx [..] dy [..] dxx [..] dyy [..] dxy [..] dxxy [..] dxyy [..] dxxyy [..]`, one per-channel vector each: level, left-to-right and top-to-bottom slope, then curvatures. The nine components (products of (1,1,1), (-1,0,1), (1,-2,1)) are orthogonal, so every 3x3 kernel has exactly one patch form, and dropping the high ones is a low-pass (DCT-like) approximation: `wir decompile --kernel-terms N` keeps the first N (lossy). Patch text is not a fixed point under `canon` (re-normalizing the expanded kernel moves the integers); `print` itself is stable.
- **Domains.** `input x: R2 in [lo, hi]` declares a box: the compiler warns about creases never active there and the canonical form drops them. `wir decompile --domain probe.json` drops creases never active on the probe's `inputs`; the samples must cover every input the network will see (a 512-pixel probe once dropped a unit that fires elsewhere in the same image).
- `wir verify <program|weights> <probe.json>` reports max and relative RMS error and argmax agreement against recorded outputs.

Compression study (Nov 7) on the colorizer head and sunrise layer 0 at equal number budgets, scored by output change on real inputs: rounding to 3 significant digits costs 0.06-0.1%, dropping inactive units is free, 3x3 DCT terms work (3 of 9: 0.7%), while FFT of weight rows (72% error at half the numbers on sunrise), magnitude pruning (64%) and low rank (44%) do not: hidden-unit order carries no frequency structure and the trained matrices are near full rank (top 16 of 128 directions hold 27% of the energy). Tokens therefore stay roughly proportional to parameters, about 2 per weight in compact text.

## Codebook vectors (Nov 7, 2026)

The target reader is a Qwen model, and Qwen's tokenizer splits numbers into single digits: `[990 -495 122] / 990` is 18 Qwen tokens, and a compact sunrise layer is about 745k Qwen tokens (406k by `uvx count-tokens`, which uses tiktoken and keeps 3-digit groups whole). Numbers cost about 4 Qwen tokens per weight, so the fix is to write words, not digits.

- **`codebook g4`** (after the header) names a fixed codebook; `{ts trend refresh ...} / 11` is then a vector: each word stands for 4 numbers, laid end to end and divided by the scale. `g4` has 4,096 words of 4 numbers (3 bits per weight, 4 weights per token); `g2` has 256 words of 2 numbers (4 bits, 2 per token). Every word is one Qwen2.5 token with its leading space. `{...}` works wherever `[...]` does.
- The entries are k-means centroids of standard normal samples (`wir/scripts/make_codebook.py`, seeded), not of any model, so one codebook serves every program and never appears in the text. A fine-tuned model can learn the 4,096 words once. Weight rows are scaled to unit RMS (`/ q`, two significant digits) before coding. A model-specific codebook was barely better on sunrise (layer 0 relative error 0.069 vs 0.072 for the universal one). Words are taken in Qwen vocabulary order (frequent first) and paired with the most used entries; profanity, sexual, violent and identity words are blocked. The files are part of the language: changing them changes what existing text means, so a new codebook gets a new name.
- A vector whose length is not a multiple of the word size ends in a word that overhangs it; the extra entries are dropped. An overhang of a whole word is still an error, and number vectors must match exactly.
- `wir decompile --codebook g4 --calibrate probe.json` codes every direction and readout. With calibration inputs (default: the `--domain` probe) rounding is error-aware, as in GPTQ: each 4-number group's rounding error is pushed onto the not-yet-coded entries through the inverse input covariance, and readouts are coded against the outputs of the already-coded units. On sunrise layer 0 this halves the error of plain nearest-word rounding (0.063 vs 0.127). Biases stay numbers.
- Coded text is a `print` fixed point and recompiles bit for bit. `canon` expands it back to numbers (renormalizing breaks the words), so re-code from weights rather than canonicalizing coded text.
- `output y1..y128: R1` declares a range of outputs (the printer uses it), saving about 7 tokens per output.

Quantization study first (Python, `--offset 1` eval samples that calibration never saw, all six sunrise MLPs replaced). Loss change and top-1 agreement: one digit per weight (10 levels, GPTQ) +0.015 / 98.8% at 1 Qwen token per weight; `g4` +0.019 / 98.7% at 1/4; `g2` +0.005 / 99.4% at 1/2. Plain rounding without error feedback has 1.5-3x the per-layer error of GPTQ rounding at the same size. Larger codebooks (16,384 entries, still single-token words): 4 numbers per word gives +0.008 / 99.1% at the same tokens as `g4` (a candidate `g4x16k`), while 8 numbers per word (1.75 bits per weight) costs +0.13 / 94.4%, so about 4 weights per token is the floor for this model.

## Real checkpoints (Nov 7, 2026)

**Yumon Paint CNN colorizer** (`checkpoints/old/yumon-colorizer-256-a`, `src/vision/colorizer.rs`). The head `out_conv(relu(head_conv(f)))`, a 3x3 conv (64 -> 64) then a 1x1 conv (64 -> 313), is at every pixel a one-hidden-layer ReLU network over the 3x3 feature patch: R576 (channel*9 + ky*3 + kx, zero padding at the border), 64 creases, 313 outputs.

```
cargo run --release --bin export_wir_colorizer     # -> tmp/wir-colorizer/head.json + probe.json (all 4,096 pixels of gt_sample0.png)
cd wir
cargo run --release -- decompile ../tmp/wir-colorizer/head.json --domain ../tmp/wir-colorizer/probe.json -o ../tmp/wir-colorizer/head.wir
cargo run --release -- verify ../tmp/wir-colorizer/head.wir ../tmp/wir-colorizer/probe.json
```

| text | tokens (tiktoken) | Qwen tokens | relative error | argmax same |
| --- | --- | --- | --- | --- |
| point-anchored, 6 digits (first decompile) | 673,659 | | 1.5e-6 | 4096/4096 |
| compact | 125,353 | | 2.6e-4 | 4085/4096 |
| compact + `--domain` (26 inactive units dropped) | 76,500 | 140,076 | 2.6e-4 | 4085/4096 |
| `--shape 64x3x3`, all 9 patch terms | 80,931 | | 2.7e-4 | 4087/4096 |
| 6 patch terms | 65,453 | | 1.1e-3 | 4016/4096 |
| 3 patch terms (level, dx, dy) | 50,081 | | 7.1e-3 | 3666/4096 |
| `--domain --codebook g2` | 25,085 | 27,982 | 8.4e-3 | 3601/4096 |
| `--domain --codebook g4` | 16,834 | 19,731 | 9.3e-3 | 3727/4096 |

Errors are against the model's own logits (up to 7.7) from a Burn WGPU forward. `--exact` text matches within 3.6e-6, the f32/f64 floor.

**yumon-base-dec-sunrise** (dense decoder, 128 wide, 6 layers, SwiGLU 128 -> 512 -> 128). Each block's MLP is a gate program: 512 gates, 128 outputs, 196,608 weights.

```
cargo run --release --bin wir_decoder -- record    # tmp/wir-sunrise/layer{l}_swiglu.json + layer{l}_probe.json
cd wir && cargo run --release -- decompile ../tmp/wir-sunrise/layer0_swiglu.json -o ../tmp/wir-sunrise/layer0.wir
cargo run --release -- compile ../tmp/wir-sunrise/layer0.wir --quiet -o ../tmp/wir-sunrise/layer0.wir.json
cd .. && cargo run --release --bin wir_decoder -- eval --mlp 0=tmp/wir-sunrise/layer0.wir.json   # ... up to --mlp 5=
# coded: decompile ... --codebook g4 --calibrate ../tmp/wir-sunrise/layer0_probe.json, then eval with --offset 1
```

On held-out MLP inputs (2,048 per layer) the canonical text is within 0.05-0.10% relative error of the SwiGLU output. With all six MLPs replaced by weights compiled from the WIR text, next-token loss on 256 cache samples (30,872 positions) is 4.1350 for both, the change is under 1e-4, and the argmax token is identical at every position.

| text, per layer | tokens (tiktoken) | Qwen tokens | layer relative error | all six layers: loss change | top-1 same |
| --- | --- | --- | --- | --- | --- |
| compact numbers | 406k | 737k-748k | 0.05-0.10% | < 1e-4 | 100% |
| `--codebook g2` | 112k | 114k | 1.4-5.8% | +0.0047 | 99.38% |
| `--codebook g4` | 63k | 65k | 2.5-10.9% | +0.0189 | 98.72% |

Coded rows are scored with `wir_decoder eval --offset 1`: 256 samples next to the ones the probes (and so the calibration) came from, 30,934 positions, base loss 4.0127. Layer 0 is the most sensitive (`g4` alone: +0.011); mixing `g2` there and `g4` elsewhere is an option. Decompiling and coding one layer takes 0.8 s.

Not covered: the colorizer backbone (stacked ReLU convs and residual blocks need depth, v0.2), and the attention, norms and embeddings of the decoder (WIR states MLPs only).

## Function-space distillation (Nov 7, 2026)

Coding weights tops out near 4 weights per Qwen token, so a sunrise layer can't drop much below ~65k tokens that way. The alternative is to describe the layer's *behaviour* with a much smaller program: fit a SwiGLU gate program with h << 512 gates to the layer's real inputs and outputs (MSE, Adam, gates started from the original's h highest-contribution units), then code that with `g4` as usual. The text says a different function that agrees with the layer where the model actually uses it.

```
cargo run --release --bin wir_decoder -- record --eval 1024 --offset 3 --probe 40000 --out tmp/wir-approx   # 40k fit rows per layer
cd python-train && uv run python ../wir/scripts/distill.py 0 --gates 32 --steps 6000 --init top --out ../tmp/wir-approx/fit/l0_g32.json
cd ../wir && cargo run --release -- decompile ../tmp/wir-approx/fit/l0_g32.json --codebook g4 --calibrate ../tmp/wir-sunrise/layer0_probe.json -o l0.wir
# compile, then wir_decoder eval --offset 1 --mlp 0=... (all six layers)
```

All six sunrise MLPs replaced, scored with `eval --offset 1` (30,934 positions; fit rows come from offset-3 samples, so neither fitting nor calibration saw them):

| per layer | weights | Qwen tokens | all six: loss change | top-1 same |
| --- | --- | --- | --- | --- |
| original 512 gates, `g4` | 196,608 | 65k | +0.019 | 98.72% |
| 64 gates, numbers | 24,832 | | +0.237 | 95.34% |
| 32 gates, numbers | 12,480 | | +0.509 | 93.38% |
| 32 gates, `g4` | 12,480 | 8.4k-8.5k | +0.541 | 92.13% |
| 16 gates, `g4` | 6,304 | 6.3k | +1.974 | 83.15% |

32 gates is the knee: 16x fewer weights, 7.6x fewer tokens, about 92% top-1 agreement. Per-layer relative error at 32 gates ranges from 0.07 (layer 5) to 0.50 (layer 1), and an exact least-squares linear map already reaches 0.15-0.64, so the layers are mostly a linear map plus a few dozen nonlinear corrections. Layers 0 and 1 cost the most (layer 0 alone at 16 gates: 86% top-1), so a fixed token budget should give them more gates.

What did not help:

- **A shared low-dimensional frame** (gates read r = 16-32 projections of x and write back through an r-dim map) was worse than plain gates at equal parameters (layer 0: 0.236 at 20.9k vs 0.193 at 24.8k).
- **A separate low-rank linear term** (48 gates + rank 16) was no better than 64 gates.
- **Sequential fitting** (`wir_decoder record --mlp ...`: inputs from the model with earlier layers already replaced, targets from the original) lowered the six-layer loss change to +0.430 but top-1 fell to 90.11%. Its targets only undo MLP drift, not attention drift.

At 32 gates the readouts are about 45% of the text: 128 output lines, each an 8-word vector, a scale and a 6-digit bias. Writing one `writes` vector per gate (128 entries, 32 words) plus one coded bias vector would bring layer 0 from 8.5k to about 5.0k Qwen tokens (estimated by tokenizing synthesized text; the syntax does not exist yet and coding the biases is untested).

## WIR 0.2: combinators and depth (Nov 7, 2026)

Small programs whose behaviour can be checked, such as one step of a game, are the near-term data for the Qwen fine-tune: they fit in a context, and a generated program can be compiled, run and scored. They need depth and logic, which v0 lacks. A `wir 0.2` header selects a separate language (`wir/src/deep.rs`); v0 programs are unchanged.

```
wir 0.2
input bx: R1 in [1, 14]            # named scalar inputs, each with an optional interval
input vx: R1 in [-1, 1]
let nx = bx + vx                   # affine arithmetic; * and / by constants (and small products, below)
let at_l = lt(nx, 1)
output bx2 = select(at_l, 2, nx)
assert bx2(1, -1) == 2             # one value per input, in declaration order
```

- **Combinators**, each an exact composition of relu and affine maps: `relu`, `abs`, `max`/`min` (n-ary; max(a, b) = b + relu(a - b)), `clip(a, lo, hi)` with constant bounds, `step(a)` = clip(a, 0, 1), comparisons `gt ge lt le eq` (steps of a - b, so exact when a - b is an integer: `ge(a, b)` = step(a - b + 1)), `and`/`or`/`not` (exact on bits: and = relu(sum - (n - 1)), or = sum - relu(sum - 1)), and `select(c, a, b)` for a bit c, via the big-M product c d = relu(d - M+ (1 - c)) - relu(-d - M- (1 - c)). `a * b` of two values compiles when one factor takes at most three integer values (added with the tactical brains, see the last section).
- **Intervals.** Every node carries a sound interval computed from the input domains. `select` takes its M from them (and refuses unbounded values), and a relu whose argument is never negative becomes the argument, one never positive becomes 0. Both hold only while inputs stay in their domains; the compiled network is wrong outside them.
- **Depth is chosen by the compiler.** Relu nodes are shared when identical (after scaling to smallest coefficient 1, so integer programs get integer weights; max coefficient 1 until Nov 7), each sits one layer above the deepest node it reads, and values read more than one layer later ride on carry units: relu(v - lo) + lo, one unit when v has a lower bound, two (relu(v) - relu(-v)) when unbounded. `wir compile` reports units per layer and statement, carries separately.
- **Decompiler.** Any ReLU MLP (`.json` with `layers`, or `nn.Sequential` safetensors with 3+ linear layers) decompiles to flat 0.2 text, one `let hK_U = relu(...)` per hidden unit. The canonical form scales each unit to max incoming weight 1, folds constant units into the next bias, drops unread units, merges duplicates and sorts each layer, so scaling and permuting hidden units leave the text unchanged. Numbers snap to rationals with denominators up to 1,000 (`35/72`), which integer-grid programs produce and six digits cannot hold.
- Not built: v0 plane/anchor creases inside 0.2, vector inputs, codebook vectors in 0.2 text, and the `in space(...)` escape hatch (relu of any expression covers its use so far).

**Pong** (`wir/examples/pong.wir`, written by `wir pong`, `wir/src/games.rs`). One step on a 16 x 12 integer grid: inputs ball position and velocity, both paddles and both controls; outputs the next state and two scoring bits. Walls reflect, paddles clip, a paddle hit reflects the ball, a miss scores and restarts it in the middle. It compiles to 4 hidden layers [12, 20, 14, 18], 1,084 parameters, and matches a reference simulator on all 604,800 states in its domain (worst |diff| 1.8e-15). The source is 1,160 Qwen tokens; the decompiled flat text is 1,801. PyTorch 2.14.1 loads the f32 safetensors as a plain `nn.Sequential` with the same outputs. Of the 64 units, 42 are carries, the main inefficiency left.

`wir pong --variants N --out DIR` draws field sizes (8-32 x 6-24), paddle sizes 1-6, ball speeds 1-3 and paddle speeds 1-2, writes each program with asserts from the simulator, checks it (all states up to 200k, else a 200k sample) and writes `manifest.jsonl`. 50 variants (seed 7): 1,084-1,604 parameters, all depth 4, 2.1-2.2 KB each, 9.0M states checked, no mismatches, 5.5 s.

```
cd wir
cargo run --release -- pong -o examples/pong.wir                       # default game, all states checked
cargo run --release -- pong --variants 50 --out ../tmp/wir-pong --seed 7
cargo test --test deep                                                 # 13 tests, ~4 s
```

## Grid programs and pathfinding (Nov 7, 2026)

Pong is one fixed board. A pathfinding step is the same small rule at every cell of a grid, so writing it out per cell (as flat 0.2) repeats the rule hundreds of times. Inputs typed `grid` make a 0.2 program **one cell's rule**: grid inputs are read at integer offsets, outside reads take the input's `pad` value, and the compiler builds one cell network that runs at every cell.

```
wir 0.2
input d: grid in [0, 64] pad 64        # distance field; outside the grid reads 64
input wall: grid in [0, 1] pad 1
input goal: grid in [0, 1] pad 0
let best = min(min(d[-1, 0], d[1, 0]), min(d[0, -1], d[0, 1]))
output d2 = clip(1 + best + 64*wall - 129*goal, 0, 64)
```

- **Lowering.** Each (input, offset) pair read becomes one scalar input of the cell network (a *tap*, `d[-1,0]`), in first-read order; everything else is ordinary 0.2. All taps are read in the first layer, so the cell network is a Conv2d stack: one (2r+1) x (2r+1) kernel over the channels, then 1 x 1 kernels. `StencilWeights` (`src/deep.rs`) holds the cell network plus taps and pads; `apply` runs it over a grid, `conv_layers` / `from_conv` convert, `.safetensors` files are `nn.Sequential(Conv2d, ReLU, ..., Conv2d)` keys with `wir.kind = stencil` and per-channel pads in the metadata (pad each channel before the first convolution), `.json` stores the layers over taps. Grid programs do not take asserts yet.
- **Decompiler.** Any such conv stack (including a trained one) decompiles to grid text: `input c: grid pad v` per channel, then flat `let hK_U = relu(...)` units over taps written `d[-1,0]`. Taps are sorted (channel, row, column) before canonicalization, so the text is a fixed point.
- **Integer weights.** Relu nodes are now normalized to smallest coefficient 1 (was largest), so integer programs compile to integer weights. The pathfinding nets went from 1/129 weights (f32 drift: 58 of 200 settled grids wrong) to exact in f32; Pong's worst error went from 1.8e-15 to 0.

**The rule** (`wir path`, `src/games.rs`). One step of value iteration: a goal is 0, a wall is the ceiling M, every other cell is one step past its cheapest neighbour, capped at M. A goal's coefficient exceeds anything the rest of the sum can reach, and the clip handles walls and the cap together. The rule never reads a cell's own old distance, so it repairs itself: stale low values rise by at least one step per iteration, and moving the goal or a wall needs no reset. From any field in [0, M] the iteration converges to the Dijkstra distances capped at M (the capped distances are a fixed point; with positive costs it is the only one). Variants: `--moves 8` (orthogonal steps cost 2, diagonal 3; a diagonal only counts when both cells beside it are open, `max(d[-1, 1], M*(wall[-1, 0] + wall[0, 1]))`), `--terrain C` (a `cost` channel in [1, C]; leaving a cell costs its terrain), `--ceiling M`.

| rule | cell network | conv weights | flat 6 x 6 MLP |
|---|---|---|---|
| 4 moves, M 64 | 6 taps, [6, 5, 2] | 218 | [158, 156, 72], 55,958 |
| 4 moves + terrain 5 | 7 taps, [7, 6, 2] | 324 | |
| 8 moves, M 128 | 14 taps, [14, 12, 10, 6, 2] | 785 | [266, 280, 257, 198, 72], 244,011 |
| 8 moves + terrain 4, M 200 | 15 taps, [15, 13, 11, 7, 2] | 1,020 | |

**Checks** (`wir path` runs them on every program it writes; `path_check`):

- **Every cell input.** The cell network against the rule written as plain code (`path_cell`) on every combination of tap values. For the default 4-move rule that is 65^4 x 2 x 2 = 71,402,500 cases, all exact (8.6 s with the rest below, release). Since a grid step is the cell network at every cell with in-domain pads, this proves the step exact on every grid of every size. 8-move and terrain rules have too many combinations and are sampled (20-100M).
- **Random grids.** 200 random 12 x 12 grids (open fields or carved mazes with loops, one or two goals, random starting fields): every step against the simulator, the settled field against Dijkstra. All variants above: 0 mismatches.
- **Every small maze.** `--all-mazes 4x4`: all 65,536 wall layouts with each of 16 goal cells (1,048,576 grids) settle to the Dijkstra distances, at most 12 steps, 5.2 s.
- **Flat expansion.** `--flat HxW` writes the same rule as a plain 0.2 program per cell and checks it against the simulator; `tests/path.rs` also checks it equals the stencil output on random grids.
- **PyTorch.** `scripts/path_planner.py check` loads the conv safetensors into Conv2d layers with per-channel padding: one step exact on 2,000 random 8 x 8 fields, 200 random 12 x 12 grids settle to BFS (f64 and f32).

## Written vs learned planners (Nov 7, 2026)

The question: does a planner trained from scratch find the same weights, and how do weights relate to behaviour? `wir/scripts/path_planner.py train` fits the compiled network's own layout (Conv2d 3 x 3, then 1 x 1, inputs d / wall / goal, ceiling 64, 4 moves) from random weights on random 8 x 8 grids (PyTorch 2.14.1, CPU, f64, Adam 3e-3 cosine, batch 64). **step** mode shows random fields (half of them near the answer) and the rule's output; **unroll** mode starts from the ceiling, runs 48 steps and scores only the final field against BFS. All networks were then scored the same way (`summary.json` in `tmp/wir-path`, 100 fresh grids per size, run until the field stops changing or 138 steps; a path is optimal when walking downhill on the network's field reaches a goal in the BFS distance):

| network | weights | one step exact | cells exact 8 x 8 / 32 x 32 | paths optimal 8 x 8 / 32 x 32 | train time |
|---|---|---|---|---|---|
| written in WIR | 218 | 100% | 100% / 100% | 100% / 100% | - |
| written, fine-tuned unroll 2k steps (lr 3e-4) | 218 | 99.999% | 100% / 87.9% | 100% / 100% | 74 s |
| 6+5+2 step, 20k (best of 3) | 218 | 40.0% | 10.8% / 1.9% | 46.3% / 4.6% | 31 s |
| 6+5+2 unroll, 3k (best of 3) | 218 | 3.8% | 11.9% / 1.7% | 39.5% / 4.0% | 117 s |
| 32+32 step, 20k (best of 3) | 1,985 | 91.4% | 61.7% / 26.7% | 99.0% / 47.9% | 43 s |
| 32+32 step, 200k | 1,985 | 99.8% | 99.8% / 71.0% | 100% / 99.8% | 462 s |
| 32+32 unroll, 3k (best of 3) | 1,985 | 11.2% | 17.0% / 3.2% | 38.9% / 3.9% | 241 s |
| 32+32 unroll, 12k | 1,985 | 4.0% | 31.4% / 8.4% | 70.4% / 13.7% | 897 s |

First-layer structure (`path_planner.py analyze`; weights scaled by input range): share of weight on taps the 4-move rule never reads (diagonal and centre distances, neighbours' walls and goals) / share of distance weight on the 4 neighbours / share of distance units whose weights cancel (relu(a - b), the min signature):

| network | unread | orthogonal | difference units |
|---|---|---|---|
| written | 0% | 100% | 50% (2 of 4; the other 2 are carries) |
| 32+32 step 20k (mean of 3) | 31.5% | 76.9% | 13.5% |
| 32+32 step 200k | 6.3% | 95.6% | 31.2% |
| 32+32 unroll 3k (mean of 3) | 67.7% | 48.3% | 8.3% |
| 32+32 unroll 12k | 62.8% | 51.8% | 0% |

Best of 3 means the seed with the highest 8 x 8 path score. Findings:

- **The written shape is not found by gradient descent.** 0 of 6 runs at 6+5+2 learned the rule (best 51% of single steps exact; several collapsed to a constant output). The 218-weight solution exists and the compiler writes it directly; training from random weights did not reach it.
- **Width plus a lot of training learns the step, and behaviour outlasts values.** At 9x the weights and 200k steps, 99.8% of single steps are exact and downhill paths are optimal on 99.8% of 32 x 32 starts, but only 71% of 32 x 32 cells hold the exact distance and the field never stops moving there. Small per-step errors compound over 60+ iterations and grow on grids larger than the 8 x 8 training ones; the ordering of neighbours survives longer than the numbers.
- **Weights drift toward the program as training continues.** Weight on unread taps falls from 31% to 6%, orthogonal share rises from 77% to 96%, difference units from 13% to 31%. Same direction as the written network, not there yet.
- **End-to-end training learns something else.** Scored only at step 48 from the ceiling, the network learns a 48-step transient, not a stable rule: most of its weight sits on taps the rule ignores, no unit is a clean difference, and it is worse on every behavioural measure despite a low training loss (4e-4 normalized MSE). Scoring it at exactly 48 steps instead of at rest barely changes this (8 x 8 paths 73%).
- **Training only protects what it measures.** Fine-tuning the exact net started at loss 1.9e-33, but Adam still moved weights by up to 0.24 (output layer weights are up to 129). It stayed exact on 8 x 8 and lost exactness on larger grids (87.9% of 32 x 32 cells) while paths stayed optimal.

For WIR this argues for programs as priors or as the answer itself rather than as a target for distillation from trained planners: the program's weights are 9x smaller than a learned net that nearly matches it, exact on all grids, and the learned nets that come close do so by drifting toward the program's structure. A trained planner can still be decompiled (`wir decompile planner.safetensors`) to inspect what it reads; its text is dense (27 taps per unit) and has no short form.

**Demo.** "WIR Pathfinder" (claude.ai artifact, private) runs the written 4-move, 8-move and mud networks on a 25 x 17 grid in the browser with a JavaScript copy of the rule and Dijkstra as live checks, an editable map, downhill runners, a live view of one cell's network, and three of the learned networks for comparison.

```
cd wir
cargo run --release -- path --local 100000000 --all-mazes 4x4 -o examples/path.wir
uv run --project ../python-train python scripts/path_planner.py train --out ../tmp/wir-path/runs/step_wide_long --mode step --hidden 32,32 --steps 200000
uv run --project ../python-train python scripts/path_planner.py analyze ../tmp/wir-path/runs/*/planner.safetensors
cargo test --test path                                       # 11 tests, ~2 s
```

## Cellular water and conservation proofs (Nov 7, 2026)

Picked from `docs/ideas/WIR_PHYSICS.md`: the flux-limited cellular liquid, with the `flow` primitive's goal (exact conservation) met by a proof instead of a new primitive. The physics doc's training ideas (`param`) are left out; nothing here is trained.

**Lets at offsets.** In a grid program, a `let` (or output) can be read at an offset: `fall[-1, 0]` is the whole `fall` expression evaluated one row up. The compiler lowers the let's text again with every grid read and every let it reads moved by the offset (nested offsets add); copies are cached, and relu nodes they share with other copies are shared by the graph. Every tap still lands in the first layer, so a two-pass rule (gravity, then levelling on the result) is still one 3 x 3 convolution stack. A let is also lowered where it is defined, so its own taps exist (with zero weights) even if only shifted copies are used.

**`conserve out = channel`.** Proves that, summed over any grid, the output equals the input channel, for every state in the declared domains (`deep::prove_conservation`). Each graph node is a *shape* at an offset (a tap, or a relu over shapes at relative offsets). Over an unbounded grid each shape contributes (sum of its coefficients) x (its sum over all cells), so the first check is that every shape's coefficients in output - input cancel and the constant is 0. On a bounded grid that cancelling fails only near the edges; near a corner the grid looks like a quadrant, so for each of the four corner orientations every cell in a window around the corner is rebuilt symbolically (reads inside the grid are inputs, reads outside take their pad constants, relus go through `Graph::relu`, which drops relus the pads make constant or one-signed), and what the cell adds to the total must come out as exactly 0. Checking shapes one at a time at the edges is not enough: at the bottom row `fall = m - relu(m - relu(16 - 0 - 16))` is zero only as a whole. The argument covers grids at least 2L cells on a side (L = copy spread + read reach + 1; 8 for the water step). A failed proof is a compile error naming the shape or the corner cell.

**The rule** (`wir fluid`, `src/fluid.rs`, `examples/fluid.wir`). Channels `m` (water units, 0 to capacity C, default 16), `wall` and `pair` (1 where the cell trades with its right-hand neighbour this step; starts as alternating columns, flips every step).

```
let fall = min(m, relu(16 - m[1, 0] - 16*wall[1, 0]))     # pour what fits into the cell below
let mv = m - fall + fall[-1, 0]
let d = mv - mv[0, 1]
let half = ge(d, 1) + ge(d, 3) + ... - le(d, -1) - ...     # half the difference, rounded up
let flow = select(and(pair, not(wall), not(wall[0, 1])), half, 0)
output m2 = mv - flow + flow[0, -1]
output pair2 = not(pair)
conserve m2 = m
```

Design notes from a numpy prototype (scratch, not kept): letting a cell give a fraction of its difference to both sides at once leaves a staircase surface (integer units cannot move a difference of 1), and a global left/right bias fixes that but makes a period-2 checkerboard. Disjoint alternating pairs (Margolus-style) settle a basin flat within one unit, and since each cell has one partner, nonnegativity and the capacity bound follow: a cell gives at most ceil(d/2) <= its water, and the receiver ends at most at the giver's old level. Requiring support below before spreading stopped water from pouring off ledges, so it was dropped. No pressure: a U-bend does not equalize. The bounds need valid states (dry walls, alternating `pair`); conservation does not.

| C | taps | cell network | conv weights |
|---|---|---|---|
| 2 | 17 | [15, 12, 14, 8] | 932 |
| 4 | 17 | [15, 12, 22, 8] | 1,100 |
| 16 | 17 | [15, 12, 70, 8] | 2,108 (all integers) |

64 of the 70 third-layer units are `half` (two copies of 16 steps of 2 relus).

**Checks** (`wir fluid` runs them on the program it writes; `fluid_check`): the proof (4 shapes cancel, 39 edge cells per program add exactly 0); the cell network against `fluid_cell` on all 5,038,848 tap combinations at C = 2 and 20M sampled ones at C = 16; 100 random 16 x 24 grids (boxes, ledges, pillars, blocks or random water) for 300 steps each against `fluid_step`: 0 differing steps, total water never changed, no cell outside 0..C, walls always dry. `tests/fluid.rs` (5 tests, ~1 s release, ~14 s debug) adds refusals and a basin that settles level.

**Speed** (`--bench 256x256`, one thread, release, Intel Core i5-12500): the network takes 26.2 ms per step (577 nonzero weights per cell), the plain rule 1.8 ms, so as a network it is about 15x slower on a CPU. GPU timing is not measured; the conv stack is what a GPU would run, but that comparison has not been done.

**Demo.** "WIR Water" (https://claude.ai/artifact/W7gEhnMaKSc9P4WZ4B97Qs, private): a 34 x 60 tank run by the compiled weights in JavaScript, with brushes for water and walls, four scenes, a water ledger (added and removed by the brush, changed by the network: always 0) and an optional per-cell comparison with the plain rule.

```
cd wir
cargo run --release -- fluid -o examples/fluid.wir --bench 256x256   # proof + 20M cell inputs + 30k grid steps, ~14 s
cargo run --release -- fluid --capacity 2 --local 6000000            # every cell input
cargo run --release -- compile examples/fluid.wir -o ../tmp/wir-fluid/fluid_c16.json --quiet
cargo test --test fluid
```

## Swarms and a shared world (Nov 7, 2026)

Picked from `docs/ideas/WIR_FLOCK.md`. The doc's per-boid controller needs neighbour averages (a division) and products of inputs; both are avoided here by scoring headings instead of steering a velocity. Nothing is trained.

**Two programs per species** (`wir swarm`, `src/swarm.rs`, `examples/{fish,bird}_{sense,brain}.wir`). Agents have sub-cell integer positions (16 units per cell) and one of 8 headings. The host counts agents into per-cell channels (`own0..own7` by heading, capped at 3; `other`, the other species, capped at 6), and the world supplies `m` (water), `wall` and, for fish, `dist` (distance to food).

- The **sense** program is a grid program with a 5 x 5 window, evaluated only at agents' cells. Outputs, 8 each: `a` neighbours already heading each way (alignment), `n` neighbours lying in each direction (cohesion), `t` the other species by direction, `b` blocked directions (a diagonal also when either side cell is), and for fish `g` = clip(dist - dist[d], -1, 1). Sectors are a pinwheel of 3 cells each (the two cells along a heading plus the knight-move offset counter-clockwise of it); rounding offsets to the nearest heading instead gives diagonals 4 cells and straight headings 2, which pulled cohesion toward diagonals. `crowd` and `open` are lets read at offsets.
- The **brain** is a plain 0.2 program (utility AI): for each heading, `align*a + cohesion*(n - (1 + spacing)*relu(n - crowd)) + other*t + goal*g + turn[steps from current] + noise*z - big*b`, then the best heading as one-hot `h2_0..h2_7`. Keys are 8*score + (7 - d), so no two tie and `ge(key, best)` is exactly one-hot. `big` is one more than the span of every other term, so a blocked heading always loses to an open one (a test sets every other input against the open heading; a penalty one smaller fails it).

Layout lesson: a value that is affine in many inputs and read several layers up makes the compiler carry each input separately. The first brain (a sequential max chain over affine keys) was 9 hidden layers and 34,731 parameters; writing each key as `relu(lin) - relu(-lin)` plus its bend (one value, 2 units) and the max as a balanced tree gives [24, 28, 28, 27, 15], 4,211 parameters. Likewise `cap - relu(cap - x)` instead of `min(x, cap)` keeps the 25-tap sums out of the carries (sense: [284, 268, 268], 244,716 conv parameters before, [56, 52] and 21,940 after).

| program | inputs | hidden | parameters (non-zero) |
|---|---|---|---|
| fish sense | 251 taps, 12 channels | [56, 52] | 21,940 conv (604) |
| fish brain | 56 | [24, 28, 28, 27, 15] | 4,211 (470) |
| bird sense | 242 taps, 11 channels | [40, 36] | 13,700 conv (540) |
| bird brain | 48 | [24, 28, 28, 27, 15] | 4,019 (454) |

Species: fish live in water at least 4 of 16 units deep, align 2, crowd 4, flee birds (-3 per bird), follow food (4); birds live where water is at most 1, align 3, crowd 3, are drawn to fish (+2). Both: turn scores [8, 4, 0, -6, -12] for 0 to 4 steps of 45 degrees, a noise kick (+6 to a random heading) on 1 step in 10. Movement (host): a step of 3 (fish) or 4 (bird) units, 2 or 3 per axis on diagonals; stay unless the destination is open, an agent whose cell closed may leave into any non-wall cell, and at most 3 of a species move into a cell. Without that cap agents sharing a cell see none of each other (sectors skip the centre) and stacks of 30 moved as one dot.

**Checks** (`swarm_check`, default `wir swarm`): 200,000 random inputs per network against `sense_cell` / `brain_choose` (0 mismatches each), then 8 worlds (32 x 48, a walled tank half full of water with rocks, 60 fish and 60 birds) for 400 steps with networks and rule in lockstep: 384,000 decisions, 0 differing steps, 0 blocked headings picked while an open one existed. The first sense check reported 18,922 of 20,000 fish inputs wrong; the plain-code rule was wrong (it tested the agent's own cell for straight headings), not the network.

Flocking, start to step 400, mean over the 8 worlds (same-species neighbours in the 5 x 5 window; local order = |mean heading vector| over an agent and those neighbours):

| | neighbours | local order | global polarization |
|---|---|---|---|
| fish | 1.9 -> 11.4 | 0.59 -> 0.85 | 0.09 -> 0.45 |
| birds | 1.8 -> 12.7 | 0.60 -> 0.88 | 0.13 -> 0.35 |
| fish, align 0 | 1.9 -> 31.2 | 0.59 -> 0.43 | 0.10 -> 0.31 |
| birds, align 0 | 2.0 -> 1.7 | 0.61 -> 0.61 | 0.11 -> 0.06 |

Without alignment fish still gather (cohesion and the food goal) but head every which way; birds stay scattered.

**Demo: six networks in one grid.** "WIR Tidepool" (https://claude.ai/artifact/4pq3RpGc1GijfbYMbCfAyC, private): a 40 x 72 pool running the water step (`fluid.wir`), a pathfinding step (`path.wir`, food as goals; rock and water under 4 units as walls, so the map reroutes when water moves) and the four swarm networks, all from compiled weights in JavaScript. Each model only reads the shared grid: water moves fish habitat and the fish map, fish and birds sense each other. Host rules outside the networks: food sinks, a fish that reaches food eats it and a fish hatches, a bird right above a fish at the surface catches it with probability 1/30 and then rests 400 ticks. Scenes: lagoon, two ponds split by a ridge (erase a gap and the school finds the food), dam break. An optional live comparison runs every plain rule (water, path, senses, brains); a Node run of the page's core (`node check.js`, scratch, not kept) over 1,500 ticks of each scene had 0 differences across 578k decisions (random noise, so runs differ) with water unchanged. Node 24.11.1 on the i5-12500, lagoon (141 agents): 4.4 ms per tick (water 2.9, map 0.35, swarms 1.1).

```
cd wir
cargo run --release -- swarm --out examples                # 800k network inputs + 384k decisions, ~19 s
cargo run --release -- swarm --no-align --cases 1000        # the control
cargo test --test swarm
```

## Tactical enemies and response curves (Nov 7, 2026)

Picked from `docs/ideas/WIR_BEHAVIOR.md`: utility AI with response curves over shared fields. The doc's example did not compile as written (`aggro_drive * dp_n` multiplies two inputs; its Pong foresight fold needs a third mirror), and its emergent-tactics claims were guesses. Below is what was built and measured. Nothing is trained.

**Small products.** `a * b` of two non-constant values now compiles when one factor's interval is two or three integers (width 1 or 2, integer ends), exact when that factor is an integer. With t = m + u, u in {-1, 0, 1}: x t = m x + x [u = 1] - x [u = -1], and both gated terms are big-M products whose gate is u itself (u = -1 pushes the first pair of relus to 0, u = 1 the second), so a product is 4 relus one layer above its inputs, with no gate bits (`Graph::mul_small`). Two wide factors are refused. This is what lets a brain write `approach*gp0`: a drive times a field gradient.

**The world** (`wir tactics`, `src/tactics.rs`). A 24 x 40 arena, one player, 9 enemies (3 archetypes x 3), one cell per tick, 8 headings or hold. The host traces the player's line of sight (`exp`) and runs two shared distance fields with the 8-move `wir path` step (ceiling 96, 3 iterations per tick): `dp` to the player, with ally cells costing 3 to leave (terrain channel), and `dc` to cover (open cells the player cannot see). Every enemy runs:

- **senses**, one grid program at its cell: `gp`, `gc` grade each neighbour against the best neighbour (1 within 1 of it, 0 within 3, -1 farther; the enemy's own cell is inflated by its ally cost, so neighbours are not compared with it), `e` exposure of each neighbour, `n` allies per pinwheel sector of the 5 x 5 window, `b` blocked headings (wall, ally, the player, wall corners), and at the cell `dps` (distance after the best step), `es`, `gcs` (the cell's own cover grade).
- **a brain** per archetype, a 0.2 program of response curves:

```
let empty = relu(1 - ammo)                                    # reloading drops the magazine
let fear = 2*relu(8 - hp) + 2*relu(4 - hp) + 7*empty
let approach = clip(dp - 10 - fear, -4, 4)                    # fear pushes the preferred range out
let cover = clip(fear - 3, 0, 10)
let peek = clip(4 - fear, -3, 4)                              # empty is always -3: 7 >= 4 + 3
let plant = 5*and(es, ge(ammo, 1), le(dp, 18), ge(dp, 4))
let s0 = approach*gp0 + cover*gc0 + peek*e0 + k0              # k0: spacing, inertia, blocked
let s8 = cover*gcs + peek*es + plant + k8                     # hold
output o2_0 = and(w01, w02, ..., w08)                         # beats every other key
output fire = and(es, ge(ammo, 1), le(cool, 0), le(dp, 18))
```

Archetypes are constants in the source, one network each: grunt (range 10, fire 18), rusher (range 0, fire 3, fearless until 3 hp), sniper (range 22, fire 34, plants, more cover). Inertia scores `[3, 1, 0, -1, -3]` for 0..4 steps of 45 degrees from the current heading (+1 or +2 to keep holding). A blocked heading loses one more than every score can span, so it always loses to holding.

**Layout lessons.** The first brain was 10 hidden layers and 93,365 parameters, almost all carries: `select(gt(gp0, 0), approach, 0)` builds a gate bit a layer before the drive is ready, each score was a long affine sum carried to the argmax, and the max tree added 4 layers. The current one: products with no gate bits; the affine rest of each score packed as `relu(x - lo) + lo` (one unit); each key one unit (`9*relu(s + 78) - 702`); the argmax as 36 pairwise `gt` plus one `and` per option. Making "empty" a single relu (reloading drops the magazine) and having the fear curve, not a select, push `peek` to its floor saved two more layers.

| network | shape | parameters (non-zero) |
|---|---|---|
| path step (8 moves, terrain 3, ceiling 96) | 15 taps, [15, 13, 11, 7, 2] | 1,020 conv (94) |
| senses | 58 taps, [74, 82, 80, 129] | 38,153 conv (820) |
| grunt / sniper brain | 56 inputs, [49, 45, 101, 18, 74, 11] | 13,888 (947) |
| rusher brain | [34, 34, 55, 12, 74, 11] | 7,644 (704) |

**Checks** (`tactics_check`): 200,000 random sense inputs and 200,000 brain inputs per archetype against the rule as plain code, then 8 arenas x 600 ticks with networks and rule in lockstep (fields, every enemy, the player): 0 mismatches, 0 blocked headings picked (6.5 s). The two control brain sets compile and check exact too.

**Behaviour** (rule, 8 arenas x 600 ticks, scripted player that walks to random cells, waits 8 ticks and shoots the nearest enemy it sees every 2 ticks):

| | step-and-reverse moves | seen while hurt or empty: steps toward cover (grunt / rusher / sniper) | spread around the player |
|---|---|---|---|
| full | 0.6% | 85.5% / 16.9% / 95.2% | 0.40 |
| no fear curve | 0.3% | 16.4% / 15.0% / 18.1% | 0.39 |
| no inertia | 7.4% | 85.3% / 13.6% / 91.3% | 0.31 |
| no spacing | 0.6% | 86.5% / 16.2% / 92.3% | 0.39 |
| no ally cost | 0.7% | 88.7% / 16.5% / 92.9% | 0.38 |

Inertia cuts step-and-reverse moves by 12x. The fear curve turns retreat to cover from about 1 decision in 6 into 85-95% for grunts and snipers (the rusher is fearless by design). The share of time hurt enemies are seen barely moves (79% vs 84%), because the player only damages enemies it sees: an enemy becomes hurt while exposed, and heals out of the bucket soon after reaching cover. Measuring the decision is the useful number. Fear adds a little step-and-reverse (0.6% vs 0.3%): a hurt enemy hides, heals a point, steps out, and is pushed back. The pincer the idea doc predicted did not appear: spread (0 = all on one side, 1 = surrounded) is 0.38-0.40 with or without spacing or ally cost; ally cost 10 with 5 enemies per archetype gave 0.44 vs 0.41. Spacing only sees a 5 x 5 window, and the ally cost only reroutes followers behind an enemy, not around the player.

Two earlier design mistakes are worth keeping in mind. With gradients as sign(dp - dp[d]), a sideways step that shaved 1 off the distance tied with a real approach, and inertia then slid enemies along walls. And a scripted player that never stops, moving at enemy speed, can never be caught by a rusher.

**Demo.** "WIR Skirmish" (https://claude.ai/artifact/CvH96ywotgdD56qR2gCtxD, private): the arena in JavaScript from the exported weights (`--export`, 160 KB with three brain sets), three maps, wall editing, overlays for sight and both fields, an inspector for one enemy (its four response curves with the current point, option scores on a compass, the network's choice, the WIR source), brain sets full / no fear / no inertia, and a live check against the plain rule. A Node run of the page's core (scratch, not kept): 3 maps x 3 brain sets x 400 ticks, about 29k decisions and 20.8M field cells, 0 differences; about 9 ms per tick with the check off.

```
cd wir
cargo run --release -- tactics --out examples --export ../tmp/wir-tactics/tactics_demo.json   # 800k inputs + 4,800 lockstep ticks
cargo run --release -- tactics --rule-only --arenas 8 --steps 600 --no-fear                    # a control, rule only
cargo test --test tactics
```

## Yumon Ant: a colony of networks (Nov 7, 2026)

A side-view ant colony on ground that extends without bound, where each ant is a pair of WIR networks. Nothing is trained. `wir ant` (`src/ant.rs`) writes and checks the programs; the colony host is plain JavaScript (`wir/demos/ant/ant_core.js`) and runs the exported weights.

**World.** Rows 0..72, sky above a noise surface, soil with rock patches and embedded pebbles, generated in 32-column chunks on demand. A cell is walkable when it is open and touches something solid, so ants walk on floors, walls and ceilings. Five distance fields run with the 4-move `wir path` step (ceiling 120, 3 iterations per tick) over the walkable cells near each colony: `dn` to the nest chamber, `df` to food one ant can lift, `dg` to cells next to soil the colony plans to dig, `dm` to the dump beside the entrance, `dh` to heavy items still short of ants. `dn`, `dg`, `dm` are per colony.

**Senses** (grid program at the ant's cell, 10 channels, 46 taps): each field graded per heading (sign of the difference: 1 nearer, -1 farther), blocked headings (no footing), the fear grade, at the nest / at the dump, which of dn, df, dg, dh are below the ceiling, whether light food, a heavy item short of ants, or dig soil is next to it, the fear here and the nest distance.

**Brain** (0.2 program, 57 inputs): drives as response curves, then the task is the strongest drive and the ant follows that task's field.

```
let home = select(kn, 6*cf + select(hs, 2*min(hungry, 4), 0) + 4*and(free, ge(dns, 60)), 0)
let forage = select(fb, 5 - store + 2*old + 3*r1, 0) + select(fb2, min(hungry, 4), 0)
let dig = select(and(free, kg, not(starving)), 4 - 2*old + 3*r0, 0)
let help = select(and(free, kh, not(starving)), 3 + 3*r2, 0)
let explore = 3*or(idle, not(kn))
let flee = 8*scared
let tkey0 = 7*home + 6                        # task: argmax of the tkeys
let s0 = 4*(task0*gn0 + task1*gf0 + task2*gg0 + task3*gm0 + task4*gh0 + task5*w0 + task6*fg0) + k0
output dig2 = and(free, ld, not(starving), not(scared))
output em4 = and(ew4_5, ...)                   # straining: the strongest feeling
```

Roles r0..r2 (digger 45%, forager 35%, helper 20%, set at birth) are individual response thresholds; age (old after 55% of a lifespan of 2,600-4,400 ticks) moves ants from digging to foraging, as in real colonies. Emotes: calm, hungry, working, carrying, straining, scared, curious; the page colours each ant by it.

**Host.** Picking, digging (a pebble becomes a heavy item of 2-3), gripping, dropping (food into the store, sand onto the dump, where it slides like sand and settles into the hill), eating. A heavy item moves one cell every 2 ticks along the nest field (food) or dump field (pebbles) only while at least its weight in ants grip it; one that makes no progress for 40 ticks slips and rests 300 ticks. The queen lays while the store holds 3 + pop/3 and the planned tunnels have room (capacity 6 + open/3); a full colony plans a branch with a chamber. Hunger +1 per 45 ticks, starvation at 15. Loose sand never enters planned tunnels.

| network | shape | parameters (non-zero) |
|---|---|---|
| path step (4 moves, ceiling 120) | 6 taps, [6, 5, 2] | 218 conv (28) |
| senses | 46 taps, [126] | 16,419 conv (358) |
| brain | 57 inputs, [68, 83, 83, 125, 132, 60, 77, 21, 36, 21] | 60,033 (1,356) |

**Checks.** `wir ant`: 200,000 random inputs each for path, senses and brain against the rule as plain code, 0 mismatches, 0 blocked headings picked (1.6 s). The page carries 256 sense and 256 brain test vectors from the Rust rule; its JavaScript port of the rule matches all 512 before the colony starts, then checks every decision and field cell live. Headless (`demos/ant/harness.js`, Node 24, i5-12500): 10,000 ticks, 846,167 decisions and 70,966,980 field cells, 0 differences. Population peaked at 129 and cycled between about 70 and 130 on spawned food; 284 born, 192 died (175 of old age, 17 starved); 506 cells dug, 68 branches, a hill 22 cells high. Mean tick over the 10,000 ticks: 3.7 ms with the check off, 5.2 ms with it (70-130 ants); a CPU profile puts the fields at ~47% of a tick and senses plus brains at ~23%.

**What did not work.** Summed utilities (every drive times its field's grade, added up) froze ants: with unit-step fields every non-best heading grades -1, so three drives pointing three ways scored every heading below holding. Picking one task fixed it. With the same thresholds everywhere, one heavy item in range sent all 41 free ants to help; roles fixed that. A one-cell shaft, a three-row chamber (its middle row touches nothing, so it is not walkable) and a per-cell ant cap deadlocked traffic: in side view every surface is a one-cell lane, so ants now pass one another and the shaft is two cells wide. Repose sliding filled the shaft from the dump and cut the nest off until loose sand was kept out of planned cells. A field ceiling of 64 left most food out of reach of the nest once the hill and tunnels grew. Staggering thoughts (each ant thinks every 2nd or 3rd tick) starved the colony both ways tried: walking on the old heading overshoots goals, and waiting makes ants slower than their hunger clock; since thinking is about a quarter of a tick, the page has every ant think every tick.

**Demo.** "Yumon Ant" (https://claude.ai/artifact/9EEF76QG5QgoLB1NfoPhsD, private): pan and zoom the endless ground, watch (inspect an ant's feeling, task, decision, drives and network = rule), drop food (crumb 1, seed 2, berry 3, grasshopper 5 ants), pour sand (it buries the entrance; the diggers clear it), dig, scare, found new colonies, overlay any field, live check, WIR sources.

```
cd wir
cargo run --release -- ant --out examples --export ../tmp/wir-ant/ant_demo.json
python demos/ant/build.py ../tmp/wir-ant/ant_demo.json ../tmp/wir-ant/yumon-ant.html
node demos/ant/harness.js 10000 1 check ../tmp/wir-ant/ant_demo.json
cargo test --test ant
```

## Yumon Park: an ecosystem that learns (Nov 8, 2026)

A side-view park with four bands (sky, trees, ground, water, top to bottom) and 30 species. Every Yumon carries its own copy of one WIR brain. Founders start with the compiled weights, so at first they decide exactly as written. After that each one learns from its own life, and a child starts from its mother's weights. `wir park` (`src/park.rs`) writes and checks the brain. The host is plain JavaScript (`wir/demos/park/park_core.js`).

**Capabilities and species.** 17 capabilities, each carried by a limb whose size gene sets how well it works: swim (fins), walk (legs), climb (claws), fly (wings), bite (jaws), claw (claws), sting (stinger), horn (horns), shell, spines, camouflage (skin), social, and five diets (graze grass, browse fruit, filter plankton, hawk gnats, scavenge meat). Claws carry both climbing and clawing, so one gene moves both. Species are permutations: every 1- and 2-way movement set (10) cycled across species, a weapon set, a guard, diets from the plants in their bands, social at 30%. Roles fix the shape of the food web: half plant eaters, a quarter omnivores, a quarter hunters. Fliers stay light (size <= 1.15) because wings must lift the body (`wings / (0.5 + 0.35 m + 0.4 shell) >= 0.9`); a lineage that outgrows its wings is grounded. Every limb adds upkeep. A Yumon can eat another species up to 1.3x its mass (2x with venom). Plant eaters are prey only; juveniles start at 55% size, so young hunters are prey too.

**Brain** (0.2 program, 20 inputs: current option, hunger, fatigue, hurt, fertile, threat / food / prey / mate distance buckets 0..3, refuge, kin, camo, social):

```
let hunt = select(and(gt(pd, 0), ge(hunger, 3)), hunger + 2*pd - 2, 0)
let flee = select(scared, 3*td + 2*refuge + 2*hurt + 2, 0)
let hide = select(and(camo, scared, lt(td, 3)), 3*td + 6, 0)
let sleep = select(scared, 0, 2*clip(fatigue - 3, 0, 7))
let roam = 2 + select(and(eq(fd, 0), eq(pd, 0)), min(hunger, 5), 0)
output key1 = 8*hunt + 16*o1 + 6
```

Hidden [34, 24, 21], 2,255 weights (227 non-zero), all dyadic, so a Float32 copy is exact (8.8 KB per Yumon; the cache keeps the current weights plus the birth weights). The host samples an option from softmax(keys / 4); a drive unit is 8 key units.

**Learning.** Reward is homeostatic: each tick pays 20 x the drop in need = 4 hunger^2 + 2 fatigue^2 + 6 hurt^2 (fractions of max), plus 1 for mating and 1 for giving birth. Each Yumon keeps up to 360 (input, option, reward) steps. When it falls asleep it takes one REINFORCE step (returns with gamma 0.97, normalized advantages, lr 0.003, gradient norm clipped at 2) plus a pull of 0.002 toward its birth weights, then clears the buffer. A child copies its mother's current weights and they become its anchor. At death the weights leave the cache. Learning costs ~1.8 ms per tick at 600-900 Yumon.

**Genes.** Size, eyes, hue and the species' limbs. A child takes, gene by gene, a uniform blend of its parents' values, then one gene mutates (x exp(0.08 N(0,1)); hue +-8 degrees). Traits (speeds per band, attack, armour, sense radius, upkeep, lift) are recomputed from genes while young.

**Checks.** `wir park`: 200,000 random inputs, 0 key mismatches, worst |diff| 0. The page carries 256 test vectors; its rule port and the compiled weights match all of them on load, and every untrained brain is checked live: 0 of 4.6M untrained decisions off the rule (seed 3, no learning). `tests/park.rs` checks written choices (hungry hunter with prey in view hunts, camouflaged freezes, close threat flees, tired and threatened never sleeps).

**What happened** (`node demos/park/harness.js 10000 <mode> <seed>`, Node 24, i5-12500):

| seed | no learning: species / Yumon | learning: species / Yumon | trained decisions off the rule |
|---|---|---|---|
| 1 | 11 / 750 | 8 / 466 | 16.4% |
| 2 | 12 / 677 | 14 / 891 | 20.9% |
| 3 | 9 / 718 | 11 / 647 | 17.6% |

About two thirds of the 30 species go extinct by tick 10,000 either way, mostly early starvation of hunters and of species crowded off a shared plant. Learning changes behaviour a lot: most common changes (seed 3) are sleeping where the rule would eat, eating through a threat the rule would flee, roaming instead of eating, fleeing where the rule would eat, and hunting where the rule would eat. With learning on, 81% of sampled decisions match the rule's top option, vs 96% with sampling alone.

Learning did not measurably help survival. In a mixed run (half the founders of each species learn, the trait passes mother to child) learner lineages ended at 40.7%, 50.0% and 48.5% of the population at tick 12,000 (seeds 1-3). At lr 0.01 they fell to 10.2%, 28.5% and 15.3%; at lr 0.03, 63.6%, 24.4% and 10.5%. So at the default rate learning is about neutral, and stronger learning loses to the written brain.

**What did not work.** The first species mix (most species armed) ran out of prey: 22 of 30 species died out by tick 6,000 and hunters made 2 kills. Predators then starved with prey in view because the written rule let hunger raise roaming above hunting whenever no plant food was in sight; roaming now grows with hunger only when neither food nor prey is visible. Prey was faster than the bigger hunters until hunters got a tiring pounce (1.4x speed). Litters of two let small species fill the park. A reward of energy eaten minus damage taught brains to skip sleep and ignore threats (sleep never paid); the homeostatic reward fixed that. A Yumon that dies never learns from its death, since its buffer goes with it: only near misses (damage survived) teach caution.

**Spawning, patches and commitment (second pass, Nov 8).** Measured with a headless probe (seeds 1-2, no learning): species started in one spot (spread 0.02, where 0 is every member at one x and 1 is evenly round the ring) and the median Yumon moved 32-39 units of x per 500 ticks in a 2,400-wide world. Three changes:

- Founders arrive in mixed-sex groups of 2-3, each group at its own random x: start spread 0.54-0.59.
- Roaming keeps a goal 300-900 units away (70% in the same direction as the last one) for up to 700 ticks at 0.75 pace, and every band speed is 1.4x. Alone this left median travel at 30-45: roaming was ~6% of time because evenly carpeted food (a tuft every 24 units) kept everyone eating where they stood.
- Plants grow in patches with sparse stretches between: 8 meadows, 7 fruit groves, 7 drifting plankton blooms, 6 drifting gnat swarms, with the same total regrowth. An option, once sampled, is kept for 12 ticks unless the brain gives it under 5% (a semi-MDP decision; rewards in between go to it; gamma 0.85 per decision, min buffer 16).

Option switches fell from 11.5-12.3 to 5.1 per 100 ticks; the 90th percentile of travel went from 119-123 to 171-187 units per 500 ticks (max 806). The median stayed at 41-48: a fed plant eater stays in its meadow. At tick 10,000 (no learning) 12 and 10 species were alive (435 and 532 Yumon). Training per sleep covers whole decisions, about 10x cheaper (1.1 s of training over 10,000 ticks, seed 1). Mixed runs at tick 12,000: learners 66.3%, 55.0%, 39.2% (seeds 1-3), still about neutral on average. The table above is from the first pass.

**Moving patches (third pass, Nov 8).** Fixed patches still let a group settle in one meadow. Now a patch withers when eaten below 10% of its full amount or when its season ends (2,000-4,000 ticks, 400-tick grace after sprouting), and sprouts at 35% full 150-450 units along its band (`reformPatches`; about 90-130 moves per 10,000 ticks). Results (no learning, tick 10,000, seeds 1-3):

| version | travel per 500 ticks, median / p90 | species alive | Yumon |
|---|---|---|---|
| fixed patches (second pass, seeds 1-2) | 41-48 / 171-187 | 12, 10 | 435, 532 |
| patches move anywhere, sprout at 10% | 54-61 / 202-233 | 5, 3 | 99, 265 |
| moves 250-700, sprout 35%, plants x1.4 | 51-57 / 182-247 | 6, 3, 8 | 361, 580, 346 |
| moves 150-450, plants x1.25 | 44-63 / 195-249 | 6, 2, 4 | 254, 720, 239 |
| same plus mating calls (current) | 48-67 / 212-229 | 8, 9, 5 | 554, 553, 304 |

Moving food costs the ecosystem food (sprouts start part grown, eaters lose track), so total plant growth is 1.25x the even-carpet figure. The bigger loss was demographic: scattered founders following moving food rarely met a mate, and small species died of old age without replacing themselves. Mating calls (a fertile Yumon hears fertile mates of its species at 2.5x its sense radius; the mate bucket uses that radius) brought survival back to 5-9 species, still below the 10-12 of fixed patches.

**Diet setups (Nov 8).** `createPark` takes `diets`: `mixed` (default), `hunters` (every species armed, still eating the plants of its bands) or `carnivores` (armed, meat only), plus `movingPatches`, and tuning overrides (`plantRate`, `call`, `lr`; the tuning object is now per park). No learning, tick 10,000, seeds 1-2 (`PARK_SETUP='{"diets":"..."}' node harness.js 10000 nolearn SEED`):

| diets | species alive | Yumon | kills | births |
|---|---|---|---|---|
| mixed (current, seeds 1-3) | 8, 9, 5 | 554, 553, 304 | 151, 295, 170 | 1,379, 1,529, 876 |
| all hunters | 5, 8 | 158, 304 | 43, 66 | 407, 891 |
| all carnivores | 0, 0 (extinct by tick 5,000) | 0 | 49, 42 | 6, 2 |

Meat only has no energy source: plants are the only way energy enters, so the founders' bodies are all there is and the largest species have no prey (prey can be at most 1.3x the hunter's mass, 2x with venom). All hunters is more direct competition and keeps fewer, leaner populations; kills stay low because prey must be smaller and everyone flees everyone. The page has an Ecosystem setup tab (diets, species 8-40, seed, plant growth, fixed or moving patches, mating-call range, learning) with Save & restart; settings stay in the viewer's browser.

**Demo.** "Yumon Park" (private artifact, link in the cc-manager card `yumon-park`): pan and zoom, inspect a Yumon (feelings, its brain's option probabilities with the rule's choice tagged, genes against its species' founders, whether its brain is still exact), plant food, toggle learning and its rate, reintroduce extinct species, population lines, most common learned changes.

```
cd wir
cargo run --release -- park --out examples --export ../tmp/wir-park/park_demo.json
python demos/park/build.py ../tmp/wir-park/park_demo.json ../tmp/wir-park/yumon-park.html
node demos/park/harness.js 10000 mixed 1 ../tmp/wir-park/park_demo.json
cargo test --test park
```
