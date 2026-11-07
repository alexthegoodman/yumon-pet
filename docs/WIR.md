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
let nx = bx + vx                   # affine arithmetic; * and / by constants only
let at_l = lt(nx, 1)
output bx2 = select(at_l, 2, nx)
assert bx2(1, -1) == 2             # one value per input, in declaration order
```

- **Combinators**, each an exact composition of relu and affine maps: `relu`, `abs`, `max`/`min` (n-ary; max(a, b) = b + relu(a - b)), `clip(a, lo, hi)` with constant bounds, `step(a)` = clip(a, 0, 1), comparisons `gt ge lt le eq` (steps of a - b, so exact when a - b is an integer: `ge(a, b)` = step(a - b + 1)), `and`/`or`/`not` (exact on bits: and = relu(sum - (n - 1)), or = sum - relu(sum - 1)), and `select(c, a, b)` for a bit c, via the big-M product c d = relu(d - M+ (1 - c)) - relu(-d - M- (1 - c)).
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
