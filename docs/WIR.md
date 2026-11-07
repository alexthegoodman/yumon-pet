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

**Planned for v0.2, depth via combinators:** `max(f,g)`, `min(f,g)`, `clip(f,lo,hi)`, `and(...)` of half-spaces, compiled to known ReLU constructions (for example max(f,g) = g + relu(f − g), with an identity passthrough for g). The author stays in input space and the compiler chooses depth. A raw feature-space escape hatch (`in space(a,b,c): crease ...`) comes alongside.

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

- [ ] `tent.wir` and `xor.wir` compile to exactly the expected weights in the syntax section, and all asserts pass.
- [ ] `xor_solve.wir` compiles with residual < 1e-9 and reproduces the XOR truth table.
- [ ] Each compile error case (anchor on plane, non-positive anchor, unsolvable `solve`, failed assert) has a test with a clear message naming the statement and line.
- [ ] `print(parse(src))` is idempotent: printing twice gives identical text.
- [ ] Round-trip source: `decompile(compile(src))` equals `canonicalize(src)` as text.
- [ ] Round-trip weights: for 1,000 random networks (D ∈ {1,2,3}, 1–16 hidden units), `compile(decompile(W))` matches W as a function on 10,000 sampled inputs within 1e-6.
- [ ] Scaling and permuting hidden units of any network leaves its decompiled text unchanged.
- [ ] The CLI prints a per-statement cost report on compile.

**Milestones**

1. Data model, parser and compiler pass the tent and XOR examples.
2. Printer and canonicalization pass the idempotence test.
3. Decompiler passes both round-trip tests and the symmetry test.
4. Fuzzing and CLI polish; then open v0.2 (combinators and depth).
