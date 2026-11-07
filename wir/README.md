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
```

`decompile` options: `--keys W1,b1,W2,b2` for tensors with other names (slices
of larger checkpoints), `--transpose` for `[in, out]` storage (Burn `Linear`),
`--input`/`--outputs` to name things, `--exact` for lossless numbers.

## Weight files

Layout is PyTorch `nn.Linear`: W1 `[H, D]`, b1 `[H]`, W2 `[O, H]`, b2 `[O]`.

| format | tensor names written | notes |
| --- | --- | --- |
| `.npz` | `W1 b1 W2 b2` + `wir_meta` | `np.load` works; read side needs `np.savez`, not `savez_compressed` |
| `.safetensors` | `0.weight 0.bias 2.weight 2.bias` | loads into `nn.Sequential(Linear, ReLU, Linear)` with `load_state_dict` |
| `.json` | `W1 b1 W2 b2` | human-readable |

All three accept any of these names on read. f64 by default, `--dtype f32` on
compile. F16/BF16 safetensors and `<f2` npy are readable.

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

## Tests

```
cargo test                      # 51 tests, ~20 s debug (fuzz is the slow one)
uv run --project ../python-train python scripts/interop.py   # NumPy 2.5 / PyTorch 2.14 cross-check
```

`tests/fuzz.rs` is the weights round-trip acceptance test: 1,000 random networks
(D 1-3, 1-16 units, including constant, dead, duplicated and opposite units)
through decompile, exact print and compile, each compared on 10,000 points.
`tests/decompile.rs` checks that scaling and permuting hidden units leaves the
text unchanged; `tests/roundtrip.rs` checks `decompile(compile(src)) ==
canonicalize(src)` on 600 generated programs.
