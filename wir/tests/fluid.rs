//! Lets read at offsets, `conserve` proofs, and the cellular water step: the cell
//! network against the rule, whole grids against the simulator, total water.

use wir::deep::{self, StencilWeights};
use wir::fluid::{self, FluidGrid, FluidRule};

fn stencil(src: &str) -> StencilWeights {
    let c = deep::compile(src, 1e-6).unwrap_or_else(|e| panic!("{e}\n{src}"));
    StencilWeights::from_compiled(&c).expect("a grid program")
}

/// A let read at an offset is the let's expression with every read moved, including
/// reads of other lets (nested shifts add up).
#[test]
fn lets_read_at_offsets() {
    let s = stencil("wir 0.2\ninput a: grid in [0, 9] pad 0\nlet r = a + 2*a[0, 1]\nlet q = max(r, r[1, 0])\noutput y = q[0, -1] - r\n");
    let mut taps = s.taps.clone();
    taps.sort();
    // q is also lowered where it is defined, so a[1, 1] is a tap (with zero weights).
    assert_eq!(taps, vec![(0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 1, -1), (0, 1, 0), (0, 1, 1)]);
    let (h, w) = (4, 5);
    let a: Vec<f64> = (0..h * w).map(|i| ((i * 7 + 3) % 10) as f64).collect();
    let get = |r: i64, c: i64| if r < 0 || c < 0 || r >= h as i64 || c >= w as i64 { 0.0 } else { a[(r * w as i64 + c) as usize] };
    let rr = |r: i64, c: i64| get(r, c) + 2.0 * get(r, c + 1);
    let out = s.apply(&[a.clone()], h, w);
    for r in 0..h as i64 {
        for c in 0..w as i64 {
            let want = rr(r, c - 1).max(rr(r + 1, c - 1)) - rr(r, c);
            assert_eq!(out[0][(r * w as i64 + c) as usize], want, "cell ({r}, {c})");
        }
    }
}

/// Two programs that move a cell's surplus to its right-hand neighbour. With a wall
/// channel gating the flow, nothing leaves the grid and the proof goes through; without
/// it, the last column pours into the pad, and a missing inflow leaks everywhere.
#[test]
fn conserve_proofs_and_refusals() {
    let ok = "wir 0.2\ninput a: grid in [0, 9] pad 0\ninput wall: grid in [0, 1] pad 1\nlet f = select(not(wall[0, 1]), relu(a - a[0, 1]), 0)\noutput b = a - f + f[0, -1]\nconserve b = a\n";
    let c = deep::compile(ok, 1e-6).unwrap();
    let p = &c.program.conserves[0];
    assert!(p.classes == 1 && p.boundary_cases > 0, "{p:?}");
    let bad = [
        (ok.replace("select(not(wall[0, 1]), relu(a - a[0, 1]), 0)", "relu(a - a[0, 1])"), "corner of the grid"),
        (ok.replace("- f + f[0, -1]", "- f"), "does not cancel"),
        (ok.replace("output b = a - f + f[0, -1]", "output b = a - f + f[0, -1] + 1"), "constant term"),
        (ok.replace("conserve b = a", "conserve b = q"), "not a grid input"),
        ("wir 0.2\ninput a: R1\noutput b = a\nconserve b = a\n".to_string(), "for grid programs"),
    ];
    for (src, msg) in bad {
        let e = deep::compile(&src, 1e-6).unwrap_err();
        assert!(e.to_string().contains(msg), "expected `{msg}` in `{e}`");
    }
}

/// Capacity 2: every combination of the cell network's 15 taps (3^9 * 2^8) matches the
/// rule, and random grids match the simulator with total water unchanged.
#[test]
fn small_fluid_is_exact_on_every_cell_input() {
    let g = FluidRule { capacity: 2 };
    let src = fluid::fluid_program(&g).unwrap();
    let c = fluid::fluid_check(&g, &src, 6_000_000, 6, 120, 10, 14, 3).unwrap();
    assert!(c.local_exhaustive && c.local_cases == 5_038_848, "{c:?}");
    assert_eq!((c.local_mismatches, c.step_mismatches, c.mass_changes, c.bound_violations), (0, 0, 0, 0), "{c:?}");
}

#[test]
fn default_fluid_matches_the_simulator() {
    let g = FluidRule::default();
    let src = fluid::fluid_program(&g).unwrap();
    let c = deep::compile(&src, 1e-6).unwrap();
    assert_eq!(c.program.conserves.len(), 1);
    let check = fluid::fluid_check(&g, &src, 200_000, 8, 150, 12, 18, 5).unwrap();
    assert_eq!((check.local_mismatches, check.step_mismatches, check.mass_changes, check.bound_violations), (0, 0, 0, 0), "{check:?}");
}

/// A block of water dropped into a walled basin falls, spreads and settles: full rows at
/// the bottom, and a top row whose cells differ by at most one unit.
#[test]
fn water_settles_level() {
    let g = FluidRule::default();
    let s = stencil(&fluid::fluid_program(&g).unwrap());
    let (h, w) = (12, 20);
    let mut grid = FluidGrid::new(h, w);
    for r in 0..h {
        for c in 0..w {
            let i = r * w + c;
            grid.fields[1][i] = (r + 1 == h || c == 0 || c + 1 == w) as i64;
            if (1..5).contains(&r) && (2..8).contains(&c) {
                grid.fields[0][i] = g.capacity;
            }
        }
    }
    let mass = grid.mass();
    let mut net = deep::SparseNet::new(&s.net);
    for _ in 0..3000 {
        let out = s.apply_with(&mut net, &grid.inputs(), h, w);
        grid.fields[0] = out[0].iter().map(|v| v.round() as i64).collect();
        grid.fields[2] = out[1].iter().map(|v| v.round() as i64).collect();
    }
    assert_eq!(grid.mass(), mass);
    // 24 full cells over an 18-wide basin: one full row, a top row of 6/18 of a row.
    let row = |r: usize| grid.fields[0][r * w + 1..r * w + w - 1].to_vec();
    assert!(row(h - 2).iter().all(|&v| v == g.capacity), "{:?}", row(h - 2));
    let top = row(h - 3);
    let (lo, hi) = (*top.iter().min().unwrap(), *top.iter().max().unwrap());
    assert!(hi - lo <= 1, "top row {top:?}");
    assert!((0..h - 3).all(|r| row(r).iter().all(|&v| v == 0)));
}
