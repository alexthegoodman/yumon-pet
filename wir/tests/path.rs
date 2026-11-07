//! Grid (stencil) programs and the pathfinding step: the cell network against the
//! rule, whole grids against Dijkstra, the flat expansion, and Conv2d files.

mod common;

use common::*;
use wir::deep::{self, StencilWeights};
use wir::games::{self, PathGrid, PathRule};
use wir::io::DType;
use wir::NumFormat;

fn stencil(src: &str) -> StencilWeights {
    let c = deep::compile(src, 1e-6).unwrap_or_else(|e| panic!("{e}\n{src}"));
    StencilWeights::from_compiled(&c).expect("a grid program")
}

fn field_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    a.iter().flatten().zip(b.iter().flatten()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

#[test]
fn grid_programs_read_neighbours_and_pads() {
    // A 3x3 box sum with pad 1 outside: corners see 5 outside cells.
    let s = stencil("wir 0.2\ninput a: grid in [0, 9] pad 1\noutput s = a[-1,-1] + a[-1,0] + a[-1,1] + a[0,-1] + a + a[0,1] + a[1,-1] + a[1,0] + a[1,1]\n");
    assert_eq!(s.taps.len(), 9);
    assert_eq!(s.radius(), 1);
    let (h, w) = (3, 4);
    let a: Vec<f64> = (0..h * w).map(|i| (i % 7) as f64).collect();
    let got = s.apply(&[a.clone()], h, w);
    for r in 0..h as i64 {
        for c in 0..w as i64 {
            let mut want = 0.0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (y, x) = (r + dy, c + dx);
                    want += if y < 0 || x < 0 || y >= h as i64 || x >= w as i64 { 1.0 } else { a[(y * w as i64 + x) as usize] };
                }
            }
            assert_eq!(got[0][(r * w as i64 + c) as usize], want, "cell ({r}, {c})");
        }
    }
}

#[test]
fn grid_program_errors() {
    let cases = [
        ("wir 0.2\ninput a: grid\ninput b: R1\noutput y = a\n", "either all R1 or all grid"),
        ("wir 0.2\ninput a: grid in [1, 2]\noutput y = a\n", "default pad 0 is outside"),
        ("wir 0.2\ninput a: grid in [0, 2] pad 3\noutput y = a\n", "pad value must lie"),
        ("wir 0.2\ninput a: R1\nlet b = a\noutput y = b[1, 0]\n", "only grid inputs and the lets of grid programs take offsets"),
        ("wir 0.2\ninput a: grid\noutput y = a[1/2, 0]\n", "grid offsets are integers"),
        ("wir 0.2\ninput a: grid in [0, 1]\noutput y = a\nassert y(1) == 1\n", "do not take asserts"),
    ];
    for (src, msg) in cases {
        let e = deep::compile(src, 1e-6).unwrap_err();
        assert!(e.to_string().contains(msg), "expected `{msg}` in `{e}`");
    }
}

/// The default 4-neighbour step with a small ceiling: every combination of cell inputs
/// (13^4 * 2 * 2 = 114,244) matches the rule, which makes every grid step exact.
#[test]
fn path_cell_network_is_exact_on_every_input() {
    let g = PathRule { ceiling: 12, ..Default::default() };
    let src = games::path_program(&g).unwrap();
    let c = games::path_check(&g, &src, 1_000_000, 20, 8, 8, 3).unwrap();
    assert!(c.local_exhaustive);
    assert_eq!(c.local_cases, 13u64.pow(4) * 4);
    assert_eq!((c.local_mismatches, c.step_mismatches, c.solve_mismatches), (0, 0, 0), "{c:?}");
    assert_eq!(c.hidden, [6, 5, 2]);
}

#[test]
fn path_variants_match_dijkstra() {
    let rules = [
        PathRule { moves: 4, ceiling: 64, terrain: 1 },
        PathRule { moves: 4, ceiling: 40, terrain: 5 },
        PathRule { moves: 8, ceiling: 64, terrain: 1 },
        PathRule { moves: 8, ceiling: 90, terrain: 3 },
    ];
    for g in rules {
        let src = games::path_program(&g).unwrap();
        let c = games::path_check(&g, &src, 200_000, 15, 10, 9, 7).unwrap();
        assert_eq!((c.local_mismatches, c.step_mismatches, c.solve_mismatches), (0, 0, 0), "{}: {c:?}", g.tag());
        assert!(c.steps > 15, "{c:?}");
    }
}

/// Every 3x3 wall layout with every goal cell settles to the Dijkstra distances.
#[test]
fn path_all_3x3_mazes() {
    for moves in [4, 8] {
        let g = PathRule { moves, ceiling: 32, terrain: 1 };
        let (count, bad, most) = games::path_all_mazes(&g, &games::path_program(&g).unwrap(), 3, 3).unwrap();
        assert_eq!((count, bad), (512 * 9, 0), "moves {moves}");
        assert!(most >= 8, "a 3x3 grid has paths of length 8; settled in {most}");
    }
}

#[test]
fn self_healing_after_the_goal_moves() {
    let g = PathRule::default();
    let s = stencil(&games::path_program(&g).unwrap());
    let mut net = deep::SparseNet::new(&s.net);
    let mut grid = PathGrid::new(&g, 9, 9);
    grid.fields[2][0] = 1;
    let (field, _) = games::path_settle(&s, &mut net, &g, &grid, 500);
    grid.fields[0] = field;
    // Move the goal to the far corner: old low values must rise, not stick.
    grid.fields[2][0] = 0;
    grid.fields[2][80] = 1;
    let (field, steps) = games::path_settle(&s, &mut net, &g, &grid, 500);
    assert!(steps < 500);
    assert_eq!(field, games::path_distances(&g, &grid));
    assert_eq!(field[0], 16);
}

/// The flat program for a small grid computes the same step as the stencil.
#[test]
fn flat_program_matches_the_stencil() {
    for g in [PathRule { moves: 4, ceiling: 20, terrain: 3 }, PathRule { moves: 8, ceiling: 30, terrain: 1 }] {
        let s = stencil(&games::path_program(&g).unwrap());
        let flat = deep::compile(&games::path_flat_program(&g, 3, 4).unwrap(), 1e-6).unwrap();
        let mut rng = 11u64;
        for _ in 0..200 {
            let grid = games::path_random_grid(&g, 3, 4, &mut rng);
            let inputs = grid.inputs(&g);
            let x: Vec<f64> = inputs.iter().flatten().copied().collect();
            let a = flat.weights.eval(&x);
            let b = s.apply(&inputs, 3, 4).remove(0);
            assert_eq!(a, b, "{}", g.tag());
        }
    }
    let (hidden, params, bad) = games::path_flat_check(&PathRule::default(), 4, 4, 300, 5).unwrap();
    assert_eq!(bad, 0);
    assert!(params > 10 * stencil(&games::path_program(&PathRule::default()).unwrap()).param_count(), "{hidden:?} {params}");
}

#[test]
fn compiled_path_weights_are_integers() {
    for g in [PathRule::default(), PathRule { moves: 8, ceiling: 100, terrain: 4 }] {
        let s = stencil(&games::path_program(&g).unwrap());
        for (k, l) in s.net.layers.iter().enumerate() {
            assert!(l.w.data.iter().chain(&l.b).all(|v| v.fract() == 0.0), "{} layer {k}", g.tag());
        }
    }
}

fn random_grid_fields(rng: &mut Rng, channels: usize, h: usize, w: usize) -> Vec<Vec<f64>> {
    (0..channels).map(|_| (0..h * w).map(|_| rng.uniform(-3.0, 3.0)).collect()).collect()
}

/// Conv2d files (.safetensors, f64 and f32) and .json load back as the same network,
/// and the decompiled grid text compiles to the same function.
#[test]
fn stencil_files_and_decompile_round_trip() {
    let dir = std::env::temp_dir().join(format!("wir-path-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let g = PathRule { moves: 8, ceiling: 64, terrain: 3 };
    let s = stencil(&games::path_program(&g).unwrap());
    let mut rng = 5u64;
    let grids: Vec<PathGrid> = (0..20).map(|_| games::path_random_grid(&g, 7, 6, &mut rng)).collect();
    let want: Vec<Vec<Vec<f64>>> = grids.iter().map(|grid| s.apply(&grid.inputs(&g), 7, 6)).collect();
    for (file, dtype) in [("p.safetensors", DType::F64), ("p32.safetensors", DType::F32), ("p.json", DType::F64)] {
        let path = dir.join(file);
        deep::save_stencil(&s, &path, dtype).unwrap();
        assert!(deep::load(&path).unwrap().is_none(), "{file} must not load as a plain deep network");
        let back = deep::load_stencil(&path).unwrap().unwrap();
        assert_eq!(back.channels, s.channels);
        for (grid, want) in grids.iter().zip(&want) {
            // Integer weights and inputs: exact in f32 too.
            assert_eq!(field_diff(&back.apply(&grid.inputs(&g), 7, 6), want), 0.0, "{file}");
        }
    }
    let text = deep::decompile_stencil(&s, NumFormat::CANONICAL).unwrap();
    let again = stencil(&text);
    for (grid, want) in grids.iter().zip(&want) {
        assert!(field_diff(&again.apply(&grid.inputs(&g), 7, 6), want) < 1e-9, "{text}");
    }
    assert_eq!(deep::decompile_stencil(&again, NumFormat::CANONICAL).unwrap(), text, "canonical text is a fixed point");
    std::fs::remove_dir_all(&dir).ok();
}

/// Random Conv2d stacks (as a learned planner would be): conv weights -> exact grid
/// text -> weights is the same function on random fields.
#[test]
fn random_conv_stacks_round_trip() {
    let mut rng = Rng::new(31);
    let mut worst = 0.0f64;
    for case in 0..100 {
        let c = 1 + rng.below(3);
        let k = [1, 3, 5][rng.below(3)];
        let widths: Vec<usize> = (0..1 + rng.below(3)).map(|_| 1 + rng.below(6)).collect();
        let mut layers = Vec::new();
        let mut cin = c;
        for (i, &u) in widths.iter().chain([&(1 + rng.below(2))]).enumerate() {
            let kk = if i == 0 { k } else { 1 };
            let w: Vec<f64> = (0..u * cin * kk * kk).map(|_| if rng.chance(0.1) { 0.0 } else { rng.uniform(-1.0, 1.0) }).collect();
            let b: Vec<f64> = (0..u).map(|_| rng.uniform(-0.5, 0.5)).collect();
            layers.push((vec![u, cin, kk, kk], w, b));
            cin = u;
        }
        let names: Vec<String> = (0..c).map(|i| format!("c{i}")).collect();
        let pads: Vec<f64> = (0..c).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let s = StencilWeights::from_conv(&layers, names, pads, None).unwrap();
        let text = deep::decompile_stencil(&s, NumFormat::EXACT).unwrap();
        let back = stencil(&text);
        let (h, w) = (2 + rng.below(5), 2 + rng.below(5));
        for _ in 0..5 {
            let f = random_grid_fields(&mut rng, c, h, w);
            let d = field_diff(&s.apply(&f, h, w), &back.apply(&f, h, w));
            worst = worst.max(d);
            assert!(d < 1e-9, "case {case}: {d:e}\n{text}");
        }
        assert_eq!(s.conv_layers().len(), layers.len());
    }
    eprintln!("conv round-trip: worst |diff| over 100 stacks = {worst:e}");
}

#[test]
fn path_example_is_the_generator_default() {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/path.wir")).unwrap();
    assert_eq!(src.replace("\r\n", "\n"), games::path_program(&PathRule::default()).unwrap());
}
