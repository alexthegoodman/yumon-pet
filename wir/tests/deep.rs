//! WIR 0.2: combinators, layering, the Pong example against a reference
//! simulator, and the deep decompiler.

mod common;

use common::*;
use wir::deep::{self, DeepWeights, Layer};
use wir::{Mat, NumFormat};

const PONG: &str = include_str!("../examples/pong.wir");

fn compile(src: &str) -> deep::DeepCompiled {
    deep::compile(src, 1e-6).unwrap_or_else(|e| panic!("{e}\n{src}"))
}

/// The same step as examples/pong.wir, written as ordinary branching code.
fn pong_step(s: [i64; 8]) -> [i64; 8] {
    let [bx, by, vx, vy, pl, pr, ul, ur] = s;
    let pl2 = (pl + ul).clamp(0, 9);
    let pr2 = (pr + ur).clamp(0, 9);
    let ny = by + vy;
    let (y, vy2) = if ny < 0 {
        (-ny, -vy)
    } else if ny > 11 {
        (22 - ny, -vy)
    } else {
        (ny, vy)
    };
    let nx = bx + vx;
    let (mut bx2, mut by2, mut vx2, mut point_l, mut point_r) = (nx, y, vx, 0, 0);
    if nx < 1 || nx > 14 {
        let (paddle, reflected) = if nx < 1 { (pl2, 2 - nx) } else { (pr2, 28 - nx) };
        vx2 = -vx;
        if paddle <= y && y <= paddle + 2 {
            bx2 = reflected;
        } else {
            (bx2, by2) = (8, 6);
            if nx < 1 { point_r = 1 } else { point_l = 1 }
        }
    }
    [pl2, pr2, bx2, by2, vx2, vy2, point_l, point_r]
}

fn pong_states(pr_values: &[i64]) -> impl Iterator<Item = [i64; 8]> + '_ {
    let pm = [-1, 1];
    let ctl = [-1, 0, 1];
    (1..=14).flat_map(move |bx| {
        (0..=11).flat_map(move |by| {
            pm.into_iter().flat_map(move |vx| {
                pm.into_iter().flat_map(move |vy| {
                    (0..=9).flat_map(move |pl| {
                        pr_values.iter().flat_map(move |&pr| {
                            ctl.into_iter().flat_map(move |ul| ctl.into_iter().map(move |ur| [bx, by, vx, vy, pl, pr, ul, ur]))
                        })
                    })
                })
            })
        })
    })
}

/// Every state with the right paddle at 0, 4 or 9 (181,440 states; the left side
/// is exhaustive and the right side mirrors it): the network matches the simulator.
#[test]
fn pong_network_matches_simulator() {
    let c = compile(PONG);
    assert_eq!(c.weights.outputs, ["pl2", "pr2", "bx2", "by2", "vx2", "vy2", "point_l", "point_r"]);
    let (mut n, mut worst) = (0, 0.0f64);
    for s in pong_states(&[0, 4, 9]) {
        let x: Vec<f64> = s.iter().map(|v| *v as f64).collect();
        let got = c.weights.eval(&x);
        let want = pong_step(s);
        for (g, w) in got.iter().zip(want) {
            worst = worst.max((g - w as f64).abs());
        }
        assert!(got.iter().zip(want).all(|(g, w)| (g - w as f64).abs() < 1e-9), "state {s:?}: network {got:?}, simulator {want:?}");
        n += 1;
    }
    assert_eq!(n, 181_440);
    eprintln!("pong: {n} states, worst |diff| {worst:e}, hidden {:?}, {} parameters", c.weights.hidden(), c.weights.param_count());
}

/// A thousand-step game driven by the network alone stays on the simulator's track.
#[test]
fn pong_plays_a_game() {
    let c = compile(PONG);
    let mut rng = Rng::new(7);
    let mut s = [3, 4, 1, -1, 4, 4, 0, 0];
    let mut points = [0, 0];
    for t in 0..1000 {
        s[6] = rng.below(3) as i64 - 1;
        s[7] = rng.below(3) as i64 - 1;
        let want = pong_step(s);
        let got: Vec<i64> = c.weights.eval(&s.map(|v| v as f64)).iter().map(|v| v.round() as i64).collect();
        assert_eq!(got, want, "step {t} from {s:?}");
        points[0] += want[6];
        points[1] += want[7];
        s = [want[2], want[3], want[4], want[5], want[0], want[1], 0, 0];
    }
    assert!(points[0] + points[1] > 0, "random paddles should miss sometimes");
}

/// Layering is exact: off the integer grid the network still equals the program's own graph.
#[test]
fn layers_match_the_graph_off_grid() {
    let c = compile(PONG);
    let mut rng = Rng::new(3);
    let domain = [(1.0, 14.0), (0.0, 11.0), (-1.0, 1.0), (-1.0, 1.0), (0.0, 9.0), (0.0, 9.0), (-1.0, 1.0), (-1.0, 1.0)];
    for _ in 0..20_000 {
        let x: Vec<f64> = domain.iter().map(|(lo, hi)| rng.uniform(*lo, *hi)).collect();
        let (a, b) = (c.weights.eval(&x), c.program.eval(&x));
        assert!(a.iter().zip(&b).all(|(p, q)| (p - q).abs() < 1e-9), "{x:?}: {a:?} vs {b:?}");
    }
}

#[test]
fn combinators_are_exact() {
    let src = "wir 0.2
input a, b: R1 in [-3, 3]
output mx = max(a, b, 0)
output mn = min(a, b)
output ab = abs(a - b)
output cl = clip(2*a + 1, -1, 2)
output st = step(a)
";
    let c = compile(src);
    let mut rng = Rng::new(5);
    for _ in 0..5000 {
        let (a, b) = (rng.uniform(-3.0, 3.0), rng.uniform(-3.0, 3.0));
        let y = c.weights.eval(&[a, b]);
        let want = [a.max(b).max(0.0), a.min(b), (a - b).abs(), (2.0 * a + 1.0).clamp(-1.0, 2.0), a.clamp(0.0, 1.0)];
        assert!(y.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-12), "({a}, {b}): {y:?} vs {want:?}");
    }
}

#[test]
fn logic_and_select_on_bits_and_integers() {
    let src = "wir 0.2
input p, q: R1 in [0, 1]
input m, n: R1 in [-5, 5]
output both = and(p, q)
output either = or(p, q)
output neither = not(or(p, q))
output pick = select(p, m, n)
output same = eq(m, n)
output greater = gt(m, n)
output at_most = le(m, n)
";
    let c = compile(src);
    for p in [0, 1] {
        for q in [0, 1] {
            for m in -5..=5 {
                for n in -5..=5 {
                    let y = c.weights.eval(&[p as f64, q as f64, m as f64, n as f64]);
                    let b = |v: bool| v as i64 as f64;
                    let want = [b(p & q == 1), b(p | q == 1), b(p | q == 0), if p == 1 { m } else { n } as f64, b(m == n), b(m > n), b(m <= n)];
                    assert!(y.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-9), "{p} {q} {m} {n}: {y:?} vs {want:?}");
                }
            }
        }
    }
}

#[test]
fn intervals_elide_relus_that_cannot_bend() {
    // a is never negative on its domain, so relu(a) is a itself: no hidden layer.
    let c = compile("wir 0.2\ninput a: R1 in [2, 5]\noutput y = relu(a) + max(a, 0)\n");
    assert!(c.weights.hidden().is_empty(), "{:?}", c.weights.hidden());
    // The same program without a domain needs the unit.
    let c = compile("wir 0.2\ninput a: R1\noutput y = relu(a)\n");
    assert_eq!(c.weights.hidden(), [1]);
}

#[test]
fn errors_name_the_statement() {
    let cases = [
        ("wir 0.2\ninput a, b: R1\noutput y = a * b\n", "constant on one side"),
        ("wir 0.2\ninput a, c: R1\noutput y = select(c, a, 0)\n", "select needs bounded values"),
        ("wir 0.2\ninput a, b: R1\noutput y = clip(a, 0, b)\n", "clip bounds must be constants"),
        ("wir 0.2\ninput a: R1\noutput y = a + z\n", "unknown name `z`"),
        ("wir 0.2\ninput a: R1\noutput y = frob(a)\n", "unknown function `frob`"),
        ("wir 0.2\ninput a: R1\nlet max = a\noutput y = a\n", "`max` is reserved"),
        ("wir 0.2\ninput a: R1 in [0, 1]\noutput y = 2*a\nassert y(1) == 3\n", "assert failed: y(1) = 2 (expected 3)"),
        ("wir 0.2\ninput a: R1\noutput y = a\nassert y(1, 2) == 1\n", "needs 1 values"),
    ];
    for (src, msg) in cases {
        let e = deep::compile(src, 1e-6).unwrap_err();
        assert!(e.to_string().contains(msg), "expected `{msg}` in `{e}`");
        assert!(e.line.is_some(), "no line in `{e}`");
    }
}

fn random_deep(rng: &mut Rng, d: usize, widths: &[usize], o: usize) -> DeepWeights {
    let mut layers = Vec::new();
    let mut cols = d;
    for &rows in widths.iter().chain([&o]) {
        let mut w = Mat::zeros(rows, cols);
        for v in w.data.iter_mut() {
            *v = if rng.chance(0.15) { 0.0 } else { rng.uniform(-2.0, 2.0) };
        }
        layers.push(Layer { w, b: (0..rows).map(|_| rng.uniform(-1.0, 1.0)).collect() });
        cols = rows;
    }
    DeepWeights { layers, inputs: (1..=d).map(|i| format!("x{i}")).collect(), outputs: (1..=o).map(|i| format!("y{i}")).collect() }
}

fn deep_diff(rng: &mut Rng, a: &DeepWeights, b: &DeepWeights, n: usize) -> f64 {
    (0..n)
        .map(|_| {
            let x = random_point(rng, a.input_dim(), 3.0);
            a.eval(&x).iter().zip(b.eval(&x)).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max)
        })
        .fold(0.0, f64::max)
}

/// 500 random deep networks (1-3 inputs, 2-3 hidden layers of 1-8 units, 1-3 outputs):
/// weights -> exact 0.2 text -> weights is the same function.
#[test]
fn random_deep_networks_round_trip() {
    let mut rng = Rng::new(2027);
    let mut worst = 0.0f64;
    for case in 0..500 {
        let d = 1 + case % 3;
        let widths: Vec<usize> = (0..2 + rng.below(2)).map(|_| 1 + rng.below(8)).collect();
        let o = 1 + rng.below(3);
        let w = random_deep(&mut rng, d, &widths, o);
        let text = deep::decompile(&w, NumFormat::EXACT).unwrap();
        let back = compile(&text).weights;
        let diff = deep_diff(&mut rng, &w, &back, 2000);
        worst = worst.max(diff);
        assert!(diff <= 1e-9, "case {case} {widths:?}: max |diff| {diff:e}\n{text}");
    }
    eprintln!("deep round-trip: worst |diff| over 500 networks x 2,000 points = {worst:e}");
}

/// Positive scaling and permutation of hidden units leave the decompiled text unchanged.
#[test]
fn deep_decompile_ignores_scaling_and_permutation() {
    let mut rng = Rng::new(9);
    for case in 0..200 {
        let widths = [1 + rng.below(6), 1 + rng.below(6)];
        let w = random_deep(&mut rng, 2, &widths, 2);
        let mut v = w.clone();
        for k in 0..v.layers.len() - 1 {
            let n = v.layers[k].w.rows;
            let mut perm: Vec<usize> = (0..n).collect();
            rng.shuffle(&mut perm);
            let scale: Vec<f64> = (0..n).map(|_| rng.uniform(0.25, 4.0)).collect();
            let (cur, nxt) = (v.layers[k].clone(), v.layers[k + 1].clone());
            for (new, &old) in perm.iter().enumerate() {
                for c in 0..cur.w.cols {
                    v.layers[k].w.set(new, c, cur.w.get(old, c) * scale[old]);
                }
                v.layers[k].b[new] = cur.b[old] * scale[old];
                for r in 0..nxt.w.rows {
                    v.layers[k + 1].w.set(r, new, nxt.w.get(r, old) / scale[old]);
                }
            }
        }
        let (a, b) = (deep::decompile(&w, NumFormat::CANONICAL).unwrap(), deep::decompile(&v, NumFormat::CANONICAL).unwrap());
        assert_eq!(a, b, "case {case}");
    }
}

/// Pong survives weights -> canonical text -> weights, and both file formats.
#[test]
fn pong_decompiles_and_saves() {
    let c = compile(PONG);
    let text = deep::decompile(&c.weights, NumFormat::CANONICAL).unwrap();
    let back = compile(&text).weights;
    for s in pong_states(&[4]).step_by(7) {
        let x = s.map(|v| v as f64);
        let (a, b) = (c.weights.eval(&x), back.eval(&x));
        assert!(a.iter().zip(&b).all(|(p, q)| (p - q).abs() < 1e-6), "{s:?}: {a:?} vs {b:?}");
    }
    let dir = std::env::temp_dir().join(format!("wir-deep-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for name in ["pong.json", "pong.safetensors"] {
        let path = dir.join(name);
        deep::save(&c.weights, &path, wir::io::DType::F64).unwrap();
        let loaded = deep::load(&path).unwrap().expect("a deep network");
        assert_eq!(loaded, c.weights, "{name}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn pong_example_is_the_generator_default() {
    let generated = wir::games::pong_program(&wir::games::Pong::default()).unwrap();
    assert_eq!(PONG.replace("\r\n", "\n"), generated, "regenerate with `wir pong -o examples/pong.wir`");
}

/// The independent simulator above agrees with the library's on the default game.
#[test]
fn simulators_agree() {
    let g = wir::games::Pong::default();
    for s in pong_states(&[0, 4, 9]) {
        assert_eq!(wir::games::pong_step(&g, s), pong_step(s), "{s:?}");
    }
}

/// Generated variants compile, pass their own asserts and match the simulator.
#[test]
fn pong_variants_match_their_simulator() {
    use wir::games::{pong_check, pong_program, Pong};
    let variants = [
        Pong { width: 8, height: 6, paddle: 1, speed: 1, paddle_speed: 1 },
        Pong { width: 10, height: 7, paddle: 2, speed: 2, paddle_speed: 2 },
        Pong { width: 24, height: 18, paddle: 5, speed: 3, paddle_speed: 1 },
        Pong { width: 7, height: 4, paddle: 3, speed: 3, paddle_speed: 1 },
    ];
    for g in variants {
        let src = pong_program(&g).unwrap();
        let c = pong_check(&g, &src, 50_000, 1).unwrap_or_else(|e| panic!("{e}\n{src}"));
        assert_eq!(c.mismatches, 0, "{g:?}: {c:?}");
    }
    assert!(pong_program(&Pong { width: 6, height: 6, paddle: 2, speed: 4, paddle_speed: 1 }).is_err());
}
