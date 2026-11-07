//! Compact syntax (vectors, `ramp`, vector readouts), gates (SwiGLU), `patch`
//! directions and domains.

mod common;

use common::*;
use wir::{canonicalize, compile, decompile, lower_str, print, DecompileOptions, Mat, NumFormat, Weights};

fn swiglu_network(rng: &mut Rng, d: usize, h: usize, o: usize) -> Weights {
    let m = |rng: &mut Rng, r: usize, c: usize| Mat {
        rows: r,
        cols: c,
        data: (0..r * c).map(|_| rng.uniform(-1.0, 1.0)).collect(),
    };
    let v = |rng: &mut Rng, n: usize| (0..n).map(|_| rng.uniform(-0.5, 0.5)).collect::<Vec<_>>();
    Weights {
        w1: m(rng, h, d),
        b1: v(rng, h),
        w2: m(rng, o, h),
        b2: v(rng, o),
        up: Some((m(rng, h, d), v(rng, h))),
        input: None,
        outputs: None,
        shape: None,
    }
}

fn rel_diff(rng: &mut Rng, a: &Weights, b: &Weights, n: usize) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for _ in 0..n {
        let x = random_point(rng, a.input_dim(), 2.0);
        for (ya, yb) in a.eval(&x).iter().zip(b.eval(&x)) {
            num += (ya - yb) * (ya - yb);
            den += ya * ya;
        }
    }
    (num / den).sqrt()
}

#[test]
fn vector_literals_and_dot_readouts() {
    let src = "wir 0.1\ninput x: R4\noutput y: R1\n\
               crease a: ramp along [10 -5 0 25] / 10 above 1/2\n\
               crease b: ramp along [0 0 1 0]\n\
               y = [3 -1] / 2 . (a, b) + 1\n";
    let c = compile(src, Default::default()).unwrap();
    assert_eq!(c.weights.w1.row(0), &[1.0, -0.5, 0.0, 2.5]);
    assert_eq!(c.weights.b1, vec![-0.5, 0.0]);
    assert_eq!(c.weights.w2.row(0), &[1.5, -0.5]);
    assert_eq!(c.weights.b2, vec![1.0]);
    let x = [1.0, 0.0, 2.0, 0.0];
    assert_eq!(c.weights.eval(&x)[0], 1.0 + 1.5 * 0.5 - 0.5 * 2.0);
}

#[test]
fn name_ranges_expand() {
    let src = "wir 0.1\ninput x: R4\noutput y: R1\n\
               crease c1: ramp along [1 0 0 0]\ncrease c2: ramp along [0 1 0 0]\ncrease c3: ramp along [0 0 1 0]\n\
               y = [1 2 3] . (c1..c3)\n";
    let w = compile(src, Default::default()).unwrap().weights;
    assert_eq!(w.w2.row(0), &[1.0, 2.0, 3.0]);
    let bad = "wir 0.1\ninput x: R4\noutput y: R1\ncrease c1: ramp along [1 0 0 0]\ny = [1 2] . (c1..c2)\n";
    assert!(compile(bad, Default::default()).unwrap_err().to_string().contains("c2"));
}

#[test]
fn gates_compile_to_swiglu() {
    let src = "wir 0.1\ninput x: R2\noutput y: R1\n\
               gate g: silu along [1 2] above 1, times along [0 1] above -3\n\
               y = 2*g\n";
    let w = compile(src, Default::default()).unwrap().weights;
    let (w3, b3) = w.up.as_ref().unwrap();
    assert_eq!(w3.row(0), &[0.0, 1.0]);
    assert_eq!(b3, &vec![3.0]);
    let x = [0.5, -1.0];
    let z: f64 = 0.5 - 2.0 - 1.0;
    assert!((w.eval(&x)[0] - 2.0 * z / (1.0 + (-z).exp()) * (-1.0 + 3.0)).abs() < 1e-12);

    let mixed = "wir 0.1\ninput x: R2\noutput y: R1\ngate g: silu along [1 0], times along [0 1]\n\
                 crease c: ramp along [1 1]\ny = g + c\n";
    assert!(compile(mixed, Default::default()).unwrap_err().to_string().contains("mixes"));
}

#[test]
fn patch_directions_expand_over_3x3_kernels() {
    let src = "wir 0.1\ninput f: R2x3x3\noutput y: R1\n\
               crease a: ramp along patch level [1 0] dx [0 2]\ny = a\n";
    let w = compile(src, Default::default()).unwrap().weights;
    let row = w.w1.row(0);
    assert_eq!(&row[..9], &[1.0; 9]);
    assert_eq!(&row[9..], &[-2.0, 0.0, 2.0, -2.0, 0.0, 2.0, -2.0, 0.0, 2.0]);

    // Any kernel has exactly one patch form: decompiling with the shape and
    // recompiling the exact text gives back the same function.
    let mut rng = Rng::new(5);
    let mut net = random_network(&mut rng, 18, 4, 1);
    net.shape = Some(vec![2, 3, 3]);
    let p = decompile(&net, &DecompileOptions::default()).unwrap();
    let text = print(&p, NumFormat::EXACT);
    assert!(text.contains("patch level"), "{text}");
    let back = compile(&text, Default::default()).unwrap().weights;
    assert!(max_diff(&mut rng, &net, &back, 2000, 2.0) < 1e-9);
}

#[test]
fn kernel_terms_project_onto_low_frequencies() {
    let mut rng = Rng::new(9);
    let mut net = random_network(&mut rng, 9, 3, 1);
    net.shape = Some(vec![1, 3, 3]);
    let opts = DecompileOptions { kernel_terms: Some(1), ..Default::default() };
    let text = print(&decompile(&net, &opts).unwrap(), NumFormat::EXACT);
    assert!(text.contains("patch level") && !text.contains(" dx "), "{text}");
    for row in compile(&text, Default::default()).unwrap().weights.w1.to_rows() {
        assert!(row.iter().all(|x| (x - row[0]).abs() < 1e-12), "level-only kernels are constant: {row:?}");
    }
    let nine = DecompileOptions { kernel_terms: Some(9), ..Default::default() };
    let back = compile(&print(&decompile(&net, &nine).unwrap(), NumFormat::EXACT), Default::default()).unwrap().weights;
    assert!(max_diff(&mut rng, &net, &back, 2000, 2.0) < 1e-9);
}

#[test]
fn box_domain_drops_inactive_creases() {
    let src = "wir 0.1\ninput x: R2 in [0, 1]\noutput y: R1\n\
               crease a: zero on line (0,0)->(1,-1), equals 1 at (1,0)\n\
               crease b: zero on line (5,0)->(5,1), equals 1 at (6,0)\n\
               y = a + b\n";
    let c = compile(src, Default::default()).unwrap();
    assert!(c.warnings.iter().any(|w| w.message.contains("`b` is never active")));
    let canon = canonicalize(&c.program, Default::default());
    assert_eq!(canon.units.len(), 1);
    let text = print(&canon, NumFormat::CANONICAL);
    assert!(text.contains("input  x: R2 in [0, 1]"), "{text}");
    assert_eq!(lower_str(&text).unwrap().domain, Some((0.0, 1.0)));
}

#[test]
fn domain_samples_drop_units_and_keep_the_function_there() {
    let mut rng = Rng::new(17);
    let w = random_network(&mut rng, 5, 12, 2);
    let samples: Vec<Vec<f64>> = (0..50).map(|_| random_point(&mut rng, 5, 0.3)).collect();
    let opts = DecompileOptions { domain_samples: Some(samples.clone()), ..Default::default() };
    let p = decompile(&w, &opts).unwrap();
    let full = decompile(&w, &DecompileOptions::default()).unwrap();
    assert!(p.units.len() <= full.units.len());
    let back = compile(&print(&p, NumFormat::EXACT), Default::default()).unwrap().weights;
    for x in &samples {
        for (a, b) in w.eval(x).iter().zip(back.eval(x)) {
            assert!((a - b).abs() < 1e-9);
        }
    }
}

/// High-dimensional ReLU and SwiGLU networks through compact text: exact numbers
/// round-trip the function, canonical (3-digit vectors) stays close and is a
/// print fixed point.
#[test]
fn compact_networks_round_trip() {
    let mut rng = Rng::new(2027);
    let mut worst_exact = 0.0f64;
    let mut worst_canon = 0.0f64;
    for case in 0..300 {
        let d = 4 + rng.below(6);
        let h = 1 + rng.below(12);
        let o = 1 + rng.below(3);
        let w = if case % 2 == 0 { random_network(&mut rng, d, h, o) } else { swiglu_network(&mut rng, d, h, o) };
        let p = decompile(&w, &DecompileOptions::default()).unwrap();

        let exact = print(&p, NumFormat::EXACT);
        let back = compile(&exact, Default::default()).unwrap_or_else(|e| panic!("{e}\n{exact}")).weights;
        let diff = max_diff(&mut rng, &w, &back, 500, 2.0);
        worst_exact = worst_exact.max(diff);
        assert!(diff < 1e-9, "case {case}: exact |diff| {diff:e}\n{exact}");

        let canon = print(&p, NumFormat::CANONICAL);
        assert!(canon.contains(" . ("), "case {case}: expected a vector readout\n{canon}");
        let reread = print(&lower_str(&canon).unwrap(), NumFormat::CANONICAL);
        assert_eq!(reread, canon, "case {case}: canonical text is not a print fixed point");
        let back = compile(&canon, Default::default()).unwrap().weights;
        let rel = rel_diff(&mut rng, &w, &back, 500);
        worst_canon = worst_canon.max(rel);
        assert!(rel < 5e-3, "case {case}: canonical relative error {rel:e}\n{canon}");
    }
    eprintln!("compact round-trip: exact worst |diff| {worst_exact:e}, canonical worst relative error {worst_canon:e}");
}

/// Scaling and permuting units (and, for gates, scaling and flipping the up
/// factor) leaves the decompiled text unchanged.
#[test]
fn gate_symmetries_canonicalize() {
    let mut rng = Rng::new(31);
    for _ in 0..50 {
        let w = swiglu_network(&mut rng, 5, 6, 2);
        let mut v = w.clone();
        let (w3, b3) = v.up.as_mut().unwrap();
        for (i, bi) in b3.iter_mut().enumerate() {
            let a = if rng.chance(0.5) { -1.0 } else { 1.0 } * rng.uniform(0.1, 5.0);
            w3.row_mut(i).iter_mut().for_each(|x| *x *= a);
            *bi *= a;
            for k in 0..2 {
                v.w2.set(k, i, w.w2.get(k, i) / a);
            }
        }
        let text = |w: &Weights| print(&decompile(w, &DecompileOptions::default()).unwrap(), NumFormat::CANONICAL);
        assert_eq!(text(&w), text(&v));
    }
}

#[test]
fn output_ranges() {
    let src = "wir 0.1\ninput x: R4\noutput y1..y3: R1\ncrease a: ramp along [1 0 0 0]\n\
               y1 = a\ny2 = 2*a\ny3 = 3*a\n";
    let p = lower_str(src).unwrap();
    let names: Vec<&str> = p.outputs.iter().map(|o| o.name.as_str()).collect();
    assert_eq!(names, ["y1", "y2", "y3"]);
    assert!(print(&p, NumFormat::CANONICAL).contains("output y1..y3: R1\n"));
    let dup = "wir 0.1\ninput x: R4\noutput y1..y3: R1\noutput y2: R1\ncrease a: ramp along [1 0 0 0]\ny1 = a\n";
    assert!(compile(dup, Default::default()).unwrap_err().to_string().contains("`y2` is already declared"));
}
