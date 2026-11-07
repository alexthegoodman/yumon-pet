use wir::{compile, CompileOptions, Weights};

fn example(name: &str) -> String {
    std::fs::read_to_string(format!("{}/examples/{name}.wir", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

fn ok(src: &str) -> wir::Compiled {
    compile(src, CompileOptions::default()).unwrap_or_else(|e| panic!("{e}"))
}

fn err(src: &str) -> wir::WirError {
    match compile(src, CompileOptions::default()) {
        Ok(_) => panic!("expected a compile error for:\n{src}"),
        Err(e) => e,
    }
}

fn rows(w: &wir::Mat) -> Vec<Vec<f64>> {
    w.to_rows()
}

const XOR_HEAD: &str = "wir 0.1\ninput  x: R2\noutput y: R1\n";

#[test]
fn tent_compiles_to_expected_weights() {
    let c = ok(&example("tent"));
    let w: &Weights = &c.weights;
    assert_eq!(rows(&w.w1), vec![vec![1.0], vec![1.0], vec![1.0]]);
    assert_eq!(w.b1, vec![0.0, -0.5, -1.0]);
    assert_eq!(rows(&w.w2), vec![vec![1.0, -2.0, 1.0]]);
    assert_eq!(w.b2, vec![0.0]);
    assert_eq!(w.param_count(), 10);
    assert_eq!(c.program.asserts.len(), 2);
}

#[test]
fn xor_compiles_to_expected_weights() {
    let c = ok(&example("xor"));
    let w = &c.weights;
    assert_eq!(rows(&w.w1), vec![vec![1.0, 1.0], vec![1.0, 1.0]]);
    assert_eq!(w.b1, vec![0.0, -1.0]);
    assert_eq!(rows(&w.w2), vec![vec![1.0, -2.0]]);
    assert_eq!(w.b2, vec![0.0]);
    assert_eq!(w.param_count(), 9);
    assert_eq!(c.program.asserts.len(), 4);
}

#[test]
fn xor_solve_fits_truth_table() {
    let c = ok(&example("xor_solve"));
    assert_eq!(c.solves.len(), 1);
    assert!(c.solves[0].max_residual < 1e-9, "residual {}", c.solves[0].max_residual);
    assert_eq!(c.solves[0].rank, 3);
    for (x, y) in [([0.0, 0.0], 0.0), ([1.0, 0.0], 1.0), ([0.0, 1.0], 1.0), ([1.0, 1.0], 0.0)] {
        assert!((c.weights.eval(&x)[0] - y).abs() < 1e-9);
    }
    // Same geometry as xor.wir, so the readout matches it too.
    let w2 = rows(&c.weights.w2);
    assert!((w2[0][0] - 1.0).abs() < 1e-9 && (w2[0][1] + 2.0).abs() < 1e-9);
    assert!(c.warnings.is_empty());
}

#[test]
fn cost_report_sums_to_parameter_count() {
    for name in ["tent", "xor", "xor_solve"] {
        let c = ok(&example(name));
        let total: usize = c.costs.iter().map(|l| l.params).sum();
        let units: usize = c.costs.iter().map(|l| l.units).sum();
        assert_eq!(total, c.weights.param_count(), "{name}");
        assert_eq!(units, c.weights.hidden(), "{name}");
    }
    let c = ok(&example("xor"));
    let per: Vec<(usize, usize, usize)> = c.costs.iter().map(|l| (l.line, l.units, l.params)).collect();
    assert_eq!(per, vec![(5, 1, 4), (6, 1, 4), (7, 0, 1)]);
}

// ---------------------------------------------------------------- required error cases

fn assert_names(e: &wir::WirError, line: usize, stmt: &str, msg: &str) {
    let text = e.to_string();
    assert_eq!(e.line, Some(line), "{text}");
    assert!(text.contains(stmt), "message should quote the statement `{stmt}`: {text}");
    assert!(text.contains(msg), "message should mention `{msg}`: {text}");
}

#[test]
fn error_anchor_on_plane() {
    let e = err(&format!("{XOR_HEAD}crease a: zero on line (0,0)->(1,-1), equals 1 at (2,-2)\ny = a\n"));
    assert_names(&e, 4, "crease a: zero on line (0,0)->(1,-1), equals 1 at (2,-2)", "lies on the crease's plane");
}

#[test]
fn error_non_positive_anchor() {
    for v in ["0", "-1"] {
        let e = err(&format!("{XOR_HEAD}\ncrease a: between (0,0) and (1,0), equals {v} at (1,0)\ny = a\n"));
        assert_names(&e, 5, "crease a: between (0,0) and (1,0)", "anchor value must be positive");
    }
}

#[test]
fn error_unsolvable_solve() {
    // One crease cannot produce XOR.
    let e = err(&format!(
        "{XOR_HEAD}crease a: zero on line (0,0)->(1,-1), equals 1 at (1,0)\ny = solve over {{a}}: (0,0)->0, (1,0)->1, (0,1)->1, (1,1)->0\n"
    ));
    assert_names(&e, 5, "y = solve over {a}", "cannot express the target");
}

#[test]
fn error_failed_assert() {
    let e = err(&format!(
        "{XOR_HEAD}crease a: zero on line (0,0)->(1,-1), equals 1 at (1,0)\ny = a\n# comment\nassert y(0,0)==0, y(1,1)==0\n"
    ));
    assert_names(&e, 7, "assert y(0,0)==0, y(1,1)==0", "assert failed: y(1, 1) = 2 but expected 0");
}

#[test]
fn other_errors_name_their_statement() {
    let cases: &[(&str, usize, &str)] = &[
        ("wir 0.2\ninput x: R1\noutput y: R1\ny = 1\n", 1, "unsupported WIR version"),
        ("wir 0.1\ninput x: R2\noutput y: R1\ny = hinge at 0 turn 1\n", 4, "1D sugar"),
        ("wir 0.1\ninput x: R2\noutput y: R1\ny = a\n", 4, "not a declared crease"),
        ("wir 0.1\ninput x: R3\noutput y: R1\ncrease a: zero on line (0,0)->(1,0), equals 1 at (0,1)\ny = a\n", 4, "2D only"),
        ("wir 0.1\ninput x: R2\noutput y: R1\ncrease a: between (0,0) and (1,0,0), equals 1 at (1,0)\ny = a\n", 4, "3 coordinates"),
        ("wir 0.1\ninput x: R1\noutput y: R1\ny = 1\ny = 2\n", 5, "already has a readout"),
        ("wir 0.1\ninput x: R1\noutput y: R1\nassert z(0) == 1\ny = 0\n", 4, "not a declared output"),
        ("wir 0.1\ninput x: R1\noutput y: R2\ny = 0\n", 3, "outputs are scalar"),
        ("wir 0.1\ninput x: R2\noutput y: R1\ncrease a: zero on line (1,1)->(1,1), equals 1 at (0,0)\ny = a\n", 4, "coincide"),
        ("wir 0.1\ninput x: R1\noutput y: R1\ncrease y: zero on plane normal (1) offset 0, equals 1 at 1\ny = 0\n", 4, "already declared"),
        ("wir 0.1\ninput x: R1\noutput y: R1\ny = 2 * turn\n", 4, "keyword"),
    ];
    for (src, line, msg) in cases {
        let e = err(src);
        assert_eq!(e.line, Some(*line), "{e}");
        assert!(e.to_string().contains(msg), "expected `{msg}` in: {e}");
    }
    assert!(err("wir 0.1\ninput x: R1\noutput y: R1\n").to_string().contains("has no readout"));
}

#[test]
fn unused_crease_warns_and_is_dropped() {
    let c = ok(&format!(
        "{XOR_HEAD}crease a: zero on line (0,0)->(1,-1), equals 1 at (1,0)\ncrease b: between (0,0) and (0,1), equals 1 at (0,1)\ny = 2*a\n"
    ));
    assert_eq!(c.weights.hidden(), 1);
    assert_eq!(c.warnings.len(), 1);
    assert_eq!(c.warnings[0].line, Some(5));
    assert!(c.warnings[0].message.contains("never used"));
    let line5 = c.costs.iter().find(|l| l.line == 5).unwrap();
    assert_eq!((line5.units, line5.params), (0, 0));
}

#[test]
fn rank_deficient_solve_warns() {
    // Two copies of the same crease: the solve is consistent but not unique.
    let c = ok(&format!(
        "{XOR_HEAD}crease a: between (0,0) and (2,0), equals 1 at (2,0)\ncrease b: between (0,0) and (2,0), equals 2 at (2,0)\ny = solve over {{a, b}}: (0,0)->0, (2,0)->3, (3,0)->6\n"
    ));
    assert!(c.warnings.iter().any(|w| w.message.contains("rank-deficient")), "{:?}", c.warnings);
    assert!((c.weights.eval(&[2.5, 7.0])[0] - 4.5).abs() < 1e-9);
}

#[test]
fn plane_forms_place_planes_and_slopes() {
    // between: bisector of (0,0,0)-(2,0,0) is x1 = 1, rising toward +x1, value 3 at x1 = 2.
    let src = "wir 0.1\ninput p: R3\noutput q: R1\n\
               crease a: between (0,0,0) and (2,0,0), equals 3 at (2,5,5)\n\
               crease n: zero on plane normal (0,0,-2) offset 1, equals 1 at (0,0,-2)\n\
               q = a + 10*n - 1\n";
    let c = ok(src);
    let y = |x: [f64; 3]| c.weights.eval(&x)[0];
    assert!((y([1.0, 0.0, 0.0]) - -1.0).abs() < 1e-12);
    assert!((y([3.0, 9.0, 0.0]) - 5.0).abs() < 1e-12); // a = 3 * 2
    assert!((y([0.0, 0.0, -1.0]) - -1.0).abs() < 1e-12); // on the plane z = -1
    assert!((y([0.0, 0.0, -3.0]) - 19.0).abs() < 1e-12); // distance 2, slope 1, times 10
    assert!((y([0.0, 0.0, 5.0]) - -1.0).abs() < 1e-12);
}

#[test]
fn anchor_flips_orientation() {
    // Same line, anchors on opposite sides: the units rise in opposite directions.
    let a = ok(&format!("{XOR_HEAD}crease a: zero on line (0,0)->(0,1), equals 2 at (1,0)\ny = a\n"));
    let b = ok(&format!("{XOR_HEAD}crease a: zero on line (0,0)->(0,1), equals 2 at (-1,0)\ny = a\n"));
    assert_eq!(rows(&a.weights.w1), vec![vec![2.0, 0.0]]);
    assert_eq!(rows(&b.weights.w1), vec![vec![-2.0, 0.0]]);
}

#[test]
fn hinge_left_and_constants() {
    let c = ok("wir 0.1\ninput x: R1\noutput y: R1\ny = 0.5 + hinge left at 1 turn 2 - 3*hinge at 2 turn 1\n");
    assert_eq!(rows(&c.weights.w1), vec![vec![-1.0], vec![1.0]]);
    assert_eq!(c.weights.b1, vec![1.0, -2.0]);
    assert_eq!(rows(&c.weights.w2), vec![vec![2.0, -3.0]]);
    assert_eq!(c.weights.b2, vec![0.5]);
    assert_eq!(c.weights.eval(&[0.0])[0], 2.5);
}

#[test]
fn multiple_outputs_share_creases() {
    let c = ok("wir 0.1\ninput x: R2\noutput u: R1\noutput v: R1\n\
               crease a: zero on line (0,0)->(0,1), equals 1 at (1,0)\n\
               u = a\nv = -a + 1\nassert u(2,0) == 2, v(2,0) == -1\n");
    assert_eq!(c.weights.hidden(), 1);
    assert_eq!(rows(&c.weights.w2), vec![vec![1.0], vec![-1.0]]);
    assert_eq!(c.weights.b2, vec![0.0, 1.0]);
    let per: usize = c.costs.iter().map(|l| l.params).sum();
    assert_eq!(per, c.weights.param_count());
}

#[test]
fn custom_tolerance() {
    let src = "wir 0.1\ninput x: R1\noutput y: R1\ny = hinge at 0 turn 1\nassert y(1) == 1.001\n";
    assert!(compile(src, CompileOptions::default()).is_err());
    assert!(compile(src, CompileOptions { tolerance: 1e-2 }).is_ok());
}
