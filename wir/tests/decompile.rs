mod common;

use common::*;
use wir::{compile, decompile, print, DecompileOptions, Mat, NumFormat, Weights};

fn weights(w1: Vec<Vec<f64>>, b1: Vec<f64>, w2: Vec<Vec<f64>>, b2: Vec<f64>) -> Weights {
    let d = w1.first().map_or(1, |r| r.len());
    let w1 = if w1.is_empty() { Mat::zeros(0, d) } else { Mat::from_rows(&w1).unwrap() };
    Weights { w1, b1, w2: Mat::from_rows(&w2).unwrap(), b2, up: None, input: None, outputs: None, shape: None }
}

fn text(w: &Weights) -> String {
    print(&decompile(w, &DecompileOptions::default()).unwrap(), NumFormat::CANONICAL)
}

#[test]
fn tent_weights_decompile_to_hinges() {
    let w = weights(vec![vec![1.0], vec![1.0], vec![1.0]], vec![0.0, -0.5, -1.0], vec![vec![1.0, -2.0, 1.0]], vec![0.0]);
    assert_eq!(
        text(&w),
        "wir 0.1\ninput  x: R1\noutput y: R1\n\ny = hinge at 0 turn 1 + hinge at 0.5 turn -2 + hinge at 1 turn 1\n"
    );
}

#[test]
fn hinge_left_and_scaling_in_1d() {
    // 3*relu(-2x + 1) = 6*relu(0.5 - x): a left hinge at 0.5 with turn 6.
    let w = weights(vec![vec![-2.0]], vec![1.0], vec![vec![3.0]], vec![0.25]);
    assert!(text(&w).contains("y = hinge left at 0.5 turn 6 + 0.25\n"), "{}", text(&w));
}

#[test]
fn xor_weights_decompile_to_lines() {
    let w = weights(vec![vec![1.0, 1.0], vec![1.0, 1.0]], vec![0.0, -1.0], vec![vec![1.0, -2.0]], vec![0.0]);
    assert_eq!(
        text(&w),
        "wir 0.1\ninput  x: R2\noutput y: R1\n\n\
         crease c1: zero on line (0, 0) -> (-1, 1), equals 1 at (1, 0)\n\
         crease c2: zero on line (1, 0) -> (0, 1), equals 1 at (2, 0)\n\n\
         y = c1 - 2*c2\n"
    );
}

#[test]
fn three_d_uses_plane_normal_form() {
    let w = weights(vec![vec![0.0, 3.0, 4.0]], vec![-10.0], vec![vec![1.0]], vec![0.0]);
    // max|w| = 4: canonical w = (0, 0.75, 1), b = -2.5, coefficient 4.
    let t = text(&w);
    assert!(t.contains("crease c1: zero on plane normal (0, 0.75, 1) through (0, 0, 2.5), equals 1 at (0, 0, 3.5)\n"), "{t}");
    assert!(t.contains("y = 4*c1\n"), "{t}");
}

#[test]
fn constant_units_fold_into_bias() {
    // Unit 1 has w = 0, b = 2: always relu(2) = 2, times 3 -> +6. Unit 2 has b < 0: always 0.
    let w = weights(vec![vec![0.0, 0.0], vec![0.0, 0.0], vec![1.0, 0.0]], vec![2.0, -1.0, 0.0], vec![vec![3.0, 5.0, 1.0]], vec![1.0]);
    let p = decompile(&w, &DecompileOptions::default()).unwrap();
    assert_eq!(p.units.len(), 1);
    assert_eq!(p.outputs[0].bias, 7.0);
}

#[test]
fn dead_and_duplicate_units() {
    // Unit 2 is unit 1 scaled by 4 (merges), unit 3 has c = 0 (dropped),
    // unit 4 is unit 1 flipped (opposite orientation, stays distinct).
    let w = weights(
        vec![vec![1.0, 2.0], vec![4.0, 8.0], vec![3.0, -1.0], vec![-1.0, -2.0]],
        vec![0.5, 2.0, 0.0, -0.5],
        vec![vec![1.0, 0.25, 0.0, 1.0]],
        vec![0.0],
    );
    let p = decompile(&w, &DecompileOptions::default()).unwrap();
    assert_eq!(p.units.len(), 2);
    let mut coeffs: Vec<f64> = p.units.iter().map(|u| u.coeffs[0]).collect();
    coeffs.sort_by(f64::total_cmp);
    // Scale is max|w| = 2. Merged: (1 + 0.25*4) * 2; opposite: 1 * 2.
    assert_eq!(coeffs, vec![2.0, 4.0]);
}

#[test]
fn cancelling_duplicates_vanish() {
    let w = weights(vec![vec![1.0], vec![2.0]], vec![-1.0, -2.0], vec![vec![2.0, -1.0]], vec![0.0]);
    assert_eq!(text(&w), "wir 0.1\ninput  x: R1\noutput y: R1\n\ny = 0\n");
}

#[test]
fn names_and_asserts_on_request() {
    let w = weights(vec![vec![1.0]], vec![0.0], vec![vec![1.0], vec![-1.0]], vec![0.0, 2.0]);
    let opts = DecompileOptions {
        input: Some("t".into()),
        outputs: Some(vec!["up".into(), "down".into()]),
        assert_at: vec![vec![3.0]],
        ..Default::default()
    };
    let src = print(&decompile(&w, &opts).unwrap(), NumFormat::CANONICAL);
    assert!(src.contains("input  t: R1\noutput up: R1\noutput down: R1\n"), "{src}");
    // Two outputs: the unit is shared, so it prints as a named crease, not a hinge.
    assert!(src.contains("crease c1: zero on plane normal (1) through (0), equals 1 at (1)"), "{src}");
    assert!(src.contains("assert up(3) == 3\nassert down(3) == -1\n"), "{src}");
    // And it compiles back with the asserts passing.
    let c = compile(&src, Default::default()).unwrap();
    assert_eq!(c.weights.eval(&[3.0]), vec![3.0, -1.0]);
}

#[test]
fn shape_errors() {
    let mut w = weights(vec![vec![1.0]], vec![0.0], vec![vec![1.0]], vec![0.0]);
    w.b1 = vec![0.0, 1.0];
    assert!(decompile(&w, &DecompileOptions::default()).unwrap_err().to_string().contains("b1 has length 2"));
    let mut w = weights(vec![vec![1.0]], vec![0.0], vec![vec![1.0]], vec![0.0]);
    w.w1.data[0] = f64::NAN;
    assert!(decompile(&w, &DecompileOptions::default()).is_err());
}

#[test]
fn scaling_and_permuting_units_leaves_text_unchanged() {
    let mut rng = Rng::new(7);
    for case in 0..300 {
        let d = 1 + case % 3;
        let o = if case % 10 == 9 { 2 } else { 1 };
        let h = 1 + rng.below(16);
        let w = random_network(&mut rng, d, h, o);
        let base = text(&w);
        for _ in 0..3 {
            let v = scale_and_permute(&mut rng, &w);
            assert_eq!(text(&v), base, "case {case}: D={d} H={h}");
        }
    }
}
