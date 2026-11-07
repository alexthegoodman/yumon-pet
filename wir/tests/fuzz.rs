//! Weight round-trip fuzzing: random networks -> decompile -> print -> compile
//! must be the same function, even though the tensors differ by symmetry.

mod common;

use common::*;
use wir::{compile, decompile, print, DecompileOptions, NumFormat, Weights};

fn through_text(w: &Weights, f: NumFormat) -> (String, Weights) {
    let p = decompile(w, &DecompileOptions::default()).unwrap();
    let text = print(&p, f);
    let back = compile(&text, Default::default()).unwrap_or_else(|e| panic!("{e}\n{text}")).weights;
    (text, back)
}

/// Acceptance: 1,000 networks, D in {1,2,3}, 1-16 hidden units, each checked
/// on 10,000 inputs within 1e-6 through exact text.
#[test]
fn thousand_random_networks_round_trip_exactly() {
    let mut rng = Rng::new(2026);
    let mut worst = 0.0f64;
    for case in 0..1000 {
        let d = 1 + case % 3;
        let h = 1 + rng.below(16);
        let w = random_network(&mut rng, d, h, 1);
        let (text, back) = through_text(&w, NumFormat::EXACT);
        let diff = max_diff(&mut rng, &w, &back, 10_000, 3.0);
        worst = worst.max(diff);
        assert!(diff <= 1e-6, "case {case} (D={d}, H={h}): max |diff| {diff:e}\n{text}");
        assert!(back.hidden() <= w.hidden(), "case {case}: canonical form added units");
    }
    eprintln!("exact round-trip: worst |diff| over 1000 networks x 10,000 points = {worst:e}");
}

/// Vector outputs share creases; the same guarantee holds.
#[test]
fn vector_output_networks_round_trip() {
    let mut rng = Rng::new(11);
    for case in 0..200 {
        let d = 1 + case % 3;
        let h = 1 + rng.below(12);
        let o = 2 + rng.below(2);
        let w = random_network(&mut rng, d, h, o);
        let (text, back) = through_text(&w, NumFormat::EXACT);
        let diff = max_diff(&mut rng, &w, &back, 2_000, 3.0);
        assert!(diff <= 1e-6, "case {case}: {diff:e}\n{text}");
    }
}

/// The 6-digit canonical text is a lossy view; check that its error stays at
/// the scale rounding allows (relative 1e-5 of the activation mass).
#[test]
fn canonical_text_is_close() {
    let mut rng = Rng::new(5);
    for case in 0..300 {
        let d = 1 + case % 3;
        let h = 1 + rng.below(16);
        let w = random_network(&mut rng, d, h, 1);
        let (text, back) = through_text(&w, NumFormat::CANONICAL);
        for _ in 0..1000 {
            let x = random_point(&mut rng, d, 3.0);
            let mass: f64 = (0..w.hidden())
                .map(|i| {
                    let pre: f64 = w.w1.row(i).iter().zip(&x).map(|(a, b)| (a * b).abs()).sum::<f64>() + w.b1[i].abs();
                    w.w2.get(0, i).abs() * pre
                })
                .sum::<f64>()
                + w.b2[0].abs()
                + 1.0;
            let diff = (w.eval(&x)[0] - back.eval(&x)[0]).abs();
            assert!(diff <= 1e-5 * mass, "case {case}: {diff:e} vs mass {mass}\n{text}");
        }
    }
}

/// Decompiling the recompiled network gives the same text again.
#[test]
fn decompile_is_a_fixed_point() {
    let mut rng = Rng::new(99);
    for case in 0..300 {
        let d = 1 + case % 3;
        let h = 1 + rng.below(16);
        let w = random_network(&mut rng, d, h, 1);
        let (text, back) = through_text(&w, NumFormat::CANONICAL);
        let (again, _) = through_text(&back, NumFormat::CANONICAL);
        assert_eq!(again, text, "case {case}");
    }
}
