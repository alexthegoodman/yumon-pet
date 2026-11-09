//! WIR Chess: the compiled attention network against the plain-code reference.

use wir::chess::{self, board_from_fen, check, mirror, random_board, reference};
use wir::seq::SeqRunner;

#[test]
fn network_matches_the_reference_on_random_boards() {
    let (_, w, bound) = chess::build().unwrap();
    // per-square outputs (attackers, lowest attacker, hanging) round exactly
    for b in &bound[..6] {
        assert!(*b < 0.5, "per-square bound {b}");
    }
    let mut rng = 11u64;
    let r = check(&w, (0..300).map(|_| random_board(&mut rng)));
    assert_eq!(r.square_wrong, 0, "{r:?}");
    assert!(r.worst_total < 1e-5, "{r:?}");
}

#[test]
fn reference_on_known_positions() {
    let start = board_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR").unwrap();
    let r = reference(&start);
    // e3 is covered by the d2 and f2 pawns; f3 by the g1 knight and the e2 and g2 pawns
    assert_eq!((r.att_w[20], r.att_w[21]), (2, 3));
    assert_eq!(r.hanging_w.iter().sum::<i64>() + r.hanging_b.iter().sum::<i64>(), 0);
    assert!(r.eval.abs() < 1e-9, "the start is symmetric: {}", r.eval);
    // a rook on e4 attacked by a black pawn on d5 and defended by nothing hangs
    let b = board_from_fen("4k3/8/8/3p4/4R3/8/8/4K3").unwrap();
    let r = reference(&b);
    assert_eq!((r.hanging_w[28], r.low_b[28]), (1, 1));
    // the d5 pawn: attacked by the rook along the rank? no (e4 rook attacks the e-file and
    // rank 4); it is not hanging
    assert_eq!(r.hanging_b[35], 0);
    let (_, w, _) = chess::build().unwrap();
    let net = SeqRunner::new(&w).eval_multi(&chess::tokens(&b));
    assert_eq!(net[28][4].round(), 1.0);
    // mirrored, the colours swap
    let m = reference(&mirror(&b));
    assert_eq!(m.hanging_b[28 ^ 56], 1);
}
