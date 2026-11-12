//! WIR Beam: one beam search step as a written attention network.

use wir::beam::{self, all_steps, check, random_step, Config};

#[test]
fn program_text_compiles_with_small_bounds() {
    let cfg = Config::default();
    let (p, w, bound) = beam::build(&cfg).unwrap();
    assert_eq!(p.ctx.len, 32);
    assert!(bound.iter().all(|b| *b < 1e-3), "{bound:?}");
    assert!(w.cost().nonzero < 10_000);
}

#[test]
fn every_step_of_a_small_configuration_matches() {
    // 2 beams x 3 tokens, costs 0..2: (3 * 27 * 2)^2 = 26,244 steps, all of them.
    let cfg = Config { k: 2, v: 3, cap: 2, qmax: 2 };
    let (_, w, _) = beam::build(&cfg).unwrap();
    let r = check(&cfg, &w, all_steps(&cfg));
    assert_eq!(r.steps, 26_244);
    assert_eq!(r.wrong, 0, "{r:?}");
    assert!(r.worst < 1e-3, "{r:?}");
}

#[test]
fn random_steps_with_many_ties_match_at_full_size() {
    let cfg = Config::default();
    let (_, w, _) = beam::build(&cfg).unwrap();
    let mut rng = 11u64;
    let tight: Vec<_> = (0..1500).map(|_| random_step(&cfg, &mut rng, true)).collect();
    let r = check(&cfg, &w, tight.into_iter());
    assert_eq!(r.wrong, 0, "{r:?}");
    let wide: Vec<_> = (0..1500).map(|_| random_step(&cfg, &mut rng, false)).collect();
    let r = check(&cfg, &w, wide.into_iter());
    assert_eq!(r.wrong, 0, "{r:?}");
}

#[test]
fn finished_beams_only_offer_the_pad_token() {
    let cfg = Config { k: 2, v: 3, cap: 4, qmax: 4 };
    let st = beam::Step { s: vec![1, 3], l: vec![vec![0, 0, 0], vec![0, 4, 4]], fin: vec![1, 0] };
    let want = beam::reference(&cfg, &st);
    // Finished beam 0 keeps cost 1 through token 0; beam 1 offers 3, 7, 7.
    assert_eq!((want[0].cost, want[0].from, want[0].tok), (1, 0, 0));
    assert_eq!((want[1].cost, want[1].from, want[1].tok), (3, 1, 0));
    let (_, w, _) = beam::build(&cfg).unwrap();
    assert_eq!(check(&cfg, &w, std::iter::once(st)).wrong, 0);
}
