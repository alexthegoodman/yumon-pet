//! Tactical enemies: the networks against the rule, the blocked-heading bound, the
//! small-integer product, and the behaviour the response curves are for.

use wir::tactics::{self, Archetype, BrainIn, TacticsNets, HOLD};

/// Senses and brains match the rule on random inputs, and whole arenas (fields, every
/// enemy, the player) run tick for tick the same with the networks as with the rule.
#[test]
fn networks_match_the_rule() {
    let c = tactics::tactics_check(&Archetype::all(), 5_000, 2, 150, 20, 32, 2, tactics::ALLY_COST, 3).unwrap();
    assert_eq!((c.sense_mismatches, c.brain_mismatches, c.step_mismatches, c.behaviour.ticks.blocked_picks), (0, 0, 0, 0), "{c:?}");
    assert!(c.behaviour.ticks.decisions > 500);
}

/// With every other term for a blocked heading and against every open one, holding
/// still (never blocked) still wins: the penalty is one more than the score span.
#[test]
fn a_blocked_heading_never_wins() {
    let archetypes = Archetype::all();
    let mut nets = TacticsNets::compile(&archetypes).unwrap();
    for (k, a) in archetypes.iter().enumerate() {
        // Empty always reads as afraid enough to avoid being seen.
        assert!(a.fe >= a.peek + a.hide, "{}", a.name);
        let (lo, hi) = a.score_range();
        assert_eq!(a.blocked_penalty(), hi - lo + 1);
        // Heading 2 is blocked and has everything for it: straight on, the best step for
        // both fields, exposed while calm. Every heading is blocked, so holding must win.
        let mut s = vec![0; tactics::SENSES];
        for d in 0..8 {
            s[d] = if d == 2 { 1 } else { -1 };
            s[8 + d] = if d == 2 { 1 } else { -1 };
            s[16 + d] = (d == 2) as i64;
            s[24 + d] = if d == 2 { 0 } else { tactics::ALLY_CAP };
            s[32 + d] = 1;
        }
        s[40] = tactics::CEILING;
        s[41] = 0;
        s[42] = -1;
        let x = BrainIn { option: 2, hp: 1, ammo: 0, cool: 0, reload: 0, senses: s };
        assert_eq!(tactics::brain_rule(a, &x).0, HOLD, "{}", a.name);
        assert_eq!(nets.decide(k, &x).unwrap().0, HOLD, "{}", a.name);
    }
}

/// `a * b` with one factor in {-1, 0, 1} (or {0, 1}, or three other integers) compiles
/// exactly; two wide factors are refused.
#[test]
fn small_integer_products() {
    let src = "wir 0.2\ninput x: R1 in [-7, 5]\ninput t: R1 in [-1, 1]\ninput u: R1 in [2, 4]\ninput c: R1 in [0, 1]\noutput y = x*t\noutput z = u*x\noutput w = x*c\n";
    let w = wir::deep::compile(src, 1e-9).unwrap().weights;
    for x in -7..=5 {
        for t in -1..=1 {
            for u in 2..=4 {
                for c in 0..=1 {
                    let out = w.eval(&[x as f64, t as f64, u as f64, c as f64]);
                    assert_eq!(out, vec![(x * t) as f64, (u * x) as f64, (x * c) as f64], "x {x} t {t} u {u} c {c}");
                }
            }
        }
    }
    let err = wir::deep::compile("wir 0.2\ninput x: R1 in [0, 5]\ninput y: R1 in [0, 3]\noutput z = x*y\n", 1e-9).unwrap_err();
    assert!(err.to_string().contains("three integer values"), "{err}");
}

/// Fear sends hurt or empty enemies that the player can see toward cover; without the
/// fear curve they mostly do not. Inertia cuts step-and-reverse moves.
#[test]
fn curves_change_behaviour() {
    let run = |f: &dyn Fn(&mut Archetype)| {
        let mut a = Archetype::all();
        a.iter_mut().for_each(f);
        tactics::tactics_behaviour(&a, 3, 300, 24, 40, 3, tactics::ALLY_COST, 5)
    };
    let base = run(&|_| {});
    let calm = run(&|a| (a.f1, a.f2, a.fe) = (0, 0, 0));
    let jumpy = run(&|a| (a.keep, a.keep_hold) = ([0; 5], 0));
    let rate = |b: &tactics::Behaviour, k: usize| b.ticks.to_cover[k] as f64 / b.ticks.afraid[k].max(1) as f64;
    for k in [0, 2] {
        assert!(rate(&base, k) > 0.6 && rate(&calm, k) < 0.3, "archetype {k}: {:.2} vs {:.2}", rate(&base, k), rate(&calm, k));
    }
    let bf = |b: &tactics::Behaviour| b.ticks.back_forth as f64 / b.ticks.moves.max(1) as f64;
    assert!(bf(&base) * 3.0 < bf(&jumpy), "{:.3} vs {:.3}", bf(&base), bf(&jumpy));
}
