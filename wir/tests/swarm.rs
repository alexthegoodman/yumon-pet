//! Swarm senses and brains: the networks against the rule, the blocked-heading bound,
//! the sense windows, and flocking that comes from the alignment term.

use wir::swarm::{self, Deciders, Species, SwarmNets};

/// Senses and brains match the rule on random inputs, and whole worlds run agent for
/// agent the same with the networks as with the rule.
#[test]
fn networks_match_the_rule() {
    let species = [Species::fish(), Species::bird()];
    let c = swarm::swarm_check(&species, 4_000, 2, 120, 20, 30, 30, 9).unwrap();
    assert_eq!((c.sense_mismatches, c.brain_mismatches, c.step_mismatches, c.blocked_picks), (0, 0, 0, 0), "{c:?}");
    assert!(c.decisions > 0);
}

/// The blocked penalty is one more than the span of every other term, so with all
/// other inputs at their worst for an open heading and best for a blocked one, the
/// open heading still wins.
#[test]
fn an_open_heading_always_beats_a_blocked_one() {
    for sp in [Species::fish(), Species::bird()] {
        let (lo, hi) = sp.score_range();
        assert_eq!(sp.blocked_penalty(), hi - lo + 1);
        let mut net = SwarmNets::compile(&sp).unwrap();
        // Heading 7 is open and has everything against it (a full turn away, crowded, the
        // wrong way on the map); heading 3 is blocked and has everything for it (straight
        // on, the noise kick). Ties go to the lower index, so a penalty one smaller would
        // pick heading 3 here.
        let (open, blocked) = (7, 3);
        let mut senses = vec![0; sp.sense_count()];
        senses[blocked] = swarm::ALIGN_CAP;
        senses[8 + open] = swarm::CROWD_CAP;
        senses[8 + blocked] = sp.crowd;
        senses[16 + if sp.other > 0 { blocked } else { open }] = swarm::OTHER_SENSE_CAP;
        for d in (0..8).filter(|&d| d != open) {
            senses[24 + d] = 1;
        }
        if sp.has_goal() {
            senses[32 + open] = -1;
            senses[32 + blocked] = 1;
        }
        assert_eq!(swarm::brain_choose(&sp, blocked, Some(blocked), &senses), open, "{}", sp.name);
        let out = net.brain_outputs(blocked, Some(blocked), &senses);
        assert!(out.iter().enumerate().all(|(k, v)| (v - (k == open) as i64 as f64).abs() < 1e-9), "{out:?}");
    }
}

/// The 24 cells around the centre split into 8 sectors of 3, and turning the window by
/// 90 degrees turns sector d into d + 2.
#[test]
fn sense_sectors_are_an_even_pinwheel() {
    let mut count = [0; 8];
    for dy in -swarm::RADIUS..=swarm::RADIUS {
        for dx in -swarm::RADIUS..=swarm::RADIUS {
            let Some(d) = swarm::sector(dy, dx) else {
                assert_eq!((dy, dx), (0, 0));
                continue;
            };
            count[d] += 1;
            // (dy, dx) turned 90 degrees counter-clockwise on screen is (-dx, dy).
            assert_eq!(swarm::sector(-dx, dy), Some((d + 2) % 8), "({dy}, {dx})");
        }
    }
    assert_eq!(count, [3; 8]);
    for (d, (dy, dx)) in swarm::DIRS.iter().enumerate() {
        assert_eq!(swarm::sector(*dy, *dx), Some(d));
    }
}

/// With the alignment term, neighbours end up heading the same way; without it they do not.
#[test]
fn alignment_makes_flocks() {
    let order = |align: bool| {
        let mut species = [Species::fish(), Species::bird()];
        if !align {
            for sp in species.iter_mut() {
                sp.align = 0;
            }
        }
        let mut total = 0.0;
        for seed in 1..=3 {
            let mut w = swarm::swarm_world(&species, 24, 36, 40, seed);
            for _ in 0..300 {
                w.step(&species, &mut Deciders::Rule(&species));
            }
            total += w.local_stats(0).1 + w.local_stats(1).1;
        }
        total / 6.0
    };
    let (with, without) = (order(true), order(false));
    assert!(with > 0.8 && without < 0.7, "local order with alignment {with:.2}, without {without:.2}");
}

/// The generated programs compile, and the sense program is a 5 x 5 grid program.
#[test]
fn programs_have_the_expected_shape() {
    for sp in [Species::fish(), Species::bird()] {
        let n = SwarmNets::compile(&sp).unwrap();
        assert_eq!(n.sense.radius(), 2);
        assert_eq!(n.sense.net.outputs.len(), sp.sense_count());
        assert_eq!(n.brain.input_dim(), 16 + sp.sense_count());
        assert_eq!(n.brain.output_dim(), 8);
        assert_eq!(n.sense.channels, sp.channels());
    }
}
