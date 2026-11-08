//! Yumon Ant: the networks against the rule, the blocked-heading bound, and the task
//! choice the colony relies on.

use wir::ant::{self, AntIn, AntNets, HOLD, SENSES, S_B, S_KG, S_KN, S_KF, S_LD};

/// Path, senses and brain match the rule on random inputs.
#[test]
fn networks_match_the_rule() {
    let c = ant::ant_check(5_000, 3).unwrap();
    assert_eq!((c.path_mismatches, c.sense_mismatches, c.brain_mismatches, c.blocked_picks), (0, 0, 0, 0), "{c:?}");
    // Every option and every emote shows up in the random inputs.
    assert!(c.options.iter().all(|&n| n > 0), "{:?}", c.options);
    assert!(c.emotes.iter().all(|&n| n > 0), "{:?}", c.emotes);
}

fn calm(senses: Vec<i64>) -> AntIn {
    AntIn { option: HOLD, cf: 0, cs: 0, gr: 0, hunger: 0, old: 0, store: 3, wander: 0, role: 0, senses }
}

/// Every heading blocked, each one on the task's gradient: holding still wins.
#[test]
fn a_blocked_heading_never_wins() {
    let mut nets = AntNets::compile().unwrap();
    let mut s = vec![1; SENSES];
    s[ant::S_FS] = 0;
    s[ant::S_DN] = 5;
    for d in 0..4 {
        s[S_B + d] = 1;
    }
    for option in 0..5 {
        let x = AntIn { option, ..calm(s.clone()) };
        assert_eq!(ant::brain_rule(&x).option, HOLD);
        assert_eq!(nets.decide(&x).unwrap().option, HOLD);
    }
}

/// A free young digger that smells the dig face follows the dig field (gg), not the
/// food field, and a forager with the same senses follows food.
#[test]
fn roles_pick_the_task() {
    let mut nets = AntNets::compile().unwrap();
    let mut s = vec![0; SENSES];
    // food is west (gf2 = 1), the dig face is down (gg3 = 1); everything else farther
    for f in 0..5 {
        for d in 0..4 {
            s[4 * f + d] = -1;
        }
    }
    s[4 + 2] = 1;
    s[8 + 3] = 1;
    s[S_KN] = 1;
    s[S_KF] = 1;
    s[S_KG] = 1;
    s[ant::S_DN] = 10;
    let digger = AntIn { store: 2, ..calm(s.clone()) };
    assert_eq!(ant::TASKS[ant::task(&digger)], "dig");
    assert_eq!(nets.decide(&digger).unwrap().option, 3);
    let forager = AntIn { role: 1, ..digger.clone() };
    assert_eq!(ant::TASKS[ant::task(&forager)], "forage");
    assert_eq!(nets.decide(&forager).unwrap().option, 2);
    // Next to the dig face the digger holds and digs, and feels like working.
    let mut at_face = digger.clone();
    at_face.senses[S_LD] = 1;
    let d = nets.decide(&at_face).unwrap();
    assert_eq!((d.option, d.actions[1], ant::EMOTES[d.emote]), (HOLD, true, "working"));
}
