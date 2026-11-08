//! Yumon Park: the brain against the rule, and the choices the park relies on.

use wir::park::{self, ParkIn, ParkNet, OPTIONS};

#[test]
fn brain_matches_the_rule() {
    let c = park::park_check(20_000, 5).unwrap();
    assert_eq!(c.key_mismatches, 0, "{c:?}");
    assert_eq!(c.worst, 0.0);
    // Every option is the written choice somewhere in the random inputs.
    assert!(c.options.iter().all(|&n| n > 0), "{:?}", c.options);
}

fn choice(net: &mut ParkNet, x: &ParkIn) -> &'static str {
    let k = net.keys(x);
    let want = park::keys(x);
    assert!(k.iter().zip(want).all(|(g, w)| (g - w as f64).abs() < 1e-9));
    OPTIONS[park::best(&want)]
}

#[test]
fn written_choices() {
    let mut net = ParkNet::compile().unwrap();
    let calm = ParkIn { option: 7, hunger: 1, fatigue: 2, ..Default::default() };
    assert_eq!(choice(&mut net, &calm), "roam");
    // A hungry hunter with prey close hunts, even with food far off.
    assert_eq!(choice(&mut net, &ParkIn { hunger: 6, pd: 3, fd: 1, ..calm }), "hunt");
    // Prey in view keeps a hungry hunter from wandering off (roam only grows with
    // hunger when nothing to eat or hunt is in sight).
    assert_eq!(choice(&mut net, &ParkIn { hunger: 8, pd: 1, ..calm }), "hunt");
    // A threat near: a camouflaged Yumon freezes, others flee; close up both flee.
    assert_eq!(choice(&mut net, &ParkIn { td: 2, camo: 1, ..calm }), "hide");
    assert_eq!(choice(&mut net, &ParkIn { td: 2, ..calm }), "flee");
    assert_eq!(choice(&mut net, &ParkIn { td: 3, camo: 1, ..calm }), "flee");
    // Tired and safe: sleep. Tired and threatened: never sleep.
    assert_eq!(choice(&mut net, &ParkIn { fatigue: 9, ..calm }), "sleep");
    assert_ne!(choice(&mut net, &ParkIn { fatigue: 10, td: 1, ..calm }), "sleep");
    // Fertile, fed, safe, mate close: mate.
    assert_eq!(choice(&mut net, &ParkIn { fertile: 1, md: 3, ..calm }), "mate");
    // Hunting comes first: near prey beats close plants; far prey loses to them.
    assert_eq!(choice(&mut net, &ParkIn { hunger: 4, pd: 2, fd: 3, ..calm }), "hunt");
    assert_eq!(choice(&mut net, &ParkIn { hunger: 4, pd: 1, fd: 3, ..calm }), "eat");
    // Struck by an attacker close by: armed with better or even odds fight back, worse odds
    // flee; a shell braces unless a refuge is near.
    let struck = ParkIn { td: 3, hit: 3, ..calm };
    assert_eq!(choice(&mut net, &ParkIn { armed: 1, odds: 2, ..struck }), "fight");
    assert_eq!(choice(&mut net, &ParkIn { armed: 1, odds: 1, ..struck }), "fight");
    assert_eq!(choice(&mut net, &ParkIn { armed: 1, odds: 0, ..struck }), "flee");
    assert_eq!(choice(&mut net, &ParkIn { armed: 1, odds: 1, hurt: 2, ..struck }), "flee");
    assert_eq!(choice(&mut net, &ParkIn { guard: 1, ..struck }), "brace");
    assert_eq!(choice(&mut net, &ParkIn { guard: 1, refuge: 1, ..struck }), "flee");
}
