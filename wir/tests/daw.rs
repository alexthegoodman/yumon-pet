//! Entropy DAW next-step brain: the network against the rule, the written choices and the
//! parameters it predicts.

use wir::daw::{self, DawIn, DawNet, Step, ACTIONS};

#[test]
fn brain_matches_the_rule() {
    let c = daw::daw_check(20_000, 5).unwrap();
    assert_eq!(c.mismatches, 0, "{c:?}");
    assert_eq!(c.worst, 0.0);
}

fn id(name: &str) -> usize {
    ACTIONS.iter().position(|a| *a == name).unwrap()
}

/// The network's top action and parameters, checked against the rule.
fn choice(net: &mut DawNet, x: &DawIn) -> (&'static str, Vec<f64>) {
    let out = net.outputs(x);
    let want = daw::outputs(x);
    assert!(out.iter().zip(&want).all(|(g, &w)| (g - w as f64).abs() < 1e-9));
    (ACTIONS[daw::best(&want)], out[ACTIONS.len()..].to_vec())
}

fn after(name: &str, x: DawIn) -> DawIn {
    DawIn { last: Some(id(name)), ..x }
}

#[test]
fn written_choices() {
    let mut net = DawNet::compile().unwrap();
    let base = DawIn { fam: 2, tracks: 2, nd: 4, steps: 16, cn: 4, ..Default::default() };
    // A blank song starts with the tempo; an existing one does not.
    assert_eq!(choice(&mut net, &base).0, "set_bpm");
    assert_ne!(choice(&mut net, &DawIn { tracks: 4, sf: 2, ..base }).0, "set_bpm");
    // Track setup: instrument, then the family's preset, the scale, the root, the length, notes.
    assert_eq!(choice(&mut net, &after("add_track", base)).0, "set_instrument");
    assert_eq!(choice(&mut net, &after("set_instrument", DawIn { fam: 4, ..base })).0, "load_strings");
    assert_eq!(choice(&mut net, &after("set_instrument", DawIn { fam: 7, ..base })).0, "load_matter_preset");
    assert_eq!(choice(&mut net, &after("load_wavetable_preset", base)).0, "set_scale");
    assert_eq!(choice(&mut net, &after("set_scale", base)).0, "set_root_note");
    assert_eq!(choice(&mut net, &after("set_root_note", base)).0, "set_pattern_length");
    assert_eq!(choice(&mut net, &after("set_pattern_length", base)).0, "add_note");
    // Arrangement: a clip run continues until a clip reaches the song's end.
    assert_eq!(choice(&mut net, &after("create_clip", base)).0, "create_clip");
    assert_eq!(choice(&mut net, &after("create_clip", DawIn { cend: 1, ..base })).0, "delete_clip");
    // Mixer: a solo is followed by the level; with the solo on, the level by unsoloing.
    let mixing = DawIn { mixed: 3, ..base };
    assert_eq!(choice(&mut net, &after("solo_track", DawIn { on: 1, ..mixing })).0, "set_volume");
    assert_eq!(choice(&mut net, &after("set_volume", DawIn { soloing: 1, ..mixing })).0, "solo_track");
    assert_eq!(choice(&mut net, &after("set_volume", mixing)).0, "set_eq_preset");
    // The gate knob brings up the gate pattern; the WAV brings up the video.
    assert_eq!(choice(&mut net, &after("set_character", DawIn { gate: 1, ..base })).0, "set_gate_pattern");
    assert_eq!(choice(&mut net, &after("export_wav", base)).0, "export_video");
}

#[test]
fn written_parameters() {
    let mut net = DawNet::compile().unwrap();
    let p = |name: &str| daw::PARAMS.iter().position(|q| *q == name).unwrap();
    // The next note keeps the spacing and wraps at the pattern's end; a chord stacks rows.
    let x = DawIn { nr: 3, ns: 8, nd: 4, steps: 16, tracks: 2, cn: 4, ..Default::default() };
    let (_, out) = choice(&mut net, &after("add_note", x));
    assert_eq!((out[p("note_row")], out[p("note_step")]), (3.0, 12.0));
    let (_, out) = choice(&mut net, &after("add_note", DawIn { ns: 12, ..x }));
    assert_eq!(out[p("note_step")], 0.0);
    let (_, out) = choice(&mut net, &after("add_note", DawIn { nd: 0, dr: 2, ..x }));
    assert_eq!((out[p("note_row")], out[p("note_step")]), (5.0, 8.0));
    // The next clip starts where the last ended; after a short section comes a full one; after a
    // deleted clip, the next lane.
    let (_, out) = choice(&mut net, &after("create_clip", DawIn { cl: 2, cb: 4, cn: 8, ..x }));
    assert_eq!((out[p("clip_lane")], out[p("clip_bar")], out[p("clip_bars")]), (2.0, 12.0, 8.0));
    let (_, out) = choice(&mut net, &after("create_clip", DawIn { cl: 2, cb: 0, cn: 4, ..x }));
    assert_eq!((out[p("clip_bar")], out[p("clip_bars")]), (4.0, 8.0));
    let (_, out) = choice(&mut net, &after("delete_clip", DawIn { cl: 2, cb: 24, cn: 8, ..x }));
    assert_eq!((out[p("clip_lane")], out[p("clip_bar")], out[p("clip_bars")]), (3.0, 4.0, 8.0));
    // Mixing moves on once a track's level is set with no solo, and stays while soloing.
    let m = DawIn { mt: 1, tracks: 4, mixed: 3, ..x };
    assert_eq!(choice(&mut net, &after("set_volume", m)).1[p("mix_track")], 2.0);
    assert_eq!(choice(&mut net, &after("set_volume", DawIn { soloing: 1, ..m })).1[p("mix_track")], 1.0);
    assert_eq!(choice(&mut net, &after("set_volume", DawIn { mt: 3, ..m })).1[p("mix_track")], 3.0);
}

#[test]
fn features_read_the_history() {
    let step = |name: &str, params: &[f64]| Step {
        action: id(name), params: params.to_vec(), fam: 1, tracks: 3, pattern_fill: 0.3, song_fill: 0.0, steps: 16,
    };
    let h = vec![
        step("add_note", &[0.0, 0.0, 1.0, 0.85]),
        step("add_note", &[0.0, 4.0, 1.0, 0.85]),
        step("create_clip", &[1.0, 0.0, 8.0]),
        step("duplicate_clip", &[]),
        step("solo_track", &[2.0, 1.0]),
    ];
    let x = daw::features(&h);
    assert_eq!(x.last, Some(id("solo_track")));
    assert_eq!((x.nr, x.ns, x.nd), (0, 4, 4));
    assert_eq!((x.cl, x.cb, x.cn, x.cend), (1, 8, 8, 1));
    assert_eq!((x.mt, x.soloing, x.on, x.mixed, x.pf), (2, 1, 1, 1, 1));
    assert_eq!(daw::features(&[]).last, None);
}
