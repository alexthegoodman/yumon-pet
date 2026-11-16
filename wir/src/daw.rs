//! The Entropy DAW's next-step brain: a written WIR program that ranks what a musician is
//! likely to do next in the DAW and fills in the parameters it can reason about.
//!
//! The host (entropy-engine `src/prediction`) keeps the session's substantive actions (edits
//! to the song, never view switches, windows, selection or the transport) and turns them into
//! the inputs below: which action came last, the active track's instrument family, how many
//! tracks there are, how full the pattern and the song are, how far mixing has got, and the
//! last note, clip and mixer track touched. The program gives one key per action (the host
//! ranks them and takes a softmax over `key / KEY_SCALE` for confidences) and the predicted
//! parameters for notes, clips, the mixer and pattern length.
//!
//! The keys are written the way a producer would describe a session: a new song starts with
//! the tempo; a new track gets an instrument, the instrument's preset, the song's scale, a
//! root, a pattern length and then notes; a filter tweak is followed by the envelope; a solo
//! is followed by the level; the WAV is exported before the video. Nothing is trained.
//! `daw_check` runs the compiled network against the rule as plain code, and `evaluate`
//! scores it on recorded sessions.

use crate::deep;
use crate::error::Result;
use crate::swarm::xorshift;
use crate::tactics::sparse_layers;

/// Every substantive DAW action, in the order of entropy-engine's `DawAction` (vocabulary v3).
pub const ACTIONS: [&str; 56] = [
    "add_track", "remove_track", "set_instrument", "set_bpm", "add_note", "remove_note", "paint_notes", "erase_notes",
    "new_pattern", "duplicate_pattern", "set_pattern_length", "use_pattern_in_clip", "create_clip", "move_clip",
    "resize_clip", "delete_clip", "duplicate_clip", "set_song_bars", "set_volume", "mute_track", "solo_track",
    "set_waveform", "set_scale", "set_root_note", "set_filter", "set_envelope", "set_delay", "set_reverb",
    "set_reverb_preset", "set_eq_preset", "set_character", "set_gate_pattern", "load_wavetable_preset",
    "set_wavetable_position", "load_strings", "load_brass", "set_brass_mute", "set_brass_style", "load_piano_preset",
    "set_piano_pedal", "load_matter_preset", "load_water_preset", "load_vst3", "assign_sample", "add_pad",
    "move_variation", "move_thin_out", "move_stutter", "move_octave_spark", "move_answer", "move_kick_lock",
    "move_build", "move_drop_gap", "move_echo_out", "export_wav", "export_video",
];
pub const NACT: usize = ACTIONS.len();
/// Keys are KEY_SCALE * score plus a tie-break below KEY_SCALE (earlier actions win ties).
pub const KEY_SCALE: i64 = 64;
/// Confidence is softmax(score / TEMPERATURE) over the actions.
pub const TEMPERATURE: f64 = 2.0;

/// Instrument families (entropy-engine's FAMILIES list).
pub mod fam {
    pub const DRUM_RACK: i64 = 1;
    pub const WAVETABLE: i64 = 3;
    pub const STRINGS: i64 = 4;
    pub const BRASS: i64 = 5;
    pub const PIANO: i64 = 6;
    pub const MATTER: i64 = 7;
    pub const WATER: i64 = 8;
}
/// The gate knob's index in CHARACTER_KNOBS.
pub const GATE_KNOB: i64 = 2;

/// Inputs after the 56 last-action bits, with their ranges, in program order.
pub const STATE: [(&str, i64, i64); 19] = [
    ("start", 0, 1),
    ("fam", 0, 9),
    ("tracks", 0, 16),
    ("pf", 0, 4),
    ("sf", 0, 4),
    ("mixed", 0, 4),
    ("on", 0, 1),
    ("gate", 0, 1),
    ("soloing", 0, 1),
    ("nr", 0, 47),
    ("ns", 0, 63),
    ("nd", 0, 16),
    ("dr", -4, 4),
    ("steps", 8, 64),
    ("cl", 0, 15),
    ("cb", 0, 127),
    ("cn", 1, 32),
    ("cend", 0, 1),
    ("mt", 0, 15),
];

/// Parameter outputs after the 56 keys, in program order.
pub const PARAMS: [&str; 10] =
    ["note_row", "note_step", "erase_row", "erase_step", "clip_lane", "clip_bar", "clip_bars", "mix_track", "solo_on", "pattern_bars"];

/// What the brain reads.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DawIn {
    /// The last substantive action (index into ACTIONS), None at the start of a session.
    pub last: Option<usize>,
    /// Active track's instrument family, number of tracks.
    pub fam: i64,
    pub tracks: i64,
    /// Pattern fill and song fill as quarters (0..=4).
    pub pf: i64,
    pub sf: i64,
    /// Mixer actions so far (level, solo, mute, EQ, reverb space), capped at 4.
    pub mixed: i64,
    /// The last action's on/off parameter (solo, mute).
    pub on: i64,
    /// The last action set the gate character knob.
    pub gate: i64,
    /// The most recent solo is still on.
    pub soloing: i64,
    /// The last note written (row, step), the step gap from the note before it (0 for a
    /// chord), the row change when the gap is 0, and the active pattern's length in steps.
    pub nr: i64,
    pub ns: i64,
    pub nd: i64,
    pub dr: i64,
    pub steps: i64,
    /// The last clip drawn (lane, bar, length in bars).
    pub cl: i64,
    pub cb: i64,
    pub cn: i64,
    /// The last clip reaches the end of the song.
    pub cend: i64,
    /// The last mixer track touched.
    pub mt: i64,
}

impl DawIn {
    pub fn state(&self) -> [i64; 19] {
        [
            self.last.is_none() as i64, self.fam, self.tracks, self.pf, self.sf, self.mixed, self.on, self.gate, self.soloing,
            self.nr, self.ns, self.nd, self.dr, self.steps, self.cl, self.cb, self.cn, self.cend, self.mt,
        ]
    }

    /// Network input vector in program order.
    pub fn inputs(&self) -> Vec<f64> {
        let mut v: Vec<f64> = (0..NACT).map(|k| (Some(k) == self.last) as i64 as f64).collect();
        v.extend(self.state().map(|k| k as f64));
        v
    }
}

/// The brain program. Every key reads "how likely is this next", term by term: `after_x` is
/// 1 when the last action was x.
pub fn brain_program() -> String {
    let lasts = ACTIONS.iter().map(|a| format!("after_{a}")).collect::<Vec<_>>().join(", ");
    let mut src = format!(
        "wir 0.2
# Entropy DAW next-step brain (generated by `wir daw`). The host keeps the session's
# substantive actions (song edits; never views, windows, selection or the transport).
# after_x = 1 when the last of them was x; start = 1 before the first. fam is the active
# track's instrument (1 drum rack, 3 wavetable, 4 strings, 5 brass, 6 piano, 7 matter kit,
# 8 water), pf and sf how full the pattern and the song are in quarters, mixed how many mixer
# moves so far (capped at 4), on the last action's on/off, gate whether it set the gate knob,
# soloing whether a solo is on. nr ns: the last note's row and step, nd its step gap from the
# note before (0 = a chord), dr the row change when nd = 0, steps the pattern length. cl cb cn:
# the last clip's lane, bar and length, cend whether it reaches the song's end. mt: the last mixer track. One key per action
# ({KEY_SCALE} * score + a tie-break); the host ranks them and takes softmax(score / {TEMPERATURE}).

input {lasts}: R1 in [0, 1]
"
    );
    for (name, lo, hi) in STATE {
        src += &format!("input {name}: R1 in [{lo}, {hi}]\n");
    }
    src += &format!(
        "
let new_song = and(start, le(tracks, 2), eq(sf, 0))
let old_song = and(start, not(new_song))
let drums = or(eq(fam, {DR}), eq(fam, {MA}))
let pitched_track = and(after_add_track, not(drums))
let mixing = ge(mixed, 2)
let composing_level = and(after_set_volume, not(mixing))
let solo_level = and(after_set_volume, soloing)
let plain_level = and(after_set_volume, not(soloing), mixing)
let solo_eq = and(after_set_eq_preset, soloing)
let plain_eq = and(after_set_eq_preset, not(soloing))
let solo_space = and(after_set_reverb_preset, soloing)
let plain_space = and(after_set_reverb_preset, not(soloing))
let solo_off = and(after_solo_track, not(on))
let clip_run = and(after_create_clip, not(cend), gt(cn, 1))
let clip_done = and(after_create_clip, or(cend, eq(cn, 1)))

# Tracks: a fresh track gets an instrument; a sound tweak or a locked bass often ends with the next track.
let s_add_track = 3*new_song + 2*old_song + 14*after_remove_track + select(composing_level, 7, 0) + select(plain_level, 4, 0) + 7*after_set_character + 8*after_set_gate_pattern + 7*after_set_delay + 8*after_move_kick_lock + 3*after_set_envelope
let s_remove_track = 0
let s_set_instrument = select(pitched_track, 10, 0)
let s_set_bpm = 12*new_song

# Notes: once a pattern has a length (or a kit), notes follow, and notes follow notes.
let s_add_note = 2 + 4*new_song + 10*after_add_note + 10*after_remove_note + 12*after_set_pattern_length + 12*after_new_pattern + 12*after_add_pad + 10*after_load_matter_preset + 8*after_assign_sample + 6*after_paint_notes + 10*after_move_variation + 10*after_move_thin_out + 10*after_move_stutter + 7*after_move_octave_spark + 7*after_set_bpm + select(and(after_set_song_bars, eq(sf, 0)), 6, 0) + select(and(after_add_track, drums), 6, 0)
# a slip: the note just written gets clicked off again
let s_remove_note = 4*old_song + 5*after_add_note + 7*after_remove_note + 12*after_erase_notes
let s_paint_notes = 0
let s_erase_notes = 0

# Patterns
let s_new_pattern = 2*after_add_note + 4*after_paint_notes
let s_duplicate_pattern = 3*old_song + 6*after_move_clip + 6*after_resize_clip + 6*after_move_build + 6*after_move_drop_gap + 6*after_move_echo_out
let s_set_pattern_length = 12*after_set_root_note + 5*after_set_scale + 8*after_load_matter_preset + 12*after_load_water_preset + 6*after_assign_sample + 5*after_set_bpm + 8*after_add_pad + select(and(after_add_track, drums), 8, 0)
let s_use_pattern_in_clip = 6*after_move_variation + 6*after_move_thin_out + 6*after_move_stutter + 4*after_move_octave_spark

# Arrangement: clips come in runs; a replaced song-long clip is deleted first.
let s_create_clip = 1 + 12*after_delete_clip + 12*after_duplicate_clip + select(clip_run, 10, 0) + select(clip_done, 6, 0)
let s_move_clip = select(clip_done, 3, 0)
let s_resize_clip = select(clip_done, 3, 0)
let s_delete_clip = select(clip_done, 12, 0) + select(and(after_set_song_bars, gt(sf, 0)), 12, 0)
let s_duplicate_clip = select(and(clip_run, eq(cn, 8)), 8, 0)
let s_set_song_bars = 2*old_song + 10*after_set_bpm + 4*after_set_character + 4*after_set_gate_pattern + 6*after_set_delay + 5*after_move_answer

# Mixer: solo a track, set its level, shape it, unsolo, next track.
let s_set_volume = 1 + 3*old_song + select(and(after_solo_track, on), 12, 0) + 9*after_mute_track + select(plain_eq, 7, 0) + select(plain_space, 9, 0) + select(solo_off, 8, 0) + select(plain_level, 6, 0) + 4*after_add_note + 9*after_move_kick_lock + 9*after_move_answer + 7*after_move_octave_spark + 6*after_use_pattern_in_clip + 6*after_move_build + 6*after_move_drop_gap + 6*after_move_echo_out + 5*after_paint_notes
let s_mute_track = select(solo_off, 4, 0) + select(plain_space, 4, 0) + select(plain_eq, 3, 0)
let s_solo_track = select(solo_level, 10, 0) + select(plain_level, 4, 0) + select(composing_level, 2, 0) + select(solo_off, 6, 0) + select(solo_eq, 10, 0) + select(plain_eq, 5, 0) + select(solo_space, 12, 0) + select(plain_space, 7, 0) + 8*after_mute_track + 5*after_use_pattern_in_clip + 5*after_move_build + 5*after_move_echo_out
let s_set_eq_preset = select(solo_level, 8, 0) + select(plain_level, 8, 0) + select(composing_level, 3, 0)
let s_set_reverb_preset = select(solo_eq, 8, 0) + select(plain_eq, 9, 0) + select(solo_level, 3, 0)

# Sound: a new instrument gets the song's key; a filter tweak goes with the envelope.
let s_set_waveform = select(pitched_track, 3, 0)
let s_set_scale = 10*after_load_wavetable_preset + 10*after_load_piano_preset + 10*after_set_waveform + 10*after_set_brass_mute + 9*after_load_strings + 7*after_load_brass + 7*after_set_brass_style + 6*after_set_instrument + select(pitched_track, 6, 0) + 5*after_set_bpm + 3*after_paint_notes + select(composing_level, 5, 0)
let s_set_root_note = 12*after_set_scale + 7*after_load_wavetable_preset + 7*after_load_piano_preset + 7*after_set_waveform + 5*after_load_strings + 4*after_set_brass_mute + select(pitched_track, 4, 0)
let s_set_filter = 3*old_song + select(composing_level, 4, 0) + 7*after_set_filter + 8*after_set_envelope + 12*after_set_wavetable_position + 6*after_set_delay + 5*after_move_kick_lock
let s_set_envelope = 8*after_set_filter
let s_set_delay = select(composing_level, 5, 0) + 7*after_move_answer + 4*after_move_octave_spark
let s_set_reverb = 0
let s_set_character = 5*after_set_filter + 6*after_set_envelope + 10*after_set_reverb + 6*after_set_piano_pedal
let s_set_gate_pattern = select(and(after_set_character, gate), 12, 0)

# Instruments: the preset of the family just picked.
let s_load_wavetable_preset = select(and(after_set_instrument, eq(fam, {WT})), 12, 0)
let s_set_wavetable_position = 0
let s_load_strings = select(and(after_set_instrument, eq(fam, {ST})), 12, 0) + 6*after_load_strings
let s_load_brass = select(and(after_set_instrument, eq(fam, {BR})), 12, 0)
let s_set_brass_mute = 5*after_load_brass + 5*after_set_brass_style
let s_set_brass_style = 9*after_load_brass + 5*after_set_brass_style
let s_load_piano_preset = select(and(after_set_instrument, eq(fam, {PI})), 10, 0)
let s_set_piano_pedal = 8*after_set_piano_pedal
let s_load_matter_preset = select(and(after_set_instrument, eq(fam, {MA})), 12, 0)
let s_load_water_preset = select(and(after_set_instrument, eq(fam, {WA})), 12, 0) + 3*after_load_water_preset
let s_load_vst3 = 0
let s_assign_sample = 4*after_set_bpm + select(and(after_add_track, eq(fam, {DR})), 6, 0)
let s_add_pad = 7*after_assign_sample

# Moves: a duplicated pattern gets reworked; a reworked section is followed by a transition move.
let s_move_variation = 8*after_duplicate_pattern
let s_move_thin_out = 9*after_duplicate_pattern
let s_move_stutter = select(and(after_duplicate_pattern, drums), 8, 0)
let s_move_octave_spark = select(and(after_duplicate_pattern, not(drums)), 8, 0)
let s_move_answer = 0
let s_move_kick_lock = 0
let s_move_build = 5*after_move_clip + 5*after_resize_clip
let s_move_drop_gap = 5*after_move_clip + 5*after_resize_clip
let s_move_echo_out = 5*after_move_clip + 5*after_resize_clip

# Finishing: export once the mix or the arrangement settles; the video follows the WAV.
let s_export_wav = 6*after_use_pattern_in_clip + 6*after_mute_track + select(solo_off, 3, 0) + select(plain_space, 3, 0) + 6*after_move_echo_out + 6*after_move_drop_gap + 5*after_move_build + 4*after_resize_clip
let s_export_video = 14*after_export_wav

",
        DR = fam::DRUM_RACK,
        MA = fam::MATTER,
        WT = fam::WAVETABLE,
        ST = fam::STRINGS,
        BR = fam::BRASS,
        PI = fam::PIANO,
        WA = fam::WATER,
    );
    for (i, a) in ACTIONS.iter().enumerate() {
        src += &format!("output key_{a} = {KEY_SCALE}*s_{a} + {}\n", NACT - 1 - i);
    }
    src += "
# Parameters. The next note continues the last one's spacing (wrapping at the pattern's end);
# in a chord it stacks the next row. The note to click off is the one just written.
let over = ge(ns + nd - steps, 0)
output note_row = clip(nr + select(eq(nd, 0), dr, 0), 0, 47)
output note_step = clip(ns + nd - select(over, steps, 0), 0, 63)
output erase_row = nr
output erase_step = ns
# The next clip starts where the last one ended, on the same lane; a short section (an intro or
# a break) is followed by a full 8 bars. A deleted clip means the next track: its lane, from
# the end of the intro.
output clip_lane = min(cl + after_delete_clip, 15)
output clip_bar = select(after_delete_clip, 4, min(cb + cn, 127))
output clip_bars = select(or(after_delete_clip, lt(cn, 8)), 8, cn)
# Mixing moves to the next track once a track is done: its level set with no solo on, its
# EQ or space chosen, a mute, or its solo released.
let done_track = or(and(not(soloing), or(after_set_volume, after_set_eq_preset, after_set_reverb_preset, after_mute_track)), solo_off)
output mix_track = max(min(mt + done_track, tracks - 1), 0)
output solo_on = 1 - soloing
# Pattern length: 1 bar for a kit, 2 bars for anything pitched (PATTERN_LENGTHS indices).
output pattern_bars = select(drums, 1, 2)
";
    src
}

/// The scores as plain code, in action order.
pub fn scores(x: &DawIn) -> [i64; NACT] {
    let idx = |name: &str| ACTIONS.iter().position(|a| *a == name).unwrap();
    let after = |name: &str| (x.last == Some(idx(name))) as i64;
    let b = |c: bool| c as i64;
    let start = b(x.last.is_none());
    let new_song = b(start == 1 && x.tracks <= 2 && x.sf == 0);
    let old_song = b(start == 1 && new_song == 0);
    let drums = x.fam == fam::DRUM_RACK || x.fam == fam::MATTER;
    let pitched_track = b(after("add_track") == 1 && !drums);
    let mixing = x.mixed >= 2;
    let composing_level = b(after("set_volume") == 1 && !mixing);
    let solo = x.soloing == 1;
    let solo_level = b(after("set_volume") == 1 && solo);
    let plain_level = b(after("set_volume") == 1 && !solo && mixing);
    let solo_eq = b(after("set_eq_preset") == 1 && solo);
    let plain_eq = b(after("set_eq_preset") == 1 && !solo);
    let solo_space = b(after("set_reverb_preset") == 1 && solo);
    let plain_space = b(after("set_reverb_preset") == 1 && !solo);
    let solo_off = b(after("solo_track") == 1 && x.on == 0);
    let clip_run = b(after("create_clip") == 1 && x.cend == 0 && x.cn > 1);
    let clip_done = b(after("create_clip") == 1 && (x.cend == 1 || x.cn == 1));
    let a = after;
    let mut s = [0i64; NACT];
    let mut set = |name: &str, v: i64| s[idx(name)] = v;

    set("add_track", 3 * new_song + 2 * old_song + 14 * a("remove_track") + 7 * composing_level + 4 * plain_level + 7 * a("set_character")
        + 8 * a("set_gate_pattern") + 7 * a("set_delay") + 8 * a("move_kick_lock") + 3 * a("set_envelope"));
    set("set_instrument", 10 * pitched_track);
    set("set_bpm", 12 * new_song);
    set("add_note", 2 + 4 * new_song + 10 * a("add_note") + 10 * a("remove_note") + 12 * a("set_pattern_length") + 12 * a("new_pattern")
        + 12 * a("add_pad") + 10 * a("load_matter_preset") + 8 * a("assign_sample") + 6 * a("paint_notes") + 10 * a("move_variation")
        + 10 * a("move_thin_out") + 10 * a("move_stutter") + 7 * a("move_octave_spark") + 7 * a("set_bpm")
        + 6 * b(a("set_song_bars") == 1 && x.sf == 0) + 6 * b(a("add_track") == 1 && drums));
    set("remove_note", 4 * old_song + 5 * a("add_note") + 7 * a("remove_note") + 12 * a("erase_notes"));
    set("new_pattern", 2 * a("add_note") + 4 * a("paint_notes"));
    set("duplicate_pattern", 3 * old_song + 6 * a("move_clip") + 6 * a("resize_clip") + 6 * a("move_build") + 6 * a("move_drop_gap") + 6 * a("move_echo_out"));
    set("set_pattern_length", 12 * a("set_root_note") + 5 * a("set_scale") + 8 * a("load_matter_preset") + 12 * a("load_water_preset")
        + 6 * a("assign_sample") + 5 * a("set_bpm") + 8 * a("add_pad") + 8 * b(a("add_track") == 1 && drums));
    set("use_pattern_in_clip", 6 * a("move_variation") + 6 * a("move_thin_out") + 6 * a("move_stutter") + 4 * a("move_octave_spark"));
    set("create_clip", 1 + 12 * a("delete_clip") + 12 * a("duplicate_clip") + 10 * clip_run + 6 * clip_done);
    set("move_clip", 3 * clip_done);
    set("resize_clip", 3 * clip_done);
    set("delete_clip", 12 * clip_done + 12 * b(a("set_song_bars") == 1 && x.sf > 0));
    set("duplicate_clip", 8 * b(clip_run == 1 && x.cn == 8));
    set("set_song_bars", 2 * old_song + 10 * a("set_bpm") + 4 * a("set_character") + 4 * a("set_gate_pattern") + 6 * a("set_delay") + 5 * a("move_answer"));
    set("set_volume", 1 + 3 * old_song + 12 * b(a("solo_track") == 1 && x.on == 1) + 9 * a("mute_track") + 7 * plain_eq
        + 9 * plain_space + 8 * solo_off + 6 * plain_level + 4 * a("add_note") + 9 * a("move_kick_lock")
        + 9 * a("move_answer") + 7 * a("move_octave_spark") + 6 * a("use_pattern_in_clip") + 6 * a("move_build")
        + 6 * a("move_drop_gap") + 6 * a("move_echo_out") + 5 * a("paint_notes"));
    set("mute_track", 4 * solo_off + 4 * plain_space + 3 * plain_eq);
    set("solo_track", 10 * solo_level + 4 * plain_level + 2 * composing_level + 6 * solo_off + 10 * solo_eq + 5 * plain_eq
        + 12 * solo_space + 7 * plain_space + 8 * a("mute_track") + 5 * a("use_pattern_in_clip") + 5 * a("move_build") + 5 * a("move_echo_out"));
    set("set_eq_preset", 8 * solo_level + 8 * plain_level + 3 * composing_level);
    set("set_reverb_preset", 8 * solo_eq + 9 * plain_eq + 3 * solo_level);
    set("set_waveform", 3 * pitched_track);
    set("set_scale", 10 * a("load_wavetable_preset") + 10 * a("load_piano_preset") + 10 * a("set_waveform") + 10 * a("set_brass_mute")
        + 9 * a("load_strings") + 7 * a("load_brass") + 7 * a("set_brass_style") + 6 * a("set_instrument") + 6 * pitched_track
        + 5 * a("set_bpm") + 3 * a("paint_notes") + 5 * composing_level);
    set("set_root_note", 12 * a("set_scale") + 7 * a("load_wavetable_preset") + 7 * a("load_piano_preset") + 7 * a("set_waveform")
        + 5 * a("load_strings") + 4 * a("set_brass_mute") + 4 * pitched_track);
    set("set_filter", 3 * old_song + 4 * composing_level + 7 * a("set_filter") + 8 * a("set_envelope") + 12 * a("set_wavetable_position") + 6 * a("set_delay") + 5 * a("move_kick_lock"));
    set("set_envelope", 8 * a("set_filter"));
    set("set_delay", 5 * composing_level + 7 * a("move_answer") + 4 * a("move_octave_spark"));
    set("set_character", 5 * a("set_filter") + 6 * a("set_envelope") + 10 * a("set_reverb") + 6 * a("set_piano_pedal"));
    set("set_gate_pattern", 12 * b(a("set_character") == 1 && x.gate == 1));
    let after_instrument = |f: i64| a("set_instrument") == 1 && x.fam == f;
    set("load_wavetable_preset", 12 * b(after_instrument(fam::WAVETABLE)));
    set("load_strings", 12 * b(after_instrument(fam::STRINGS)) + 6 * a("load_strings"));
    set("load_brass", 12 * b(after_instrument(fam::BRASS)));
    set("set_brass_mute", 5 * a("load_brass") + 5 * a("set_brass_style"));
    set("set_brass_style", 9 * a("load_brass") + 5 * a("set_brass_style"));
    set("load_piano_preset", 10 * b(after_instrument(fam::PIANO)));
    set("set_piano_pedal", 8 * a("set_piano_pedal"));
    set("load_matter_preset", 12 * b(after_instrument(fam::MATTER)));
    set("load_water_preset", 12 * b(after_instrument(fam::WATER)) + 3 * a("load_water_preset"));
    set("assign_sample", 4 * a("set_bpm") + 6 * b(a("add_track") == 1 && x.fam == fam::DRUM_RACK));
    set("add_pad", 7 * a("assign_sample"));
    set("move_variation", 8 * a("duplicate_pattern"));
    set("move_thin_out", 9 * a("duplicate_pattern"));
    set("move_stutter", 8 * b(a("duplicate_pattern") == 1 && drums));
    set("move_octave_spark", 8 * b(a("duplicate_pattern") == 1 && !drums));
    for m in ["move_build", "move_drop_gap", "move_echo_out"] {
        set(m, 5 * a("move_clip") + 5 * a("resize_clip"));
    }
    set("export_wav", 6 * a("use_pattern_in_clip") + 6 * a("mute_track") + 3 * solo_off + 3 * plain_space + 6 * a("move_echo_out") + 6 * a("move_drop_gap")
        + 5 * a("move_build") + 4 * a("resize_clip"));
    set("export_video", 14 * a("export_wav"));
    s
}

/// The keys the network outputs, as plain code.
pub fn keys(x: &DawIn) -> [i64; NACT] {
    let s = scores(x);
    std::array::from_fn(|i| KEY_SCALE * s[i] + (NACT - 1 - i) as i64)
}

/// The parameter outputs, as plain code, in PARAMS order.
pub fn params(x: &DawIn) -> [i64; PARAMS.len()] {
    let after = |name: &str| x.last == ACTIONS.iter().position(|a| *a == name);
    let drums = x.fam == fam::DRUM_RACK || x.fam == fam::MATTER;
    let over = x.ns + x.nd - x.steps >= 0;
    let solo_off = after("solo_track") && x.on == 0;
    let done_track = (x.soloing == 0 && (after("set_volume") || after("set_eq_preset") || after("set_reverb_preset") || after("mute_track"))) || solo_off;
    [
        (x.nr + if x.nd == 0 { x.dr } else { 0 }).clamp(0, 47),
        (x.ns + x.nd - if over { x.steps } else { 0 }).clamp(0, 63),
        x.nr,
        x.ns,
        (x.cl + after("delete_clip") as i64).min(15),
        if after("delete_clip") { 4 } else { (x.cb + x.cn).min(127) },
        if after("delete_clip") || x.cn < 8 { 8 } else { x.cn },
        (x.mt + done_track as i64).min(x.tracks - 1).max(0),
        1 - x.soloing,
        if drums { 1 } else { 2 },
    ]
}

/// Every output (keys then parameters), as plain code.
pub fn outputs(x: &DawIn) -> Vec<i64> {
    let mut v = keys(x).to_vec();
    v.extend(params(x));
    v
}

/// The written choice: the largest key (keys never tie).
pub fn best(k: &[i64]) -> usize {
    (0..NACT).max_by_key(|&o| k[o]).unwrap()
}

/// The compiled brain.
pub struct DawNet {
    pub brain: deep::DeepWeights,
    net: deep::SparseNet,
}

impl DawNet {
    pub fn compile() -> Result<Self> {
        let brain = deep::compile(&brain_program(), 1e-6)?.weights;
        Ok(Self { net: deep::SparseNet::new(&brain), brain })
    }

    pub fn outputs(&mut self, x: &DawIn) -> Vec<f64> {
        self.net.eval(&x.inputs()).to_vec()
    }
}

/// Random input over the declared domain.
pub fn random_daw_in(rng: &mut u64) -> DawIn {
    let r = |rng: &mut u64, lo: i64, hi: i64| lo + xorshift(rng, (hi - lo + 1) as u64) as i64;
    let last = xorshift(rng, NACT as u64 + 1) as usize;
    DawIn {
        last: (last < NACT).then_some(last),
        fam: r(rng, 0, 9),
        tracks: r(rng, 0, 16),
        pf: r(rng, 0, 4),
        sf: r(rng, 0, 4),
        mixed: r(rng, 0, 4),
        on: r(rng, 0, 1),
        gate: r(rng, 0, 1),
        soloing: r(rng, 0, 1),
        nr: r(rng, 0, 47),
        ns: r(rng, 0, 63),
        nd: r(rng, 0, 16),
        dr: r(rng, -4, 4),
        steps: r(rng, 8, 64),
        cl: r(rng, 0, 15),
        cb: r(rng, 0, 127),
        cn: r(rng, 1, 32),
        cend: r(rng, 0, 1),
        mt: r(rng, 0, 15),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DawCheck {
    pub cases: u64,
    /// Inputs where some output differs from the rule's by more than 1e-6.
    pub mismatches: u64,
    pub worst: f64,
    pub shape: (Vec<usize>, usize, usize),
}

/// The brain against the rule on `cases` random inputs.
pub fn daw_check(cases: u64, seed: u64) -> Result<DawCheck> {
    let mut net = DawNet::compile()?;
    let mut rng = seed.max(1);
    let (mut mismatches, mut worst) = (0, 0f64);
    for _ in 0..cases {
        let x = random_daw_in(&mut rng);
        let want = outputs(&x);
        let got = net.outputs(&x);
        let diff = want.iter().zip(&got).map(|(&w, &g)| (w as f64 - g).abs()).fold(0.0, f64::max);
        worst = worst.max(diff);
        mismatches += (diff > 1e-6) as u64;
    }
    let nonzero = net.brain.layers.iter().map(|l| l.w.data.iter().filter(|v| **v != 0.0).count() + l.b.len()).sum();
    Ok(DawCheck { cases, mismatches, worst, shape: (net.brain.hidden(), net.brain.param_count(), nonzero) })
}

// ---------------------------------------------------------------- sessions

/// One recorded substantive step: the action, its parameters in natural units, and the
/// context after it (family, tracks, pattern fill, song fill, pattern steps).
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    pub action: usize,
    pub params: Vec<f64>,
    pub fam: i64,
    pub tracks: i64,
    pub pattern_fill: f64,
    pub song_fill: f64,
    pub steps: i64,
}

const MIXER: [&str; 5] = ["set_volume", "solo_track", "mute_track", "set_eq_preset", "set_reverb_preset"];

/// The brain's inputs after `history` (oldest first). The same reading as entropy-engine's
/// `prediction::features`.
pub fn features(history: &[Step]) -> DawIn {
    let id = |name: &str| ACTIONS.iter().position(|a| *a == name).unwrap();
    let p = |s: &Step, i: usize| s.params.get(i).copied().unwrap_or(0.0);
    let quarter = |f: f64| ((f.clamp(0.0, 1.0) * 4.0).floor() as i64).min(4);
    let mut x = DawIn { tracks: 2, fam: 1, nd: 4, steps: 16, cn: 4, ..Default::default() };
    let mut notes: Vec<(i64, i64)> = Vec::new();
    let mut song_bars = 16;
    for s in history {
        let name = ACTIONS[s.action];
        if MIXER.contains(&name) {
            x.mixed = (x.mixed + 1).min(4);
            if name != "set_eq_preset" && name != "set_reverb_preset" {
                x.mt = (p(s, 0) as i64).clamp(0, 15);
            }
        }
        if name == "solo_track" {
            x.soloing = (p(s, 1) >= 0.5) as i64;
        }
        if name == "add_note" {
            notes.push(((p(s, 0) as i64).clamp(0, 47), (p(s, 1) as i64).clamp(0, 63)));
        }
        if name == "create_clip" {
            x.cl = (p(s, 0) as i64).clamp(0, 15);
            x.cb = (p(s, 1) as i64).clamp(0, 127);
            x.cn = (p(s, 2) as i64).clamp(1, 32);
        }
        if name == "duplicate_clip" {
            x.cb = (x.cb + x.cn).min(127);
        }
        if name == "set_song_bars" {
            song_bars = (p(s, 0) as i64).max(1);
        }
    }
    x.cend = (x.cb + x.cn >= song_bars) as i64;
    if let Some(&(r, s)) = notes.last() {
        x.nr = r;
        x.ns = s;
        if notes.len() >= 2 {
            let (pr, ps) = notes[notes.len() - 2];
            x.nd = (s - ps).clamp(0, 16);
            x.dr = if s == ps { (r - pr).clamp(-4, 4) } else { 0 };
        }
    }
    if let Some(s) = history.last() {
        let name = ACTIONS[s.action];
        x.last = Some(s.action);
        x.fam = s.fam.clamp(0, 9);
        x.tracks = s.tracks.clamp(0, 16);
        x.pf = quarter(s.pattern_fill);
        x.sf = quarter(s.song_fill);
        x.steps = s.steps.clamp(8, 64);
        x.on = if name == "solo_track" || name == "mute_track" { (p(s, 1) >= 0.5) as i64 } else { 0 };
        x.gate = (s.action == id("set_character") && p(s, 0) as i64 == GATE_KNOB) as i64;
    }
    x
}

#[derive(Debug, Clone, Default)]
pub struct DawEval {
    pub queries: u64,
    pub top1: u64,
    pub top3: u64,
    /// Sum of ln p(actual next action) under softmax(score / temperature).
    pub log_lik: f64,
    /// Per actual action: (count, top-1 hits).
    pub per_action: Vec<(u64, u64)>,
    /// Top-1 misses by (last action or None, predicted, actual).
    pub misses: std::collections::BTreeMap<(Option<usize>, usize, usize), u64>,
}

/// Scores the network on sessions: at the start and after every step, the next step's action
/// against the ranked keys. `temperature` is the softmax temperature for the likelihood.
pub fn evaluate(sessions: &[Vec<Step>], temperature: f64) -> Result<DawEval> {
    let mut net = DawNet::compile()?;
    let mut e = DawEval { per_action: vec![(0, 0); NACT], ..Default::default() };
    for s in sessions {
        for t in 0..s.len() {
            let x = features(&s[..t]);
            let out = net.outputs(&x);
            let target = s[t].action;
            let mut order: Vec<usize> = (0..NACT).collect();
            order.sort_by(|&a, &b| out[b].partial_cmp(&out[a]).unwrap());
            e.queries += 1;
            e.top1 += (order[0] == target) as u64;
            e.top3 += order[..3].contains(&target) as u64;
            e.per_action[target].0 += 1;
            e.per_action[target].1 += (order[0] == target) as u64;
            if order[0] != target {
                *e.misses.entry((x.last, order[0], target)).or_default() += 1;
            }
            let sc: Vec<f64> = out[..NACT].iter().map(|k| (k / KEY_SCALE as f64).floor() / temperature).collect();
            let m = sc.iter().cloned().fold(f64::MIN, f64::max);
            let z: f64 = sc.iter().map(|v| (v - m).exp()).sum();
            e.log_lik += sc[target] - m - z.ln();
        }
    }
    Ok(e)
}

/// Reads sessions as JSON: `[[[action_name, [params..], [fam, tracks, pattern_fill, song_fill,
/// steps]], ..], ..]`. Unknown action names are skipped (they are not substantive).
pub fn read_sessions(text: &str) -> Result<Vec<Vec<Step>>> {
    let v: serde_json::Value = serde_json::from_str(text).map_err(|e| crate::error::WirError::new(format!("sessions: {e}")))?;
    let bad = || crate::error::WirError::new("sessions: expected [[[name, [params], [fam, tracks, pf, sf, steps]], ..], ..]");
    let mut out = Vec::new();
    for s in v.as_array().ok_or_else(bad)? {
        let mut steps = Vec::new();
        for st in s.as_array().ok_or_else(bad)? {
            let st = st.as_array().ok_or_else(bad)?;
            let name = st.first().and_then(|n| n.as_str()).ok_or_else(bad)?;
            let Some(action) = ACTIONS.iter().position(|a| *a == name) else { continue };
            let params = st.get(1).and_then(|p| p.as_array()).ok_or_else(bad)?.iter().map(|p| p.as_f64().unwrap_or(0.0)).collect();
            let c: Vec<f64> = st.get(2).and_then(|c| c.as_array()).ok_or_else(bad)?.iter().map(|c| c.as_f64().unwrap_or(0.0)).collect();
            let c = |i: usize| c.get(i).copied().unwrap_or(0.0);
            steps.push(Step { action, params, fam: c(0) as i64, tracks: c(1) as i64, pattern_fill: c(2), song_fill: c(3), steps: c(4) as i64 });
        }
        out.push(steps);
    }
    Ok(out)
}

/// The brain (sparse layers, source), the constants, and test vectors (inputs with the rule's
/// outputs) for the host's port.
pub fn export_json(vectors: u64, seed: u64) -> Result<String> {
    let net = DawNet::compile()?;
    let mut rng = seed.max(1);
    let vs: Vec<serde_json::Value> = (0..vectors)
        .map(|_| {
            let x = random_daw_in(&mut rng);
            let input: Vec<i64> = x.inputs().iter().map(|&v| v as i64).collect();
            serde_json::json!([input, outputs(&x)])
        })
        .collect();
    let v = serde_json::json!({
        "format": "wir-daw-brain",
        "consts": {
            "actions": ACTIONS.to_vec(), "params": PARAMS.to_vec(), "key_scale": KEY_SCALE, "temperature": TEMPERATURE,
            "state": STATE.iter().map(|(n, lo, hi)| serde_json::json!([n, lo, hi])).collect::<Vec<_>>(),
        },
        "brain": {
            "inputs": net.brain.inputs, "outputs": net.brain.outputs,
            "hidden": net.brain.hidden(), "params": net.brain.param_count(),
            "layers": sparse_layers(&net.brain),
            "source": brain_program(),
        },
        "vectors": vs,
    });
    Ok(serde_json::to_string(&v).unwrap())
}
