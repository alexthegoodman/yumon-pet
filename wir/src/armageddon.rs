//! Armageddon 2070: the WIR networks behind every bot on a 128 vs 128 battlefield.
//!
//! The tactical enemies (`tactics.rs`) were one archetype fighting one player in a small arena.
//! Armageddon needs a chain of command and machines, so this module writes a family of 0.2
//! programs, one network each, all checked exact against their own program semantics:
//!
//! - **senses** (`ar_sense`): one grid program read at an agent's cell, shared by infantry, ground
//!   vehicles, mecha and helicopters (the host fills the channels for each scale): grades of four
//!   distance fields per heading (objective, threat, cover, squad anchor), exposure, allies per
//!   pinwheel sector, blocked headings.
//! - **infantry** (`ar_<role>_brain`, rifle / support / marksman / antiarmor): the tactical grunt
//!   upgraded with the squad's mode (advance, assault, defend, regroup, flank, fall back, mount),
//!   suppression, squad cohesion with a leash and the player's squad call, grenades and launchers.
//! - **squad** (`ar_squad_brain`): picks the squad's mode from its strength, contact, armor and the
//!   objective's state.
//! - **strategic** (`ar_strategic_brain`): picks one of up to 12 territories for a squad.
//! - **vehicles**: `ar_tank_*` (hull heading, target class, main gun, coax, smoke: the turret is
//!   aimed by the host at the chosen class, so hull and turret are separate decisions), `ar_apc_*`
//!   (deliver, pick up, dismount, support fire), `ar_heli_*` (heading, altitude band, landing,
//!   flares, random jitter bits so approaches are not predictable), `ar_jet_*` (attack-run phases,
//!   drops), `ar_mecha_*` (component damage changes movement, bracing and weapon choice).
//!
//! Everything is integer, so every product has a factor with at most three integer values (a
//! grade or a bit) and every argmax key is `unit(n * score + tie-break)`.
//! `armageddon_check` evaluates each compiled network against `DeepProgram::eval` on random
//! in-domain inputs; the game (entropy-engine, `ar_wir.ts`) checks the exported weights against
//! the same rules written as plain TypeScript.

use crate::deep;
use crate::error::{Result, WirError};
use crate::swarm::{sector, turn, xorshift, DIRS};

/// Local distance fields are passed relative to the agent's cell and clipped to this.
pub const CEIL: i64 = 48;
pub const ALLY_CAP: i64 = 3;
pub const HOLD: usize = 8;

/// One network's source with its declared inputs, for checks and export.
#[derive(Debug, Clone)]
pub struct Net {
    pub name: String,
    pub kind: String,
    pub variant: String,
    pub src: String,
    /// Scalar inputs in order (name, lo, hi); empty for the grid program.
    pub inputs: Vec<(String, i64, i64)>,
    pub consts: serde_json::Value,
}

struct Prog {
    src: String,
    inputs: Vec<(String, i64, i64)>,
}

impl Prog {
    fn new(header: &str) -> Self {
        Self { src: format!("wir 0.2\n{header}\n"), inputs: vec![] }
    }
    fn input(&mut self, names: &[String], lo: i64, hi: i64) {
        self.src += &format!("input {}: R1 in [{lo}, {hi}]\n", names.join(", "));
        for n in names {
            self.inputs.push((n.clone(), lo, hi));
        }
    }
    fn one(&mut self, name: &str, lo: i64, hi: i64) {
        self.input(&[name.to_string()], lo, hi);
    }
    fn line(&mut self, s: impl AsRef<str>) {
        self.src += s.as_ref();
        self.src += "\n";
    }
    /// One-hot argmax over `scores` (let names), outputs `{out}0..`, keys unique by tie-break
    /// (lower index wins ties).
    fn argmax(&mut self, scores: &[String], out: &str) {
        let n = scores.len() as i64;
        for (d, s) in scores.iter().enumerate() {
            self.line(format!("let {out}key{d} = unit({n}*({s}) + {})", n - 1 - d as i64));
        }
        for d in 0..scores.len() {
            for e in d + 1..scores.len() {
                self.line(format!("let {out}w{d}_{e} = gt({out}key{d}, {out}key{e})"));
            }
        }
        for d in 0..scores.len() {
            let parts: Vec<String> = (0..scores.len()).filter(|&e| e != d).map(|e| if e > d { format!("{out}w{d}_{e}") } else { format!("not({out}w{e}_{d})") }).collect();
            if parts.len() == 1 {
                self.line(format!("output {out}{d} = {}", parts[0]));
            } else {
                self.line(format!("output {out}{d} = and({})", parts.join(", ")));
            }
        }
    }
}

fn names(p: &str, n: usize) -> Vec<String> {
    (0..n).map(|k| format!("{p}{k}")).collect()
}

fn term(k: i64, x: &str) -> String {
    match k {
        0 => String::new(),
        1 => format!(" + {x}"),
        -1 => format!(" - {x}"),
        k if k < 0 => format!(" - {}*{x}", -k),
        k => format!(" + {k}*{x}"),
    }
}

fn affine(parts: &[(i64, String)], c: i64) -> String {
    let mut s: String = parts.iter().map(|(k, x)| term(*k, x)).collect();
    if c != 0 || s.is_empty() {
        s += &term(c.signum(), &c.abs().to_string());
        if c == 0 {
            s = " + 0".into();
        }
    }
    match s.strip_prefix(" - ") {
        Some(r) => format!("-{r}"),
        None => s.trim_start_matches(" + ").to_string(),
    }
}

/// Keep-the-current-option scores for 0..4 steps of 45 degrees, and for holding again.
pub const KEEP: [i64; 5] = [3, 1, 0, -1, -3];
pub const KEEP_HOLD: i64 = 1;

fn inertia(option: usize, current: usize) -> i64 {
    match (option == HOLD, current == HOLD) {
        (false, false) => KEEP[turn(option, current)],
        (true, true) => KEEP_HOLD,
        _ => 0,
    }
}

/// The packed rest of a heading score: spacing, inertia, blocked penalty.
fn heading_rest(p: &mut Prog, spacing: i64, big: i64, extra: &[(i64, String)]) {
    for d in 0..9 {
        let mut parts: Vec<(i64, String)> = vec![];
        if d < 8 {
            parts.push((-spacing, format!("n{d}")));
        }
        for k in 0..9 {
            parts.push((inertia(d, k), format!("o{k}")));
        }
        if d < 8 {
            parts.push((-big, format!("b{d}")));
            parts.extend(extra.iter().cloned());
        }
        p.line(format!("let k{d} = unit({})", affine(&parts, 0)));
    }
}

/// The 58 senses (in order) the brains read: go gt gc gl e n b (8 each), es, gcs.
pub fn sense_names() -> Vec<String> {
    let mut v = vec![];
    for p in ["go", "gt", "gc", "gl", "e", "n", "b"] {
        v.extend(names(p, 8));
    }
    v.push("es".into());
    v.push("gcs".into());
    v
}

fn sense_inputs(p: &mut Prog) {
    for g in ["go", "gt", "gc", "gl"] {
        p.input(&names(g, 8), -1, 1);
    }
    p.input(&names("e", 8), 0, 1);
    p.input(&names("n", 8), 0, ALLY_CAP);
    p.input(&names("b", 8), 0, 1);
    p.one("es", 0, 1);
    p.one("gcs", -1, 1);
}

fn tap(ch: &str, dy: i64, dx: i64) -> String {
    if dy == 0 && dx == 0 { ch.to_string() } else { format!("{ch}[{dy}, {dx}]") }
}

/// The shared sense grid program.
pub fn sense_program() -> String {
    let mut src = format!(
        "wir 0.2
# Armageddon 2070 senses (generated by `wir armageddon`). A grid program read at an agent's
# cell; the host fills a 5 x 5 window around it. Distance fields are relative to the agent
# and clipped to [0, {CEIL}]: dobj to its objective, dthr to the nearest threat, dcov to cover,
# dlead to its squad anchor (leader, rally point or the squad it supports). exp is 1 where a
# known enemy sees the cell, wall 1 where the agent cannot go, ally 1 where a friend stands.
# Headings 0..7 are E NE N NW W SW S SE; a straight step costs 2, a diagonal 3.

input dobj: grid in [0, {CEIL}] pad {CEIL}
input dthr: grid in [0, {CEIL}] pad {CEIL}
input dcov: grid in [0, {CEIL}] pad {CEIL}
input dlead: grid in [0, {CEIL}] pad {CEIL}
input exp: grid in [0, 1] pad 0
input wall: grid in [0, 1] pad 1
input ally: grid in [0, 1] pad 0

# grades: 1 within 1 of the best neighbour, 0 within 3, -1 farther
"
    );
    let fields = [("dobj", "go", "mo"), ("dthr", "gt", "mt"), ("dcov", "gc", "mc"), ("dlead", "gl", "ml")];
    for (field, _, best) in fields {
        let t: Vec<String> = DIRS.iter().map(|&(dy, dx)| tap(field, dy, dx)).collect();
        src += &format!("let {best} = min({})\n", t.join(", "));
    }
    for (field, g, best) in fields {
        for (d, (dy, dx)) in DIRS.iter().enumerate() {
            let t = tap(field, *dy, *dx);
            src += &format!("output {g}{d} = le({t} - {best}, 1) - ge({t} - {best}, 4)\n");
        }
    }
    src += "\n# e: a known enemy sees the next cell\n";
    for (d, (dy, dx)) in DIRS.iter().enumerate() {
        src += &format!("output e{d} = {}\n", tap("exp", *dy, *dx));
    }
    src += &format!("\n# n: friends per pinwheel sector of the 5 x 5 window, capped at {ALLY_CAP}\n");
    for d in 0..8 {
        let cells: Vec<String> = (-2..=2i64)
            .flat_map(|dy| (-2..=2i64).map(move |dx| (dy, dx)))
            .filter(|&(dy, dx)| sector(dy, dx) == Some(d))
            .map(|(dy, dx)| tap("ally", dy, dx))
            .collect();
        src += &format!("output n{d} = {ALLY_CAP} - relu({ALLY_CAP} - ({}))\n", cells.join(" + "));
    }
    src += "\n# b: blocked by a wall or a friend; a diagonal also by a wall beside it\n";
    for (d, (dy, dx)) in DIRS.iter().enumerate() {
        let mut parts = vec![tap("wall", *dy, *dx), tap("ally", *dy, *dx)];
        if *dy != 0 && *dx != 0 {
            parts.push(tap("wall", *dy, 0));
            parts.push(tap("wall", 0, *dx));
        }
        src += &format!("output b{d} = or({})\n", parts.join(", "));
    }
    src += "\n# the agent's own cell: seen, and how its cover compares with the best neighbour\noutput es = exp\noutput gcs = le(dcov - mc, 1) - ge(dcov - mc, 4)\n";
    src
}

// --- infantry --------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Infantry {
    pub name: &'static str,
    /// Preferred threat distance (5 m units), fire range, min range for planting, launcher range.
    pub range: i64,
    pub fire: i64,
    pub min_range: i64,
    pub launcher: i64,
    pub knee1: i64,
    pub f1: i64,
    pub knee2: i64,
    pub f2: i64,
    pub fe: i64,
    pub fs: i64,
    pub approach: i64,
    pub obj: i64,
    pub leash: i64,
    pub coh: i64,
    pub cover_from: i64,
    pub cover: i64,
    pub peek: i64,
    pub hide: i64,
    pub plant: i64,
    pub spacing: i64,
    pub reload_at: i64,
    pub crouch: i64,
}

impl Infantry {
    pub fn all() -> Vec<Self> {
        let rifle = Self { name: "rifle", range: 8, fire: 30, min_range: 2, launcher: 24, knee1: 6, f1: 1, knee2: 3, f2: 2, fe: 6, fs: 2, approach: 4, obj: 5, leash: 3, coh: 6, cover_from: 3, cover: 8, peek: 3, hide: 3, plant: 4, spacing: 2, reload_at: 2, crouch: 5 };
        vec![
            rifle.clone(),
            Self { name: "support", range: 14, fire: 34, min_range: 4, plant: 7, fs: 1, approach: 3, peek: 4, ..rifle.clone() },
            Self { name: "marksman", range: 24, fire: 40, min_range: 10, plant: 8, knee1: 7, cover_from: 2, cover: 10, peek: 5, hide: 4, leash: 6, spacing: 3, approach: 3, ..rifle.clone() },
            Self { name: "antiarmor", range: 12, fire: 24, min_range: 3, launcher: 40, fs: 2, cover_from: 2, ..rifle },
        ]
    }

    fn big(&self) -> i64 {
        let hi = self.approach + self.obj + self.coh + self.cover + self.peek + KEEP[0];
        let lo = -self.approach - self.obj - self.coh - self.cover - self.hide - ALLY_CAP * self.spacing + KEEP[4];
        let hold_hi = self.cover + self.peek + self.plant + 2 + KEEP_HOLD;
        let hold_lo = -self.cover - self.hide;
        hi.max(hold_hi) - lo.min(hold_lo) + 1
    }

    pub fn consts(&self) -> serde_json::Value {
        serde_json::json!({
            "range": self.range, "fire": self.fire, "min_range": self.min_range, "launcher": self.launcher,
            "knee1": self.knee1, "f1": self.f1, "knee2": self.knee2, "f2": self.f2, "fe": self.fe, "fs": self.fs,
            "approach": self.approach, "obj": self.obj, "leash": self.leash, "coh": self.coh,
            "cover_from": self.cover_from, "cover": self.cover, "peek": self.peek, "hide": self.hide,
            "plant": self.plant, "spacing": self.spacing, "reload_at": self.reload_at, "crouch": self.crouch,
            "big": self.big(),
        })
    }
}

pub fn infantry_program(a: &Infantry) -> Net {
    let mut p = Prog::new(&format!(
        "# Armageddon 2070 infantry brain: {} (generated by `wir armageddon`).
# o: current option (8 headings, 8 hold). Modes m0..m6 come from the squad brain: advance,
# assault, defend, regroup, flank, fall back, mount. dt is the nearest threat (5 m units),
# dl the distance to the squad anchor (the leader, or the player's call point while call is 1).
# Outputs: the option one-hot, fire, reload, crouch, grenade, launcher.",
        a.name
    ));
    p.input(&names("o", 9), 0, 1);
    p.one("hp", 1, 10);
    p.one("ammo", 0, 6);
    p.one("supp", 0, 4);
    p.input(&names("m", 7), 0, 1);
    p.one("call", 0, 1);
    p.one("dt", 0, 40);
    p.one("vis", 0, 1);
    p.one("armor", 0, 1);
    p.one("dl", 0, 12);
    p.one("gren", 0, 1);
    p.one("hasl", 0, 1);
    sense_inputs(&mut p);
    p.line("\n# --- response curves ---");
    p.line("let empty = relu(1 - ammo)");
    p.line(format!("let fear = {}*relu({} - hp) + {}*relu({} - hp) + {}*empty + {}*supp", a.f1, a.knee1, a.f2, a.knee2, a.fe, a.fs));
    p.line(format!("let approach = clip(dt - {} - fear + 4*m1 - 6*m2 - 8*m3 - 12*m5 - 8*m6, -{}, {})", a.range, a.approach, a.approach));
    p.line(format!("let obj = clip(3*m0 + 4*m1 + m2 + 3*m4 + 5*m5 + 6*m6 - 2*vis, 0, {})", a.obj));
    p.line(format!("let coh = clip(dl - {} + 4*m3 + 6*call, 0, {})", a.leash, a.coh));
    p.line(format!("let cover = clip(fear - {} + 2*m2 + 2*m5, 0, {})", a.cover_from, a.cover));
    p.line(format!("let peek = clip({} - fear, -{}, {})", a.peek, a.hide, a.peek));
    p.line(format!("let plant = {}*and(es, vis, ge(ammo, 1), le(dt, {}), ge(dt, {})) + 2*m2", a.plant, a.fire, a.min_range));
    p.line("\n# --- scores: drive x field grade, plus spacing, inertia and blocked headings ---");
    heading_rest(&mut p, a.spacing, a.big(), &[]);
    let mut scores = vec![];
    for d in 0..8 {
        p.line(format!("let s{d} = approach*gt{d} + obj*go{d} + coh*gl{d} + cover*gc{d} + peek*e{d} + k{d}"));
        scores.push(format!("s{d}"));
    }
    p.line("let s8 = cover*gcs + peek*es + plant + k8");
    scores.push("s8".into());
    p.argmax(&scores, "o2_");
    p.line(format!("output fire = and(vis, ge(ammo, 1), le(dt, {}))", a.fire));
    p.line(format!("output reload = or(le(ammo, 0), and(not(vis), le(ammo, {})))", a.reload_at));
    p.line(format!("output crouch = or(ge(fear, {}), and(vis, ge(dt, 4)))", a.crouch));
    p.line("output grenade = and(gren, vis, le(dt, 6), ge(dt, 2))");
    p.line(format!("output launch = and(hasl, armor, vis, le(dt, {}))", a.launcher));
    Net { name: format!("ar_{}_brain", a.name), kind: "infantry".into(), variant: a.name.into(), src: p.src, inputs: p.inputs, consts: a.consts() }
}

// --- squad and strategic ----------------------------------------------------------------------

pub fn squad_program() -> Net {
    let mut p = Prog::new(
        "# Armageddon 2070 squad brain (generated by `wir armageddon`): the squad's mode.
# c: current mode (advance, assault, defend, regroup, flank, fall back, mount). alive and hp
# (mean, 0..10) measure strength; contact enemies seen, armor enemy vehicles seen, farmor a
# friendly vehicle with the squad; dobj the objective's distance (25 m units); ow0..ow2 who holds
# it (enemy, nobody, us); cont it is being contested; spread the squad's radius (5 m units);
# call the player called the squad; veh an empty friendly transport is near; mounted aboard.",
    );
    p.input(&names("c", 7), 0, 1);
    p.one("alive", 0, 8);
    p.one("hp", 0, 10);
    p.one("contact", 0, 8);
    p.one("armor", 0, 3);
    p.one("farmor", 0, 1);
    p.one("dobj", 0, 20);
    p.input(&names("ow", 3), 0, 1);
    p.one("cont", 0, 1);
    p.one("supp", 0, 4);
    p.one("spread", 0, 10);
    p.one("call", 0, 1);
    p.one("veh", 0, 1);
    p.one("mounted", 0, 1);
    p.line("let strength = alive + hp");
    p.line("let danger = min(contact, 6) + 3*armor - 2*farmor + supp");
    p.line("let s0 = 4 + min(dobj, 8) - 2*min(contact, 3) - 3*ow2 + 2*c0");
    p.line("let s1 = select(and(le(dobj, 3), not(ow2)), 6 + min(contact, 3) + 2*cont, 0) + 2*c1");
    p.line("let s2 = 3*ow2 + 4*and(ow2, or(cont, ge(contact, 1))) + 2*and(ow2, le(dobj, 2)) + 2*c2");
    p.line("let s3 = clip(spread - 4, 0, 6) + 14*call + 3*le(alive, 3) + 2*c3");
    p.line("let s4 = select(and(ge(contact, 2), le(dobj, 8), ge(strength, 12), le(armor, 0)), 5 + min(contact, 3), 0) + 2*c4");
    p.line("let s5 = clip(danger + 6 - strength, 0, 12) + 2*c5");
    p.line("let s6 = 7*and(veh, ge(dobj, 10), not(mounted)) + 9*and(mounted, ge(dobj, 4)) + 2*c6");
    p.argmax(&names("s", 7), "mode");
    Net { name: "ar_squad_brain".into(), kind: "squad".into(), variant: "squad".into(), src: p.src, inputs: p.inputs, consts: serde_json::json!({}) }
}

pub const SLOTS: usize = 12;

pub fn strategic_program() -> Net {
    let mut p = Prog::new(&format!(
        "# Armageddon 2070 strategic brain (generated by `wir armageddon`): which of up to {SLOTS}
# territories a squad goes for. Per slot k: v valid, en held by the enemy, ne by nobody
# (else us), ct contested, val its value (1..3), d its distance (64 m units), fr friendly squads
# already sent there, fo enemy strength seen there (squads), cur the squad's current pick."
    ));
    for k in 0..SLOTS {
        p.input(&[format!("v{k}"), format!("en{k}"), format!("ne{k}"), format!("ct{k}"), format!("cur{k}")], 0, 1);
        p.one(&format!("val{k}"), 1, 3);
        p.one(&format!("d{k}"), 0, 15);
        p.input(&[format!("fr{k}"), format!("fo{k}")], 0, 6);
    }
    let mut scores = vec![];
    for k in 0..SLOTS {
        p.line(format!("let ours{k} = 1 - en{k} - ne{k}"));
        p.line(format!("let th{k} = min(fo{k}, 3)"));
        p.line(format!(
            "let s{k} = 3*val{k} + 5*en{k} + 3*ne{k} + 4*and(ct{k}, ours{k}) + 2*th{k}*ours{k} - th{k}*en{k} - d{k} - 3*fr{k} + 3*cur{k} - 60*(1 - v{k})"
        ));
        scores.push(format!("s{k}"));
    }
    p.argmax(&scores, "a");
    Net { name: "ar_strategic_brain".into(), kind: "strategic".into(), variant: "strategic".into(), src: p.src, inputs: p.inputs, consts: serde_json::json!({ "slots": SLOTS }) }
}

// --- vehicles --------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Tank {
    pub name: &'static str,
    /// Preferred distance (25 m units), fire range; fear weights; target priorities (tank, at, inf, air).
    pub range: i64,
    pub fire: i64,
    pub f1: i64,
    pub f2: i64,
    pub fa: i64,
    pub ft: i64,
    pub approach: i64,
    pub obj: i64,
    pub cover_from: i64,
    pub cover: i64,
    pub peek: i64,
    pub hide: i64,
    pub plant: i64,
    pub spacing: i64,
    pub prio: [i64; 4],
}

impl Tank {
    pub fn all() -> Vec<Self> {
        let mbt = Self { name: "mbt", range: 10, fire: 40, f1: 2, f2: 2, fa: 2, ft: 1, approach: 4, obj: 5, cover_from: 2, cover: 6, peek: 3, hide: 3, plant: 5, spacing: 2, prio: [9, 7, 4, 0] };
        vec![
            mbt.clone(),
            Self { name: "destroyer", range: 18, fire: 40, ft: 0, plant: 7, cover: 8, prio: [10, 3, 2, 0], ..mbt.clone() },
            Self { name: "antiair", range: 14, fire: 40, fa: 3, obj: 4, prio: [1, 2, 3, 12], ..mbt.clone() },
            Self { name: "artillery", range: 30, fire: 40, fa: 3, ft: 2, obj: 3, plant: 8, prio: [6, 5, 6, 0], ..mbt },
        ]
    }
    fn big(&self) -> i64 {
        let hi = self.approach + self.obj + 4 + self.cover + self.peek + KEEP[0];
        let lo = -self.approach - self.obj - 4 - self.cover - self.hide - ALLY_CAP * self.spacing + KEEP[4];
        let hold_hi = self.cover + self.peek + self.plant + 3 + KEEP_HOLD;
        hi.max(hold_hi) - lo.min(-self.cover - self.hide) + 1
    }
    pub fn consts(&self) -> serde_json::Value {
        serde_json::json!({ "range": self.range, "fire": self.fire, "f1": self.f1, "f2": self.f2, "fa": self.fa, "ft": self.ft, "approach": self.approach, "obj": self.obj, "cover_from": self.cover_from, "cover": self.cover, "peek": self.peek, "hide": self.hide, "plant": self.plant, "spacing": self.spacing, "prio": self.prio, "big": self.big() })
    }
}

fn target_keys(p: &mut Prog, prio: [i64; 4]) {
    // Target class: tank, anti-tank infantry, infantry, aircraft, or none.
    let counts = ["tanks", "at", "inf", "air"];
    let mut scores = vec![];
    for (c, w) in counts.iter().zip(prio) {
        p.line(format!("let t_{c} = {}*ge({c}, 1)", 2 * w));
        scores.push(format!("t_{c}"));
    }
    p.line("let t_none = 1");
    scores.push("t_none".into());
    p.argmax(&scores, "tgt");
}

pub fn tank_program(a: &Tank) -> Net {
    let mut p = Prog::new(&format!(
        "# Armageddon 2070 tank brain: {} (generated by `wir armageddon`). The hull picks a heading
# (or holds, hull-down when cover is good); the turret is aimed by the host at the class this
# network picks (tgt0..tgt4: tank, anti-tank infantry, infantry, aircraft, none). tdist is the
# nearest threat (25 m units); g0..g2 the order (attack, support a squad, hold).",
        a.name
    ));
    p.input(&names("o", 9), 0, 1);
    p.one("hp", 1, 10);
    p.one("loaded", 0, 1);
    p.one("smoke", 0, 1);
    p.one("tanks", 0, 3);
    p.one("at", 0, 3);
    p.one("inf", 0, 8);
    p.one("air", 0, 2);
    p.one("tdist", 0, 40);
    p.one("vis", 0, 1);
    p.input(&names("g", 3), 0, 1);
    p.one("dl", 0, 12);
    sense_inputs(&mut p);
    p.line(format!("let fear = {}*relu(5 - hp) + {}*relu(3 - hp) + {}*min(at, 3) + {}*min(tanks, 2)", a.f1, a.f2, a.fa, a.ft));
    p.line(format!("let approach = clip(tdist - {} - fear - 6*g2, -{}, {})", a.range, a.approach, a.approach));
    p.line(format!("let obj = clip({} - fear + 2*g0 - 3*g2, 0, {})", a.obj, a.obj + 2));
    p.line("let sup = g1*clip(dl - 3, 0, 4)");
    p.line(format!("let cover = clip(fear - {} + 2*min(tanks, 1), 0, {})", a.cover_from, a.cover));
    p.line(format!("let peek = clip({} - fear, -{}, {})", a.peek, a.hide, a.peek));
    p.line(format!("let plant = {}*and(es, vis, loaded, le(tdist, {})) + 3*g2", a.plant, a.fire));
    heading_rest(&mut p, a.spacing, a.big(), &[]);
    let mut scores = vec![];
    for d in 0..8 {
        p.line(format!("let s{d} = approach*gt{d} + obj*go{d} + sup*gl{d} + cover*gc{d} + peek*e{d} + k{d}"));
        scores.push(format!("s{d}"));
    }
    p.line("let s8 = cover*gcs + peek*es + plant + k8");
    scores.push("s8".into());
    p.argmax(&scores, "o2_");
    target_keys(&mut p, a.prio);
    p.line(format!("output fire = and(vis, loaded, le(tdist, {}))", a.fire));
    p.line("output coax = and(vis, ge(inf, 1), le(tdist, 16))");
    p.line("output smoke2 = and(smoke, or(ge(at, 2), and(le(hp, 4), vis)))");
    Net { name: format!("ar_tank_{}", a.name), kind: "tank".into(), variant: a.name.into(), src: p.src, inputs: p.inputs, consts: a.consts() }
}

#[derive(Debug, Clone)]
pub struct Apc {
    pub name: &'static str,
    pub range: i64,
    pub fire: i64,
    pub gun: bool,
    pub f1: i64,
    pub fa: i64,
    pub ft: i64,
    pub approach: i64,
    pub deliver: i64,
    pub cover_from: i64,
    pub cover: i64,
    pub peek: i64,
    pub hide: i64,
    pub plant: i64,
    pub spacing: i64,
}

impl Apc {
    pub fn all() -> Vec<Self> {
        let apc = Self { name: "apc", range: 14, fire: 32, gun: true, f1: 2, fa: 2, ft: 3, approach: 3, deliver: 6, cover_from: 2, cover: 6, peek: 2, hide: 3, plant: 3, spacing: 2 };
        vec![
            apc.clone(),
            Self { name: "ifv", range: 10, fire: 40, approach: 4, plant: 5, ft: 2, ..apc.clone() },
            Self { name: "light", range: 20, fire: 14, f1: 3, fa: 3, ft: 4, cover: 8, plant: 2, ..apc },
        ]
    }
    fn big(&self) -> i64 {
        let hi = self.approach + self.deliver + 6 + self.cover + self.peek + KEEP[0];
        let lo = -self.approach - self.deliver - 6 - self.cover - self.hide - ALLY_CAP * self.spacing + KEEP[4];
        let hold_hi = self.cover + self.peek + self.plant + 6 + KEEP_HOLD;
        hi.max(hold_hi) - lo.min(-self.cover - self.hide) + 1
    }
    pub fn consts(&self) -> serde_json::Value {
        serde_json::json!({ "range": self.range, "fire": self.fire, "gun": self.gun, "f1": self.f1, "fa": self.fa, "ft": self.ft, "approach": self.approach, "deliver": self.deliver, "cover_from": self.cover_from, "cover": self.cover, "peek": self.peek, "hide": self.hide, "plant": self.plant, "spacing": self.spacing, "big": self.big() })
    }
}

pub fn apc_program(a: &Apc) -> Net {
    let mut p = Prog::new(&format!(
        "# Armageddon 2070 transport brain: {} (generated by `wir armageddon`). Carries its squad to
# the objective (go grades), comes back for members not aboard (gl grades toward them), dismounts
# them near the objective, in a firefight or when badly hurt, and supports with its gun.
# pax aboard, wait members still to pick up, near some of them within 15 m.",
        a.name
    ));
    p.input(&names("o", 9), 0, 1);
    p.one("hp", 1, 10);
    p.one("pax", 0, 6);
    p.one("wait", 0, 6);
    p.one("near", 0, 1);
    p.one("dobj", 0, 20);
    p.one("at", 0, 3);
    p.one("tanks", 0, 3);
    p.one("inf", 0, 8);
    p.one("tdist", 0, 40);
    p.one("vis", 0, 1);
    p.one("atgm", 0, 1);
    p.one("smoke", 0, 1);
    sense_inputs(&mut p);
    p.line(format!("let fear = {}*relu(5 - hp) + {}*min(at, 3) + {}*min(tanks, 2) + min(pax, 3)", a.f1, a.fa, a.ft));
    p.line(format!("let approach = clip(tdist - {} - fear, -{}, {})", a.range, a.approach, a.approach));
    p.line(format!("let deliver = clip(3 + min(pax, 3) - fear + 2*le(wait, 0), 0, {})", a.deliver));
    p.line("let pickup = select(le(pax, 0), clip(2*min(wait, 3), 0, 6), 0)");
    p.line(format!("let cover = clip(fear - {}, 0, {})", a.cover_from, a.cover));
    p.line(format!("let peek = clip({} - fear, -{}, {})", a.peek, a.hide, a.peek));
    let gunplant = if a.gun { format!("{}*and(es, vis, le(tdist, {}))", a.plant, a.fire) } else { "0".into() };
    p.line(format!("let plant = 6*and(near, ge(wait, 1)) + {gunplant}"));
    heading_rest(&mut p, a.spacing, a.big(), &[]);
    let mut scores = vec![];
    for d in 0..8 {
        p.line(format!("let s{d} = approach*gt{d} + deliver*go{d} + pickup*gl{d} + cover*gc{d} + peek*e{d} + k{d}"));
        scores.push(format!("s{d}"));
    }
    p.line("let s8 = cover*gcs + peek*es + plant + k8");
    scores.push("s8".into());
    p.argmax(&scores, "o2_");
    p.line("output dismount = and(ge(pax, 1), or(le(dobj, 2), and(le(tdist, 8), ge(inf, 1)), le(hp, 3)))");
    if a.gun {
        p.line(format!("output fire = and(vis, le(tdist, {}))", a.fire));
    } else {
        p.line("output fire = 0*vis");
    }
    p.line("output atgm2 = and(atgm, vis, ge(tanks, 1), le(tdist, 40))");
    p.line("output smoke2 = and(smoke, vis, or(ge(at, 1), le(hp, 4)))");
    Net { name: format!("ar_apc_{}", a.name), kind: "apc".into(), variant: a.name.into(), src: p.src, inputs: p.inputs, consts: a.consts() }
}

#[derive(Debug, Clone)]
pub struct Heli {
    pub name: &'static str,
    pub range: i64,
    pub fire: i64,
    pub f1: i64,
    pub fa: i64,
    pub approach: i64,
    pub obj: i64,
    pub cover_from: i64,
    pub cover: i64,
    pub plant: i64,
    pub spacing: i64,
}

impl Heli {
    pub fn all() -> Vec<Self> {
        let attack = Self { name: "attack", range: 16, fire: 36, f1: 2, fa: 2, approach: 4, obj: 5, cover_from: 3, cover: 6, plant: 5, spacing: 2 };
        vec![
            attack.clone(),
            Self { name: "transport", range: 30, fire: 12, fa: 3, approach: 2, obj: 6, plant: 2, ..attack.clone() },
            Self { name: "drone", range: 10, fire: 24, f1: 1, fa: 1, approach: 5, cover: 3, ..attack },
        ]
    }
    fn big(&self) -> i64 {
        let hi = self.approach + self.obj + 2 + self.cover + 1 + KEEP[0];
        let lo = -self.approach - self.obj - 2 - self.cover - 4 - ALLY_CAP * self.spacing + KEEP[4];
        let hold_hi = 6 + self.plant + KEEP_HOLD;
        hi.max(hold_hi) - lo + 1
    }
    pub fn consts(&self) -> serde_json::Value {
        serde_json::json!({ "range": self.range, "fire": self.fire, "f1": self.f1, "fa": self.fa, "approach": self.approach, "obj": self.obj, "cover_from": self.cover_from, "cover": self.cover, "plant": self.plant, "spacing": self.spacing, "big": self.big() })
    }
}

pub fn heli_program(a: &Heli) -> Net {
    let mut p = Prog::new(&format!(
        "# Armageddon 2070 helicopter brain: {} (generated by `wir armageddon`). On a 16 m air grid:
# heading or hover, altitude band (low: nap of the earth, mid, high), land, fire, flares.
# q0..q2 mission (attack, transport, extract); r0..r7 random bits that nudge headings so no two
# approaches are the same; aa anti-air threats near, lock a missile lock on us; lz the cell below
# is a good landing zone.",
        a.name
    ));
    p.input(&names("o", 9), 0, 1);
    p.one("hp", 1, 10);
    p.input(&names("a", 3), 0, 1);
    p.one("pax", 0, 8);
    p.one("aa", 0, 3);
    p.one("lock", 0, 1);
    p.one("flares", 0, 1);
    p.one("dobj", 0, 20);
    p.one("lz", 0, 1);
    p.input(&names("q", 3), 0, 1);
    p.one("tg", 0, 8);
    p.one("tdist", 0, 40);
    p.one("vis", 0, 1);
    p.input(&names("r", 8), 0, 1);
    sense_inputs(&mut p);
    p.line(format!("let fear = {}*relu(5 - hp) + {}*min(aa, 3) + 6*lock", a.f1, a.fa));
    p.line(format!("let obj = clip({} - fear + 2*q1 + 2*q2, 0, {})", a.obj, a.obj + 2));
    p.line(format!("let approach = q0*clip(tdist - {} - fear, -{}, {})", a.range, a.approach, a.approach));
    p.line(format!("let cover = clip(fear - {}, 0, {})", a.cover_from, a.cover));
    p.line("let avoid = clip(2*aa, 0, 4)");
    p.line(format!("let plant = 6*and(or(q1, q2), le(dobj, 1)) + {}*and(q0, vis, le(tdist, {}))", a.plant, a.fire));
    heading_rest(&mut p, a.spacing, a.big(), &[]);
    let mut scores = vec![];
    for d in 0..8 {
        p.line(format!("let s{d} = obj*go{d} + approach*gt{d} + cover*gc{d} - avoid*e{d} + r{d} + k{d}"));
        scores.push(format!("s{d}"));
    }
    p.line("let s8 = plant + k8");
    scores.push("s8".into());
    p.argmax(&scores, "o2_");
    p.line("let band0 = 2*min(aa, 2) + 4*le(dobj, 2) + 2*and(q0, vis) + 2*a0");
    p.line("let band1 = 3 + 2*a1");
    p.line("let band2 = 4*ge(dobj, 8) - 3*min(aa, 2) - 3*q0 + 2*a2");
    p.argmax(&["band0".into(), "band1".into(), "band2".into()], "alt");
    p.line("output land = and(or(q1, q2), le(dobj, 1), lz, le(aa, 0))");
    p.line(format!("output fire = and(vis, le(tdist, {}), or(q0, ge(tg, 1)))", a.fire));
    p.line("output flare = and(lock, flares)");
    Net { name: format!("ar_heli_{}", a.name), kind: "heli".into(), variant: a.name.into(), src: p.src, inputs: p.inputs, consts: a.consts() }
}

#[derive(Debug, Clone)]
pub struct Jet {
    pub name: &'static str,
    /// Attack weights for ground targets and aircraft; fire range (100 m units); bombs.
    pub ground: i64,
    pub air: i64,
    pub fire: i64,
    pub bombs: bool,
}

impl Jet {
    pub fn all() -> Vec<Self> {
        vec![
            Self { name: "fighter", ground: 2, air: 9, fire: 9, bombs: false },
            Self { name: "attack", ground: 9, air: 2, fire: 10, bombs: true },
            Self { name: "transport", ground: 0, air: 0, fire: 0, bombs: false },
        ]
    }
    pub fn consts(&self) -> serde_json::Value {
        serde_json::json!({ "ground": self.ground, "air": self.air, "fire": self.fire, "bombs": self.bombs })
    }
}

pub fn jet_program(a: &Jet) -> Net {
    let mut p = Prog::new(&format!(
        "# Armageddon 2070 fixed-wing brain: {} (generated by `wir armageddon`). Phases p0..p4: patrol,
# attack run, pull out, evade, return to base. The host flies the phase. tg a ground target is
# designated, air enemy aircraft near, aa anti-air near, tdist the target (100 m units), alt
# height (50 m units), payload a vehicle aboard (transport) and dz over the drop zone.",
        a.name
    ));
    p.input(&names("p", 5), 0, 1);
    p.one("hp", 1, 10);
    p.one("ammo", 0, 6);
    p.one("tg", 0, 1);
    p.one("air", 0, 3);
    p.one("aa", 0, 3);
    p.one("lock", 0, 1);
    p.one("tdist", 0, 20);
    p.one("alt", 0, 6);
    p.one("flares", 0, 1);
    p.one("payload", 0, 1);
    p.one("dz", 0, 1);
    p.line("let s0 = 3 + 2*p0");
    p.line(format!("let s1 = select(ge(ammo, 1), {}*tg + {}*min(air, 1), 0) + 8*payload - 2*min(aa, 3) + 3*p1", a.ground, a.air));
    p.line("let s2 = 9*and(p1, or(le(tdist, 1), le(alt, 1))) + 4*and(p2, le(alt, 3))");
    p.line("let s3 = 8*lock + 2*min(aa, 3) + 2*p3");
    p.line("let s4 = 7*and(le(ammo, 0), le(payload, 0)) + 6*le(hp, 3) + 2*p4");
    p.argmax(&names("s", 5), "phase");
    if a.fire > 0 {
        p.line(format!("output fire = and(p1, ge(ammo, 1), le(tdist, {}), or(tg, ge(air, 1)))", a.fire));
    } else {
        p.line("output fire = 0*p1");
    }
    p.line("output flare = and(lock, flares)");
    if a.bombs {
        p.line("output drop = or(and(payload, dz), and(p1, tg, le(tdist, 2), ge(ammo, 1)))");
    } else {
        p.line("output drop = and(payload, dz)");
    }
    Net { name: format!("ar_jet_{}", a.name), kind: "jet".into(), variant: a.name.into(), src: p.src, inputs: p.inputs, consts: a.consts() }
}

#[derive(Debug, Clone)]
pub struct Mecha {
    pub name: &'static str,
    pub range: i64,
    pub fire: i64,
    pub approach: i64,
    pub obj: i64,
    pub cover_from: i64,
    pub cover: i64,
    pub peek: i64,
    pub hide: i64,
    pub plant: i64,
    /// Priorities [tank, at, inf, air] for the left arm, right arm and shoulder weapons.
    pub left: [i64; 4],
    pub right: [i64; 4],
    pub shoulder: [i64; 4],
}

impl Mecha {
    pub fn all() -> Vec<Self> {
        vec![
            Self { name: "titan", range: 10, fire: 40, approach: 4, obj: 6, cover_from: 4, cover: 5, peek: 3, hide: 2, plant: 4, left: [8, 5, 3, 0], right: [8, 5, 3, 0], shoulder: [3, 6, 7, 6] },
            Self { name: "wraith", range: 22, fire: 40, approach: 4, obj: 5, cover_from: 2, cover: 8, peek: 4, hide: 4, plant: 7, left: [1, 6, 8, 3], right: [10, 4, 2, 4], shoulder: [0, 0, 0, 0] },
            Self { name: "bastion", range: 12, fire: 36, approach: 2, obj: 4, cover_from: 5, cover: 4, peek: 4, hide: 2, plant: 8, left: [2, 7, 8, 5], right: [2, 7, 8, 5], shoulder: [5, 5, 6, 0] },
        ]
    }
    fn big(&self) -> i64 {
        let hi = self.approach + self.obj + self.cover + self.peek + KEEP[0];
        let lo = -self.approach - self.obj - self.cover - self.hide - 2 - ALLY_CAP * 2 + KEEP[4];
        let hold_hi = self.cover + self.peek + self.plant + 4 + KEEP_HOLD;
        hi.max(hold_hi) - lo.min(-self.cover - self.hide) + 1
    }
    pub fn consts(&self) -> serde_json::Value {
        serde_json::json!({ "range": self.range, "fire": self.fire, "approach": self.approach, "obj": self.obj, "cover_from": self.cover_from, "cover": self.cover, "peek": self.peek, "hide": self.hide, "plant": self.plant, "left": self.left, "right": self.right, "shoulder": self.shoulder, "big": self.big() })
    }
}

pub fn mecha_program(a: &Mecha) -> Net {
    let mut p = Prog::new(&format!(
        "# Armageddon 2070 mecha brain: {} (generated by `wir armageddon`). Component health in
# quarters (legs, left arm, right arm, torso, 0 = destroyed): lame legs make holding and bracing
# better than walking, a lost arm takes its weapon out of the choice (wpn0..wpn3: left, right,
# shoulder, none), a failing torso makes it fearful.",
        a.name
    ));
    p.input(&names("o", 9), 0, 1);
    p.one("legs", 0, 4);
    p.one("larm", 0, 4);
    p.one("rarm", 0, 4);
    p.one("torso", 1, 4);
    p.one("tanks", 0, 3);
    p.one("at", 0, 3);
    p.one("inf", 0, 8);
    p.one("air", 0, 2);
    p.one("tdist", 0, 40);
    p.one("vis", 0, 1);
    sense_inputs(&mut p);
    p.line("let fear = 3*relu(2 - torso) + relu(2 - legs) + 2*min(at, 3) + min(tanks, 2)");
    p.line("let lame = relu(2 - legs)");
    p.line(format!("let approach = clip(tdist - {} - fear, -{}, {})", a.range, a.approach, a.approach));
    p.line(format!("let obj = clip({} - fear, 0, {})", a.obj, a.obj));
    p.line(format!("let cover = clip(fear - {}, 0, {})", a.cover_from, a.cover));
    p.line(format!("let peek = clip({} - fear, -{}, {})", a.peek, a.hide, a.peek));
    p.line(format!("let plant = {}*and(es, vis, le(tdist, {})) + 2*lame", a.plant, a.fire));
    heading_rest(&mut p, 2, a.big(), &[(-1, "lame".into())]);
    let mut scores = vec![];
    for d in 0..8 {
        p.line(format!("let s{d} = approach*gt{d} + obj*go{d} + cover*gc{d} + peek*e{d} + k{d}"));
        scores.push(format!("s{d}"));
    }
    p.line("let s8 = cover*gcs + peek*es + plant + k8");
    scores.push("s8".into());
    p.argmax(&scores, "o2_");
    p.line("let tp = ge(tanks, 1)\nlet ap = ge(at, 1)\nlet ip = ge(inf, 1)\nlet airp = ge(air, 1)");
    let mut keys = vec![];
    for (w, (pr, avail)) in [(a.left, "ge(larm, 1)"), (a.right, "ge(rarm, 1)"), (a.shoulder, "ge(torso, 2)")].iter().enumerate() {
        p.line(format!("let wp{w} = select({avail}, {}, -1)", affine(&[(2 * pr[0], "tp".into()), (2 * pr[1], "ap".into()), (2 * pr[2], "ip".into()), (2 * pr[3], "airp".into())], 0)));
        keys.push(format!("wp{w}"));
    }
    p.line("let wp3 = 1");
    keys.push("wp3".into());
    p.argmax(&keys, "wpn");
    p.line("output brace = and(vis, or(ge(at, 2), le(torso, 1), ge(tanks, 2)))");
    p.line(format!("output fire = and(vis, le(tdist, {}))", a.fire));
    Net { name: format!("ar_mecha_{}", a.name), kind: "mecha".into(), variant: a.name.into(), src: p.src, inputs: p.inputs, consts: a.consts() }
}

/// Every brain (the sense grid program is separate).
pub fn all_brains() -> Vec<Net> {
    let mut v: Vec<Net> = Infantry::all().iter().map(infantry_program).collect();
    v.push(squad_program());
    v.push(strategic_program());
    v.extend(Tank::all().iter().map(tank_program));
    v.extend(Apc::all().iter().map(apc_program));
    v.extend(Heli::all().iter().map(heli_program));
    v.extend(Jet::all().iter().map(jet_program));
    v.extend(Mecha::all().iter().map(mecha_program));
    v
}

pub struct Compiled {
    pub net: Net,
    pub weights: deep::DeepWeights,
    pub program: deep::DeepProgram,
}

pub fn compile(net: Net) -> Result<Compiled> {
    let c = deep::compile(&net.src, 1e-6).map_err(|e| WirError::new(format!("{}: {e}", net.name)))?;
    Ok(Compiled { net, weights: c.weights, program: c.program })
}

#[derive(Debug, Default, Clone)]
pub struct CheckReport {
    pub name: String,
    pub cases: u64,
    pub mismatches: u64,
    pub worst: f64,
    pub params: usize,
    pub hidden: Vec<usize>,
}

/// Random in-domain integer inputs: the compiled network against the program's own semantics.
pub fn check(c: &Compiled, cases: u64, seed: u64) -> CheckReport {
    let mut rng = seed.max(1);
    let ranges: Vec<(i64, i64)> = c.program.inputs.iter().map(|(_, (lo, hi))| (*lo as i64, *hi as i64)).collect();
    let mut net = deep::SparseNet::new(&c.weights);
    let mut r = CheckReport { name: c.net.name.clone(), cases, params: c.weights.param_count(), hidden: c.weights.hidden(), ..Default::default() };
    let mut x = vec![0.0; ranges.len()];
    for case in 0..cases {
        for (k, (lo, hi)) in ranges.iter().enumerate() {
            // One-hot groups (o, c, m, g, q, a, p, cur) are sampled as bits; the network is exact
            // on every in-domain point, so independent bits are a stronger test.
            let span = (hi - lo + 1) as u64;
            // Bias a third of the cases toward the ends of each interval.
            let v = if case % 3 == 0 && xorshift(&mut rng, 2) == 0 { if xorshift(&mut rng, 2) == 0 { *lo } else { *hi } } else { lo + xorshift(&mut rng, span) as i64 };
            x[k] = v as f64;
        }
        let want = c.program.eval(&x);
        let got = net.eval(&x);
        let mut bad = false;
        for (a, b) in want.iter().zip(got) {
            let d = (a - b).abs();
            if d > r.worst {
                r.worst = d;
            }
            if d > 1e-6 {
                bad = true;
            }
        }
        if bad {
            r.mismatches += 1;
        }
    }
    r
}

/// The export the game reads: per network sparse layers, input and output names, constants and
/// the source; the sense program as a stencil.
pub fn export_json(cases: u64, seed: u64) -> Result<(String, Vec<CheckReport>)> {
    let mut reports = vec![];
    let sc = deep::compile(&sense_program(), 1e-6)?;
    let sense = deep::StencilWeights::from_compiled(&sc).ok_or_else(|| WirError::new("sense is not a grid program"))?;
    let sense_c = Compiled { net: Net { name: "ar_sense".into(), kind: "sense".into(), variant: "sense".into(), src: sense_program(), inputs: vec![], consts: serde_json::json!({}) }, weights: sc.weights.clone(), program: sc.program.clone() };
    reports.push(check(&sense_c, cases, seed));
    let mut nets = serde_json::Map::new();
    for (i, net) in all_brains().into_iter().enumerate() {
        let c = compile(net)?;
        reports.push(check(&c, cases, seed + i as u64 + 1));
        nets.insert(
            c.net.name.clone(),
            serde_json::json!({
                "kind": c.net.kind, "variant": c.net.variant, "consts": c.net.consts,
                "inputs": c.program.inputs.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
                "ranges": c.program.inputs.iter().map(|(_, (lo, hi))| [*lo, *hi]).collect::<Vec<_>>(),
                "outputs": c.program.outputs.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
                "hidden": c.weights.hidden(), "params": c.weights.param_count(),
                "layers": crate::tactics::sparse_layers(&c.weights),
                "source": c.net.src,
            }),
        );
    }
    let v = serde_json::json!({
        "format": "wir-armageddon",
        "consts": { "ceil": CEIL, "ally_cap": ALLY_CAP, "keep": KEEP, "keep_hold": KEEP_HOLD, "slots": SLOTS },
        "sense": crate::tactics::stencil_json(&sense, &sense_program()),
        "nets": nets,
    });
    Ok((serde_json::to_string(&v).unwrap(), reports))
}

/// Writes every program's source into `dir`.
pub fn write_sources(dir: &std::path::Path) -> Result<Vec<String>> {
    std::fs::create_dir_all(dir).map_err(|e| WirError::new(e.to_string()))?;
    let mut files = vec![("ar_sense.wir".to_string(), sense_program())];
    for n in all_brains() {
        files.push((format!("{}.wir", n.name), n.src));
    }
    let mut out = vec![];
    for (name, src) in files {
        std::fs::write(dir.join(&name), src).map_err(|e| WirError::new(e.to_string()))?;
        out.push(name);
    }
    Ok(out)
}
