//! Tactical enemies: response curves over shared fields, with a reference simulator.
//!
//! The arena is a grid (walls, one player, a few enemies of several archetypes). Two
//! shared fields come from the `wir path` step, run a few iterations per tick: `dp`,
//! the distance to the player, and `dc`, the distance to cover (open cells the player
//! cannot see). The host traces the player's line of sight (`exp`, 1 where a cell is
//! visible). Every enemy runs the same **sense** grid program at its cell (gradients of
//! both fields, exposure and crowding by heading, blocked headings) and its archetype's
//! **brain**, a 0.2 utility program. The brain shapes hp, ammo and range into response
//! curves (fear, approach, cover, peek), scores 8 headings plus holding still, and
//! decides whether to fire or reload. Field gradients are -1, 0 or 1, so a drive times a
//! gradient is two selects and stays exact. Archetypes are constants in the source: each
//! compiles to its own network. `tactics_check` runs the networks against the rule as
//! plain code, enemy for enemy, and measures the behaviour.

use crate::deep;
use crate::error::{Result, WirError};
use crate::games::{self, PathGrid, PathRule};
use crate::swarm::{sector, turn, xorshift, DIRS};

/// Distance field ceiling (8 moves: a straight step costs 2, a diagonal 3).
pub const CEILING: i64 = 96;
/// Allies counted per sector of the 5 x 5 window are capped here.
pub const ALLY_CAP: i64 = 4;
pub const HP: i64 = 12;
pub const AMMO: i64 = 6;
/// Ticks between shots, ticks a reload takes.
pub const COOL: i64 = 2;
pub const RELOAD: i64 = 4;
/// The hold option (headings are 0..8).
pub const HOLD: usize = 8;
/// Sense outputs: gp, gc, e, n, b (8 each), then dp (after the best step), e and the
/// cover grade of the cell itself.
pub const SENSES: usize = 43;
pub const SENSE_CHANNELS: [&str; 5] = ["dp", "dc", "exp", "wall", "ally"];

/// Leaving a cell an ally stands on costs this many steps in the distance-to-player
/// field, so followers route around the enemies ahead of them.
pub const ALLY_COST: i64 = 3;

pub fn path_rule() -> PathRule {
    PathRule { moves: 8, ceiling: CEILING, terrain: ALLY_COST }
}

/// How a neighbour compares with the best neighbour: 1 within 1 of it (the best step or
/// as good), 0 within 3 (partial progress), -1 farther (sideways or back).
pub fn grade(v: i64, best: i64) -> i64 {
    (v - best <= 1) as i64 - (v - best >= 4) as i64
}

/// One kind of enemy: its curves and weights. Distances are in field units (2 per
/// straight cell).
#[derive(Debug, Clone, PartialEq)]
pub struct Archetype {
    pub name: String,
    /// Preferred distance to the player; fear adds to it.
    pub range: i64,
    /// Fires when it sees the player at this distance or closer.
    pub fire_range: i64,
    /// Plants (holds and fires) only this far out or farther.
    pub min_range: i64,
    /// Fear curve: `f1 * relu(knee1 - hp) + f2 * relu(knee2 - hp) + fe * empty` (empty:
    /// no ammo; reloading drops the magazine). The built-in archetypes keep
    /// `fe >= peek + hide`, so an empty enemy's peek drive bottoms out at -hide without a
    /// separate gate (the no-fear control drops that).
    pub knee1: i64,
    pub f1: i64,
    pub knee2: i64,
    pub f2: i64,
    pub fe: i64,
    /// Approach drive is `dp - range - fear`, clipped to +-approach.
    pub approach: i64,
    /// Cover drive is `fear - cover_from`, clipped to [0, cover].
    pub cover_from: i64,
    pub cover: i64,
    /// Exposure drive: `peek - fear` clipped to [-hide, peek].
    pub peek: i64,
    pub hide: i64,
    /// Holding while exposed, ready and in range scores this.
    pub plant: i64,
    /// Each ally in a heading's sector costs this.
    pub spacing: i64,
    /// Score for a heading 0..4 steps of 45 degrees from the current one, and for
    /// holding again after holding.
    pub keep: [i64; 5],
    pub keep_hold: i64,
    /// Reloads when out of ammo, or when hidden with this much or less.
    pub reload_at: i64,
    /// Damage per hit (host).
    pub damage: i64,
}

impl Archetype {
    /// Mid range, reasonably brave, keeps a little spacing.
    pub fn grunt() -> Self {
        Self {
            name: "grunt".into(),
            range: 10,
            fire_range: 18,
            min_range: 4,
            knee1: 8,
            f1: 2,
            knee2: 4,
            f2: 2,
            fe: 7,
            approach: 4,
            cover_from: 3,
            cover: 10,
            peek: 4,
            hide: 3,
            plant: 5,
            spacing: 2,
            keep: [3, 1, 0, -1, -3],
            keep_hold: 1,
            reload_at: 2,
            damage: 1,
        }
    }

    /// Closes to melee and barely flinches.
    pub fn rusher() -> Self {
        Self {
            name: "rusher".into(),
            range: 0,
            fire_range: 3,
            min_range: 0,
            knee1: 3,
            f1: 1,
            knee2: 0,
            f2: 0,
            fe: 2,
            approach: 6,
            cover_from: 6,
            cover: 4,
            peek: 1,
            hide: 0,
            plant: 2,
            spacing: 1,
            keep: [3, 1, 0, -1, -3],
            keep_hold: 0,
            reload_at: 0,
            damage: 3,
        }
    }

    /// Keeps its distance, likes cover, plants to shoot.
    pub fn sniper() -> Self {
        Self {
            name: "sniper".into(),
            range: 22,
            fire_range: 34,
            min_range: 14,
            knee1: 10,
            f1: 2,
            knee2: 6,
            f2: 2,
            fe: 10,
            approach: 4,
            cover_from: 2,
            cover: 12,
            peek: 5,
            hide: 5,
            plant: 8,
            spacing: 3,
            keep: [3, 1, 0, -1, -3],
            keep_hold: 2,
            reload_at: 3,
            damage: 2,
        }
    }

    pub fn all() -> Vec<Self> {
        vec![Self::grunt(), Self::rusher(), Self::sniper()]
    }

    pub fn validate(&self) -> Result<()> {
        let ok = self.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            && !self.name.is_empty()
            && (0..CEILING).contains(&self.range)
            && (1..CEILING).contains(&self.fire_range)
            && (0..CEILING).contains(&self.min_range)
            && (0..=HP).contains(&self.knee1)
            && (0..=HP).contains(&self.knee2)
            && (0..AMMO).contains(&self.reload_at)
            && self.approach >= 0
            && self.cover >= 0
            && self.peek >= 0
            && self.hide >= 0
            && self.plant >= 0
            && self.spacing >= 0
            && [self.f1, self.f2, self.fe].iter().all(|v| (0..=20).contains(v))
            && [self.approach, self.cover_from, self.cover, self.peek, self.hide, self.plant, self.spacing, self.keep_hold, self.damage]
                .iter()
                .chain(&self.keep)
                .all(|v| v.abs() <= 100);
        if ok { Ok(()) } else { Err(WirError::new(format!("archetype `{}` is out of range", self.name))) }
    }

    /// Bounds of fear over valid states.
    pub fn fear_range(&self) -> (i64, i64) {
        (0, self.f1 * (self.knee1 - 1).max(0) + self.f2 * (self.knee2 - 1).max(0) + self.fe)
    }

    fn inertia(&self, option: usize, current: usize) -> i64 {
        match (option == HOLD, current == HOLD) {
            (false, false) => self.keep[turn(option, current)],
            (true, true) => self.keep_hold,
            _ => 0,
        }
    }

    /// Bounds of an option's score without the blocked term, over every option.
    pub fn score_range(&self) -> (i64, i64) {
        let exposure = (-self.hide, self.peek);
        let k = (*self.keep.iter().min().unwrap(), *self.keep.iter().max().unwrap());
        let heading = (
            -self.approach - self.cover + exposure.0.min(0) - ALLY_CAP * self.spacing + k.0.min(0),
            self.approach + self.cover + exposure.1.max(0) + k.1.max(0),
        );
        let hold = (
            -self.cover + exposure.0.min(0) + self.keep_hold.min(0),
            self.cover + exposure.1.max(0) + self.plant + self.keep_hold.max(0),
        );
        (heading.0.min(hold.0), heading.1.max(hold.1))
    }

    /// A blocked heading loses this: more than every score can span, so it loses to
    /// holding still, which is never blocked.
    pub fn blocked_penalty(&self) -> i64 {
        let (lo, hi) = self.score_range();
        hi - lo + 1
    }
}

/// The senses as plain code. `get(channel, dy, dx)` reads dp, dc, exp, wall, ally
/// around the cell. Order: gp0..7, gc0..7, e0..7, n0..7, b0..7, dp, e.
pub fn sense_cell(get: impl Fn(usize, i64, i64) -> i64) -> Vec<i64> {
    let mut s = Vec::with_capacity(SENSES);
    // Neighbours are graded against the best neighbour, not the cell itself: in the
    // distance-to-player field an enemy's own cell carries its ally cost.
    let best = |ch: usize| DIRS.iter().map(|&(dy, dx)| get(ch, dy, dx)).min().unwrap();
    let (mp, mc) = (best(0), best(1));
    for (dy, dx) in DIRS {
        s.push(grade(get(0, dy, dx), mp));
    }
    for (dy, dx) in DIRS {
        s.push(grade(get(1, dy, dx), mc));
    }
    for (dy, dx) in DIRS {
        s.push(get(2, dy, dx));
    }
    for d in 0..8 {
        let mut n = 0;
        for dy in -2..=2 {
            for dx in -2..=2 {
                if sector(dy, dx) == Some(d) {
                    n += get(4, dy, dx);
                }
            }
        }
        s.push(n.min(ALLY_CAP));
    }
    for (dy, dx) in DIRS {
        let mut b = get(3, dy, dx) == 1 || get(4, dy, dx) == 1 || get(0, dy, dx) <= 0;
        if dy != 0 && dx != 0 {
            b = b || get(3, dy, 0) == 1 || get(3, 0, dx) == 1;
        }
        s.push(b as i64);
    }
    s.push(mp);
    s.push(get(2, 0, 0));
    s.push(grade(get(1, 0, 0), mc));
    s
}

fn tap(ch: &str, dy: i64, dx: i64) -> String {
    if dy == 0 && dx == 0 { ch.to_string() } else { format!("{ch}[{dy}, {dx}]") }
}

/// The sense program: a grid program whose outputs at a cell are an enemy's senses there.
pub fn sense_program() -> String {
    let mut src = format!(
        "wir 0.2
# Senses of a tactical enemy (generated by `wir tactics`). A grid program, read at the
# enemy's cell. dp is the distance to the player, dc the distance to cover (8-move
# fields, a straight step costs 2), exp 1 where the player can see the cell, ally 1
# where another enemy stands. Headings 0..7 are E NE N NW W SW S SE.

input dp: grid in [0, {CEILING}] pad {CEILING}
input dc: grid in [0, {CEILING}] pad {CEILING}
input exp: grid in [0, 1] pad 0
input wall: grid in [0, 1] pad 1
input ally: grid in [0, 1] pad 0

# Allies cost extra to pass in dp, so an enemy's own cell reads high there: neighbours
# are graded against the best neighbour instead. gp, gc: 1 within 1 of the best step,
# 0 within 3 (partial progress), -1 farther (sideways or back).
"
    );
    for (field, best) in [("dp", "mp"), ("dc", "mc")] {
        let t: Vec<String> = DIRS.iter().map(|&(dy, dx)| tap(field, dy, dx)).collect();
        src += &format!(
            "let {best} = min(min(min({}, {}), min({}, {})), min(min({}, {}), min({}, {})))\n",
            t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]
        );
    }
    for (field, name, best) in [("dp", "gp", "mp"), ("dc", "gc", "mc")] {
        for (d, (dy, dx)) in DIRS.iter().enumerate() {
            let t = tap(field, *dy, *dx);
            src += &format!("output {name}{d} = le({t} - {best}, 1) - ge({t} - {best}, 4)\n");
        }
    }
    src += "\n# e: the player can see the next cell\n";
    for (d, (dy, dx)) in DIRS.iter().enumerate() {
        src += &format!("output e{d} = {}\n", tap("exp", *dy, *dx));
    }
    src += &format!("\n# n: allies in each heading's sector of the 5 x 5 window (3 cells each), capped at {ALLY_CAP}\n");
    for d in 0..8 {
        let cells: Vec<String> = (-2..=2i64)
            .flat_map(|dy| (-2..=2i64).map(move |dx| (dy, dx)))
            .filter(|&(dy, dx)| sector(dy, dx) == Some(d))
            .map(|(dy, dx)| tap("ally", dy, dx))
            .collect();
        src += &format!("output n{d} = {ALLY_CAP} - relu({ALLY_CAP} - ({}))\n", cells.join(" + "));
    }
    src += "\n# b: blocked by a wall, an ally or the player (dp 0); a diagonal also by a wall beside it\n";
    for (d, (dy, dx)) in DIRS.iter().enumerate() {
        let mut parts = vec![tap("wall", *dy, *dx), tap("ally", *dy, *dx), format!("le({}, 0)", tap("dp", *dy, *dx))];
        if *dy != 0 && *dx != 0 {
            parts.push(tap("wall", *dy, 0));
            parts.push(tap("wall", 0, *dx));
        }
        src += &format!("output b{d} = or({})\n", parts.join(", "));
    }
    src += "\n# the enemy's own cell: distance after the best step, seen, how its cover compares\noutput dps = mp\noutput es = exp\noutput gcs = le(dc - mc, 1) - ge(dc - mc, 4)\n";
    src
}

/// What a brain reads: the current option (0..8, HOLD = 8), its own state and the senses.
#[derive(Debug, Clone, PartialEq)]
pub struct BrainIn {
    pub option: usize,
    pub hp: i64,
    pub ammo: i64,
    pub cool: i64,
    pub reload: i64,
    pub senses: Vec<i64>,
}

/// The response curves at one state, as plain code.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Drives {
    pub fear: i64,
    pub approach: i64,
    pub cover: i64,
    pub peek: i64,
    pub plant: i64,
}

pub fn drives(a: &Archetype, x: &BrainIn) -> Drives {
    let (dp, e) = (x.senses[40], x.senses[41]);
    let empty = (x.ammo <= 0) as i64;
    let fear = a.f1 * (a.knee1 - x.hp).max(0) + a.f2 * (a.knee2 - x.hp).max(0) + a.fe * empty;
    let ready = x.ammo >= 1;
    Drives {
        fear,
        approach: (dp - a.range - fear).clamp(-a.approach, a.approach),
        cover: (fear - a.cover_from).clamp(0, a.cover),
        peek: (a.peek - fear).clamp(-a.hide, a.peek),
        plant: if e == 1 && ready && dp <= a.fire_range && dp >= a.min_range { a.plant } else { 0 },
    }
}

/// Scores of the 9 options (blocked term included), as plain code.
pub fn scores(a: &Archetype, x: &BrainIn) -> [i64; 9] {
    let dv = drives(a, x);
    let s = &x.senses;
    let big = a.blocked_penalty();
    let mut out = [0; 9];
    for d in 0..8 {
        out[d] = dv.approach * s[d] + dv.cover * s[8 + d] + if s[16 + d] == 1 { dv.peek } else { 0 } - a.spacing * s[24 + d]
            + a.inertia(d, x.option)
            - big * s[32 + d];
    }
    out[HOLD] = dv.cover * s[42] + if s[41] == 1 { dv.peek } else { 0 } + dv.plant + a.inertia(HOLD, x.option);
    out
}

/// The brain as plain code: (option, fire, reload).
pub fn brain_rule(a: &Archetype, x: &BrainIn) -> (usize, bool, bool) {
    let sc = scores(a, x);
    // Ties go to the lower option index.
    let option = (0..9).max_by_key(|&d| 9 * sc[d] + (8 - d as i64)).unwrap();
    let (dp, e) = (x.senses[40], x.senses[41]);
    let fire = e == 1 && x.ammo >= 1 && x.cool <= 0 && dp <= a.fire_range;
    let reload = x.reload <= 0 && (x.ammo <= 0 || (e == 0 && x.ammo <= a.reload_at));
    (option, fire, reload)
}

/// The brain program: response curves, scores for 8 headings and holding, the best
/// option as one-hot o2_0..o2_8, and the fire and reload bits.
pub fn brain_program(a: &Archetype) -> Result<String> {
    a.validate()?;
    let big = a.blocked_penalty();
    let lo = a.score_range().0 - big;
    let list = |p: &str, n: usize| (0..n).map(|k| format!("{p}{k}")).collect::<Vec<_>>().join(", ");
    let term = |k: i64, x: &str| match k {
        0 => String::new(),
        1 => format!(" + {x}"),
        -1 => format!(" - {x}"),
        k if k < 0 => format!(" - {}*{x}", -k),
        k => format!(" + {k}*{x}"),
    };
    let fear_terms = [
        if a.f1 > 0 && a.knee1 > 1 { term(a.f1, &format!("relu({} - hp)", a.knee1)) } else { String::new() },
        if a.f2 > 0 && a.knee2 > 1 { term(a.f2, &format!("relu({} - hp)", a.knee2)) } else { String::new() },
        term(a.fe, "empty"),
    ]
    .concat();
    let fear = if fear_terms.is_empty() { "0".to_string() } else { fear_terms.trim_start_matches(" + ").to_string() };
    let mut src = format!(
        "wir 0.2
# Brain of a {name} (generated by `wir tactics`): response curves, then a score for each
# heading (0..7: E NE N NW W SW S SE) and for holding still (8), the best one-hot, and
# whether to fire or reload. o is the current option (one-hot); the senses come from the
# sense program at the enemy's cell. Distances are field units (2 per straight cell).
# Reloading drops the magazine, so ammo is 0 until a reload is done.

input {o}: R1 in [0, 1]
input hp: R1 in [1, {HP}]
input ammo: R1 in [0, {AMMO}]
input cool: R1 in [0, {COOL}]
input reload: R1 in [0, {RELOAD}]
input {gp}: R1 in [-1, 1]
input {gc}: R1 in [-1, 1]
input {e}: R1 in [0, 1]
input {n}: R1 in [0, {ALLY_CAP}]
input {b}: R1 in [0, 1]
input dp: R1 in [0, {CEILING}]
input es: R1 in [0, 1]
input gcs: R1 in [-1, 1]

# --- response curves ---
let empty = relu(1 - ammo)
# fear rises below {knee1} hp, faster below {knee2}, and a lot while empty
let fear = {fear}
# approach: + to close in, - to back off; fear pushes the preferred range out
let approach = clip(dp - {range} - fear, -{ap}, {ap})
# cover: the pull toward cells the player cannot see
let cover = clip(fear - {cf}, 0, {cv})
# exposure: worth seeking when calm, avoided when afraid (empty is always -{hide})
let peek = clip({peek} - fear, -{hide}, {peek})
# plant: hold still and fire when seen, loaded and in range but not too close
let plant = {plant}*and(es, ge(ammo, 1), le(dp, {fr}), ge(dp, {minr}))

# --- scores. Gradients gp, gc are -1, 0 or 1 and e is 0 or 1, so drive * gradient is
# an exact product. The rest of each score (spacing, keeping the current option, a
# blocked heading losing {big}) is packed into one unit: relu(x - lo) + lo is x.
",
        name = a.name,
        o = list("o", 9),
        gp = list("gp", 8),
        gc = list("gc", 8),
        e = list("e", 8),
        n = list("n", 8),
        b = list("b", 8),
        knee1 = a.knee1,
        knee2 = a.knee2,
        range = a.range,
        ap = a.approach,
        cf = a.cover_from,
        cv = a.cover,
        peek = a.peek,
        hide = a.hide,
        fr = a.fire_range,
        minr = a.min_range,
        plant = a.plant,
    );
    let signed = |v: i64| if v < 0 { format!("- {}", -v) } else { format!("+ {v}") };
    for d in 0..9 {
        let mut l = String::new();
        if d < 8 {
            l += &term(-a.spacing, &format!("n{d}"));
        }
        for k in 0..9 {
            l += &term(a.inertia(d, k), &format!("o{k}"));
        }
        if d < 8 {
            l += &term(-big, &format!("b{d}"));
        }
        let l = match l.strip_prefix(" - ") {
            Some(rest) => format!("-{rest}"),
            None => l.trim_start_matches(" + ").to_string(),
        };
        let l = if l.is_empty() { "0".to_string() } else { l };
        // The packed part never goes below this.
        let floor = if d < 8 { -ALLY_CAP * a.spacing + a.keep.iter().min().unwrap().min(&0) - big } else { a.keep_hold.min(0) };
        if l == "0" {
            src += &format!("let k{d} = 0\n");
        } else if floor == 0 {
            src += &format!("let k{d} = relu({l})\n");
        } else {
            src += &format!("let k{d} = relu({l} {}) {}\n", signed(-floor), signed(floor));
        }
    }
    src += "\n";
    for d in 0..9 {
        let s = if d < 8 { format!("approach*gp{d} + cover*gc{d} + peek*e{d} + k{d}") } else { "cover*gcs + peek*es + plant + k8".to_string() };
        src += &format!("let s{d} = {s}\n");
    }
    src += &format!("\n# key: the score as one unit (scores are at least {lo}), times 9 plus a tie-break\n");
    for d in 0..9 {
        src += &format!("let key{d} = 9*relu(s{d} {}) {}\n", signed(-lo), signed(9 * lo + 8 - d as i64));
    }
    src += "\n# the best option beats every other key (keys are never equal)\n";
    for d in 0..9 {
        for e in d + 1..9 {
            src += &format!("let w{d}{e} = gt(key{d}, key{e})\n");
        }
    }
    for d in 0..9 {
        let parts: Vec<String> = (0..9).filter(|&e| e != d).map(|e| if e > d { format!("w{d}{e}") } else { format!("not(w{e}{d})") }).collect();
        src += &format!("output o2_{d} = and({})\n", parts.join(", "));
    }
    src += &format!(
        "\n# fire when the player is visible and in range; reload when empty, or hidden and low\noutput fire = and(es, ge(ammo, 1), le(cool, 0), le(dp, {}))\noutput reload2 = and(le(reload, 0), or(le(ammo, 0), and(not(es), le(ammo, {}))))\n",
        a.fire_range, a.reload_at
    );
    Ok(src)
}

/// One archetype's compiled brain plus the shared sense and path networks.
pub struct TacticsNets {
    pub path: deep::StencilWeights,
    pub sense: deep::StencilWeights,
    pub brains: Vec<deep::DeepWeights>,
    path_net: deep::SparseNet,
    sense_net: deep::SparseNet,
    brain_nets: Vec<deep::SparseNet>,
}

impl TacticsNets {
    pub fn compile(archetypes: &[Archetype]) -> Result<Self> {
        let pc = deep::compile(&games::path_program(&path_rule())?, 1e-6)?;
        let path = deep::StencilWeights::from_compiled(&pc).ok_or_else(|| WirError::new("path is not a grid program"))?;
        let sc = deep::compile(&sense_program(), 1e-6)?;
        let sense = deep::StencilWeights::from_compiled(&sc).ok_or_else(|| WirError::new("sense is not a grid program"))?;
        let brains: Vec<deep::DeepWeights> =
            archetypes.iter().map(|a| Ok(deep::compile(&brain_program(a)?, 1e-6)?.weights)).collect::<Result<_>>()?;
        Ok(Self {
            path_net: deep::SparseNet::new(&path.net),
            sense_net: deep::SparseNet::new(&sense.net),
            brain_nets: brains.iter().map(deep::SparseNet::new).collect(),
            path,
            sense,
            brains,
        })
    }

    pub fn senses(&mut self, get: impl Fn(usize, i64, i64) -> i64) -> Vec<i64> {
        let x: Vec<f64> = self.sense.taps.iter().map(|&(ch, dy, dx)| get(ch, dy, dx) as f64).collect();
        self.sense_net.eval(&x).iter().map(|v| v.round() as i64).collect()
    }

    /// Raw brain outputs: o2_0..o2_8, fire, reload2.
    pub fn brain_outputs(&mut self, archetype: usize, x: &BrainIn) -> Vec<f64> {
        let mut v: Vec<f64> = (0..9).map(|k| (k == x.option) as i64 as f64).collect();
        v.extend([x.hp, x.ammo, x.cool, x.reload].map(|k| k as f64));
        v.extend(x.senses.iter().map(|&k| k as f64));
        self.brain_nets[archetype].eval(&v).to_vec()
    }

    /// The decision from the network, or None when its outputs are not clean bits.
    pub fn decide(&mut self, archetype: usize, x: &BrainIn) -> Option<(usize, bool, bool)> {
        let out = self.brain_outputs(archetype, x);
        let bit = |v: f64| if v.abs() < 1e-6 { Some(false) } else if (v - 1.0).abs() < 1e-6 { Some(true) } else { None };
        let bits: Option<Vec<bool>> = out.iter().map(|&v| bit(v)).collect();
        let bits = bits?;
        let hot: Vec<usize> = (0..9).filter(|&k| bits[k]).collect();
        (hot.len() == 1).then(|| (hot[0], bits[9], bits[10]))
    }

    fn path_step(&mut self, s: &PathGrid) -> Vec<i64> {
        let out = self.path.apply_with(&mut self.path_net, &s.inputs(&path_rule()), s.h, s.w);
        out[0].iter().map(|v| v.round() as i64).collect()
    }
}

/// One enemy.
#[derive(Debug, Clone, PartialEq)]
pub struct Enemy {
    pub archetype: usize,
    pub r: i64,
    pub c: i64,
    pub option: usize,
    pub hp: i64,
    pub ammo: i64,
    pub cool: i64,
    pub reload: i64,
    /// Ticks until respawn while dead (hp 0).
    pub dead: i64,
}

/// What happened in one tick.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TickStats {
    pub decisions: u64,
    /// Decisions that picked a blocked heading.
    pub blocked_picks: u64,
    pub moves: u64,
    /// Moves back into the cell held two ticks ago (a step and its reverse).
    pub back_forth: u64,
    pub shots: u64,
    pub hits: u64,
    pub reloads: u64,
    pub kills: u64,
    /// Decisions by an exposed enemy that is hurt (4 hp or less) or empty, and those
    /// that stepped to a cell nearer cover (lower dc), per archetype (up to 4).
    pub afraid: [u64; 4],
    pub to_cover: [u64; 4],
}

/// An arena: walls, the player, the enemies and the shared fields.
#[derive(Debug, Clone, PartialEq)]
pub struct Arena {
    pub h: usize,
    pub w: usize,
    pub wall: Vec<i64>,
    pub player: (i64, i64),
    pub target: (i64, i64),
    /// Ticks the player still waits at its target.
    pub wait: u64,
    pub enemies: Vec<Enemy>,
    /// Each enemy's cell two ticks ago and one tick ago.
    pub trail: Vec<[(i64, i64); 2]>,
    pub dp: Vec<i64>,
    pub dc: Vec<i64>,
    pub exp: Vec<i64>,
    pub tick: u64,
    pub rng: u64,
    pub player_hits_taken: i64,
    /// Cost of leaving an ally's cell in the dp field (1 to ALLY_COST; 1 turns it off).
    pub ally_cost: i64,
}

/// How the arena runs: the rule as plain code, or compiled networks.
pub enum Runner<'a> {
    Rule,
    Nets(&'a mut TacticsNets),
}

/// Field iterations per tick.
pub const FIELD_ITERS: usize = 3;
/// Player: shoots every this many ticks, reaches this far (field units), hits 1 in 2.
pub const PLAYER_COOL: u64 = 2;
pub const PLAYER_REACH: i64 = 20;
pub const PLAYER_DAMAGE: i64 = 3;
/// The player waits this many ticks at each target before picking the next.
pub const PLAYER_WAIT: u64 = 8;
/// Hidden enemies heal 1 hp every this many ticks.
pub const REGEN: u64 = 6;
pub const RESPAWN: i64 = 20;

impl Arena {
    pub fn at(&self, r: i64, c: i64) -> Option<usize> {
        (r >= 0 && c >= 0 && r < self.h as i64 && c < self.w as i64).then(|| (r * self.w as i64 + c) as usize)
    }

    pub fn open(&self, r: i64, c: i64) -> bool {
        self.at(r, c).is_some_and(|i| self.wall[i] == 0)
    }

    /// Line of sight between two cell centres: sample the segment at quarter cells.
    pub fn sight(&self, a: (i64, i64), b: (i64, i64)) -> bool {
        let n = 4 * (a.0 - b.0).abs().max((a.1 - b.1).abs());
        for k in 1..n {
            // Cell of the point a + (b - a) k / n, rounding from cell centres.
            let y = (2 * a.0 * n + 2 * (b.0 - a.0) * k + n).div_euclid(2 * n);
            let x = (2 * a.1 * n + 2 * (b.1 - a.1) * k + n).div_euclid(2 * n);
            if !self.open(y, x) {
                return false;
            }
        }
        true
    }

    /// Host: which cells the player sees.
    pub fn update_exposure(&mut self) {
        let mut exp = vec![0; self.h * self.w];
        for r in 0..self.h as i64 {
            for c in 0..self.w as i64 {
                if self.open(r, c) && self.sight(self.player, (r, c)) {
                    exp[(r * self.w as i64 + c) as usize] = 1;
                }
            }
        }
        self.exp = exp;
    }

    fn field_step(&self, d: &[i64], goal: Vec<i64>, cost: &[i64], runner: &mut Runner) -> Vec<i64> {
        let g = path_rule();
        let mut s = PathGrid::new(&g, self.h, self.w);
        s.fields[0] = d.to_vec();
        s.fields[1] = self.wall.clone();
        s.fields[2] = goal;
        s.fields[3] = cost.to_vec();
        match runner {
            Runner::Rule => games::path_step(&g, &s),
            Runner::Nets(n) => n.path_step(&s),
        }
    }

    /// Run the distance-to-player and distance-to-cover fields `FIELD_ITERS` steps.
    pub fn update_fields(&mut self, runner: &mut Runner) {
        let n = self.h * self.w;
        let mut goal_p = vec![0; n];
        goal_p[self.at(self.player.0, self.player.1).unwrap()] = 1;
        let goal_c: Vec<i64> = (0..n).map(|i| (self.wall[i] == 0 && self.exp[i] == 0) as i64).collect();
        let ally = self.ally_grid();
        let cost: Vec<i64> = ally.iter().map(|&a| if a == 1 { self.ally_cost } else { 1 }).collect();
        let flat = vec![1; n];
        for _ in 0..FIELD_ITERS {
            self.dp = self.field_step(&self.dp, goal_p.clone(), &cost, runner);
            self.dc = self.field_step(&self.dc, goal_c.clone(), &flat, runner);
        }
    }

    fn ally_grid(&self) -> Vec<i64> {
        let mut g = vec![0; self.h * self.w];
        for e in self.enemies.iter().filter(|e| e.dead == 0) {
            g[self.at(e.r, e.c).unwrap()] = 1;
        }
        g
    }

    /// The reader for the sense program at (r, c): dp, dc, exp, wall, ally (the enemy
    /// itself is not its own ally: its cell reads 0).
    fn reader<'a>(&'a self, ally: &'a [i64], r: i64, c: i64) -> impl Fn(usize, i64, i64) -> i64 + 'a {
        move |ch, dy, dx| match self.at(r + dy, c + dx) {
            None => [CEILING, CEILING, 0, 1, 0][ch],
            Some(i) => match ch {
                0 => self.dp[i],
                1 => self.dc[i],
                2 => self.exp[i],
                3 => self.wall[i],
                _ => if dy == 0 && dx == 0 { 0 } else { ally[i] },
            },
        }
    }

    /// The brain input of a living enemy (senses from the rule or the sense network).
    pub fn brain_in(&self, k: usize, ally: &[i64], runner: &mut Runner) -> BrainIn {
        let e = &self.enemies[k];
        let get = self.reader(ally, e.r, e.c);
        let senses = match runner {
            Runner::Rule => sense_cell(get),
            Runner::Nets(n) => n.senses(get),
        };
        BrainIn { option: e.option, hp: e.hp, ammo: e.ammo, cool: e.cool, reload: e.reload, senses }
    }

    /// Advance one tick: the host updates sight and fields, every living enemy decides
    /// from the same snapshot, then enemies act and move in order, then the player.
    pub fn step(&mut self, archetypes: &[Archetype], runner: &mut Runner) -> TickStats {
        let mut st = TickStats::default();
        self.update_exposure();
        self.update_fields(runner);
        let ally = self.ally_grid();
        let mut plans = Vec::with_capacity(self.enemies.len());
        for k in 0..self.enemies.len() {
            if self.enemies[k].dead > 0 {
                plans.push(None);
                continue;
            }
            let x = self.brain_in(k, &ally, runner);
            let a = self.enemies[k].archetype;
            let plan = match runner {
                Runner::Rule => brain_rule(&archetypes[a], &x),
                Runner::Nets(n) => n.decide(a, &x).unwrap_or((usize::MAX, false, false)),
            };
            st.decisions += 1;
            let afraid = x.senses[41] == 1 && (x.hp <= 4 || x.ammo == 0);
            if a < 4 && afraid {
                st.afraid[a] += 1;
                st.to_cover[a] += (plan.0 < 8 && x.senses[8 + plan.0] == 1) as u64;
            }
            if plan.0 < 8 && x.senses[32 + plan.0] == 1 {
                st.blocked_picks += 1;
            }
            plans.push(Some(plan));
        }
        for k in 0..self.enemies.len() {
            let Some((option, fire, reload)) = plans[k] else {
                let e = &mut self.enemies[k];
                e.dead -= 1;
                if e.dead == 0 {
                    self.respawn(k);
                }
                continue;
            };
            let dmg = archetypes[self.enemies[k].archetype].damage;
            let roll = xorshift(&mut self.rng, 2) == 0;
            let e = &mut self.enemies[k];
            e.option = option.min(HOLD);
            e.cool = (e.cool - 1).max(0);
            if e.reload > 0 {
                e.reload -= 1;
                if e.reload == 0 {
                    e.ammo = AMMO;
                }
            }
            if fire {
                e.ammo -= 1;
                e.cool = COOL;
                st.shots += 1;
                if roll {
                    st.hits += 1;
                    self.player_hits_taken += dmg;
                }
            } else if reload {
                // Reloading drops the magazine: an enemy is empty until it is done.
                e.ammo = 0;
                e.reload = RELOAD;
                st.reloads += 1;
            }
            let (r, c) = (self.enemies[k].r, self.enemies[k].c);
            if option < 8 {
                let (dy, dx) = DIRS[option];
                let (y, x) = (r + dy, c + dx);
                let corner = dy != 0 && dx != 0 && !(self.open(y, c) && self.open(r, x));
                let taken = (y, x) == self.player || self.enemies.iter().any(|o| o.dead == 0 && (o.r, o.c) == (y, x));
                if self.open(y, x) && !corner && !taken {
                    if self.trail[k][0] == (y, x) && self.trail[k][1] == (r, c) {
                        st.back_forth += 1;
                    }
                    st.moves += 1;
                    self.enemies[k].r = y;
                    self.enemies[k].c = x;
                }
            }
            let e = &self.enemies[k];
            self.trail[k] = [self.trail[k][1], (e.r, e.c)];
            if self.tick % REGEN == 0 && e.hp < HP && self.exp[self.at(e.r, e.c).unwrap()] == 0 {
                self.enemies[k].hp += 1;
            }
        }
        st.kills = self.player_turn();
        self.tick += 1;
        st
    }

    fn respawn(&mut self, k: usize) {
        // A random open cell the player cannot see and nobody stands on.
        for _ in 0..10_000 {
            let (r, c) = (xorshift(&mut self.rng, self.h as u64) as i64, xorshift(&mut self.rng, self.w as u64) as i64);
            let i = self.at(r, c).unwrap();
            if self.wall[i] == 0 && self.exp[i] == 0 && (r, c) != self.player && !self.enemies.iter().any(|o| o.dead == 0 && (o.r, o.c) == (r, c)) {
                let e = &mut self.enemies[k];
                *e = Enemy { archetype: e.archetype, r, c, option: HOLD, hp: HP, ammo: AMMO, cool: 0, reload: 0, dead: 0 };
                self.trail[k] = [(r, c); 2];
                return;
            }
        }
        self.enemies[k].dead = 1;
    }

    /// The scripted player: walks toward a random target cell (re-picked on arrival),
    /// and every few ticks shoots the nearest enemy it can see. Returns kills.
    fn player_turn(&mut self) -> u64 {
        if self.player == self.target && self.wait > 0 {
            self.wait -= 1;
        } else if self.player == self.target {
            self.wait = PLAYER_WAIT;
            loop {
                let (r, c) = (xorshift(&mut self.rng, self.h as u64) as i64, xorshift(&mut self.rng, self.w as u64) as i64);
                if self.open(r, c) {
                    self.target = (r, c);
                    break;
                }
            }
        }
        // Step to the neighbour closest (BFS, 8 moves) to the target, if it is free.
        let dist = self.bfs(self.target);
        let here = dist[self.at(self.player.0, self.player.1).unwrap()];
        let mut best = (here, self.player);
        for (dy, dx) in DIRS {
            let (y, x) = (self.player.0 + dy, self.player.1 + dx);
            let corner = dy != 0 && dx != 0 && !(self.open(y, self.player.1) && self.open(self.player.0, x));
            if !self.open(y, x) || corner || self.enemies.iter().any(|o| o.dead == 0 && (o.r, o.c) == (y, x)) {
                continue;
            }
            let d = dist[self.at(y, x).unwrap()];
            if d < best.0 {
                best = (d, (y, x));
            }
        }
        self.player = best.1;
        if best.0 == here {
            self.target = self.player;
        }
        let mut kills = 0;
        if self.tick % PLAYER_COOL == 0 {
            let mut near: Option<(i64, usize)> = None;
            for (k, e) in self.enemies.iter().enumerate() {
                let (dy, dx) = ((e.r - self.player.0).abs(), (e.c - self.player.1).abs());
                let d = 2 * dy.max(dx) + dy.min(dx);
                if e.dead == 0 && d <= PLAYER_REACH && self.sight(self.player, (e.r, e.c)) && near.is_none_or(|(b, _)| d < b) {
                    near = Some((d, k));
                }
            }
            if let Some((_, k)) = near {
                if xorshift(&mut self.rng, 2) == 0 {
                    let e = &mut self.enemies[k];
                    e.hp -= PLAYER_DAMAGE;
                    if e.hp <= 0 {
                        e.hp = 0;
                        e.dead = RESPAWN;
                        kills += 1;
                    }
                }
            }
        }
        kills
    }

    fn bfs(&self, from: (i64, i64)) -> Vec<i64> {
        let mut dist = vec![i64::MAX; self.h * self.w];
        let mut q = std::collections::VecDeque::new();
        dist[self.at(from.0, from.1).unwrap()] = 0;
        q.push_back(from);
        while let Some((r, c)) = q.pop_front() {
            let d = dist[self.at(r, c).unwrap()];
            for (dy, dx) in DIRS {
                let (y, x) = (r + dy, c + dx);
                if !self.open(y, x) || (dy != 0 && dx != 0 && !(self.open(y, c) && self.open(r, x))) {
                    continue;
                }
                let j = self.at(y, x).unwrap();
                if dist[j] > d + 1 {
                    dist[j] = d + 1;
                    q.push_back((y, x));
                }
            }
        }
        dist
    }
}

/// A random arena: border walls, pillars and wall segments, the player in the middle,
/// `per_archetype` enemies of each archetype at random open cells away from the player.
pub fn arena(archetypes: usize, h: usize, w: usize, per_archetype: usize, seed: u64) -> Arena {
    let mut rng = seed.max(1);
    let n = h * w;
    let mut wall = vec![0; n];
    for r in 0..h {
        for c in 0..w {
            if r == 0 || c == 0 || r + 1 == h || c + 1 == w {
                wall[r * w + c] = 1;
            }
        }
    }
    let player = (h as i64 / 2, w as i64 / 2);
    for _ in 0..(n / 40).max(2) {
        let (r, c) = (1 + xorshift(&mut rng, h as u64 - 2) as i64, 1 + xorshift(&mut rng, w as u64 - 2) as i64);
        let shape: Vec<(i64, i64)> = match xorshift(&mut rng, 3) {
            0 => vec![(0, 0), (0, 1), (1, 0), (1, 1)],
            1 => (0..1 + xorshift(&mut rng, 5) as i64).map(|k| (0, k)).collect(),
            _ => (0..1 + xorshift(&mut rng, 4) as i64).map(|k| (k, 0)).collect(),
        };
        for (dy, dx) in shape {
            let (y, x) = (r + dy, c + dx);
            if y > 0 && x > 0 && y + 1 < h as i64 && x + 1 < w as i64 && (y - player.0).abs().max((x - player.1).abs()) > 1 {
                wall[(y * w as i64 + x) as usize] = 1;
            }
        }
    }
    let mut a = Arena {
        h,
        w,
        wall,
        player,
        target: player,
        wait: 0,
        enemies: vec![],
        trail: vec![],
        dp: vec![CEILING; n],
        dc: vec![CEILING; n],
        exp: vec![0; n],
        tick: 0,
        rng: 0,
        player_hits_taken: 0,
        ally_cost: ALLY_COST,
    };
    for k in 0..archetypes {
        let mut placed = 0;
        let mut tries = 0;
        while placed < per_archetype && tries < 100_000 {
            tries += 1;
            let (r, c) = (xorshift(&mut rng, h as u64) as i64, xorshift(&mut rng, w as u64) as i64);
            let far = (r - player.0).abs().max((c - player.1).abs()) >= 6;
            if a.open(r, c) && far && !a.enemies.iter().any(|e| (e.r, e.c) == (r, c)) {
                a.enemies.push(Enemy { archetype: k, r, c, option: HOLD, hp: HP, ammo: AMMO, cool: 0, reload: 0, dead: 0 });
                a.trail.push([(r, c); 2]);
                placed += 1;
            }
        }
    }
    a.rng = 1 + xorshift(&mut rng, u64::MAX - 1);
    a
}

/// Behaviour over a run, summed over arenas and ticks.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Behaviour {
    pub ticks: TickStats,
    /// Living enemy-ticks, and those spent where the player can see them: overall, at
    /// 4 hp or less, and while empty (no ammo or reloading).
    pub alive: u64,
    pub exposed: u64,
    pub hurt: u64,
    pub hurt_exposed: u64,
    pub empty: u64,
    pub empty_exposed: u64,
    /// Per archetype: summed distance to the player in cells (Chebyshev) and ticks.
    pub distance: Vec<(f64, u64)>,
    /// Ticks with 2+ enemies within 8 cells of the player, and the summed spread
    /// 1 - |mean unit vector from the player to them| (0 all on one side, 1 surrounded).
    pub engaged: u64,
    pub spread: f64,
    pub player_hits_taken: i64,
}

impl Behaviour {
    fn observe(&mut self, a: &Arena, archetypes: usize) {
        if self.distance.len() < archetypes {
            self.distance.resize(archetypes, (0.0, 0));
        }
        let (mut sy, mut sx, mut near) = (0.0, 0.0, 0);
        for e in a.enemies.iter().filter(|e| e.dead == 0) {
            let seen = a.sight(a.player, (e.r, e.c));
            self.alive += 1;
            self.exposed += seen as u64;
            if e.hp <= 4 {
                self.hurt += 1;
                self.hurt_exposed += seen as u64;
            }
            if e.ammo == 0 || e.reload > 0 {
                self.empty += 1;
                self.empty_exposed += seen as u64;
            }
            let (dy, dx) = ((e.r - a.player.0) as f64, (e.c - a.player.1) as f64);
            let cheb = dy.abs().max(dx.abs());
            self.distance[e.archetype].0 += cheb;
            self.distance[e.archetype].1 += 1;
            if cheb <= 8.0 && cheb > 0.0 {
                let l = (dy * dy + dx * dx).sqrt();
                sy += dy / l;
                sx += dx / l;
                near += 1;
            }
        }
        if near >= 2 {
            self.engaged += 1;
            self.spread += 1.0 - (sy * sy + sx * sx).sqrt() / near as f64;
        }
    }

    fn add(&mut self, t: &TickStats) {
        let s = &mut self.ticks;
        s.decisions += t.decisions;
        s.blocked_picks += t.blocked_picks;
        s.moves += t.moves;
        s.back_forth += t.back_forth;
        s.shots += t.shots;
        s.hits += t.hits;
        s.reloads += t.reloads;
        s.kills += t.kills;
        for k in 0..4 {
            s.afraid[k] += t.afraid[k];
            s.to_cover[k] += t.to_cover[k];
        }
    }
}

/// Outcome of `tactics_check`.
#[derive(Debug, Clone, PartialEq)]
pub struct TacticsCheck {
    pub sense_cases: u64,
    pub sense_mismatches: u64,
    pub brain_cases: u64,
    pub brain_mismatches: u64,
    pub arenas: u64,
    pub steps: u64,
    pub step_mismatches: u64,
    pub behaviour: Behaviour,
    /// Path taps and hidden, sense taps, hidden and conv parameters, then per archetype
    /// brain hidden and parameters.
    pub path_shape: (usize, Vec<usize>, usize),
    pub sense_shape: (usize, Vec<usize>, usize),
    pub brain_shapes: Vec<(Vec<usize>, usize)>,
}

/// Random brain input within the declared domains (all headings blocked 1 time in 20).
pub fn random_brain_in(rng: &mut u64) -> BrainIn {
    let mut s = Vec::with_capacity(SENSES);
    for _ in 0..16 {
        s.push(xorshift(rng, 3) as i64 - 1);
    }
    for _ in 0..8 {
        s.push(xorshift(rng, 2) as i64);
    }
    for _ in 0..8 {
        s.push(if xorshift(rng, 2) == 0 { 0 } else { xorshift(rng, ALLY_CAP as u64 + 1) as i64 });
    }
    let all_blocked = xorshift(rng, 20) == 0;
    for _ in 0..8 {
        s.push((all_blocked || xorshift(rng, 4) == 0) as i64);
    }
    // dp near the interesting thresholds half the time.
    s.push(if xorshift(rng, 2) == 0 { xorshift(rng, 40) as i64 } else { xorshift(rng, CEILING as u64 + 1) as i64 });
    s.push(xorshift(rng, 2) as i64);
    s.push(xorshift(rng, 3) as i64 - 1);
    BrainIn {
        option: xorshift(rng, 9) as usize,
        hp: 1 + xorshift(rng, HP as u64) as i64,
        ammo: xorshift(rng, AMMO as u64 + 1) as i64,
        cool: xorshift(rng, COOL as u64 + 1) as i64,
        reload: if xorshift(rng, 2) == 0 { 0 } else { xorshift(rng, RELOAD as u64 + 1) as i64 },
        senses: s,
    }
}

/// Compile the programs and check them: the sense network against `sense_cell` and
/// each brain against `brain_rule` on `cases` random inputs, then `arenas` random
/// h x w arenas run `steps` ticks with networks and rule in lockstep (every enemy,
/// field and the player must agree every tick). Behaviour is measured on the rule run.
#[allow(clippy::too_many_arguments)]
pub fn tactics_check(archetypes: &[Archetype], cases: u64, arenas: u64, steps: u64, h: usize, w: usize, per_archetype: usize, ally_cost: i64, seed: u64) -> Result<TacticsCheck> {
    let mut nets = TacticsNets::compile(archetypes)?;
    let mut rng = seed.max(1);
    let mut sense_mismatches = 0;
    let spans: [(i64, i64); 5] = [(0, CEILING), (0, CEILING), (0, 1), (0, 1), (0, 1)];
    for _ in 0..cases {
        let mut vals = std::collections::HashMap::new();
        let base = xorshift(&mut rng, CEILING as u64 + 1) as i64;
        for &(ch, dy, dx) in &nets.sense.taps {
            let v = match ch {
                // Fields mostly near each other, so gradients are -1, 0 and 1 alike.
                0 | 1 if xorshift(&mut rng, 4) != 0 => (base + xorshift(&mut rng, 7) as i64 - 3).clamp(0, CEILING),
                2 | 4 => xorshift(&mut rng, 2) as i64,
                3 => (xorshift(&mut rng, 4) == 0) as i64,
                _ => spans[ch].0 + xorshift(&mut rng, (spans[ch].1 - spans[ch].0 + 1) as u64) as i64,
            };
            vals.insert((ch, dy, dx), v);
        }
        let get = |ch: usize, dy: i64, dx: i64| -> i64 {
            *vals.get(&(ch, dy, dx)).unwrap_or_else(|| panic!("the rule reads channel {ch} at [{dy}, {dx}], which the program does not"))
        };
        let want = sense_cell(get);
        sense_mismatches += (nets.senses(get) != want) as u64;
    }
    let mut brain_mismatches = 0;
    for (k, a) in archetypes.iter().enumerate() {
        for _ in 0..cases {
            let x = random_brain_in(&mut rng);
            brain_mismatches += (nets.decide(k, &x) != Some(brain_rule(a, &x))) as u64;
        }
    }
    let mut behaviour = Behaviour::default();
    let (mut total, mut step_mismatches) = (0, 0);
    for _ in 0..arenas {
        let arena_seed = 1 + xorshift(&mut rng, u64::MAX - 1);
        let mut a = arena(archetypes.len(), h, w, per_archetype, arena_seed);
        a.ally_cost = ally_cost;
        let mut b = a.clone();
        for _ in 0..steps {
            let t = a.step(archetypes, &mut Runner::Rule);
            b.step(archetypes, &mut Runner::Nets(&mut nets));
            behaviour.add(&t);
            behaviour.observe(&a, archetypes.len());
            total += 1;
            step_mismatches += (a != b) as u64;
        }
        behaviour.player_hits_taken += a.player_hits_taken;
    }
    Ok(TacticsCheck {
        sense_cases: cases,
        sense_mismatches,
        brain_cases: cases * archetypes.len() as u64,
        brain_mismatches,
        arenas,
        steps: total,
        step_mismatches,
        behaviour,
        path_shape: (nets.path.taps.len(), nets.path.net.hidden(), nets.path.param_count()),
        sense_shape: (nets.sense.taps.len(), nets.sense.net.hidden(), nets.sense.param_count()),
        brain_shapes: nets.brains.iter().map(|b| (b.hidden(), b.param_count())).collect(),
    })
}

#[allow(clippy::too_many_arguments)]
/// Behaviour of the rule alone (no networks), for comparing archetype settings quickly.
pub fn tactics_behaviour(archetypes: &[Archetype], arenas: u64, steps: u64, h: usize, w: usize, per_archetype: usize, ally_cost: i64, seed: u64) -> Behaviour {
    let mut rng = seed.max(1);
    let mut behaviour = Behaviour::default();
    for _ in 0..arenas {
        let arena_seed = 1 + xorshift(&mut rng, u64::MAX - 1);
        let mut a = arena(archetypes.len(), h, w, per_archetype, arena_seed);
        a.ally_cost = ally_cost;
        for _ in 0..steps {
            let t = a.step(archetypes, &mut Runner::Rule);
            behaviour.add(&t);
            behaviour.observe(&a, archetypes.len());
        }
        behaviour.player_hits_taken += a.player_hits_taken;
    }
    behaviour
}

fn sparse_layers(w: &deep::DeepWeights) -> serde_json::Value {
    let layers: Vec<serde_json::Value> = w
        .layers
        .iter()
        .map(|l| {
            let rows: Vec<Vec<f64>> = (0..l.w.rows)
                .map(|r| l.w.row(r).iter().enumerate().filter(|(_, v)| **v != 0.0).flat_map(|(j, v)| [j as f64, *v]).collect())
                .collect();
            serde_json::json!({ "rows": rows, "b": l.b })
        })
        .collect();
    serde_json::Value::Array(layers)
}

fn stencil_json(s: &deep::StencilWeights, src: &str) -> serde_json::Value {
    serde_json::json!({
        "channels": s.channels,
        "pads": s.pads,
        "taps": s.taps,
        "outputs": s.net.outputs,
        "hidden": s.net.hidden(),
        "params": s.param_count(),
        "layers": sparse_layers(&s.net),
        "source": src,
    })
}

/// Everything a page needs to run the arena from compiled weights: the path and sense
/// stencils, then per named variant each archetype's brain (sparse layers, constants,
/// source), and the host constants.
pub fn export_json(variants: &[(&str, Vec<Archetype>)]) -> Result<String> {
    let first = &variants.first().ok_or_else(|| WirError::new("no variants to export"))?.1;
    let nets = TacticsNets::compile(first)?;
    let mut sets = serde_json::Map::new();
    for (name, archetypes) in variants {
        let compiled = TacticsNets::compile(archetypes)?;
        let brains: Vec<serde_json::Value> = archetypes
            .iter()
            .zip(&compiled.brains)
            .map(|(a, b)| {
                Ok(serde_json::json!({
                    "name": a.name,
                    "range": a.range, "fire_range": a.fire_range, "min_range": a.min_range,
                    "knee1": a.knee1, "f1": a.f1, "knee2": a.knee2, "f2": a.f2, "fe": a.fe,
                    "approach": a.approach, "cover_from": a.cover_from, "cover": a.cover,
                    "peek": a.peek, "hide": a.hide, "plant": a.plant, "spacing": a.spacing,
                    "keep": a.keep, "keep_hold": a.keep_hold, "reload_at": a.reload_at,
                    "damage": a.damage, "big": a.blocked_penalty(),
                    "inputs": b.inputs, "outputs": b.outputs,
                    "hidden": b.hidden(), "params": b.param_count(),
                    "layers": sparse_layers(b),
                    "source": brain_program(a)?,
                }))
            })
            .collect::<Result<_>>()?;
        sets.insert(name.to_string(), serde_json::Value::Array(brains));
    }
    let v = serde_json::json!({
        "format": "wir-tactics-demo",
        "consts": {
            "ceiling": CEILING, "ally_cap": ALLY_CAP, "ally_cost": ALLY_COST, "hp": HP, "ammo": AMMO, "cool": COOL,
            "reload": RELOAD, "field_iters": FIELD_ITERS, "regen": REGEN, "respawn": RESPAWN,
            "player_cool": PLAYER_COOL, "player_reach": PLAYER_REACH, "player_damage": PLAYER_DAMAGE,
        },
        "path": stencil_json(&nets.path, &games::path_program(&path_rule())?),
        "sense": stencil_json(&nets.sense, &sense_program()),
        "brains": sets,
    });
    Ok(serde_json::to_string(&v).unwrap())
}
