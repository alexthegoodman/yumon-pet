//! Swarming as two programs per species, with a reference simulator.
//!
//! The world is a grid (water, walls, a distance field) with agents moving through it
//! at sub-cell positions. Each species has a **sense** program, a grid program that
//! turns the cells around an agent into 40 numbers (neighbours by heading, crowding by
//! direction, the other species by direction, blocked directions, the goal gradient),
//! and a **brain** program, a plain 0.2 program that scores the 8 headings and picks
//! the best one (utility AI: alignment, cohesion with a crowding limit, the other
//! species, the goal, a turn cost and a noise kick). The host only counts agents into
//! cells, reads the senses at each agent's cell and moves agents: there is no
//! neighbour search. `swarm_check` runs the compiled networks against the rule as
//! plain code, agent for agent.

use crate::deep;
use crate::error::{Result, WirError};

/// Heading k points along DIRS[k] = (dy, dx): E, NE, N, NW, W, SW, S, SE.
pub const DIRS: [(i64, i64); 8] = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)];
pub const DIR_NAMES: [&str; 8] = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"];

/// Sub-cell units per cell: agent positions are integers in these units.
pub const CELL: i64 = 16;
/// Water units in a full cell (the `wir fluid` default).
pub const WATER: i64 = 16;
/// Senses reach this many cells out (a 5 x 5 window).
pub const RADIUS: i64 = 2;
/// Per-cell caps on the counted channels: agents of one heading, agents of the other species.
pub const HEADING_CAP: i64 = 3;
pub const OTHER_CAP: i64 = 6;
/// Bodies take room: at most this many agents of one species move into a cell.
pub const ROOM: i64 = 3;
/// Caps on the senses: neighbours with a heading, crowding in a direction, the other species.
pub const ALIGN_CAP: i64 = 12;
pub const CROWD_CAP: i64 = 16;
pub const OTHER_SENSE_CAP: i64 = 12;
/// Distance field range (the `wir path` default ceiling).
pub const CEILING: i64 = 64;

/// Steps between two headings, 0 to 4.
pub fn turn(a: usize, b: usize) -> usize {
    let t = (a + 8 - b) % 8;
    t.min(8 - t)
}

/// The sector of the 5 x 5 window an offset lies in, None at the centre. Sectors are a
/// pinwheel of 3 cells each: the 2 cells along a heading plus the knight-move offset just
/// counter-clockwise of it (26.6 degrees past E goes to E, 63.4 to NE), so all 8 are equal.
pub fn sector(dy: i64, dx: i64) -> Option<usize> {
    if dy == 0 && dx == 0 {
        return None;
    }
    let a = (-dy as f64).atan2(dx as f64).to_degrees();
    Some((((a + 18.0) / 45.0).floor() as i64).rem_euclid(8) as usize)
}

/// One species: where it may be and how it weighs its urges.
#[derive(Debug, Clone, PartialEq)]
pub struct Species {
    pub name: String,
    /// Open cells hold between these amounts of water (walls never are).
    pub water: (i64, i64),
    /// Score per neighbour already heading this way.
    pub align: i64,
    /// Score per neighbour in this direction, up to `crowd` of them; past that each one
    /// costs `spacing`.
    pub cohesion: i64,
    pub crowd: i64,
    pub spacing: i64,
    /// Score per agent of the other species in this direction (negative: flee).
    pub other: i64,
    /// Score for a direction that lowers the distance field by one (0: no goal).
    pub goal: i64,
    /// Score for turning 0, 1, 2, 3 or 4 steps (of 45 degrees) from the current heading.
    pub turn: [i64; 5],
    /// Score for the heading the noise input points at.
    pub noise: i64,
    /// One noise kick every this many steps on average.
    pub noise_every: u64,
    /// Sub-cell units per step: straight, then each axis on a diagonal.
    pub speed: (i64, i64),
}

impl Species {
    /// Fish: live in water at least a quarter deep, school, follow food, flee birds.
    pub fn fish() -> Self {
        Self {
            name: "fish".into(),
            water: (4, WATER),
            align: 2,
            cohesion: 1,
            crowd: 4,
            spacing: 2,
            other: -3,
            goal: 4,
            turn: [8, 4, 0, -6, -12],
            noise: 6,
            noise_every: 10,
            speed: (3, 2),
        }
    }

    /// Birds: fly in dry air, flock, and gather over fish.
    pub fn bird() -> Self {
        Self {
            name: "bird".into(),
            water: (0, 1),
            align: 3,
            cohesion: 1,
            crowd: 3,
            spacing: 2,
            other: 2,
            goal: 0,
            turn: [8, 4, 0, -6, -12],
            noise: 6,
            noise_every: 10,
            speed: (4, 3),
        }
    }

    pub fn validate(&self) -> Result<()> {
        let ok = self.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            && !self.name.is_empty()
            && 0 <= self.water.0
            && self.water.0 <= self.water.1
            && self.water.1 <= WATER
            && (1..CROWD_CAP).contains(&self.crowd)
            && self.spacing >= 0
            && (1..CELL).contains(&self.speed.0)
            && (1..CELL).contains(&self.speed.1)
            && self.noise_every >= 1
            && [self.align, self.cohesion, self.spacing, self.other, self.goal, self.noise].iter().chain(&self.turn).all(|v| v.abs() <= 100);
        if ok { Ok(()) } else { Err(WirError::new(format!("swarm species `{}` is out of range", self.name))) }
    }

    pub fn has_goal(&self) -> bool {
        self.goal != 0
    }

    /// Sense channels in program order.
    pub fn channels(&self) -> Vec<String> {
        let mut c: Vec<String> = (0..8).map(|k| format!("own{k}")).collect();
        c.extend(["other", "m", "wall"].map(String::from));
        if self.has_goal() {
            c.push("dist".into());
        }
        c
    }

    pub fn pads(&self) -> Vec<i64> {
        let mut p = vec![0; 8];
        p.extend([0, 0, 1]);
        if self.has_goal() {
            p.push(CEILING);
        }
        p
    }

    pub fn span(&self, ch: usize) -> (i64, i64) {
        match ch {
            0..8 => (0, HEADING_CAP),
            8 => (0, OTHER_CAP),
            9 => (0, WATER),
            10 => (0, 1),
            _ => (0, CEILING),
        }
    }

    /// Senses per agent: a, n, t, b (and g with a goal), 8 each.
    pub fn sense_count(&self) -> usize {
        if self.has_goal() { 40 } else { 32 }
    }

    /// Bounds of a heading's score without the blocked term.
    pub fn score_range(&self) -> (i64, i64) {
        let coh = |n: i64| self.cohesion * (n - (1 + self.spacing) * (n - self.crowd).max(0));
        let ends = |a: i64, b: i64| (a.min(b), a.max(b));
        let (cl, ch) = (0..=CROWD_CAP).map(coh).fold((i64::MAX, i64::MIN), |(l, h), v| (l.min(v), h.max(v)));
        let parts = [
            ends(0, self.align * ALIGN_CAP),
            (cl, ch),
            ends(0, self.other * OTHER_SENSE_CAP),
            ends(-self.goal, self.goal),
            (*self.turn.iter().min().unwrap(), *self.turn.iter().max().unwrap()),
            ends(0, self.noise),
        ];
        parts.iter().fold((0, 0), |(l, h), &(a, b)| (l + a, h + b))
    }

    /// Penalty on a blocked heading: larger than the whole score range, so a blocked
    /// heading always loses to an open one.
    pub fn blocked_penalty(&self) -> i64 {
        let (lo, hi) = self.score_range();
        hi - lo + 1
    }
}

/// The senses as plain code. `get(channel, dy, dx)` reads the grid around the cell.
/// Order: a0..a7, n0..n7, t0..t7, b0..b7, then g0..g7 with a goal.
pub fn sense_cell(sp: &Species, get: impl Fn(usize, i64, i64) -> i64) -> Vec<i64> {
    let window = || (-RADIUS..=RADIUS).flat_map(|dy| (-RADIUS..=RADIUS).map(move |dx| (dy, dx)));
    let crowd = |dy: i64, dx: i64| (0..8).map(|k| get(k, dy, dx)).sum::<i64>();
    let open = |dy: i64, dx: i64| get(10, dy, dx) == 0 && (sp.water.0..=sp.water.1).contains(&get(9, dy, dx));
    let mut s = Vec::with_capacity(sp.sense_count());
    for d in 0..8 {
        s.push(window().map(|(dy, dx)| get(d, dy, dx)).sum::<i64>().min(ALIGN_CAP));
    }
    for d in 0..8 {
        s.push(window().filter(|&(dy, dx)| sector(dy, dx) == Some(d)).map(|(dy, dx)| crowd(dy, dx)).sum::<i64>().min(CROWD_CAP));
    }
    for d in 0..8 {
        s.push(window().filter(|&(dy, dx)| sector(dy, dx) == Some(d)).map(|(dy, dx)| get(8, dy, dx)).sum::<i64>().min(OTHER_SENSE_CAP));
    }
    for (dy, dx) in DIRS {
        // A diagonal is blocked when either side cell is: no cutting past a corner.
        let clear = open(dy, dx) && (dy == 0 || dx == 0 || (open(dy, 0) && open(0, dx)));
        s.push(!clear as i64);
    }
    if sp.has_goal() {
        for (dy, dx) in DIRS {
            s.push((get(11, 0, 0) - get(11, dy, dx)).clamp(-1, 1));
        }
    }
    s
}

/// The brain as plain code: the new heading from the current one, the noise kick
/// (a heading or None) and the senses.
pub fn brain_choose(sp: &Species, heading: usize, kick: Option<usize>, senses: &[i64]) -> usize {
    let big = sp.blocked_penalty();
    (0..8)
        .max_by_key(|&d| {
            let n = senses[8 + d];
            let mut s = sp.align * senses[d]
                + sp.cohesion * (n - (1 + sp.spacing) * (n - sp.crowd).max(0))
                + sp.other * senses[16 + d]
                + sp.turn[turn(d, heading)]
                + sp.noise * (kick == Some(d)) as i64
                - big * senses[24 + d];
            if sp.has_goal() {
                s += sp.goal * senses[32 + d];
            }
            // Ties go to the lower heading index.
            8 * s + (7 - d as i64)
        })
        .unwrap()
}

fn tap(ch: &str, dy: i64, dx: i64) -> String {
    if dy == 0 && dx == 0 { ch.to_string() } else { format!("{ch}[{dy}, {dx}]") }
}

/// The sense program: a grid program whose outputs at a cell are an agent's senses there.
pub fn sense_program(sp: &Species) -> Result<String> {
    sp.validate()?;
    let window: Vec<(i64, i64)> = (-RADIUS..=RADIUS).flat_map(|dy| (-RADIUS..=RADIUS).map(move |dx| (dy, dx))).collect();
    let sum = |ch: &str, keep: &dyn Fn(i64, i64) -> bool| {
        window.iter().filter(|&&(dy, dx)| keep(dy, dx)).map(|&(dy, dx)| tap(ch, dy, dx)).collect::<Vec<_>>().join(" + ")
    };
    let (lo, hi) = sp.water;
    // One step when only one water bound matters (wall pushes the step below 0).
    let open = match (lo > 0, hi < WATER) {
        (false, false) => "not(wall)".to_string(),
        (true, false) => format!("clip(m - {} - {WATER}*wall, 0, 1)", lo - 1),
        (false, true) => format!("clip({} - m - {}*wall, 0, 1)", hi + 1, hi + 1),
        (true, true) => format!("and(not(wall), ge(m, {lo}), le(m, {hi}))"),
    };
    let mut src = format!(
        "wir 0.2
# Senses of one {name} (generated by `wir swarm`). A grid program: the outputs at a cell are
# what an agent in that cell perceives, read from the 5 x 5 window around it.
# own0..own7 count {name} per cell by heading (E NE N NW W SW S SE, capped at {hc}),
# other counts the other species (capped at {oc}), m is water (0 to {water}).
# Open cells for a {name}: no wall, {lo} to {hi} units of water.

",
        name = sp.name,
        hc = HEADING_CAP,
        oc = OTHER_CAP,
        water = WATER,
    );
    for k in 0..8 {
        src += &format!("input own{k}: grid in [0, {HEADING_CAP}] pad 0\n");
    }
    src += &format!("input other: grid in [0, {OTHER_CAP}] pad 0\ninput m: grid in [0, {WATER}] pad 0\ninput wall: grid in [0, 1] pad 1\n");
    if sp.has_goal() {
        src += &format!("input dist: grid in [0, {CEILING}] pad {CEILING}\n");
    }
    src += &format!(
        "\nlet crowd = {}\nlet open = {}\n\n# a: neighbours heading each way (alignment). cap - relu(cap - x) is min(x, cap).\n",
        (0..8).map(|k| format!("own{k}")).collect::<Vec<_>>().join(" + "),
        open
    );
    for d in 0..8 {
        src += &format!("output a{d} = {ALIGN_CAP} - relu({ALIGN_CAP} - ({}))\n", sum(&format!("own{d}"), &|_, _| true));
    }
    src += "\n# n: neighbours lying in each direction (cohesion and crowding)\n";
    for d in 0..8 {
        src += &format!("output n{d} = {CROWD_CAP} - relu({CROWD_CAP} - ({}))\n", sum("crowd", &|dy, dx| sector(dy, dx) == Some(d)));
    }
    src += "\n# t: the other species in each direction\n";
    for d in 0..8 {
        src += &format!("output t{d} = {OTHER_SENSE_CAP} - relu({OTHER_SENSE_CAP} - ({}))\n", sum("other", &|dy, dx| sector(dy, dx) == Some(d)));
    }
    src += "\n# b: blocked directions (a diagonal also when either side is)\n";
    for (d, (dy, dx)) in DIRS.iter().enumerate() {
        let b = if *dy != 0 && *dx != 0 {
            format!("not(and({}, {}, {}))", tap("open", *dy, *dx), tap("open", *dy, 0), tap("open", 0, *dx))
        } else {
            format!("not({})", tap("open", *dy, *dx))
        };
        src += &format!("output b{d} = {b}\n");
    }
    if sp.has_goal() {
        src += "\n# g: 1 where a step lowers the distance to the goal, -1 where it raises it\n";
        for (d, (dy, dx)) in DIRS.iter().enumerate() {
            src += &format!("output g{d} = clip(dist - {}, -1, 1)\n", tap("dist", *dy, *dx));
        }
    }
    Ok(src)
}

/// The brain program: scores each heading and outputs the best one as one-hot h2_0..h2_7.
pub fn brain_program(sp: &Species) -> Result<String> {
    sp.validate()?;
    let big = sp.blocked_penalty();
    let list = |p: &str| (0..8).map(|k| format!("{p}{k}")).collect::<Vec<_>>().join(", ");
    let mut src = format!(
        "wir 0.2
# Brain of one {name} (generated by `wir swarm`): scores the 8 headings and takes the best.
# h is the current heading (one-hot), z a noise kick (one-hot or all 0), and a, n, t, b{g}
# are the senses at the agent's cell. Blocked headings lose {big}, more than the rest of
# the score can span, so a {name} never picks a blocked heading while an open one exists.

input {h}: R1 in [0, 1]
input {z}: R1 in [0, 1]
input {a}: R1 in [0, {ALIGN_CAP}]
input {n}: R1 in [0, {CROWD_CAP}]
input {t}: R1 in [0, {OTHER_SENSE_CAP}]
input {b}: R1 in [0, 1]
",
        name = sp.name,
        g = if sp.has_goal() { ", g" } else { "" },
        h = list("h"),
        z = list("z"),
        a = list("a"),
        n = list("n"),
        t = list("t"),
        b = list("b"),
    );
    if sp.has_goal() {
        src += &format!("input {}: R1 in [-1, 1]\n", list("g"));
    }
    let term = |k: i64, x: &str| match k {
        0 => String::new(),
        1 => format!(" + {x}"),
        -1 => format!(" - {x}"),
        k if k < 0 => format!(" - {}*{x}", -k),
        k => format!(" + {k}*{x}"),
    };
    src += "
# score of each heading: the straight part, then cohesion's bend (past `crowd`
";
    src += "# neighbours, each extra one costs). relu(x) - relu(-x) is x, kept as one value so the
";
    src += "# layers above carry 2 units per key instead of its 50 inputs. Times 8 plus a
";
    src += "# tie-break, so no two keys are equal.
";
    for d in 0..8 {
        let mut s = String::new();
        s += &term(sp.align, &format!("a{d}"));
        s += &term(sp.cohesion, &format!("n{d}"));
        s += &term(sp.other, &format!("t{d}"));
        if sp.has_goal() {
            s += &term(sp.goal, &format!("g{d}"));
        }
        for k in 0..8 {
            s += &term(sp.turn[turn(d, k)], &format!("h{k}"));
        }
        s += &term(sp.noise, &format!("z{d}"));
        s += &term(-big, &format!("b{d}"));
        let s = s.trim_start_matches(" + ").to_string();
        src += &format!("let lin{d} = {s}
");
    }
    src += "
";
    for d in 0..8 {
        src += &format!(
            "let key{d} = 8*(relu(lin{d}) - relu(-lin{d}){}) + {}
",
            term(-sp.cohesion * (1 + sp.spacing), &format!("relu(n{d} - {})", sp.crowd)),
            7 - d
        );
    }
    src += "let best = max(max(max(key0, key1), max(key2, key3)), max(max(key4, key5), max(key6, key7)))

";
    for d in 0..8 {
        src += &format!("output h2_{d} = ge(key{d}, best)\n");
    }
    Ok(src)
}

pub(crate) fn xorshift(rng: &mut u64, n: u64) -> u64 {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    *rng % n.max(1)
}

/// One agent: species index, position in sub-cell units, heading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Agent {
    pub species: usize,
    pub y: i64,
    pub x: i64,
    pub heading: usize,
}

impl Agent {
    pub fn cell(&self) -> (i64, i64) {
        (self.y.div_euclid(CELL), self.x.div_euclid(CELL))
    }
}

/// A world: water, walls, the goal distance field and the agents of every species.
#[derive(Debug, Clone, PartialEq)]
pub struct SwarmWorld {
    pub h: usize,
    pub w: usize,
    pub m: Vec<i64>,
    pub wall: Vec<i64>,
    pub dist: Vec<i64>,
    pub agents: Vec<Agent>,
    pub rng: u64,
}

/// How the agents decide: the rule as plain code, or compiled networks.
pub enum Deciders<'a> {
    Rule(&'a [Species]),
    Nets(&'a mut [SwarmNets]),
}

/// One species' compiled programs, ready to run.
pub struct SwarmNets {
    pub species: Species,
    pub sense: deep::StencilWeights,
    pub brain: deep::DeepWeights,
    sense_net: deep::SparseNet,
    brain_net: deep::SparseNet,
}

impl SwarmNets {
    pub fn compile(sp: &Species) -> Result<Self> {
        let sc = deep::compile(&sense_program(sp)?, 1e-6)?;
        let sense = deep::StencilWeights::from_compiled(&sc).ok_or_else(|| WirError::new("sense is not a grid program"))?;
        let brain = deep::compile(&brain_program(sp)?, 1e-6)?.weights;
        Ok(Self { species: sp.clone(), sense_net: deep::SparseNet::new(&sense.net), brain_net: deep::SparseNet::new(&brain), sense, brain })
    }

    /// Senses from the sense network (`get` as for `sense_cell`).
    pub fn senses(&mut self, get: impl Fn(usize, i64, i64) -> i64) -> Vec<i64> {
        let x: Vec<f64> = self.sense.taps.iter().map(|&(ch, dy, dx)| get(ch, dy, dx) as f64).collect();
        self.sense_net.eval(&x).iter().map(|v| v.round() as i64).collect()
    }

    /// Brain network outputs (one-hot when the program is right).
    pub fn brain_outputs(&mut self, heading: usize, kick: Option<usize>, senses: &[i64]) -> Vec<f64> {
        let mut x: Vec<f64> = (0..8).map(|k| (k == heading) as i64 as f64).collect();
        x.extend((0..8).map(|k| (kick == Some(k)) as i64 as f64));
        x.extend(senses.iter().map(|&v| v as f64));
        self.brain_net.eval(&x).to_vec()
    }
}

impl SwarmWorld {
    pub fn at(&self, r: i64, c: i64) -> Option<usize> {
        (r >= 0 && c >= 0 && r < self.h as i64 && c < self.w as i64).then(|| (r * self.w as i64 + c) as usize)
    }

    pub fn open(&self, sp: &Species, r: i64, c: i64) -> bool {
        self.at(r, c).is_some_and(|i| self.wall[i] == 0 && (sp.water.0..=sp.water.1).contains(&self.m[i]))
    }

    /// Per-cell agent counts, species after species: by heading, and in total (uncapped).
    pub fn counts(&self, species: usize) -> (Vec<[i64; 8]>, Vec<i64>) {
        let mut heads = vec![vec![[0i64; 8]; self.h * self.w]; species];
        let mut dens = vec![vec![0i64; self.h * self.w]; species];
        for a in &self.agents {
            let (r, c) = a.cell();
            if let Some(i) = self.at(r, c) {
                heads[a.species][i][a.heading] += 1;
                dens[a.species][i] += 1;
            }
        }
        (heads.concat(), dens.concat())
    }

    /// Advance one step: every agent decides from the same snapshot, then all move.
    /// Returns (decisions that picked a blocked heading while an open one existed,
    /// agents that could not move).
    pub fn step(&mut self, species: &[Species], deciders: &mut Deciders) -> (u64, u64) {
        let n = species.len();
        let cells = self.h * self.w;
        let (heads, dens) = self.counts(n);
        let cap_heads: Vec<[i64; 8]> = heads.iter().map(|h| h.map(|v| v.min(HEADING_CAP))).collect();
        // `other` is every other species together.
        let other: Vec<Vec<i64>> = (0..n)
            .map(|s| (0..cells).map(|i| (0..n).filter(|&o| o != s).map(|o| dens[o * cells + i]).sum::<i64>().min(OTHER_CAP)).collect())
            .collect();
        let mut chosen = Vec::with_capacity(self.agents.len());
        let mut bad = 0;
        for a in self.agents.clone() {
            let sp = &species[a.species];
            let kick = (xorshift(&mut self.rng, sp.noise_every) == 0).then(|| xorshift(&mut self.rng, 8) as usize);
            let (r, c) = a.cell();
            let pads = sp.pads();
            let get = |ch: usize, dy: i64, dx: i64| match self.at(r + dy, c + dx) {
                None => pads[ch],
                Some(i) => match ch {
                    0..8 => cap_heads[a.species * cells + i][ch],
                    8 => other[a.species][i],
                    9 => self.m[i],
                    10 => self.wall[i],
                    _ => self.dist[i],
                },
            };
            let (senses, h2) = match deciders {
                Deciders::Rule(sps) => {
                    let s = sense_cell(&sps[a.species], get);
                    let h2 = brain_choose(&sps[a.species], a.heading, kick, &s);
                    (s, h2)
                }
                Deciders::Nets(nets) => {
                    let net = &mut nets[a.species];
                    let s = net.senses(get);
                    let out = net.brain_outputs(a.heading, kick, &s);
                    let hot: Vec<usize> = (0..8).filter(|&k| (out[k] - 1.0).abs() < 1e-6).collect();
                    let h2 = if hot.len() == 1 && out.iter().all(|v| v.abs() < 1e-6 || (v - 1.0).abs() < 1e-6) { hot[0] } else { usize::MAX };
                    (s, h2)
                }
            };
            if h2 < 8 && senses[24 + h2] == 1 && (0..8).any(|d| senses[24 + d] == 0) {
                bad += 1;
            }
            chosen.push(h2);
        }
        let mut stuck = 0;
        let mut agents = std::mem::take(&mut self.agents);
        let mut room = vec![0i64; n * cells];
        for a in &agents {
            let (r, c) = a.cell();
            if let Some(i) = self.at(r, c) {
                room[a.species * cells + i] += 1;
            }
        }
        for (a, h2) in agents.iter_mut().zip(chosen) {
            a.heading = h2.min(7);
            let sp = &species[a.species];
            let (dy, dx) = DIRS[a.heading];
            let v = if dy != 0 && dx != 0 { sp.speed.1 } else { sp.speed.0 };
            let (y, x) = (a.y + dy * v, a.x + dx * v);
            let (r0, c0) = a.cell();
            let (r1, c1) = (y.div_euclid(CELL), x.div_euclid(CELL));
            let here = self.at(r0, c0).is_some_and(|i| self.wall[i] == 0 && (sp.water.0..=sp.water.1).contains(&self.m[i]));
            let there = self.at(r1, c1).is_some_and(|i| self.wall[i] == 0 && (sp.water.0..=sp.water.1).contains(&self.m[i]));
            // Stay unless the destination is open; an agent whose own cell closed (water
            // drained or rose) may move anywhere inside the grid to get out.
            // Agents move in order and take room as they go: a full cell turns others away.
            let fits = (r1, c1) == (r0, c0) || self.at(r1, c1).is_some_and(|i| room[a.species * cells + i] < ROOM);
            if fits && (there || (!here && self.at(r1, c1).is_some_and(|i| self.wall[i] == 0))) {
                if (r1, c1) != (r0, c0) {
                    if let Some(i) = self.at(r0, c0) {
                        room[a.species * cells + i] -= 1;
                    }
                    room[a.species * cells + self.at(r1, c1).unwrap()] += 1;
                }
                a.y = y;
                a.x = x;
            } else {
                stuck += 1;
            }
        }
        self.agents = agents;
        (bad, stuck)
    }

    /// How aligned a species is: |mean heading vector|, 0 (random) to 1 (all the same way).
    pub fn polarization(&self, species: usize) -> f64 {
        let (mut sx, mut sy, mut n) = (0.0, 0.0, 0.0);
        for a in self.agents.iter().filter(|a| a.species == species) {
            let (dy, dx) = DIRS[a.heading];
            let l = ((dy * dy + dx * dx) as f64).sqrt();
            sx += dx as f64 / l;
            sy += dy as f64 / l;
            n += 1.0;
        }
        if n == 0.0 { 0.0 } else { (sx * sx + sy * sy).sqrt() / n }
    }

    /// Per species, averaged over its agents: same-species neighbours in the 5 x 5 window,
    /// and local order, |mean heading vector| over the agent and those neighbours (1 when
    /// they all fly the same way; a lone agent counts 1, so read it with the neighbours).
    pub fn local_stats(&self, species: usize) -> (f64, f64) {
        let unit = |h: usize| {
            let (dy, dx) = DIRS[h];
            let l = ((dy * dy + dx * dx) as f64).sqrt();
            (dy as f64 / l, dx as f64 / l)
        };
        let mut grid = vec![(0.0, 0.0, 0i64); self.h * self.w];
        let mine: Vec<&Agent> = self.agents.iter().filter(|a| a.species == species).collect();
        for a in &mine {
            let (r, c) = a.cell();
            if let Some(i) = self.at(r, c) {
                let (uy, ux) = unit(a.heading);
                grid[i].0 += uy;
                grid[i].1 += ux;
                grid[i].2 += 1;
            }
        }
        if mine.is_empty() {
            return (0.0, 0.0);
        }
        let (mut nb, mut order) = (0.0, 0.0);
        for a in &mine {
            let (r, c) = a.cell();
            let (mut sy, mut sx, mut n) = (0.0, 0.0, 0);
            for dy in -RADIUS..=RADIUS {
                for dx in -RADIUS..=RADIUS {
                    if let Some(i) = self.at(r + dy, c + dx) {
                        sy += grid[i].0;
                        sx += grid[i].1;
                        n += grid[i].2;
                    }
                }
            }
            nb += (n - 1) as f64;
            order += (sy * sy + sx * sx).sqrt() / n as f64;
        }
        (nb / mine.len() as f64, order / mine.len() as f64)
    }

    /// Distance to the nearest goal cell through cells open to `sp` (4 moves, cost 1),
    /// capped at the ceiling: what `wir path` converges to with those cells as walls.
    pub fn goal_distances(&mut self, sp: &Species, goals: &[(i64, i64)]) {
        let mut dist = vec![CEILING; self.h * self.w];
        let mut queue = std::collections::VecDeque::new();
        for &(r, c) in goals {
            if let Some(i) = self.at(r, c).filter(|_| self.open(sp, r, c)) {
                dist[i] = 0;
                queue.push_back((r, c));
            }
        }
        while let Some((r, c)) = queue.pop_front() {
            let d = dist[self.at(r, c).unwrap()];
            for (dy, dx) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
                if let Some(j) = self.at(r + dy, c + dx).filter(|_| self.open(sp, r + dy, c + dx)) {
                    if dist[j] > d + 1 {
                        dist[j] = d + 1;
                        queue.push_back((r + dy, c + dx));
                    }
                }
            }
        }
        self.dist = dist;
    }
}

/// A test world: a walled tank with water below `waterline`, a few rocks, fish in the
/// water and birds in the air, all with random headings; food at one spot in the water.
pub fn swarm_world(species: &[Species], h: usize, w: usize, per_species: usize, seed: u64) -> SwarmWorld {
    let mut rng = seed.max(1);
    let n = h * w;
    let mut s = SwarmWorld { h, w, m: vec![0; n], wall: vec![0; n], dist: vec![CEILING; n], agents: vec![], rng: 0 };
    let waterline = h / 2;
    for r in 0..h {
        for c in 0..w {
            let i = r * w + c;
            if r + 1 == h || c == 0 || c + 1 == w {
                s.wall[i] = 1;
            } else if r >= waterline {
                s.m[i] = WATER;
            }
        }
    }
    for _ in 0..(n / 120).max(1) {
        let (r, c) = (xorshift(&mut rng, h as u64) as usize, xorshift(&mut rng, w as u64) as usize);
        for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            if r + dy < h && c + dx < w {
                s.wall[(r + dy) * w + c + dx] = 1;
                s.m[(r + dy) * w + c + dx] = 0;
            }
        }
    }
    for (k, sp) in species.iter().enumerate() {
        let mut placed = 0;
        let mut tries = 0;
        while placed < per_species && tries < 100_000 {
            tries += 1;
            let (r, c) = (xorshift(&mut rng, h as u64) as i64, xorshift(&mut rng, w as u64) as i64);
            if s.open(sp, r, c) {
                let (oy, ox) = (xorshift(&mut rng, CELL as u64) as i64, xorshift(&mut rng, CELL as u64) as i64);
                s.agents.push(Agent { species: k, y: r * CELL + oy, x: c * CELL + ox, heading: xorshift(&mut rng, 8) as usize });
                placed += 1;
            }
        }
    }
    if let Some(sp) = species.iter().find(|sp| sp.has_goal()) {
        let goal = (h as i64 * 3 / 4, w as i64 * 3 / 4);
        let sp = sp.clone();
        s.goal_distances(&sp, &[goal]);
    }
    s.rng = 1 + xorshift(&mut rng, u64::MAX - 1);
    s
}

/// Outcome of `swarm_check`.
#[derive(Debug, Clone, PartialEq)]
pub struct SwarmCheck {
    /// Random cell inputs where the sense network matched `sense_cell`, per species.
    pub sense_cases: u64,
    pub sense_mismatches: u64,
    /// Random brain inputs where the brain network matched `brain_choose`.
    pub brain_cases: u64,
    pub brain_mismatches: u64,
    /// Worlds run with networks and with the rule side by side.
    pub worlds: u64,
    pub steps: u64,
    pub decisions: u64,
    pub step_mismatches: u64,
    /// Decisions that picked a blocked heading while an open one existed.
    pub blocked_picks: u64,
    /// Per species, at the start and end, averaged over the worlds: global polarization
    /// (|mean heading vector|), same-species neighbours and local order (`local_stats`).
    pub polarization: Vec<(f64, f64)>,
    pub neighbours: Vec<(f64, f64)>,
    pub order: Vec<(f64, f64)>,
    /// Per species: sense taps, sense hidden layers, sense conv parameters, brain hidden, brain parameters.
    pub shapes: Vec<(usize, Vec<usize>, usize, Vec<usize>, usize)>,
}

/// Compile every species' programs and check them: the sense network against
/// `sense_cell` on `cases` random cell inputs, the brain against `brain_choose` on
/// `cases` random inputs, then `worlds` random h x w worlds run `steps` steps with the
/// networks and the rule in lockstep (every agent's state must agree every step).
#[allow(clippy::too_many_arguments)]
pub fn swarm_check(species: &[Species], cases: u64, worlds: u64, steps: u64, h: usize, w: usize, per_species: usize, seed: u64) -> Result<SwarmCheck> {
    let mut nets: Vec<SwarmNets> = species.iter().map(SwarmNets::compile).collect::<Result<_>>()?;
    let mut rng = seed.max(1);
    let (mut sense_mismatches, mut brain_mismatches) = (0, 0);
    for net in nets.iter_mut() {
        let sp = net.species.clone();
        for _ in 0..cases {
            let mut vals = std::collections::HashMap::new();
            for &(ch, dy, dx) in &net.sense.taps {
                let (lo, hi) = sp.span(ch);
                // Mostly small counts and a mix of full, empty and partial water, so the
                // caps and the open test are both exercised.
                let v = match ch {
                    0..9 if xorshift(&mut rng, 3) != 0 => 0,
                    9 => {
                        let edges = [0, (sp.water.0 - 1).max(0), sp.water.0, sp.water.1, (sp.water.1 + 1).min(WATER), WATER];
                        if xorshift(&mut rng, 2) == 0 { edges[xorshift(&mut rng, 6) as usize] } else { xorshift(&mut rng, (hi + 1) as u64) as i64 }
                    }
                    10 => (xorshift(&mut rng, 5) == 0) as i64,
                    _ => lo + xorshift(&mut rng, (hi - lo + 1) as u64) as i64,
                };
                vals.insert((ch, dy, dx), v);
            }
            let get = |ch: usize, dy: i64, dx: i64| -> i64 {
                *vals.get(&(ch, dy, dx)).unwrap_or_else(|| panic!("the rule reads channel {ch} at [{dy}, {dx}], which the program does not"))
            };
            let want = sense_cell(&sp, get);
            sense_mismatches += (net.senses(get) != want) as u64;
        }
        for _ in 0..cases {
            let heading = xorshift(&mut rng, 8) as usize;
            let kick = (xorshift(&mut rng, 2) == 0).then(|| xorshift(&mut rng, 8) as usize);
            let all_blocked = xorshift(&mut rng, 20) == 0;
            let mut s: Vec<i64> = Vec::with_capacity(sp.sense_count());
            for _ in 0..8 {
                s.push(xorshift(&mut rng, ALIGN_CAP as u64 + 1) as i64);
            }
            for _ in 0..8 {
                s.push(xorshift(&mut rng, CROWD_CAP as u64 + 1) as i64);
            }
            for _ in 0..8 {
                s.push(xorshift(&mut rng, OTHER_SENSE_CAP as u64 + 1) as i64);
            }
            for _ in 0..8 {
                s.push((all_blocked || xorshift(&mut rng, 3) == 0) as i64);
            }
            if sp.has_goal() {
                for _ in 0..8 {
                    s.push(xorshift(&mut rng, 3) as i64 - 1);
                }
            }
            let want = brain_choose(&sp, heading, kick, &s);
            let out = net.brain_outputs(heading, kick, &s);
            let ok = out.iter().enumerate().all(|(k, v)| (v - (k == want) as i64 as f64).abs() < 1e-6);
            brain_mismatches += !ok as u64;
        }
    }
    let (mut total_steps, mut decisions, mut step_mismatches, mut blocked_picks) = (0, 0, 0, 0);
    let ns = species.len();
    let (mut polarization, mut neighbours, mut order) = (vec![(0.0, 0.0); ns], vec![(0.0, 0.0); ns], vec![(0.0, 0.0); ns]);
    let mut stats = |w: &SwarmWorld, end: bool| {
        for s in 0..ns {
            let (nb, ord) = w.local_stats(s);
            let k = worlds.max(1) as f64;
            for (acc, v) in [(&mut polarization[s], w.polarization(s)), (&mut neighbours[s], nb), (&mut order[s], ord)] {
                if end { acc.1 += v / k } else { acc.0 += v / k }
            }
        }
    };
    for _ in 0..worlds {
        let world_seed = 1 + xorshift(&mut rng, u64::MAX - 1);
        let mut a = swarm_world(species, h, w, per_species, world_seed);
        let mut b = a.clone();
        stats(&a, false);
        for _ in 0..steps {
            let (bad, _) = a.step(species, &mut Deciders::Rule(species));
            b.step(species, &mut Deciders::Nets(&mut nets));
            blocked_picks += bad;
            total_steps += 1;
            decisions += a.agents.len() as u64;
            step_mismatches += (a.agents != b.agents) as u64;
        }
        stats(&a, true);
    }
    Ok(SwarmCheck {
        sense_cases: cases * species.len() as u64,
        sense_mismatches,
        brain_cases: cases * species.len() as u64,
        brain_mismatches,
        worlds,
        steps: total_steps,
        decisions,
        step_mismatches,
        blocked_picks,
        polarization,
        neighbours,
        order,
        shapes: nets
            .iter()
            .map(|n| (n.sense.taps.len(), n.sense.net.hidden(), n.sense.param_count(), n.brain.hidden(), n.brain.param_count()))
            .collect(),
    })
}
