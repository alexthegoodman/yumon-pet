//! Yumon Park: an ecosystem where every Yumon carries its own copy of one WIR brain.
//!
//! The brain is a 0.2 utility program written once for every species: response curves
//! over what an individual feels (hunger, fatigue, hurt, fertile) and what it perceives
//! (the nearest threat, food, prey, mate and kin as distance buckets 0..3, whether a
//! refuge band is reachable) give one key per option. The host (the page) samples an
//! option from a softmax over the keys, steers, and runs the world.
//!
//! Founders start with the compiled weights, which give the written keys exactly. After
//! that each individual learns: rewards along the way are kept in a small buffer, and
//! every time it falls asleep it takes policy-gradient steps on its own weights. A child
//! starts from its mother's weights at birth; a Yumon's weights are dropped when it dies.
//! `park_check` runs the compiled network against the rule as plain code.
//!
//! Hunting comes first for an armed Yumon with prey in sight; plants are the fallback.
//! An attacked Yumon can fight back (armed, weighing its odds) or brace (shell or spines).

use crate::deep;
use crate::error::Result;
use crate::swarm::xorshift;
use crate::tactics::sparse_layers;

/// Options, in key order.
pub const OPTIONS: [&str; 10] = ["eat", "hunt", "flee", "hide", "mate", "sleep", "flock", "roam", "fight", "brace"];
/// Number of options.
pub const NOPT: usize = OPTIONS.len();
/// Hunger and fatigue run 0..=LEVEL (the host buckets energy and tiredness).
pub const LEVEL: i64 = 10;
/// Distance buckets: 0 nothing seen, 1 far, 2 near, 3 close.
pub const NEAR: i64 = 3;
/// Hurt runs 0..=HURT (share of health lost, bucketed).
pub const HURT: i64 = 3;
/// How recently an attacker still in sight struck: 0 not, 1..=HIT (3 = just now).
pub const HIT: i64 = 3;
/// Odds against the attacker: 0 worse, 1 even, 2 better.
pub const ODDS: i64 = 2;
/// The current option's drive gets this much more (hysteresis).
pub const INERTIA: i64 = 2;
/// Keys are KEY_SCALE * drive plus a tie-break below KEY_SCALE.
pub const KEY_SCALE: i64 = 16;

/// Inputs after the 8 option bits, with their ranges.
pub const STATE: [(&str, i64); 16] = [
    ("hunger", LEVEL),
    ("fatigue", LEVEL),
    ("hurt", HURT),
    ("fertile", 1),
    ("td", NEAR),
    ("refuge", 1),
    ("fd", NEAR),
    ("pd", NEAR),
    ("md", NEAR),
    ("kc", NEAR),
    ("camo", 1),
    ("social", 1),
    ("hit", HIT),
    ("odds", ODDS),
    ("armed", 1),
    ("guard", 1),
];

/// What a brain reads.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ParkIn {
    pub option: usize,
    pub hunger: i64,
    pub fatigue: i64,
    pub hurt: i64,
    pub fertile: i64,
    /// Nearest threat (a Yumon that can eat this one), food, prey, mate: buckets 0..3.
    pub td: i64,
    /// A band this Yumon can enter and the nearest threat cannot.
    pub refuge: i64,
    pub fd: i64,
    pub pd: i64,
    pub md: i64,
    /// Kin nearby, capped at 3.
    pub kc: i64,
    /// Species traits the brain reads as inputs.
    pub camo: i64,
    pub social: i64,
    /// Struck recently by an attacker still in sight (0..=HIT), and the odds against it.
    pub hit: i64,
    pub odds: i64,
    /// Has a weapon; has a shell or spines.
    pub armed: i64,
    pub guard: i64,
}

impl ParkIn {
    pub fn state(&self) -> [i64; 16] {
        [
            self.hunger, self.fatigue, self.hurt, self.fertile, self.td, self.refuge, self.fd, self.pd, self.md, self.kc, self.camo,
            self.social, self.hit, self.odds, self.armed, self.guard,
        ]
    }

    /// Network input vector in program order.
    pub fn inputs(&self) -> Vec<f64> {
        let mut v: Vec<f64> = (0..NOPT).map(|k| (k == self.option) as i64 as f64).collect();
        v.extend(self.state().map(|k| k as f64));
        v
    }
}

/// The drives as plain code, in option order.
pub fn drives(x: &ParkIn) -> [i64; NOPT] {
    let scared = x.td > 0;
    let eat = if x.fd > 0 && x.hunger >= 2 { x.hunger + 2 * x.fd - 1 } else { 0 };
    let hunt = if x.pd > 0 && x.hunger >= 3 { x.hunger + 3 * x.pd + 1 } else { 0 };
    let flee = if scared { 3 * x.td + 2 * x.refuge + 2 * x.hurt + 2 } else { 0 };
    let hide = if x.camo == 1 && scared && x.td < NEAR { 3 * x.td + 6 } else { 0 };
    let mate = if x.fertile == 1 && x.md > 0 && x.hunger < 6 && x.td < 2 { 6 + 2 * x.md } else { 0 };
    let sleep = if scared { 0 } else { 2 * (x.fatigue - 3).clamp(0, 7) };
    let flock = if x.social == 1 { 2 + x.td - x.kc.min(2) } else { 0 };
    let roam = 2 + if x.fd == 0 && x.pd == 0 { x.hunger.min(5) } else { 0 };
    let fight = if x.armed == 1 && x.hit > 0 { (4 * x.hit + 4 * x.odds - 2 * x.hurt - 2).max(0) } else { 0 };
    let brace = if x.guard == 1 && x.hit >= 2 { 3 * x.hit + 3 } else { 0 };
    [eat, hunt, flee, hide, mate, sleep, flock, roam, fight, brace]
}

/// The keys the network outputs, as plain code.
pub fn keys(x: &ParkIn) -> [i64; NOPT] {
    let d = drives(x);
    let mut k = [0; NOPT];
    for o in 0..NOPT {
        k[o] = KEY_SCALE * (d[o] + INERTIA * (o == x.option) as i64) + (NOPT as i64 - 1 - o as i64);
    }
    k
}

/// The written choice: the largest key (keys never tie).
pub fn best(k: &[i64; NOPT]) -> usize {
    (0..NOPT).max_by_key(|&o| k[o]).unwrap()
}

/// The brain program.
pub fn brain_program() -> String {
    let opts = (0..NOPT).map(|k| format!("o{k}")).collect::<Vec<_>>().join(", ");
    let mut src = format!(
        "wir 0.2
# Brain of a Yumon (generated by `wir park`), one program for every species. It scores
# {n} options: {names}. o is the current option. td fd pd md
# are the nearest threat, food, prey and mate as distance buckets (0 none, 1 far, 2 near,
# 3 close), kc the kin nearby; refuge = a band this Yumon can enter and the threat cannot.
# hit = how recently an attacker still in sight struck (3 just now), odds against it
# (0 worse, 1 even, 2 better). camo, social, armed and guard are traits. The host samples an option from a softmax over the
# keys; every Yumon starts from these weights and then learns its own.

input {opts}: R1 in [0, 1]
",
        n = NOPT,
        names = OPTIONS.join(" "),
    );
    for (name, hi) in STATE {
        src += &format!("input {name}: R1 in [0, {hi}]\n");
    }
    src += &format!(
        "
let scared = gt(td, 0)
# hunting comes first: prey in sight beats plants unless the prey is far and food close
let eat = select(and(gt(fd, 0), ge(hunger, 2)), hunger + 2*fd - 1, 0)
let hunt = select(and(gt(pd, 0), ge(hunger, 3)), hunger + 3*pd + 1, 0)
# flee harder when the threat is close, a refuge is reachable, or already hurt
let flee = select(scared, 3*td + 2*refuge + 2*hurt + 2, 0)
# camouflaged species freeze instead, until the threat is close
let hide = select(and(camo, scared, lt(td, {NEAR})), 3*td + 6, 0)
# a far threat does not stop mating
let mate = select(and(fertile, gt(md, 0), lt(hunger, 6), lt(td, 2)), 6 + 2*md, 0)
let sleep = select(scared, 0, 2*clip(fatigue - 3, 0, 7))
# social species keep together, more so under threat
let flock = select(social, 2 + td - min(kc, 2), 0)
# roam: wander farther when hungry and nothing to eat or hunt is in sight
let roam = 2 + select(and(eq(fd, 0), eq(pd, 0)), min(hunger, 5), 0)
# struck: the armed fight back when the odds allow (less so when hurt); a shell or
# spines can brace instead, unless a refuge or being hurt makes fleeing better
let fight = select(and(armed, gt(hit, 0)), max(4*hit + 4*odds - 2*hurt - 2, 0), 0)
let brace = select(and(guard, ge(hit, 2)), 3*hit + 3, 0)

# keys: {KEY_SCALE} * (drive + {INERTIA} on the current option) + a tie-break
"
    );
    for (o, name) in OPTIONS.iter().enumerate() {
        src += &format!("output key{o} = {KEY_SCALE}*{name} + {}*o{o} + {}\n", KEY_SCALE * INERTIA, NOPT - 1 - o);
    }
    src
}

/// The compiled brain.
pub struct ParkNet {
    pub brain: deep::DeepWeights,
    net: deep::SparseNet,
}

impl ParkNet {
    pub fn compile() -> Result<Self> {
        let brain = deep::compile(&brain_program(), 1e-6)?.weights;
        Ok(Self { net: deep::SparseNet::new(&brain), brain })
    }

    pub fn keys(&mut self, x: &ParkIn) -> Vec<f64> {
        self.net.eval(&x.inputs()).to_vec()
    }
}

/// Random input over the declared domain, weighted toward states where several drives
/// compete.
pub fn random_park_in(rng: &mut u64) -> ParkIn {
    let b = |rng: &mut u64, n: i64| xorshift(rng, n as u64 + 1) as i64;
    let seen = |rng: &mut u64| if xorshift(rng, 3) == 0 { 0 } else { 1 + xorshift(rng, 3) as i64 };
    ParkIn {
        option: xorshift(rng, NOPT as u64) as usize,
        hunger: b(rng, LEVEL),
        fatigue: b(rng, LEVEL),
        hurt: if xorshift(rng, 2) == 0 { 0 } else { b(rng, HURT) },
        fertile: b(rng, 1),
        td: if xorshift(rng, 2) == 0 { 0 } else { seen(rng) },
        refuge: b(rng, 1),
        fd: seen(rng),
        pd: seen(rng),
        md: seen(rng),
        kc: b(rng, NEAR),
        camo: b(rng, 1),
        social: b(rng, 1),
        hit: if xorshift(rng, 2) == 0 { 0 } else { b(rng, HIT) },
        odds: b(rng, ODDS),
        armed: b(rng, 1),
        guard: b(rng, 1),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParkCheck {
    pub cases: u64,
    /// Inputs where some key differs from the rule's by more than 1e-6.
    pub key_mismatches: u64,
    pub worst: f64,
    /// How often each option is the written choice, for coverage.
    pub options: [u64; NOPT],
    pub shape: (Vec<usize>, usize, usize),
}

/// The brain against the rule on `cases` random inputs.
pub fn park_check(cases: u64, seed: u64) -> Result<ParkCheck> {
    let mut net = ParkNet::compile()?;
    let mut rng = seed.max(1);
    let (mut key_mismatches, mut worst, mut options) = (0, 0f64, [0; NOPT]);
    for _ in 0..cases {
        let x = random_park_in(&mut rng);
        let want = keys(&x);
        let got = net.keys(&x);
        let diff = want.iter().zip(&got).map(|(&w, &g)| (w as f64 - g).abs()).fold(0.0, f64::max);
        worst = worst.max(diff);
        key_mismatches += (diff > 1e-6) as u64;
        options[best(&want)] += 1;
    }
    let nonzero = net.brain.layers.iter().map(|l| l.w.data.iter().filter(|v| **v != 0.0).count() + l.b.len()).sum();
    Ok(ParkCheck { cases, key_mismatches, worst, options, shape: (net.brain.hidden(), net.brain.param_count(), nonzero) })
}

/// The brain (sparse layers, source), the constants, and test vectors (inputs with the
/// rule's keys) for the page's JavaScript port of the rule.
pub fn export_json(vectors: u64, seed: u64) -> Result<String> {
    let net = ParkNet::compile()?;
    let mut rng = seed.max(1);
    let vs: Vec<serde_json::Value> = (0..vectors)
        .map(|_| {
            let x = random_park_in(&mut rng);
            let input: Vec<i64> = x.inputs().iter().map(|&v| v as i64).collect();
            serde_json::json!([input, keys(&x)])
        })
        .collect();
    let v = serde_json::json!({
        "format": "wir-park-demo",
        "consts": {
            "options": OPTIONS, "level": LEVEL, "near": NEAR, "hurt": HURT, "hit": HIT, "odds": ODDS,
            "inertia": INERTIA, "key_scale": KEY_SCALE,
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
