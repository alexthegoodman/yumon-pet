// Yumon Park simulation core. Every Yumon carries its own copy of the exported WIR brain
// (dense Float32 weights). Founders start from the compiled weights, which give the
// written keys exactly; afterwards each Yumon samples options from a softmax over its
// keys, keeps a small buffer of what happened, and takes policy-gradient steps on its own
// weights whenever it falls asleep. A child starts from its mother's weights; a Yumon's
// weights are dropped when it dies. The host does the world: four bands (sky, trees,
// ground, water), plants, species built from capability permutations, genes, limbs,
// hunting, mating and death.
(function (root) {
  'use strict';
  const WORLD_W = 2400, WORLD_H = 600;
  // Bands top to bottom: sky, trees, ground, water.
  const BANDS = ['sky', 'trees', 'ground', 'water'];
  const BAND_Y = [[0, 190], [190, 330], [330, 400], [400, 600]];
  const bandOf = (y) => (y < 190 ? 0 : y < 330 ? 1 : y < 400 ? 2 : 3);

  // ---------------------------------------------------------------- capabilities
  // Each capability is carried by a limb; a limb's size gene sets how well the
  // capability works. Limbs shared by two capabilities (claws: climbing and clawing)
  // tie them together.
  const CAPS = [
    { id: 'swim', kind: 'move', limb: 'fins', label: 'Swim' },
    { id: 'walk', kind: 'move', limb: 'legs', label: 'Walk' },
    { id: 'climb', kind: 'move', limb: 'claws', label: 'Climb' },
    { id: 'fly', kind: 'move', limb: 'wings', label: 'Fly' },
    { id: 'bite', kind: 'weapon', limb: 'jaws', label: 'Bite' },
    { id: 'claw', kind: 'weapon', limb: 'claws', label: 'Claw' },
    { id: 'sting', kind: 'weapon', limb: 'stinger', label: 'Sting' },
    { id: 'horn', kind: 'weapon', limb: 'horns', label: 'Horn' },
    { id: 'shell', kind: 'guard', limb: 'shell', label: 'Shell' },
    { id: 'spines', kind: 'guard', limb: 'spines', label: 'Spines' },
    { id: 'camo', kind: 'guard', limb: 'skin', label: 'Camouflage' },
    { id: 'social', kind: 'mind', limb: null, label: 'Social' },
    { id: 'graze', kind: 'diet', food: 'grass', label: 'Graze' },
    { id: 'browse', kind: 'diet', food: 'fruit', label: 'Browse' },
    { id: 'filter', kind: 'diet', food: 'plankton', label: 'Filter' },
    { id: 'hawk', kind: 'diet', food: 'gnats', label: 'Hawk' },
    { id: 'scavenge', kind: 'diet', food: 'meat', label: 'Scavenge' },
  ];
  const CAP = Object.fromEntries(CAPS.map((c) => [c.id, c]));
  const LIMBS = ['fins', 'legs', 'claws', 'wings', 'jaws', 'stinger', 'horns', 'shell', 'spines', 'skin'];
  const FOODS = ['gnats', 'fruit', 'grass', 'plankton', 'meat'];
  const FOOD_BAND = { gnats: 0, fruit: 1, grass: 2, plankton: 3 };
  // Movement sets: every 1- and 2-capability combination of the four.
  const LOCO = [['swim'], ['walk'], ['climb'], ['fly'], ['swim', 'walk'], ['walk', 'climb'], ['walk', 'fly'], ['swim', 'fly'], ['climb', 'fly'], ['swim', 'climb']];
  const WEAPONS = [[], [], ['bite'], ['claw'], ['sting'], ['horn'], ['bite', 'claw'], ['bite', 'sting'], ['bite', 'horn']];
  const GUARDS = [[], ['shell'], ['spines'], ['camo']];

  // ---------------------------------------------------------------- tuning
  const DEFAULTS = {
    temp: 8, // softmax temperature in key units (a drive unit is 16 key units)
    // Rewards on top of the homeostatic one: a kill, a blow landed (hunting or fighting
    // back), getting away from an attacker alive. A killer gorges on up to gorge of the body.
    killR: 3, hitR: 0.3, escapeR: 1, gorge: 0.5,
    // Evolution: every gene x exp(mutSmall N), one gene x exp(mutBig N), hue +- hueStep;
    // a capability is gained or lost with probability capRate per birth. Every splitEvery
    // ticks a group of at least splitMin (both sexes) with another capability set, or a
    // gene cluster splitDist away (log units), becomes a new species.
    mutSmall: 0.03, mutBig: 0.15, hueStep: 6, capRate: 0.06,
    splitEvery: 250, splitMin: 5, splitDist: 0.3, maxSpecies: 160,
    wallLife: 0, // ticks a wall stands; 0 until removed
    breedCd: 400, twinSize: 0.85, // mating cooldown at size 1; unarmed Yumon smaller than twinSize have twins
    gamma: 0.85, lr: 0.003, anchor: 0.002, clip: 2.0, minBuf: 16, maxBuf: 360, epochs: 1,
    foodE: 25, popCap: 900, needScale: 20,
    speed: 1.4, // all band speeds
    // A sampled option is kept this many ticks (one decision) unless its probability
    // falls below drop; rewards in between go to that decision.
    commit: 12, drop: 0.05,
    // Plant patches: a season lasts season[0..1] ticks; a patch eaten below wither of its
    // full amount withers early; it sprouts at sprout of full, reformAway[0..1] along.
    call: 2.5, // mating-call range, times the sense radius
    plantRate: 1.25, // moving patches lose food (sprouts start small, eaters lose track)
    season: [2000, 4000], wither: 0.1, sprout: 0.35, reformAway: [150, 450], grace: 400,
    roamFar: [300, 900], roamTicks: 700, roamPace: 0.75,
  };

  function makeRng(s) {
    let seed = (s >>> 0) || 1;
    const rnd = () => {
      seed = (seed + 0x6d2b79f5) >>> 0;
      let t = seed;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    return rnd;
  }

  // ---------------------------------------------------------------- the brain
  // Dense layout of the exported sparse layers: per layer W [rows x cols] then b.
  function brainShape(spec) {
    let cols = spec.inputs.length, size = 0;
    const layers = spec.layers.map((l) => {
      const rows = l.rows.length, L = { rows, cols, w: size, b: size + rows * cols };
      size += rows * cols + rows;
      cols = rows;
      return L;
    });
    const init = new Float32Array(size);
    spec.layers.forEach((l, k) => {
      const L = layers[k];
      l.rows.forEach((r, i) => {
        for (let j = 0; j < r.length; j += 2) init[L.w + i * L.cols + r[j]] = r[j + 1];
      });
      l.b.forEach((v, i) => (init[L.b + i] = v));
    });
    const width = Math.max(spec.inputs.length, ...layers.map((L) => L.rows));
    // Activation buffers: acts[k] is the input of layer k, acts[n] the output.
    const acts = [new Float64Array(spec.inputs.length), ...layers.map((L) => new Float64Array(L.rows))];
    const deltas = layers.map((L) => new Float64Array(L.rows));
    function forward(w, x) {
      acts[0].set(x);
      for (let k = 0; k < layers.length; k++) {
        const L = layers[k], a = acts[k], out = acts[k + 1], last = k === layers.length - 1;
        for (let r = 0; r < L.rows; r++) {
          let z = w[L.b + r];
          const o = L.w + r * L.cols;
          for (let c = 0; c < L.cols; c++) z += w[o + c] * a[c];
          out[r] = last || z > 0 ? z : 0;
        }
      }
      return acts[layers.length];
    }
    // Adds d(sum_k dOut_k * out_k)/dw into g, after forward(w, x).
    function backward(w, dOut, g) {
      const n = layers.length;
      deltas[n - 1].set(dOut);
      for (let k = n - 1; k >= 0; k--) {
        const L = layers[k], a = acts[k], d = deltas[k];
        for (let r = 0; r < L.rows; r++) {
          const dr = d[r];
          if (dr === 0) continue;
          g[L.b + r] += dr;
          const o = L.w + r * L.cols;
          for (let c = 0; c < L.cols; c++) g[o + c] += dr * a[c];
        }
        if (k === 0) break;
        const prev = deltas[k - 1];
        for (let c = 0; c < L.cols; c++) {
          if (a[c] <= 0) { prev[c] = 0; continue; }
          let s = 0;
          for (let r = 0; r < L.rows; r++) s += w[L.w + r * L.cols + c] * d[r];
          prev[c] = s;
        }
      }
    }
    return { layers, size, init, forward, backward, nIn: spec.inputs.length };
  }

  function softmax(keys, temp, p) {
    let mx = -Infinity;
    for (let k = 0; k < keys.length; k++) mx = Math.max(mx, keys[k]);
    let s = 0;
    for (let k = 0; k < keys.length; k++) s += (p[k] = Math.exp((keys[k] - mx) / temp));
    for (let k = 0; k < keys.length; k++) p[k] /= s;
    return p;
  }

  // ---------------------------------------------------------------- the written rule
  function ruleKeys(x, C) {
    const N = C.options.length;
    const opt = x.slice(0, N).indexOf(1);
    const [hunger, fatigue, hurt, fertile, td, refuge, fd, pd, md, kc, camo, social, hit, odds, armed, guard] = x.slice(N);
    const scared = td > 0;
    const d = [
      fd > 0 && hunger >= 2 ? hunger + 2 * fd - 1 : 0,
      pd > 0 && hunger >= 3 ? hunger + 3 * pd + 1 : 0,
      scared ? 3 * td + 2 * refuge + 2 * hurt + 2 : 0,
      camo === 1 && scared && td < C.near ? 3 * td + 6 : 0,
      fertile === 1 && md > 0 && hunger < 6 && td < 2 ? 6 + 2 * md : 0,
      scared ? 0 : 2 * Math.min(7, Math.max(0, fatigue - 3)),
      social === 1 ? 2 + td - Math.min(kc, 2) : 0,
      2 + (fd === 0 && pd === 0 ? Math.min(hunger, 5) : 0),
      armed === 1 && hit > 0 ? Math.max(0, 4 * hit + 4 * odds - 2 * hurt - 2) : 0,
      guard === 1 && hit >= 2 ? 3 * hit + 3 : 0,
    ];
    return d.map((v, o) => C.key_scale * (v + C.inertia * (o === opt ? 1 : 0)) + (N - 1 - o));
  }
  const argmax = (a) => { let b = 0; for (let k = 1; k < a.length; k++) if (a[k] > a[b]) b = k; return b; };

  // ---------------------------------------------------------------- species
  const SYL = {
    swim: ['Fin', 'Gill', 'Rip', 'Bub', 'Kelp'], walk: ['Paw', 'Trot', 'Mud', 'Burr', 'Stomp'], climb: ['Bough', 'Twig', 'Lim', 'Nook', 'Vine'],
    fly: ['Wisp', 'Puff', 'Sky', 'Flit', 'Zeph'],
  };
  const TAIL = { bite: ['fang', 'chomp', 'maw'], claw: ['claw', 'rake'], sting: ['sting', 'barb'], horn: ['horn', 'butt'], shell: ['shell', 'plate'], spines: ['quill', 'thorn'], camo: ['shade', 'murk'], none: ['let', 'kin', 'mo', 'bit', 'ling', 'pip'] };

  // Roles fix the shape of the food web: half the species eat only plants, a quarter
  // hunt and also eat plants, a quarter only hunt (and scavenge).
  // diets: 'mixed' (half plant eaters, a quarter omnivores, a quarter hunters), 'hunters'
  // (every species armed and still eating plants), 'carnivores' (armed, meat only).
  function makeSpecies(rnd, count, diets = 'mixed') {
    const pick = (a) => a[Math.floor(rnd() * a.length)];
    const roles = [];
    for (let i = 0; i < count; i++) roles.push(diets === 'hunters' ? 'omni' : diets === 'carnivores' ? 'carn' : i % 4 < 2 ? 'herb' : i % 4 === 2 ? 'omni' : 'carn');
    for (let i = roles.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [roles[i], roles[j]] = [roles[j], roles[i]]; }
    const out = [], seen = new Set(), names = new Set();
    let guard = 0;
    while (out.length < count && guard++ < 5000) {
      const role = roles[out.length];
      const loco = LOCO[out.length % LOCO.length];
      const weapons = role === 'herb' ? [] : pick(WEAPONS.filter((w) => w.length));
      const guards = pick(GUARDS);
      const bands = new Set();
      for (const c of loco) for (const b of bandsFor(c)) bands.add(b);
      const plants = Object.keys(FOOD_BAND).filter((f) => bands.has(FOOD_BAND[f]));
      const diet = new Set();
      const carn = weapons.length > 0;
      if (role !== 'carn') {
        diet.add(pick(plants));
        if (rnd() < 0.35) diet.add(pick(plants));
      }
      if (carn || rnd() < 0.15) diet.add('meat');
      const caps = [...loco, ...weapons, ...guards];
      if (rnd() < 0.3) caps.push('social');
      for (const f of diet) {
        const c = CAPS.find((k) => k.food === f);
        if (c && (f !== 'meat' || !carn)) caps.push(c.id);
      }
      const sig = [...caps].sort().join('+') + '|' + [...diet].sort().join('+');
      if (seen.has(sig)) continue;
      seen.add(sig);
      // Fliers stay light enough to lift; hunters run a little bigger.
      const lo = carn ? 0.85 : 0.6, span = loco.includes('fly') ? 1.15 - lo : carn ? 1.0 : 0.9;
      const genes = { size: lo + rnd() * span, eyes: 0.7 + rnd() * 0.6, hue: (out.length * 137.5) % 360 };
      for (const l of LIMBS) genes[l] = 0;
      for (const c of caps) if (CAP[c].limb) genes[CAP[c].limb] = 0.75 + rnd() * 0.5;
      const cs = new Set(caps);
      out.push({ id: out.length, name: nameFor(cs, rnd, names), role, caps: cs, caps0: new Set(cs), diet: dietOf(cs), genes, carn, alive: 0, born: 0, extinctAt: null, maxGen: 0, parent: null, since: 0 });
    }
    return out;
  }
  // A name from the capabilities: a syllable for how it moves, a tail for its weapon or guard.
  function nameFor(caps, rnd, names) {
    const pick = (a) => a[Math.floor(rnd() * a.length)];
    const loco = ['swim', 'walk', 'climb', 'fly'].filter((c) => caps.has(c));
    const tailKey = ['bite', 'claw', 'sting', 'horn', 'shell', 'spines', 'camo'].find((c) => caps.has(c)) || 'none';
    let name;
    for (let t = 0; t < 30; t++) {
      name = pick(SYL[loco[loco.length - 1] || 'walk']) + pick(TAIL[tailKey]);
      if (!names.has(name)) break;
    }
    for (let k = 2; names.has(name); k++) if (!names.has(name + ' ' + roman(k))) name += ' ' + roman(k);
    names.add(name);
    return name;
  }
  const roman = (n) => ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X'][n] || String(n);
  // What a capability set can eat: the plants of its diet capabilities, and meat with a
  // weapon or scavenging.
  function dietOf(caps) {
    const d = new Set();
    for (const c of caps) if (CAP[c].food && CAP[c].food !== 'meat') d.add(CAP[c].food);
    if (caps.has('scavenge') || [...caps].some((c) => CAP[c].kind === 'weapon')) d.add('meat');
    return d;
  }
  function roleOf(caps) {
    const armed = [...caps].some((c) => CAP[c].kind === 'weapon');
    const plants = [...caps].some((c) => CAP[c].food && CAP[c].food !== 'meat');
    return !armed ? 'herb' : plants ? 'omni' : 'carn';
  }
  const sigOf = (caps) => [...caps].sort().join('+');
  function bandsFor(cap) {
    return cap === 'swim' ? [3] : cap === 'walk' ? [2] : cap === 'climb' ? [1, 2] : cap === 'fly' ? [0, 1, 2] : [];
  }

  // ---------------------------------------------------------------- traits
  // Quantities from genes and growth. Every limb costs upkeep; size buys reach and
  // strength at the cost of speed and food.
  function traits(caps, g, age, life) {
    const grow = 0.55 + 0.45 * Math.min(1, age / (0.18 * life));
    const s = g.size * grow, m = s * s, has = (c) => caps.has(c);
    const drag = 0.6 + 0.4 * m + 0.5 * g.shell, k = 1 / Math.sqrt(drag);
    const lift = g.wings / (0.5 + 0.35 * m + 0.4 * g.shell);
    const fly = has('fly') && lift >= 0.9 ? 2.2 * g.wings * k : 0;
    const swim = has('swim') ? 1.5 * g.fins * k : 0;
    const walk = has('walk') ? 1.3 * g.legs * k : 0;
    const climb = has('climb') ? 1.0 * g.claws * k : 0;
    const spd = [fly, Math.max(climb, 0.6 * fly), Math.max(walk, 0.45 * fly, 0.45 * climb, has('fly') ? 0.25 : 0), swim];
    let limbs = 0;
    for (const l of LIMBS) limbs += g[l];
    return {
      s, m, r: 3 + 4 * s, spd, lift,
      bite: has('bite') ? 10 * g.jaws * s : 0, claw: has('claw') ? 7 * g.claws * s : 0, horn: has('horn') ? 8 * g.horns * s : 0,
      venom: has('sting') ? 3 * g.stinger * s : 0,
      armor: has('shell') ? Math.min(0.75, 0.45 * g.shell) : 0, thorn: has('spines') ? 1.8 * g.spines * s : 0,
      camo: has('camo') ? 1 + 1.5 * g.skin : 1,
      sense: 80 + 70 * g.eyes + 10 * s,
      emax: 100 * m, hpmax: 30 * m * (1 + 0.6 * g.shell),
      metab: 100 * m * 0.0007 * (1 + 0.1 * limbs + 0.15 * g.eyes),
      eat: 0.04 * (0.6 + 0.4 * s) * (1 + 0.3 * g.jaws),
    };
  }
  const weaponOf = (t) => t.bite + t.claw + t.horn + t.venom > 0;
  // A can eat B: a weapon, another species, and B not too big for it.
  function canEat(a, b) {
    if (a.sp === b.sp || !weaponOf(a.t)) return false;
    return b.t.m <= a.t.m * (a.t.venom > 0 ? 2.0 : 1.3) && b.t.m >= a.t.m * 0.04;
  }

  // ---------------------------------------------------------------- the park
  function createPark(data, opts = {}) {
    const C = data.consts;
    const T = Object.assign({}, DEFAULTS, opts.tune || {}); // per park, so a restart starts clean
    const NOPT = C.options.length;
    const brain = brainShape(opts.brain || data.brain);
    const rnd = makeRng(opts.seed || 1);
    let learn = opts.learn !== false;
    const gauss = () => Math.sqrt(-2 * Math.log(1 - rnd())) * Math.cos(2 * Math.PI * rnd());
    const species = makeSpecies(rnd, opts.species || 30, opts.diets || 'mixed');
    const names = new Set(species.map((sp) => sp.name));
    const events = []; // [tick, text]: new species, capability changes that spread
    // Walls the observer draws: { id, x, y0, y1, until }. Nothing crosses or sees through.
    const walls = [];
    let nextWall = 1, aliveN = 0;
    const moving = opts.movingPatches !== false;
    const yumon = [];
    const byId = new Map();
    // The model cache: id -> { w, anchor }. One entry per living Yumon.
    const models = new Map();
    const food = [];
    let nextId = 1;
    const stats = {
      tick: 0, born: 0, died: 0, causes: { eaten: 0, fought: 0, starved: 0, old: 0, poison: 0, thorns: 0 }, sleeps: 0, trainSteps: 0, trainMs: 0,
      hunts: 0, attacks: 0, kills: 0, fights: 0, braces: 0, escapes: 0, gained: 0, lost: 0, splits: 0, reformed: 0, decisions: 0, written: 0, untrained: 0, exactMiss: 0, drift: 0, driftN: 0,
    };
    const history = []; // [tick, pops per species]
    const changes = new Map(); // "rule->net" option pairs from learned brains
    const p = new Float64Array(NOPT);

    const wrapX = (x) => ((x % WORLD_W) + WORLD_W) % WORLD_W;
    const dxw = (a, b) => { let d = b - a; if (d > WORLD_W / 2) d -= WORLD_W; else if (d < -WORLD_W / 2) d += WORLD_W; return d; };

    // ---------------------------------------------------------------- plants
    function addFood(kind, x, y, cap, regrow, drift) {
      food.push({ kind, x, y, amt: cap * (0.4 + 0.6 * rnd()), cap, regrow, vx: drift ? (rnd() - 0.5) * 0.3 : 0, band: bandOf(y) });
    }
    // Plants grow in patches with sparse stretches between them: meadows, fruit groves,
    // drifting plankton blooms and gnat swarms. A patch eaten down sends its eaters on.
    // Totals (items x regrowth) match an even carpet: grass 1.3, fruit 0.72, plankton
    // 0.84, gnats 0.5 units per tick.
    const groups = [];
    // A patch withers when it is eaten down (below wither of its full amount) or its
    // season ends, and sprouts small somewhere new in its band, so eaters must follow it.
    const patchList = [];
    function patches(kind, n, per, cap, total, spanLo, spanHi, y0, y1, drift) {
      const regrow = (T.plantRate * total) / (n * per);
      for (let k = 0; k < n; k++) {
        const P = { kind, items: [], cx: (k + 0.15 + 0.7 * rnd()) * (WORLD_W / n), span: 0, y0, y1, spanLo, spanHi, grp: null, until: 0, moves: 0 };
        if (drift) { P.grp = { vx: (rnd() - 0.5) * 0.3, cy: 0 }; groups.push(P.grp); }
        for (let i = 0; i < per; i++) {
          addFood(kind, 0, y0, cap, regrow, false);
          const f = food[food.length - 1];
          f.grp = P.grp; f.patch = P;
          P.items.push(f);
        }
        placePatch(P, P.cx, 0.4 + 0.6 * rnd());
        P.until = stats.tick + T.season[0] * rnd() + T.season[0]; // stagger the first seasons
      }
    }
    function placePatch(P, cx, fill) {
      P.cx = wrapX(cx);
      P.span = P.spanLo + rnd() * (P.spanHi - P.spanLo);
      if (P.grp) P.grp.cy = P.y0 + rnd() * (P.y1 - P.y0);
      for (const f of P.items) {
        f.x = wrapX(P.cx + (rnd() - 0.5) * P.span);
        f.y = P.grp ? Math.min(P.y1, Math.max(P.y0, P.grp.cy + (rnd() - 0.5) * 50)) : P.y0 + rnd() * (P.y1 - P.y0);
        f.band = bandOf(f.y);
        f.amt = f.cap * fill;
      }
      P.born = stats.tick;
      P.until = stats.tick + T.season[0] + rnd() * (T.season[1] - T.season[0]);
    }
    function reformPatches() {
      for (const P of patchList) {
        let amt = 0, cap = 0;
        for (const f of P.items) { amt += f.amt; cap += f.cap; }
        if (P.grp) P.cx = wrapX(P.cx + P.grp.vx * 25);
        if (stats.tick < P.until && (amt > T.wither * cap || stats.tick - P.born < T.grace)) continue;
        const [lo, hi] = T.reformAway;
        const cx = P.cx + (rnd() < 0.5 ? -1 : 1) * (lo + rnd() * (hi - lo));
        placePatch(P, cx, T.sprout);
        P.moves++; stats.reformed++;
      }
    }
    patches('grass', 8, 13, 6, 1.3, 120, 240, 340, 395, false);
    patches('fruit', 7, 9, 5, 0.72, 90, 180, 205, 315, false);
    patches('plankton', 7, 10, 5, 0.84, 60, 120, 410, 590, true);
    patches('gnats', 6, 8, 4, 0.5, 50, 100, 15, 180, true);
    for (const f of food) if (f.patch && !patchList.includes(f.patch)) patchList.push(f.patch);

    // ---------------------------------------------------------------- yumon
    function spawn(sp, genes, x, y, mother, father, caps) {
      const life = 3500 * (0.7 + 0.3 * genes.size) * (0.85 + 0.3 * rnd());
      const y0 = { id: nextId++, sp: sp.id, s: sp, sex: rnd() < 0.5 ? 0 : 1, genes, x: wrapX(x), y, vx: 0, vy: 0, age: 0, life, alive: true,
        option: 7, sleeping: false, fatigue: rnd() * 4, poison: 0, mateCd: 300, preg: null, atkCd: 0, wander: rnd() * Math.PI * 2,
        gen: mother ? Math.max(mother.gen, father ? father.gen : 0) + 1 : 0, learned: mother ? mother.learned : 0, sleeps: 0,
        learns: mother ? mother.learns : opts.mixed ? rnd() < 0.5 : true, buf: [], r: 0, keys: null, rule: null, exact: true, lastEat: 0, kills: 0, children: 0, mom: mother ? mother.id : 0 };
      y0.caps = caps || new Set(sp.caps);
      y0.diet = dietOf(y0.caps);
      y0.hitBy = 0; y0.hitAt = -1e9; y0.attacker = null; y0.wasHit = 0; y0.bracing = false; y0.blockedAt = -1;
      y0.t = traits(y0.caps, genes, 0, life);
      y0.energy = y0.t.emax * (mother ? 0.7 : 0.8);
      y0.hp = y0.t.hpmax;
      // The child's brain is a copy of its mother's at birth; founders get the compiled weights.
      const w = mother ? models.get(mother.id).w.slice() : brain.init.slice();
      models.set(y0.id, { w, anchor: w.slice() });
      y0.exact = !mother || mother.exact;
      placeIn(y0);
      yumon.push(y0);
      byId.set(y0.id, y0);
      sp.alive++; sp.born++; sp.maxGen = Math.max(sp.maxGen, y0.gen);
      return y0;
    }
    function maskOf(t) { return t.spd.map((v) => v > 0); }
    function placeIn(a) {
      const m = maskOf(a.t);
      if (m[bandOf(a.y)]) return;
      const ok = [0, 1, 2, 3].filter((b) => m[b]);
      const b = ok[Math.floor(rnd() * ok.length)];
      a.y = BAND_Y[b][0] + 6 + rnd() * (BAND_Y[b][1] - BAND_Y[b][0] - 12);
    }
    function homeBand(t) { let b = 0; for (let k = 1; k < 4; k++) if (t.spd[k] > t.spd[b]) b = k; return b; }
    // Founders arrive in small mixed-sex groups of 2-3, each group somewhere else in the
    // park, so species start spread out and still find mates.
    function founders(sp, n) {
      const t = traits(sp.caps, sp.genes, 3000, 3500);
      const hb = homeBand(t);
      let cx = 0, left = 0;
      for (let i = 0; i < n; i++) {
        // Hunters start as packs of 4 so they still find mates while they chase prey.
        if (left === 0) { cx = rnd() * WORLD_W; left = sp.carn ? Math.min(n - i, 4) : n - i === 4 ? 2 : Math.min(n - i, 2 + (rnd() < 0.4 ? 1 : 0)); }
        left--;
        const g = { ...sp.genes };
        for (const k of Object.keys(g)) if (k !== 'hue' && g[k] > 0) g[k] *= Math.exp(0.05 * gauss());
        const y = BAND_Y[hb][0] + 6 + rnd() * (BAND_Y[hb][1] - BAND_Y[hb][0] - 12);
        const a = spawn(sp, g, cx + (rnd() - 0.5) * 60, y, null, null);
        a.sex = i % 2;
        a.age = a.life * (0.2 + 0.3 * rnd());
        a.t = traits(a.caps, a.genes, a.age, a.life);
        a.energy = a.t.emax * 0.8; a.hp = a.t.hpmax;
      }
    }
    for (const sp of species) {
      const t = traits(sp.caps, sp.genes, 3000, 3500);
      sp.adult = t;
      const n = sp.role === 'herb' ? Math.max(9, Math.min(24, 21 / t.m)) : sp.role === 'omni' ? 8 : 8;
      founders(sp, Math.round(n));
    }

    // ---------------------------------------------------------------- spatial hash
    const CELL = 100, GX = WORLD_W / CELL, GY = Math.ceil(WORLD_H / CELL);
    let grid = [], fgrid = [];
    function rebuild() {
      grid = Array.from({ length: GX * GY }, () => []);
      fgrid = Array.from({ length: GX * GY }, () => []);
      for (const a of yumon) if (a.alive) grid[Math.floor(a.x / CELL) % GX + GX * Math.min(GY - 1, Math.floor(a.y / CELL))].push(a);
      for (const f of food) if (f.amt >= 0.3) fgrid[Math.floor(wrapX(f.x) / CELL) % GX + GX * Math.min(GY - 1, Math.floor(f.y / CELL))].push(f);
    }
    function near(g, x, y, R, fn) {
      const cx = Math.floor(x / CELL), cy = Math.floor(y / CELL), n = Math.ceil(R / CELL);
      for (let j = Math.max(0, cy - n); j <= Math.min(GY - 1, cy + n); j++)
        for (let i = cx - n; i <= cx + n; i++) for (const o of g[((i % GX) + GX) % GX + GX * j]) fn(o);
    }
    const bucket = (d, R, contact) => (d > R ? 0 : d <= contact ? 3 : d <= 0.4 * R ? 2 : 1);

    // ---------------------------------------------------------------- perception
    function perceive(a) {
      const t = a.t, mask = maskOf(t), R = t.sense * (a.sleeping ? 0.35 : 1), myBand = bandOf(a.y);
      const fertile = isFertile(a);
      let threat = null, td = Infinity, prey = null, pdist = Infinity, mate = null, mdist = Infinity, kin = 0, kx = 0, ky = 0;
      // Mating calls carry farther than sight: a fertile Yumon hears fertile mates of its
      // species out to call x its sense radius.
      const RC = fertile && !a.sleeping ? R * T.call : R;
      near(grid, a.x, a.y, RC, (o) => {
        if (o === a || !o.alive) return;
        const dx = dxw(a.x, o.x), dy = o.y - a.y, d = Math.hypot(dx, dy);
        if (walls.length && walled(a.x, a.y, dx, dy)) return;
        if (d > R) {
          if (d <= RC && o.sp === a.sp && o.sex !== a.sex && d < mdist && isFertile(o)) { mate = o; mdist = d; }
          return;
        }
        if (o.sp === a.sp) {
          if (d < 0.6 * R) { kin++; kx += dx; ky += dy; }
          if (fertile && o.sex !== a.sex && isFertile(o) && d < mdist) { mate = o; mdist = d; }
          return;
        }
        // Still camouflaged Yumon are seen only from closer.
        const seen = (o.option === 3 || o.sleeping) ? d * o.t.camo : d;
        if (seen > R) return;
        if (canEat(o, a) && maskOf(o.t)[myBand] && seen < td) { threat = o; td = seen; }
        if (canEat(a, o) && mask[bandOf(o.y)] && seen < pdist) { prey = o; pdist = seen; }
      });
      let fdist = Infinity, fd = null;
      near(fgrid, a.x, a.y, R, (f) => {
        if (f.amt < 0.3 || !a.diet.has(f.kind) || !mask[f.band]) return;
        const dx = dxw(a.x, f.x), d = Math.hypot(dx, f.y - a.y);
        if (d >= fdist || (walls.length && walled(a.x, a.y, dx, f.y - a.y))) return;
        if (d < fdist) { fdist = d; fd = f; }
      });
      let refuge = 0;
      if (threat) { const tm = maskOf(threat.t); for (let b = 0; b < 4; b++) if (mask[b] && !tm[b]) refuge = 1; }
      a.threat = threat; a.prey = prey; a.mate = mate; a.food = fd; a.kin = kin ? [kx / kin, ky / kin] : null;
      // The attacker: who struck last, if it is alive and still in sight.
      let hit = 0, odds = 0;
      const at = a.hitBy ? byId.get(a.hitBy) : null;
      a.attacker = null;
      if (at && at.alive) {
        const dx = dxw(a.x, at.x), dy = at.y - a.y, ago = stats.tick - a.hitAt;
        if (Math.hypot(dx, dy) <= R && !(walls.length && walled(a.x, a.y, dx, dy))) {
          hit = ago <= 12 ? 3 : ago <= 40 ? 2 : ago <= 100 ? 1 : 0;
          if (hit) {
            a.attacker = at;
            // Odds: how long it takes me to finish it against how long it takes it to finish me.
            const mine = at.hp / Math.max(1e-6, blow(a, at)), theirs = a.hp / Math.max(1e-6, blow(at, a));
            odds = !weaponOf(t) ? 0 : theirs >= 1.5 * mine ? 2 : theirs >= 0.67 * mine ? 1 : 0;
          }
        }
      }
      if (a.wasHit && !hit && a.hitBy) { a.r += T.escapeR; stats.escapes++; a.hitBy = 0; } // got away alive
      a.wasHit = hit;
      const contact = (o) => t.r + (o ? o.t.r : 4) + 4;
      const hunger = Math.max(0, Math.min(C.level, Math.round(C.level * (1 - a.energy / t.emax))));
      const O = NOPT, x = new Array(O + 16).fill(0);
      x[a.option] = 1;
      x[O] = hunger;
      x[O + 1] = Math.min(C.level, Math.floor(a.fatigue));
      x[O + 2] = Math.min(C.hurt, Math.floor(4 * (1 - a.hp / t.hpmax)));
      x[O + 3] = fertile ? 1 : 0;
      x[O + 4] = threat ? bucket(td, R, contact(threat) + 20) : 0;
      x[O + 5] = refuge;
      x[O + 6] = fd ? bucket(fdist, R, t.r + 6) : 0;
      x[O + 7] = prey ? bucket(pdist, R, contact(prey)) : 0;
      x[O + 8] = mate ? bucket(mdist, RC, contact(mate)) : 0;
      x[O + 9] = Math.min(3, kin);
      x[O + 10] = a.caps.has('camo') ? 1 : 0;
      x[O + 11] = a.caps.has('social') ? 1 : 0;
      x[O + 12] = hit;
      x[O + 13] = odds;
      x[O + 14] = weaponOf(t) ? 1 : 0;
      x[O + 15] = t.armor > 0 || t.thorn > 0 ? 1 : 0;
      return x;
    }
    // The reward is homeostatic: what a tick does to a convex need over hunger, fatigue
    // and hurt (Keramati and Gutkin's drive reduction), plus a bonus for mating and birth.
    function need(a) {
      const hu = 1 - Math.max(0, a.energy) / a.t.emax, fa = a.fatigue / C.level, hr = 1 - Math.max(0, a.hp) / a.t.hpmax;
      return 4 * hu * hu + 2 * fa * fa + 6 * hr * hr;
    }
    function isFertile(a) {
      return a.age > 0.18 * a.life && !a.preg && a.mateCd <= 0 && a.energy > 0.55 * a.t.emax && aliveN < T.popCap;
    }

    // ---------------------------------------------------------------- learning
    // REINFORCE with a per-buffer baseline, an anchor pulling back toward the weights the
    // Yumon was born with, and a clipped step.
    function train(a) {
      const buf = a.buf;
      if (buf.length < T.minBuf) return;
      const t0 = (typeof performance !== 'undefined' ? performance : Date).now();
      const n = buf.length, G = new Float64Array(n);
      let run = 0;
      for (let i = n - 1; i >= 0; i--) { run = buf[i].r + T.gamma * run; G[i] = run; }
      let mean = 0; for (let i = 0; i < n; i++) mean += G[i]; mean /= n;
      let v = 0; for (let i = 0; i < n; i++) v += (G[i] - mean) ** 2;
      const sd = Math.sqrt(v / n);
      if (sd < 1e-4) { buf.length = 0; return; }
      const M = models.get(a.id), w = M.w, g = new Float64Array(brain.size), dOut = new Float64Array(NOPT);
      for (let e = 0; e < T.epochs; e++) {
        g.fill(0);
        for (let i = 0; i < n; i++) {
          const A = (G[i] - mean) / sd;
          const keys = brain.forward(w, buf[i].x);
          softmax(keys, T.temp, p);
          for (let k = 0; k < NOPT; k++) dOut[k] = (A * ((k === buf[i].a ? 1 : 0) - p[k])) / T.temp / n;
          brain.backward(w, dOut, g);
        }
        let norm = 0;
        for (let i = 0; i < g.length; i++) norm += g[i] * g[i];
        norm = Math.sqrt(norm);
        const sc = norm > T.clip ? T.clip / norm : 1;
        for (let i = 0; i < g.length; i++) w[i] += T.lr * (sc * g[i] - T.anchor * (w[i] - M.anchor[i]));
      }
      a.exact = false;
      a.learned++; a.sleeps++;
      stats.sleeps++; stats.trainSteps += T.epochs;
      stats.trainMs += (typeof performance !== 'undefined' ? performance : Date).now() - t0;
      buf.length = 0;
    }

    // ---------------------------------------------------------------- one tick
    function die(a, cause) {
      if (!a.alive) return;
      a.alive = false;
      a.cause = cause;
      stats.died++; stats.causes[cause]++;
      a.s.alive--;
      if (a.s.alive === 0) a.s.extinctAt = stats.tick;
      models.delete(a.id); // the model cache keeps only the living
      a.buf = [];
      const fy = bandOf(a.y) === 3 ? a.y : Math.min(395, Math.max(a.y, 340 + rnd() * 50));
      const body = { kind: 'meat', x: a.x, y: fy, amt: 4 * a.t.m + 1, cap: 0, regrow: -0.004, vx: 0, band: bandOf(fy) };
      food.push(body);
      return body;
    }

    function steer(a, tx, ty, frac) {
      const dx = dxw(a.x, tx), dy = ty - a.y, d = Math.hypot(dx, dy) || 1;
      const sp = T.speed * a.t.spd[bandOf(a.y)] * frac * (1 - 0.06 * Math.max(0, a.fatigue - 5)); // tired is slow
      a.dvx = (dx / d) * sp; a.dvy = (dy / d) * sp;
    }

    function tick() {
      stats.tick++;
      for (let i = walls.length - 1; i >= 0; i--) if (walls[i].until && walls[i].until <= stats.tick) walls.splice(i, 1);
      if (stats.tick % T.splitEvery === 0) speciate();
      rebuild();
      const alive = yumon.filter((a) => a.alive);
      aliveN = alive.length;
      for (const a of alive) {
        if (!a.alive) continue; // killed earlier this tick
        a.need0 = need(a);
        const x = perceive(a);
        const M = models.get(a.id);
        const keys = Array.from(brain.forward(M.w, x));
        if (opts.onDecide) opts.onDecide(a, x, keys);
        const rule = ruleKeys(x, C);
        a.keys = keys; a.rule = rule; a.x_in = x;
        const rb = argmax(rule), nb = argmax(keys);
        if (a.exact) {
          stats.untrained++;
          let bad = false;
          for (let k = 0; k < NOPT; k++) if (Math.abs(keys[k] - rule[k]) > 1e-6) bad = true;
          if (bad) stats.exactMiss++;
        } else {
          stats.driftN++; a.s.dN = (a.s.dN || 0) + 1;
          if (rb !== nb) {
            stats.drift++; a.s.dD = (a.s.dD || 0) + 1;
            const key = C.options[rb] + '>' + C.options[nb];
            changes.set(key, (changes.get(key) || 0) + 1);
          }
        }
        a.ruleBest = rb; a.netBest = nb;
        let opt;
        softmax(keys, T.temp, p);
        if (a.fatigue >= C.level && !a.attacker) { opt = 5; a.commit = 0; } // exhausted: falls asleep, not a decision
        else if (a.commit > 0 && p[a.option] >= T.drop) { opt = a.option; a.commit--; } // still committed
        else {
          a.commit = T.commit;
          let u = rnd(); opt = NOPT - 1;
          for (let k = 0; k < NOPT; k++) { u -= p[k]; if (u <= 0) { opt = k; break; } }
          stats.decisions++;
          if (opt === rb) stats.written++;
          if (learn && a.learns) {
            a.buf.push({ x: Int8Array.from(x), a: opt, r: 0 });
            if (a.buf.length > T.maxBuf) a.buf.shift();
          }
        }
        const wasAsleep = a.sleeping;
        a.option = opt;
        a.sleeping = opt === 5;
        if (a.sleeping && !wasAsleep && learn && a.learns) train(a);
        act(a);
      }
      for (const a of alive) if (a.alive) move(a);
      for (const a of alive) if (a.alive) body(a);
      for (const a of alive) if (a.alive && a.buf.length) { a.buf[a.buf.length - 1].r += a.r + T.needScale * (a.need0 - need(a)); }
      for (const a of alive) a.r = 0;
      plants();
      if (stats.tick % 25 === 0) {
        for (let i = yumon.length - 1; i >= 0; i--) if (!yumon[i].alive) { byId.delete(yumon[i].id); yumon.splice(i, 1); }
        history.push([stats.tick, species.map((s) => s.alive)]);
        if (history.length > 2000) history.splice(0, history.length - 2000);
      }
    }

    function act(a) {
      a.dvx = 0; a.dvy = 0; a.bracing = false;
      const t = a.t;
      switch (a.option) {
        case 0: { // eat
          const f = a.food;
          if (!f) break;
          const d = Math.hypot(dxw(a.x, f.x), f.y - a.y);
          if (d > t.r + 6) steer(a, f.x, f.y, 0.8);
          else if (a.energy < t.emax) {
            const u = Math.min(f.amt, t.eat), e = Math.min(u * T.foodE, t.emax - a.energy);
            f.amt -= u; a.energy += e;
          }
          break;
        }
        case 1: { // hunt
          const o = a.prey;
          if (!o || !o.alive) break;
          stats.hunts++;
          const d = Math.hypot(dxw(a.x, o.x), o.y - a.y);
          // A pounce: faster than a run, and it tires.
          if (d > t.r + o.t.r + 4) { steer(a, o.x, o.y, 1.4); a.fatigue += 0.02; }
          else if (a.atkCd <= 0) attack(a, o);
          break;
        }
        case 2: { // flee
          const o = a.threat;
          if (!o) break;
          let tx = a.x - dxw(a.x, o.x) * 3, ty = a.y - (o.y - a.y) * 3;
          const tm = maskOf(o.t), m = maskOf(t);
          let best = -1, bd = Infinity;
          for (let b = 0; b < 4; b++) if (m[b] && !tm[b]) { const cy = (BAND_Y[b][0] + BAND_Y[b][1]) / 2, dd = Math.abs(cy - a.y); if (dd < bd) { bd = dd; best = b; } }
          if (best >= 0) ty = (BAND_Y[best][0] + BAND_Y[best][1]) / 2;
          steer(a, tx, ty, 1);
          break;
        }
        case 3: break; // hide: hold still (camouflage works only while still)
        case 4: { // mate
          const o = a.mate;
          if (!o || !o.alive) break;
          const d = Math.hypot(dxw(a.x, o.x), o.y - a.y);
          if (d > t.r + o.t.r + 4) steer(a, o.x, o.y, 0.8);
          else if (isFertile(a) && isFertile(o)) conceive(a.sex === 0 ? a : o, a.sex === 0 ? o : a);
          break;
        }
        case 5: break; // sleep
        case 6: { // flock
          if (!a.kin) { roam(a); break; }
          steer(a, a.x + a.kin[0], a.y + a.kin[1], 0.6);
          break;
        }
        case 8: { // fight: turn on the attacker
          const o = a.attacker;
          if (!o || !o.alive) break;
          const d = Math.hypot(dxw(a.x, o.x), o.y - a.y);
          if (d > t.r + o.t.r + 4) { steer(a, o.x, o.y, 1.2); a.fatigue += 0.015; }
          else if (a.atkCd <= 0) { stats.fights++; attack(a, o, true); }
          break;
        }
        case 9: a.bracing = true; stats.braces++; break; // brace: hold still behind shell or spines
        default: roam(a);
      }
    }

    // Roaming heads for a far goal and keeps it until it gets there or gives up, so a
    // Yumon with nothing to do travels instead of milling about.
    function roam(a) {
      const t = a.t, g = a.goal, m = maskOf(t);
      if (g && a.blockedAt >= stats.tick - 1) g.dir = -g.dir, g.until = 0; // walked into a wall: turn back
      if (!g || stats.tick > g.until || !m[bandOf(g.y)] || Math.hypot(dxw(a.x, g.x), g.y - a.y) < 25) {
        const [lo, hi] = T.roamFar, dist = lo + rnd() * (hi - lo);
        const dir = g && rnd() < 0.7 ? g.dir : rnd() < 0.5 ? -1 : 1; // mostly keep going the same way
        const ok = [0, 1, 2, 3].filter((b) => m[b]), hb = homeBand(t);
        const b = rnd() < 0.6 ? hb : ok[Math.floor(rnd() * ok.length)];
        a.goal = { x: wrapX(a.x + dir * dist), y: BAND_Y[b][0] + 8 + rnd() * (BAND_Y[b][1] - BAND_Y[b][0] - 16), dir, until: stats.tick + T.roamTicks };
      }
      a.wander += (rnd() - 0.5) * 0.2;
      steer(a, a.goal.x, a.goal.y + Math.sin(a.wander) * 12, T.roamPace);
    }

    // One blow from a to o. A braced shell takes less and braced spines hurt more.
    function blow(a, o) {
      const t = a.t, armor = o.bracing && o.t.armor > 0 ? Math.min(0.9, o.t.armor + 0.3) : o.t.armor;
      return (t.bite + t.claw + t.horn) * (1 - armor) + 0.3 * t.venom;
    }
    function attack(a, o, back) {
      const t = a.t;
      a.atkCd = 12;
      stats.attacks++;
      o.hp -= blow(a, o);
      o.poison += t.venom;
      o.hitBy = a.id; o.hitAt = stats.tick;
      a.r += T.hitR;
      if (o.t.thorn > 0) {
        a.hp -= o.t.thorn * (o.bracing ? 2 : 1);
        if (a.hp <= 0) { die(a, 'thorns'); o.r += T.killR; return; }
      }
      if (o.hp > 0) return;
      const body = die(o, back ? 'fought' : 'eaten');
      a.kills++; stats.kills++;
      a.r += T.killR;
      // The killer eats first: up to gorge of the body, as far as it has room.
      if (a.diet.has('meat') && body) {
        const e = Math.min(a.t.emax - a.energy, T.gorge * body.amt * T.foodE);
        if (e > 0) { a.energy += e; body.amt -= e / T.foodE; }
      }
    }

    function conceive(mom, dad) {
      mom.preg = { dad: dad.id, dadGenes: { ...dad.genes }, dadCaps: new Set(dad.caps), dadGen: dad.gen, at: stats.tick + 80 };
      mom.energy -= 0.25 * mom.t.emax;
      // Small Yumon breed sooner (r-strategists); big ones wait longer.
      mom.mateCd = T.breedCd * (0.45 + 0.55 * mom.genes.size); dad.mateCd = T.breedCd * (0.45 + 0.55 * dad.genes.size);
      mom.r += 1; dad.r += 1;
    }

    // A child's genome: each gene a random blend of the parents', a little noise on every
    // gene and one bigger step on one of them. Capabilities both parents carry pass on;
    // one parent's pass half the time. Now and then a capability is gained (likelier on a
    // limb it already has, so claws for climbing make clawing easy) or lost.
    function childGenome(mom, pr) {
      const a = mom.genes, b = pr.dadGenes, g = {};
      for (const k of Object.keys(a)) {
        const u = rnd();
        g[k] = k === 'hue' ? a.hue + (((b.hue - a.hue + 540) % 360) - 180) * u : a[k] * (1 - u) + b[k] * u;
      }
      const caps = new Set();
      for (const c of new Set([...mom.caps, ...pr.dadCaps])) if ((mom.caps.has(c) && pr.dadCaps.has(c)) || rnd() < 0.5) caps.add(c);
      if (![...caps].some((c) => CAP[c].kind === 'move')) caps.add([...mom.caps].find((c) => CAP[c].kind === 'move'));
      let event = null;
      if (rnd() < T.capRate) event = rnd() < 0.55 ? gainCap(caps, g) : loseCap(caps, g);
      fitLimbs(caps, g);
      const keys = Object.keys(g).filter((k) => k !== 'hue' && g[k] > 0);
      for (const k of keys) g[k] *= Math.exp(T.mutSmall * gauss());
      const k = keys[Math.floor(rnd() * keys.length)];
      g[k] *= Math.exp(T.mutBig * gauss());
      for (const k of keys) g[k] = Math.min(k === 'size' ? 2.6 : 2.5, Math.max(k === 'size' ? 0.4 : 0.15, g[k]));
      g.hue = (g.hue + T.hueStep * gauss() + 360) % 360;
      // A limb that has withered away takes its capabilities with it (never the last way to move).
      for (const c of [...caps]) {
        const L = CAP[c].limb;
        if (L && g[L] < 0.22 && !(CAP[c].kind === 'move' && moveCount(caps) === 1)) { caps.delete(c); event = event || '-' + c; }
      }
      fitLimbs(caps, g);
      return { g, caps, event, mutated: k };
    }
    const moveCount = (caps) => [...caps].filter((c) => CAP[c].kind === 'move').length;
    // Limbs follow capabilities: a limb no capability uses is gone, a new one starts small.
    function fitLimbs(caps, g) {
      const used = new Set([...caps].map((c) => CAP[c].limb).filter(Boolean));
      for (const L of LIMBS) {
        if (!used.has(L)) g[L] = 0;
        else if (g[L] < 0.3) g[L] = 0.3 + 0.15 * rnd();
      }
    }
    function gainCap(caps, g) {
      const pool = CAPS.filter((c) => !caps.has(c.id) && !(caps.size >= 9));
      if (!pool.length) return null;
      const wt = pool.map((c) => (c.limb && g[c.limb] > 0 ? 4 : 1));
      let u = rnd() * wt.reduce((x, y) => x + y, 0), c = pool[pool.length - 1];
      for (let i = 0; i < pool.length; i++) { u -= wt[i]; if (u <= 0) { c = pool[i]; break; } }
      caps.add(c.id);
      if (c.limb && !(g[c.limb] > 0)) g[c.limb] = 0.35 + 0.2 * rnd();
      return '+' + c.id;
    }
    function loseCap(caps, g) {
      const pool = [...caps].filter((c) => !(CAP[c].kind === 'move' && moveCount(caps) === 1));
      if (!pool.length) return null;
      const c = pool[Math.floor(rnd() * pool.length)];
      caps.delete(c);
      void g;
      return '-' + c;
    }

    function birth(mom) {
      const pr = mom.preg;
      mom.preg = null;
      // Small unarmed Yumon have twins.
      const n = Math.min(!weaponOf(mom.t) && mom.genes.size < T.twinSize ? 2 : 1, Math.max(0, T.popCap - aliveN));
      for (let i = 0; i < n; i++) {
        const G = childGenome(mom, pr);
        const c = spawn(mom.s, G.g, mom.x + (rnd() - 0.5) * 10, mom.y, mom, { gen: pr.dadGen }, G.caps);
        c.mutated = G.mutated;
        c.capEvent = G.event;
        if (G.event) stats[G.event[0] === '+' ? 'gained' : 'lost']++;
        stats.born++;
        mom.children++;
      }
      mom.r += 1;
    }

    // ---------------------------------------------------------------- new species
    // A group inside a species that carries another capability set, or a gene cluster far
    // from the rest, splits off once it has splitMin members of both sexes. After the split
    // the two can no longer mate. The species itself takes its majority's capabilities.
    function speciate() {
      for (const sp of species.slice()) {
        if (!sp.alive) continue;
        const live = yumon.filter((a) => a.alive && a.sp === sp.id);
        const groups = new Map();
        for (const a of live) { const k = sigOf(a.caps); if (!groups.has(k)) groups.set(k, []); groups.get(k).push(a); }
        const sorted = [...groups.values()].sort((x, y) => y.length - x.length);
        for (const grp of sorted.slice(1)) if (grp.length >= T.splitMin && bothSexes(grp)) split(sp, grp, 'caps');
        const rest = live.filter((a) => a.sp === sp.id);
        if (rest.length >= 2 * T.splitMin) {
          const [A, B, dist] = twoMeans(rest);
          const small = A.length < B.length ? A : B;
          if (dist > T.splitDist && A.length >= T.splitMin && B.length >= T.splitMin && bothSexes(A) && bothSexes(B)) split(sp, small, 'genes');
        }
        refresh(sp);
      }
    }
    const bothSexes = (grp) => grp.some((a) => a.sex === 0) && grp.some((a) => a.sex === 1);
    const geneVec = (g) => ['size', 'eyes', ...LIMBS].map((k) => Math.log(Math.max(0.1, g[k] || 0)));
    // Two clusters of log genes (farthest-point start, a few Lloyd steps); the distance
    // between their centres.
    function twoMeans(live) {
      const V = live.map((a) => geneVec(a.genes)), D = V[0].length;
      const dist = (u, v) => { let s = 0; for (let k = 0; k < D; k++) s += (u[k] - v[k]) ** 2; return Math.sqrt(s); };
      let c0 = V[0], c1 = V[0];
      for (const v of V) if (dist(v, c0) > dist(c1, c0)) c1 = v;
      for (const v of V) if (dist(v, c1) > dist(c0, c1)) c0 = v;
      let lab = [];
      for (let it = 0; it < 6; it++) {
        lab = V.map((v) => (dist(v, c0) <= dist(v, c1) ? 0 : 1));
        const m = [new Array(D).fill(0), new Array(D).fill(0)], n = [0, 0];
        V.forEach((v, i) => { n[lab[i]]++; for (let k = 0; k < D; k++) m[lab[i]][k] += v[k]; });
        if (!n[0] || !n[1]) return [live, [], 0];
        c0 = m[0].map((x) => x / n[0]); c1 = m[1].map((x) => x / n[1]);
      }
      return [live.filter((_, i) => lab[i] === 0), live.filter((_, i) => lab[i] === 1), dist(c0, c1)];
    }
    function meanGenes(grp) {
      const g = {};
      for (const k of Object.keys(grp[0].genes)) g[k] = grp.reduce((s, a) => s + a.genes[k], 0) / grp.length;
      g.hue = grp[0].genes.hue;
      return g;
    }
    function split(sp, grp, why) {
      if (species.length >= T.maxSpecies) return;
      const caps = new Set(grp[0].caps);
      const shift = (rnd() < 0.5 ? -1 : 1) * (35 + 50 * rnd()); // a new species looks new
      for (const a of grp) a.genes.hue = (a.genes.hue + shift + 360) % 360;
      const genes = meanGenes(grp);
      const ns = { id: species.length, name: nameFor(caps, rnd, names), role: roleOf(caps), caps, caps0: new Set(caps), diet: dietOf(caps), genes,
        carn: [...caps].some((c) => CAP[c].kind === 'weapon'), alive: 0, born: grp.length, extinctAt: null, maxGen: 0, parent: sp.id, since: stats.tick, why };
      ns.adult = traits(caps, genes, 3000, 3500);
      species.push(ns);
      for (const a of grp) { a.sp = ns.id; a.s = ns; sp.alive--; ns.alive++; ns.maxGen = Math.max(ns.maxGen, a.gen); }
      const diff = [...caps].filter((c) => !sp.caps.has(c)).map((c) => '+' + CAP[c].label).concat([...sp.caps].filter((c) => !caps.has(c)).map((c) => '-' + CAP[c].label));
      events.push([stats.tick, `${ns.name} split from ${sp.name} (${why === 'caps' ? diff.join(' ') : 'genes drifted apart'})`]);
      stats.splits++;
    }
    // A species shows its majority's capabilities and adult body.
    function refresh(sp) {
      const live = yumon.filter((a) => a.alive && a.sp === sp.id);
      if (!live.length) return;
      const n = new Map();
      for (const a of live) { const k = sigOf(a.caps); n.set(k, (n.get(k) || 0) + 1); }
      const top = [...n.entries()].sort((x, y) => y[1] - x[1])[0][0];
      const caps = new Set(live.find((a) => sigOf(a.caps) === top).caps);
      if (sigOf(caps) !== sigOf(sp.caps)) {
        const diff = [...caps].filter((c) => !sp.caps.has(c)).map((c) => '+' + CAP[c].label).concat([...sp.caps].filter((c) => !caps.has(c)).map((c) => '-' + CAP[c].label));
        events.push([stats.tick, `${sp.name} changed: ${diff.join(' ')}`]);
      }
      sp.caps = caps; sp.diet = dietOf(caps); sp.role = roleOf(caps); sp.carn = [...caps].some((c) => CAP[c].kind === 'weapon');
      sp.adult = traits(caps, meanGenes(live), 3000, 3500);
    }

    function move(a) {
      a.vx += (a.dvx - a.vx) * 0.3; a.vy += (a.dvy - a.vy) * 0.3;
      const x0 = a.x;
      a.x = wrapX(a.x + a.vx);
      let y = a.y + a.vy;
      const m = maskOf(a.t);
      let lo = 0, hi = 3;
      while (!m[lo]) lo++;
      while (!m[hi]) hi--;
      const ymin = BAND_Y[lo][0] + 4, ymax = BAND_Y[hi][1] - 4;
      if (y < ymin) y = Math.min(a.y + 3, ymin); // fell out of a band it can no longer hold (lost flight)
      if (y > ymax) y = ymax;
      if (a.y < ymin) y = Math.min(a.y + 3, ymax);
      a.y = y;
      if (walls.length) wallBlock(a, x0);
    }
    // A wall stops anything that would cross it or end up inside it, on the side it came from.
    function wallBlock(a, x0) {
      for (const w of walls) {
        if (a.y < w.y0 - a.t.r || a.y > w.y1 + a.t.r) continue;
        const pad = a.t.r + 2, o0 = dxw(x0, w.x), o1 = dxw(a.x, w.x);
        const side = o0 > 0 ? 1 : o0 < 0 ? -1 : a.vx > 0 ? 1 : -1; // 1: the wall is to its right
        if (Math.sign(o1) !== side || Math.abs(o1) < pad) { a.x = wrapX(w.x - side * pad); a.vx = 0; a.blockedAt = stats.tick; }
      }
    }
    // Does a wall stand between (x, y) and (x + dx, y + dy)?
    function walled(x, y, dx, dy) {
      if (dx === 0) return false;
      for (const w of walls) {
        const off = dxw(x, w.x);
        if (off * dx <= 0 || Math.abs(off) >= Math.abs(dx)) continue;
        const yy = y + (dy * off) / dx;
        if (yy >= w.y0 && yy <= w.y1) return true;
      }
      return false;
    }

    function body(a) {
      const t = a.t, speed = Math.hypot(a.vx, a.vy), top = Math.max(...t.spd) || 1;
      const frac = Math.min(1, speed / top), band = bandOf(a.y);
      a.energy -= t.metab + 0.0011 * t.emax * frac * (band === 0 ? 1.6 : 1);
      a.fatigue = a.sleeping ? Math.max(0, a.fatigue - 0.06) : Math.min(C.level, a.fatigue + 0.012 * (1 + frac));
      if (a.poison > 0.01) { a.hp -= 0.03 * a.poison; a.poison *= 0.985; if (a.hp <= 0) return die(a, 'poison'); }
      if (a.energy <= 0) { a.energy = 0; a.hp -= 0.02 * t.hpmax; if (a.hp <= 0) return die(a, 'starved'); }
      else if (a.hp < t.hpmax) a.hp = Math.min(t.hpmax, a.hp + 0.002 * t.hpmax);
      a.atkCd--; a.mateCd--;
      a.age++;
      if (a.age > a.life) return die(a, 'old');
      if (a.age < 0.2 * a.life && a.age % 50 === 0) {
        const f = a.energy / t.emax, h = a.hp / t.hpmax;
        a.t = traits(a.caps, a.genes, a.age, a.life);
        a.energy = f * a.t.emax; a.hp = h * a.t.hpmax;
      }
      if (a.preg && stats.tick >= a.preg.at) birth(a);
    }

    function plants() {
      if (moving && stats.tick % 25 === 0) reformPatches();
      for (const g of groups) if (rnd() < 0.002) g.vx = (rnd() - 0.5) * 0.3;
      for (let i = food.length - 1; i >= 0; i--) {
        const f = food[i];
        if (f.kind === 'meat') { f.amt += f.regrow * (1 + f.amt * 0.2); if (f.amt <= 0.05) food.splice(i, 1); continue; }
        f.amt = Math.min(f.cap, f.amt + f.regrow);
        if (f.vx) { f.x = wrapX(f.x + f.vx); if (rnd() < 0.002) f.vx = (rnd() - 0.5) * 0.3; }
        if (f.grp) f.x = wrapX(f.x + f.grp.vx);
      }
    }

    // ---------------------------------------------------------------- checks and views
    function verifyPort() {
      let bad = 0, netBad = 0;
      for (const [x, want] of data.vectors) {
        const got = ruleKeys(x, C);
        if (got.some((v, k) => v !== want[k])) bad++;
        const keys = brain.forward(brain.init, x);
        for (let k = 0; k < NOPT; k++) if (Math.abs(keys[k] - want[k]) > 1e-6) { netBad++; break; }
      }
      return { vectors: data.vectors.length, ruleMismatches: bad, netMismatches: netBad };
    }
    function foodWeb() {
      // Adult means: who can eat whom where their bands overlap.
      const live = species.map((s) => ({ sp: s.id, t: s.adult }));
      const web = [];
      for (const a of live) for (const b of live) {
        if (a.sp === b.sp) continue;
        const ma = maskOf(a.t), mb = maskOf(b.t);
        if (canEat(a, b) && ma.some((v, k) => v && mb[k])) web.push([a.sp, b.sp]);
      }
      return web;
    }
    function speciesStats(sp) {
      const live = yumon.filter((a) => a.alive && a.sp === sp.id);
      const mean = (f) => (live.length ? live.reduce((s, a) => s + f(a), 0) / live.length : 0);
      const genes = {};
      for (const k of Object.keys(sp.genes)) if (k !== 'hue' && sp.genes[k] > 0) genes[k] = mean((a) => a.genes[k]);
      return { n: live.length, genes, gen: mean((a) => a.gen), learned: mean((a) => a.learned) };
    }
    function brainDelta(a) {
      const M = models.get(a.id);
      if (!M) return 0;
      let s = 0; for (let i = 0; i < M.w.length; i++) s += Math.abs(M.w[i] - brain.init[i]);
      return s;
    }

    return {
      WORLD_W, WORLD_H, BANDS, BAND_Y, CAPS, LIMBS, T, species, yumon, food, stats, history, changes, models, brain,
      tick, verifyPort, foodWeb, speciesStats, brainDelta, bandOf, traits, canEat, maskOf,
      feed(x, y) {
        const b = bandOf(y), kind = ['gnats', 'fruit', 'grass', 'plankton'][b];
        for (let i = 0; i < 4; i++) addFood(kind, wrapX(x + (rnd() - 0.5) * 30), Math.min(BAND_Y[b][1] - 3, Math.max(BAND_Y[b][0] + 3, y + (rnd() - 0.5) * 20)), [4, 5, 6, 5][b], [0.01, 0.012, 0.013, 0.012][b], b === 0 || b === 3);
        return kind;
      },
      reintroduce(sp) { if (sp.alive === 0) { sp.extinctAt = null; founders(sp, 6); } },
      walls, events,
      addWall(x, y0, y1) {
        const w = { id: nextWall++, x: wrapX(x), y0: Math.max(0, Math.min(y0, y1)), y1: Math.min(WORLD_H, Math.max(y0, y1)), until: T.wallLife ? stats.tick + T.wallLife : 0 };
        walls.push(w);
        return w;
      },
      removeWall(x, y, R) {
        let best = -1, bd = R;
        walls.forEach((w, i) => { const d = Math.abs(dxw(x, w.x)) + (y < w.y0 ? w.y0 - y : y > w.y1 ? y - w.y1 : 0); if (d < bd) { bd = d; best = i; } });
        return best >= 0 ? walls.splice(best, 1)[0] : null;
      },
      clearWalls() { walls.length = 0; },
      setLearn(v) { learn = !!v; },
      get learn() { return learn; },
      byId,
    };
  }

  const api = { createPark, makeSpecies, traits, dietOf, CAPS, LIMBS, BANDS };
  if (typeof module !== 'undefined' && module.exports) module.exports = { ParkCore: api };
  else root.ParkCore = api;
})(typeof window !== 'undefined' ? window : globalThis);
