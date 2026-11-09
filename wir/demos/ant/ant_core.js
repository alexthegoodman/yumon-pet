// Yumon Ant simulation core. Every ant runs the exported WIR sense and brain networks;
// the five distance fields run the exported 4-move path step. The host does the world:
// terrain, digging plans, gravity, heavy items, the queen, eggs, hunger and death.
(function (root) {
  'use strict';
  const CW = 32, H = 72;
  const DIR4 = [[0, 1], [-1, 0], [0, -1], [1, 0]]; // [dy, dx]: E N W S
  const AIR = 0, SOIL = 1, ROCK = 2, SAND = 3, PEB = 4;

  function makeNet(spec) {
    const layers = spec.layers.map((l) => ({
      rows: l.rows.map((r) => {
        const n = r.length / 2, idx = new Int32Array(n), w = new Float64Array(n);
        for (let i = 0; i < n; i++) { idx[i] = r[2 * i]; w[i] = r[2 * i + 1]; }
        return { idx, w };
      }),
      b: Float64Array.from(l.b),
    }));
    const width = Math.max(spec.taps ? spec.taps.length : spec.inputs.length, ...layers.map((l) => l.rows.length));
    let a = new Float64Array(width), b = new Float64Array(width);
    return {
      eval(x) {
        a.set(x);
        for (let k = 0; k < layers.length; k++) {
          const { rows, b: bias } = layers[k], last = k === layers.length - 1;
          for (let r = 0; r < rows.length; r++) {
            const { idx, w } = rows[r];
            let z = bias[r];
            for (let i = 0; i < idx.length; i++) z += w[i] * a[idx[i]];
            b[r] = last || z > 0 ? z : 0;
          }
          const t = a; a = b; b = t;
        }
        return a;
      },
    };
  }

  function createSim(data, opts = {}) {
    const C = data.consts;
    const CEIL = C.ceiling, HUN = C.hunger, STARV = C.starving, PECK = C.peckish, FAR = C.far;
    const KEEP = C.keep, KEEP_HOLD = C.keep_hold, BIG = C.big, HOLD_BONUS = C.hold_bonus, PULL = C.pull, RB = C.role_bonus;
    const pathNet = makeNet(data.path), senseNet = makeNet(data.sense), brainNet = makeNet(opts.brain || data.brain);
    const pathTaps = data.path.taps, senseTaps = data.sense.taps;
    const BI = {}; data.brain.inputs.forEach((n, i) => (BI[n] = i));
    const BO = {}; data.brain.outputs.forEach((n, i) => (BO[n] = i));
    const SO = {}; data.sense.outputs.forEach((n, i) => (SO[n] = i));
    const NB = data.brain.inputs.length;

    // ---------------------------------------------------------------- randomness
    let seed = (opts.seed >>> 0) || 1;
    function rnd() {
      seed = (seed + 0x6d2b79f5) >>> 0;
      let t = seed;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    const ri = (n) => Math.floor(rnd() * n);
    function h2(x, y, s) {
      let h = (Math.imul(x | 0, 374761393) + Math.imul(y | 0, 668265263) + Math.imul(s, 2246822519)) | 0;
      h = Math.imul(h ^ (h >>> 13), 1274126177);
      return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
    }
    function vnoise(x, s) {
      const i = Math.floor(x), f = x - i, u = f * f * (3 - 2 * f);
      return h2(i, 0, s) * (1 - u) + h2(i + 1, 0, s) * u;
    }
    function vnoise2(x, y, s) {
      const i = Math.floor(x), j = Math.floor(y), fx = x - i, fy = y - j;
      const ux = fx * fx * (3 - 2 * fx), uy = fy * fy * (3 - 2 * fy);
      const a = h2(i, j, s) * (1 - ux) + h2(i + 1, j, s) * ux;
      const b = h2(i, j + 1, s) * (1 - ux) + h2(i + 1, j + 1, s) * ux;
      return a * (1 - uy) + b * uy;
    }
    const SURF = 24;
    const surfaceAt = (x) => SURF + Math.round(7 * (vnoise(x / 52, 11) - 0.5) + 3 * (vnoise(x / 15, 23) - 0.5));

    // ---------------------------------------------------------------- world
    const key = (x, y) => x * H + y;
    const kx = (k) => Math.floor(k / H), ky = (k) => k - Math.floor(k / H) * H;
    const chunks = new Map();
    const items = new Map(); // cell key -> item
    const itemsById = new Map();
    let nextItem = 1;
    const stats = { tick: 0, born: 0, died: 0, starved: 0, oldAge: 0, dug: 0, foodIn: 0, sandOut: 0 };

    function addItem(kind, w, x, y) {
      if (y < 0 || y >= H || items.has(key(x, y)) || cellType(x, y) !== AIR) return null;
      const it = { id: nextItem++, kind, w, x, y, grip: new Set(), stuck: 0, best: CEIL, rest: 0 };
      items.set(key(x, y), it);
      itemsById.set(it.id, it);
      touched(x, y);
      return it;
    }
    function removeItem(it) {
      items.delete(key(it.x, it.y));
      itemsById.delete(it.id);
      for (const id of it.grip) { const a = antsById.get(id); if (a) a.grip = 0; }
      it.grip.clear();
      touched(it.x, it.y);
    }
    function moveItem(it, x, y) {
      items.delete(key(it.x, it.y));
      touched(it.x, it.y);
      it.x = x; it.y = y;
      items.set(key(x, y), it);
      touched(x, y);
    }

    function genChunk(ci) {
      const cells = new Uint8Array(CW * H);
      const c = { ci, cells, walk: null };
      chunks.set(ci, c);
      for (let lx = 0; lx < CW; lx++) {
        const x = ci * CW + lx, s = surfaceAt(x);
        for (let y = 0; y < H; y++) {
          let t = AIR;
          if (y >= H - 2) t = ROCK;
          else if (y >= s) {
            t = SOIL;
            if (y > s + 5 && vnoise2(x / 7, y / 4, 5) > 0.8) t = ROCK;
            else if (y > s + 2 && h2(x, y, 7) < 0.014) t = PEB;
          }
          cells[lx * H + y] = t;
        }
      }
      for (let lx = 0; lx < CW; lx++) {
        const x = ci * CW + lx;
        if (Math.abs(x) > 6 && h2(x, 0, 91) < 0.045) {
          const r = h2(x, 1, 92), w = r < 0.55 ? 1 : r < 0.75 ? 2 : r < 0.9 ? 3 : 5;
          const it = { id: nextItem++, kind: 'food', w, x, y: surfaceAt(x) - 1, grip: new Set(), stuck: 0, best: CEIL, rest: 0 };
          items.set(key(it.x, it.y), it);
          itemsById.set(it.id, it);
        }
      }
      return c;
    }
    const chunk = (ci) => chunks.get(ci) || genChunk(ci);
    function cellType(x, y) {
      if (y < 0) return AIR;
      if (y >= H) return ROCK;
      const ci = Math.floor(x / CW);
      return chunk(ci).cells[(x - ci * CW) * H + y];
    }
    function setCell(x, y, t) {
      if (y < 0 || y >= H) return;
      const ci = Math.floor(x / CW);
      chunk(ci).cells[(x - ci * CW) * H + y] = t;
      touched(x, y);
    }
    const solid = (x, y) => (y < 0 ? false : y >= H ? true : cellType(x, y) !== AIR || items.has(key(x, y)));
    function walkRaw(x, y) {
      if (y < 0 || y >= H || solid(x, y)) return 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) if ((dy || dx) && solid(x + dx, y + dy)) return 1;
      return 0;
    }
    function walkArr(c) {
      if (!c.walk) {
        c.walk = new Uint8Array(CW * H);
        for (let lx = 0; lx < CW; lx++) for (let y = 0; y < H; y++) c.walk[lx * H + y] = walkRaw(c.ci * CW + lx, y);
      }
      return c.walk;
    }
    function walk(x, y) {
      if (y < 0 || y >= H) return 0;
      const ci = Math.floor(x / CW);
      return walkArr(chunk(ci))[(x - ci * CW) * H + y];
    }
    // A cell changed: its 3 x 3 neighbourhood may change walkability.
    function touched(x, y) {
      for (let dx = -1; dx <= 1; dx++) {
        const X = x + dx, ci = Math.floor(X / CW), c = chunks.get(ci);
        if (!c || !c.walk) continue;
        for (let dy = -1; dy <= 1; dy++) {
          const Y = y + dy;
          if (Y >= 0 && Y < H) c.walk[(X - ci * CW) * H + Y] = walkRaw(X, Y);
        }
      }
      topCache.delete(x);
    }
    const topCache = new Map();
    function skyTop(x) {
      let t = topCache.get(x);
      if (t === undefined) {
        t = 0;
        while (t < H && !solid(x, t)) t++;
        topCache.set(x, t);
      }
      return t;
    }

    // ---------------------------------------------------------------- fields
    function makeField() { return new Map(); }
    function farr(f, ci) {
      let a = f.get(ci);
      if (!a) { a = new Uint8Array(CW * H).fill(CEIL); f.set(ci, a); }
      return a;
    }
    function fget(f, x, y) {
      if (y < 0 || y >= H || !walk(x, y)) return CEIL;
      const ci = Math.floor(x / CW), a = f.get(ci);
      return a ? a[(x - ci * CW) * H + y] : CEIL;
    }
    const shared = { df: makeField(), dh: makeField() };

    // ---------------------------------------------------------------- ants and colonies
    const ants = [];
    const antsById = new Map();
    const colonies = [];
    let nextAnt = 1;
    const crowd = new Map();
    const crowdAt = (x, y) => crowd.get(key(x, y)) || 0;
    function crowdAdd(x, y, n) {
      const k = key(x, y), v = (crowd.get(k) || 0) + n;
      if (v > 0) crowd.set(k, v); else crowd.delete(k);
    }
    const fear = new Map();
    const fearAt = (x, y) => fear.get(key(x, y)) || 0;

    const diggable = (t) => t === SOIL || t === SAND || t === PEB;
    const digTarget = (col, x, y) => col.plan.has(key(x, y)) && diggable(cellType(x, y));
    const lift = (x, y) => { const it = items.get(key(x, y)); return it && it.kind === 'food' && it.w === 1 ? 1 : 0; };
    const pull = (x, y) => { const it = items.get(key(x, y)); return it && it.w > 1 && !(it.rest > 0) && it.grip.size < it.w ? 1 : 0; };

    function spawnAnt(col, x, y, adult) {
      const a = {
        id: nextAnt++, col: col.id, x, y, opt: 4, carry: 0, grip: 0, hunger: 2, age: adult ? ri(900) : 0,
        life: 2600 + ri(1800), role: (() => { const r = rnd(); return r < 0.45 ? 0 : r < 0.8 ? 1 : 2; })(), wander: ri(4), wt: 10 + ri(30), emote: 0, last: null, alive: true,
      };
      ants.push(a);
      antsById.set(a.id, a);
      crowdAdd(x, y, 1);
      return a;
    }

    const HUES = [28, 200, 300, 120, 50, 250];
    function foundColony(x0) {
      const s = skyTop(x0);
      if (s >= H - 16) return null;
      const id = colonies.length;
      const col = {
        id, ex: x0, hue: HUES[id % HUES.length], store: 14, queen: null, chamber: new Set(), plan: new Set(),
        eggs: [], dn: makeField(), dg: makeField(), dm: makeField(), alive: true, born: 0, died: 0, hungerQ: 0,
        floor: s + 11, top: s, branches: 0,
      };
      // The founding queen dug a shaft and a first chamber.
      // A two-lane shaft, so ants can pass, down to a two-row chamber.
      const bottom = s + 10;
      for (const X of [x0, x0 + 1]) {
        for (let y = 0; y < bottom; y++) col.plan.add(key(X, y));
        for (let y = s - 1; y < bottom; y++) if (cellType(X, y) !== AIR) setCell(X, y, AIR);
      }
      for (let dx = -5; dx <= 6; dx++) for (let dy = 0; dy < 2; dy++) {
        const X = x0 + dx, Y = bottom + dy;
        col.plan.add(key(X, Y));
        if (cellType(X, Y) !== AIR) setCell(X, Y, AIR);
        col.chamber.add(key(X, Y));
      }
      col.floor = bottom + 1;
      col.queen = { x: x0, y: bottom + 1, age: 0 };
      colonies.push(col);
      addBranch(col);
      for (let k = 0; k < 10; k++) spawnAnt(col, x0 - 2 + ri(5), bottom + 2, true);
      return col;
    }

    function addBranch(col) {
      const open = [];
      for (const k of col.plan) { const y = ky(k); if (y >= col.top + 6 && cellType(kx(k), y) === AIR) open.push(k); }
      if (!open.length) return;
      const k0 = open[ri(open.length)];
      let x = kx(k0), y = ky(k0);
      const dir = rnd() < 0.5 ? -1 : 1, len = 6 + ri(10);
      const cells = [];
      for (let i = 0; i < len; i++) {
        if (rnd() < 0.3 && y < H - 6) y++;
        else x += dir;
        if (cellType(x, y) === ROCK) break;
        cells.push(key(x, y));
      }
      if (cells.length >= 4) {
        const ex = x, ey = y;
        for (let dx = -2; dx <= 2; dx++) for (let dy = -1; dy <= 0; dy++) if (cellType(ex + dx, ey + dy) !== ROCK) cells.push(key(ex + dx, ey + dy));
      }
      for (const k of cells) col.plan.add(k);
      col.branches++;
    }

    function release(a) {
      const it = itemsById.get(a.grip);
      if (it) it.grip.delete(a.id);
      a.grip = 0;
    }
    function moveAnt(a, x, y) {
      crowdAdd(a.x, a.y, -1);
      a.x = x; a.y = y;
      crowdAdd(x, y, 1);
    }
    function kill(a, why) {
      a.alive = false;
      release(a);
      crowdAdd(a.x, a.y, -1);
      if (a.carry === 1) addItem('food', 1, a.x, a.y);
      a.carry = 0;
      stats.died++;
      if (why === 'starved') stats.starved++; else stats.oldAge++;
      colonies[a.col].died++;
      corpses.push({ x: a.x, y: a.y, t: 300, hue: colonies[a.col].hue });
      antsById.delete(a.id);
    }
    const corpses = [];

    // ---------------------------------------------------------------- the rule as plain code (JS port)
    const sgn = (v) => (v > 0 ? 1 : v < 0 ? -1 : 0);
    function senseRule(get) {
      const s = [];
      for (let f = 0; f < 5; f++) for (const [dy, dx] of DIR4) s.push(sgn(get(f, 0, 0) - get(f, dy, dx)));
      for (const [dy, dx] of DIR4) s.push(get(5, dy, dx) === 0 ? 1 : 0);
      for (const [dy, dx] of DIR4) s.push(sgn(get(9, 0, 0) - get(9, dy, dx)));
      s.push(get(0, 0, 0) <= 0 ? 1 : 0);
      s.push(get(3, 0, 0) <= 0 ? 1 : 0);
      for (const f of [0, 1, 2, 4]) s.push(get(f, 0, 0) < CEIL ? 1 : 0);
      for (const ch of [6, 7, 8]) s.push(DIR4.some(([dy, dx]) => get(ch, dy, dx) === 1) ? 1 : 0);
      s.push(get(9, 0, 0));
      s.push(get(0, 0, 0));
      return s;
    }
    const turn4 = (a, b) => { const t = (a + 4 - b) % 4; return Math.min(t, 4 - t); };
    const inertia = (o, cur) => (o < 4 && cur < 4 ? KEEP[turn4(o, cur)] : o === 4 && cur === 4 ? KEEP_HOLD : 0);
    // The brain on a flat input vector (program order), as plain code.
    function brainParts(v) {
      const g = (n) => v[BI[n]];
      let opt = 0, wander = 0, role = 0;
      for (let k = 0; k < 3; k++) if (g('r' + k) === 1) role = k;
      for (let k = 0; k < 5; k++) if (g('o' + k) === 1) opt = k;
      for (let k = 0; k < 4; k++) if (g('w' + k) === 1) wander = k;
      const cf = g('cf'), cs = g('cs'), gr = g('gr'), hunger = g('hunger'), old = g('old'), store = g('store');
      const free = !cf && !cs && !gr;
      const hungry = Math.min(10, Math.max(0, hunger - 5));
      const starving = hunger >= STARV, scared = g('fs') >= 1, hs = store >= 1;
      const kf = g('kf') === 1, kg = g('kg') === 1, kh = g('kh') === 1, kn = g('kn') === 1;
      const idle = free && !kf && !kg && !kh;
      const drives = {
        home: kn ? 6 * cf + (hs ? 2 * Math.min(hungry, 4) : 0) + (free && g('dns') >= FAR ? 4 : 0) : 0,
        forage: (free && kf ? 5 - store + 2 * old + (role === 1 ? RB : 0) : 0) + (free && kf && !hs ? Math.min(hungry, 4) : 0),
        dig: free && kg && !starving ? 4 - 2 * old + (role === 0 ? RB : 0) : 0,
        dump: cs === 1 && kn ? 6 : 0,
        help: free && kh && !starving ? 3 + (role === 2 ? RB : 0) : 0,
        explore: idle || !kn ? 3 : 0,
        flee: scared ? 8 : 0,
      };
      const pick = free && g('lf') === 1 && !scared;
      const dig = free && g('ld') === 1 && !starving && !scared;
      const grip = (gr === 1 || (free && g('lp') === 1)) && !starving && !scared;
      const drop = (cf === 1 && g('zn') === 1) || (cs === 1 && g('zm') === 1);
      const eat = (g('zn') === 1 && hs && hunger >= PECK) || (cf === 1 && starving);
      const actions = [pick, dig, grip, drop, eat];
      const dv = [drives.home, drives.forage, drives.dig, drives.dump, drives.help, drives.explore, drives.flee];
      let task = 0;
      for (let f = 1; f < 7; f++) if (7 * dv[f] + (6 - f) > 7 * dv[task] + (6 - task)) task = f;
      const G = ['gn', 'gf', 'gg', 'gm', 'gh'];
      const follow = (d) => (task < 5 ? g(G[task] + d) : task === 5 ? (wander === d ? 1 : 0) : g('fg' + d));
      const scores = [];
      for (let d = 0; d < 4; d++) scores.push(PULL * follow(d) + inertia(d, opt) - BIG * g('b' + d));
      scores.push(HOLD_BONUS * (gr === 1 || actions.some(Boolean) ? 1 : 0) + inertia(4, opt));
      let option = 0;
      for (let d = 1; d < 5; d++) if (5 * scores[d] + (4 - d) > 5 * scores[option] + (4 - option)) option = d;
      const lv = [1, hungry, 3 * (cs === 1 || dig || (free && kg) ? 1 : 0), 4 * cf, 5 * (grip ? 1 : 0), 9 * (scared ? 1 : 0), 2 * (idle || !kn ? 1 : 0)];
      let emote = 0;
      for (let e = 1; e < 7; e++) if (7 * lv[e] + (6 - e) > 7 * lv[emote] + (6 - emote)) emote = e;
      return { drives, task, actions, scores, option, emote, levels: lv };
    }
    function pathRule(get) {
      if (get(2, 0, 0) === 1) return 0;
      if (get(1, 0, 0) === 1) return CEIL;
      let best = CEIL;
      for (const [dy, dx] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) best = Math.min(best, 1 + get(0, dy, dx));
      return best;
    }

    // Check the JS port of the rule against the Rust rule's vectors.
    function verifyPort() {
      let sense = 0, brain = 0;
      for (const [x, y] of data.vectors.sense) {
        const m = new Map();
        senseTaps.forEach((t, i) => m.set(t.join(','), x[i]));
        const got = senseRule((ch, dy, dx) => m.get(ch + ',' + dy + ',' + dx));
        if (got.every((v, i) => v === y[i])) sense++;
      }
      for (const [v, option, actions, emote] of data.vectors.brain) {
        const p = brainParts(v);
        if (p.option === option && p.emote === emote && p.actions.every((b, i) => b === actions[i])) brain++;
      }
      return { sense, brain, senseTotal: data.vectors.sense.length, brainTotal: data.vectors.brain.length };
    }

    // ---------------------------------------------------------------- networks at an ant
    const check = { on: !!opts.check, sense: 0, brain: 0, path: 0, decisions: 0, cells: 0 };
    const senseX = new Float64Array(senseTaps.length), brainX = new Float64Array(NB), pathX = new Float64Array(pathTaps.length);
    function reader(a) {
      const col = colonies[a.col];
      return (ch, dy, dx) => {
        const X = a.x + dx, Y = a.y + dy;
        switch (ch) {
          case 0: return fget(col.dn, X, Y);
          case 1: return fget(shared.df, X, Y);
          case 2: return fget(col.dg, X, Y);
          case 3: return fget(col.dm, X, Y);
          case 4: return fget(shared.dh, X, Y);
          case 5: return walk(X, Y);
          case 6: return lift(X, Y);
          case 7: return pull(X, Y);
          case 8: return digTarget(col, X, Y) ? 1 : 0;
          default: return Y < 0 || Y >= H ? 0 : fearAt(X, Y);
        }
      };
    }
    function storeBucket(s) { return s <= 0 ? 0 : s < 10 ? 1 : s < 30 ? 2 : 3; }
    function brainInput(a, senses) {
      const v = new Array(NB).fill(0);
      v[BI['o' + a.opt]] = 1;
      v[BI.cf] = a.carry === 1 ? 1 : 0;
      v[BI.cs] = a.carry === 2 ? 1 : 0;
      v[BI.gr] = a.grip ? 1 : 0;
      v[BI.hunger] = Math.min(HUN, a.hunger);
      v[BI.old] = a.age > 0.55 * a.life ? 1 : 0;
      v[BI.store] = storeBucket(colonies[a.col].store);
      v[BI['w' + a.wander]] = 1;
      v[BI['r' + a.role]] = 1;
      for (let i = 0; i < senses.length; i++) v[BI.gn0 + i] = senses[i];
      return v;
    }
    function decide(a) {
      const get = reader(a);
      senseTaps.forEach(([ch, dy, dx], i) => (senseX[i] = get(ch, dy, dx)));
      const so = senseNet.eval(senseX);
      const senses = Array.from(so.subarray ? so.subarray(0, data.sense.outputs.length) : so.slice(0, data.sense.outputs.length), Math.round);
      const v = brainInput(a, senses);
      brainX.set(v);
      const out = brainNet.eval(brainX);
      const bit = (n) => out[BO[n]] > 0.5;
      let option = 0, emote = 0;
      for (let k = 0; k < 5; k++) if (bit('o2_' + k)) option = k;
      for (let k = 0; k < 7; k++) if (bit('em' + k)) emote = k;
      const dec = { option, emote, actions: [bit('pick'), bit('dig2'), bit('grip'), bit('drop'), bit('eat')], senses, input: v };
      if (check.on) {
        check.decisions++;
        const rs = senseRule(get);
        if (rs.some((x, i) => x !== senses[i])) check.sense++;
        const p = brainParts(brainInput(a, rs));
        if (p.option !== option || p.emote !== emote || p.actions.some((b, i) => b !== dec.actions[i])) check.brain++;
      }
      if (opts.onDecide) opts.onDecide(a, dec);
      return dec;
    }

    // ---------------------------------------------------------------- fields step
    const FIELD_ITERS = 3;
    function walkList(x0, x1) {
      const L = [];
      for (let x = x0; x <= x1; x++) for (let y = 0; y < H; y++) if (walk(x, y)) L.push(x, y);
      return L;
    }
    function fieldStep(f, L, goal) {
      const n = L.length / 2, nv = new Uint8Array(n);
      for (let i = 0; i < n; i++) {
        const x = L[2 * i], y = L[2 * i + 1], gl = goal(x, y) ? 1 : 0;
        const get = (ch, dy, dx) => (ch === 0 ? fget(f, x + dx, y + dy) : ch === 1 ? 0 : gl);
        for (let t = 0; t < pathTaps.length; t++) { const [ch, dy, dx] = pathTaps[t]; pathX[t] = get(ch, dy, dx); }
        const v = Math.round(pathNet.eval(pathX)[0]);
        nv[i] = v;
        if (check.on) { check.cells++; if (pathRule(get) !== v) check.path++; }
      }
      for (let i = 0; i < n; i++) {
        const x = L[2 * i], y = L[2 * i + 1], ci = Math.floor(x / CW);
        farr(f, ci)[(x - ci * CW) * H + y] = nv[i];
      }
    }
    const anyNeighbor = (x, y, fn) => fn(x + 1, y) || fn(x - 1, y) || fn(x, y + 1) || fn(x, y - 1);
    function colonyRange(col) {
      let lo = col.ex, hi = col.ex;
      for (const a of ants) if (a.alive && a.col === col.id) { if (a.x < lo) lo = a.x; if (a.x > hi) hi = a.x; }
      return [lo - 24, hi + 24];
    }
    function inDump(col, x, y) {
      const d = Math.min(Math.abs(x - col.ex), Math.abs(x - col.ex - 1));
      return d >= 3 && d <= 12 && y === skyTop(x) - 1;
    }
    function updateFields() {
      let lo = Infinity, hi = -Infinity;
      for (const col of colonies) {
        if (!col.alive) continue;
        const [a, b] = colonyRange(col);
        const L = walkList(a, b);
        lo = Math.min(lo, a); hi = Math.max(hi, b);
        for (let k = 0; k < FIELD_ITERS; k++) {
          fieldStep(col.dn, L, (x, y) => col.chamber.has(key(x, y)));
          fieldStep(col.dg, L, (x, y) => anyNeighbor(x, y, (X, Y) => digTarget(col, X, Y)));
          fieldStep(col.dm, L, (x, y) => inDump(col, x, y));
        }
      }
      if (lo > hi) return;
      const L = walkList(lo, hi);
      for (let k = 0; k < FIELD_ITERS; k++) {
        fieldStep(shared.df, L, (x, y) => anyNeighbor(x, y, lift));
        fieldStep(shared.dh, L, (x, y) => anyNeighbor(x, y, pull));
      }
    }

    // ---------------------------------------------------------------- host physics
    const sandActive = new Map(); // key -> still ticks
    function placeSand(x, y) {
      if (y < 0 || y >= H || solid(x, y)) return false;
      setCell(x, y, SAND);
      sandActive.set(key(x, y), 0);
      return true;
    }
    // Loose sand never moves into a colony's planned tunnels (the mound keeps a rim).
    const planned = (x, y) => colonies.some((c) => c.alive && c.plan.has(key(x, y)));
    const free = (x, y) => !solid(x, y) && !planned(x, y);
    function stepSand() {
      const keys = [...sandActive.keys()];
      for (const k of keys) {
        const still = sandActive.get(k);
        sandActive.delete(k);
        const x = kx(k), y = ky(k);
        if (cellType(x, y) !== SAND) continue;
        let to = null;
        if (free(x, y + 1)) to = [x, y + 1];
        else if (solid(x, y + 1)) {
          const s = rnd() < 0.5 ? 1 : -1;
          for (const dx of [s, -s]) if (!solid(x + dx, y) && free(x + dx, y + 1)) { to = [x + dx, y + 1]; break; }
        }
        if (to) {
          setCell(x, y, AIR);
          setCell(to[0], to[1], SAND);
          sandActive.set(key(to[0], to[1]), 0);
          if (cellType(x, y - 1) === SAND) sandActive.set(key(x, y - 1), 0);
        } else if (still < 30) sandActive.set(k, still + 1);
        else setCell(x, y, SOIL);
      }
    }
    // Ants give up on an item that will not move: it slips, and nobody pulls it for a while.
    function slip(it) {
      for (const id of it.grip) { const a = antsById.get(id); if (a) a.grip = 0; }
      it.grip.clear();
      it.stuck = 0;
      it.best = CEIL;
      it.rest = 300;
    }
    function stepItems() {
      for (const it of [...itemsById.values()]) {
        if (!itemsById.has(it.id)) continue;
        if (it.rest > 0) it.rest--;
        if (it.grip.size >= it.w) {
          if (stats.tick % 2) continue;
          const col = colonies[antsById.get([...it.grip][0])?.col ?? 0];
          const f = it.kind === 'food' ? col.dn : col.dm;
          let best = null, bv = CEIL;
          for (const [dy, dx] of DIR4) {
            const X = it.x + dx, Y = it.y + dy;
            if (solid(X, Y) || !walk(X, Y)) continue;
            const v = fget(f, X, Y);
            if (v < bv) { bv = v; best = [X, Y, dy, dx]; }
          }
          if (!best || bv >= it.best) { if (++it.stuck > 40) slip(it); if (!best) continue; }
          else { it.stuck = 0; it.best = bv; }
          const [X, Y, dy, dx] = best;
          moveItem(it, X, Y);
          for (const id of [...it.grip]) {
            const a = antsById.get(id);
            if (!a) { it.grip.delete(id); continue; }
            if (!solid(a.x + dx, a.y + dy)) moveAnt(a, a.x + dx, a.y + dy);
            if (Math.max(Math.abs(a.x - it.x), Math.abs(a.y - it.y)) > 1) release(a);
          }
          if (bv === 0) {
            if (it.kind === 'food') { col.store += 4 * it.w; stats.foodIn += it.w; removeItem(it); }
            else { const x = it.x, y = it.y; removeItem(it); placeSand(x, y); stats.sandOut++; }
          }
        } else if (it.grip.size && ++it.stuck > 90) slip(it);
        else if (!solid(it.x, it.y + 1)) {
          moveItem(it, it.x, it.y + 1);
          for (const id of [...it.grip]) { const a = antsById.get(id); if (!a || Math.max(Math.abs(a.x - it.x), Math.abs(a.y - it.y)) > 1) { if (a) release(a); else it.grip.delete(id); } }
        }
      }
    }

    // ---------------------------------------------------------------- one tick
    let thinkEvery = opts.thinkEvery || 1;
    function act(a, dec) {
      const col = colonies[a.col];
      const [pick, dig, grip, drop, eat] = dec.actions;
      if (grip) {
        if (!a.grip) {
          for (const [dy, dx] of DIR4) {
            const it = items.get(key(a.x + dx, a.y + dy));
            if (it && it.w > 1 && !(it.rest > 0) && it.grip.size < it.w) { it.grip.add(a.id); a.grip = it.id; break; }
          }
        }
      } else if (a.grip) release(a);
      if (!a.grip && a.carry === 0) {
        if (pick) {
          for (const [dy, dx] of DIR4) {
            const it = items.get(key(a.x + dx, a.y + dy));
            if (it && it.kind === 'food' && it.w === 1) { removeItem(it); a.carry = 1; break; }
          }
        } else if (dig) {
          for (const [dy, dx] of DIR4) {
            const X = a.x + dx, Y = a.y + dy;
            if (!digTarget(col, X, Y)) continue;
            const t = cellType(X, Y);
            setCell(X, Y, AIR);
            sandActive.delete(key(X, Y));
            stats.dug++;
            if (t === PEB) addItem('peb', 2 + ri(2), X, Y);
            else a.carry = 2;
            break;
          }
        }
      }
      if (drop) {
        if (a.carry === 1 && fget(col.dn, a.x, a.y) === 0) { col.store += 3; stats.foodIn++; a.carry = 0; }
        else if (a.carry === 2 && fget(col.dm, a.x, a.y) === 0) {
          const x = a.x, y = a.y;
          if (!solid(x, y - 1)) moveAnt(a, x, y - 1);
          if (placeSand(x, y)) { a.carry = 0; stats.sandOut++; }
        }
      }
      if (eat) {
        if (a.carry === 1 && a.hunger >= STARV) { a.carry = 0; a.hunger = Math.max(0, a.hunger - 8); }
        else if (col.store > 0 && fget(col.dn, a.x, a.y) === 0 && a.hunger >= PECK) { col.store--; a.hunger = Math.max(0, a.hunger - 8); }
      }
    }
    function stepAnt(a) {
      if (--a.wt <= 0) {
        const outside = a.y < skyTop(a.x) + 1, col = colonies[a.col];
        const r = rnd();
        if (fget(col.dn, a.x, a.y) >= CEIL && rnd() < 0.7) a.wander = a.x < col.ex ? 0 : 2; // lost: head for home
        else a.wander = outside ? (r < 0.4 ? 0 : r < 0.8 ? 2 : r < 0.9 ? 1 : 3) : ri(4);
        a.wt = 15 + ri(40);
      }
      const think = (stats.tick + a.id) % thinkEvery === 0;
      if (think) {
        const dec = decide(a);
        a.last = dec;
        a.opt = dec.option;
        a.emote = dec.emote;
        act(a, dec);
      }
      // Between thoughts an ant waits: walking on an old heading overshoots its goals.
      if (think && !a.grip && a.opt < 4) {
        const [dy, dx] = DIR4[a.opt], X = a.x + dx, Y = a.y + dy;
        if (walk(X, Y)) moveAnt(a, X, Y);
      }
    }
    function settleAnt(a) {
      if (solid(a.x, a.y)) {
        let y = a.y;
        while (y > 0 && solid(a.x, y)) y--;
        moveAnt(a, a.x, y);
      } else if (!walk(a.x, a.y) && !solid(a.x, a.y + 1)) moveAnt(a, a.x, a.y + 1);
      if (a.grip) {
        const it = itemsById.get(a.grip);
        if (!it || Math.max(Math.abs(a.x - it.x), Math.abs(a.y - it.y)) > 1) release(a);
      }
    }
    function stepColony(col) {
      if (!col.alive) return;
      const q = col.queen;
      const pop = ants.filter((a) => a.alive && a.col === col.id).length;
      if (q) {
        q.age++;
        if (stats.tick % 60 === 0) { if (col.store > 0) { col.store--; col.hungerQ = 0; } else col.hungerQ++; }
        if (col.hungerQ > 25) col.queen = null;
      }
      let open = 0, pending = 0;
      for (const k of col.plan) { const t = cellType(kx(k), ky(k)); if (t === AIR) open++; else if (diggable(t)) pending++; }
      const capacity = 6 + Math.floor(open / 3);
      col.capacity = capacity; col.open = open; col.pending = pending; col.pop = pop;
      if (col.queen && stats.tick % 25 === 0 && col.store >= 3 + Math.floor(pop / 3) && pop + col.eggs.length < capacity) {
        const spots = [...col.chamber].filter((k) => walk(kx(k), ky(k)));
        if (spots.length) {
          const k = spots[ri(spots.length)];
          col.eggs.push({ x: kx(k), y: ky(k), t: 160 });
          col.store -= 3;
        }
      }
      for (const e of col.eggs) e.t--;
      for (const e of col.eggs.filter((e) => e.t <= 0)) { spawnAnt(col, e.x, e.y, false); col.born++; stats.born++; }
      col.eggs = col.eggs.filter((e) => e.t > 0);
      if (pending < 6 && pop + col.eggs.length > 0.7 * capacity && col.branches < 90) addBranch(col);
      if (!col.queen && pop === 0 && col.eggs.length === 0) col.alive = false;
    }
    function spawnFood() {
      const live = colonies.filter((c) => c.alive);
      if (!live.length || stats.tick % 10) return;
      let n = 0;
      for (const it of itemsById.values()) if (it.kind === 'food') n++;
      if (n > 70 * live.length) return;
      const col = live[ri(live.length)];
      const x = col.ex + (rnd() < 0.5 ? -1 : 1) * (8 + ri(40));
      const r = rnd(), w = r < 0.55 ? 1 : r < 0.75 ? 2 : r < 0.9 ? 3 : 5;
      addItem('food', w, x, skyTop(x) - 1);
    }
    function tick() {
      stats.tick++;
      stepSand();
      stepItems();
      updateFields();
      for (const a of ants) if (a.alive) stepAnt(a);
      for (const a of ants) if (a.alive) settleAnt(a);
      for (const a of ants) {
        if (!a.alive) continue;
        a.age++;
        if ((stats.tick + a.id) % 45 === 0) a.hunger++;
        if (a.hunger >= HUN) kill(a, 'starved');
        else if (a.age >= a.life) kill(a, 'old');
      }
      if (stats.tick % 50 === 0) for (let i = ants.length - 1; i >= 0; i--) if (!ants[i].alive) ants.splice(i, 1);
      for (const c of colonies) stepColony(c);
      for (const c of corpses) c.t--;
      for (let i = corpses.length - 1; i >= 0; i--) if (corpses[i].t <= 0) corpses.splice(i, 1);
      if (stats.tick % 10 === 0) for (const [k, v] of fear) { if (v <= 1) fear.delete(k); else fear.set(k, v - 1); }
      spawnFood();
    }

    // ---------------------------------------------------------------- observer tools
    function disk(x, y, r, fn) {
      for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) if (dx * dx + dy * dy <= r * r + r) fn(x + dx, y + dy);
    }
    const tools = {
      food(x, y, w) { if (!solid(x, y)) return !!addItem('food', w, x, y); return false; },
      sand(x, y, r) { let n = 0; disk(x, y, r, (X, Y) => { if (placeSand(X, Y)) n++; }); return n > 0; },
      dig(x, y, r) {
        disk(x, y, r, (X, Y) => { const t = cellType(X, Y); if (Y < H - 2 && (t === SOIL || t === SAND || t === PEB)) { setCell(X, Y, AIR); sandActive.delete(key(X, Y)); } });
        return true;
      },
      scare(x, y, r) { disk(x, y, r, (X, Y) => { if (Y >= 0 && Y < H) { const d = Math.sqrt((X - x) ** 2 + (Y - y) ** 2); const v = Math.max(1, 3 - Math.floor((3 * d) / (r + 1))); fear.set(key(X, Y), Math.max(fearAt(X, Y), v)); } }); return true; },
      colony(x) { for (const c of colonies) if (c.alive && Math.abs(c.ex - x) < 20) return null; return foundColony(x); },
    };

    foundColony(0);

    return {
      H, CW, CEIL, AIR, SOIL, ROCK, SAND, PEB, DIR4,
      tick, tools, stats, ants, colonies, items, itemsById, corpses, fear, check, shared,
      cellType, solid, walk, skyTop, surfaceAt, fget, key, h2, crowdAt,
      brainParts, verifyPort, digTarget,
      get thinkEvery() { return thinkEvery; },
      set thinkEvery(v) { thinkEvery = v; },
      hillHeight(col) {
        let m = 0;
        for (let x = col.ex - 14; x <= col.ex + 15; x++) m = Math.max(m, surfaceAt(x) - skyTop(x));
        return m;
      },
    };
  }

  root.AntCore = { createSim, makeNet };
})(typeof module !== 'undefined' ? module.exports : window);
