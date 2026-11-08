(function () {
  'use strict';
  const DATA = JSON.parse(document.getElementById('park-data').textContent);
  // Ecosystem setup: saved per viewer, applied by Save & restart.
  const DEFAULT_SETUP = { seed: 11, species: 30, diets: 'mixed', plantRate: 1.25, moving: true, call: 2.5, learn: true, lr: 0.003 };
  let setup = { ...DEFAULT_SETUP };
  try { setup = { ...DEFAULT_SETUP, ...JSON.parse(localStorage.getItem('yumon-park-setup') || '{}') }; } catch (e) { /* storage blocked: defaults */ }
  const makeSim = (st) => ParkCore.createPark(DATA, { seed: st.seed, species: st.species, diets: st.diets, movingPatches: st.moving, learn: st.learn,
    tune: { plantRate: st.plantRate, call: st.call, lr: st.lr } });
  let sim = makeSim(setup);
  const C = DATA.consts;
  const $ = (id) => document.getElementById(id);
  const canvas = $('world'), ctx = canvas.getContext('2d');
  const OPT_LABEL = { eat: 'Eat', hunt: 'Hunt', flee: 'Flee', hide: 'Hide', mate: 'Mate', sleep: 'Sleep', flock: 'Flock', roam: 'Roam' };
  const W0 = sim.WORLD_W, H0 = sim.WORLD_H;
  let founder = new Map(sim.species.map((sp) => [sp.id, { ...sp.genes }]));
  const hue = (sp, a) => `hsl(${(a ? a.genes.hue : sp.genes.hue).toFixed(0)} 62% ${dark() ? 62 : 48}%)`;
  const dark = () => getComputedStyle(document.documentElement).colorScheme === 'dark';

  let pal = {};
  function readPalette() {
    const cs = getComputedStyle(document.documentElement), v = (n) => cs.getPropertyValue(n).trim();
    pal = { skyTop: v('--sky-top'), skyBot: v('--sky-bot'), canopy: v('--canopy'), canopy2: v('--canopy-2'), trunk: v('--trunk'), ground: v('--ground'),
      grass: v('--grass'), waterTop: v('--water-top'), waterBot: v('--water-bot'), ink: v('--ink'), accent: v('--accent'), bad: v('--bad'), line: v('--line'), muted: v('--muted'), panel: v('--panel') };
  }
  readPalette();
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', readPalette);
  new MutationObserver(readPalette).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  // ---------------------------------------------------------------- camera
  const cam = { x: W0 / 2, y: H0 / 2, z: 1 };
  let W = 0, Hpx = 0, dpr = 1;
  function resize() {
    const r = canvas.getBoundingClientRect();
    dpr = Math.min(2, window.devicePixelRatio || 1);
    W = r.width; Hpx = r.height;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(Hpx * dpr);
  }
  new ResizeObserver(resize).observe(canvas);
  resize();
  const scale = () => (Hpx / H0) * cam.z;
  function clampCam() {
    const s = scale(), half = Hpx / 2 / s;
    cam.y = Math.max(half, Math.min(H0 - half, cam.y));
    cam.x = ((cam.x % W0) + W0) % W0;
  }
  const sx = (x) => { let d = x - cam.x; if (d > W0 / 2) d -= W0; else if (d < -W0 / 2) d += W0; return d * scale() + W / 2; };
  const sy = (y) => (y - cam.y) * scale() + Hpx / 2;
  const toWorld = (px, py) => [cam.x + (px - W / 2) / scale(), cam.y + (py - Hpx / 2) / scale()];

  // ---------------------------------------------------------------- state
  const state = { tool: 'watch', running: true, speed: 2, selected: null, follow: false };
  let acc = 0, last = performance.now(), tickMs = 0;
  const TPS = 12;
  function step() {
    const t0 = performance.now();
    sim.tick();
    tickMs = tickMs * 0.9 + (performance.now() - t0) * 0.1;
  }

  // ---------------------------------------------------------------- scenery (fixed per world x)
  const h2 = (x, s) => { let h = Math.imul(x | 0, 374761393) + Math.imul(s, 668265263); h = Math.imul(h ^ (h >>> 13), 1274126177); return ((h ^ (h >>> 16)) >>> 0) / 4294967296; };
  const TREES = [];
  for (let x = 20; x < W0; x += 70 + Math.floor(h2(x, 3) * 50)) TREES.push({ x, w: 6 + h2(x, 4) * 6, top: 212 + h2(x, 5) * 40, r: 46 + h2(x, 6) * 30 });

  function drawScene() {
    const s = scale();
    const [b0, b1, b2, b3] = sim.BAND_Y;
    let g = ctx.createLinearGradient(0, sy(0), 0, sy(330));
    g.addColorStop(0, pal.skyTop); g.addColorStop(1, pal.skyBot);
    ctx.fillStyle = g; ctx.fillRect(0, sy(0), W, sy(330) - sy(0));
    // trees: trunks to the ground, canopies across the tree band
    for (const t of TREES) for (const X of copies(t.x, t.r)) {
      ctx.fillStyle = pal.trunk;
      ctx.fillRect(X - (t.w * s) / 2, sy(t.top), t.w * s, sy(336) - sy(t.top));
    }
    for (const t of TREES) for (const X of copies(t.x, t.r)) {
      ctx.fillStyle = h2(t.x, 9) < 0.5 ? pal.canopy : pal.canopy2;
      ctx.beginPath(); ctx.ellipse(X, sy(t.top + 8), t.r * s, (t.r * 0.62) * s, 0, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.ellipse(X + t.r * 0.55 * s, sy(t.top + 34), t.r * 0.6 * s, t.r * 0.42 * s, 0, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.ellipse(X - t.r * 0.6 * s, sy(t.top + 30), t.r * 0.55 * s, t.r * 0.4 * s, 0, 0, Math.PI * 2); ctx.fill();
    }
    ctx.fillStyle = pal.ground; ctx.fillRect(0, sy(b2[0]), W, sy(b2[1]) - sy(b2[0]));
    ctx.fillStyle = pal.grass; ctx.fillRect(0, sy(b2[0]), W, Math.max(2, 5 * s));
    g = ctx.createLinearGradient(0, sy(b3[0]), 0, sy(b3[1]));
    g.addColorStop(0, pal.waterTop); g.addColorStop(1, pal.waterBot);
    ctx.fillStyle = g; ctx.fillRect(0, sy(b3[0]), W, sy(b3[1]) - sy(b3[0]));
    void b0; void b1;
  }
  function copies(x, r) {
    const out = [], X = sx(x), span = W0 * scale(), pad = (r || 10) * scale();
    for (const k of [-1, 0, 1]) { const v = X + k * span; if (v > -pad && v < W + pad) out.push(v); }
    return out;
  }

  const FOOD_COL = { gnats: '#3a3a3a', fruit: '#d6453d', grass: '#8bcf5a', plankton: '#7fe0c4', meat: '#9b3b2e' };
  function drawFood() {
    const s = scale();
    for (const f of sim.food) {
      if (f.amt < 0.3) continue;
      for (const X of copies(f.x, 10)) {
        const Y = sy(f.y), k = Math.min(1, f.amt / Math.max(1, f.cap || 6));
        ctx.fillStyle = f.kind === 'gnats' && dark() ? '#c9d2d0' : FOOD_COL[f.kind];
        if (f.kind === 'grass') {
          ctx.strokeStyle = FOOD_COL.grass; ctx.lineWidth = Math.max(1, 1.3 * s);
          ctx.beginPath();
          for (let i = -1; i <= 1; i++) { ctx.moveTo(X + i * 2 * s, Y + 3 * s); ctx.lineTo(X + i * 3 * s, Y + 3 * s - (3 + 7 * k) * s); }
          ctx.stroke();
        } else if (f.kind === 'gnats') {
          for (let i = 0; i < 2 + Math.round(5 * k); i++) ctx.fillRect(X + (h2(i, f.x | 0) - 0.5) * 14 * s, Y + (h2(i, 7 + (f.y | 0)) - 0.5) * 10 * s, Math.max(1, s), Math.max(1, s));
        } else if (f.kind === 'plankton') {
          ctx.globalAlpha = 0.85;
          for (let i = 0; i < 2 + Math.round(5 * k); i++) { ctx.beginPath(); ctx.arc(X + (h2(i, f.x | 0) - 0.5) * 14 * s, Y + (h2(i, 3) - 0.5) * 10 * s, Math.max(1, 1.2 * s), 0, 7); ctx.fill(); }
          ctx.globalAlpha = 1;
        } else {
          ctx.beginPath(); ctx.arc(X, Y, Math.max(1.5, (f.kind === 'meat' ? 2 + Math.min(5, f.amt * 0.5) : 2 + 2 * k) * s), 0, 7); ctx.fill();
        }
      }
    }
  }

  // A Yumon: body by size and hue, with the limbs its species has, sized by its genes.
  function drawYumon(a, X, Y, now) {
    const s = scale(), R = a.t.r * s, g = a.genes, face = a.vx >= 0 ? 1 : -1, sp = a.s;
    const hidden = a.option === 3 || (a.sleeping && sp.caps.has('camo'));
    ctx.globalAlpha = hidden ? 0.45 : 1;
    const col = hue(sp, a);
    ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = Math.max(1, 0.18 * R);
    const flap = Math.sin(now / 90 + a.id) * 0.5;
    if (g.wings && a.t.spd[0] > 0) { ctx.beginPath(); ctx.ellipse(X - face * 0.2 * R, Y - 0.6 * R, R * 0.9 * g.wings, R * (0.35 + 0.25 * flap), -face * 0.5, 0, Math.PI * 2); ctx.globalAlpha *= 0.7; ctx.fill(); ctx.globalAlpha = hidden ? 0.45 : 1; }
    if (g.fins) { ctx.beginPath(); ctx.moveTo(X - face * R * 0.8, Y); ctx.lineTo(X - face * R * (1.3 + 0.4 * g.fins), Y - R * 0.6); ctx.lineTo(X - face * R * (1.3 + 0.4 * g.fins), Y + R * 0.6); ctx.closePath(); ctx.fill(); }
    if (g.legs) { ctx.beginPath(); for (const o of [-0.45, 0.45]) { ctx.moveTo(X + o * R, Y + 0.6 * R); ctx.lineTo(X + o * R + Math.sin(now / 80 + a.id + o * 3) * 0.2 * R * (Math.abs(a.vx) > 0.1), Y + R * (0.9 + 0.5 * g.legs)); } ctx.stroke(); }
    if (g.stinger) { ctx.beginPath(); ctx.moveTo(X - face * R * 0.85, Y - 0.1 * R); ctx.lineTo(X - face * R * (1.2 + 0.5 * g.stinger), Y - 0.5 * R); ctx.stroke(); }
    // body
    ctx.beginPath(); ctx.ellipse(X, Y, R, R * 0.82, 0, 0, Math.PI * 2); ctx.fill();
    if (g.shell) { ctx.fillStyle = 'rgba(0,0,0,0.28)'; ctx.beginPath(); ctx.ellipse(X, Y - 0.1 * R, R * 1.02, R * 0.78, 0, Math.PI, 0); ctx.fill(); }
    if (g.spines) { ctx.strokeStyle = 'rgba(0,0,0,0.45)'; ctx.beginPath(); for (let k = 0; k < 5; k++) { const t = Math.PI * (1.1 + k * 0.2); ctx.moveTo(X + Math.cos(t) * R, Y + Math.sin(t) * R * 0.82); ctx.lineTo(X + Math.cos(t) * R * (1.3 + 0.2 * g.spines), Y + Math.sin(t) * R * (1.1 + 0.2 * g.spines)); } ctx.stroke(); }
    if (g.skin) { ctx.fillStyle = 'rgba(0,0,0,0.22)'; for (let k = 0; k < 3; k++) { ctx.beginPath(); ctx.arc(X + (k - 1) * 0.45 * R, Y + 0.1 * R, 0.16 * R, 0, 7); ctx.fill(); } }
    ctx.strokeStyle = col;
    if (g.horns) { ctx.beginPath(); ctx.moveTo(X + face * 0.5 * R, Y - 0.6 * R); ctx.lineTo(X + face * R * (0.8 + 0.3 * g.horns), Y - R * (1.1 + 0.3 * g.horns)); ctx.stroke(); }
    if (g.claws) { ctx.beginPath(); ctx.arc(X + face * 1.05 * R, Y + 0.35 * R, 0.25 * R * g.claws, face > 0 ? -1 : 2, face > 0 ? 1.5 : 4.6); ctx.stroke(); }
    // eye and mouth
    ctx.fillStyle = '#ffffff'; ctx.beginPath(); ctx.arc(X + face * 0.45 * R, Y - 0.2 * R, Math.max(1, 0.24 * R), 0, 7); ctx.fill();
    ctx.fillStyle = '#10181a'; ctx.beginPath(); ctx.arc(X + face * 0.5 * R, Y - 0.2 * R, Math.max(0.6, 0.12 * R), 0, 7); ctx.fill();
    if (a.sleeping) { ctx.strokeStyle = '#10181a'; ctx.beginPath(); ctx.moveTo(X + face * 0.3 * R, Y - 0.2 * R); ctx.lineTo(X + face * 0.65 * R, Y - 0.2 * R); ctx.stroke(); }
    if (g.jaws) { ctx.strokeStyle = '#10181a'; ctx.lineWidth = Math.max(1, 0.12 * R); ctx.beginPath(); ctx.moveTo(X + face * 0.95 * R, Y + 0.15 * R); ctx.lineTo(X + face * 0.5 * R, Y + 0.25 * R); ctx.stroke(); }
    ctx.globalAlpha = 1;
    if (a.sleeping && s > 1.4) { ctx.fillStyle = pal.ink; ctx.font = `700 ${Math.round(9 + 2 * s)}px ${getComputedStyle(document.body).fontFamily}`; ctx.fillText('z', X + R, Y - R); }
  }

  function draw(now) {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (state.follow && state.selected && state.selected.alive) { cam.x = state.selected.x; cam.y = state.selected.y; }
    clampCam();
    drawScene();
    drawFood();
    for (const a of sim.yumon) {
      if (!a.alive) continue;
      for (const X of copies(a.x, 20)) {
        drawYumon(a, X, sy(a.y), now);
        if (a === state.selected) {
          ctx.strokeStyle = pal.panel; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(X, sy(a.y), a.t.r * scale() * 1.8 + 4, 0, 7); ctx.stroke();
          ctx.strokeStyle = pal.accent; ctx.setLineDash([4, 3]);
          ctx.beginPath(); ctx.arc(X, sy(a.y), a.t.sense * scale(), 0, 7); ctx.stroke(); ctx.setLineDash([]);
        }
      }
    }
  }

  // ---------------------------------------------------------------- input
  let drag = null;
  canvas.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, cx: cam.x, cy: cam.y, moved: false }; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointermove', (e) => {
    const r = canvas.getBoundingClientRect(), [wx, wy] = toWorld(e.clientX - r.left, e.clientY - r.top);
    $('hover').textContent = `x ${Math.round(((wx % W0) + W0) % W0)} · ${sim.BANDS[sim.bandOf(Math.max(0, Math.min(H0 - 1, wy)))]}`;
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 4) { drag.moved = true; state.follow = false; $('follow').checked = false; }
    cam.x = drag.cx - dx / scale(); cam.y = drag.cy - dy / scale();
    if (!state.running) draw(performance.now());
  });
  canvas.addEventListener('pointerup', (e) => {
    const d = drag; drag = null;
    if (!d || d.moved) return;
    const r = canvas.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top, [wx, wy] = toWorld(px, py);
    if (state.tool === 'feed') { toast(`Planted ${sim.feed(((wx % W0) + W0) % W0, Math.max(2, Math.min(H0 - 2, wy)))}`); return; }
    let best = null, bd = 18;
    for (const a of sim.yumon) if (a.alive) for (const X of copies(a.x, 20)) { const dd = Math.hypot(X - px, sy(a.y) - py) - a.t.r * scale(); if (dd < bd) { bd = dd; best = a; } }
    state.selected = best;
    renderInspector(); renderSpecies();
  });
  canvas.addEventListener('wheel', (e) => { e.preventDefault(); zoom(e.deltaY < 0 ? 1.2 : 1 / 1.2); }, { passive: false });
  function zoom(f) { cam.z = Math.max(1, Math.min(6, cam.z * f)); if (!state.running) draw(performance.now()); }
  $('zin').onclick = () => zoom(1.4);
  $('zout').onclick = () => zoom(1 / 1.4);
  document.querySelectorAll('input[name=tool]').forEach((r) => r.addEventListener('change', () => { state.tool = r.value; canvas.style.cursor = r.value === 'watch' ? 'grab' : 'crosshair'; }));
  $('play').onclick = () => { state.running = !state.running; $('play').textContent = state.running ? 'Pause' : 'Play'; $('play').setAttribute('aria-pressed', String(!state.running)); };
  $('stepbtn').onclick = () => { step(); draw(performance.now()); renderAll(); };
  $('speed').onchange = (e) => (state.speed = +e.target.value);
  $('follow').onchange = (e) => (state.follow = e.target.checked);
  $('learn').onchange = (e) => { sim.setLearn(e.target.checked); toast(e.target.checked ? 'Sleep learning on' : 'Sleep learning off: brains keep what they have'); };
  $('lr').onchange = (e) => { sim.T.lr = +e.target.value; toast(`Learning rate ${e.target.value}`); };
  let toastT = 0;
  function toast(msg) { const t = $('toast'); t.textContent = msg; t.hidden = false; clearTimeout(toastT); toastT = setTimeout(() => (t.hidden = true), 1800); }

  // ---------------------------------------------------------------- panels
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
  const fmt = (n) => n.toLocaleString('en-US');
  const bar = (v, warn) => `<span class="bar${warn ? ' warn' : ''}"><i style="width:${(100 * Math.max(0, Math.min(1, v))).toFixed(0)}%"></i></span>`;
  const capChips = (sp) => `<span class="chips">${[...sp.caps].map((c) => { const k = sim.CAPS.find((x) => x.id === c); return `<span class="chip k-${k.kind}">${esc(k.label)}</span>`; }).join('')}</span>`;
  let eats = new Map(), eatenBy = new Map();
  function computeWeb() {
    eats = new Map(); eatenBy = new Map();
    for (const [a, b] of sim.foodWeb()) { eats.set(a, (eats.get(a) || 0) + 1); eatenBy.set(b, (eatenBy.get(b) || 0) + 1); }
  }
  computeWeb();

  function renderInspector() {
    const a = state.selected, el = $('inspector');
    if (!a) { el.innerHTML = '<p class="muted" style="margin:0">Click a Yumon in the park to see what it feels, what its brain says next to the written rule, and how its genes have moved from its species\' founders.</p>'; return; }
    const sp = a.s, t = a.t;
    if (!a.alive) { el.innerHTML = `<p style="margin:0"><span class="dot" style="background:${hue(sp, a)}"></span> <b>${esc(sp.name)}</b> #${a.id} died (${esc({ eaten: 'eaten', starved: 'starved', old: 'old age', poison: 'poison', thorns: 'spines of its prey' }[a.cause] || a.cause)}) at generation ${a.gen}. Its brain has left the cache.</p>`; return; }
    const p = softmax(a.keys, sim.T.temp);
    const keyRows = C.options.map((o, k) => `<span${k === a.option ? ' style="font-weight:700"' : ''}>${OPT_LABEL[o]}</span>${bar(p[k])}<span class="num w">${k === a.ruleBest ? 'rule' : ''}</span>`).join('');
    const f = founder.get(sp.id);
    const genes = Object.keys(f).filter((k) => k !== 'hue' && f[k] > 0).map((k) => {
      const d = (a.genes[k] / f[k] - 1) * 100;
      return `<span class="chip"${k === a.mutated ? ' title="mutated at birth" style="border-color:var(--accent)"' : ''}>${k} ${a.genes[k].toFixed(2)} <span class="muted">${d >= 0 ? '+' : ''}${d.toFixed(0)}%</span></span>`;
    }).join('');
    const delta = sim.brainDelta(a);
    const flyNote = sp.caps.has('fly') && t.spd[0] === 0 ? '<dt>Flight</dt><dd><span class="pill bad">grounded</span> wings no longer lift it</dd>' : '';
    el.innerHTML = `
      <p style="margin:0"><span class="dot" style="background:${hue(sp, a)}"></span> <b>${esc(sp.name)}</b> #${a.id} · ${a.sex ? 'male' : 'female'} · gen ${a.gen}${a.mom ? ` · mother #${a.mom}` : ' · founder'}</p>
      <div style="margin-top:6px">${capChips(sp)}</div>
      <dl class="kv">
        <dt>Doing</dt><dd><b>${OPT_LABEL[C.options[a.option]]}</b>${a.ruleBest !== a.netBest ? ` <span class="pill mid">rule: ${OPT_LABEL[C.options[a.ruleBest]]}</span>` : ''}</dd>
        <dt>Energy</dt><dd>${bar(a.energy / t.emax, a.energy < 0.25 * t.emax)}<span class="num">${Math.round(100 * a.energy / t.emax)}%</span></dd>
        <dt>Health</dt><dd>${bar(a.hp / t.hpmax, a.hp < 0.5 * t.hpmax)}<span class="num">${Math.round(100 * a.hp / t.hpmax)}%</span></dd>
        <dt>Fatigue</dt><dd>${bar(a.fatigue / C.level)}<span class="num">${a.fatigue.toFixed(1)}</span></dd>
        <dt>Age</dt><dd>${bar(a.age / a.life)}<span class="num">${Math.round(100 * a.age / a.life)}% of life</span></dd>
        ${flyNote}
      </dl>
      <h4>Its brain's options</h4>
      <div class="keys">${keyRows}</div>
      <h4>Genes vs founders</h4>
      <div class="chips">${genes}</div>
      <h4>Brain</h4>
      <p class="num" style="margin:0">${a.exact ? '<span class="pill good">exact</span> still the compiled weights' : `<span class="pill mid">learned</span> ${a.learned} sleep updates along its line, ${a.sleeps} its own · |change| ${delta.toFixed(2)}`}</p>
      <p class="note" style="margin-top:4px">${a.learns ? '' : 'This lineage does not learn. '}${fmt(sim.brain.size)} weights, ${(sim.brain.size * 4 / 1024).toFixed(1)} KB. The rule tag marks the written rule's choice for what it senses now.</p>`;
  }
  function softmax(keys, T) { const m = Math.max(...keys), e = keys.map((k) => Math.exp((k - m) / T)), s = e.reduce((x, y) => x + y, 0); return e.map((v) => v / s); }

  function renderLearned() {
    const st = sim.stats, el = $('learned');
    const total = [...sim.changes.values()].reduce((s, v) => s + v, 0);
    const rows = [...sim.changes.entries()].sort((a, b) => b[1] - a[1]).slice(0, 7).map(([k, v]) => {
      const [r, n] = k.split('>');
      return `<li><span>${OPT_LABEL[n]} <span class="muted">where the rule says</span> ${OPT_LABEL[r]}</span><span class="num">${(100 * v / Math.max(1, st.driftN)).toFixed(1)}%</span></li>`;
    }).join('');
    el.innerHTML = `
      <dl class="kv" style="grid-template-columns: 1fr auto">
        <dt>Trained brains disagree with the rule</dt><dd class="num">${st.driftN ? (100 * st.drift / st.driftN).toFixed(1) : '0.0'}%</dd>
        <dt>Decisions on the rule's choice</dt><dd class="num">${st.decisions ? (100 * st.written / st.decisions).toFixed(1) : '0.0'}%</dd>
        <dt>Sleep updates</dt><dd class="num">${fmt(st.sleeps)}</dd>
        <dt>Untrained brains off the rule</dt><dd class="num">${fmt(st.exactMiss)} of ${fmt(st.untrained)}</dd>
      </dl>
      ${total ? `<h4>Most common learned changes</h4><ul class="learned">${rows}</ul>` : '<p class="note">No brain has slept yet. The first sleeps come a few hundred ticks in.</p>'}
      <p class="note">Shares are of all decisions by trained brains. Sampling from the softmax already strays from the rule's top choice now and then, before any learning.</p>`;
  }

  function renderSpecies() {
    const rows = sim.species.map((sp) => {
      const st = sim.speciesStats(sp), f = founder.get(sp.id);
      let drift = '';
      if (st.n) {
        let bk = null, bv = 0;
        for (const k of Object.keys(st.genes)) { const d = st.genes[k] / f[k] - 1; if (Math.abs(d) > Math.abs(bv)) { bv = d; bk = k; } }
        if (bk) drift = `${bk} ${bv >= 0 ? '+' : ''}${(100 * bv).toFixed(0)}%`;
      }
      const learned = sp.dN ? `${(100 * (sp.dD || 0) / sp.dN).toFixed(0)}%` : '-';
      const status = sp.alive ? '<span class="pill good">alive</span>' : `<span class="pill bad">extinct</span> <button type="button" data-re="${sp.id}">Reintroduce</button>`;
      const sel = state.selected && state.selected.sp === sp.id ? ' class="sel"' : sp.alive ? '' : ' class="gone"';
      return `<tr${sel}><td><span class="dot" style="background:${hue(sp)}"></span> <b>${esc(sp.name)}</b><br><span class="muted" style="font-size:12px">${sp.role === 'herb' ? 'plant eater' : sp.role === 'omni' ? 'omnivore' : 'hunter'} · ${[...sp.diet].join(', ')}</span></td><td>${capChips(sp)}</td><td class="num">${sp.alive}</td><td class="num">${sp.maxGen}</td><td class="num">${drift}</td><td class="num">${eats.get(sp.id) || 0} / ${eatenBy.get(sp.id) || 0}</td><td class="num">${learned}</td><td>${status}</td></tr>`;
    });
    $('species').innerHTML = rows.join('');
  }
  $('species').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-re]');
    if (!b) return;
    const sp = sim.species[+b.dataset.re];
    sim.reintroduce(sp); toast(`${sp.name} reintroduced`); renderSpecies();
  });

  const pc = $('pop'), pctx = pc.getContext('2d');
  function renderPop() {
    const r = pc.getBoundingClientRect(), d = Math.min(2, window.devicePixelRatio || 1);
    pc.width = Math.round(r.width * d); pc.height = Math.round(r.height * d);
    pctx.setTransform(d, 0, 0, d, 0, 0);
    const w = r.width, h = r.height, L = 34, B = 18, Tp = 6;
    pctx.clearRect(0, 0, w, h);
    const hist = sim.history;
    if (hist.length < 2) return;
    let mx = 10;
    for (const [, pops] of hist) for (const v of pops) mx = Math.max(mx, v);
    mx = Math.ceil(mx / 10) * 10;
    const t0 = hist[0][0], t1 = hist[hist.length - 1][0];
    const X = (t) => L + ((t - t0) / Math.max(1, t1 - t0)) * (w - L - 6), Y = (v) => h - B - (v / mx) * (h - B - Tp);
    pctx.strokeStyle = pal.line; pctx.fillStyle = pal.muted; pctx.lineWidth = 1; pctx.font = '11px ' + getComputedStyle(document.body).fontFamily;
    for (let k = 0; k <= 4; k++) { const v = (mx * k) / 4, y = Y(v); pctx.beginPath(); pctx.moveTo(L, y); pctx.lineTo(w - 6, y); pctx.stroke(); pctx.fillText(String(Math.round(v)), 2, y + 4); }
    pctx.fillText(`tick ${t0}`, L, h - 4); const lbl = `tick ${t1}`; pctx.fillText(lbl, w - 6 - pctx.measureText(lbl).width, h - 4);
    sim.species.forEach((sp, i) => {
      pctx.strokeStyle = hue(sp); pctx.lineWidth = state.selected && state.selected.sp === sp.id ? 2.6 : 1.3;
      pctx.beginPath();
      hist.forEach(([t, pops], j) => (j ? pctx.lineTo(X(t), Y(pops[i])) : pctx.moveTo(X(t), Y(pops[i]))));
      pctx.stroke();
    });
    const alive = sim.species.filter((s) => s.alive).length;
    $('pop-note').textContent = `${alive} of ${sim.species.length} species alive. One line per species in its colour; the selected Yumon's species is drawn thicker.`;
  }

  function renderStats() {
    const alive = sim.yumon.filter((a) => a.alive);
    $('s-pop').textContent = fmt(alive.length);
    $('s-sp').textContent = `${sim.species.filter((s) => s.alive).length}/${sim.species.length}`;
    $('s-born').textContent = fmt(sim.stats.born);
    $('s-died').textContent = fmt(sim.stats.died);
    $('s-gen').textContent = String(Math.max(...sim.species.map((s) => s.maxGen)));
    $('s-models').textContent = `${fmt(sim.models.size)} · ${((sim.models.size * sim.brain.size * 8) / 1048576).toFixed(1)} MB`;
    $('s-tick').textContent = fmt(sim.stats.tick);
  }
  function renderAll() { renderStats(); renderInspector(); renderLearned(); renderSpecies(); renderPop(); }

  // static cards
  function renderStatic() {
  $('caps').innerHTML = sim.CAPS.map((c) => `<div><span class="chip k-${c.kind}">${esc(c.label)}</span><span class="muted" style="font-size:12px">${c.limb || (c.food ? c.food : 'mind')}</span><span class="num">${sim.species.filter((s) => s.caps.has(c.id)).length}</span></div>`).join('');
  $('src').textContent = DATA.brain.source;
  $('net').textContent = `${DATA.brain.inputs.length} inputs, hidden ${JSON.stringify(DATA.brain.hidden)}, ${fmt(DATA.brain.params)} weights (${(DATA.brain.params * 4 / 1024).toFixed(1)} KB as float32)`;
  const port = sim.verifyPort();
  $('port').textContent = `Checked on load: the compiled weights and this page's copy of the rule give the Rust rule's keys on all ${port.vectors} test vectors (${port.netMismatches + port.ruleMismatches} differences). The cache holds one brain per living Yumon: a copy (current weights) plus the weights it was born with.`;

  }

  // ---------------------------------------------------------------- setup tab
  function fillSetup(st) {
    $('set-seed').value = st.seed; $('set-species').value = st.species; $('set-diets').value = st.diets;
    $('set-plants').value = String(st.plantRate); $('set-moving').value = st.moving ? 'moving' : 'fixed';
    $('set-call').value = String(st.call); $('set-learn').checked = st.learn; $('set-lr').value = String(st.lr);
    dietNote();
  }
  function readSetup() {
    const n = (id, lo, hi, d) => { const v = Math.round(+$(id).value); return Number.isFinite(v) ? Math.max(lo, Math.min(hi, v)) : d; };
    return { seed: n('set-seed', 1, 999999, 11), species: n('set-species', 8, 40, 30), diets: $('set-diets').value, plantRate: +$('set-plants').value,
      moving: $('set-moving').value === 'moving', call: +$('set-call').value, learn: $('set-learn').checked, lr: +$('set-lr').value };
  }
  const DIET_NOTES = {
    mixed: 'Half the species eat only plants, a quarter are omnivores, a quarter hunt. In headless runs 5-9 of 30 species were alive at tick 10,000.',
    hunters: 'Every species is armed and hunts, and still eats the plants of its bands. In headless runs 5 and 8 of 30 species were alive at tick 10,000, with fewer Yumon than the mixed park.',
    carnivores: 'Meat only. No species can eat plants, so the only food is other Yumon and nothing brings new energy in. In headless runs every species was extinct by tick 5,000.',
  };
  function dietNote() { $('diet-note').textContent = DIET_NOTES[$('set-diets').value]; }
  $('set-diets').addEventListener('change', dietNote);
  function describe(st) {
    return `Seed ${st.seed} · ${st.species} species · ${{ mixed: 'mixed diets', hunters: 'all hunters', carnivores: 'all carnivores' }[st.diets]} · plants ${st.plantRate}x · ${st.moving ? 'moving' : 'fixed'} patches · mating calls ${st.call}x · learning ${st.learn ? 'on (' + st.lr + ')' : 'off'}`;
  }
  function restart(st) {
    setup = st;
    try { localStorage.setItem('yumon-park-setup', JSON.stringify(st)); } catch (e) { /* not saved: still applies now */ }
    sim = makeSim(st);
    founder = new Map(sim.species.map((sp) => [sp.id, { ...sp.genes }]));
    computeWeb();
    $('learn').checked = st.learn; $('lr').value = String(st.lr);
    if (![...$('lr').options].some((o) => o.value === String(st.lr))) $('lr').value = '0.003';
    renderStatic();
    warm();
    $('current-setup').textContent = describe(st);
  }
  $('setup-form').addEventListener('submit', (e) => {
    e.preventDefault();
    restart(readSetup());
    showTab('park');
    toast('Park restarted with the new setup');
  });
  $('set-defaults').addEventListener('click', () => fillSetup(DEFAULT_SETUP));
  function showTab(name) {
    for (const b of document.querySelectorAll('[role=tab][data-view]')) b.setAttribute('aria-selected', String(b.dataset.view === name));
    $('view-park').hidden = name !== 'park';
    $('view-setup').hidden = name !== 'setup';
    if (name === 'setup') fillSetup(setup);
    else { resize(); renderPop(); }
    try { localStorage.setItem('yumon-park-tab', name); } catch (e) { /* ignore */ }
  }
  for (const b of document.querySelectorAll('[role=tab][data-view]')) b.addEventListener('click', () => showTab(b.dataset.view));

  // ---------------------------------------------------------------- loop
  // Start a little way in so the first view shows a living park.
  function warm() {
    for (let i = 0; i < 300; i++) sim.tick();
    state.selected = sim.yumon.find((a) => a.alive && a.s.carn) || sim.yumon.find((a) => a.alive) || null;
    if (state.selected) cam.x = state.selected.x;
    renderAll();
  }
  renderStatic();
  $('current-setup').textContent = describe(setup);
  warm();
  let firstTab = 'park';
  try { firstTab = location.hash === '#setup' ? 'setup' : localStorage.getItem('yumon-park-tab') || 'park'; } catch (e) { /* ignore */ }
  showTab(firstTab === 'setup' ? 'setup' : 'park');
  let panelT = 0;
  function frame(now) {
    const dt = Math.min(100, now - last); last = now;
    if (state.running) {
      acc += (dt / 1000) * TPS * state.speed;
      const t0 = performance.now();
      while (acc >= 1 && performance.now() - t0 < 30) { step(); acc -= 1; }
      if (acc > 4) acc = 0;
    }
    draw(now);
    if (now - panelT > 500 && !$('view-park').hidden) { panelT = now; renderAll(); }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
