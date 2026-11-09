(function () {
  'use strict';
  const DATA = JSON.parse(document.getElementById('ant-data').textContent);
  // Brain: the exact compiled rule, or a variant from `wir ant --variant` (e.g. a
  // superposed brain from `wir superpose-choice-layer`); picked by #brain=N, page reloads.
  const BRAINS = [{ label: 'Exact (the rule, compiled)', spec: DATA.brain }, ...(DATA.brains || []).map((b) => ({ label: b.label, spec: b }))];
  const PICK = Math.max(0, Math.min(BRAINS.length - 1, +((/brain=(\d+)/.exec(location.hash) || [])[1] || 0)));
  const BRAIN = BRAINS[PICK].spec;
  const sim = AntCore.createSim(DATA, { seed: 7, check: true, thinkEvery: 1, brain: BRAIN });
  const C = DATA.consts;
  const $ = (id) => document.getElementById(id);
  const canvas = $('world'), ctx = canvas.getContext('2d');
  const EMOTE_TOKENS = ['--em-calm', '--em-hungry', '--em-working', '--em-carrying', '--em-straining', '--em-scared', '--em-curious'];
  const EMOTE_LABELS = ['Calm', 'Hungry', 'Working', 'Carrying food', 'Straining', 'Scared', 'Curious'];
  const TASK_LABELS = { home: 'Go home', forage: 'Forage', dig: 'Dig', dump: 'Dump sand', help: 'Help carry', explore: 'Explore', flee: 'Flee' };
  const OPT = ['east', 'up', 'west', 'down', 'hold'];
  const FOOD_NAMES = { 1: 'crumb', 2: 'seed', 3: 'berry', 5: 'grasshopper' };

  // ---------------------------------------------------------------- theme colours
  let pal = {};
  function readPalette() {
    const cs = getComputedStyle(document.documentElement);
    const v = (n) => cs.getPropertyValue(n).trim();
    pal = {
      skyTop: v('--sky-top'), skyBot: v('--sky-bot'), tunnel: v('--tunnel'), rock: v('--rock'), sand: v('--sand'),
      soil: [v('--soil-1'), v('--soil-2'), v('--soil-3'), v('--soil-4')], deep: v('--soil-deep'),
      emotes: EMOTE_TOKENS.map(v), egg: v('--egg'), pebble: v('--pebble'), ink: v('--ink'), field: v('--field'), fear: v('--em-scared'),
      food: { 1: v('--food-1'), 2: v('--food-2'), 3: v('--food-3'), 5: v('--food-5') }, ring: v('--accent'),
    };
  }
  readPalette();
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', readPalette);
  new MutationObserver(readPalette).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  // ---------------------------------------------------------------- camera
  const col0 = sim.colonies[0];
  const cam = { x: col0.ex + 0.5, y: col0.top + 6, s: 11 };
  let W = 0, Hpx = 0, dpr = 1;
  function resize() {
    const r = canvas.getBoundingClientRect();
    dpr = Math.min(2, window.devicePixelRatio || 1);
    W = r.width; Hpx = r.height;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(Hpx * dpr);
  }
  new ResizeObserver(resize).observe(canvas);
  resize();
  const toCell = (px, py) => [Math.floor((px - W / 2) / cam.s + cam.x), Math.floor((py - Hpx / 2) / cam.s + cam.y)];

  // ---------------------------------------------------------------- state
  const state = { tool: 'watch', foodSize: 3, sandSize: 2, running: true, speed: 1, overlay: 'none', selected: null, follow: false };
  const prev = new Map(); // ant id -> [x, y] before the last tick
  let lastTick = performance.now(), acc = 0, tickMs = 0;
  const TPS = 10;

  function step() {
    prev.clear();
    for (const a of sim.ants) if (a.alive) prev.set(a.id, [a.x, a.y]);
    const t0 = performance.now();
    sim.tick();
    tickMs = tickMs * 0.9 + (performance.now() - t0) * 0.1;
    lastTick = performance.now();
  }

  // ---------------------------------------------------------------- drawing
  function soilColor(x, y) {
    const top = sim.surfaceAt(x), d = y - top;
    if (d > 30) return pal.deep;
    return pal.soil[(Math.floor(sim.h2(x, y, 3) * 3) + (d > 14 ? 1 : 0)) % 4];
  }
  function draw(now) {
    const s = cam.s;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const g = ctx.createLinearGradient(0, 0, 0, Hpx);
    g.addColorStop(0, pal.skyTop); g.addColorStop(1, pal.skyBot);
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, Hpx);
    const x0 = Math.floor(cam.x - W / 2 / s) - 1, x1 = Math.ceil(cam.x + W / 2 / s) + 1;
    const y0 = Math.max(0, Math.floor(cam.y - Hpx / 2 / s) - 1), y1 = Math.min(sim.H - 1, Math.ceil(cam.y + Hpx / 2 / s) + 1);
    const sx = (x) => (x - cam.x) * s + W / 2, sy = (y) => (y - cam.y) * s + Hpx / 2;
    const ov = state.overlay !== 'none' && state.overlay !== 'fear' ? fieldFor(state.overlay) : null;
    for (let x = x0; x <= x1; x++) {
      // Air under the ground line is a tunnel: a shaft's own column has no ground on top.
      const side = (d) => Math.min(sim.skyTop(x + d), sim.skyTop(x + 2 * d));
      const top = Math.min(sim.skyTop(x), Math.max(side(-1), side(1)) - 1), X = sx(x);
      for (let y = Math.max(y0, 0); y <= y1; y++) {
        const t = sim.cellType(x, y), Y = sy(y);
        if (t === sim.AIR) {
          if (y > top) { ctx.fillStyle = pal.tunnel; ctx.fillRect(X, Y, s + 0.6, s + 0.6); }
        } else {
          ctx.fillStyle = t === sim.ROCK ? pal.rock : t === sim.SAND ? pal.sand : soilColor(x, y);
          ctx.fillRect(X, Y, s + 0.6, s + 0.6);
          if (t === sim.PEB && s >= 5) { ctx.fillStyle = pal.pebble; ctx.beginPath(); ctx.arc(X + s / 2, Y + s / 2, s * 0.3, 0, 7); ctx.fill(); }
        }
        if (ov) {
          const v = sim.fget(ov, x, y);
          if (v < C.ceiling) { ctx.globalAlpha = 0.55 * (1 - v / C.ceiling); ctx.fillStyle = pal.field; ctx.fillRect(X, Y, s + 0.6, s + 0.6); ctx.globalAlpha = 1; }
        }
      }
    }
    // fear
    if (sim.fear.size) {
      ctx.fillStyle = pal.fear;
      for (const [k, v] of sim.fear) {
        const x = Math.floor(k / sim.H), y = k - x * sim.H;
        if (x < x0 || x > x1) continue;
        ctx.globalAlpha = 0.12 * v; ctx.fillRect(sx(x), sy(y), s + 0.6, s + 0.6);
      }
      ctx.globalAlpha = 1;
    }
    // items
    for (const it of sim.itemsById.values()) {
      if (it.x < x0 || it.x > x1) continue;
      const cx = sx(it.x) + s / 2, cy = sy(it.y) + s / 2, r = s * (0.22 + 0.13 * Math.sqrt(it.w));
      ctx.fillStyle = it.kind === 'food' ? pal.food[it.w] || pal.food[3] : pal.pebble;
      ctx.beginPath();
      if (it.kind === 'food' && it.w === 5) ctx.ellipse(cx, cy, r * 1.25, r * 0.7, 0, 0, 7);
      else ctx.arc(cx, cy, r, 0, 7);
      ctx.fill();
      if (it.w > 1 && s >= 8) {
        ctx.fillStyle = pal.ink; ctx.font = `600 ${Math.max(9, s * 0.8)}px "JetBrains Mono", monospace`; ctx.textAlign = 'center';
        ctx.fillText(`${it.grip.size}/${it.w}`, cx, cy - r - 3);
      }
    }
    // colonies: eggs and queen
    for (const col of sim.colonies) {
      for (const e of col.eggs) { ctx.fillStyle = pal.egg; ctx.beginPath(); ctx.ellipse(sx(e.x) + s / 2, sy(e.y) + s * 0.6, s * 0.22, s * 0.32, 0.4, 0, 7); ctx.fill(); }
      if (col.queen) drawAnt(sx(col.queen.x) + s / 2, sy(col.queen.y) + s / 2, s * 1.7, `hsl(${col.hue} 55% 42%)`, 0, 0, `hsl(${col.hue} 70% 60%)`);
    }
    for (const c of sim.corpses) { ctx.globalAlpha = Math.min(0.6, c.t / 300); ctx.fillStyle = pal.rock; ctx.fillRect(sx(c.x) + s * 0.25, sy(c.y) + s * 0.55, s * 0.5, s * 0.25); }
    ctx.globalAlpha = 1;
    // ants
    const alpha = Math.min(1, (now - lastTick) / (1000 / (TPS * state.speed)));
    for (const a of sim.ants) {
      if (!a.alive || a.x < x0 - 1 || a.x > x1 + 1) continue;
      const p = state.running ? prev.get(a.id) : null;
      const fx = p ? p[0] + (a.x - p[0]) * alpha : a.x, fy = p ? p[1] + (a.y - p[1]) * alpha : a.y;
      const jx = ((a.id * 37) % 7 - 3) * s * 0.06, jy = ((a.id * 53) % 5 - 2) * s * 0.05;
      const old = a.age > 0.55 * a.life;
      ctx.globalAlpha = old ? 0.72 : 1;
      drawAnt(sx(fx) + s / 2 + jx, sy(fy) + s / 2 + jy, s, pal.emotes[a.emote], a.opt, a.carry, null);
      ctx.globalAlpha = 1;
      if (state.selected === a.id) {
        ctx.strokeStyle = pal.ring; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(sx(fx) + s / 2, sy(fy) + s / 2, s * 1.1 + 3, 0, 7); ctx.stroke();
      }
    }
  }
  // An ant in side view: gaster, thorax, head along the heading, three legs, antennae.
  function drawAnt(cx, cy, s, color, opt, carry, crown) {
    const dir = opt === 2 ? -1 : 1, vert = opt === 1 || opt === 3;
    ctx.save();
    ctx.translate(cx, cy);
    if (vert) ctx.rotate(opt === 1 ? -Math.PI / 2 : Math.PI / 2);
    else ctx.scale(dir, 1);
    if (s < 4) { ctx.fillStyle = color; ctx.fillRect(-s * 0.45, -s * 0.2, s * 0.9, s * 0.4); ctx.restore(); return; }
    ctx.strokeStyle = color; ctx.lineWidth = Math.max(0.8, s * 0.07);
    ctx.beginPath();
    for (const lx of [-0.12, 0.02, 0.16]) { ctx.moveTo(lx * s, 0); ctx.lineTo((lx - 0.08) * s, 0.36 * s); }
    ctx.moveTo(0.42 * s, -0.08 * s); ctx.lineTo(0.56 * s, -0.3 * s);
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.ellipse(-0.27 * s, 0.02 * s, 0.2 * s, 0.15 * s, 0, 0, 7); ctx.fill();
    ctx.beginPath(); ctx.ellipse(0.04 * s, 0, 0.12 * s, 0.08 * s, 0, 0, 7); ctx.fill();
    ctx.beginPath(); ctx.arc(0.32 * s, -0.04 * s, 0.11 * s, 0, 7); ctx.fill();
    if (carry) {
      ctx.fillStyle = carry === 1 ? pal.food[1] : pal.sand;
      ctx.beginPath(); ctx.arc(0.5 * s, -0.02 * s, 0.11 * s, 0, 7); ctx.fill();
    }
    if (crown) { ctx.strokeStyle = crown; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.ellipse(-0.27 * s, 0.02 * s, 0.24 * s, 0.19 * s, 0, 0, 7); ctx.stroke(); }
    ctx.restore();
  }
  function fieldFor(name) {
    const col = sim.colonies[0];
    if (name === 'df' || name === 'dh') return sim.shared[name];
    return col[name];
  }

  // ---------------------------------------------------------------- input
  let down = null;
  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    down = { x: e.offsetX, y: e.offsetY, cx: cam.x, cy: cam.y, moved: false };
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!down) { hover(e); return; }
    const dx = e.offsetX - down.x, dy = e.offsetY - down.y;
    if (Math.abs(dx) + Math.abs(dy) > 5) down.moved = true;
    if (down.moved) { cam.x = down.cx - dx / cam.s; cam.y = clampY(down.cy - dy / cam.s); state.follow = false; $('follow').checked = false; }
  });
  canvas.addEventListener('pointerup', (e) => {
    if (down && !down.moved) apply(e.offsetX, e.offsetY);
    down = null;
  });
  canvas.addEventListener('pointercancel', () => (down = null));
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    zoomAt(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.offsetX, e.offsetY);
  }, { passive: false });
  const clampY = (y) => Math.max(0, Math.min(sim.H, y));
  function zoomAt(f, px, py) {
    const [wx, wy] = [(px - W / 2) / cam.s + cam.x, (py - Hpx / 2) / cam.s + cam.y];
    cam.s = Math.max(3, Math.min(28, cam.s * f));
    cam.x = wx - (px - W / 2) / cam.s; cam.y = clampY(wy - (py - Hpx / 2) / cam.s);
  }
  function hover(e) {
    const [x, y] = toCell(e.offsetX, e.offsetY);
    const it = sim.items.get(sim.key(x, y));
    let txt = `x ${x}, depth ${y - sim.surfaceAt(x)}`;
    if (it) txt += it.kind === 'food' ? ` - ${FOOD_NAMES[it.w] || 'food'} (needs ${it.w} ant${it.w > 1 ? 's' : ''})` : ` - pebble (needs ${it.w} ants)`;
    if (state.overlay !== 'none' && state.overlay !== 'fear') { const v = sim.fget(fieldFor(state.overlay), x, y); txt += ` - ${state.overlay} ${v >= C.ceiling ? 'out of range' : v}`; }
    $('hover').textContent = txt;
  }
  function apply(px, py) {
    const [x, y] = toCell(px, py);
    let msg = '';
    switch (state.tool) {
      case 'watch': {
        let best = null, bd = 2.5;
        for (const a of sim.ants) if (a.alive) { const d = Math.hypot(a.x - x, a.y - y); if (d < bd) { bd = d; best = a; } }
        state.selected = best ? best.id : null;
        renderInspector();
        return;
      }
      case 'food': msg = sim.tools.food(x, y, state.foodSize) ? `Dropped a ${FOOD_NAMES[state.foodSize]}` : 'That cell is solid. Drop food into open air.'; break;
      case 'sand': sim.tools.sand(x, y, state.sandSize); msg = 'Poured sand'; break;
      case 'dig': sim.tools.dig(x, y, 2); msg = 'Dug a hole'; break;
      case 'scare': sim.tools.scare(x, y, 4); msg = 'Ants nearby are scared'; break;
      case 'colony': { const c = sim.tools.colony(x); msg = c ? `Founded colony ${c.id + 1}` : 'Too close to another colony, or no room underground here.'; break; }
    }
    toast(msg);
  }
  let toastTimer = 0;
  function toast(m) { const t = $('toast'); t.textContent = m; t.hidden = !m; clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.hidden = true), 2200); }

  document.querySelectorAll('[name="tool"]').forEach((r) => r.addEventListener('change', () => { state.tool = r.value; syncToolOpts(); }));
  function syncToolOpts() {
    $('food-size').hidden = state.tool !== 'food';
    $('sand-size').hidden = state.tool !== 'sand';
    canvas.style.cursor = state.tool === 'watch' ? 'grab' : 'crosshair';
  }
  syncToolOpts();
  $('food-size').addEventListener('change', (e) => (state.foodSize = +e.target.value));
  $('sand-size').addEventListener('change', (e) => (state.sandSize = +e.target.value));
  $('play').addEventListener('click', () => { state.running = !state.running; $('play').textContent = state.running ? 'Pause' : 'Play'; $('play').setAttribute('aria-pressed', String(!state.running)); });
  $('stepbtn').addEventListener('click', () => { step(); renderStats(); renderInspector(); });
  $('speed').addEventListener('change', (e) => (state.speed = +e.target.value));
  $('overlay').addEventListener('change', (e) => (state.overlay = e.target.value));
  $('check').addEventListener('change', (e) => (sim.check.on = e.target.checked));
  $('follow').addEventListener('change', (e) => (state.follow = e.target.checked));
  $('zin').addEventListener('click', () => zoomAt(1.25, W / 2, Hpx / 2));
  $('zout').addEventListener('click', () => zoomAt(0.8, W / 2, Hpx / 2));
  $('home').addEventListener('click', () => { const c = sim.colonies.find((c) => c.alive) || sim.colonies[0]; cam.x = c.ex + 0.5; cam.y = c.top + 6; });
  $('scatter').addEventListener('click', () => {
    const c = sim.colonies.find((c) => c.alive) || sim.colonies[0];
    let n = 0;
    for (let i = 0; i < 40 && n < 16; i++) {
      const x = c.ex + Math.round((Math.random() * 2 - 1) * 45), w = [1, 1, 1, 2, 2, 3, 3, 5][Math.floor(Math.random() * 8)];
      if (Math.abs(x - c.ex) > 4 && sim.tools.food(x, sim.skyTop(x) - 1, w)) n++;
    }
    toast(`Scattered ${n} pieces of food`);
  });

  // ---------------------------------------------------------------- panels
  function renderStats() {
    const live = sim.ants.filter((a) => a.alive);
    let eggs = 0, store = 0;
    for (const c of sim.colonies) { eggs += c.eggs.length; store += c.store; }
    const col = sim.colonies[0];
    $('s-ants').textContent = live.length;
    $('s-eggs').textContent = eggs;
    $('s-store').textContent = store;
    $('s-born').textContent = sim.stats.born;
    $('s-died').textContent = sim.stats.died;
    $('s-dug').textContent = sim.stats.dug;
    $('s-hill').textContent = sim.hillHeight(col);
    $('s-tick').textContent = sim.stats.tick;
    const counts = new Array(7).fill(0);
    for (const a of live) counts[a.emote]++;
    counts.forEach((n, i) => ($('em-n-' + i).textContent = n));
    const ch = sim.check;
    const bad = ch.sense + ch.brain + ch.path;
    const shared = PICK > 0 ? ` This brain shares neurons between features, so some decisions are expected to differ (${(100 * ch.brain / Math.max(1, ch.decisions)).toFixed(2)}% so far).` : '';
    $('check-out').innerHTML = ch.decisions
      ? `<span class="pill ${bad ? 'bad' : 'good'}">${bad ? bad + ' differences' : 'No differences'}</span> ${fmt(ch.decisions)} decisions and ${fmt(ch.cells)} field cells checked against the rule as plain code.${shared} One tick takes ${tickMs.toFixed(1)} ms here.`
      : `Check is off. One tick takes ${tickMs.toFixed(1)} ms here.`;
    $('colonies').innerHTML = sim.colonies.map((c) => `<li><span class="sw" style="background:hsl(${c.hue} 55% 45%)"></span><span>Colony ${c.id + 1}</span><span class="num">${c.alive ? `${c.pop ?? 0} ants, ${c.eggs.length} eggs, ${c.store} food${c.queen ? '' : ', no queen'}` : 'gone'}</span></li>`).join('');
  }
  const fmt = (n) => n.toLocaleString('en-US');
  function bar(v, max, cls = '') { return `<span class="bar ${cls}"><i style="width:${Math.max(0, Math.min(100, (100 * v) / max))}%"></i></span>`; }
  function renderInspector() {
    const box = $('inspector');
    const a = state.selected && sim.ants.find((a) => a.id === state.selected && a.alive);
    if (!a) {
      box.innerHTML = `<p class="muted">${state.selected ? 'That ant has died.' : 'Pick the Watch tool and click an ant to see what its network decided and why.'}</p>`;
      return;
    }
    const p = a.last ? sim.brainParts(a.last.input) : null;
    const agree = p && a.last && p.option === a.last.option && p.emote === a.last.emote && p.actions.every((b, i) => b === a.last.actions[i]);
    const role = C.roles[a.role];
    const carry = a.grip ? 'gripping a heavy item' : a.carry === 1 ? 'carrying food' : a.carry === 2 ? 'carrying sand' : 'nothing';
    const dv = p ? p.drives : {};
    const max = 18;
    const acts = a.last ? C.actions.filter((_, i) => a.last.actions[i]) : [];
    box.innerHTML = `
      <div class="ant-head"><span class="dot" style="background:${pal.emotes[a.emote]}"></span><strong>Ant ${a.id}</strong><span class="muted">${role}, colony ${a.col + 1}</span></div>
      <dl class="kv">
        <dt>Feeling</dt><dd>${EMOTE_LABELS[a.emote]}</dd>
        <dt>Task</dt><dd>${p ? TASK_LABELS[C.tasks[p.task]] : '-'}</dd>
        <dt>Decision</dt><dd>${a.last ? (a.last.option === 4 ? 'hold' : 'move ' + OPT[a.last.option]) : '-'}${acts.length ? ', ' + acts.join(', ') : ''}</dd>
        <dt>Holding</dt><dd>${carry}</dd>
        <dt>Hunger</dt><dd>${bar(a.hunger, C.hunger, a.hunger >= C.starving ? 'warn' : '')}<span class="num">${a.hunger}/${C.hunger}</span></dd>
        <dt>Age</dt><dd>${bar(a.age, a.life)}<span class="num">${Math.round((100 * a.age) / a.life)}%</span></dd>
      </dl>
      <h4>Drives (the strongest one is the task)</h4>
      <dl class="kv drives">${C.tasks.map((t) => `<dt>${TASK_LABELS[t]}</dt><dd>${bar(dv[t] || 0, max, p && C.tasks[p.task] === t ? 'on' : '')}<span class="num">${dv[t] || 0}</span></dd>`).join('')}</dl>
      <p class="agree">${p ? (agree ? '<span class="pill good">Network = rule</span>' : '<span class="pill bad">Network differs from rule</span>') : ''}</p>`;
  }

  // ---------------------------------------------------------------- networks panel and sources
  const port = sim.verifyPort();
  const nz = (spec) => spec.layers.reduce((n, l) => n + l.rows.reduce((m, r) => m + r.length / 2, 0), 0);
  $('nets').innerHTML = [
    ['Path step (one per field)', `${DATA.path.taps.length} taps, hidden ${DATA.path.hidden.join('-')}`, DATA.path.params, nz(DATA.path)],
    ['Senses (grid program)', `${DATA.sense.taps.length} taps, hidden ${DATA.sense.hidden.join('-')}`, DATA.sense.params, nz(DATA.sense)],
    ['Brain', `${BRAIN.inputs.length} inputs, hidden ${BRAIN.hidden.join('-')}`, BRAIN.params, nz(BRAIN)],
  ].map(([n, shape, p, z]) => `<tr><th scope="row">${n}<small>${shape}</small></th><td class="num">${fmt(p)}</td><td class="num">${fmt(z)}</td></tr>`).join('');
  $('port').textContent = `This page's JavaScript copy of the rule matched ${port.sense}/${port.senseTotal} sense and ${port.brain}/${port.brainTotal} brain test vectors from the Rust reference before the colony started.`;
  const sources = { brain: BRAIN.source, sense: DATA.sense.source, path: DATA.path.source };
  const pickEl = $('brain-pick');
  pickEl.innerHTML = BRAINS.map((b, i) => `<option value="${i}"${i === PICK ? ' selected' : ''}>${b.label}</option>`).join('');
  pickEl.disabled = BRAINS.length < 2;
  pickEl.addEventListener('change', () => { location.hash = 'brain=' + pickEl.value; location.reload(); });
  function showSource(k) {
    $('src').textContent = sources[k];
    document.querySelectorAll('[data-src]').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.src === k)));
  }
  document.querySelectorAll('[data-src]').forEach((b) => b.addEventListener('click', () => showSource(b.dataset.src)));
  showSource('brain');
  $('legend').innerHTML = EMOTE_LABELS.map((l, i) => `<li><span class="dot" style="background:var(${EMOTE_TOKENS[i]})"></span>${l}<span class="num" id="em-n-${i}">0</span></li>`).join('');

  // ---------------------------------------------------------------- loop
  let lastFrame = performance.now(), panelT = 0;
  function frame(now) {
    const dt = Math.min(250, now - lastFrame);
    lastFrame = now;
    if (state.running) {
      acc += dt;
      const per = 1000 / (TPS * state.speed);
      let n = 0;
      while (acc >= per && n < 6) { step(); acc -= per; n++; }
      if (n === 6) acc = 0;
    }
    if (state.follow && state.selected) {
      const a = sim.ants.find((a) => a.id === state.selected && a.alive);
      if (a) { cam.x += (a.x + 0.5 - cam.x) * 0.15; cam.y = clampY(cam.y + (a.y + 0.5 - cam.y) * 0.15); }
    }
    draw(now);
    if (now - panelT > 300) { panelT = now; renderStats(); renderInspector(); }
    requestAnimationFrame(frame);
  }
  // A short warm-up so the page opens on a working colony.
  for (let i = 0; i < 400; i++) step();
  renderStats();
  requestAnimationFrame(frame);
})();
