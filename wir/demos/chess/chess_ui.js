// WIR Chess page: the board, the engine (in a worker when available) and the lens.
(function () {
  'use strict';
  const C = window.ChessCore;
  const data = JSON.parse(document.getElementById('chess-data').textContent);
  const twinData = { pstMg: data.pst_mg, pstEg: data.pst_eg };
  const eng = C.createEngine(data, { twinData });
  const outIdx = Object.fromEntries(data.outputs.map((o, i) => [o, i]));
  const defaults = Object.fromEntries(data.style);
  const GLYPH = ['', '♟', '♞', '♝', '♜', '♛', '♚'];
  const DIRNAMES = ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW'];
  const LEVELS = [
    ['Instinct', 'The network looks once at every legal move and plays the one it likes best. No lookahead at all.'],
    ['1 ply', 'Looks one move ahead, then follows captures until the position is quiet. The network judges every position it reaches.'],
    ['2 ply', 'Looks two moves ahead (its move and your reply), then follows captures.'],
    ['3 ply', 'Looks three moves ahead. Slower; uses the think time.'],
  ];
  const STYLE_INFO = {
    pst: ['Material', 0, 2, 0.05, 'Material and piece placement, blended from opening to endgame'],
    mobility: ['Mobility', 0, 12, 0.5, 'Per attacked square not holding an own piece'],
    king: ['King attack', 0, 24, 1, 'Per attack on squares next to a king'],
    hang_us: ['Own hanging', -1, 0, 0.05, 'Per centipawn of own pieces left hanging'],
    hang_them: ['Their hanging', 0, 1, 0.05, 'Per centipawn of enemy pieces hanging'],
    passed: ['Passed pawns', 0, 3, 0.1, 'Per point of passed pawn bonus'],
    rooks: ['Open files', 0, 50, 2, 'Per rook on a file with no pawns'],
  };

  const store = {
    get(k, d) { try { const v = localStorage.getItem('wirchess.' + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem('wirchess.' + k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
  };

  const S = {
    pos: C.fromFen(C.START),
    moves: [],        // move ints from the start
    sans: [],
    human: C.WHITE,
    flipped: false,
    level: store.get('level', 2),
    time: store.get('time', 4),
    style: Object.assign({}, defaults, store.get('style', {})),
    sel: -1,
    promo: null,
    thinking: false,
    last: null,
    lens: { attack: store.get('lens.attack', false), hang: store.get('lens.hang', true), rays: store.get('lens.rays', false) },
    raySquare: -1,
    search: null,
    reqId: 0,
  };
  Object.assign(eng.style, S.style);

  // ---------------------------------------------------------------- engine worker
  let worker = null;
  const pending = new Map();
  try {
    const src = document.getElementById('chess-core-src').textContent + '\n;(' + workerMain.toString() + ')();';
    worker = new Worker(URL.createObjectURL(new Blob([src], { type: 'text/javascript' })));
    worker.onmessage = e => { const cb = pending.get(e.data.id); if (cb) { pending.delete(e.data.id); cb(e.data); } };
    worker.onerror = () => { worker = null; };
    worker.postMessage({ type: 'init', data });
  } catch { worker = null; }
  function workerMain() {
    let engine = null;
    self.onmessage = e => {
      const m = e.data;
      if (m.type === 'init') { engine = self.ChessCore.createEngine(m.data); return; }
      Object.assign(engine.style, m.style);
      const pos = self.ChessCore.fromFen(self.ChessCore.START);
      for (const mv of m.moves) self.ChessCore.make(pos, mv);
      const before = engine.stats.evals;
      const r = engine.search(pos, m.opts);
      self.postMessage({ id: m.id, move: r.move, score: r.score, depth: r.depth, nodes: r.nodes, evals: engine.stats.evals - before, ms: r.ms });
    };
  }
  function think(opts) {
    const id = ++S.reqId;
    return new Promise(resolve => {
      if (worker) {
        pending.set(id, resolve);
        worker.postMessage({ id, moves: S.moves, style: S.style, opts });
      } else {
        setTimeout(() => {
          const before = eng.stats.evals;
          const r = eng.search(S.pos, opts);
          resolve({ id, move: r.move, score: r.score, depth: r.depth, nodes: r.nodes, evals: eng.stats.evals - before, ms: r.ms });
        }, 30);
      }
    });
  }

  // ---------------------------------------------------------------- board
  const boardEl = document.getElementById('board');
  const squares = [];
  for (let i = 0; i < 64; i++) {
    const b = document.createElement('button');
    b.className = 'sq';
    b.type = 'button';
    b.addEventListener('click', () => clickSquare(b._sq));
    b.addEventListener('mouseenter', () => { if (S.lens.rays) { S.raySquare = b._sq; drawRays(); } });
    boardEl.appendChild(b);
    squares.push(b);
  }
  const svgNS = 'http://www.w3.org/2000/svg';
  const rays = document.createElementNS(svgNS, 'svg');
  rays.setAttribute('class', 'rays');
  rays.setAttribute('viewBox', '0 0 8 8');
  boardEl.appendChild(rays);

  const cell = s => { const f = s & 7, r = s >> 3; return S.flipped ? [7 - f, r] : [f, 7 - r]; };

  let lensCache = { hash: null };
  function lens() {
    if (lensCache.hash === S.pos.hash) return lensCache;
    const out = eng.net.run(S.pos.b, { record: true });
    const get = k => Array.from({ length: 64 }, (_, s) => out.output(outIdx[k], s));
    lensCache = { hash: S.pos.hash, attW: get('att_w'), attB: get('att_b'), hangW: get('hanging_w'), hangB: get('hanging_b'), attn: out.attention };
    return lensCache;
  }

  function render() {
    const pos = S.pos;
    const L = lens();
    const legal = S.sel >= 0 ? C.legal(pos).filter(m => C.mFrom(m) === S.sel) : [];
    const checkSq = C.inCheck(pos) ? pos.b.indexOf(pos.side === C.WHITE ? 6 : 12) : -1;
    for (let i = 0; i < 64; i++) {
      const s = S.flipped ? 63 - (i ^ 56) : i ^ 56;
      const el = squares[i];
      el._sq = s;
      const f = s & 7, r = s >> 3;
      el.className = 'sq ' + ((f + r) % 2 ? 'l' : 'd') + (s === S.sel ? ' sel' : '') + (S.last && (s === C.mFrom(S.last) || s === C.mTo(S.last)) ? ' last' : '') + (s === checkSq ? ' check' : '');
      const p = pos.b[s];
      let html = '';
      if (S.lens.attack) {
        const w = Math.round(L.attW[s]), b = Math.round(L.attB[s]);
        const net = w - b;
        if (w || b) {
          const a = Math.min(1, 0.25 + 0.2 * Math.abs(net));
          html += `<span class="ctl" style="background:${net > 0 ? 'var(--white-ctl)' : net < 0 ? 'var(--black-ctl)' : 'transparent'};opacity:${net ? a : 0}"></span>`;
          html += `<span class="num">${w}:${b}</span>`;
        }
      }
      if ((S.flipped ? r === 7 : r === 0)) html += `<span class="coord f">${'abcdefgh'[f]}</span>`;
      if ((S.flipped ? f === 7 : f === 0)) html += `<span class="coord r">${r + 1}</span>`;
      if (p) html += `<span class="pc ${p <= 6 ? 'w' : 'b'}">${GLYPH[(p - 1) % 6 + 1]}︎</span>`;
      if (S.lens.hang && (Math.round(L.hangW[s]) || Math.round(L.hangB[s]))) html += '<span class="hang"></span>';
      const m = legal.find(x => C.mTo(x) === s);
      if (m) html += `<span class="dot${p || C.mFlag(m) === 2 ? ' cap' : ''}"></span>`;
      el.innerHTML = html;
      el.setAttribute('aria-label', `${C.name(s)}${p ? ' ' + 'PNBRQK'[(p - 1) % 6] + (p <= 6 ? ' white' : ' black') : ''}`);
    }
    drawRays();
    renderStatus();
    renderTerms();
    renderMoves();
    renderLegend();
  }

  function drawRays() {
    rays.innerHTML = '';
    if (!S.lens.rays) return;
    let q = S.raySquare >= 0 ? S.raySquare : S.sel;
    if (q < 0) q = S.pos.b.indexOf(S.pos.side === C.WHITE ? 6 : 12);
    const L = lens();
    const [qx, qy] = cell(q);
    const mk = (tag, attrs) => { const e = document.createElementNS(svgNS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); rays.appendChild(e); return e; };
    mk('rect', { x: qx + 0.06, y: qy + 0.06, width: 0.88, height: 0.88, fill: 'none', stroke: 'var(--accent)', 'stroke-width': 0.08, rx: 0.08 });
    for (let d = 0; d < 8; d++) {
      const j = L.attn[d][q + 1];
      if (j === 0) continue; // the head read BOS: nothing on this ray
      const [kx, ky] = cell(j - 1);
      mk('line', { x1: qx + 0.5, y1: qy + 0.5, x2: kx + 0.5, y2: ky + 0.5, stroke: 'var(--accent)', 'stroke-width': 0.07, 'stroke-linecap': 'round', opacity: 0.9 });
      mk('circle', { cx: kx + 0.5, cy: ky + 0.5, r: 0.14, fill: 'var(--accent)' });
      const t = mk('text', { x: kx + 0.5, y: ky + 0.24, 'text-anchor': 'middle', 'font-size': 0.2, fill: 'var(--ink)', 'font-family': 'IBM Plex Mono, monospace' });
      t.textContent = DIRNAMES[d];
    }
  }

  function renderLegend() {
    const el = document.getElementById('legend');
    const parts = [];
    if (S.lens.attack) parts.push('<span><i style="background:var(--white-ctl)"></i>white controls more</span>', '<span><i style="background:var(--black-ctl)"></i>black controls more</span>', '<span>numbers: white:black attackers</span>');
    if (S.lens.hang) parts.push('<span><i style="border:2px dashed var(--danger);border-radius:50%"></i>hanging: attacked and undefended, or attacked by something cheaper</span>');
    if (S.lens.rays) parts.push('<span><i style="background:var(--accent)"></i>the 8 ray heads: the key each head picked for the square under the pointer</span>');
    el.innerHTML = parts.join('');
  }

  function renderStatus() {
    const st = C.status(S.pos);
    const turn = S.pos.side === C.WHITE ? 'White' : 'Black';
    const who = S.pos.side === S.human ? 'your move' : 'network to move';
    const el = document.getElementById('status');
    el.innerHTML = `<b>${st ? st[0].toUpperCase() + st.slice(1) : turn + ' to move'}</b>` +
      (st ? '' : `<span>${who}</span>`) +
      `<span class="chip${S.thinking ? ' on' : ''}">${S.thinking ? 'thinking' : 'idle'}</span>` +
      (worker ? '' : '<span class="chip">search on page thread</span>');
    // eval rail: white's share from the network's judgment
    const e = eng.judge(S.pos) * (S.pos.side === C.WHITE ? 1 : -1);
    document.getElementById('railfill').style.height = `${(100 / (1 + Math.exp(-e / 250))).toFixed(1)}%`;
    document.getElementById('rail').title = `Network evaluation: ${(e / 100).toFixed(2)} pawns for white`;
  }

  function renderTerms() {
    const t = eng.terms(S.pos);
    let total = 0;
    const rows = C.TERMS.map((k, i) => {
      const c = S.style[k] * t[i];
      total += c;
      return `<tr><td>${STYLE_INFO[k][0]}</td><td>${fmt(t[i])}</td><td>&times; ${S.style[k]}</td><td>${fmt(c)}</td></tr>`;
    });
    document.getElementById('terms').innerHTML = `<tr><th>Term</th><th style="text-align:right">Network</th><th style="text-align:right">Weight</th><th style="text-align:right">cp</th></tr>${rows.join('')}<tr class="total"><td>Judgment</td><td></td><td></td><td>${fmt(total)}</td></tr>`;
    // instinct: depth 0 on the page thread
    const cands = document.getElementById('cands');
    if (C.status(S.pos)) { cands.innerHTML = '<span class="hint">Game over.</span>'; return; }
    const r = eng.search(S.pos, { depth: 0 });
    const list = r.scored.sort((a, b) => b[1] - a[1]).slice(0, 5);
    const top = list[0][1], low = list[list.length - 1][1];
    cands.innerHTML = list.map(([m, s]) => {
      const w = list.length > 1 && top !== low ? 15 + 85 * (s - low) / (top - low) : 100;
      return `<span>${C.san(S.pos, m)}</span><span class="bar" style="width:${w.toFixed(0)}%"></span><span>${fmt(s)}</span>`;
    }).join('');
  }

  const fmt = v => (Math.abs(v) >= 90000 ? (v > 0 ? 'mate' : '-mate') : (v >= 0 ? '+' : '') + v.toFixed(Math.abs(v) < 10 ? 1 : 0));

  function renderMoves() {
    const el = document.getElementById('moves');
    let html = '';
    for (let i = 0; i < S.sans.length; i += 2) html += `<span class="n">${i / 2 + 1}.</span><span>${S.sans[i]}</span><span>${S.sans[i + 1] || ''}</span>`;
    el.innerHTML = html || '<span class="n"></span><span class="hint">No moves yet.</span><span></span>';
    el.scrollTop = el.scrollHeight;
  }

  // ---------------------------------------------------------------- play
  function play(m) {
    S.sans.push(C.san(S.pos, m));
    C.make(S.pos, m);
    S.moves.push(m);
    S.last = m;
    S.sel = -1;
    save();
    render();
    maybeEngine();
  }

  function clickSquare(s) {
    if (S.thinking || C.status(S.pos) || S.pos.side !== S.human || S.promo) return;
    const p = S.pos.b[s];
    if (S.sel >= 0) {
      const opts = C.legal(S.pos).filter(m => C.mFrom(m) === S.sel && C.mTo(m) === s);
      if (opts.length > 1) { showPromo(opts); return; }
      if (opts.length === 1) { play(opts[0]); return; }
    }
    S.sel = p && C.colorOf(p) === S.pos.side ? s : -1;
    S.raySquare = S.sel;
    render();
  }

  function showPromo(opts) {
    const box = document.createElement('div');
    box.className = 'promo';
    const inner = document.createElement('div');
    for (const pr of [4, 3, 2, 1]) {
      const m = opts.find(x => C.mPromo(x) === pr);
      const b = document.createElement('button');
      b.type = 'button';
      b.innerHTML = `<span class="pc ${S.human === C.WHITE ? 'w' : 'b'}" style="font-size:40px">${GLYPH[pr + 1]}︎</span>`;
      b.setAttribute('aria-label', 'Promote to ' + ['knight', 'bishop', 'rook', 'queen'][pr - 1]);
      b.addEventListener('click', () => { box.remove(); S.promo = null; play(m); });
      inner.appendChild(b);
    }
    box.appendChild(inner);
    boardEl.appendChild(box);
    S.promo = box;
  }

  async function maybeEngine() {
    if (C.status(S.pos) || S.pos.side === S.human || S.thinking) return;
    S.thinking = true;
    renderStatus();
    const id = S.reqId + 1;
    const r = await think({ depth: S.level, timeMs: S.time * 1000, qdepth: S.level ? 4 : 0 });
    S.thinking = false;
    if (r.id !== S.reqId || S.pos.side === S.human) { render(); return; }
    void id;
    S.search = r;
    renderSearch();
    if (r.move) play(r.move); else render();
  }

  function renderSearch() {
    const r = S.search;
    const el = document.getElementById('lastsearch');
    if (!r) { el.innerHTML = '<span class="hint">The network has not moved yet.</span>'; return; }
    el.innerHTML = `<span class="chip">depth ${r.depth}</span><span class="chip">${r.nodes} positions</span><span class="chip">${r.evals} network runs</span><span class="chip">${(r.ms / 1000).toFixed(2)} s</span><span class="chip">score ${fmt(r.score)}</span>`;
  }

  function newGame() {
    S.reqId++;
    S.thinking = false;
    S.pos = C.fromFen(C.START);
    S.moves = []; S.sans = []; S.sel = -1; S.last = null; S.search = null;
    if (S.promo) { S.promo.remove(); S.promo = null; }
    S.flipped = S.human === C.BLACK;
    save();
    renderSearch();
    render();
    maybeEngine();
  }

  function undo() {
    if (!S.moves.length) return;
    S.reqId++;
    S.thinking = false;
    let n = S.pos.side === S.human ? 2 : 1;
    n = Math.min(n, S.moves.length);
    for (let i = 0; i < n; i++) { C.unmake(S.pos); S.moves.pop(); S.sans.pop(); }
    S.last = S.moves[S.moves.length - 1] || null;
    S.sel = -1;
    save();
    render();
    maybeEngine();
  }

  function save() { store.set('game', { moves: S.moves, human: S.human }); }

  function restore(saved) {
    if (!saved || !Array.isArray(saved.moves)) return;
    S.human = saved.human === C.BLACK ? C.BLACK : C.WHITE;
    S.flipped = S.human === C.BLACK;
    for (const m of saved.moves) {
      if (!C.legal(S.pos).includes(m)) break;
      S.sans.push(C.san(S.pos, m));
      C.make(S.pos, m);
      S.moves.push(m);
      S.last = m;
    }
  }

  // ---------------------------------------------------------------- controls
  const levelsEl = document.getElementById('levels');
  LEVELS.forEach(([label], i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = label;
    b.addEventListener('click', () => { S.level = i; store.set('level', i); renderLevels(); });
    levelsEl.appendChild(b);
  });
  function renderLevels() {
    [...levelsEl.children].forEach((b, i) => b.setAttribute('aria-pressed', String(i === S.level)));
    document.getElementById('levelhint').textContent = LEVELS[S.level][1];
  }
  const timeEl = document.getElementById('time');
  timeEl.value = S.time;
  const timeOut = () => { document.getElementById('timeout').textContent = `${S.time} s`; };
  timeEl.addEventListener('input', () => { S.time = +timeEl.value; store.set('time', S.time); timeOut(); });

  const slidersEl = document.getElementById('sliders');
  function renderSliders() {
    slidersEl.innerHTML = '';
    for (const k of C.TERMS) {
      const [label, lo, hi, step, title] = STYLE_INFO[k];
      const row = document.createElement('div');
      row.className = 'slider';
      row.title = title;
      row.innerHTML = `<label class="lab" for="style-${k}">${label}</label><input type="range" id="style-${k}" min="${lo}" max="${hi}" step="${step}" value="${S.style[k]}"><output>${S.style[k]}</output>`;
      const input = row.querySelector('input'), out = row.querySelector('output');
      input.addEventListener('input', () => {
        S.style[k] = +input.value;
        eng.style[k] = S.style[k];
        out.textContent = S.style[k];
        store.set('style', S.style);
        renderStatus();
        renderTerms();
      });
      slidersEl.appendChild(row);
    }
  }
  document.getElementById('style-reset').addEventListener('click', () => {
    S.style = Object.assign({}, defaults);
    Object.assign(eng.style, S.style);
    store.set('style', S.style);
    renderSliders();
    render();
  });

  for (const k of ['attack', 'hang', 'rays']) {
    const el = document.getElementById('lens-' + k);
    el.checked = S.lens[k];
    el.addEventListener('change', () => { S.lens[k] = el.checked; store.set('lens.' + k, el.checked); render(); });
  }
  boardEl.addEventListener('mouseleave', () => { S.raySquare = S.sel; drawRays(); });

  const setSide = side => {
    S.human = side;
    document.getElementById('as-white').setAttribute('aria-pressed', String(side === C.WHITE));
    document.getElementById('as-black').setAttribute('aria-pressed', String(side === C.BLACK));
    newGame();
  };
  document.getElementById('as-white').addEventListener('click', () => setSide(C.WHITE));
  document.getElementById('as-black').addEventListener('click', () => setSide(C.BLACK));
  document.getElementById('newgame').addEventListener('click', newGame);
  document.getElementById('undo').addEventListener('click', undo);
  document.getElementById('flip').addEventListener('click', () => { S.flipped = !S.flipped; render(); });

  // ---------------------------------------------------------------- the network panel
  const cost = data.cost;
  document.getElementById('netstats').innerHTML = [
    [cost.heads, 'attention heads'], [cost.blocks, 'blocks'], [cost.d_model, 'residual width'],
    [cost.nonzero.toLocaleString(), 'non-zero weights'], [data.outputs.length, 'outputs'],
  ].map(([v, l]) => `<div><b>${v}</b><span>${l}</span></div>`).join('');
  const bAtt = data.bounds[outIdx.att_w], bHang = data.bounds[outIdx.hanging_w];
  document.getElementById('netcheck').textContent =
    `Compiled from the WIR source below, no training. Proven before any run: attack counts within ${bAtt.toExponential(1)} and hanging flags within ${bHang.toExponential(1)} of the program's exact answer on every board, so rounding them is exact. Checked against a plain-code twin on 100,000 random boards and on positions from self-play games, with no differences.`;
  document.getElementById('src').textContent = data.program.split('\n').map(l => (l.length > 160 ? l.slice(0, 157) + '...' : l)).join('\n');

  // ---------------------------------------------------------------- start
  function start(hot) {
    restore(hot && hot.game ? hot.game : store.get('game', null));
    document.getElementById('as-white').setAttribute('aria-pressed', String(S.human === C.WHITE));
    document.getElementById('as-black').setAttribute('aria-pressed', String(S.human === C.BLACK));
    renderLevels();
    timeOut();
    renderSliders();
    renderSearch();
    render();
    maybeEngine();
  }
  window.claude?.hot?.snapshot?.(() => ({ game: { moves: S.moves, human: S.human } }));
  if (window.claude?.hot?.ready) window.claude.hot.ready(start); else start(window.claude?.hot?.data ?? {});
})();
