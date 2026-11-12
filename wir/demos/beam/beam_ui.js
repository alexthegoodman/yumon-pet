// WIR Beam page: drives BeamCore and draws the candidate grid. Every ranking on screen is the
// compiled network's own output; a plain-code twin checks each step.
(function () {
  'use strict';
  const B = window.BeamCore;
  const data = JSON.parse(document.getElementById('beam-data').textContent);
  const cfg = { ...data.config, maxSteps: 8 };
  const net = B.createNet(data.weights);
  const $ = id => document.getElementById(id);
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
  const say = words => (words.length ? B.text(words) : '—');

  const state = { ctx: B.START, width: 4, unit: 6, S: null, cur: null, running: null, selected: null, refs: null };

  // ---------------------------------------------------------------- controls
  const prompt = $('prompt');
  for (let c = 1; c <= B.START; c++) { const o = el('option', '', c === B.START ? '(nothing: start of sentence)' : B.WORDS[c]); o.value = c; prompt.appendChild(o); }
  prompt.value = state.ctx;
  const unitSel = $('unit');
  for (const u of [2, 4, 6]) { const o = el('option', '', `${u} per nat (1/${u})`); o.value = u; unitSel.appendChild(o); }
  unitSel.value = state.unit;
  const widthBox = $('width');
  for (let w = 1; w <= cfg.k; w++) { const b = el('button', '', String(w)); b.type = 'button'; b.dataset.w = w; b.setAttribute('aria-pressed', w === state.width); b.setAttribute('aria-label', `Width ${w}`); widthBox.appendChild(b); }

  function legend() {
    const l = $('legend'); l.textContent = '';
    const add = (cls, style, text) => { const s = el('span'); const i = el('i'); if (style) i.style.cssText = style; s.append(i, text); l.appendChild(s); };
    add('', 'background:var(--accent)', 'survives (network rank < width)');
    add('', 'background:color-mix(in srgb, var(--accent) 30%, transparent)', 'candidate, tint = cheaper');
    add('', 'background:repeating-linear-gradient(135deg, transparent 0 3px, var(--line) 3px 4px)', 'cannot be chosen (ended beam, other word)');
    l.appendChild(el('span', '', 'number = total cost of that candidate, small number = its rank'));
  }

  // ---------------------------------------------------------------- the search
  function reset() {
    stop();
    state.S = B.newSearch(cfg, state.ctx, state.width, state.unit);
    state.cur = null; state.selected = null;
    render();
  }
  function peek() {
    const S = state.S;
    const st = B.prepare(S);
    const sel = B.networkStep(net, cfg, st);
    const twin = B.twinStep(cfg, st);
    let same = true;
    for (let p = 0; p < sel.rank.length; p++) if (sel.rank[p] !== twin.rank[p]) { same = false; break; }
    return { st, sel, twin, same, beams: S.beams.slice(0, S.width).map(b => ({ ...b })) };
  }
  function step() {
    const S = state.S;
    if (S.done) return;
    const v = state.cur || peek();
    B.advance(S, () => v.sel);
    state.last = v;
    state.cur = null; state.selected = null;
    render();
  }
  function run() {
    if (state.running) { stop(); return; }
    if (state.S.done) reset();
    $('run').textContent = 'Pause';
    const tick = () => { step(); if (state.S.done) stop(); else state.running = setTimeout(tick, 650); };
    state.running = setTimeout(tick, 0);
  }
  function stop() { if (state.running) clearTimeout(state.running); state.running = null; $('run').textContent = 'Run'; }

  // ---------------------------------------------------------------- drawing
  function drawGrid(v, S, showPicks) {
    const t = $('grid'); t.textContent = '';
    const head = el('tr'); head.appendChild(el('th', 'beam', 'beam'));
    B.WORDS.forEach((w, j) => head.appendChild(el('th', '', j === 0 ? 'end' : w)));
    const thead = el('thead'); thead.appendChild(head); t.appendChild(thead);
    const body = el('tbody'); t.appendChild(body);
    const pickSet = new Set(showPicks ? v.sel.picks.slice(0, S.width).map(p => p.from * cfg.v + p.tok) : []);
    const span = cfg.cap + cfg.qmax;
    for (let i = 0; i < cfg.k; i++) {
      const b = v.beams[i], idle = !b;
      const tr = el('tr');
      const th = el('th', 'beam' + (idle ? ' idle' : '') + (b && b.fin ? ' ended' : ''));
      th.scope = 'row';
      if (idle) { th.textContent = 'idle'; } else {
        th.appendChild(document.createTextNode(say(b.words) + (b.fin ? ' · ended' : '')));
        th.appendChild(el('small', '', `+${v.st.s[i]}${v.st.s[i] >= cfg.cap ? ' (capped)' : ''} · ${b.logp.toFixed(2)} nats`));
      }
      tr.appendChild(th);
      for (let j = 0; j < cfg.v; j++) {
        const p = i * cfg.v + j;
        const cost = v.twin.cost[p];
        const dead = !idle && cost >= cfg.dead;
        const td = el('td', 'cell' + (pickSet.has(p) ? ' pick' : '') + (dead ? ' dead' : '') + (idle ? ' idle' : ''));
        td.tabIndex = idle ? -1 : 0;
        td.dataset.p = p;
        if (!idle && !dead) {
          td.textContent = cost;
          td.appendChild(el('span', 'rk', String(v.sel.rank[p])));
          td.style.setProperty('--t', String(Math.max(0, 1 - cost / span)));
          td.title = `${say(b.words)} + ${j === 0 ? 'end' : B.WORDS[j]}: cost ${cost} (beam +${v.st.s[i]}, word ${b.fin ? 0 : v.st.l[i][j]}), network rank ${v.sel.rank[p]}`;
        } else if (dead) { td.title = 'An ended beam offers only the end token.'; }
        if (state.selected === p) td.classList.add('sel');
        tr.appendChild(td);
      }
      body.appendChild(tr);
    }
  }

  function drawBeams(S) {
    const ol = $('beams'); ol.textContent = '';
    S.beams.forEach((b, i) => {
      const li = el('li');
      li.append(el('span', 'n', String(i + 1)), el('span', b.fin ? 'done' : '', say(b.words) + (b.fin ? ' • ended' : '')), el('span', '', b.logp.toFixed(2)));
      ol.appendChild(li);
    });
  }

  function status(v, S) {
    const s = $('status'); s.textContent = '';
    const b = S.beams[0];
    if (S.done) {
      const best = S.beams[0];
      s.append(el('b', '', 'Done. '), document.createTextNode(`Best beam after ${S.steps} step${S.steps === 1 ? '' : 's'}: “${say(best.words)}”${best.fin ? '' : ' (not ended; step limit)'}, ${best.logp.toFixed(2)} nats. The cheapest beam ended, so nothing can overtake it.`));
    } else if (S.steps === 0) {
      s.append(el('b', '', 'Step 1. '), document.createTextNode(`Width ${S.width}: the network has ranked ${S.width} × 16 candidates; the highlighted cells are what it keeps. Press Step to take them.`));
    } else {
      s.append(el('b', '', `Step ${S.steps + 1}. `), document.createTextNode(`Best so far “${say(b.words)}” (${b.logp.toFixed(2)} nats). Highlighted cells are the next survivors.`));
    }
  }

  function netCheck(v) {
    const n = cfg.k * cfg.v;
    const chose = v.sel.picks.slice(0, state.width).map(p => {
      const b = v.beams[p.from];
      return (b && b.words.length ? B.text(b.words) + ' ' : '') + (b && b.fin ? '(ended)' : p.tok === 0 ? 'end' : B.WORDS[p.tok]);
    }).join(', ');
    $('netcheck').textContent = v.same
      ? `This step: the network's rank for all ${n} candidates equals a plain sort by (cost, position). It keeps: ${chose}.`
      : 'This step: network and plain sort DISAGREE.';
  }

  function render() {
    const S = state.S;
    let v;
    if (S.done) v = state.last; else { v = state.cur = peek(); }
    drawGrid(v, S, true);
    drawBeams(S);
    status(v, S);
    netCheck(v);
    $('step').disabled = S.done;
    for (const b of widthBox.children) b.setAttribute('aria-pressed', String(+b.dataset.w === state.width));
    refs();
  }

  // ---------------------------------------------------------------- references
  function refs() {
    const key = state.ctx + ':' + state.unit;
    if (!state.refs || state.refs.key !== key) {
      const opt = B.optimal(state.ctx, cfg.maxSteps), gr = B.greedy(state.ctx, cfg.maxSteps);
      const net_ = [];
      for (let w = 1; w <= cfg.k; w++) { const S = B.runSearch(cfg, state.ctx, w, state.unit, B.netSelect(net, cfg)); net_.push({ S, best: S.beams[0] }); }
      state.refs = { key, opt, gr, net: net_ };
    }
    const R = state.refs;
    const t = $('refs'); t.textContent = '';
    const hd = el('tr'); ['', 'sentence', 'nats', ''].forEach(h => hd.appendChild(el('th', '', h))); t.appendChild(hd);
    const row = (label, words, logp, fin, cur, optimum) => {
      const tr = el('tr', (cur ? 'cur ' : '') + (optimum ? 'opt' : ''));
      tr.appendChild(el('td', '', label));
      tr.appendChild(el('td', '', say(words) + (fin === false ? ' (unfinished)' : '')));
      tr.appendChild(el('td', 'num', logp.toFixed(2)));
      const hit = fin !== false && Math.abs(logp - R.opt.logp) < 1e-9;
      tr.appendChild(el('td', 'mark ' + (hit ? 'ok' : 'miss'), optimum ? 'best' : hit ? '= best' : (logp - R.opt.logp).toFixed(2)));
      t.appendChild(tr);
    };
    row('exact best (DP)', R.opt.words, R.opt.logp, true, false, true);
    row('greedy', R.gr.words, R.gr.logp, R.gr.finished, false);
    R.net.forEach((r, i) => row(`network beam, width ${i + 1}`, r.best.words, r.best.logp, r.best.fin, i + 1 === state.width));
    const f = B.floatBeam(state.ctx, state.width, cfg.maxSteps);
    row(`float beam, width ${state.width}`, f.words, f.logp, f.fin, false);
    const same = R.net[state.width - 1].best.words.join() === f.words.join();
    $('refnote').textContent = `Network beams use integer costs at ${state.unit} per nat and ignore beams more than ${cfg.cap} units behind the best; float beams use exact log-probabilities. This one ${same ? 'agrees with' : 'differs from'} the float beam at width ${state.width}. A wider beam can still land on a worse sentence than a narrower one.`;
  }

  // ---------------------------------------------------------------- all prompts
  function score() {
    const btn = $('score'); btn.disabled = true;
    const hit = Array(cfg.k + 1).fill(0); let greedyHit = 0; const total = B.START;
    let c = 1;
    const prog = $('scoreprog');
    const one = () => {
      const opt = B.optimal(c, cfg.maxSteps), g = B.greedy(c, cfg.maxSteps);
      if (g.finished && Math.abs(g.logp - opt.logp) < 1e-9) greedyHit++;
      for (let w = 1; w <= cfg.k; w++) {
        const S = B.runSearch(cfg, c, w, state.unit, B.netSelect(net, cfg));
        const b = S.beams[0];
        if (b.fin && Math.abs(b.logp - opt.logp) < 1e-9) hit[w]++;
      }
      prog.textContent = `${c} of ${total}`;
      c++;
      if (c <= total) { setTimeout(one, 0); return; }
      const t = $('scoreboard'); t.textContent = '';
      const hd = el('tr'); ['search', `exact best (of ${total})`].forEach(h => hd.appendChild(el('th', '', h))); t.appendChild(hd);
      const add = (label, n) => { const tr = el('tr'); tr.appendChild(el('td', '', label)); tr.appendChild(el('td', 'num', `${n} / ${total}`)); t.appendChild(tr); };
      add('greedy', greedyHit);
      for (let w = 1; w <= cfg.k; w++) add(`network beam, width ${w}`, hit[w]);
      prog.textContent = `at ${state.unit} per nat`;
      btn.disabled = false;
    };
    setTimeout(one, 0);
  }

  // ---------------------------------------------------------------- network facts
  function facts() {
    const s = $('netstats'); s.textContent = '';
    const add = (n, l) => { const d = el('div'); d.append(el('b', '', n), el('span', '', l)); s.appendChild(d); };
    add(`${cfg.k} × ${cfg.v}`, 'beams × words');
    add(String(data.cost.blocks), 'blocks');
    add(String(data.cost.heads), 'attention heads');
    add(data.cost.nonzero.toLocaleString('en-US'), 'non-zero weights');
    add(Math.max(...data.bounds).toExponential(0), 'proven worst error');
    $('src').textContent = data.program;
  }

  // ---------------------------------------------------------------- events
  prompt.addEventListener('change', () => { state.ctx = +prompt.value; reset(); });
  unitSel.addEventListener('change', () => { state.unit = +unitSel.value; reset(); });
  widthBox.addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; state.width = +b.dataset.w; reset(); });
  $('step').addEventListener('click', () => { stop(); step(); });
  $('run').addEventListener('click', run);
  $('reset').addEventListener('click', reset);
  $('score').addEventListener('click', score);
  $('grid').addEventListener('click', e => {
    const td = e.target.closest('td.cell'); if (!td || td.classList.contains('idle')) return;
    state.selected = +td.dataset.p; for (const c of $('grid').querySelectorAll('td.sel')) c.classList.remove('sel'); td.classList.add('sel');
  });

  legend(); facts(); reset();
  window.__beam = { state, B, net, cfg };
})();
