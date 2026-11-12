// WIR Beam core: a toy bigram language model (host), the compiled WIR network that picks
// each step's beams (selection), a plain-code twin of the network for checks, and the
// reference searches (greedy, float beam, exact dynamic programming). Browser and Node.
(function (root) {
  'use strict';

  // ------------------------------------------------------------- language model
  // Sixteen words; id 0 is the end token (also the pad of a finished beam). Rows are
  // unnormalized counts of the next word; the host turns them into costs.
  const WORDS = ['<end>', 'the', 'a', 'yumon', 'pet', 'desk', 'small', 'happy', 'sleeps', 'plays', 'eats', 'on', 'softly', 'quietly', 'and', 'wakes'];
  const START = 16; // context row for "nothing said yet"
  const COUNTS = {
    '<s>': { the: 45, a: 20, yumon: 30, small: 5 },
    the: { small: 25, pet: 25, yumon: 15, desk: 20, happy: 15 },
    a: { small: 40, happy: 30, pet: 30 },
    yumon: { sleeps: 55, plays: 20, eats: 15, and: 10 },
    small: { pet: 35, yumon: 30, desk: 20, happy: 15 },
    happy: { pet: 40, yumon: 30, small: 15, desk: 15 },
    pet: { sleeps: 30, plays: 30, eats: 20, on: 10, and: 10 },
    desk: { '<end>': 30, and: 20, sleeps: 10, plays: 10, eats: 10, wakes: 20 },
    sleeps: { '<end>': 40, on: 20, softly: 20, quietly: 10, and: 10 },
    plays: { '<end>': 30, on: 15, softly: 10, quietly: 15, and: 30 },
    eats: { '<end>': 35, on: 10, softly: 10, quietly: 5, and: 40 },
    on: { the: 60, a: 30, desk: 10 },
    softly: { '<end>': 50, and: 50 },
    quietly: { '<end>': 60, and: 40 },
    and: { sleeps: 25, plays: 25, eats: 25, wakes: 25 },
    wakes: { '<end>': 70, softly: 15, quietly: 15 },
  };
  const FLOOR = 0.05; // smoothing so every word has some probability

  // logp[context][word], contexts 0..15 are words (0 = <end> is never a context) and 16 = start.
  const LOGP = [];
  for (let c = 0; c <= START; c++) {
    const row = COUNTS[c === START ? '<s>' : WORDS[c]] || {};
    const raw = WORDS.map(w => (row[w] || 0) + FLOOR);
    const z = raw.reduce((a, b) => a + b, 0);
    LOGP.push(raw.map(v => Math.log(v / z)));
  }
  const contextName = c => (c === START ? '(start)' : WORDS[c]);

  // ------------------------------------------------------------- network
  function sparse(m) {
    const rows = Array.from({ length: m.rows }, () => []);
    const cols = Array.from({ length: m.cols }, () => []);
    for (const [r, c, v] of m.nz) { rows[r].push(c, v); cols[c].push(r, v); }
    return { rows: rows.map(r => Float64Array.from(r)), cols: cols.map(c => Float64Array.from(c)), nrows: m.rows, ncols: m.cols };
  }
  const SKIP = 36; // keys this far below a query's best score are skipped (weight < e^-36)

  function createNet(w) {
    const d = w.dims.length, n = w.len + 1;
    const embeds = w.embeds.map(e => ({ lo: e.lo, t: sparse(e.table) }));
    const posE = sparse(w.pos_embed);
    const blocks = w.blocks.map(b => ({
      heads: b.heads.map(h => {
        const dh = h.wq.rows, nv = h.wv.rows;
        return { q: sparse(h.wq), k: sparse(h.wk), v: sparse(h.wv), o: sparse(h.wo), Q: new Float64Array(n * dh), Kd: new Float64Array(n * dh), V: new Float64Array(n * nv), K: Array.from({ length: n }, () => []) };
      }),
      win: sparse(b.w_in), wout: sparse(b.w_out),
    }));
    const unembed = sparse(w.unembed);
    const x = new Float64Array(n * d);
    const dotRow = (row, base) => { let s = 0; for (let i = 0; i < row.length; i += 2) s += row[i + 1] * x[base + row[i]]; return s; };
    for (const blk of blocks) for (const h of blk.heads) {
      const cols = new Set();
      for (const m of [h.q, h.k, h.v]) m.cols.forEach((c, i) => { if (c.length) cols.add(i); });
      h.read = Int32Array.from([...cols].sort((a, b) => a - b));
    }
    // tokens: [position][input]. Returns { at(p) -> all outputs, output(o, p), attention }.
    function run(tokens, opts = {}) {
      x.fill(0);
      const add = (m, r, base) => { const row = m.rows[r]; for (let i = 0; i < row.length; i += 2) x[base + row[i]] += row[i + 1]; };
      embeds.forEach(e => add(e.t, 0, 0));
      add(posE, 0, 0);
      for (let s = 0; s < w.len; s++) {
        embeds.forEach((e, k) => add(e.t, 1 + tokens[s][k] - e.lo, (s + 1) * d));
        add(posE, 1 + s, (s + 1) * d);
      }
      const record = opts.record ? [] : null;
      const scores = new Float64Array(n);
      for (const blk of blocks) {
        const outs = [];
        for (const h of blk.heads) {
          const dh = h.q.nrows, nv = h.v.nrows;
          const { Q, Kd, V, K, read } = h;
          Q.fill(0); Kd.fill(0); V.fill(0);
          for (let i = 0; i < n; i++) {
            const base = i * d, qo = i * dh, vo = i * nv;
            for (let t = 0; t < read.length; t++) {
              const c = read[t], xv = x[base + c];
              if (xv === 0) continue;
              let col = h.q.cols[c];
              for (let k = 0; k < col.length; k += 2) Q[qo + col[k]] += col[k + 1] * xv;
              col = h.k.cols[c];
              for (let k = 0; k < col.length; k += 2) Kd[qo + col[k]] += col[k + 1] * xv;
              col = h.v.cols[c];
              for (let k = 0; k < col.length; k += 2) V[vo + col[k]] += col[k + 1] * xv;
            }
            const kk = K[i];
            kk.length = 0;
            for (let r = 0; r < dh; r++) { const v = Kd[qo + r]; if (v !== 0) kk.push(r, v); }
          }
          const pattern = record ? new Int16Array(n) : null;
          const res = new Float64Array(n * nv);
          for (let i = 0; i < n; i++) {
            let m = -Infinity;
            const qb = i * dh;
            for (let j = 0; j < n; j++) {
              const kk = K[j];
              let s = 0;
              for (let t = 0; t < kk.length; t += 2) s += kk[t + 1] * Q[qb + kk[t]];
              scores[j] = s;
              if (s > m) { m = s; if (pattern) pattern[i] = j; }
            }
            let z = 0;
            const ro = i * nv;
            for (let j = 0; j < n; j++) {
              if (scores[j] < m - SKIP) continue;
              const e = Math.exp(scores[j] - m);
              z += e;
              for (let r = 0; r < nv; r++) res[ro + r] += e * V[j * nv + r];
            }
            for (let r = 0; r < nv; r++) res[ro + r] /= z;
          }
          outs.push(res);
          if (record) record.push(pattern);
        }
        blk.heads.forEach((h, hi) => {
          const res = outs[hi], nv = h.v.nrows;
          for (let i = 0; i < n; i++) for (let r = 0; r < nv; r++) {
            const y = res[i * nv + r];
            if (y === 0) continue;
            const col = h.o.cols[r];
            for (let t = 0; t < col.length; t += 2) x[i * d + col[t]] += col[t + 1] * y;
          }
        });
        const units = blk.win.nrows;
        if (!units) continue;
        const hid = new Float64Array(units);
        for (let i = 0; i < n; i++) {
          const base = i * d;
          for (let u = 0; u < units; u++) hid[u] = dotRow(blk.win.rows[u], base);
          for (let u = 0; u < units; u++) {
            const hval = hid[u];
            if (hval <= 0) continue;
            const col = blk.wout.cols[u];
            for (let t = 0; t < col.length; t += 2) x[base + col[t]] += col[t + 1] * hval;
          }
        }
      }
      return {
        at(p) { const base = (p + 1) * d; return unembed.rows.map(r => dotRow(r, base)); },
        output(o, p) { return dotRow(unembed.rows[o], (p + 1) * d); },
        attention: record,
      };
    }
    return { run, d, len: w.len, outputs: w.outputs };
  }

  // ------------------------------------------------------------- one step: network and twin
  // A step: s[beam] (relative cost), l[beam][token], fin[beam]. Candidate position p = beam * v + token.
  function candidateCosts(cfg, st) {
    const out = [];
    for (let i = 0; i < cfg.k; i++) for (let j = 0; j < cfg.v; j++) {
      out.push(st.fin[i] ? (j === 0 ? st.s[i] : cfg.dead) : st.s[i] + st.l[i][j]);
    }
    return out;
  }
  // Plain-code twin: ranks by (cost, position).
  function twinStep(cfg, st) {
    const c = candidateCosts(cfg, st);
    const order = c.map((_, p) => p).sort((a, b) => c[a] - c[b] || a - b);
    const rank = new Array(c.length);
    order.forEach((p, r) => { rank[p] = r; });
    return { cost: c, order, rank };
  }
  function stepTokens(cfg, st) {
    const t = [];
    for (let i = 0; i < cfg.k; i++) for (let j = 0; j < cfg.v; j++) t.push([st.s[i], st.l[i][j], st.fin[i]]);
    return t;
  }
  // The network's answer: picks[r] = { cost, from, tok } for the best k slots, rank[p] for every candidate.
  function networkStep(net, cfg, st, opts) {
    const out = net.run(stepTokens(cfg, st), opts);
    const idx = Object.fromEntries(net.outputs.map((o, i) => [o.name || o, i]));
    const rank = [], picks = [];
    for (let p = 0; p < cfg.k * cfg.v; p++) {
      const o = out.at(p);
      rank.push(Math.round(o[idx.rank]));
      if (p < cfg.k) picks.push({ cost: Math.round(o[idx.cost]), from: Math.round(o[idx.from]), tok: Math.round(o[idx.tok]), raw: [o[idx.cost], o[idx.from], o[idx.tok]] });
    }
    return { picks, rank, attention: out.attention };
  }

  // ------------------------------------------------------------- the beam search host
  // Costs are integers: round(unit * -ln p), at most qmax. A live beam's `cum` is its integer
  // cost so far; the network sees it relative to the best beam, clipped at cap.
  function quant(cfg, unit, lp) { return Math.min(cfg.qmax, Math.round(unit * -lp)); }

  function newSearch(cfg, context, width, unit) {
    return {
      cfg, width, unit, context, steps: 0, done: false,
      beams: [{ words: [], last: context, cum: 0, logp: 0, fin: false }],
      history: [],
    };
  }

  // The network's input for the next step: costs relative to the best live beam, clipped at cap.
  function prepare(S) {
    const { cfg, width, unit } = S;
    const live = S.beams.slice(0, width);
    const best = Math.min(...live.map(b => b.cum));
    const st = { s: [], l: [], fin: [] };
    for (let i = 0; i < cfg.k; i++) {
      const b = live[i];
      if (!b) { st.s.push(cfg.cap); st.l.push(new Array(cfg.v).fill(cfg.qmax)); st.fin.push(0); continue; }
      st.s.push(Math.min(cfg.cap, b.cum - best));
      st.fin.push(b.fin ? 1 : 0);
      st.l.push(b.fin ? new Array(cfg.v).fill(0) : LOGP[b.last].map(lp => quant(cfg, unit, lp)));
    }
    return st;
  }

  // One step with a given selection function (network or twin); returns the step record.
  function advance(S, select) {
    const { width } = S;
    const live = S.beams.slice(0, width);
    const st = prepare(S);
    const sel = select(st);
    const next = [];
    for (let r = 0; r < width; r++) {
      const pk = sel.picks[r];
      const parent = live[pk.from];
      const fin = parent.fin || pk.tok === 0;
      if (parent.fin) { next.push({ ...parent }); continue; }
      next.push({
        words: pk.tok === 0 ? parent.words.slice() : parent.words.concat(pk.tok),
        last: pk.tok === 0 ? parent.last : pk.tok,
        cum: parent.cum + st.l[pk.from][pk.tok],
        logp: parent.logp + LOGP[parent.last][pk.tok],
        fin,
      });
    }
    const rec = { step: S.steps, input: st, rank: sel.rank, picks: sel.picks, before: live.map(b => ({ ...b })), after: next.map(b => ({ ...b })) };
    S.beams = next;
    S.steps++;
    S.history.push(rec);
    // slot 0 is the cheapest beam; once it has ended nothing can overtake it
    if (S.beams[0].fin || S.steps >= S.cfg.maxSteps) S.done = true;
    return rec;
  }

  function runSearch(cfg, context, width, unit, select) {
    const S = newSearch(cfg, context, width, unit);
    while (!S.done) advance(S, select);
    return S;
  }

  const netSelect = (net, cfg) => st => networkStep(net, cfg, st);
  const twinSelect = cfg => st => {
    const t = twinStep(cfg, st);
    return { rank: t.rank, picks: t.order.slice(0, cfg.k).map(p => ({ cost: t.cost[p], from: Math.floor(p / cfg.v), tok: p % cfg.v })) };
  };

  // ------------------------------------------------------------- reference searches (plain floats)
  function greedy(context, maxSteps) {
    let last = context; const words = []; let logp = 0;
    for (let t = 0; t < maxSteps; t++) {
      let bj = 0;
      for (let j = 1; j < 16; j++) if (LOGP[last][j] > LOGP[last][bj]) bj = j;
      logp += LOGP[last][bj];
      if (bj === 0) return { words, logp, finished: true };
      words.push(bj); last = bj;
    }
    return { words, logp, finished: false };
  }
  // Beam search on exact log-probabilities, no quantization, no cap.
  function floatBeam(context, width, maxSteps) {
    let beams = [{ words: [], last: context, logp: 0, fin: false }];
    for (let t = 0; t < maxSteps; t++) {
      const cand = [];
      beams.forEach((b, i) => {
        if (b.fin) { cand.push({ ...b, order: i * 16 }); return; }
        for (let j = 0; j < 16; j++) cand.push({ words: j === 0 ? b.words : b.words.concat(j), last: j === 0 ? b.last : j, logp: b.logp + LOGP[b.last][j], fin: j === 0, order: i * 16 + j });
      });
      cand.sort((a, b) => b.logp - a.logp || a.order - b.order);
      beams = cand.slice(0, width);
      if (beams[0].fin) break;
    }
    return beams[0];
  }
  // Exact best finished sentence of at most maxSteps tokens (words plus the end token), by DP.
  function optimal(context, maxSteps) {
    // best[t][w] = max logp of t words ending in w (t >= 1), with parent pointers
    let best = null, bestEnd = { logp: -Infinity };
    const dp = [new Map([[context, { logp: 0, prev: null }]])];
    for (let t = 0; t < maxSteps; t++) {
      const cur = dp[t], nxt = new Map();
      for (const [w, st] of cur) {
        const endLp = st.logp + LOGP[w][0];
        if (endLp > bestEnd.logp) bestEnd = { logp: endLp, t, w };
        if (t + 1 >= maxSteps) continue;
        for (let j = 1; j < 16; j++) {
          const lp = st.logp + LOGP[w][j];
          if (!nxt.has(j) || nxt.get(j).logp < lp) nxt.set(j, { logp: lp, prev: w });
        }
      }
      dp.push(nxt);
    }
    // walk back
    const words = []; let w = bestEnd.w;
    for (let t = bestEnd.t; t > 0; t--) { words.push(w); w = dp[t].get(w).prev; }
    return { words: words.reverse(), logp: bestEnd.logp, finished: true };
  }

  const text = words => words.map(w => WORDS[w]).join(' ');

  const api = { WORDS, START, LOGP, COUNTS, contextName, createNet, candidateCosts, twinStep, stepTokens, networkStep, quant, newSearch, prepare, advance, runSearch, netSelect, twinSelect, greedy, floatBeam, optimal, text };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.BeamCore = api;
})(typeof self !== 'undefined' ? self : this);
