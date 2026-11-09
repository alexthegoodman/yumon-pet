// WIR Chess core: rules (host), the compiled WIR network (judgment), a plain-code twin
// of the network for checks, and search. Works in the browser and in Node.
// Squares a1 = 0 .. h8 = 63; pieces 0 empty, 1-6 white P N B R Q K, 7-12 black.
(function (root) {
  'use strict';
  const WHITE = 0, BLACK = 1;
  const DIRS = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [-1, 1], [1, -1], [-1, -1]];
  const KNIGHT = [[1, 2], [2, 1], [-1, 2], [-2, 1], [1, -2], [2, -1], [-1, -2], [-2, -1]];
  const KING = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1]];
  const LETTERS = '.PNBRQKpnbrqk';
  const MATE = 100000;
  const kind = p => (p === 0 ? -1 : (p - 1) % 6); // 0 P 1 N 2 B 3 R 4 Q 5 K
  const colorOf = p => (p === 0 ? -1 : p <= 6 ? WHITE : BLACK);
  const fileOf = s => s & 7, rankOf = s => s >> 3;
  const sq = (f, r) => (f >= 0 && f < 8 && r >= 0 && r < 8 ? f + 8 * r : -1);
  const name = s => 'abcdefgh'[fileOf(s)] + (rankOf(s) + 1);

  // Precomputed targets.
  const knightTo = [], kingTo = [], rays = [];
  for (let s = 0; s < 64; s++) {
    const f = fileOf(s), r = rankOf(s);
    knightTo.push(KNIGHT.map(([a, b]) => sq(f + a, r + b)).filter(x => x >= 0));
    kingTo.push(KING.map(([a, b]) => sq(f + a, r + b)).filter(x => x >= 0));
    rays.push(DIRS.map(([a, b]) => { const out = []; let t = 1; while (sq(f + a * t, r + b * t) >= 0) { out.push(sq(f + a * t, r + b * t)); t++; } return out; }));
  }

  // Zobrist keys (deterministic).
  let zs = 0x9e3779b9;
  const rnd = () => { zs ^= zs << 13; zs ^= zs >>> 17; zs ^= zs << 5; return zs >>> 0; };
  const ZP = [], ZSIDE = [rnd(), rnd()], ZC = [], ZEP = [];
  for (let i = 0; i < 13 * 64; i++) ZP.push([rnd(), rnd()]);
  for (let i = 0; i < 16; i++) ZC.push([rnd(), rnd()]);
  for (let i = 0; i < 8; i++) ZEP.push([rnd(), rnd()]);

  // ------------------------------------------------------------- position
  function fromFen(fen) {
    const [place, side, castle, ep, half, full] = fen.trim().split(/\s+/);
    const b = new Uint8Array(64);
    const rows = place.split('/');
    for (let i = 0; i < 8; i++) {
      let f = 0;
      for (const ch of rows[i]) {
        if (/\d/.test(ch)) f += +ch;
        else { b[f + 8 * (7 - i)] = LETTERS.indexOf(ch); f++; }
      }
    }
    let c = 0;
    if (castle && castle !== '-') for (const ch of castle) c |= { K: 1, Q: 2, k: 4, q: 8 }[ch] || 0;
    const pos = { b, side: side === 'b' ? BLACK : WHITE, castle: c, ep: ep && ep !== '-' ? 'abcdefgh'.indexOf(ep[0]) + 8 * (+ep[1] - 1) : -1, half: +half || 0, full: +full || 1, hist: [], undo: [] };
    pos.hash = hashOf(pos);
    return pos;
  }
  const START = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

  function toFen(pos) {
    let out = '';
    for (let r = 7; r >= 0; r--) {
      let e = 0;
      for (let f = 0; f < 8; f++) {
        const p = pos.b[f + 8 * r];
        if (!p) e++; else { if (e) out += e; e = 0; out += LETTERS[p]; }
      }
      if (e) out += e;
      if (r) out += '/';
    }
    let c = '';
    if (pos.castle & 1) c += 'K'; if (pos.castle & 2) c += 'Q'; if (pos.castle & 4) c += 'k'; if (pos.castle & 8) c += 'q';
    return `${out} ${pos.side ? 'b' : 'w'} ${c || '-'} ${pos.ep >= 0 ? name(pos.ep) : '-'} ${pos.half} ${pos.full}`;
  }

  function hashOf(pos) {
    let h = 0, l = 0;
    for (let s = 0; s < 64; s++) if (pos.b[s]) { const z = ZP[pos.b[s] * 64 + s]; h ^= z[0]; l ^= z[1]; }
    h ^= ZSIDE[pos.side]; l ^= ZSIDE[pos.side] * 7;
    h ^= ZC[pos.castle][0]; l ^= ZC[pos.castle][1];
    if (pos.ep >= 0) { h ^= ZEP[fileOf(pos.ep)][0]; l ^= ZEP[fileOf(pos.ep)][1]; }
    return (h >>> 0) * 2097152 + ((l >>> 0) & 0x1fffff);
  }

  function attacked(b, s, by) {
    const f = fileOf(s), r = rankOf(s);
    const P = by === WHITE ? 1 : 7;
    const pr = by === WHITE ? r - 1 : r + 1;
    for (const df of [-1, 1]) { const t = sq(f + df, pr); if (t >= 0 && b[t] === P) return true; }
    for (const t of knightTo[s]) if (b[t] === P + 1) return true;
    for (const t of kingTo[s]) if (b[t] === P + 5) return true;
    for (let d = 0; d < 8; d++) {
      for (const t of rays[s][d]) {
        const p = b[t];
        if (!p) continue;
        if (colorOf(p) === by) { const k = kind(p); if (k === 4 || (d < 4 ? k === 3 : k === 2)) return true; }
        break;
      }
    }
    return false;
  }
  const kingSq = (b, side) => b.indexOf(side === WHITE ? 6 : 12);
  const inCheck = pos => { const k = kingSq(pos.b, pos.side); return k >= 0 && attacked(pos.b, k, 1 - pos.side); };

  // Move: from | to << 6 | promo kind (1..4 = N B R Q) << 12 | flag << 15 (1 double, 2 ep, 3 castle)
  const mv = (from, to, promo = 0, flag = 0) => from | (to << 6) | (promo << 12) | (flag << 15);
  const mFrom = m => m & 63, mTo = m => (m >> 6) & 63, mPromo = m => (m >> 12) & 7, mFlag = m => (m >> 15) & 3;

  function pseudo(pos, capturesOnly) {
    const b = pos.b, us = pos.side, out = [];
    const own = p => p && colorOf(p) === us, foe = p => p && colorOf(p) !== us;
    for (let s = 0; s < 64; s++) {
      const p = b[s];
      if (!own(p)) continue;
      const k = kind(p), f = fileOf(s), r = rankOf(s);
      if (k === 0) {
        const dir = us === WHITE ? 1 : -1, last = us === WHITE ? 7 : 0, start = us === WHITE ? 1 : 6;
        const push = (to, flag) => { if (rankOf(to) === last) for (const pr of [4, 1, 2, 3]) out.push(mv(s, to, pr, flag)); else out.push(mv(s, to, 0, flag)); };
        const one = sq(f, r + dir);
        if (one >= 0 && !b[one] && (!capturesOnly || rankOf(one) === last)) {
          push(one, 0);
          const two = sq(f, r + 2 * dir);
          if (!capturesOnly && r === start && !b[two]) out.push(mv(s, two, 0, 1));
        }
        for (const df of [-1, 1]) {
          const t = sq(f + df, r + dir);
          if (t < 0) continue;
          if (foe(b[t])) push(t, 0);
          else if (t === pos.ep) out.push(mv(s, t, 0, 2));
        }
      } else if (k === 1 || k === 5) {
        for (const t of (k === 1 ? knightTo : kingTo)[s]) if (!own(b[t]) && (!capturesOnly || b[t])) out.push(mv(s, t));
        if (k === 5 && !capturesOnly) {
          const home = us === WHITE ? 4 : 60, them = 1 - us;
          if (s === home && !attacked(b, s, them)) {
            const [kc, qc] = us === WHITE ? [1, 2] : [4, 8];
            if (pos.castle & kc && !b[s + 1] && !b[s + 2] && b[s + 3] === (us === WHITE ? 4 : 10) && !attacked(b, s + 1, them) && !attacked(b, s + 2, them)) out.push(mv(s, s + 2, 0, 3));
            if (pos.castle & qc && !b[s - 1] && !b[s - 2] && !b[s - 3] && b[s - 4] === (us === WHITE ? 4 : 10) && !attacked(b, s - 1, them) && !attacked(b, s - 2, them)) out.push(mv(s, s - 2, 0, 3));
          }
        }
      } else {
        for (let d = k === 2 ? 4 : 0; d < (k === 3 ? 4 : 8); d++) {
          for (const t of rays[s][d]) {
            if (own(b[t])) break;
            if (!capturesOnly || b[t]) out.push(mv(s, t));
            if (b[t]) break;
          }
        }
      }
    }
    return out;
  }

  const CASTLE_MASK = new Uint8Array(64).fill(15);
  CASTLE_MASK[4] = 15 & ~3; CASTLE_MASK[7] = 15 & ~1; CASTLE_MASK[0] = 15 & ~2;
  CASTLE_MASK[60] = 15 & ~12; CASTLE_MASK[63] = 15 & ~4; CASTLE_MASK[56] = 15 & ~8;

  function make(pos, m) {
    const b = pos.b, from = mFrom(m), to = mTo(m), flag = mFlag(m), promo = mPromo(m);
    const p = b[from];
    pos.undo.push([m, b[to], pos.castle, pos.ep, pos.half, pos.hash]);
    pos.hist.push(pos.hash);
    let cap = b[to];
    b[to] = promo ? (pos.side === WHITE ? promo + 1 : promo + 7) : p;
    b[from] = 0;
    if (flag === 2) { const cs = to + (pos.side === WHITE ? -8 : 8); cap = b[cs]; b[cs] = 0; pos.undo[pos.undo.length - 1].push(cs, cap); }
    if (flag === 3) { if (to > from) { b[from + 1] = b[from + 3]; b[from + 3] = 0; } else { b[from - 1] = b[from - 4]; b[from - 4] = 0; } }
    pos.castle &= CASTLE_MASK[from] & CASTLE_MASK[to];
    pos.ep = flag === 1 ? (from + to) >> 1 : -1;
    pos.half = kind(p) === 0 || cap ? 0 : pos.half + 1;
    if (pos.side === BLACK) pos.full++;
    pos.side ^= 1;
    pos.hash = hashOf(pos);
  }

  function unmake(pos) {
    const u = pos.undo.pop();
    pos.hist.pop();
    const [m, captured, castle, ep, half, hash] = u;
    const b = pos.b, from = mFrom(m), to = mTo(m), flag = mFlag(m), promo = mPromo(m);
    pos.side ^= 1;
    if (pos.side === BLACK) pos.full--;
    b[from] = promo ? (pos.side === WHITE ? 1 : 7) : b[to];
    b[to] = captured;
    if (flag === 2) b[u[6]] = u[7];
    if (flag === 3) { if (to > from) { b[from + 3] = b[from + 1]; b[from + 1] = 0; } else { b[from - 4] = b[from - 1]; b[from - 1] = 0; } }
    pos.castle = castle; pos.ep = ep; pos.half = half; pos.hash = hash;
  }

  function legal(pos, capturesOnly) {
    const out = [];
    for (const m of pseudo(pos, capturesOnly)) {
      make(pos, m);
      const k = kingSq(pos.b, 1 - pos.side);
      if (k < 0 || !attacked(pos.b, k, pos.side)) out.push(m);
      unmake(pos);
    }
    return out;
  }

  function perft(pos, depth) {
    if (depth === 0) return 1;
    let n = 0;
    for (const m of legal(pos)) { make(pos, m); n += perft(pos, depth - 1); unmake(pos); }
    return n;
  }

  function san(pos, m) {
    const from = mFrom(m), to = mTo(m), p = pos.b[from], k = kind(p);
    let s;
    if (mFlag(m) === 3) s = to > from ? 'O-O' : 'O-O-O';
    else {
      const cap = pos.b[to] || mFlag(m) === 2;
      if (k === 0) s = (cap ? 'abcdefgh'[fileOf(from)] + 'x' : '') + name(to) + (mPromo(m) ? '=' + 'NBRQ'[mPromo(m) - 1] : '');
      else {
        const others = legal(pos).filter(o => o !== m && mTo(o) === to && pos.b[mFrom(o)] === p);
        let dis = '';
        if (others.length) dis = others.some(o => fileOf(mFrom(o)) === fileOf(from)) ? (others.some(o => rankOf(mFrom(o)) === rankOf(from)) ? name(from) : String(rankOf(from) + 1)) : 'abcdefgh'[fileOf(from)];
        s = 'NBRQK'[k - 1] + dis + (cap ? 'x' : '') + name(to);
      }
    }
    make(pos, m);
    const check = inCheck(pos), none = legal(pos).length === 0;
    unmake(pos);
    return s + (check ? (none ? '#' : '+') : '');
  }

  // Game state from the host's point of view (rules, not judgment).
  function status(pos) {
    const moves = legal(pos);
    if (!moves.length) return inCheck(pos) ? (pos.side === WHITE ? 'black wins' : 'white wins') : 'stalemate';
    if (pos.half >= 100) return 'draw (50 moves)';
    if (pos.hist.filter(h => h === pos.hash).length >= 2) return 'draw (repetition)';
    if (insufficient(pos.b)) return 'draw (material)';
    return null;
  }
  function insufficient(b) {
    const ps = [...b].filter(p => p && kind(p) !== 5);
    return ps.length === 0 || (ps.length === 1 && (kind(ps[0]) === 1 || kind(ps[0]) === 2));
  }

  // ------------------------------------------------------------- the network
  function sparse(m) {
    const rows = Array.from({ length: m.rows }, () => []);
    const cols = Array.from({ length: m.cols }, () => []);
    for (const [r, c, v] of m.nz) { rows[r].push(c, v); cols[c].push(r, v); }
    return { rows: rows.map(r => Float64Array.from(r)), cols: cols.map(c => Float64Array.from(c)), nrows: m.rows, ncols: m.cols };
  }

  // Keys this far below a query's best score are skipped (weight < e^-36).
  const SKIP = 36;

  function createNet(w) {
    const d = w.dims.length, n = w.len + 1;
    const embed = sparse(w.embeds[0].table), posE = sparse(w.pos_embed);
    const blocks = w.blocks.map(b => ({
      heads: b.heads.map(h => {
        const dh = h.wq.rows, nv = h.wv.rows, n1 = w.len + 1;
        return { q: sparse(h.wq), k: sparse(h.wk), v: sparse(h.wv), o: sparse(h.wo), Q: new Float64Array(n1 * dh), Kd: new Float64Array(n1 * dh), V: new Float64Array(n1 * nv), K: Array.from({ length: n1 }, () => []) };
      }),
      win: sparse(b.w_in), wout: sparse(b.w_out),
    }));
    const unembed = sparse(w.unembed);
    const x = new Float64Array(n * d);
    const lo = w.embeds[0].lo;
    let lastAttn = null;
    const dotRow = (row, base) => { let s = 0; for (let i = 0; i < row.length; i += 2) s += row[i + 1] * x[base + row[i]]; return s; };
    // Each head precomputes the dimensions it reads (nonzero columns of Wq, Wk, Wv), so a
    // product touches only those; the residual stream is mostly one-hot. Head outputs go
    // straight into x: they are fresh dimensions no head of the same block reads.
    for (const blk of blocks) for (const h of blk.heads) {
      const cols = new Set();
      for (const m of [h.q, h.k, h.v]) m.cols.forEach((c, i) => { if (c.length) cols.add(i); });
      h.read = Int32Array.from([...cols].sort((a, b) => a - b));
    }
    function run(board, opts = {}) {
      x.fill(0);
      const add = (m, r, base) => { const row = m.rows[r]; for (let i = 0; i < row.length; i += 2) x[base + row[i]] += row[i + 1]; };
      add(embed, 0, 0); add(posE, 0, 0);
      for (let s = 0; s < 64; s++) { add(embed, 1 + board[s] - lo, (s + 1) * d); add(posE, 1 + s, (s + 1) * d); }
      const record = opts.record ? [] : null;
      const scores = new Float64Array(n);
      for (const blk of blocks) {
        const heads = blk.heads;
        // all heads read x before any writes it (outputs gathered per head, then added)
        const outs = [];
        for (const h of heads) {
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
        heads.forEach((h, hi) => {
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
        // all units of a position read x before any writes it (hidden values first)
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
      lastAttn = record;
      return {
        // outputs at square s (0-based), all outputs
        at(s) { const base = (s + 1) * d; return unembed.rows.map(r => dotRow(r, base)); },
        output(o, s) { return dotRow(unembed.rows[o], (s + 1) * d); },
        attention: record,
      };
    }
    return { run, dims: w.dims, outputs: w.outputs, d, get lastAttn() { return lastAttn; } };
  }

  // ------------------------------------------------------------- plain-code twin
  // Same numbers as the network (and wir::chess::reference in Rust), for checks.
  function reference(board, data) {
    const UNIT = [1, 3, 3, 5, 9, 20], HANG = [100, 320, 330, 500, 900, 0], PHASE = [0, 1, 1, 2, 4, 0];
    const PASSED = [0, 5, 10, 20, 35, 60, 100, 0];
    const attW = new Array(64).fill(0), attB = new Array(64).fill(0), lowW = new Array(64).fill(30), lowB = new Array(64).fill(30);
    for (let q = 0; q < 64; q++) {
      const hit = p => { const v = UNIT[kind(p)]; if (colorOf(p) === WHITE) { attW[q]++; lowW[q] = Math.min(lowW[q], v); } else { attB[q]++; lowB[q] = Math.min(lowB[q], v); } };
      for (let d = 0; d < 8; d++) for (const t of rays[q][d]) { const p = board[t]; if (!p) continue; const k = kind(p); if (k === 4 || (d < 4 ? k === 3 : k === 2)) hit(p); break; }
      for (const t of knightTo[q]) if (kind(board[t]) === 1) hit(board[t]);
      const f = fileOf(q), r = rankOf(q);
      for (const df of [-1, 1]) { let t = sq(f + df, r - 1); if (t >= 0 && board[t] === 1) hit(1); t = sq(f + df, r + 1); if (t >= 0 && board[t] === 7) hit(7); }
      for (const t of kingTo[q]) if (kind(board[t]) === 5) hit(board[t]);
    }
    const unit = (p, w) => (p && (colorOf(p) === WHITE) === w ? UNIT[kind(p)] : 0);
    const hangW = new Array(64).fill(0), hangB = new Array(64).fill(0);
    let mob = 0, king = 0, hus = 0, hthem = 0, passed = 0, rooks = 0, mg = 0, eg = 0, phase = 0;
    const wk = board.indexOf(6), bk = board.indexOf(12);
    const nearK = (q, kp) => { for (const t of kingTo[q]) if (board[t] === kp) return true; return false; };
    for (let q = 0; q < 64; q++) {
      const p = board[q], uw = unit(p, true), ub = unit(p, false);
      hangW[q] = attB[q] > 0 && (attW[q] === 0 || lowB[q] < uw) && uw > 0 ? 1 : 0;
      hangB[q] = attW[q] > 0 && (attB[q] === 0 || lowW[q] < ub) && ub > 0 ? 1 : 0;
      if (hangW[q]) hus += HANG[kind(p)];
      if (hangB[q]) hthem += HANG[kind(p)];
      if (colorOf(p) !== WHITE) mob += attW[q];
      if (colorOf(p) !== BLACK) mob -= attB[q];
      if (nearK(q, 12)) king += attW[q];
      if (nearK(q, 6)) king -= attB[q];
      const f = fileOf(q), r = rankOf(q);
      const zone = (s, ahead) => Math.abs(fileOf(s) - f) <= 1 && (ahead ? rankOf(s) > r : rankOf(s) < r);
      if (p === 1 && !board.some((x, s) => x === 7 && zone(s, true))) passed += PASSED[r];
      if (p === 7 && !board.some((x, s) => x === 1 && zone(s, false))) passed -= PASSED[7 - r];
      const open = !board.some((x, s) => s !== q && fileOf(s) === f && (x === 1 || x === 7));
      if (open && p === 4) rooks++;
      if (open && p === 10) rooks--;
      if (p) { mg += data.pstMg[p][q]; eg += data.pstEg[p][q]; phase += PHASE[kind(p)]; }
    }
    void wk; void bk;
    const pst = eg + phase * (mg - eg) / 24;
    return { attW, attB, lowW, lowB, hangW, hangB, totals: [mob, king, hus, hthem, passed, rooks, pst] };
  }

  // ------------------------------------------------------------- judgment + search
  const TERMS = ['mobility', 'king', 'hang_us', 'hang_them', 'passed', 'rooks', 'pst'];

  function mirror(b) {
    const m = new Uint8Array(64);
    for (let s = 0; s < 64; s++) { const p = b[s ^ 56]; m[s] = p === 0 ? 0 : p <= 6 ? p + 6 : p - 6; }
    return m;
  }

  function createEngine(data, opts = {}) {
    const net = createNet(data.weights);
    const outIdx = Object.fromEntries(data.outputs.map((o, i) => [o, i]));
    const style = Object.fromEntries(data.style.map(([k, v]) => [k, v]));
    const cache = new Map();
    const stats = { evals: 0, cached: 0, nodes: 0 };
    // Twin tables for the plain-code evaluator (same piece-square values as the network).
    const twinData = opts.twinData || null;

    // The network's terms for the side to move.
    function terms(pos) {
      const key = pos.hash;
      const hit = cache.get(key);
      if (hit) { stats.cached++; return hit; }
      stats.evals++;
      const b = pos.side === WHITE ? pos.b : mirror(pos.b);
      const out = net.run(b);
      const t = TERMS.map(k => out.output(outIdx[k], 0));
      if (cache.size > 200000) cache.clear();
      cache.set(key, t);
      return t;
    }
    // eval = the last layer with the current style weights (a linear readout of the terms).
    function judge(pos) {
      const t = terms(pos);
      let e = 0;
      for (let i = 0; i < TERMS.length; i++) e += style[TERMS[i]] * t[i];
      return quantize(e);
    }
    // Scores are kept to 1/1000 centipawn, so exact ties stay ties (the network's error,
    // about 1e-7, would otherwise break them at random).
    const quantize = e => Math.round(e * 1000) / 1000;
    function twinJudge(pos) {
      const b = pos.side === WHITE ? pos.b : mirror(pos.b);
      const t = reference(b, twinData).totals;
      let e = 0;
      for (let i = 0; i < TERMS.length; i++) e += style[TERMS[i]] * t[i];
      return quantize(e);
    }

    const VAL = [100, 320, 330, 500, 900, 2000];
    function order(pos, moves, best) {
      const score = m => {
        if (m === best) return 1e9;
        const v = pos.b[mTo(m)];
        let s = 0;
        if (v) s += 10 * VAL[kind(v)] - VAL[kind(pos.b[mFrom(m)])];
        if (mFlag(m) === 2) s += 900;
        if (mPromo(m)) s += 800 + mPromo(m);
        return s;
      };
      return moves.map(m => [score(m), m]).sort((a, b) => b[0] - a[0]).map(x => x[1]);
    }

    function search(pos, o = {}) {
      const evalFn = o.twin ? twinJudge : judge;
      const depthMax = o.depth ?? 2, timeMs = o.timeMs ?? 3000, qMax = o.qdepth ?? 4;
      const t0 = Date.now();
      const tt = new Map();
      stats.nodes = 0;
      let stop = false;
      const rootMoves = legal(pos);
      if (!rootMoves.length) return { move: 0, score: inCheck(pos) ? -MATE : 0, depth: 0 };
      const terminal = (ply) => (inCheck(pos) ? -MATE + ply : 0);
      // A position seen before (same side to move, since the last irreversible move) is a draw.
      const countRep = () => { let c = 0; for (let i = pos.hist.length - 2; i >= 0 && i >= pos.hist.length - 1 - pos.half; i -= 2) if (pos.hist[i] === pos.hash) c++; return c; };
      const repeated = () => pos.half >= 100 || countRep() >= 1;

      function quiesce(alpha, beta, ply, qd) {
        stats.nodes++;
        const stand = evalFn(pos);
        if (qd <= 0) return stand;
        if (stand >= beta) return stand;
        if (stand > alpha) alpha = stand;
        for (const m of order(pos, legal(pos, true), 0)) {
          make(pos, m);
          const s = -quiesce(-beta, -alpha, ply + 1, qd - 1);
          unmake(pos);
          if (s >= beta) return s;
          if (s > alpha) alpha = s;
        }
        return alpha;
      }

      function negamax(depth, alpha, beta, ply) {
        stats.nodes++;
        if (ply > 0 && repeated()) return 0;
        if (depth <= 0) return qMax > 0 ? quiesce(alpha, beta, ply, qMax) : evalFn(pos);
        if ((stats.nodes & 63) === 0 && Date.now() - t0 > timeMs) stop = true;
        if (stop) return 0;
        const e = tt.get(pos.hash);
        if (e && e.depth >= depth) return e.score;
        const moves = legal(pos);
        if (!moves.length) return terminal(ply);
        let best = -Infinity, bestMove = moves[0];
        for (const m of order(pos, moves, e ? e.move : 0)) {
          make(pos, m);
          const s = -negamax(depth - 1, -beta, -alpha, ply + 1);
          unmake(pos);
          if (stop) return 0;
          if (s > best) { best = s; bestMove = m; }
          if (s > alpha) alpha = s;
          if (alpha >= beta) break;
        }
        tt.set(pos.hash, { depth, score: best, move: bestMove });
        return best;
      }

      // Depth 0: the network alone picks among the legal moves (the host only reports
      // checkmate and stalemate, which are rules).
      if (depthMax === 0) {
        let best = null;
        const scored = [];
        for (const m of rootMoves) {
          make(pos, m);
          const replies = legal(pos);
          const s = replies.length ? -evalFn(pos) : terminal(1) === -MATE + 1 ? MATE - 1 : 0;
          unmake(pos);
          scored.push([m, s]);
          if (!best || s > best[1]) best = [m, s];
        }
        return { move: best[0], score: best[1], depth: 0, nodes: rootMoves.length, scored, ms: Date.now() - t0 };
      }

      let result = { move: rootMoves[0], score: 0, depth: 0 };
      for (let depth = 1; depth <= depthMax; depth++) {
        let alpha = -Infinity, bestMove = 0, best = -Infinity;
        const prev = tt.get(pos.hash);
        const scored = [];
        for (const m of order(pos, rootMoves, prev ? prev.move : result.move)) {
          make(pos, m);
          const s = -negamax(depth - 1, -Infinity, -alpha, 1);
          unmake(pos);
          if (stop) break;
          scored.push([m, s]);
          if (s > best) { best = s; bestMove = m; }
          if (s > alpha) alpha = s;
        }
        if (stop && depth > 1) break;
        if (bestMove) {
          result = { move: bestMove, score: best, depth, scored };
          tt.set(pos.hash, { depth, score: best, move: bestMove });
        }
        if (stop) break;
      }
      result.nodes = stats.nodes;
      result.ms = Date.now() - t0;
      return result;
    }

    return { net, terms, judge, twinJudge, search, style, stats, cache };
  }

  const api = { WHITE, BLACK, MATE, START, LETTERS, fromFen, toFen, legal, make, unmake, perft, san, status, inCheck, attacked, kind, colorOf, name, mFrom, mTo, mPromo, mFlag, mirror, createNet, createEngine, reference, TERMS, rays, knightTo, kingTo };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ChessCore = api;
})(typeof self !== 'undefined' ? self : this);
