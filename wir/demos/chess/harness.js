// node harness.js chess_demo.json [games] [mode]
//   perft: move generation against the standard counts
//   vectors: the plain-code twin and the network against the Rust reference vectors
//   lockstep: network vs twin on every position of self-play games (random + engine moves)
//   speed: network evals per second; a short engine game
const C = require('./chess_core.js');
const fs = require('fs');
const data = JSON.parse(fs.readFileSync(process.argv[2] || 'chess_demo.json', 'utf8'));
const games = +(process.argv[3] || 20);
const mode = process.argv[4] || 'all';
const twinData = { pstMg: data.pst_mg, pstEg: data.pst_eg };
let failed = false;
const fail = msg => { console.log('FAIL', msg); failed = true; };

if (mode === 'all' || mode === 'perft') {
  // Published perft counts (chessprogramming wiki): start position, "Kiwipete", position 3, 4, 5.
  const cases = [
    [C.START, [20, 400, 8902, 197281]],
    ['r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1', [48, 2039, 97862]],
    ['8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1', [14, 191, 2812, 43238]],
    ['r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1', [6, 264, 9467]],
    ['rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8', [44, 1486, 62379]],
  ];
  const t0 = Date.now();
  for (const [fen, counts] of cases) {
    const pos = C.fromFen(fen);
    counts.forEach((want, i) => {
      const got = C.perft(pos, i + 1);
      if (got !== want) fail(`perft ${fen} depth ${i + 1}: ${got} != ${want}`);
    });
    if (C.toFen(pos) !== fen.trim()) fail(`fen roundtrip ${fen} -> ${C.toFen(pos)}`);
  }
  console.log(`perft: ${cases.length} positions, ${cases.reduce((a, c) => a + c[1].length, 0)} counts (${Date.now() - t0} ms)`);
}

const eng = C.createEngine(data, { twinData });
const outIdx = Object.fromEntries(data.outputs.map((o, i) => [o, i]));

function compareBoard(board, label, ref) {
  // network (all per-square outputs and totals) against a reference (twin or Rust)
  const out = eng.net.run(board);
  const per = [['att_w', ref.attW], ['att_b', ref.attB], ['low_w', ref.lowW], ['low_b', ref.lowB], ['hanging_w', ref.hangW], ['hanging_b', ref.hangB]];
  let worst = 0;
  for (const [k, want] of per) {
    for (let s = 0; s < 64; s++) {
      const v = out.output(outIdx[k], s);
      if (Math.round(v) !== want[s]) { fail(`${label}: ${k}[${s}] network ${v} reference ${want[s]}`); return Infinity; }
      worst = Math.max(worst, Math.abs(v - want[s]));
    }
  }
  C.TERMS.forEach((k, i) => { worst = Math.max(worst, Math.abs(out.output(outIdx[k], 0) - ref.totals[i])); });
  return worst;
}

if (mode === 'all' || mode === 'vectors') {
  let worstTwin = 0, worstNet = 0;
  for (const v of data.vectors) {
    const b = Uint8Array.from(v.board);
    const t = C.reference(b, twinData);
    const rust = { attW: v.att_w, attB: v.att_b, lowW: v.low_w, lowB: v.low_b, hangW: v.hanging_w, hangB: v.hanging_b, totals: v.totals };
    for (const k of ['attW', 'attB', 'lowW', 'lowB', 'hangW', 'hangB']) if (t[k].some((x, i) => x !== rust[k][i])) fail(`twin ${k} differs from Rust`);
    t.totals.forEach((x, i) => { worstTwin = Math.max(worstTwin, Math.abs(x - v.totals[i])); });
    worstNet = Math.max(worstNet, compareBoard(b, 'vector', rust));
  }
  // The page draws ray heads from the attention pattern: head d's best key for square q
  // must be the first piece on that ray (or BOS, index 0, when the ray is empty).
  let rayChecks = 0;
  for (const v of data.vectors.slice(0, 64)) {
    const b = Uint8Array.from(v.board);
    const att = eng.net.run(b, { record: true }).attention;
    for (let q = 0; q < 64; q++) for (let d = 0; d < 8; d++) {
      const first = C.rays[q][d].find(t => b[t]);
      const want = first === undefined ? 0 : first + 1;
      if (att[d][q + 1] !== want) fail(`ray head ${d} at ${q}: key ${att[d][q + 1]} want ${want}`);
      rayChecks++;
    }
  }
  console.log(`rays: ${rayChecks} (square, direction) pairs: each ray head attends to the first piece on its ray`);
  console.log(`vectors: ${data.vectors.length} Rust reference boards; twin worst |diff| ${worstTwin.toExponential(1)}, network worst |diff| ${worstNet.toExponential(1)}`);
}

if (mode === 'all' || mode === 'lockstep') {
  // Self-play: mostly random legal moves, every 4th move the engine at depth 1, so games
  // pass through ordinary middlegames and endings.
  let seed = 12345;
  const rnd = n => { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return (seed >>> 0) % n; };
  let positions = 0, worst = 0;
  const t0 = Date.now();
  for (let g = 0; g < games; g++) {
    const pos = C.fromFen(C.START);
    for (let ply = 0; ply < 160 && !C.status(pos); ply++) {
      for (const side of [0, 1]) {
        const b = side ? C.mirror(pos.b) : pos.b;
        worst = Math.max(worst, compareBoard(b, `game ${g} ply ${ply}`, C.reference(b, twinData)));
        positions++;
      }
      const moves = C.legal(pos);
      const m = ply % 4 === 3 ? eng.search(pos, { depth: 1, timeMs: 1000, qdepth: 2 }).move : moves[rnd(moves.length)];
      C.make(pos, m);
    }
  }
  console.log(`lockstep: ${games} games, ${positions} boards (both sides to move): network vs twin worst |diff| ${worst.toExponential(1)} (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
}

if (mode === 'all' || mode === 'speed') {
  const pos = C.fromFen('r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R2QKB1R w KQ - 0 8');
  const n = 200;
  let t0 = Date.now();
  for (let i = 0; i < n; i++) eng.net.run(pos.b);
  const ms = (Date.now() - t0) / n;
  console.log(`speed: ${ms.toFixed(2)} ms per network eval`);
  for (const depth of [0, 1, 2]) {
    eng.cache.clear();
    eng.stats.evals = 0;
    t0 = Date.now();
    const r = eng.search(C.fromFen(C.START), { depth, timeMs: 60000, qdepth: 4 });
    const p0 = C.fromFen(C.START);
    console.log(`  depth ${depth}: ${C.san(p0, r.move)} score ${r.score.toFixed(1)}, ${r.nodes} nodes, ${eng.stats.evals} network evals, ${Date.now() - t0} ms`);
  }
}

if (mode === 'twin') {
  // The engine with the network and with the plain-code twin must choose the same moves.
  let same = 0, total = 0;
  const pos = C.fromFen(C.START);
  for (let ply = 0; ply < (games || 40) && !C.status(pos); ply++) {
    eng.cache.clear();
    const a = eng.search(pos, { depth: 2, timeMs: 1e9, qdepth: 3 });
    const b = eng.search(pos, { depth: 2, timeMs: 1e9, qdepth: 3, twin: true });
    total++;
    if (a.move === b.move && Math.abs(a.score - b.score) < 1e-6) same++;
    else fail(`ply ${ply}: network ${C.san(pos, a.move)} ${a.score} vs twin ${C.san(pos, b.move)} ${b.score}`);
    C.make(pos, a.move);
  }
  console.log(`twin: ${same}/${total} moves and scores identical (depth 2)`);
}

if (mode === 'match') {
  // Short matches: [label, white player, black player]; colours alternate per game.
  let seed = 777;
  const rnd = n => { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return (seed >>> 0) % n; };
  const players = {
    random: pos => { const m = C.legal(pos); return m[rnd(m.length)]; },
    instinct: pos => eng.search(pos, { depth: 0 }).move,
    d1: pos => eng.search(pos, { depth: 1, timeMs: 2000, qdepth: 3 }).move,
    d2: pos => eng.search(pos, { depth: 2, timeMs: 3000, qdepth: 3 }).move,
  };
  for (const [a, b] of [['instinct', 'random'], ['d1', 'random'], ['d2', 'instinct']]) {
    const score = { win: 0, draw: 0, loss: 0 };
    const t0 = Date.now();
    for (let g = 0; g < games; g++) {
      const pos = C.fromFen(C.START);
      const white = g % 2 ? b : a;
      let st = null, ply = 0;
      for (; ply < 300 && !(st = C.status(pos)); ply++) C.make(pos, players[pos.side === 0 ? white : (white === a ? b : a)](pos));
      const aWhite = white === a;
      if (!st || st.startsWith('draw') || st === 'stalemate') score.draw++;
      else if ((st === 'white wins') === aWhite) score.win++;
      else score.loss++;
    }
    console.log(`match ${a} vs ${b}: ${score.win} wins, ${score.draw} draws, ${score.loss} losses for ${a} (${games} games, colours alternating, 300-ply cap; ${((Date.now() - t0) / 1000).toFixed(0)} s)`);
  }
}

console.log(failed ? 'FAILED' : 'ok');
process.exit(failed ? 1 : 0);
