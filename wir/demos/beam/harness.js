// node harness.js beam_demo.json [mode]
//   vectors:  the JS network against the Rust reference ranking and Rust network outputs
//   lockstep: whole searches (every prompt, widths 1-8, three cost scales (2, 4, 6 per nat)) with the network
//             vs the plain-code twin: same beams at every step
//   quality:  greedy, network beams per width, float beams and exact DP over all prompts
//   speed:    network steps per second
const B = require('./beam_core.js');
const fs = require('fs');
const data = JSON.parse(fs.readFileSync(process.argv[2] || 'beam_demo.json', 'utf8'));
const mode = process.argv[3] || 'all';
const cfg = { ...data.config, maxSteps: 8 };
const net = B.createNet(data.weights);
let failed = false;
const fail = msg => { console.log('FAIL', msg); failed = true; };
const sel = B.netSelect(net, cfg), twin = B.twinSelect(cfg);

if (mode === 'all' || mode === 'vectors') {
  let worst = 0, n = 0;
  for (const v of data.vectors) {
    const st = { s: v.s, l: v.l, fin: v.fin };
    const t = B.twinStep(cfg, st);
    const out = B.networkStep(net, cfg, st);
    for (let r = 0; r < cfg.k; r++) {
      const [cost, from, tok] = v.ranking[r];
      const p = out.picks[r];
      if (p.cost !== cost || p.from !== from || p.tok !== tok) fail(`vector ${n} slot ${r}: network ${JSON.stringify([p.cost, p.from, p.tok])} rust ${JSON.stringify([cost, from, tok])}`);
      const q = t.order[r];
      if (t.cost[q] !== cost || Math.floor(q / cfg.v) !== from || q % cfg.v !== tok) fail(`vector ${n} slot ${r}: twin disagrees with rust`);
    }
    for (let p = 0; p < cfg.k * cfg.v; p++) {
      if (out.rank[p] !== t.rank[p]) fail(`vector ${n} rank[${p}] network ${out.rank[p]} twin ${t.rank[p]}`);
    }
    // raw outputs against the Rust runner's outputs for the first slots
    const raw = net.run(B.stepTokens(cfg, st));
    for (let p = 0; p < cfg.k * cfg.v; p++) for (let o = 0; o < 4; o++) worst = Math.max(worst, Math.abs(raw.output(o, p) - v.network[p][o]));
    n++;
  }
  console.log(`vectors: ${n} Rust steps (full rankings), JS network = Rust ranking = twin; worst |JS network - Rust network| ${worst.toExponential(1)}`);
  if (worst > 1e-6) fail('JS network drifts from the Rust network');
}

const searchKey = S => S.beams.map(b => b.words.join(',') + (b.fin ? '!' : '') + ':' + b.cum).join('|');

if (mode === 'all' || mode === 'lockstep') {
  let searches = 0, steps = 0;
  const t0 = Date.now();
  for (const unit of [2, 4, 6]) for (let width = 1; width <= cfg.k; width++) for (let ctx = 0; ctx <= B.START; ctx++) {
    if (ctx === 0) continue;
    const a = B.newSearch(cfg, ctx, width, unit), b = B.newSearch(cfg, ctx, width, unit);
    while (!a.done) {
      B.advance(a, sel); B.advance(b, twin); steps++;
      if (searchKey(a) !== searchKey(b)) { fail(`unit ${unit} width ${width} context ${ctx} step ${a.steps}: network ${searchKey(a)} twin ${searchKey(b)}`); break; }
    }
    searches++;
  }
  console.log(`lockstep: ${searches} searches, ${steps} steps (${((Date.now() - t0) / steps).toFixed(1)} ms/step): network = twin at every step`);
}

if (mode === 'all' || mode === 'quality') {
  const unit = +(process.argv[4] || 6);
  const rows = [];
  let greedyOpt = 0, prompts = 0;
  const hit = Array(cfg.k + 1).fill(0), hitFloat = Array(cfg.k + 1).fill(0), agree = Array(cfg.k + 1).fill(0);
  const eq = (a, b) => Math.abs(a.logp - b.logp) < 1e-9;
  for (let ctx = 1; ctx <= B.START; ctx++) {
    const opt = B.optimal(ctx, cfg.maxSteps), g = B.greedy(ctx, cfg.maxSteps);
    prompts++;
    if (g.finished && eq(g, opt)) greedyOpt++;
    const line = [B.contextName(ctx).padEnd(8), `opt ${opt.logp.toFixed(2)} "${B.text(opt.words)}"`, `greedy ${g.logp.toFixed(2)}`];
    for (let w = 1; w <= cfg.k; w++) {
      const S = B.runSearch(cfg, ctx, w, unit, sel);
      const best = S.beams[0], f = B.floatBeam(ctx, w, cfg.maxSteps);
      if (best.fin && eq(best, opt)) hit[w]++;
      if (f.fin && eq(f, opt)) hitFloat[w]++;
      if (best.words.join() === f.words.join()) agree[w]++;
      line.push(`w${w} ${best.logp.toFixed(2)}`);
    }
    rows.push(line.join('  '));
  }
  console.log(rows.join('\n'));
  console.log(`over ${prompts} prompts (unit ${unit}): greedy optimal on ${greedyOpt}`);
  for (let w = 1; w <= cfg.k; w++) console.log(`  width ${w}: network beam optimal ${hit[w]}, float beam optimal ${hitFloat[w]}, network = float beam ${agree[w]}`);
}

if (mode === 'all' || mode === 'speed') {
  const st = { s: [0, 3, 5, 7, 24, 24, 24, 24], l: Array.from({ length: 8 }, () => Array.from({ length: 16 }, (_, j) => (j * 7) % 16)), fin: Array(8).fill(0) };
  const t0 = Date.now(); let n = 0;
  while (Date.now() - t0 < 1500) { B.networkStep(net, cfg, st); n++; }
  console.log(`speed: ${(1000 * n / (Date.now() - t0)).toFixed(0)} steps/s (${((Date.now() - t0) / n).toFixed(2)} ms/step, Node ${process.version})`);
}
process.exit(failed ? 1 : 0);
