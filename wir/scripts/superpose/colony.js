// node colony.js DEMO.json BRAIN.json|- TICKS SEED [record.json]
const fs = require('fs');
const { createSim } = require(require('path').join(__dirname, '../../demos/ant/ant_core.js')).AntCore;
const [demoP, brainP, T, seed, recP] = process.argv.slice(2);
const data = JSON.parse(fs.readFileSync(demoP, 'utf8'));
if (brainP !== '-') {
  const w = JSON.parse(fs.readFileSync(brainP, 'utf8'));
  data.brain = { ...data.brain, inputs: w.inputs, outputs: w.outputs, layers: w.layers.map((l) => ({
    b: l.b, rows: l.W.map((r) => r.flatMap((x, i) => (x !== 0 ? [i, x] : []))) })) };
}
const rec = new Map();
const diff = { decisions: 0, any: 0, option: 0, emote: 0, actions: 0 };
let sim;
sim = createSim(data, { seed: +seed, check: false, onDecide(a, dec) {
  diff.decisions++;
  const p = sim.brainParts(dec.input);
  const o = p.option !== dec.option, e = p.emote !== dec.emote, ac = p.actions.some((b, i) => b !== dec.actions[i]);
  diff.option += o; diff.emote += e; diff.actions += ac; diff.any += o || e || ac;
  if (recP) { const k = dec.input.join(','); rec.set(k, (rec.get(k) || 0) + 1); }
} });
const ticks = +T, t0 = Date.now(), pops = [];
for (let t = 1; t <= ticks; t++) { sim.tick(); if (t % 500 === 0) pops.push(sim.colonies[0].pop); }
const s = sim.stats, c = sim.colonies[0];
console.log(JSON.stringify({ seed: +seed, pop: c.pop, popMin: Math.min(...pops), popMean: Math.round(pops.reduce((a, b) => a + b) / pops.length),
  born: s.born, died: s.died, starved: s.starved, foodIn: s.foodIn, dug: s.dug, sandOut: s.sandOut, store: c.store,
  diffPct: { any: +(100 * diff.any / diff.decisions).toFixed(3), option: +(100 * diff.option / diff.decisions).toFixed(3),
    emote: +(100 * diff.emote / diff.decisions).toFixed(3), actions: +(100 * diff.actions / diff.decisions).toFixed(3) },
  decisions: diff.decisions, msPerTick: +((Date.now() - t0) / ticks).toFixed(2) }));
if (recP) fs.writeFileSync(recP, JSON.stringify({ inputs: data.brain.inputs, states: [...rec].map(([k, n]) => [k.split(',').map(Number), n]) }));
