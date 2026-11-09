// node park_run.js PARK_DEMO.json BRAIN.json|- TICKS SEED learn|nolearn
// One park with the given compiled brain (- = the exact one): population, species, deaths,
// and how often the brain's top option differs from the rule's.
const fs = require('fs');
const { createPark } = require(require('path').join(__dirname, '../../demos/park/park_core.js')).ParkCore;
const [demoP, brainP, T, seed, mode] = process.argv.slice(2);
const data = JSON.parse(fs.readFileSync(demoP, 'utf8'));
let brain;
if (brainP !== '-') {
  const w = JSON.parse(fs.readFileSync(brainP, 'utf8'));
  brain = { ...data.brain, hidden: w.layers.slice(0, -1).map((l) => l.b.length), layers: w.layers.map((l) => ({ b: l.b, rows: l.W.map((r) => r.flatMap((x, i) => (x !== 0 ? [i, x] : []))) })) };
}
let n = 0, off = 0;
const argmax = (v) => v.reduce((b, x, i) => (x > v[b] ? i : b), 0);
const sim = createPark(data, { seed: +seed, learn: mode === 'learn', brain, onDecide(a, x, keys) {} });
const t0 = Date.now();
for (let t = 1; t <= +T; t++) {
  sim.tick();
  for (const a of sim.yumon) if (a.alive && a.keys && a.rule && a.exact) { n++; if (argmax(a.keys) !== argmax(a.rule)) off++; }
}
const s = sim.stats, alive = sim.yumon.filter((a) => a.alive);
console.log(JSON.stringify({ brain: brainP === '-' ? 'exact' : brainP.split(/[\/]/).pop(), mode, seed: +seed, alive: alive.length,
  species: sim.species.filter((sp) => sp.alive > 0).length, born: s.born, died: s.died, causes: s.causes, kills: s.kills,
  topOffRulePct: +(100 * off / Math.max(1, n)).toFixed(2), msPerTick: +((Date.now() - t0) / +T).toFixed(1) }));
