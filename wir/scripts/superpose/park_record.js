// Record Yumon Park brain inputs as a probe for `wir superpose-choice-layer --states`.
// node park_record.js PARK_DEMO.json TICKS SEED OUT.json   (learning off: the exact brain)
const fs = require('fs');
const { createPark } = require(require('path').join(__dirname, '../../demos/park/park_core.js')).ParkCore;
const [demoP, T, seed, outP] = process.argv.slice(2);
const data = JSON.parse(fs.readFileSync(demoP, 'utf8'));
const seen = new Map();
const sim = createPark(data, { seed: +seed, learn: false, onDecide(a, x) {
  const k = Array.from(x).join(','); seen.set(k, (seen.get(k) || 0) + 1);
} });
for (let t = 0; t < +T; t++) sim.tick();
const rows = [...seen];
fs.writeFileSync(outP, JSON.stringify({ inputs: rows.map(([k]) => k.split(',').map(Number)), weights: rows.map(([, n]) => n) }));
console.log(`${rows.reduce((s, [, n]) => s + n, 0)} decisions, ${rows.length} distinct inputs, ${sim.yumon.filter((a) => a.alive).length} alive`);
