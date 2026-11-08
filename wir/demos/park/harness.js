const { createPark } = require('./park_core.js').ParkCore;
// node harness.js TICKS [learn|nolearn|mixed] [seed] [park_demo.json]
const data = JSON.parse(require('fs').readFileSync(process.argv[5] || 'park_demo.json', 'utf8'));
const ticks = +process.argv[2] || 3000, learn = process.argv[3] !== 'nolearn', seed = +process.argv[4] || 1;
const mixed = process.argv[3] === 'mixed';
const tune = process.env.PARK_TUNE ? JSON.parse(process.env.PARK_TUNE) : {};
const setup = process.env.PARK_SETUP ? JSON.parse(process.env.PARK_SETUP) : {};
const sim = createPark(data, { seed, learn: learn, mixed, tune, ...setup });
console.log('port', JSON.stringify(sim.verifyPort()), 'species', sim.species.length, 'yumon', sim.yumon.length, 'params', sim.brain.size);
const t0 = Date.now();
for (let t = 1; t <= ticks; t++) {
  sim.tick();
  if (t % 1000 === 0 || t === ticks) {
    const s = sim.stats, alive = sim.yumon.filter((a) => a.alive);
    const ext = sim.species.filter((sp) => sp.alive === 0).length;
    const opts = new Array(8).fill(0); alive.forEach((a) => opts[a.option]++);
    const L = alive.filter((a) => a.learns).length;
    const spMix = sim.species.filter((sp) => sp.alive > 0).map((sp) => { const l = alive.filter((a) => a.sp === sp.id); return l.filter((a) => a.learns).length / l.length; });
    if (mixed) console.log(`  learners ${L}/${alive.length} (${(100 * L / Math.max(1, alive.length)).toFixed(1)}%), species mostly learners ${spMix.filter((v) => v > 0.5).length}/${spMix.length}, all-learner ${spMix.filter((v) => v === 1).length}, no-learner ${spMix.filter((v) => v === 0).length}`);
    console.log(`t ${t} pop ${alive.length} models ${sim.models.size} species ${sim.species.length - ext}/${sim.species.length} born ${s.born} died ${s.died} ${JSON.stringify(s.causes)} sleeps ${s.sleeps} written ${(100 * s.written / s.decisions).toFixed(1)}% drift ${(100 * s.drift / Math.max(1, s.driftN)).toFixed(1)}% exactMiss ${s.exactMiss}/${s.untrained} maxGen ${Math.max(...sim.species.map((x) => x.maxGen))} opts ${opts} trainMs ${s.trainMs.toFixed(0)} ms/tick ${((Date.now() - t0) / t).toFixed(2)}`);
  }
}
const top = [...sim.changes.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10);
console.log('changes', JSON.stringify(top));
console.log('species', sim.species.map((sp) => `${sp.name}:${sp.alive}${sp.extinctAt ? '(x' + sp.extinctAt + ')' : ''}`).join(' '));
