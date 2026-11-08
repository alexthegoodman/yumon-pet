const { createSim } = require('./ant_core.js').AntCore;
// node harness.js TICKS [THINK_EVERY] [nocheck] [ant_demo.json]
const data = JSON.parse(require('fs').readFileSync(process.argv[5] || 'ant_demo.json', 'utf8'));
const ticks = +process.argv[2] || 2000, every = +process.argv[3] || 1, chk = process.argv[4] !== 'nocheck';
const sim = createSim(data, { seed: 7, check: chk, thinkEvery: every });
console.log('port', JSON.stringify(sim.verifyPort()));
const t0 = Date.now();
const log = [];
for (let t = 1; t <= ticks; t++) {
  sim.tick();
  if (t % 250 === 0) {
    const c = sim.colonies[0];
    const alive = sim.ants.filter(a => a.alive);
    const em = [0,0,0,0,0,0,0]; alive.forEach(a => em[a.emote]++);
    const carry = [0,0,0]; alive.forEach(a => carry[a.carry]++);
    console.log(`t ${t} pop ${c.pop} eggs ${c.eggs.length} store ${c.store} open ${c.open} pending ${c.pending} cap ${c.capacity} branches ${c.branches} hill ${sim.hillHeight(c)} grip ${alive.filter(a=>a.grip).length} carry ${carry} emotes ${em} | ${JSON.stringify(sim.stats)} ms/tick ${((Date.now()-t0)/t).toFixed(1)}`);
  }
}
console.log('check', JSON.stringify(sim.check));
