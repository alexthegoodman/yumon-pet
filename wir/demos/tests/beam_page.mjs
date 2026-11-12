// node beam_page.mjs wir-beam.html SHOT_DIR
// Drives beam_page.feature in the installed Chrome (Playwright from PLAYWRIGHT_FROM,
// default ../../../../allegiance).
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import path from "node:path";
import fs from "node:fs";

const [html, shots] = process.argv.slice(2);
const from = process.env.PLAYWRIGHT_FROM || path.join(import.meta.dirname, "../../../../allegiance");
const { chromium } = createRequire(path.join(from, "package.json"))("playwright");
fs.mkdirSync(shots, { recursive: true });
let failed = 0;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) failed++; };
const browser = await chromium.launch({ channel: "chrome", headless: true });
const url = pathToFileURL(path.resolve(html)).href;

async function open(opts = {}) {
  const ctx = await browser.newContext({ viewport: { width: 1360, height: 1200 }, ...opts });
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => { if (m.type() === "error" && !/fonts\.g/.test(m.text())) errors.push(m.text()); });
  await page.goto(url);
  await page.waitForSelector("#grid td.cell");
  return { page, errors };
}
const text = (page, sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, " ").trim());
const beams = (page) => page.$$eval("#beams li", (l) => l.map((x) => x.textContent.replace(/\s+/g, " ").trim()));
const waitDone = (page) => page.waitForFunction(() => document.querySelector("#status").textContent.startsWith("Done."), null, { timeout: 30000 });
const width = async (page, w) => { await page.click(`#width button[data-w="${w}"]`); };

const { page, errors } = await open();
// Scenario: The page opens on the first step
{
  const rows = await page.$$eval("#grid tbody tr", (r) => r.length);
  const cols = await page.$$eval("#grid thead th", (r) => r.length - 1);
  check(rows === 8 && cols === 16, `grid: ${rows} beam rows, ${cols} word columns`);
  const w = await page.$eval("#width button[aria-pressed=true]", (b) => b.dataset.w);
  const picks = await page.$$eval("#grid td.pick", (c) => c.length);
  check(w === "4" && picks === 4, `width ${w}, ${picks} survivors highlighted`);
  const chk = await text(page, "#netcheck");
  check(/equals a plain sort/.test(chk), `network check: ${chk.slice(0, 90)}...`);
  await page.screenshot({ path: path.join(shots, "beam_start.png"), fullPage: true });
}
// Scenario: Stepping takes the network's survivors
{
  await page.click("#step");
  const b = await beams(page);
  check(b.length === 4 && b[0].startsWith("1the"), `beams after step 1: ${b.join(" | ")}`);
  const live = await page.$$eval("#grid tbody th.beam:not(.idle)", (r) => r.length);
  check(live === 4, `grid has ${live} live beam rows for the next step`);
  await page.screenshot({ path: path.join(shots, "beam_step2.png"), fullPage: true });
}
// Scenario: Running to the end finds the exact best sentence
{
  await page.click("#reset");
  await page.click("#run");
  await waitDone(page);
  const b = await beams(page);
  const status = await text(page, "#status");
  check(/yumon sleeps/.test(b[0]) && /-2\.74/.test(b[0]), `best beam: ${b[0]}`);
  check(/cheapest beam ended/.test(status), `stopped: ${status.slice(0, 110)}...`);
  await page.screenshot({ path: path.join(shots, "beam_done.png"), fullPage: true });
}
// Scenario: Greedy gets it wrong and a wider beam fixes it
{
  await width(page, 1);
  await page.click("#run");
  await waitDone(page);
  let b = await beams(page);
  check(b.length === 1 && !/yumon sleeps/.test(b[0]), `width 1: ${b[0]}`);
  const miss = await page.$$eval("#refs tr.cur td.mark", (c) => c.map((x) => x.textContent));
  check(miss.length === 1 && miss[0].startsWith("-"), `reference table marks width 1 as ${miss[0]} nats off`);
  await width(page, 3);
  await page.click("#run");
  await waitDone(page);
  b = await beams(page);
  check(/yumon sleeps/.test(b[0]), `width 3: ${b[0]}`);
}
// Scenario: A coarse cost grain changes the picks
{
  await page.selectOption("#unit", "2");
  const rows = await page.$$eval("#refs tr", (r) => r.length);
  check(rows >= 2 + 4 + 1, `reference table has ${rows} rows at 2 per nat`);
  await page.selectOption("#unit", "6");
}
// Scenario: Scoring every prompt
{
  await page.click("#score");
  await page.waitForFunction(() => document.querySelectorAll("#scoreboard tr").length > 1, null, { timeout: 60000 });
  const rows = await page.$$eval("#scoreboard tr td", (c) => c.map((x) => x.textContent.trim()));
  const map = {};
  for (let i = 0; i < rows.length; i += 2) map[rows[i]] = rows[i + 1];
  check(map["greedy"] === "8 / 16", `greedy ${map["greedy"]}`);
  check(map["network beam, width 4"] === "13 / 16", `width 4: ${map["network beam, width 4"]}`);
  await page.screenshot({ path: path.join(shots, "beam_scoreboard.png"), fullPage: true });
}
check(errors.length === 0, `script errors: ${errors.length ? errors.join("; ") : "none"}`);
// Scenario: Phone width and dark theme
{
  const { page: p2, errors: e2 } = await open({ viewport: { width: 400, height: 900 }, colorScheme: "dark" });
  const over = await p2.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  const framed = await p2.$eval(".gridwrap", (g) => g.scrollWidth > g.clientWidth);
  check(over <= 0 && framed, `phone: page overflow ${over}px, grid scrolls inside its frame: ${framed}`);
  await p2.screenshot({ path: path.join(shots, "beam_phone_dark.png") });
  check(e2.length === 0, `phone script errors: ${e2.length ? e2.join("; ") : "none"}`);
}
await browser.close();
console.log(failed ? `${failed} failed` : "all passed");
process.exit(failed ? 1 : 0);
