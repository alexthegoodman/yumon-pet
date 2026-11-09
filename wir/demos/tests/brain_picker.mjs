// node brain_picker.mjs ANT.html PARK.html SHOT_DIR
// Drives brain_picker.feature in the installed Chrome. Playwright comes from a sibling install
// (PLAYWRIGHT_FROM, default ../../../../allegiance) so this folder needs no node_modules.
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import path from "node:path";
import fs from "node:fs";

const [antHtml, parkHtml, shots] = process.argv.slice(2);
const from = process.env.PLAYWRIGHT_FROM || path.join(import.meta.dirname, "../../../../allegiance");
const { chromium } = createRequire(path.join(from, "package.json"))("playwright");
fs.mkdirSync(shots, { recursive: true });

let failed = 0;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) failed++; };
const browser = await chromium.launch({ channel: "chrome", headless: true });

async function open(file) {
  const page = await (await browser.newContext({ viewport: { width: 1440, height: 1000 } })).newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(pathToFileURL(path.resolve(file)).href);
  return { page, errors };
}

// Scenario: Ant offers its brains and runs a superposed one
{
  const { page, errors } = await open(antHtml);
  await page.waitForSelector("#brain-pick option", { state: "attached" });
  const labels = await page.$$eval("#brain-pick option", (o) => o.map((x) => x.textContent));
  check(labels.length === 3 && labels[0].startsWith("Exact") && labels[1].includes("6 shared") && labels[2].includes("4 shared"), `ant menu: ${labels.join(" | ")}`);
  check((await page.$eval("#brain-pick", (s) => s.value)) === "0", "ant starts on the exact brain");
  await Promise.all([page.waitForEvent("load"), page.selectOption("#brain-pick", "1")]);
  await page.waitForSelector("#nets tr", { state: "attached" });
  check(page.url().endsWith("#brain=1"), `reloaded with ${new URL(page.url()).hash}`);
  const nets = await page.$eval("#nets", (t) => t.textContent);
  check(nets.includes("66-71-71-67-58-130"), "networks table shows the superposed brain's layers");
  await page.waitForTimeout(4000);
  const note = await page.$eval("#check-out", (p) => p.textContent);
  check(note.includes("shares neurons"), `check note: ${note.slice(0, 120)}...`);
  check((await page.$eval("#src", (p) => p.textContent)).includes("let sp_z0 = unit("), "source tab shows the bottleneck");
  await page.screenshot({ path: path.join(shots, "ant_superposed6.png") });
  check(errors.length === 0, `ant script errors: ${errors.length ? errors.join("; ") : "none"}`);
}

// Scenario: Park offers its brains in the ecosystem setup
{
  const { page, errors } = await open(parkHtml);
  await page.evaluate(() => { try { localStorage.removeItem("yumon-park-setup"); } catch (e) {} });
  await page.reload();
  await page.click("[role=tab][data-view=setup]");
  const labels = await page.$$eval("#set-brain option", (o) => o.map((x) => x.textContent));
  check(labels.length === 2 && labels[1].includes("6 shared"), `park menu: ${labels.join(" | ")}`);
  await page.selectOption("#set-brain", "1");
  await page.click("#setup-form button[type=submit]");
  await page.waitForTimeout(3000);
  const net = await page.$eval("#net", (p) => p.textContent);
  check(net.startsWith("Superposed, 6 shared neurons"), `park net: ${net}`);
  const port = await page.$eval("#port", (p) => p.textContent);
  check(port.includes("shares neurons"), `park load check: ${port.slice(0, 110)}...`);
  await page.screenshot({ path: path.join(shots, "park_superposed6.png") });
  check(errors.length === 0, `park script errors: ${errors.length ? errors.join("; ") : "none"}`);
}

await browser.close();
console.log(failed ? `${failed} check(s) failed` : "all checks passed");
process.exit(failed ? 1 : 0);
