// node chess_page.mjs wir-chess.html SHOT_DIR
// Drives chess_page.feature in the installed Chrome (Playwright from PLAYWRIGHT_FROM,
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
  const ctx = await browser.newContext({ viewport: { width: 1360, height: 1100 }, ...opts });
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  await page.goto(url);
  await page.evaluate(() => { try { localStorage.clear(); } catch (e) {} });
  await page.reload();
  await page.waitForSelector("#cands span");
  return { page, errors };
}
// moves in the list (ignoring move numbers and blanks)
const movesIn = () => [...document.querySelectorAll("#moves span")].map((x) => x.textContent).filter((x) => x && !x.endsWith(".") && !x.startsWith("No moves"));
// square name -> board button (white at the bottom)
const sqSel = (name) => {
  const f = "abcdefgh".indexOf(name[0]), r = +name[1] - 1;
  return `#board .sq:nth-child(${(7 - r) * 8 + f + 1})`;
};

const { page, errors } = await open();
// Scenario: The page opens on the starting position
{
  const n = await page.$$eval("#board .sq", (s) => s.length);
  const pieces = await page.$$eval("#board .pc", (s) => s.length);
  check(n === 64 && pieces === 32, `board: ${n} squares, ${pieces} pieces`);
  check((await page.textContent("#status")).includes("White to move"), "status: white to move");
  const cands = await page.$$eval("#cands span", (s) => s.length);
  check(cands === 15, `instinct ranks ${cands / 3} moves`);
  await page.screenshot({ path: path.join(shots, "chess_start.png") });
}
// Scenario: The network answers a move
{
  await page.click(sqSel("e2"));
  await page.click(sqSel("e4"));
  const t0 = Date.now();
  await page.waitForFunction(`(${movesIn})().length >= 2 && document.querySelector("#status").textContent.includes("your move")`, null, { timeout: 30000 });
  const moves = await page.evaluate(`(${movesIn})()`);
  check(moves.length === 2 && moves[0] === "e4", `moves: ${moves.join(" ")} (reply after ${Date.now() - t0} ms)`);
  check(!(await page.textContent("#status")).includes("page thread"), "search ran in a worker");
  check((await page.textContent("#lastsearch")).includes("network runs"), `search: ${(await page.textContent("#lastsearch")).replace(/\s+/g, " ")}`);
}
// Scenario: The lens shows what the network attends to
{
  await page.check("#lens-attack");
  await page.check("#lens-rays");
  await page.hover(sqSel("d1"));
  const lines = await page.$$eval("svg.rays line", (l) => l.length);
  const nums = await page.$$eval("#board .num", (s) => s.length);
  check(lines >= 2 && nums > 10, `lens: ${lines} ray lines from d1, ${nums} attacker counts`);
  await page.screenshot({ path: path.join(shots, "chess_lens.png") });
}
// Scenario: Style weights are the last layer
{
  const before = await page.textContent("#terms tr.total");
  await page.$eval("#style-mobility", (el) => { el.value = "0"; el.dispatchEvent(new Event("input", { bubbles: true })); });
  const after = await page.textContent("#terms tr.total");
  check(before !== after, `judgment ${before.replace(/\s+/g, " ").trim()} -> ${after.replace(/\s+/g, " ").trim()}`);
}
// Scenario: Playing black at Instinct level
{
  await page.click("#levels button:first-child");
  await page.click("#as-black");
  await page.waitForFunction(`(${movesIn})().length >= 1 && document.querySelector("#status").textContent.includes("your move")`, null, { timeout: 20000 });
  const first = await page.$eval("#moves span:nth-child(2)", (s) => s.textContent);
  check(/^[a-hNBRQKO]/.test(first), `network opened with ${first}`);
  await page.screenshot({ path: path.join(shots, "chess_black.png") });
}
check(errors.length === 0, `script errors: ${errors.length ? errors.join("; ") : "none"}`);
// Scenario: Phone width and dark theme
{
  const { page: p2, errors: e2 } = await open({ viewport: { width: 400, height: 900 }, colorScheme: "dark" });
  const over = await p2.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  check(over <= 0, `phone: horizontal overflow ${over}px`);
  await p2.screenshot({ path: path.join(shots, "chess_phone_dark.png"), fullPage: false });
  check(e2.length === 0, `phone script errors: ${e2.length ? e2.join("; ") : "none"}`);
}
await browser.close();
console.log(failed ? `${failed} failed` : "all passed");
process.exit(failed ? 1 : 0);
