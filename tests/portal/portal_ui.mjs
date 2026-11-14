// node portal_ui.mjs SHOT_DIR
// Drives portal_ui.feature in the installed Chrome (Playwright from PLAYWRIGHT_FROM,
// default ../../../allegiance). Builds the page the way yumon_portal.rs does: the two
// simulation pages become script-safe JSON literals in window.__simPages.
import { createRequire } from "node:module";
import path from "node:path";
import fs from "node:fs";

const [shots] = process.argv.slice(2);
const root = path.join(import.meta.dirname, "../../src/bin");
const from = process.env.PLAYWRIGHT_FROM || path.join(import.meta.dirname, "../../../allegiance");
const { chromium } = createRequire(path.join(from, "package.json"))("playwright");
fs.mkdirSync(shots, { recursive: true });
const literal = (f) => JSON.stringify(fs.readFileSync(path.join(root, "sims", f), "utf8")).replaceAll("<", String.fromCharCode(92) + "u003c");
const html = fs.readFileSync(path.join(root, "portal_ui.html"), "utf8")
  .replace("/* __SIM_PAGES__ */", () => `window.__simPages = { ant: ${literal("yumon-ant.html")}, chess: ${literal("wir-chess.html")} };`);

let failed = 0;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) failed++; };
const browser = await chromium.launch({ channel: "chrome", headless: true });
const ctx = await browser.newContext({ viewport: { width: 1440, height: 760 } });
const page = await ctx.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
// Fonts are blocked by the portal CSP on purpose (no network); ignore those.
page.on("console", (m) => { if (m.type() === "error" && !/fonts\.(googleapis|gstatic)|Content Security Policy/i.test(m.text())) errors.push(m.text()); });
await page.addInitScript(() => { window.__sent = []; window.ipc = { postMessage: (s) => window.__sent.push(JSON.parse(s)) }; });
await page.goto("about:blank");
await page.setContent(html);
await page.waitForFunction(() => window.__sent.some((m) => m.type === "ready"));

// Scenario: The portal opens with a simulation beside the chat
{
  const geo = await page.evaluate(() => {
    const r = (id) => document.getElementById(id).getBoundingClientRect();
    return { sim: r("sim-pane"), chat: r("chat-panel"), tabs: document.querySelectorAll("[role=tab]").length };
  });
  check(geo.sim.right <= geo.chat.left + 1 && geo.sim.width > geo.chat.width, `simulation (${Math.round(geo.sim.width)}px) is left of chat (${Math.round(geo.chat.width)}px)`);
  check(geo.tabs === 0, "no tabs");
  const body = await page.evaluate(() => { const c = document.body.cloneNode(true); c.querySelectorAll("script, iframe").forEach((x) => x.remove()); return c.innerHTML.toLowerCase(); });
  check(!/product hunt|paint|universe|aquarium|suburb/.test(body), "no Products, Paint or 3D universe markup");
  await page.screenshot({ path: path.join(shots, "portal-chat.png") });
}

// Scenario: Chat round trip goes through the host
{
  await page.evaluate(() => window.__yumon_portal({ chat: [], chat_busy: false, chat_status: "Say hello to Yumon.", can_chat: true }));
  await page.fill("#chat-input", "Hello Yumon");
  await page.press("#chat-input", "Enter");
  const sent = await page.evaluate(() => window.__sent.filter((m) => m.type === "chat"));
  check(sent.length === 1 && sent[0].message === "Hello Yumon", "host receives the chat message");
  await page.evaluate(() => window.__yumon_portal({ chat: [{ role: "user", text: "Hello Yumon" }, { role: "yumon", text: "Hello friend!" }], chat_busy: false, chat_status: "Say hello to Yumon.", can_chat: true }));
  const senders = await page.$$eval("#chat-messages .message small", (els) => els.map((e) => e.textContent));
  check(JSON.stringify(senders) === JSON.stringify(["You", "Yumon"]), `bubbles: ${senders.join(", ")}`);
  check((await page.inputValue("#chat-input")) === "", "composer clears once the host echoes the message");
}

// Scenario: The simulation can be switched between Yumon Ant and WIR Chess
{
  const ant = page.frameLocator("#sim-frame");
  await ant.locator("canvas").first().waitFor({ timeout: 15000 });
  check((await ant.locator("canvas").count()) >= 1, "Yumon Ant canvas is running in the frame");
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(shots, "portal-ant.png") });

  await page.click("#sim-chess");
  const chess = page.frameLocator("#sim-frame");
  await chess.locator("#board .sq").first().waitFor({ timeout: 15000 });
  check((await chess.locator("#board .sq").count()) === 64, "WIR Chess board has 64 squares");
  await chess.locator("#cands span").first().waitFor({ timeout: 15000 });
  check((await chess.locator("#cands span").count()) > 0, "the network ranks candidate moves (worker runs in the sandboxed frame)");
  check(await page.isVisible("#chat-input"), "chat is still on the right");
  await page.screenshot({ path: path.join(shots, "portal-chess.png") });
}

check(errors.length === 0, `no page errors${errors.length ? ": " + errors.join(" | ") : ""}`);
await browser.close();
process.exit(failed ? 1 : 0);
