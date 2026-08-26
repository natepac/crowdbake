#!/usr/bin/env node
// Headless GPU smoke test.
//
// The one class of bug this project cannot catch on the CPU is a shader that
// fails to compile or draws nothing. This boots the built app in headless
// Chrome/Edge over SwiftShader, fails on any console error or GL warning,
// exercises every runtime toggle (bone mode, cross-fade, chunk size, LOD debug,
// stride matching), and writes screenshots to tools/smoke-out/.
//
//   npm run build && node tools/smoke.mjs
//   node tools/smoke.mjs --headful      (watch it run)

import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dist = path.join(root, 'dist');
const outDir = path.join(root, 'tools', 'smoke-out');
const headful = process.argv.includes('--headful');
// SwiftShader is a software rasteriser -- a few hundred instances at a small
// viewport is plenty to prove the shaders and culling work.
const count = Number((process.argv.find((a) => a.startsWith('--count=')) || '').split('=')[1]) || 700;
const SETTLE_TIMEOUT = 180000;

const BROWSERS = [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
];

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json',
  '.bin': 'application/octet-stream', '.css': 'text/css', '.png': 'image/png',
};

function serve(dir) {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const url = decodeURIComponent(req.url.split('?')[0]);
      let file = path.join(dir, url === '/' ? 'index.html' : url);
      if (!file.startsWith(dir)) { res.writeHead(403).end(); return; }
      if (!fs.existsSync(file) || fs.statSync(file).isDirectory()) file = path.join(dir, 'index.html');
      const body = fs.readFileSync(file);
      res.writeHead(200, {
        'Content-Type': MIME[path.extname(file)] || 'application/octet-stream',
        'Content-Length': body.length,
      });
      res.end(body);
    });
    server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port }));
  });
}

function findBrowser() {
  for (const b of BROWSERS) if (fs.existsSync(b)) return b;
  throw new Error('no Chrome/Edge binary found; pass one via PUPPETEER_EXECUTABLE_PATH');
}

const problems = [];
const notes = [];
let currentStage = 'startup';

async function main() {
  if (!fs.existsSync(path.join(dist, 'index.html'))) {
    throw new Error('dist/ not built. Run: npm run build');
  }
  fs.mkdirSync(outDir, { recursive: true });

  const { server, port } = await serve(dist);
  const browser = await puppeteer.launch({
    executablePath: process.env.PUPPETEER_EXECUTABLE_PATH || findBrowser(),
    headless: !headful,
    args: [
      '--use-angle=swiftshader',
      '--use-gl=angle',
      '--enable-unsafe-swiftshader',
      '--enable-webgl',
      '--ignore-gpu-blocklist',
      '--no-sandbox',
      '--window-size=720,450',
    ],
    defaultViewport: { width: 720, height: 450 },
  });

  let page;
  try {
    page = await browser.newPage();
    page.on('console', (msg) => {
      const t = msg.type();
      const text = msg.text();
      if (text.startsWith('WebGL adapter:')) return;   // informational GPU-name log
      if (t === 'error' || /THREE.WebGLProgram|shader|GL_INVALID|WebGL:/i.test(text)) {
        problems.push(`console.${t} [stage: ${currentStage}]: ${text}`);
      }
    });
    page.on('pageerror', (e) => problems.push('pageerror: ' + e.message));
    page.on('requestfailed', (r) => problems.push('requestfailed: ' + r.url()));

    const url = `http://127.0.0.1:${port}/?count=${count}`;
    console.log('smoke  ' + url);
    await page.goto(url, { waitUntil: 'networkidle0', timeout: 90000 });
    await page.waitForFunction('window.__vat && window.__vat.ready', { timeout: 180000 });
    // keep the software rasteriser inside a sane budget
    await page.evaluate(() => {
      window.renderer.setPixelRatio(0.5);
      window.env.sun.shadow.mapSize.set(1024, 1024);
      if (window.env.sun.shadow.map) { window.env.sun.shadow.map.dispose(); window.env.sun.shadow.map = null; }
    });

    const shot = async (name) => {
      await page.screenshot({ path: path.join(outDir, name + '.png') });
    };
    const settle = async (frames = 8) => {
      const start = await page.evaluate(() => window.__vat.frames);
      await page.waitForFunction((s, f) => window.__vat.frames > s + f, { timeout: SETTLE_TIMEOUT }, start, frames);
    };
    const stats = () => page.evaluate(() => window.__vat.stats);

    const check = (label, s, expect) => {
    currentStage = label;
      notes.push(`${label.padEnd(26)} draws ${String(s.drawCalls).padStart(4)}  tris ${String(s.triangles).padStart(9)}  `
        + `rast ${String(s.rendered).padStart(6)}/${s.instances}  chunks ${s.renderedChunks}/${s.chunks}`);
      if (s.triangles === 0) problems.push(`${label}: nothing was drawn (0 triangles)`);
      if (s.drawCalls === 0) problems.push(`${label}: no draw calls`);
      if (expect) expect(s);
    };

    // --- baseline (VAT mode, chunked, shadows on) -------------------------
    await settle(30);
    const base = await stats();
    check('vat / chunked', base, (s) => {
      if (s.drawn > s.instances) problems.push('submitted instances exceeds total');
      if (s.rendered > s.drawn) problems.push('rasterised instances exceeds submitted');
      if (s.rendered >= s.instances) {
        notes.push('  note: no chunk was frustum-culled from this viewpoint');
      }
    });
    await shot('01-vat-chunked');

    // --- single chunk (the "one draw call" configuration) -----------------
    await page.evaluate(() => window.crowd.reconfigure({ chunkSize: 1e9 }));
    await settle();
    const single = await stats();
    check('vat / single chunk', single, (s) => {
      if (s.chunks !== 1) problems.push(`single-chunk mode produced ${s.chunks} chunks`);
      if (s.drawn !== s.instances) problems.push('single chunk should submit every instance');
    });
    await shot('02-single-chunk');
    await page.evaluate(() => window.crowd.reconfigure({ chunkSize: 24 }));
    await settle();

    // --- bone-matrix playback --------------------------------------------
    await page.evaluate(() => window.crowd.reconfigure({ mode: 'bone' }));
    await settle(30);
    const bone = await stats();
    check('bone-matrix mode', bone);
    await shot('03-bone-mode');
    await page.evaluate(() => window.crowd.reconfigure({ mode: 'vat' }));
    await settle();

    // --- cross-fade off ---------------------------------------------------
    await page.evaluate(() => window.crowd.reconfigure({ crossFade: false }));
    await settle(20);
    check('cross-fade off', await stats());
    await page.evaluate(() => window.crowd.reconfigure({ crossFade: true }));
    await settle();

    // --- LOD debug + forced far LOD --------------------------------------
    await page.evaluate(() => {
      window.crowd.setDebugLod(true);
      window.crowd.options.lodBias = 4;
    });
    await settle(20);
    const lod = await stats();
    check('lod debug (bias 4)', lod, (s) => {
      if (s.lodCounts.slice(1).reduce((a, b) => a + b, 0) === 0) {
        problems.push('lod bias 4 did not push any instance past LOD0');
      }
    });
    await shot('04-lod-debug');
    await page.evaluate(() => { window.crowd.setDebugLod(false); window.crowd.options.lodBias = 1; });

    // --- stride matching off (foot-slide comparison) ----------------------
    await page.evaluate(() => { window.sim.options.matchStride = false; window.sim.refreshRates(); });
    await settle(20);
    check('stride matching off', await stats());
    await page.evaluate(() => { window.sim.options.matchStride = true; window.sim.refreshRates(); });

    // --- scenarios --------------------------------------------------------
    for (const sc of ['stampede', 'parade', 'mill', 'gather', 'wander']) {
      await page.evaluate((s) => window.sim.setScenario(s), sc);
      await settle(15);
      check('scenario ' + sc, await stats());
    }
    await shot('05-scenario-wander');

    // --- walk inside the crowd (worst case for culling) -------------------
    await page.evaluate(() => {
      window.rig.setMode('walk');
      window.rig.walkPos.set(0, 1.72, 0);
    });
    await settle(30);
    const walk = await stats();
    check('walk-in-crowd', walk);
    await shot('06-walk-inside');

    // --- the crowd must actually be in the shadow pass ---------------------
    await page.evaluate(() => {
      window.renderer.shadowMap.enabled = false;
      window.env.sun.castShadow = false;
      window.scene.traverse((o) => { if (o.material) o.material.needsUpdate = true; });
    });
    await settle(20);
    const noShadow = await stats();
    check('shadows off', noShadow);
    const shadowDraws = walk.drawCalls - noShadow.drawCalls;
    notes.push(`shadow pass draws        ${shadowDraws} (${walk.drawCalls} with shadows, ${noShadow.drawCalls} without)`);
    if (shadowDraws <= 0) {
      problems.push('the crowd is not being drawn into the shadow map - customDepthMaterial is not taking effect');
    }

    // --- pixel-perfect sanity: is the frame actually non-empty? -----------
    await page.evaluate(() => {
      window.renderer.shadowMap.enabled = true;
      window.env.sun.castShadow = true;
      window.rig.setMode('orbit');
    });
    await settle(30);
    await page.evaluate(() => { window.__vat.wantSample = true; });
    await settle(3);
    const pixels = await page.evaluate(() => window.__vat.sample);
    notes.push(`frame colour variety     ${pixels.uniqueColours} buckets, mean luma ${pixels.meanLuma.toFixed(1)}`);
    if (pixels.uniqueColours < 40) problems.push('rendered frame is nearly flat - probably drawing nothing visible');
    if (pixels.meanLuma < 8) problems.push('rendered frame is almost black');
    await shot('07-final');

    const programs = await page.evaluate(() => window.renderer.info.programs.length);
    notes.push(`shader programs          ${programs}`);

    // CPU scaling is measured by tools/bench-cpu.mjs instead: a software
    // rasteriser saturates the cores in the background, so any timing taken in
    // this process is contended and not worth asserting on.

  } finally {
    // a leaked headless browser keeps rasterising in the background and poisons
    // every timing taken afterwards, so always tear it down
    await browser.close().catch(() => {});
    server.close();
  }
}

main()
  .then(() => {
    console.log('');
    notes.forEach((n) => console.log('  ' + n));
    console.log('');
    console.log(`screenshots -> ${path.relative(root, outDir)}`);
    if (problems.length) {
      console.log('\nFAILED');
      [...new Set(problems)].forEach((p) => console.log('  - ' + p));
      process.exit(1);
    }
    console.log('OK  renderer, both playback modes and every runtime toggle drew without errors');
  })
  .catch((err) => {
    console.error('smoke failed:', err.message);
    if (problems.length) [...new Set(problems)].forEach((p) => console.error('  - ' + p));
    process.exit(1);
  });
