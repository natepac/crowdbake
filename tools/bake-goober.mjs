#!/usr/bin/env node
// Capture a goober (SDF blend-shell critter from C:\Projects\MGameExample\goobers)
// and turn it into a frames dump that tools/bake.mjs can bake like any rig.
//
//   node tools/bake-goober.mjs --kind goober-biped --seed 7 --out tools/_out/goober-biped
//
// How: goobers.html is loaded in headless Chrome with requestAnimationFrame
// stubbed out, so the app initialises but its own loop never runs. We drive
// critter.update() ourselves at a fixed timestep, walking the critter in a
// straight line, and snapshot the prim uniforms (the exact values its shader
// would see) each frame. Everything else -- SDF vertex snapping, loop finding,
// re-centering -- happens in Node with the ported math in lib/goober-sdf.mjs.

import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { performance } from 'node:perf_hooks';
import puppeteer from 'puppeteer-core';
import { snapFrame, capsuleTemplate } from './lib/goober-sdf.mjs';

const DEFAULT_GOOBERS_DIR = 'vendor/goobers';
const GROUND_R = 260;
const groundY = (x, z) => { const q = GROUND_R * GROUND_R - x * x - z * z; return q > 0 ? Math.sqrt(q) - GROUND_R : -2; };

function parseArgs(argv) {
  const o = { kind: 'goober-biped', seed: 7, out: null, fps: 40, captureSec: 6, warmupSec: 4, quiet: false, list: false, dir: DEFAULT_GOOBERS_DIR };
  for (let i = 0; i < argv.length; i++) {
    const next = () => argv[++i];
    switch (argv[i]) {
      case '--kind': o.kind = next(); break;
      case '--seed': o.seed = parseInt(next(), 10); break;
      case '--out': o.out = next(); break;
      case '--fps': o.fps = parseFloat(next()); break;
      case '--dir': o.dir = next(); break;
      case '--list': o.list = true; break;
      case '--quiet': o.quiet = true; break;
      default: break;
    }
  }
  if (!o.out) o.out = 'tools/_out/' + o.kind;
  return o;
}

const BROWSERS = [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
];

function serve(dir) {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const url = decodeURIComponent(req.url.split('?')[0]);
      const file = path.join(dir, url === '/' ? 'index.html' : url);
      if (!fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404).end(); return; }
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(fs.readFileSync(file));
    });
    server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port }));
  });
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const log = opts.quiet ? () => {} : (...a) => console.log(...a);
  const t0 = performance.now();

  const { server, port } = await serve(path.resolve(opts.dir));
  const exe = BROWSERS.find((b) => fs.existsSync(b));
  const browser = await puppeteer.launch({
    executablePath: exe,
    headless: true,
    args: ['--use-angle=swiftshader', '--use-gl=angle', '--enable-unsafe-swiftshader', '--no-sandbox'],
    defaultViewport: { width: 640, height: 400 },
  });
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.error('  page error:', e.message));

  // serve their CDN three imports from the local copy (works offline; r180's
  // module build re-exports from ./three.core.js, so both must be routed)
  const threeFiles = {
    'three.module.js': fs.readFileSync('node_modules/three/build/three.module.js', 'utf8'),
    'three.core.js': fs.readFileSync('node_modules/three/build/three.core.js', 'utf8'),
  };
  await page.setRequestInterception(true);
  page.on('request', (req) => {
    const m = req.url().match(/cdn\.jsdelivr\.net.*\/(three\.[a-z]+\.js)$/);
    if (m && threeFiles[m[1]]) {
      req.respond({ status: 200, contentType: 'text/javascript', headers: { 'Access-Control-Allow-Origin': '*' }, body: threeFiles[m[1]] });
    } else req.continue();
  });

  // deterministic RNG + dead rAF, installed before any page script runs
  await page.evaluateOnNewDocument((seed) => {
    let s = seed >>> 0;
    Math.random = () => {
      s |= 0; s = (s + 0x6D2B79F5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    window.requestAnimationFrame = () => 0;
  }, opts.seed);

  log(`goober        ${opts.kind}  (seed ${opts.seed})`);
  await page.goto(`http://127.0.0.1:${port}/goobers.html`, { waitUntil: 'networkidle0', timeout: 60000 });
  await page.waitForFunction('window.__goobersReady === true', { timeout: 60000 });

  if (opts.list) {
    // every preset kind across every family, as machine-readable JSON
    const kinds = await page.evaluate(() => {
      const out = [];
      for (const f of window.GOOBERS.families) {
        for (const k of window.GOOBERS.presets(f)) out.push({ kind: k, family: f });
      }
      return out;
    });
    console.log(JSON.stringify(kinds));
    await browser.close();
    server.close();
    return;
  }

  // ---- capture one clip: drive update() with a fixed timestep -------------
  const capture = (mode, seconds, warmup) => page.evaluate((mode2, seconds2, warmup2, fps) => {
    const G = window.GOOBERS;
    G.clear();
    const c = G.spawn(window.__bakeDef || (window.__bakeDef = G.random(window.__bakeKind)), 0, -15);
    c.heading = 0; c.yaw = 0;

    const dt = 1 / fps;
    let t = 1000;                                  // arbitrary, away from 0
    const steer = () => {
      if (mode2 === 'walk') c.target.set(0, 0, 500);   // dead ahead, never reached
      else c.target.set(c.pos.x, 0, c.pos.z);          // nowhere to go -> idles
    };
    for (let i = 0; i < warmup2 * fps; i++) { steer(); t += dt; c.update(dt, t); }

    const frames = [];
    const count = c.uniforms.uCount.value;
    const uA = c.uniforms.uPrimA.value, uB = c.uniforms.uPrimB.value, uC = c.uniforms.uPrimC.value;
    const n = Math.round(seconds2 * fps);
    for (let f = 0; f < n; f++) {
      steer(); t += dt; c.update(dt, t);
      const prims = new Array(count * 12);
      for (let i = 0; i < count; i++) {
        const o = i * 12;
        prims[o] = uA[i].x; prims[o + 1] = uA[i].y; prims[o + 2] = uA[i].z; prims[o + 3] = uA[i].w;
        prims[o + 4] = uB[i].x; prims[o + 5] = uB[i].y; prims[o + 6] = uB[i].z; prims[o + 7] = uB[i].w;
        prims[o + 8] = uC[i].x; prims[o + 9] = uC[i].y; prims[o + 10] = uC[i].z; prims[o + 11] = uC[i].w;
      }
      const eyes = [];
      for (const e of c.eyes) {
        eyes.push({
          wx: e.w.position.x, wy: e.w.position.y, wz: e.w.position.z, wr: e.w.scale.z,
          px: e.p.position.x, py: e.p.position.y, pz: e.p.position.z, pr: e.p.scale.z,
        });
      }
      frames.push({ prims, eyes, x: c.pos.x, z: c.pos.z, heading: c.heading, speed: c.speed });
    }

    const geo = c.mesh.geometry;
    return {
      def: { kind: c.def.kind, name: c.def.name, size: c.def.size },
      count,
      consts: {
        tuckDepth: c.uniforms.uTuckDepth.value,
        tuck0: c.uniforms.uTuck0.value,
        tuck1: c.uniforms.uTuck1.value,
        gradEps: c.uniforms.uGradEps.value,
        stepMax: c.uniforms.uStepMax.value,
        offset: 0,
      },
      template: {
        pos: Array.from(geo.attributes.position.array),
        aPrim: Array.from(geo.attributes.aPrim.array),
        idx: Array.from(geo.index.array),
      },
      pupilColor: (() => {
        const m = c.eyes[0] && c.eyes[0].p.material;
        const col = m && m.color ? m.color : { r: 0.06, g: 0.06, b: 0.08 };
        return [col.r, col.g, col.b];
      })(),
      frames,
    };
  }, mode, seconds, warmup, opts.fps);

  await page.evaluate((kind) => { window.__bakeKind = kind; window.__bakeDef = null; }, opts.kind);
  const walk = await capture('walk', opts.captureSec, opts.warmupSec);
  const idle = await capture('idle', 4.5, 3);
  await browser.close();
  server.close();

  log(`captured      ${walk.frames.length} walk + ${idle.frames.length} idle frames, ${walk.count} prims, ${walk.template.aPrim.length} template verts`);

  // ---- find the best loop inside each capture -----------------------------
  const dtCap = 1 / opts.fps;
  const findLoop = (frames, minSec, maxSec) => {
    const sig = frames.map((f) => f.prims);
    const dist = (a, b) => {
      let s = 0;
      const n = Math.min(a.length, b.length);
      // endpoints only; radii/colors are static
      for (let i = 0; i < n; i += 12) {
        for (let c = 0; c < 7; c++) { if (c === 3) continue; const d = a[i + c] - b[i + c]; s += d * d; }
      }
      return s;
    };
    let best = { i: 0, j: Math.min(frames.length - 1, Math.round(1.2 / dtCap)), d: Infinity };
    const minF = Math.max(4, Math.round(minSec / dtCap));
    const maxF = Math.round(maxSec / dtCap);
    for (let i = 0; i < frames.length - minF; i++) {
      for (let j = i + minF; j < Math.min(frames.length, i + maxF); j++) {
        // compare pose AND velocity so the seam does not pop
        const d = dist(centered(frames[i]), centered(frames[j]))
          + dist(centered(frames[Math.min(i + 1, frames.length - 1)]), centered(frames[Math.min(j + 1, frames.length - 1)]));
        if (d < best.d) best = { i, j, d };
      }
    }
    return best;
  };

  // re-center a frame's prims into critter-local space (also used by findLoop)
  const centeredCache = new Map();
  function centered(frame) {
    let c = centeredCache.get(frame);
    if (c) return c;
    const gy = groundY(frame.x, frame.z);
    const cos = Math.cos(-frame.heading), sin = Math.sin(-frame.heading);
    c = frame.prims.slice();
    for (let i = 0; i < c.length; i += 12) {
      for (const off of [0, 4]) {
        const x = c[i + off] - frame.x;
        const y = c[i + off + 1] - gy;
        const z = c[i + off + 2] - frame.z;
        c[i + off] = cos * x + sin * z;
        c[i + off + 1] = y;
        c[i + off + 2] = -sin * x + cos * z;
      }
    }
    centeredCache.set(frame, c);
    return c;
  }
  function centeredEyes(frame) {
    const gy = groundY(frame.x, frame.z);
    const cos = Math.cos(-frame.heading), sin = Math.sin(-frame.heading);
    return frame.eyes.map((e) => {
      const rot = (x, y, z) => [cos * (x - frame.x) + sin * (z - frame.z), y - gy, -sin * (x - frame.x) + cos * (z - frame.z)];
      const w = rot(e.wx, e.wy, e.wz), p = rot(e.px, e.py, e.pz);
      return { w, wr: e.wr, p, pr: e.pr };
    });
  }

  const loops = {
    walk: findLoop(walk.frames, 0.55, 3.0),
    idle: findLoop(idle.frames, 1.2, 4.2),
  };
  for (const [name, l] of Object.entries(loops)) {
    log(`loop ${name.padEnd(6)}  frames ${l.i}..${l.j}  (${((l.j - l.i) * dtCap).toFixed(2)} s, seam err ${Math.sqrt(l.d).toExponential(2)})`);
  }

  // ---- build the final template (body prims + eye spheres) ----------------
  // prims parked out of sight (y < -50) in every frame are dropped outright
  const parkedAlways = [];
  for (let i = 0; i < walk.count; i++) {
    let parked = true;
    for (const f of walk.frames) { if (f.prims[i * 12 + 1] > -50) { parked = false; break; } }
    for (const f of idle.frames) { if (f.prims[i * 12 + 1] > -50) { parked = false; break; } }
    parkedAlways[i] = parked;
  }

  const eyeCount = walk.frames[0].eyes.length;

  // Eyes are NOT part of the SDF -- the original renders them as separate rigid
  // spheres, and putting them into the field makes the shell's tuck logic carve
  // craters around the embedded pupil. Eye vertices are placed directly.
  const sphereTpl = (() => {
    const segU = 10, segV = 7, pos = [], idx = [];
    for (let v = 0; v <= segV; v++) {
      const ph = (v / segV) * Math.PI;
      for (let u = 0; u <= segU; u++) {
        const th = (u / segU) * Math.PI * 2;
        pos.push(Math.sin(ph) * Math.cos(th), Math.cos(ph), Math.sin(ph) * Math.sin(th));
      }
    }
    for (let v = 0; v < segV; v++) for (let u = 0; u < segU; u++) {
      const a = v * (segU + 1) + u, b = a + 1, c = a + segU + 1, d = c + 1;
      idx.push(a, c, b, b, c, d);
    }
    return { pos, idx };
  })();
  const sphereVerts = sphereTpl.pos.length / 3;

  const keepPrim = (pi) => !parkedAlways[pi];
  const tplPos = [], tplPrim = [], tplIdx = [];
  {
    const remap = new Map();
    const src = walk.template;
    for (let t = 0; t < src.idx.length; t += 3) {
      const a = src.idx[t], b = src.idx[t + 1], cc = src.idx[t + 2];
      if (!keepPrim(src.aPrim[a])) continue;
      for (const v of [a, b, cc]) {
        if (!remap.has(v)) {
          remap.set(v, tplPrim.length);
          tplPos.push(src.pos[v * 3], src.pos[v * 3 + 1], src.pos[v * 3 + 2]);
          tplPrim.push(src.aPrim[v]);
        }
      }
      tplIdx.push(remap.get(a), remap.get(b), remap.get(cc));
    }
  }
  const bodyVerts = tplPrim.length;
  // eye spheres appended after the body: 2 per eye (white, pupil)
  for (let e = 0; e < eyeCount * 2; e++) {
    const base = tplPrim.length;
    for (let v = 0; v < sphereVerts; v++) {
      tplPos.push(sphereTpl.pos[v * 3], sphereTpl.pos[v * 3 + 1], sphereTpl.pos[v * 3 + 2]);
      tplPrim.push(walk.count + e);        // distinct id, used only for LOD clustering
    }
    for (const ix of sphereTpl.idx) tplIdx.push(base + ix);
  }
  const template = { pos: Float32Array.from(tplPos), aPrim: Uint16Array.from(tplPrim) };
  const bodyTemplate = {
    pos: template.pos.subarray(0, bodyVerts * 3),
    aPrim: template.aPrim.subarray(0, bodyVerts),
  };
  const indices = Uint32Array.from(tplIdx);
  const vertexCount = template.aPrim.length;
  const totalPrims = walk.count;
  log(`template      ${vertexCount} verts, ${indices.length / 3} tris, ${walk.count} body + ${eyeCount * 2} eye prims`
    + (parkedAlways.some(Boolean) ? ` (${parkedAlways.filter(Boolean).length} parked prims dropped)` : ''));

  // ---- snap every kept frame ---------------------------------------------
  const clips = [];
  const positions = [];
  const normals = [];
  let colors = null;

  for (const [name, cap] of [['idle', idle], ['walk', walk]]) {
    const l = loops[name];
    const frameCount = l.j - l.i;
    const framePrims = new Float32Array(totalPrims * 12);
    const outPos = [];
    const outNrm = [];
    const tSnap = performance.now();
    for (let f = l.i; f < l.j; f++) {
      const frame = cap.frames[f];
      framePrims.set(centered(frame), 0);
      const pos = new Float32Array(vertexCount * 3);
      const nrm = new Float32Array(vertexCount * 3);
      const wantColor = colors === null;
      const col = wantColor ? new Float32Array(vertexCount * 3) : null;
      snapFrame(bodyTemplate, framePrims, totalPrims, walk.consts, pos, nrm, col);

      // eyes: rigid spheres straight from the captured transforms
      const eyes = centeredEyes(frame);
      let vo = bodyVerts;
      for (let ei = 0; ei < eyes.length; ei++) {
        const e = eyes[ei];
        for (const [c, r, ecol] of [[e.w, Math.max(e.wr, 0.008), [0.97, 0.97, 1.0]], [e.p, Math.max(e.pr, 0.005), walk.pupilColor]]) {
          for (let v = 0; v < sphereVerts; v++) {
            const nx = sphereTpl.pos[v * 3], ny = sphereTpl.pos[v * 3 + 1], nz = sphereTpl.pos[v * 3 + 2];
            const i3 = (vo + v) * 3;
            pos[i3] = c[0] + nx * r; pos[i3 + 1] = c[1] + ny * r; pos[i3 + 2] = c[2] + nz * r;
            nrm[i3] = nx; nrm[i3 + 1] = ny; nrm[i3 + 2] = nz;
            if (col) { col[i3] = ecol[0]; col[i3 + 1] = ecol[1]; col[i3 + 2] = ecol[2]; }
          }
          vo += sphereVerts;
        }
      }
      if (wantColor) colors = col;
      outPos.push(pos);
      outNrm.push(nrm);
    }
    const duration = frameCount * dtCap;
    // ground truth stride: distance the critter actually travelled over the loop
    const a = cap.frames[l.i], b = cap.frames[l.j];
    const stride = Math.hypot(b.x - a.x, b.z - a.z);
    clips.push({
      name, frameCount, duration, loop: true,
      stride: name === 'walk' ? stride : 0,
      groundSpeed: name === 'walk' ? stride / duration : 0,
    });
    positions.push(...outPos);
    normals.push(...outNrm);
    log(`snapped ${name.padEnd(6)} ${frameCount} frames in ${((performance.now() - tSnap) / 1000).toFixed(1)} s`
      + (name === 'walk' ? `   stride ${stride.toFixed(3)} u/cycle (${(stride / duration).toFixed(3)} u/s)` : ''));
  }

  // ---- write the dump ------------------------------------------------------
  const totalFrames = clips.reduce((s, c) => s + c.frameCount, 0);
  const posAll = new Float32Array(totalFrames * vertexCount * 3);
  const nrmAll = new Float32Array(totalFrames * vertexCount * 3);
  positions.forEach((p, i) => posAll.set(p, i * vertexCount * 3));
  normals.forEach((p, i) => nrmAll.set(p, i * vertexCount * 3));

  fs.mkdirSync(path.dirname(path.resolve(opts.out)), { recursive: true });
  // pad the u16 aPrim block to 4 bytes so the u32 indices stay aligned
  const aPrimPadded = new Uint16Array(vertexCount + (vertexCount % 2));
  aPrimPadded.set(template.aPrim);
  const bin = Buffer.concat([
    Buffer.from(posAll.buffer),
    Buffer.from(nrmAll.buffer),
    Buffer.from(colors.buffer),
    Buffer.from(aPrimPadded.buffer),
    Buffer.from(indices.buffer),
  ]);
  const manifest = {
    format: 'goober-frames/1',
    kind: opts.kind,
    seed: opts.seed,
    def: walk.def,
    vertexCount,
    indexCount: indices.length,
    primCount: totalPrims,
    fps: opts.fps,
    clips,
    layout: {
      positions: { offset: 0, floats: posAll.length },
      normals: { offset: posAll.byteLength, floats: nrmAll.length },
      colors: { offset: posAll.byteLength + nrmAll.byteLength, floats: colors.length },
      aPrim: { offset: posAll.byteLength + nrmAll.byteLength + colors.byteLength, u16: vertexCount },
      indices: { offset: posAll.byteLength + nrmAll.byteLength + colors.byteLength + aPrimPadded.byteLength, u32: indices.length },
    },
  };
  fs.writeFileSync(opts.out + '.frames.json', JSON.stringify(manifest, null, 2));
  fs.writeFileSync(opts.out + '.frames.bin', bin);
  log(`dump          ${opts.out}.frames.{json,bin}  (${(bin.length / 1048576).toFixed(2)} MB, ${totalFrames} frames)`);
  log(`elapsed       ${((performance.now() - t0) / 1000).toFixed(1)} s`);
}

main().catch((err) => { console.error('bake-goober failed:', err.message); if (process.env.DEBUG) console.error(err.stack); process.exit(1); });
