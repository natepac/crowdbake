#!/usr/bin/env node
// Headless CPU benchmark for the crowd runtime.
//
// VATCrowd and CrowdSim never touch a GL context -- they only build three
// scene-graph objects and shuffle typed arrays -- so the whole per-frame CPU
// cost can be measured in Node, with no renderer and none of the background
// noise a software rasteriser generates. This is the number that decides
// whether a 100k crowd is GPU-bound or CPU-bound.
//
//   node tools/bench-cpu.mjs
//   node tools/bench-cpu.mjs --counts 1000,20000,100000 --iters 60

import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { PerspectiveCamera } from 'three';

import { VATAsset } from '../src/vat/VATAsset.js';
import { VATCrowd } from '../src/vat/VATCrowd.js';
import { CrowdSim } from '../src/demo/crowd-sim.js';

// crowd-sim uses performance.now(); Node has it on globalThis from perf_hooks
if (typeof globalThis.performance === 'undefined') globalThis.performance = performance;

function parseArgs(argv) {
  const o = {
    prefix: 'public/baked/crowd',
    counts: [1000, 10000, 25000, 50000, 100000],
    iters: 45, warmup: 15, worldSize: 420, chunkSize: 24, scenario: 'wander',
  };
  for (let i = 0; i < argv.length; i++) {
    const next = () => argv[++i];
    switch (argv[i]) {
      case '--counts': o.counts = next().split(',').map(Number); break;
      case '--iters': o.iters = parseInt(next(), 10); break;
      case '--warmup': o.warmup = parseInt(next(), 10); break;
      case '--chunk': o.chunkSize = parseFloat(next()); break;
      case '--scenario': o.scenario = next(); break;
      case '--json': o.json = true; break;
      default: if (!argv[i].startsWith('-')) o.prefix = argv[i];
    }
  }
  return o;
}

const opts = parseArgs(process.argv.slice(2));
const manifest = JSON.parse(fs.readFileSync(opts.prefix + '.json', 'utf8'));
const binBuf = fs.readFileSync(path.join(path.dirname(opts.prefix), manifest.binary));
const asset = new VATAsset(manifest, binBuf.buffer.slice(binBuf.byteOffset, binBuf.byteOffset + binBuf.byteLength));

const capacity = Math.max(...opts.counts);
const camera = new PerspectiveCamera(58, 16 / 9, 0.3, 900);
camera.position.set(28, 16, 44);
camera.lookAt(0, 1.2, 0);
camera.updateMatrixWorld(true);

// lodDistances deliberately left at the library default so the benchmark
// reflects what the demo actually ships with
const crowd = new VATCrowd(asset, {
  capacity,
  chunkSize: opts.chunkSize,
  worldSize: opts.worldSize,
});

const obstacles = [];
for (let i = 0; i < 26; i++) {
  const a = (i / 26) * Math.PI * 2 * 3.3;
  const r = opts.worldSize * 0.42 * Math.sqrt((i + 0.5) / 26);
  obstacles.push({ x: Math.cos(a) * r, z: Math.sin(a) * r, radius: 1.9 });
}

const sim = new CrowdSim(crowd, {
  worldSize: opts.worldSize,
  obstacles,
  separation: true,
  separationMaxAgents: 40000,
  scenario: opts.scenario,
});
sim.spawn(capacity);

const dt = 1 / 60;
const rows = [];

for (const count of opts.counts) {
  crowd.count = count;
  for (let i = 0; i < opts.warmup; i++) { sim.update(dt, opts.worldSize); crowd.update(dt, camera); }

  // Per-iteration samples, reported as medians. A laptop under sustained load
  // throttles hard enough to move a mean by 3x; the median (and the min, shown
  // alongside) is what the code actually costs when the core is not being
  // stolen by something else.
  const sSim = [], sSep = [], sCrowd = [], sBucket = [], sCopy = [], sTotal = [];
  let dirty = 0, upload = 0;
  for (let i = 0; i < opts.iters; i++) {
    let t = performance.now();
    sim.update(dt, opts.worldSize);
    const a = performance.now() - t;
    t = performance.now();
    crowd.update(dt, camera);
    const b = performance.now() - t;
    sSim.push(a); sCrowd.push(b); sTotal.push(a + b);
    sBucket.push(crowd.stats.bucketMs);
    sCopy.push(crowd.stats.copyMs);
    sSep.push(sim.stats.sepMs);
    dirty += crowd.stats.dirtyChunks;
    upload += crowd.stats.uploadBytes;
  }
  const med = (arr) => { const a = arr.slice().sort((x, y) => x - y); return a[a.length >> 1]; };
  const n = opts.iters;
  rows.push({
    count,
    sim: med(sSim),
    sep: med(sSep),
    crowd: med(sCrowd),
    bucket: med(sBucket),
    copy: med(sCopy),
    total: med(sTotal),
    best: Math.min(...sTotal),
    dirty: dirty / n,
    chunks: crowd.stats.chunks,
    uploadMB: upload / n / 1048576,
    nsPerInstance: med(sTotal) * 1e6 / count,
    triangles: crowd.stats.triangles,
    lodCounts: Array.from(crowd.stats.lodCounts),
  });
}

if (opts.json) {
  console.log(JSON.stringify(rows, null, 2));
} else {
  const p = (v, w, d = 2) => v.toFixed(d).padStart(w);
  console.log(`\ncpu cost per frame, no renderer   (${asset.name}, chunk ${opts.chunkSize} m, ${opts.scenario})`);
  console.log('  medians of ' + opts.iters + ' frames, ms');
  console.log('  instances     sim  (sep)   chunk  bucket    copy   total    best   ns/inst   upload');
  for (const r of rows) {
    console.log(`  ${String(r.count).padStart(9)}  ${p(r.sim, 6)} ${p(r.sep, 6)}  ${p(r.crowd, 6)}  `
      + `${p(r.bucket, 6)}  ${p(r.copy, 6)}  ${p(r.total, 6)}  ${p(r.best, 6)}   ${p(r.nsPerInstance, 7, 0)}   `
      + `${p(r.uploadMB, 5)} MB`);
  }
  console.log('');
  console.log('  submitted geometry (pre frustum cull), and the LOD split');
  console.log('  instances    triangles   ' + asset.lods.map((l, i) => ('L' + i + ' (' + l.triangleCount + 't)').padStart(12)).join(''));
  for (const r of rows) {
    console.log(`  ${String(r.count).padStart(9)}  ${(r.triangles / 1e6).toFixed(2).padStart(8)} M   `
      + r.lodCounts.map((c) => String(c).padStart(12)).join(''));
  }

  const first = rows[0], last = rows[rows.length - 1];
  const cr = last.count / first.count;
  const tr = last.total / first.total;
  console.log(`\n  ${cr}x instances -> ${tr.toFixed(1)}x cost `
    + `(${tr < cr * 1.5 ? 'linear' : 'super-linear - investigate'})`);
  console.log(`  budget: ${last.count.toLocaleString()} instances leaves `
    + `${(16.7 - last.total).toFixed(1)} ms of a 60 fps frame for the GPU\n`);
}
