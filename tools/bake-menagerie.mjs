#!/usr/bin/env node
// Bake EVERY goober kind and write the menagerie manifest the demo's
// `?asset=menagerie` mode loads (all kinds instanced together in one scene).
//
//   node tools/bake-menagerie.mjs             # skips kinds already baked
//   node tools/bake-menagerie.mjs --force     # rebake everything
//
// Sequential on purpose: each capture owns a headless Chrome, and reliability
// beats a few saved minutes on a one-time bake.

import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';

const force = process.argv.includes('--force');
const t0 = performance.now();

// deterministic seed per kind so a fresh clone bakes the same menagerie
function seedOf(kind) {
  let h = 2166136261;
  for (const c of kind) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619); }
  return (h >>> 8) % 0xffffff + 1;
}

const run = (args) => execFileSync(process.execPath, args, { cwd: process.cwd(), encoding: 'utf8' });

console.log('listing kinds...');
const kinds = JSON.parse(run(['tools/bake-goober.mjs', '--list', '--quiet']).trim().split('\n').pop());
console.log(`${kinds.length} kinds`);

const baked = [];
const failed = [];
for (const { kind, family } of kinds) {
  const asset = 'goober-' + kind.replace(/^goober-/, '');
  const out = 'public/baked/' + asset;
  if (!force && fs.existsSync(out + '.json')) {
    baked.push({ asset, kind, family });
    console.log(`  skip   ${kind} (already baked)`);
    continue;
  }
  const t1 = performance.now();
  try {
    run(['tools/bake-goober.mjs', '--kind', kind, '--seed', String(seedOf(kind)),
      '--out', 'tools/_out/' + asset, '--quiet']);
    run(['tools/bake.mjs', '--frames-dump', 'tools/_out/' + asset + '.frames.json',
      '--fps', '40', '--lods', '5', '--out', out, '--quiet']);
    const mb = fs.statSync(out + '.bin').size / 1048576;
    baked.push({ asset, kind, family });
    console.log(`  baked  ${kind.padEnd(14)} ${mb.toFixed(1).padStart(5)} MB  (${((performance.now() - t1) / 1000).toFixed(0)} s)`);
  } catch (err) {
    failed.push({ kind, error: String(err.message).slice(0, 200) });
    console.log(`  FAIL   ${kind}: ${String(err.message).slice(0, 120)}`);
  }
}

fs.writeFileSync('public/baked/menagerie.json', JSON.stringify({
  format: 'menagerie/1',
  assets: baked,
}, null, 2));

const total = fs.readdirSync('public/baked').filter((f) => f.endsWith('.bin'))
  .reduce((s, f) => s + fs.statSync(path.join('public/baked', f)).size, 0);
console.log(`\nmenagerie.json: ${baked.length} kinds baked, ${failed.length} failed`);
if (failed.length) failed.forEach((f) => console.log(`  - ${f.kind}: ${f.error}`));
console.log(`public/baked total: ${(total / 1048576).toFixed(0)} MB`);
console.log(`elapsed ${((performance.now() - t0) / 1000 / 60).toFixed(1)} min`);
