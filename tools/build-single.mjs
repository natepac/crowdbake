#!/usr/bin/env node
// Produce dist/crowdbake.html: the entire app in ONE file -- bundle inlined,
// baked assets embedded as base64 -- for hosts where a single HTML upload is
// the whole deployment (itch.io, a CMS page, an email attachment).
//
//   npm run build:single
//   node tools/build-single.mjs --assets crowd                 # lean file
//   node tools/build-single.mjs --assets crowd,goober-cat
//
// The page detects the embedded registry at runtime: VATAsset.load() serves
// from it instead of fetching, and the asset dropdown lists only what is
// embedded. The menagerie is deliberately not embeddable -- 40 kinds is a
// quarter-gigabyte of base64, which no single page should be.

import fs from 'node:fs';
import path from 'node:path';

const argAssets = (() => {
  const i = process.argv.indexOf('--assets');
  if (i < 0) return ['crowd', 'goober-biped', 'goober-quad', 'goober-cat'];
  return process.argv[i + 1].split(',').map((s) => s.trim()).filter(Boolean);
})();

const dist = path.resolve('dist');
const html = fs.readFileSync(path.join(dist, 'index.html'), 'utf8');

const scriptMatch = html.match(/<script type="module"[^>]*src="\.\/(assets\/index-[^"]+\.js)"[^>]*><\/script>/);
if (!scriptMatch) {
  throw new Error('dist/index.html is already inlined (or not a Vite build) - run "npm run build" first;'
    + ' "npm run build:single" does both in order');
}
const bundle = fs.readFileSync(path.join(dist, scriptMatch[1]), 'utf8')
  .replace(/<\/script>/g, '<\\/script>');

const registry = {};
let assetBytes = 0;
for (const name of argAssets) {
  const jsonPath = path.join(dist, 'baked', name + '.json');
  if (!fs.existsSync(jsonPath)) {
    if (name === 'menagerie') throw new Error('the menagerie cannot be embedded in a single file');
    throw new Error(`no bake for "${name}" in dist/baked - bake it first`);
  }
  const manifest = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
  const bin = fs.readFileSync(path.join(dist, 'baked', manifest.binary));
  registry[name] = { manifest, bin: bin.toString('base64') };
  assetBytes += bin.length;
  console.log(`  embed  ${name.padEnd(16)} ${(bin.length / 1048576).toFixed(1)} MB`);
}

const registryScript = '<script>window.__CROWDBAKE_INLINE=' + JSON.stringify(registry)
  .replace(/<\/script>/g, '<\\/script>') + '</script>';

// replacer FUNCTION, not string: the bundle contains "$&"-style sequences that
// String.replace would expand into the matched tag itself, re-injecting a raw
// </script> into the page and truncating the inline module at parse time
const out = html.replace(scriptMatch[0],
  () => registryScript + '\n<script type="module">' + bundle + '</script>');

// dist/index.html IS the all-in-one file, so posting it anywhere (or the whole
// dist folder) just works; crowdbake.html is the same bytes under a name that
// survives being dropped next to some other site's index.
fs.writeFileSync(path.join(dist, 'index.html'), out);
fs.writeFileSync(path.join(dist, 'crowdbake.html'), out);
const mbOut = (fs.statSync(path.join(dist, 'index.html')).size / 1048576).toFixed(1);
console.log(`\ndist/index.html + dist/crowdbake.html  ${mbOut} MB each `
  + `(${argAssets.length} assets, ${(assetBytes / 1048576).toFixed(1)} MB binary before base64)`);
console.log('either file is the whole app; dist/baked/ alongside them still enables ?asset=menagerie');
