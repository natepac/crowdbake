#!/usr/bin/env node
// Human-readable dump of a bake. Use it to sanity-check memory budgets and
// clip/stride data without loading the demo.
//
//   node tools/inspect.mjs public/baked/crowd

import fs from 'node:fs';
import path from 'node:path';
import { humanBytes } from './lib/pack.mjs';

const prefix = process.argv[2] || 'public/baked/crowd';
const manifest = JSON.parse(fs.readFileSync(prefix + '.json', 'utf8'));
const binSize = fs.statSync(path.join(path.dirname(prefix), manifest.binary)).size;

const pad = (s, n) => String(s).padEnd(n);
const padL = (s, n) => String(s).padStart(n);

console.log(`
${manifest.name}   (${manifest.generator} v${manifest.version})
source        ${manifest.source}
mesh          ${manifest.vertexCount.toLocaleString()} verts, ${manifest.triangleCount.toLocaleString()} tris, ${manifest.boneCount} bones
texture       ${manifest.texture.width} x ${manifest.texture.height}   pos ${manifest.texture.posFormat}   nrm ${manifest.texture.nrmFormat}${manifest.texture.hasTangents ? '   (+tangents)' : ''}
frames        ${manifest.totalFrames} total across ${manifest.clips.length} clips @ ${manifest.bakeFps} fps
bounds        min [${manifest.bounds.min.map((v) => v.toFixed(3)).join(', ')}]  extent [${manifest.bounds.extent.map((v) => v.toFixed(3)).join(', ')}]
              radius ${manifest.bounds.maxRadiusXZ.toFixed(3)} m, height ${manifest.bounds.height.toFixed(3)} m
`);

console.log('clips');
console.log(`  ${pad('name', 10)}${padL('frames', 7)}${padL('secs', 8)}${padL('stride m', 11)}${padL('m/s', 8)}   method / confidence`);
for (const c of manifest.clips) {
  const gt = c.groundTruthStride != null && c.groundTruthStride > 0
    ? `   truth ${c.groundTruthStride.toFixed(3)} (${(((c.stride - c.groundTruthStride) / c.groundTruthStride) * 100).toFixed(2)}%)`
    : '';
  console.log(`  ${pad(c.name, 10)}${padL(c.frameCount, 7)}${padL(c.duration.toFixed(2), 8)}`
    + `${padL(c.stride.toFixed(4), 11)}${padL(c.groundSpeed.toFixed(3), 8)}   ${c.strideMethod} `
    + `${(c.strideConfidence * 100).toFixed(0)}%${gt}`);
}

console.log('\nlods');
for (const l of manifest.lods) {
  const bar = '#'.repeat(Math.max(1, Math.round(l.ratio * 34)));
  console.log(`  L${l.level}  ${padL(l.vertexCount, 6)} verts  ${padL(l.triangleCount, 6)} tris  `
    + `${padL((l.ratio * 100).toFixed(1) + '%', 7)}  ${bar}`);
}

const sec = manifest.sections;
const posBytes = sec.posTex.byteLength;
const nrmBytes = sec.nrmTex.byteLength;
const boneBytes = sec.boneTex ? sec.boneTex.byteLength : 0;
const geomBytes = binSize - posBytes - nrmBytes - boneBytes;
const texels = manifest.vertexCount * manifest.totalFrames;

console.log('\nmemory');
const row = (name, bytes) => console.log(`  ${pad(name, 14)}${padL(humanBytes(bytes), 10)}   ${padL(((bytes / binSize) * 100).toFixed(1) + '%', 7)}`);
row('positions', posBytes);
row('normals', nrmBytes);
if (boneBytes) row('bone matrices', boneBytes);
row('geometry', geomBytes);
row('total', binSize);
console.log(`  per vertex-frame  ${((posBytes + nrmBytes) / texels).toFixed(1)} bytes`);
if (boneBytes) {
  console.log(`  bone-mode animation data is ${((posBytes + nrmBytes) / boneBytes).toFixed(0)}x smaller `
    + `(${humanBytes(boneBytes)} vs ${humanBytes(posBytes + nrmBytes)})`);
}

console.log('\nsockets');
for (const [name, bone] of Object.entries(manifest.sockets || {})) {
  const r = manifest.boneRest ? manifest.boneRest[bone] : null;
  console.log(`  ${pad(name, 10)} bone ${padL(bone, 3)} ${manifest.boneNames[bone]}`
    + (r ? `   rest [${r.map((v) => v.toFixed(3)).join(', ')}]` : ''));
}
console.log('');
