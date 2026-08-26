#!/usr/bin/env node
// Numerical verification of a bake.
//
// Re-runs the CPU skinning that produced the bake and decodes the texture back
// the same way the vertex shader will, then reports the error. This is the only
// way to catch fp16 / bbox-normalisation mistakes without a GPU: if the numbers
// here are good, what the shader draws is right.
//
//   node tools/verify.mjs public/baked/crowd

import fs from 'node:fs';
import path from 'node:path';

import { makeProceduralRig } from './lib/procgen.mjs';
import { loadRigFromGLTF } from './lib/rig-gltf.mjs';
import { skinFrame, computeTangents } from './lib/skin.mjs';
import { unpackHalfArray } from './lib/half.mjs';
import { octDecode } from './lib/octa.mjs';

const prefix = process.argv[2] || 'public/baked/crowd';
const manifest = JSON.parse(fs.readFileSync(prefix + '.json', 'utf8'));
const binPath = path.join(path.dirname(prefix), manifest.binary);
const binBuf = fs.readFileSync(binPath);
const bin = binBuf.buffer.slice(binBuf.byteOffset, binBuf.byteOffset + binBuf.byteLength);

function section(name) {
  const s = manifest.sections[name];
  if (!s) return null;
  const ctor = { f32: Float32Array, u32: Uint32Array, u16: Uint16Array, f16: Uint16Array, u8: Uint8Array }[s.type];
  return { meta: s, data: new ctor(bin, s.offset, s.byteLength / ctor.BYTES_PER_ELEMENT) };
}

const bo = manifest.bakeOptions || {};
const model = bo.procgen === false && bo.input
  ? loadRigFromGLTF(path.resolve(bo.input), { skin: bo.skin || 0 })
  : makeProceduralRig({ density: bo.density != null ? bo.density : 1.4 });

if (!model.materialIds) model.materialIds = new Float32Array(model.vertexCount);

const { vertexCount } = manifest;
if (model.vertexCount !== vertexCount) {
  console.error(`vertex count mismatch: manifest ${vertexCount}, rebuilt model ${model.vertexCount}`);
  process.exit(1);
}

const posTex = unpackHalfArray(section('posTex').data);
const nrmSec = section('nrmTex');
const nrmTex = unpackHalfArray(nrmSec.data);
const nrmComps = nrmSec.meta.components;
const { min, extent } = manifest.bounds;

const skinMats = new Float32Array(model.boneCount * 16);
const fPos = new Float32Array(vertexCount * 3);
const fNrm = new Float32Array(vertexCount * 3);
const bindTan = manifest.texture.hasTangents ? computeTangents(model) : null;
const fTan = bindTan ? new Float32Array(vertexCount * 3) : null;
const dec = new Float32Array(3);

let maxPosErr = 0, sumPosErr = 0, nPos = 0;
let maxNrmDeg = 0, sumNrmDeg = 0;
let maxTanDeg = 0;
let worst = null;

const nameToClip = new Map(model.clips.map((c, i) => [c.name, i]));

for (const clip of manifest.clips) {
  const ci = nameToClip.get(clip.name);
  if (ci === undefined) { console.error('clip missing from model:', clip.name); process.exit(1); }
  for (let f = 0; f < clip.frameCount; f++) {
    const t = clip.loop
      ? (f / clip.frameCount) * clip.duration
      : (f / Math.max(1, clip.frameCount - 1)) * clip.duration;
    model.sampleSkinMatrices(ci, t, skinMats);
    skinFrame(model, skinMats, fPos, fNrm, bindTan, fTan);
    const base = (clip.frameStart + f) * vertexCount;
    for (let v = 0; v < vertexCount; v++) {
      const t4 = (base + v) * 4;
      const dx = (min[0] + posTex[t4] * extent[0]) - fPos[v * 3];
      const dy = (min[1] + posTex[t4 + 1] * extent[1]) - fPos[v * 3 + 1];
      const dz = (min[2] + posTex[t4 + 2] * extent[2]) - fPos[v * 3 + 2];
      const e = Math.hypot(dx, dy, dz);
      sumPosErr += e; nPos++;
      if (e > maxPosErr) { maxPosErr = e; worst = { clip: clip.name, frame: f, vertex: v }; }

      const no = (base + v) * nrmComps;
      octDecode(nrmTex[no], nrmTex[no + 1], dec, 0);
      const dot = Math.min(1, Math.max(-1, dec[0] * fNrm[v * 3] + dec[1] * fNrm[v * 3 + 1] + dec[2] * fNrm[v * 3 + 2]));
      const deg = Math.acos(dot) * 180 / Math.PI;
      sumNrmDeg += deg;
      if (deg > maxNrmDeg) maxNrmDeg = deg;

      if (nrmComps === 4 && fTan) {
        octDecode(nrmTex[no + 2], nrmTex[no + 3], dec, 0);
        const td = Math.min(1, Math.max(-1, dec[0] * fTan[v * 3] + dec[1] * fTan[v * 3 + 1] + dec[2] * fTan[v * 3 + 2]));
        const tdeg = Math.acos(td) * 180 / Math.PI;
        if (tdeg > maxTanDeg) maxTanDeg = tdeg;
      }
    }
  }
}

// ---- structural checks ----------------------------------------------------
const problems = [];
manifest.lods.forEach((lod, i) => {
  const vids = section(`lod${i}.vids`).data;
  const idx = section(`lod${i}.index`).data;
  if (vids.length !== lod.vertexCount) problems.push(`lod${i}: vids length ${vids.length} != manifest ${lod.vertexCount}`);
  if (idx.length !== lod.triangleCount * 3) problems.push(`lod${i}: index length ${idx.length} != ${lod.triangleCount * 3}`);
  for (let k = 0; k < vids.length; k++) if (vids[k] >= vertexCount) { problems.push(`lod${i}: vid ${vids[k]} out of range`); break; }
  for (let k = 0; k < idx.length; k++) if (idx[k] >= vids.length) { problems.push(`lod${i}: index ${idx[k]} out of range`); break; }
});

const texels = vertexCount * manifest.totalFrames;
if (manifest.texture.width * manifest.texture.height < texels) problems.push('texture too small for texel count');
if (posTex.length < texels * 4) problems.push('posTex section shorter than texel count');

// ---- stride vs ground truth ----------------------------------------------
const strideRows = [];
for (const clip of manifest.clips) {
  if (clip.groundTruthStride == null || clip.groundTruthStride === 0) continue;
  const err = ((clip.stride - clip.groundTruthStride) / clip.groundTruthStride) * 100;
  strideRows.push(`  ${clip.name.padEnd(8)} baked ${clip.stride.toFixed(4)} m  truth ${clip.groundTruthStride.toFixed(4)} m  err ${err >= 0 ? '+' : ''}${err.toFixed(2)}%`);
  if (Math.abs(err) > 5) problems.push(`stride error for '${clip.name}' is ${err.toFixed(2)}% (>5%)`);
}

// fp16 quantisation floor for reference
const quantum = Math.max(...extent) / 2048;

console.log(`verify        ${prefix}`);
console.log(`samples       ${nPos.toLocaleString()} vertex-frames across ${manifest.clips.length} clips`);
console.log(`position      max ${(maxPosErr * 1000).toFixed(3)} mm   mean ${(sumPosErr / nPos * 1000).toFixed(4)} mm   (fp16 quantum ~${(quantum * 1000).toFixed(3)} mm)`);
if (worst) console.log(`  worst at    clip ${worst.clip}, frame ${worst.frame}, vertex ${worst.vertex}`);
console.log(`normal        max ${maxNrmDeg.toFixed(3)} deg   mean ${(sumNrmDeg / nPos).toFixed(4)} deg`);
if (nrmComps === 4) console.log(`tangent       max ${maxTanDeg.toFixed(3)} deg`);
if (strideRows.length) { console.log('stride'); strideRows.forEach((r) => console.log(r)); }

const posFail = maxPosErr > quantum * 3;
const nrmFail = maxNrmDeg > 2.0;
if (posFail) problems.push(`position error ${(maxPosErr * 1000).toFixed(3)} mm exceeds 3x the fp16 quantum`);
if (nrmFail) problems.push(`normal error ${maxNrmDeg.toFixed(2)} deg exceeds 2 deg`);

console.log('');
if (problems.length) {
  console.log('FAILED');
  problems.forEach((p) => console.log('  - ' + p));
  process.exit(1);
}
console.log('OK  bake round-trips within fp16 tolerance');
