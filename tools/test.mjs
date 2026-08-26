#!/usr/bin/env node
// Offline test suite: everything that can be checked without a GPU.
//
//   node tools/test.mjs
//
// Covers the numeric core (half/oct round-trips, stride extraction against a
// known ground truth), the container layout, the LOD builder, and -- the useful
// one -- a cross-check that baking the procedural rig directly and baking the
// .glb exported from it land on the same texels. That last test is what makes
// the glTF reader trustworthy.

import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { toHalf, fromHalf, packHalfArray, unpackHalfArray, HAS_NATIVE_FLOAT16 } from './lib/half.mjs';
import { octEncode, octDecode } from './lib/octa.mjs';
import { makeProceduralRig } from './lib/procgen.mjs';
import { buildLOD } from './lib/decimate.mjs';
import { measureStride } from './lib/stride.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const tmp = path.join(root, 'tools', '_out');
fs.mkdirSync(tmp, { recursive: true });

let passed = 0;
const failures = [];

function test(name, fn) {
  try {
    const detail = fn();
    passed++;
    console.log(`  ok    ${name}${detail ? '   ' + detail : ''}`);
  } catch (err) {
    failures.push({ name, message: err.message });
    console.log(`  FAIL  ${name}\n          ${err.message}`);
  }
}

function assert(cond, msg) { if (!cond) throw new Error(msg); }
function assertClose(a, b, tol, msg) {
  if (!(Math.abs(a - b) <= tol)) throw new Error(`${msg}: ${a} vs ${b} (tol ${tol})`);
}

function node(args) {
  return execFileSync(process.execPath, args, { cwd: root, encoding: 'utf8' });
}

function loadBake(prefix) {
  const manifest = JSON.parse(fs.readFileSync(prefix + '.json', 'utf8'));
  const b = fs.readFileSync(path.join(path.dirname(prefix), manifest.binary));
  const bin = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  const section = (name) => {
    const s = manifest.sections[name];
    if (!s) return null;
    const ctor = { f32: Float32Array, u32: Uint32Array, u16: Uint16Array, f16: Uint16Array, u8: Uint8Array }[s.type];
    return new ctor(bin, s.offset, s.byteLength / ctor.BYTES_PER_ELEMENT);
  };
  return { manifest, bin, section };
}

console.log('\nnumeric core');

test('float16 round-trip in the normal range', () => {
  // binary16 has 10 explicit mantissa bits, so round-to-nearest caps the
  // relative error of a NORMAL value at 2^-11. Subnormals (|v| < 6.104e-5) are
  // checked separately below because their relative error is unbounded by design.
  const SMALLEST_NORMAL = 6.103515625e-5;
  let maxRel = 0;
  for (let i = 0; i < 50000; i++) {
    const v = (Math.random() < 0.5 ? -1 : 1) * Math.pow(10, Math.random() * 6 - 4);
    if (Math.abs(v) < SMALLEST_NORMAL) continue;
    const rel = Math.abs(fromHalf(toHalf(v)) - v) / Math.abs(v);
    if (rel > maxRel) maxRel = rel;
  }
  assert(maxRel <= Math.pow(2, -11) * 1.001, `max relative error ${maxRel} exceeds the 2^-11 bound`);
  return `max rel err ${maxRel.toExponential(2)} (bound ${Math.pow(2, -11).toExponential(2)})`
    + (HAS_NATIVE_FLOAT16 ? ', native Float16Array present' : '');
});

test('float16 handles subnormals, zero and overflow', () => {
  assert(fromHalf(toHalf(0)) === 0, 'zero did not survive');
  assert(Object.is(fromHalf(toHalf(-0)), -0) || fromHalf(toHalf(-0)) === 0, 'negative zero broke');
  for (const v of [1e-6, 3e-5, 5.96e-8, -2.4e-5]) {
    const back = fromHalf(toHalf(v));
    assert(Number.isFinite(back), `subnormal ${v} became ${back}`);
    assert(Math.abs(back - v) <= 6e-8 + Math.abs(v) * 0.5, `subnormal ${v} -> ${back} is way off`);
  }
  assert(fromHalf(toHalf(1e6)) === Infinity, 'overflow should saturate to Infinity');
  assert(fromHalf(toHalf(-1e6)) === -Infinity, 'negative overflow should saturate');
  assert(Math.abs(fromHalf(toHalf(65504)) - 65504) < 1, 'largest finite half is wrong');
  return 'subnormal / zero / overflow all sane';
});

test('float16 array and scalar paths agree to 1 ULP', () => {
  // They are not required to be bit-identical: the scalar fallback rounds
  // double -> f32 -> f16, so near-midpoint values can double-round one step up
  // where the native single-rounding path goes down. What must hold is that
  // neither ever lands more than one representable step apart, and that both
  // stay inside the round-to-nearest error bound.
  const N = 200000;
  const src = new Float32Array(N);
  for (let i = 0; i < N; i++) src[i] = Math.random() * 2 - 1;
  const back = unpackHalfArray(packHalfArray(src));
  let disagreements = 0;
  let worstUlps = 0;
  let worstScalarRel = 0;
  for (let i = 0; i < N; i++) {
    const scalar = fromHalf(toHalf(src[i]));
    if (scalar !== back[i]) {
      disagreements++;
      // one ULP at this exponent
      const ulp = Math.pow(2, Math.floor(Math.log2(Math.abs(src[i]) || 1e-8)) - 10);
      worstUlps = Math.max(worstUlps, Math.abs(scalar - back[i]) / ulp);
    }
    const v = Math.abs(src[i]);
    if (v > 6.103515625e-5) worstScalarRel = Math.max(worstScalarRel, Math.abs(scalar - src[i]) / v);
  }
  assert(worstUlps <= 1.001, `paths differ by ${worstUlps.toFixed(2)} ULP`);
  assert(worstScalarRel <= Math.pow(2, -11) * 1.001,
    `scalar fallback exceeded the round-to-nearest bound: ${worstScalarRel}`);
  return `${disagreements}/${N} differ, all by <=1 ULP `
    + `(fallback worst rel err ${worstScalarRel.toExponential(2)})`;
});

test('octahedral normal round-trip', () => {
  const e = new Float32Array(2);
  const d = new Float32Array(3);
  let maxDeg = 0;
  for (let i = 0; i < 20000; i++) {
    let x = Math.random() * 2 - 1, y = Math.random() * 2 - 1, z = Math.random() * 2 - 1;
    const l = Math.hypot(x, y, z) || 1;
    x /= l; y /= l; z /= l;
    octEncode(x, y, z, e, 0);
    // through fp16, the way it is actually stored
    octDecode(fromHalf(toHalf(e[0])), fromHalf(toHalf(e[1])), d, 0);
    const dot = Math.min(1, Math.max(-1, d[0] * x + d[1] * y + d[2] * z));
    maxDeg = Math.max(maxDeg, Math.acos(dot) * 180 / Math.PI);
  }
  assert(maxDeg < 0.5, `max angular error ${maxDeg} deg`);
  return `max ${maxDeg.toFixed(3)} deg through fp16`;
});

console.log('\nrig + stride');

const rig = makeProceduralRig({ density: 1.4 });

test('procedural rig is well formed', () => {
  assert(rig.vertexCount > 500, 'suspiciously small mesh');
  assert(rig.indices.length % 3 === 0, 'index buffer is not triangles');
  let maxIdx = 0;
  for (let i = 0; i < rig.indices.length; i++) maxIdx = Math.max(maxIdx, rig.indices[i]);
  assert(maxIdx < rig.vertexCount, `index ${maxIdx} out of range`);
  for (let v = 0; v < rig.vertexCount; v++) {
    let sum = 0;
    for (let k = 0; k < 4; k++) {
      sum += rig.skinWeight[v * 4 + k];
      assert(rig.skinIndex[v * 4 + k] < rig.boneCount, 'skin index out of range');
    }
    assertClose(sum, 1, 1e-4, `vertex ${v} weights do not sum to 1`);
  }
  return `${rig.vertexCount} verts, ${rig.indices.length / 3} tris, ${rig.boneCount} bones`;
});

test('planted foot never slides in the walk clip', () => {
  // the IK gait plants the ankle exactly, so the contact foot's backwards
  // velocity should be constant across the whole stance phase
  const ci = rig.clips.findIndex((c) => c.name === 'walk');
  const frames = 96;
  const world = new Float32Array(rig.boneCount * 16);
  const footL = rig.boneNames.indexOf('footL');
  const zs = [];
  const ys = [];
  const xs = [];
  for (let f = 0; f < frames; f++) {
    rig.sampleJointWorld(ci, (f / frames) * rig.clips[ci].duration, world);
    xs.push(world[footL * 16 + 12]);
    ys.push(world[footL * 16 + 13]);
    zs.push(world[footL * 16 + 14]);
  }
  const minY = Math.min(...ys);
  const deltas = [];
  const lateral = [];
  for (let f = 1; f < frames; f++) {
    if (ys[f] < minY + 1e-3 && ys[f - 1] < minY + 1e-3) {
      deltas.push(zs[f] - zs[f - 1]);
      lateral.push(Math.abs(xs[f] - xs[f - 1]));
    }
  }
  const maxLateral = Math.max(...lateral);
  assert(maxLateral < 1e-4, `planted foot drifts sideways by up to ${(maxLateral * 1000).toFixed(3)} mm/frame`);
  assert(deltas.length > 10, `only ${deltas.length} planted frames found`);
  const mean = deltas.reduce((a, b) => a + b, 0) / deltas.length;
  const spread = Math.max(...deltas) - Math.min(...deltas);
  // also check the foot does not wander sideways while planted
  assert(Math.abs(spread / mean) < 5e-3, `planted foot velocity varies by ${(spread / mean * 100).toFixed(3)}%`);
  return `${deltas.length} planted frames, velocity spread ${(spread / mean * 100).toExponential(1)}%`;
});

for (const name of ['walk', 'jog', 'run']) {
  test(`stride extraction: ${name}`, () => {
    const ci = rig.clips.findIndex((c) => c.name === name);
    const truth = rig.clips[ci].groundTruthStride;
    const s = measureStride(rig, ci, 96);
    const err = ((s.strideLength - truth) / truth) * 100;
    assert(Math.abs(err) < 5, `stride error ${err.toFixed(2)}% exceeds 5%`);
    assert(s.confidence > 0.7, `low confidence ${s.confidence.toFixed(2)}`);
    assert(s.feet.length === 2, `expected 2 contact bones, got ${s.feet.join()}`);
    return `${s.strideLength.toFixed(4)}m vs truth ${truth.toFixed(4)}m (${err >= 0 ? '+' : ''}${err.toFixed(2)}%)`;
  });
}

test('stride extraction reports zero for in-place clips', () => {
  const ci = rig.clips.findIndex((c) => c.name === 'idle');
  const s = measureStride(rig, ci, 64);
  assert(s.strideLength < 0.05, `idle clip reported ${s.strideLength.toFixed(4)}m of travel`);
  return `${s.strideLength.toFixed(5)}m`;
});

console.log('\nlod builder');

test('LODs keep original vertex ids and valid topology', () => {
  const details = [];
  for (const ratio of [1, 0.5, 0.2, 0.07]) {
    const lod = buildLOD(rig, ratio);
    assert(lod.indices.length % 3 === 0, 'not triangles');
    assert(lod.vids.length > 0 && lod.indices.length > 0, `empty LOD at ratio ${ratio}`);
    for (let i = 0; i < lod.vids.length; i++) {
      assert(lod.vids[i] < rig.vertexCount, `vid ${lod.vids[i]} outside source mesh`);
    }
    for (let i = 0; i < lod.indices.length; i++) {
      assert(lod.indices[i] < lod.vids.length, 'index outside LOD vertex list');
    }
    for (let i = 0; i < lod.indices.length; i += 3) {
      const [a, b, c] = [lod.indices[i], lod.indices[i + 1], lod.indices[i + 2]];
      assert(a !== b && b !== c && a !== c, 'degenerate triangle survived');
    }
    assert(lod.vids.length <= rig.vertexCount, 'LOD grew the mesh');
    details.push(`${(ratio * 100).toFixed(0)}%->${lod.vids.length}v`);
  }
  return details.join(' ');
});

test('LOD vertex counts decrease monotonically', () => {
  const counts = [1, 0.5, 0.2, 0.07].map((r) => buildLOD(rig, r).vids.length);
  for (let i = 1; i < counts.length; i++) {
    assert(counts[i] < counts[i - 1], `LOD ${i} (${counts[i]}) is not smaller than LOD ${i - 1} (${counts[i - 1]})`);
  }
  return counts.join(' > ');
});

console.log('\nbake pipeline');

const procPrefix = path.join(tmp, 'proc');
const glbPath = path.join(tmp, 'humanoid.glb');
const glbPrefix = path.join(tmp, 'fromglb');

test('bake (procgen) runs', () => {
  node(['tools/bake.mjs', '--procgen', '--frames', '48', '--lods', '4', '--bones', '--tangents',
    '--out', procPrefix, '--quiet']);
  assert(fs.existsSync(procPrefix + '.json'), 'no manifest written');
  assert(fs.existsSync(procPrefix + '.bin'), 'no binary written');
  const size = fs.statSync(procPrefix + '.bin').size;
  return `${(size / 1048576).toFixed(2)} MB`;
});

test('verify (procgen) passes', () => {
  const out = node(['tools/verify.mjs', procPrefix]);
  assert(/^OK/m.test(out), 'verify did not report OK\n' + out);
  const m = out.match(/position\s+max ([\d.]+) mm/);
  return m ? `max position error ${m[1]} mm` : '';
});

test('container sections are aligned and in bounds', () => {
  const { manifest, bin } = loadBake(procPrefix);
  for (const [name, s] of Object.entries(manifest.sections)) {
    assert(s.offset % 4 === 0, `section ${name} is not 4-byte aligned`);
    assert(s.offset + s.byteLength <= bin.byteLength, `section ${name} runs past the end of the binary`);
  }
  const texels = manifest.vertexCount * manifest.totalFrames;
  assert(manifest.texture.width * manifest.texture.height >= texels,
    'texture is too small for the baked texel count');
  assert(manifest.texture.width <= 16384 && manifest.texture.height <= 16384, 'texture exceeds the 16384 limit');
  return `${Object.keys(manifest.sections).length} sections, ${manifest.texture.width}x${manifest.texture.height}`;
});

test('clip frame ranges tile the texture without gaps', () => {
  const { manifest } = loadBake(procPrefix);
  let expect = 0;
  for (const c of manifest.clips) {
    assert(c.frameStart === expect, `clip ${c.name} starts at ${c.frameStart}, expected ${expect}`);
    expect += c.frameCount;
  }
  assert(expect === manifest.totalFrames, `frames sum to ${expect}, manifest says ${manifest.totalFrames}`);
  return `${manifest.clips.length} clips, ${manifest.totalFrames} frames`;
});

test('tangent bake produces a 4-channel normal map', () => {
  const { manifest } = loadBake(procPrefix);
  assert(manifest.texture.hasTangents, 'hasTangents not set');
  assert(manifest.sections.nrmTex.components === 4, 'normal map should be RGBA when tangents are baked');
  return manifest.texture.nrmFormat;
});

test('glTF export runs', () => {
  node(['tools/export-gltf.mjs', '--frames', '48', '--out', glbPath, '--quiet']);
  const size = fs.statSync(glbPath).size;
  assert(size > 10000, 'glb is suspiciously small');
  return `${(size / 1024).toFixed(1)} KB`;
});

test('bake (glb) runs', () => {
  node(['tools/bake.mjs', glbPath, '--frames', '48', '--lods', '4', '--bones', '--tangents',
    '--out', glbPrefix, '--quiet']);
  assert(fs.existsSync(glbPrefix + '.bin'), 'no binary written');
  return '';
});

test('glTF reader reproduces the procedural bake', () => {
  const a = loadBake(procPrefix);
  const b = loadBake(glbPrefix);
  assert(a.manifest.vertexCount === b.manifest.vertexCount, 'vertex counts differ');
  assert(a.manifest.totalFrames === b.manifest.totalFrames, 'frame counts differ');
  for (let c = 0; c < 3; c++) {
    assertClose(a.manifest.bounds.min[c], b.manifest.bounds.min[c], 1e-4, `bounds.min[${c}]`);
    assertClose(a.manifest.bounds.extent[c], b.manifest.bounds.extent[c], 1e-4, `bounds.extent[${c}]`);
  }

  const pa = unpackHalfArray(a.section('posTex'));
  const pb = unpackHalfArray(b.section('posTex'));
  assert(pa.length === pb.length, 'position textures differ in size');
  const ext = a.manifest.bounds.extent;
  // Both bakes quantise to fp16, so the honest bound is one storage ULP: if the
  // underlying float differs in the last bit it can land either side of a
  // rounding boundary. What must hold is that this is rare and never worse.
  const quantum = Math.max(...ext) / 2048;
  let maxErr = 0, sumErr = 0, differing = 0;
  for (let i = 0; i < pa.length; i += 4) {
    let e = 0;
    for (let c = 0; c < 3; c++) e = Math.max(e, Math.abs(pa[i + c] - pb[i + c]) * ext[c]);
    if (e > 0) differing++;
    sumErr += e;
    if (e > maxErr) maxErr = e;
  }
  const texels = pa.length / 4;
  const meanErr = sumErr / texels;
  const frac = differing / texels;
  assert(maxErr <= quantum * 1.05, `worst disagreement ${(maxErr * 1000).toFixed(4)} mm exceeds one fp16 ULP (${(quantum * 1000).toFixed(4)} mm)`);
  assert(meanErr < quantum * 0.05, `mean disagreement ${(meanErr * 1000).toFixed(5)} mm is too high`);
  assert(frac < 0.05, `${(frac * 100).toFixed(2)}% of texels differ; expected under 5%`);
  return `${(frac * 100).toFixed(2)}% of ${texels.toLocaleString()} texels differ by <=1 ULP `
    + `(max ${(maxErr * 1e6).toFixed(1)} um, mean ${(meanErr * 1e6).toFixed(2)} um)`;
});

test('glb bake recovers the same stride lengths', () => {
  const a = loadBake(procPrefix).manifest;
  const b = loadBake(glbPrefix).manifest;
  const details = [];
  for (let i = 0; i < a.clips.length; i++) {
    assert(a.clips[i].name === b.clips[i].name, 'clip order differs');
    assertClose(a.clips[i].stride, b.clips[i].stride, 1e-3, `stride for ${a.clips[i].name}`);
    if (a.clips[i].stride > 0.1) details.push(`${a.clips[i].name} ${a.clips[i].stride.toFixed(3)}m`);
  }
  return details.join('  ');
});

test('bone texture layout matches the manifest', () => {
  const { manifest, section } = loadBake(procPrefix);
  assert(manifest.bone, 'no bone texture baked');
  const data = section('boneTex');
  assert(manifest.bone.width === manifest.boneCount * 3, 'bone texture width should be boneCount * 3');
  assert(manifest.bone.height === manifest.totalFrames, 'bone texture height should be totalFrames');
  assert(data.length === manifest.bone.width * manifest.bone.height * 4, 'bone texture length mismatch');
  // frame 0 of the idle clip: the bottom row of every matrix must be sane
  for (let b = 0; b < manifest.boneCount; b++) {
    const o = b * 12;
    const det = Math.abs(data[o] * (data[o + 5] * data[o + 10] - data[o + 6] * data[o + 9]));
    assert(Number.isFinite(det), `bone ${b} matrix has non-finite values`);
  }
  return `${manifest.bone.width}x${manifest.bone.height} ${manifest.bone.format} (${(data.byteLength / 1024).toFixed(0)} KB)`;
});

test('bone rest positions invert the bind matrices', () => {
  const { manifest } = loadBake(procPrefix);
  assert(Array.isArray(manifest.boneRest) && manifest.boneRest.length === manifest.boneCount,
    'boneRest missing or wrong length');
  const head = manifest.sockets.head;
  const y = manifest.boneRest[head][1];
  assert(y > 1.0 && y < 2.2, `head socket at y=${y} looks wrong`);
  return `head socket at y=${y.toFixed(3)}`;
});

test('texture width auto-grows when a bake will not fit', () => {
  // 16384-wide is the cap; ask for a narrow texture and check it widened
  const prefix = path.join(tmp, 'narrow');
  node(['tools/bake.mjs', '--procgen', '--frames', '48', '--lods', '1',
    '--tex-width', '64', '--out', prefix, '--quiet']);
  const { manifest } = loadBake(prefix);
  const texels = manifest.vertexCount * manifest.totalFrames;
  assert(manifest.texture.height <= 16384, 'height exceeded the limit');
  assert(manifest.texture.width * manifest.texture.height >= texels, 'texture too small');
  return `${manifest.texture.width}x${manifest.texture.height}`;
});

console.log('');
if (failures.length) {
  console.log(`FAILED  ${failures.length} of ${passed + failures.length} checks`);
  for (const f of failures) console.log(`  - ${f.name}: ${f.message}`);
  process.exit(1);
}
console.log(`OK  ${passed} checks passed`);
