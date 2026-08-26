#!/usr/bin/env node
// Writes the procedural humanoid out as a real rigged, animated .glb.
//
// Two reasons this exists:
//  1. it gives you a sample asset to open in Blender / three's editor;
//  2. it turns the glTF reader into tested code -- tools/test.mjs bakes the
//     procgen rig and the exported glb with matching sample times and asserts
//     the two bakes agree, which exercises accessors, skins, node hierarchies
//     and animation samplers end to end.
//
//   node tools/export-gltf.mjs --frames 48 --out assets/humanoid.glb

import fs from 'node:fs';
import path from 'node:path';
import { makeProceduralRig } from './lib/procgen.mjs';

function parseArgs(argv) {
  const o = { frames: 48, out: 'assets/humanoid.glb', density: 1.4, quiet: false };
  for (let i = 0; i < argv.length; i++) {
    const next = () => argv[++i];
    switch (argv[i]) {
      case '--frames': o.frames = parseInt(next(), 10); break;
      case '--out': o.out = next(); break;
      case '--density': o.density = parseFloat(next()); break;
      case '--quiet': o.quiet = true; break;
      default: break;
    }
  }
  return o;
}

class BufferBuilder {
  constructor() { this.views = []; this.chunks = []; this.length = 0; }

  push(array, target) {
    const pad = (4 - (this.length % 4)) % 4;
    if (pad) { this.chunks.push(new Uint8Array(pad)); this.length += pad; }
    const bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
    const view = { buffer: 0, byteOffset: this.length, byteLength: bytes.byteLength };
    if (target) view.target = target;
    this.views.push(view);
    this.chunks.push(bytes);
    this.length += bytes.byteLength;
    return this.views.length - 1;
  }

  build() {
    const pad = (4 - (this.length % 4)) % 4;
    if (pad) { this.chunks.push(new Uint8Array(pad)); this.length += pad; }
    const out = new Uint8Array(this.length);
    let o = 0;
    for (const c of this.chunks) { out.set(c, o); o += c.byteLength; }
    return out;
  }
}

const COMPONENT = { f32: 5126, u32: 5125, u16: 5123 };

function main() {
  const opts = parseArgs(process.argv.slice(2));
  const log = opts.quiet ? () => {} : (...a) => console.log(...a);
  const rig = makeProceduralRig({ density: opts.density });

  const buf = new BufferBuilder();
  const accessors = [];

  const addAccessor = (array, type, componentType, target, extra = {}) => {
    const comps = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 }[type];
    const view = buf.push(array, target);
    accessors.push({
      bufferView: view,
      componentType,
      count: array.length / comps,
      type,
      ...extra,
    });
    return accessors.length - 1;
  };

  const minMax = (arr, comps) => {
    const min = new Array(comps).fill(Infinity);
    const max = new Array(comps).fill(-Infinity);
    for (let i = 0; i < arr.length; i += comps) {
      for (let c = 0; c < comps; c++) {
        const v = arr[i + c];
        if (v < min[c]) min[c] = v;
        if (v > max[c]) max[c] = v;
      }
    }
    return { min, max };
  };

  // ---- geometry --------------------------------------------------------
  const posAcc = addAccessor(rig.positions, 'VEC3', COMPONENT.f32, 34962, minMax(rig.positions, 3));
  const nrmAcc = addAccessor(rig.normals, 'VEC3', COMPONENT.f32, 34962);
  const uvAcc = addAccessor(rig.uvs, 'VEC2', COMPONENT.f32, 34962);
  const jointAcc = addAccessor(rig.skinIndex, 'VEC4', COMPONENT.u16, 34962);
  const weightAcc = addAccessor(rig.skinWeight, 'VEC4', COMPONENT.f32, 34962);
  const idxAcc = addAccessor(rig.indices, 'SCALAR', COMPONENT.u32, 34963);
  const ibmAcc = addAccessor(rig.inverseBind, 'MAT4', COMPONENT.f32);

  // ---- nodes -----------------------------------------------------------
  // node 0 = skinned mesh, nodes 1..boneCount = joints
  const nodes = [{ name: 'body', mesh: 0, skin: 0 }];
  const jointNode = (b) => b + 1;
  for (let b = 0; b < rig.boneCount; b++) {
    nodes.push({
      name: rig.boneNames[b],
      translation: [rig.restLocal[b * 3], rig.restLocal[b * 3 + 1], rig.restLocal[b * 3 + 2]],
      rotation: [0, 0, 0, 1],
      scale: [1, 1, 1],
    });
  }
  for (let b = 0; b < rig.boneCount; b++) {
    const p = rig.restParent[b];
    if (p < 0) continue;
    const parentNode = nodes[jointNode(p)];
    (parentNode.children || (parentNode.children = [])).push(jointNode(b));
  }

  // ---- animations ------------------------------------------------------
  const animations = [];
  const outT = new Float32Array(rig.boneCount * 3);
  const outR = new Float32Array(rig.boneCount * 4);
  const outS = new Float32Array(rig.boneCount * 3);

  for (let ci = 0; ci < rig.clips.length; ci++) {
    const clip = rig.clips[ci];
    const n = opts.frames;
    const times = new Float32Array(n + 1);
    const rot = [];
    const trans = [];
    for (let b = 0; b < rig.boneCount; b++) {
      rot.push(new Float32Array((n + 1) * 4));
      trans.push(new Float32Array((n + 1) * 3));
    }
    for (let f = 0; f <= n; f++) {
      // the extra key at f === n repeats frame 0 so the clip loops seamlessly
      const t = (Math.min(f, n - 1) / n) * clip.duration;
      times[f] = (f / n) * clip.duration;
      rig.sampleLocalTRS(ci, f === n ? 0 : t, outT, outR, outS);
      for (let b = 0; b < rig.boneCount; b++) {
        for (let c = 0; c < 4; c++) rot[b][f * 4 + c] = outR[b * 4 + c];
        for (let c = 0; c < 3; c++) trans[b][f * 3 + c] = outT[b * 3 + c];
      }
    }

    const timeAcc = addAccessor(times, 'SCALAR', COMPONENT.f32, undefined, {
      min: [times[0]], max: [times[times.length - 1]],
    });
    const samplers = [];
    const channels = [];
    for (let b = 0; b < rig.boneCount; b++) {
      samplers.push({ input: timeAcc, output: addAccessor(rot[b], 'VEC4', COMPONENT.f32), interpolation: 'LINEAR' });
      channels.push({ sampler: samplers.length - 1, target: { node: jointNode(b), path: 'rotation' } });
      // only bones that actually translate get a translation track
      let moves = false;
      for (let f = 1; f <= n && !moves; f++) {
        for (let c = 0; c < 3; c++) if (Math.abs(trans[b][f * 3 + c] - trans[b][c]) > 1e-7) moves = true;
      }
      if (moves) {
        samplers.push({ input: timeAcc, output: addAccessor(trans[b], 'VEC3', COMPONENT.f32), interpolation: 'LINEAR' });
        channels.push({ sampler: samplers.length - 1, target: { node: jointNode(b), path: 'translation' } });
      }
    }
    // a deliberate single-key scale track: exercises the reader's n === 1 path
    const oneTime = new Float32Array([0]);
    const oneScale = new Float32Array([1, 1, 1]);
    samplers.push({
      input: addAccessor(oneTime, 'SCALAR', COMPONENT.f32, undefined, { min: [0], max: [0] }),
      output: addAccessor(oneScale, 'VEC3', COMPONENT.f32),
      interpolation: 'LINEAR',
    });
    channels.push({ sampler: samplers.length - 1, target: { node: jointNode(0), path: 'scale' } });

    animations.push({ name: clip.name, samplers, channels });
  }

  const bin = buf.build();

  const gltf = {
    asset: { version: '2.0', generator: 'instance-baking/export-gltf' },
    scene: 0,
    scenes: [{ nodes: [0, jointNode(0)] }],
    nodes,
    meshes: [{
      name: 'humanoid',
      primitives: [{
        attributes: {
          POSITION: posAcc, NORMAL: nrmAcc, TEXCOORD_0: uvAcc,
          JOINTS_0: jointAcc, WEIGHTS_0: weightAcc,
        },
        indices: idxAcc,
        mode: 4,
        material: 0,
      }],
    }],
    materials: [{ name: 'body', pbrMetallicRoughness: { baseColorFactor: [0.82, 0.78, 0.74, 1], metallicFactor: 0, roughnessFactor: 0.9 } }],
    skins: [{ name: 'rig', joints: Array.from({ length: rig.boneCount }, (_, b) => jointNode(b)), inverseBindMatrices: ibmAcc, skeleton: jointNode(0) }],
    animations,
    accessors,
    bufferViews: buf.views,
    buffers: [{ byteLength: bin.byteLength }],
  };

  const jsonBytes = Buffer.from(JSON.stringify(gltf), 'utf8');
  const jsonPad = (4 - (jsonBytes.length % 4)) % 4;
  const jsonChunk = Buffer.concat([jsonBytes, Buffer.alloc(jsonPad, 0x20)]);
  const binChunk = Buffer.from(bin.buffer, bin.byteOffset, bin.byteLength);

  const total = 12 + 8 + jsonChunk.length + 8 + binChunk.length;
  const glb = Buffer.alloc(total);
  let o = 0;
  glb.writeUInt32LE(0x46546c67, o); o += 4;   // 'glTF'
  glb.writeUInt32LE(2, o); o += 4;
  glb.writeUInt32LE(total, o); o += 4;
  glb.writeUInt32LE(jsonChunk.length, o); o += 4;
  glb.writeUInt32LE(0x4e4f534a, o); o += 4;   // 'JSON'
  jsonChunk.copy(glb, o); o += jsonChunk.length;
  glb.writeUInt32LE(binChunk.length, o); o += 4;
  glb.writeUInt32LE(0x004e4942, o); o += 4;   // 'BIN'
  binChunk.copy(glb, o);

  fs.mkdirSync(path.dirname(path.resolve(opts.out)), { recursive: true });
  fs.writeFileSync(opts.out, glb);

  log(`wrote ${opts.out}`);
  log(`  ${rig.vertexCount} verts, ${rig.indices.length / 3} tris, ${rig.boneCount} joints`);
  log(`  ${animations.length} clips @ ${opts.frames} keys, ${accessors.length} accessors`);
  log(`  ${(glb.length / 1024).toFixed(1)} KB`);
}

main();
