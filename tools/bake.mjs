#!/usr/bin/env node
// Vertex Animation Texture baker.
//
//   node tools/bake.mjs --procgen --out public/baked/crowd
//   node tools/bake.mjs assets/character.glb --fps 30 --lods 4 --bones --out public/baked/hero
//
// Produces <out>.json (manifest) + <out>.bin (textures + geometry).

import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { Matrix4 } from 'three';

import { makeProceduralRig } from './lib/procgen.mjs';
import { loadRigFromGLTF } from './lib/rig-gltf.mjs';
import { loadFrameModel } from './lib/frame-model.mjs';
import { skinFrame, computeTangents } from './lib/skin.mjs';
import { measureStride } from './lib/stride.mjs';
import { buildLOD } from './lib/decimate.mjs';
import { packHalfArray } from './lib/half.mjs';
import { octEncode } from './lib/octa.mjs';
import { BinaryPacker, humanBytes } from './lib/pack.mjs';

const MAX_TEXTURE_SIZE = 16384;

function parseArgs(argv) {
  const o = {
    input: null, procgen: false, out: null,
    fps: 30, frames: 0, clips: null,
    lodRatios: null, lods: 3,
    bones: false, tangents: false,
    texWidth: 4096, density: 1.4, skin: 0,
    framesDump: null,
    quiet: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    switch (a) {
      case '--procgen': o.procgen = true; break;
      case '--out': o.out = next(); break;
      case '--fps': o.fps = parseFloat(next()); break;
      case '--frames': o.frames = parseInt(next(), 10); break;
      case '--clips': o.clips = next().split(',').map((s) => s.trim()).filter(Boolean); break;
      case '--lods': o.lods = parseInt(next(), 10); break;
      case '--lod-ratios': o.lodRatios = next().split(',').map(Number); break;
      case '--bones': o.bones = true; break;
      case '--tangents': o.tangents = true; break;
      case '--tex-width': o.texWidth = parseInt(next(), 10); break;
      case '--density': o.density = parseFloat(next()); break;
      case '--skin': o.skin = parseInt(next(), 10); break;
      case '--frames-dump': o.framesDump = next(); break;
      case '--quiet': o.quiet = true; break;
      case '-h': case '--help': printHelp(); process.exit(0); break;
      default:
        if (a.startsWith('-')) throw new Error('unknown flag ' + a);
        o.input = a;
    }
  }
  if (!o.input && !o.procgen && !o.framesDump) o.procgen = true;
  return o;
}

function printHelp() {
  console.log(`vat baker

  node tools/bake.mjs [input.gltf|input.glb] [options]

  --procgen           bake the built-in procedural humanoid (default when no input given)
  --out <prefix>      output path prefix, writes <prefix>.json and <prefix>.bin
  --fps <n>           sampling rate per clip (default 30)
  --frames <n>        fixed frame count per clip, overrides --fps
  --clips a,b,c       only bake these clips
  --lods <n>          number of LOD levels (default 3)
  --lod-ratios a,b,c  explicit vertex ratios, e.g. 1,0.4,0.15,0.05
  --bones             also bake a bone-matrix texture (attachments + cheap memory)
  --tangents          bake per-frame tangents for normal mapping
  --tex-width <n>     animation texture width (default 4096)
  --density <f>       procedural mesh density (default 1.4)
  --skin <n>          which glTF skin to use (default 0)
`);
}

function defaultRatios(n) {
  const table = {
    1: [1],
    2: [1, 0.28],
    3: [1, 0.3, 0.08],
    4: [1, 0.35, 0.12, 0.04],
    5: [1, 0.35, 0.12, 0.045, 0.015],
  };
  return table[Math.max(1, Math.min(5, n))];
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const t0 = performance.now();
  const log = opts.quiet ? () => {} : (...a) => console.log(...a);

  // ---------------------------------------------------------------- load ---
  let model;
  if (opts.framesDump) {
    model = loadFrameModel(path.resolve(opts.framesDump));
    if (opts.bones) throw new Error('--bones needs a skinned rig; a frames dump has no bones');
    if (opts.tangents) throw new Error('--tangents is not supported for frames dumps');
    log(`source        frames dump ${opts.framesDump}`);
  } else if (opts.procgen) {
    model = makeProceduralRig({ density: opts.density });
    log(`source        procedural humanoid (density ${opts.density})`);
  } else {
    model = loadRigFromGLTF(path.resolve(opts.input), { skin: opts.skin });
    log(`source        ${opts.input}`);
  }
  if (!model.clips.length) throw new Error('model has no animation clips');
  if (!model.materialIds) model.materialIds = new Float32Array(model.vertexCount);

  let clipIdx = model.clips.map((c, i) => i);
  if (opts.clips) {
    clipIdx = clipIdx.filter((i) => opts.clips.includes(model.clips[i].name));
    if (!clipIdx.length) throw new Error('no clips matched --clips');
  }

  const out = opts.out || path.join('public', 'baked', model.name.replace(/\s+/g, '_'));
  fs.mkdirSync(path.dirname(path.resolve(out)), { recursive: true });

  const vertexCount = model.vertexCount;
  log(`mesh          ${vertexCount} verts, ${model.indices.length / 3} tris, ${model.boneCount} bones`);

  // -------------------------------------------------------------- layout ---
  const clips = clipIdx.map((i) => {
    const c = model.clips[i];
    const frameCount = opts.frames > 0
      ? opts.frames
      : Math.max(2, Math.round((c.duration || 1) * opts.fps));
    return { index: i, name: c.name, duration: c.duration || 1, loop: c.loop !== false, frameCount, groundTruthStride: c.groundTruthStride };
  });
  let frameStart = 0;
  for (const c of clips) { c.frameStart = frameStart; frameStart += c.frameCount; }
  const totalFrames = frameStart;
  const texelCount = vertexCount * totalFrames;

  let texWidth = Math.min(opts.texWidth, MAX_TEXTURE_SIZE);
  if (texelCount < texWidth) texWidth = Math.max(1, texelCount);
  let texHeight = Math.ceil(texelCount / texWidth);
  while (texHeight > MAX_TEXTURE_SIZE && texWidth < MAX_TEXTURE_SIZE) {
    texWidth = Math.min(MAX_TEXTURE_SIZE, texWidth * 2);
    texHeight = Math.ceil(texelCount / texWidth);
  }
  if (texHeight > MAX_TEXTURE_SIZE) {
    throw new Error(`baked data needs ${texelCount} texels which will not fit in a ${MAX_TEXTURE_SIZE}^2 texture. `
      + 'Reduce --fps / --frames, or split the mesh.');
  }

  // ------------------------------------------------------- pass 1: bounds --
  const skinMats = new Float32Array(model.boneCount * 16);
  const framePos = new Float32Array(vertexCount * 3);
  const frameNrm = new Float32Array(vertexCount * 3);
  const bindTangents = opts.tangents ? computeTangents(model) : null;
  const frameTan = opts.tangents ? new Float32Array(vertexCount * 3) : null;

  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  let maxRadiusXZ = 0;

  const eachFrame = (fn) => {
    for (const c of clips) {
      for (let f = 0; f < c.frameCount; f++) {
        const t = c.loop
          ? (f / c.frameCount) * c.duration
          : (f / Math.max(1, c.frameCount - 1)) * c.duration;
        if (model.sampleFrameDirect) model.sampleFrameDirect(c.index, t, framePos, frameNrm);
        else {
          model.sampleSkinMatrices(c.index, t, skinMats);
          skinFrame(model, skinMats, framePos, frameNrm, bindTangents, frameTan);
        }
        fn(c.frameStart + f, c);
      }
    }
  };

  eachFrame(() => {
    for (let v = 0; v < vertexCount; v++) {
      const x = framePos[v * 3], y = framePos[v * 3 + 1], z = framePos[v * 3 + 2];
      if (x < min[0]) min[0] = x; if (x > max[0]) max[0] = x;
      if (y < min[1]) min[1] = y; if (y > max[1]) max[1] = y;
      if (z < min[2]) min[2] = z; if (z > max[2]) max[2] = z;
      const r = Math.hypot(x, z);
      if (r > maxRadiusXZ) maxRadiusXZ = r;
    }
  });

  const extent = [
    Math.max(max[0] - min[0], 1e-5),
    Math.max(max[1] - min[1], 1e-5),
    Math.max(max[2] - min[2], 1e-5),
  ];
  log(`bounds        min [${min.map((v) => v.toFixed(3)).join(', ')}]  extent [${extent.map((v) => v.toFixed(3)).join(', ')}]`);

  // -------------------------------------------------------- pass 2: bake ---
  const posF = new Float32Array(texelCount * 4);
  const nrmComps = opts.tangents ? 4 : 2;
  const nrmF = new Float32Array(texelCount * nrmComps);
  const oct = new Float32Array(2);

  eachFrame((frameIndex) => {
    const base = frameIndex * vertexCount;
    for (let v = 0; v < vertexCount; v++) {
      const t4 = (base + v) * 4;
      posF[t4] = (framePos[v * 3] - min[0]) / extent[0];
      posF[t4 + 1] = (framePos[v * 3 + 1] - min[1]) / extent[1];
      posF[t4 + 2] = (framePos[v * 3 + 2] - min[2]) / extent[2];
      posF[t4 + 3] = bindTangents ? bindTangents[v * 4 + 3] : 1;

      const no = (base + v) * nrmComps;
      octEncode(frameNrm[v * 3], frameNrm[v * 3 + 1], frameNrm[v * 3 + 2], oct, 0);
      nrmF[no] = oct[0]; nrmF[no + 1] = oct[1];
      if (opts.tangents) {
        octEncode(frameTan[v * 3], frameTan[v * 3 + 1], frameTan[v * 3 + 2], oct, 0);
        nrmF[no + 2] = oct[0]; nrmF[no + 3] = oct[1];
      }
    }
  });

  const posHalf = packHalfArray(posF);
  const nrmHalf = packHalfArray(nrmF);

  // ------------------------------------------------------- bone matrices ---
  let boneTex = null;
  if (opts.bones) {
    const bw = model.boneCount * 3;
    const data = new Float32Array(bw * totalFrames * 4);
    for (const c of clips) {
      for (let f = 0; f < c.frameCount; f++) {
        const t = c.loop ? (f / c.frameCount) * c.duration : (f / Math.max(1, c.frameCount - 1)) * c.duration;
        model.sampleSkinMatrices(c.index, t, skinMats);
        const row = (c.frameStart + f) * bw * 4;
        for (let b = 0; b < model.boneCount; b++) {
          const m = b * 16, o = row + b * 12;
          // three rows of the 3x4 affine matrix (column-major source)
          data[o + 0] = skinMats[m + 0]; data[o + 1] = skinMats[m + 4]; data[o + 2] = skinMats[m + 8]; data[o + 3] = skinMats[m + 12];
          data[o + 4] = skinMats[m + 1]; data[o + 5] = skinMats[m + 5]; data[o + 6] = skinMats[m + 9]; data[o + 7] = skinMats[m + 13];
          data[o + 8] = skinMats[m + 2]; data[o + 9] = skinMats[m + 6]; data[o + 10] = skinMats[m + 10]; data[o + 11] = skinMats[m + 14];
        }
      }
    }
    boneTex = { width: bw, height: totalFrames, data, texelsPerBone: 3, format: 'RGBA32F' };
  }

  // -------------------------------------------------------------- stride ---
  for (const c of clips) {
    const src = model.clips[c.index];
    const s = src.strideOverride != null
      ? { strideLength: src.strideOverride, groundSpeed: src.groundSpeedOverride || 0,
          method: 'captured', confidence: 1, facing: 1, feet: [] }
      : measureStride(model, c.index, Math.max(c.frameCount, 24));
    c.stride = s.strideLength;
    c.groundSpeed = s.groundSpeed;
    c.strideMethod = s.method;
    c.strideConfidence = s.confidence;
    c.facing = s.facing;
    c.feet = s.feet;
    const gt = c.groundTruthStride;
    const err = gt ? ((s.strideLength - gt) / gt) * 100 : null;
    log(`stride ${c.name.padEnd(8)} ${s.strideLength.toFixed(4)} m/cycle  ${s.groundSpeed.toFixed(3)} m/s  `
      + `[${s.method}, conf ${(s.confidence * 100).toFixed(0)}%${err !== null ? `, err ${err.toFixed(2)}%` : ''}]`);
  }

  // ----------------------------------------------------------------- LODs ---
  const ratios = opts.lodRatios || defaultRatios(opts.lods);
  const lods = ratios.map((r) => buildLOD(model, r));
  lods.forEach((l, i) => log(`lod${i}          ${l.vids.length} verts, ${l.indices.length / 3} tris (${(l.ratio * 100).toFixed(1)}%)`));

  // rest-pose world position of each joint, so attachment sockets can be
  // authored in world units without re-reading the source rig
  const boneRest = [];
  {
    const inv = new Matrix4();
    for (let b = 0; b < model.boneCount && model.inverseBind.length >= (b + 1) * 16; b++) {
      inv.fromArray(model.inverseBind, b * 16).invert();
      boneRest.push([inv.elements[12], inv.elements[13], inv.elements[14]]);
    }
  }

  // ---------------------------------------------------------------- write ---
  const packer = new BinaryPacker();
  packer.add('posTex', posHalf, { type: 'f16', components: 4 });
  packer.add('nrmTex', nrmHalf, { type: 'f16', components: nrmComps });
  packer.add('bindPosition', model.positions, { components: 3 });
  packer.add('bindNormal', model.normals, { components: 3 });
  packer.add('uv', model.uvs, { components: 2 });
  packer.add('color', model.colors, { components: 3 });
  packer.add('materialId', model.materialIds, { components: 1 });
  if (opts.bones) {
    packer.add('boneTex', boneTex.data, { components: 4 });
    packer.add('skinIndex', model.skinIndex, { components: 4 });
    packer.add('skinWeight', model.skinWeight, { components: 4 });
  }
  lods.forEach((l, i) => {
    packer.add(`lod${i}.vids`, l.vids, { components: 1 });
    packer.add(`lod${i}.index`, l.indices, { components: 1 });
  });

  const bin = packer.build();
  const manifest = {
    version: 1,
    generator: 'instance-baking/vat-bake',
    name: model.name,
    source: model.source,
    vertexCount,
    triangleCount: model.indices.length / 3,
    boneCount: model.boneCount,
    boneNames: model.boneNames,
    boneRest: boneRest,
    sockets: model.sockets || null,
    bakeFps: opts.fps,
    bakeOptions: {
      procgen: opts.procgen, density: opts.density, fps: opts.fps, frames: opts.frames,
      clips: opts.clips, lodRatios: ratios, bones: opts.bones, tangents: opts.tangents,
      texWidth: opts.texWidth, skin: opts.skin, input: opts.input,
    },
    totalFrames,
    texture: {
      width: texWidth,
      height: texHeight,
      posFormat: 'RGBA16F',
      nrmFormat: opts.tangents ? 'RGBA16F' : 'RG16F',
      hasTangents: !!opts.tangents,
    },
    bounds: { min, extent, maxRadiusXZ, height: max[1] },
    vertexColorMode: !!model.vertexColorMode,
    clips: clips.map((c) => ({
      name: c.name,
      frameStart: c.frameStart,
      frameCount: c.frameCount,
      duration: c.duration,
      loop: c.loop,
      fps: c.frameCount / c.duration,
      stride: c.stride,
      groundSpeed: c.groundSpeed,
      strideMethod: c.strideMethod,
      strideConfidence: c.strideConfidence,
      facing: c.facing,
      feet: c.feet,
      groundTruthStride: c.groundTruthStride != null ? c.groundTruthStride : undefined,
    })),
    lods: lods.map((l, i) => ({
      level: i,
      vertexCount: l.vids.length,
      triangleCount: l.indices.length / 3,
      ratio: l.ratio,
    })),
    bone: boneTex ? { width: boneTex.width, height: boneTex.height, texelsPerBone: 3, format: boneTex.format } : null,
    sections: packer.sections,
    binary: path.basename(out) + '.bin',
  };

  fs.writeFileSync(out + '.json', JSON.stringify(manifest, null, 2));
  fs.writeFileSync(out + '.bin', bin);

  const posBytes = posHalf.byteLength;
  const nrmBytes = nrmHalf.byteLength;
  log('');
  log(`texture       ${texWidth} x ${texHeight}  (${totalFrames} frames x ${vertexCount} verts = ${texelCount} texels)`);
  log(`  positions   ${humanBytes(posBytes)}  RGBA16F`);
  log(`  normals     ${humanBytes(nrmBytes)}  ${manifest.texture.nrmFormat}`);
  if (boneTex) log(`  bones       ${humanBytes(boneTex.data.byteLength)}  ${boneTex.format} (${boneTex.width} x ${boneTex.height})`);
  log(`  per vertex  ${((posBytes + nrmBytes) / texelCount).toFixed(1)} bytes/vertex/frame`);
  log(`total         ${humanBytes(bin.byteLength)} -> ${out}.bin`);
  log(`manifest      ${out}.json`);
  log(`elapsed       ${(performance.now() - t0).toFixed(0)} ms`);
}

main().catch((err) => {
  console.error('bake failed:', err.message);
  if (process.env.DEBUG) console.error(err.stack);
  process.exit(1);
});
