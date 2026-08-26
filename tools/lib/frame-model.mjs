// Adapts a frames dump (tools/bake-goober.mjs output) to the model interface
// tools/bake.mjs consumes. Instead of skinning, frames are sampled directly:
// sampleFrameDirect() lerps between captured frames, loop-aware.

import fs from 'node:fs';

export function loadFrameModel(jsonPath) {
  const manifest = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
  if (manifest.format !== 'goober-frames/1') {
    throw new Error(jsonPath + ': not a goober-frames/1 dump');
  }
  const binPath = jsonPath.replace(/\.json$/, '.bin');
  const buf = fs.readFileSync(binPath);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const L = manifest.layout;

  const positions = new Float32Array(ab, L.positions.offset, L.positions.floats);
  const normals = new Float32Array(ab, L.normals.offset, L.normals.floats);
  const colors = new Float32Array(ab, L.colors.offset, L.colors.floats);
  const aPrim = new Uint16Array(ab, L.aPrim.offset, L.aPrim.u16);
  const indices = new Uint32Array(ab, L.indices.offset, L.indices.u32);

  const V = manifest.vertexCount;
  const clipStart = [];
  let acc = 0;
  for (const c of manifest.clips) { clipStart.push(acc); acc += c.frameCount; }

  // static per-vertex data: aPrim doubles as the "dominant bone" for the LOD
  // clusterer, so limbs do not weld across prims
  const skinIndex = new Uint16Array(V * 4);
  const skinWeight = new Float32Array(V * 4);
  const materialIds = new Float32Array(V).fill(1);   // 1 = instance-tintable
  for (let v = 0; v < V; v++) {
    skinIndex[v * 4] = aPrim[v];
    skinWeight[v * 4] = 1;
  }

  function frameBase(clipIndex, frame) {
    return (clipStart[clipIndex] + frame) * V * 3;
  }

  function sampleFrameDirect(clipIndex, time, outPos, outNrm) {
    const clip = manifest.clips[clipIndex];
    const fc = clip.frameCount;
    let ft = (time / clip.duration) * fc;
    ft = ((ft % fc) + fc) % fc;
    const f0 = Math.floor(ft) % fc;
    const f1 = (f0 + 1) % fc;
    const a = ft - Math.floor(ft);
    const b0 = frameBase(clipIndex, f0);
    const b1 = frameBase(clipIndex, f1);
    for (let i = 0; i < V * 3; i++) {
      outPos[i] = positions[b0 + i] + (positions[b1 + i] - positions[b0 + i]) * a;
    }
    for (let v = 0; v < V; v++) {
      const i = v * 3;
      let nx = normals[b0 + i] + (normals[b1 + i] - normals[b0 + i]) * a;
      let ny = normals[b0 + i + 1] + (normals[b1 + i + 1] - normals[b0 + i + 1]) * a;
      let nz = normals[b0 + i + 2] + (normals[b1 + i + 2] - normals[b0 + i + 2]) * a;
      const l = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
      outNrm[i] = nx / l; outNrm[i + 1] = ny / l; outNrm[i + 2] = nz / l;
    }
  }

  return {
    name: manifest.kind,
    source: jsonPath,
    vertexCount: V,
    boneCount: 0,
    boneNames: [],
    positions: positions.slice(frameBase(0, 0), frameBase(0, 0) + V * 3),  // frame 0 as "bind pose"
    normals: normals.slice(frameBase(0, 0), frameBase(0, 0) + V * 3),
    uvs: new Float32Array(V * 2),
    colors,
    materialIds,
    skinIndex,
    skinWeight,
    indices,
    inverseBind: new Float32Array(0),
    clips: manifest.clips.map((c) => ({
      name: c.name,
      duration: c.duration,
      loop: c.loop !== false,
      strideOverride: c.stride,
      groundSpeedOverride: c.groundSpeed,
    })),
    sockets: null,
    vertexColorMode: true,
    sampleFrameDirect,
  };
}
