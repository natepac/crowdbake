// Turns a parsed glTF into the baker's RigModel: one merged skinned mesh plus a
// joint hierarchy that can be sampled at arbitrary times for every clip.

import { Matrix4, Quaternion, Vector3 } from 'three';
import { readGLTF } from './gltf.mjs';

const _p = new Vector3();
const _q = new Quaternion();
const _s = new Vector3();

function binarySearchKey(times, t) {
  let lo = 0, hi = times.length - 1;
  if (t <= times[0]) return 0;
  if (t >= times[hi]) return hi;
  while (lo < hi - 1) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= t) lo = mid; else hi = mid;
  }
  return lo;
}

function slerpQuat(out, o, values, ai, bi, t) {
  let ax = values[ai], ay = values[ai + 1], az = values[ai + 2], aw = values[ai + 3];
  const bx = values[bi], by = values[bi + 1], bz = values[bi + 2], bw = values[bi + 3];
  let cos = ax * bx + ay * by + az * bz + aw * bw;
  if (cos < 0) { cos = -cos; ax = -ax; ay = -ay; az = -az; aw = -aw; }
  let s0, s1;
  if (cos > 0.9995) { s0 = 1 - t; s1 = t; }
  else {
    const theta = Math.acos(cos);
    const sin = Math.sin(theta);
    s0 = Math.sin((1 - t) * theta) / sin;
    s1 = Math.sin(t * theta) / sin;
  }
  out[o] = ax * s0 + bx * s1;
  out[o + 1] = ay * s0 + by * s1;
  out[o + 2] = az * s0 + bz * s1;
  out[o + 3] = aw * s0 + bw * s1;
}

export function computeVertexNormals(positions, indices, out) {
  out.fill(0);
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i] * 3, b = indices[i + 1] * 3, c = indices[i + 2] * 3;
    const e1x = positions[b] - positions[a], e1y = positions[b + 1] - positions[a + 1], e1z = positions[b + 2] - positions[a + 2];
    const e2x = positions[c] - positions[a], e2y = positions[c + 1] - positions[a + 1], e2z = positions[c + 2] - positions[a + 2];
    const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
    out[a] += nx; out[a + 1] += ny; out[a + 2] += nz;
    out[b] += nx; out[b + 1] += ny; out[b + 2] += nz;
    out[c] += nx; out[c + 1] += ny; out[c + 2] += nz;
  }
  for (let i = 0; i < out.length; i += 3) {
    const l = Math.hypot(out[i], out[i + 1], out[i + 2]) || 1;
    out[i] /= l; out[i + 1] /= l; out[i + 2] /= l;
  }
  return out;
}

function basename(p) {
  return p.split(/[\\/]/).pop().replace(/\.(gltf|glb)$/i, '');
}

export function loadRigFromGLTF(file, opts = {}) {
  const { json, readAccessor } = readGLTF(file);
  const nodes = json.nodes || [];
  const nodeCount = nodes.length;

  // ---- hierarchy -------------------------------------------------------
  const parent = new Int32Array(nodeCount).fill(-1);
  for (let i = 0; i < nodeCount; i++) {
    for (const c of nodes[i].children || []) parent[c] = i;
  }
  const order = [];
  const seen = new Uint8Array(nodeCount);
  const visit = (i) => {
    if (seen[i]) return;
    seen[i] = 1;
    if (parent[i] >= 0) visit(parent[i]);
    order.push(i);
  };
  for (let i = 0; i < nodeCount; i++) visit(i);

  const restT = new Float32Array(nodeCount * 3);
  const restR = new Float32Array(nodeCount * 4);
  const restS = new Float32Array(nodeCount * 3);
  const tmpM = new Matrix4();
  for (let i = 0; i < nodeCount; i++) {
    const n = nodes[i];
    if (n.matrix) {
      tmpM.fromArray(n.matrix);
      tmpM.decompose(_p, _q, _s);
      restT[i * 3] = _p.x; restT[i * 3 + 1] = _p.y; restT[i * 3 + 2] = _p.z;
      restR[i * 4] = _q.x; restR[i * 4 + 1] = _q.y; restR[i * 4 + 2] = _q.z; restR[i * 4 + 3] = _q.w;
      restS[i * 3] = _s.x; restS[i * 3 + 1] = _s.y; restS[i * 3 + 2] = _s.z;
    } else {
      restT.set(n.translation || [0, 0, 0], i * 3);
      restR.set(n.rotation || [0, 0, 0, 1], i * 4);
      restS.set(n.scale || [1, 1, 1], i * 3);
    }
  }

  // ---- skin ------------------------------------------------------------
  const skinIdx = opts.skin != null ? opts.skin : 0;
  const skin = (json.skins || [])[skinIdx];
  if (!skin) throw new Error(file + ': no skin found (VAT baking needs a rigged mesh)');
  const joints = skin.joints.slice();
  const boneCount = joints.length;
  let inverseBind;
  if (skin.inverseBindMatrices != null) {
    inverseBind = readAccessor(skin.inverseBindMatrices);
  } else {
    inverseBind = new Float32Array(boneCount * 16);
    for (let i = 0; i < boneCount; i++) new Matrix4().toArray(inverseBind, i * 16);
  }
  const boneNames = joints.map((j, i) => nodes[j].name || ('joint_' + i));

  // ---- geometry (merge every primitive that uses this skin) -------------
  const chunks = [];
  for (let ni = 0; ni < nodeCount; ni++) {
    const n = nodes[ni];
    if (n.mesh == null || n.skin !== skinIdx) continue;
    for (const prim of json.meshes[n.mesh].primitives) {
      if (prim.mode != null && prim.mode !== 4) continue;
      const a = prim.attributes;
      const pos = readAccessor(a.POSITION);
      if (!pos) continue;
      chunks.push({
        count: pos.length / 3,
        position: pos,
        normal: a.NORMAL != null ? readAccessor(a.NORMAL) : null,
        uv: a.TEXCOORD_0 != null ? readAccessor(a.TEXCOORD_0) : null,
        color: a.COLOR_0 != null ? readAccessor(a.COLOR_0) : null,
        colorComps: a.COLOR_0 != null && json.accessors[a.COLOR_0].type === 'VEC4' ? 4 : 3,
        joints: a.JOINTS_0 != null ? readAccessor(a.JOINTS_0, { asInt: true }) : null,
        weights: a.WEIGHTS_0 != null ? readAccessor(a.WEIGHTS_0) : null,
        index: prim.indices != null ? readAccessor(prim.indices, { asInt: true }) : null,
      });
    }
  }
  if (!chunks.length) throw new Error(file + ': skin ' + skinIdx + ' has no triangle primitives');

  const vertexCount = chunks.reduce((s, c) => s + c.count, 0);
  const indexCount = chunks.reduce((s, c) => s + (c.index ? c.index.length : c.count), 0);

  const positions = new Float32Array(vertexCount * 3);
  const normals = new Float32Array(vertexCount * 3);
  const uvs = new Float32Array(vertexCount * 2);
  const colors = new Float32Array(vertexCount * 3).fill(1);
  const skinIndex = new Uint16Array(vertexCount * 4);
  const skinWeight = new Float32Array(vertexCount * 4);
  const indices = new Uint32Array(indexCount);

  let vo = 0, io = 0;
  let hadNormals = false;
  for (const c of chunks) {
    positions.set(c.position, vo * 3);
    if (c.normal) { normals.set(c.normal, vo * 3); hadNormals = true; }
    if (c.uv) uvs.set(c.uv, vo * 2);
    if (c.color) {
      for (let i = 0; i < c.count; i++) {
        colors[(vo + i) * 3] = c.color[i * c.colorComps];
        colors[(vo + i) * 3 + 1] = c.color[i * c.colorComps + 1];
        colors[(vo + i) * 3 + 2] = c.color[i * c.colorComps + 2];
      }
    }
    if (c.joints) for (let i = 0; i < c.count * 4; i++) skinIndex[vo * 4 + i] = c.joints[i];
    if (c.weights) skinWeight.set(c.weights, vo * 4);
    else for (let i = 0; i < c.count; i++) skinWeight[(vo + i) * 4] = 1;
    if (c.index) for (let i = 0; i < c.index.length; i++) indices[io + i] = c.index[i] + vo;
    else for (let i = 0; i < c.count; i++) indices[io + i] = vo + i;
    io += c.index ? c.index.length : c.count;
    vo += c.count;
  }

  for (let i = 0; i < vertexCount; i++) {
    const o = i * 4;
    const sum = skinWeight[o] + skinWeight[o + 1] + skinWeight[o + 2] + skinWeight[o + 3];
    if (sum > 1e-6) { for (let k = 0; k < 4; k++) skinWeight[o + k] /= sum; }
    else { skinWeight[o] = 1; skinWeight[o + 1] = 0; skinWeight[o + 2] = 0; skinWeight[o + 3] = 0; }
  }
  if (!hadNormals) computeVertexNormals(positions, indices, normals);

  // ---- animations ------------------------------------------------------
  const rawClips = (json.animations || []).map((anim, ci) => {
    const tracks = [];
    let duration = 0;
    for (const ch of anim.channels) {
      const node = ch.target.node;
      if (node == null) continue;
      if (ch.target.path === 'weights') continue; // morph targets are out of scope
      const smp = anim.samplers[ch.sampler];
      const times = readAccessor(smp.input);
      const values = readAccessor(smp.output);
      duration = Math.max(duration, times[times.length - 1]);
      tracks.push({ node, path: ch.target.path, times, values, interp: smp.interpolation || 'LINEAR' });
    }
    return { name: anim.name || ('clip_' + ci), duration, tracks };
  });

  // ---- sampling --------------------------------------------------------
  const localT = new Float32Array(nodeCount * 3);
  const localR = new Float32Array(nodeCount * 4);
  const localS = new Float32Array(nodeCount * 3);
  const world = new Float32Array(nodeCount * 16);
  const mA = new Matrix4(), mB = new Matrix4();

  function applyTrack(tr, time) {
    const { times, values, interp, path } = tr;
    const comps = path === 'rotation' ? 4 : 3;
    const out = path === 'rotation' ? localR : path === 'translation' ? localT : localS;
    const o = tr.node * comps;
    const n = times.length;
    const cubicOffset = interp === 'CUBICSPLINE' ? comps : 0;
    const stride = interp === 'CUBICSPLINE' ? comps * 3 : comps;
    if (n === 1) {
      for (let c = 0; c < comps; c++) out[o + c] = values[cubicOffset + c];
      return;
    }
    const i0 = binarySearchKey(times, time);
    const i1 = Math.min(i0 + 1, n - 1);
    const t0 = times[i0], t1 = times[i1];
    const a = t1 > t0 ? Math.min(Math.max((time - t0) / (t1 - t0), 0), 1) : 0;

    if (interp === 'STEP') {
      for (let c = 0; c < comps; c++) out[o + c] = values[i0 * comps + c];
      return;
    }
    if (interp === 'CUBICSPLINE') {
      const dt = t1 - t0;
      const a2 = a * a, a3 = a2 * a;
      const h00 = 2 * a3 - 3 * a2 + 1, h10 = a3 - 2 * a2 + a;
      const h01 = -2 * a3 + 3 * a2, h11 = a3 - a2;
      for (let c = 0; c < comps; c++) {
        const v0 = values[i0 * stride + comps + c];
        const b0 = values[i0 * stride + comps * 2 + c];
        const v1 = values[i1 * stride + comps + c];
        const m1 = values[i1 * stride + c];
        out[o + c] = h00 * v0 + h10 * dt * b0 + h01 * v1 + h11 * dt * m1;
      }
      if (comps === 4) {
        const len = Math.hypot(out[o], out[o + 1], out[o + 2], out[o + 3]) || 1;
        for (let c = 0; c < 4; c++) out[o + c] /= len;
      }
      return;
    }
    if (comps === 4) slerpQuat(out, o, values, i0 * 4, i1 * 4, a);
    else for (let c = 0; c < 3; c++) out[o + c] = values[i0 * 3 + c] * (1 - a) + values[i1 * 3 + c] * a;
  }

  function sampleJointWorld(clipIndex, time, outWorld) {
    localT.set(restT); localR.set(restR); localS.set(restS);
    const clip = rawClips[clipIndex];
    if (clip) for (const tr of clip.tracks) applyTrack(tr, time);
    for (let k = 0; k < order.length; k++) {
      const ni = order[k];
      _p.set(localT[ni * 3], localT[ni * 3 + 1], localT[ni * 3 + 2]);
      _q.set(localR[ni * 4], localR[ni * 4 + 1], localR[ni * 4 + 2], localR[ni * 4 + 3]);
      _s.set(localS[ni * 3], localS[ni * 3 + 1], localS[ni * 3 + 2]);
      mA.compose(_p, _q, _s);
      if (parent[ni] >= 0) { mB.fromArray(world, parent[ni] * 16); mA.premultiply(mB); }
      mA.toArray(world, ni * 16);
    }
    for (let b = 0; b < boneCount; b++) {
      const src = joints[b] * 16, dst = b * 16;
      for (let k = 0; k < 16; k++) outWorld[dst + k] = world[src + k];
    }
    return outWorld;
  }

  const scratchWorld = new Float32Array(boneCount * 16);
  function sampleSkinMatrices(clipIndex, time, outSkin) {
    sampleJointWorld(clipIndex, time, scratchWorld);
    for (let b = 0; b < boneCount; b++) {
      mA.fromArray(scratchWorld, b * 16);
      mB.fromArray(inverseBind, b * 16);
      mA.multiply(mB);
      mA.toArray(outSkin, b * 16);
    }
    return outSkin;
  }

  return {
    name: basename(file),
    source: file,
    vertexCount, boneCount, boneNames,
    positions, normals, uvs, colors, skinIndex, skinWeight, indices,
    inverseBind,
    clips: rawClips.map((c) => ({ name: c.name, duration: c.duration, loop: true })),
    sampleJointWorld,
    sampleSkinMatrices,
  };
}
