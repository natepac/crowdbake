// Loads a bake (manifest + binary) and turns it into GPU textures plus one
// shared BufferGeometry template per LOD.
//
// The LOD trick: every LOD stores the ORIGINAL vertex ids of the vertices it
// kept, so all levels address the same animation texture. An LOD therefore
// costs an index buffer and a handful of static attributes -- zero extra
// animation memory.

import {
  BufferAttribute,
  ClampToEdgeWrapping,
  DataTexture,
  FloatType,
  HalfFloatType,
  NearestFilter,
  RGBAFormat,
  RGFormat,
  Sphere,
  Vector3,
} from 'three';

const CTOR = { f32: Float32Array, u32: Uint32Array, u16: Uint16Array, f16: Uint16Array, u8: Uint8Array };

function makeDataTexture(data, width, height, format, type) {
  const tex = new DataTexture(data, width, height, format, type);
  tex.minFilter = NearestFilter;
  tex.magFilter = NearestFilter;
  tex.wrapS = ClampToEdgeWrapping;
  tex.wrapT = ClampToEdgeWrapping;
  tex.generateMipmaps = false;
  tex.flipY = false;
  tex.needsUpdate = true;
  return tex;
}

export class VATAsset {
  constructor(manifest, bin) {
    this.manifest = manifest;
    this.bin = bin;

    this.name = manifest.name;
    this.vertexCount = manifest.vertexCount;
    this.boneCount = manifest.boneCount;
    this.boneNames = manifest.boneNames || [];
    this.sockets = manifest.sockets || {};
    this.bounds = manifest.bounds;
    this.texture = manifest.texture;
    this.bone = manifest.bone;
    this.totalFrames = manifest.totalFrames;
    this.clips = manifest.clips;
    this.clipIndex = new Map(manifest.clips.map((c, i) => [c.name, i]));

    // rate = speed * rateScale[clip] + rateBase[clip]
    // Locomotion clips scale with speed through their stride; in-place clips
    // ignore speed and run at their authored rate. Precomputed so the per-agent
    // path in a 100k crowd is two array reads and a multiply-add.
    this.rateScale = new Float32Array(manifest.clips.length);
    this.rateBase = new Float32Array(manifest.clips.length);
    manifest.clips.forEach((c, i) => {
      if (c.stride && c.stride > 1e-4) this.rateScale[i] = 1 / c.stride;
      else this.rateBase[i] = 1 / c.duration;
    });

    const { width, height } = manifest.texture;
    const texels = width * height;

    // ---- animation textures ------------------------------------------
    const posSrc = this.section('posTex');
    const posData = new Uint16Array(texels * 4);
    posData.set(posSrc);
    this.posTex = makeDataTexture(posData, width, height, RGBAFormat, HalfFloatType);

    const nrmSrc = this.section('nrmTex');
    const nrmComps = manifest.sections.nrmTex.components;
    const nrmData = new Uint16Array(texels * nrmComps);
    nrmData.set(nrmSrc);
    this.nrmTex = makeDataTexture(nrmData, width, height, nrmComps === 4 ? RGBAFormat : RGFormat, HalfFloatType);

    this.boneTex = null;
    if (manifest.bone && manifest.sections.boneTex) {
      const bw = manifest.bone.width, bh = manifest.bone.height;
      const src = this.section('boneTex');
      const data = new Float32Array(bw * bh * 4);
      data.set(src);
      this.boneTex = makeDataTexture(data, bw, bh, RGBAFormat, FloatType);
    }

    // ---- shared static geometry data ---------------------------------
    this.bindPosition = this.section('bindPosition');
    this.bindNormal = this.section('bindNormal');
    this.uv = this.section('uv');
    this.color = this.section('color');
    this.materialId = this.section('materialId');
    this.skinIndexData = this.section('skinIndex');
    this.skinWeightData = this.section('skinWeight');

    this.lods = manifest.lods.map((lod, i) => this.buildLODAttributes(i, lod));

    // a sphere big enough to contain one instance at unit scale
    const b = manifest.bounds;
    this.instanceRadius = Math.max(
      Math.hypot(b.maxRadiusXZ, Math.max(Math.abs(b.min[1]), Math.abs(b.min[1] + b.extent[1]))),
      b.extent[1] * 0.5,
    );
    this.instanceHeight = b.min[1] + b.extent[1];

    this.memory = {
      positions: posSrc.byteLength,
      normals: nrmSrc.byteLength,
      bones: this.boneTex ? this.section('boneTex').byteLength : 0,
      geometry: this.lods.reduce((s, l) => s + l.byteLength, 0),
      total: 0,
    };
    this.memory.total = this.memory.positions + this.memory.normals + this.memory.bones + this.memory.geometry;
  }

  section(name) {
    const s = this.manifest.sections[name];
    if (!s) return null;
    const ctor = CTOR[s.type];
    return new ctor(this.bin, s.offset, s.byteLength / ctor.BYTES_PER_ELEMENT);
  }

  buildLODAttributes(level) {
    const vids = this.section(`lod${level}.vids`);
    const index = this.section(`lod${level}.index`);
    const n = vids.length;

    const position = new Float32Array(n * 3);
    const normal = new Float32Array(n * 3);
    const uv = new Float32Array(n * 2);
    const vid = new Float32Array(n);
    const materialId = new Float32Array(n);
    const color = this.color ? new Float32Array(n * 3) : null;
    const skinIndex = this.skinIndexData ? new Float32Array(n * 4) : null;
    const skinWeight = this.skinWeightData ? new Float32Array(n * 4) : null;

    for (let i = 0; i < n; i++) {
      const v = vids[i];
      position[i * 3] = this.bindPosition[v * 3];
      position[i * 3 + 1] = this.bindPosition[v * 3 + 1];
      position[i * 3 + 2] = this.bindPosition[v * 3 + 2];
      normal[i * 3] = this.bindNormal[v * 3];
      normal[i * 3 + 1] = this.bindNormal[v * 3 + 1];
      normal[i * 3 + 2] = this.bindNormal[v * 3 + 2];
      uv[i * 2] = this.uv[v * 2];
      uv[i * 2 + 1] = this.uv[v * 2 + 1];
      vid[i] = v;
      materialId[i] = this.materialId ? this.materialId[v] : 0;
      if (color) {
        color[i * 3] = this.color[v * 3];
        color[i * 3 + 1] = this.color[v * 3 + 1];
        color[i * 3 + 2] = this.color[v * 3 + 2];
      }
      if (skinIndex) {
        for (let k = 0; k < 4; k++) {
          skinIndex[i * 4 + k] = this.skinIndexData[v * 4 + k];
          skinWeight[i * 4 + k] = this.skinWeightData[v * 4 + k];
        }
      }
    }

    const attrs = {
      position: new BufferAttribute(position, 3),
      normal: new BufferAttribute(normal, 3),
      uv: new BufferAttribute(uv, 2),
      aVid: new BufferAttribute(vid, 1),
      aMaterialId: new BufferAttribute(materialId, 1),
    };
    if (color) attrs.color = new BufferAttribute(color, 3);
    if (skinIndex) {
      attrs.aSkinIndex = new BufferAttribute(skinIndex, 4);
      attrs.aSkinWeight = new BufferAttribute(skinWeight, 4);
    }

    let byteLength = 0;
    for (const a of Object.values(attrs)) byteLength += a.array.byteLength;
    byteLength += index.byteLength;

    return {
      level,
      vertexCount: n,
      triangleCount: index.length / 3,
      attributes: attrs,
      index: new BufferAttribute(index, 1),
      byteLength,
    };
  }

  /** cycles per second so the character's feet match `speed` metres/second */
  rateForSpeed(clipIndex, speed) {
    return speed * this.rateScale[clipIndex] + this.rateBase[clipIndex];
  }

  /** the rate the clip was authored at */
  naturalRate(clipIndex) {
    const c = this.clips[clipIndex];
    return c ? 1 / c.duration : 0;
  }

  clipByName(name) {
    const i = this.clipIndex.get(name);
    return i === undefined ? -1 : i;
  }

  boundingSphere(scale = 1) {
    return new Sphere(new Vector3(0, this.instanceHeight * 0.5 * scale, 0), this.instanceRadius * scale);
  }

  dispose() {
    this.posTex.dispose();
    this.nrmTex.dispose();
    if (this.boneTex) this.boneTex.dispose();
  }

  /**
   * onProgress(loadedBytes, totalBytes) fires as the binary streams in;
   * totalBytes is 0 when the server does not say (compressed transfer without
   * a usable Content-Length), so treat 0 as "indeterminate".
   */
  static async load(url, { fetchImpl = fetch, onProgress } = {}) {
    // single-file builds embed bakes as base64 (tools/build-single.mjs); the
    // registry, when present, replaces the network entirely
    const inline = globalThis.__CROWDBAKE_INLINE;
    const name = url.split('/').pop().replace(/\.json$/, '');
    if (inline && inline[name]) {
      const bin = await (await fetch('data:application/octet-stream;base64,' + inline[name].bin)).arrayBuffer();
      if (onProgress) onProgress(bin.byteLength, bin.byteLength);
      return new VATAsset(inline[name].manifest, bin);
    }
    const manRes = await fetchImpl(url);
    if (!manRes.ok) throw new Error(`could not load ${name}.json (HTTP ${manRes.status})`);
    let manifest;
    try {
      manifest = await manRes.json();
    } catch (e) {
      // SPA-style hosts answer missing files with index.html and HTTP 200
      throw new Error(`there is no bake named "${name}" here (the server sent a page instead of a manifest)`);
    }
    const binURL = new URL(manifest.binary, new URL(url, location.href)).href;
    const res = await fetchImpl(binURL);
    if (!res.ok) throw new Error(`could not load ${manifest.binary} (HTTP ${res.status})`);

    let bin;
    if (onProgress && res.body) {
      // stream so the loading bar can show real byte progress. Content-Length
      // on a compressed transfer counts compressed bytes while we count
      // decompressed ones, so the fraction is clamped at the call site.
      const total = parseInt(res.headers.get('Content-Length') || '0', 10) || 0;
      const reader = res.body.getReader();
      const chunks = [];
      let loaded = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.byteLength;
        onProgress(loaded, total);
      }
      const buf = new Uint8Array(loaded);
      let o = 0;
      for (const c of chunks) { buf.set(c, o); o += c.byteLength; }
      bin = buf.buffer;
    } else {
      bin = await res.arrayBuffer();
      if (onProgress) onProgress(bin.byteLength, bin.byteLength);
    }
    return new VATAsset(manifest, bin);
  }
}
