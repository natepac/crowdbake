// Chunked instanced crowd renderer.
//
// Two problems this solves that a plain InstancedMesh does not:
//
// 1. Culling. An InstancedMesh has one bounding sphere, so it is all-or-nothing:
//    walk into the middle of a 100k crowd and you submit all 100k. Here
//    instances are counting-sorted into a spatial grid every frame, each chunk
//    gets its own bounding sphere and its own draw, and each chunk picks an LOD
//    from its distance to the camera. Set chunkSize >= worldSize to collapse
//    back to the classic single draw call.
//
// 2. Upload cost. Per-instance state lives in ONE interleaved 16-float record
//    (64 bytes, one cache line). Building a chunk's buffer is a gather -- a
//    scattered read per instance -- so touching four separate arrays meant four
//    cache misses per instance, and the cost went super-linear once the working
//    set outgrew L3. Interleaved it is one miss per instance, and one
//    bufferSubData instead of four.
//
// Animation phase is evaluated on the GPU as fract(offset + time * rate), so a
// crowd that is not moving uploads nothing at all and still animates.

import {
  DynamicDrawUsage,
  InstancedBufferGeometry,
  InstancedInterleavedBuffer,
  InterleavedBufferAttribute,
  Mesh,
  Object3D,
  Sphere,
  Vector3,
} from 'three';

import { createVATDepthMaterial, createVATMaterial, createVATUniforms } from './vat-shader.js';

/** floats per instance record */
export const STRIDE = 16;

/** offsets within an instance record */
export const OFF = {
  posX: 0, posY: 1, posZ: 2, yaw: 3,
  clipA: 4, phaseA: 5, rateA: 6, scale: 7,
  clipB: 8, phaseB: 9, rateB: 10, fadeStart: 11,
  tintR: 12, tintG: 13, tintB: 14, variant: 15,
};

const DEFAULTS = {
  capacity: 20000,
  mode: 'vat',
  chunkSize: 24,
  worldSize: 512,
  lodDistances: [6, 14, 34, 80],
  // Where in a chunk the LOD distance is measured from. 'edge' uses the chunk's
  // near edge, which never under-detais an instance but means a big chunk holds
  // LOD0 until its centre is (chunkRadius + lodDistance[0]) away -- with 42 m
  // chunks that is 52 m, and essentially the whole crowd renders at full detail.
  // 'centre' is what you actually want; use smaller chunks for finer granularity.
  lodPivot: 'centre',
  castShadow: true,
  receiveShadow: true,
  // Chunks at or beyond this LOD stop casting. A character 45 m away is a few
  // pixels in the shadow map, but it still costs a full extra draw call -- and
  // the shadow pass is otherwise an exact duplicate of the colour pass's draw
  // count, which is what the renderer becomes bound by once LODs are working.
  shadowMaxLod: 2,
  crossFade: true,
  crossFadeMaxLod: 1,
  fadeDuration: 0.28,
  lodBias: 1,
  // Merge every chunk sitting at the last LOD into ONE unculled instanced draw.
  // Zoomed out, per-chunk draws dominate: ~290 chunks x 2 passes of a 44-tri
  // mesh is pure draw-call overhead. The pool trades frustum culling (worthless
  // for 44-tri instances) for a single draw.
  farPool: true,
  // 'lambert' (cheap diffuse) or 'pbr' (full MeshStandardMaterial)
  quality: 'lambert',
  // Playback rates snap to this many cycles/second, with a deadband either side.
  // A steering crowd's speed jitters every frame; without this every agent
  // rewrites its rate continuously and no chunk is ever considered unchanged.
  rateQuantum: 0.05,
  initialChunkCapacity: 128,
  materialOptions: {},
  timeWrap: 3600,
};

// three's geometry.dispose() deletes the GL buffer of EVERY attribute on the
// geometry -- including attributes shared with other geometries (the LOD
// templates every chunk references). Other live geometries keep cached VAOs
// pointing at the deleted buffers and draw garbage (or warn) for a frame.
// Stripping shared attributes first means dispose only frees per-chunk data.

const OWNED_ATTRS = new Set(['iXform', 'iAnimA', 'iAnimB', 'iTint']);
function disposeKeepingShared(geom) {
  geom.setIndex(null);
  for (const name of Object.keys(geom.attributes)) {
    if (!OWNED_ATTRS.has(name)) geom.deleteAttribute(name);
  }
  geom.dispose();
}

class Chunk {

  constructor(crowd, id, cx, cz, expected = 0) {
    this.crowd = crowd;
    this.id = id;
    this.cx = cx;
    this.cz = cz;
    this.capacity = 0;
    this.count = 0;
    this.staticDirty = true;
    this.wasFar = false;
    // start from the cell's own bounds so a chunk born far still gets a sane
    // LOD decision before its first gather
    this.sphere = new Sphere(
      new Vector3(
        crowd.gridOrigin + (cx + 0.5) * crowd.cellSize,
        crowd.asset.instanceHeight * 0.55,
        crowd.gridOrigin + (cz + 0.5) * crowd.cellSize,
      ),
      Math.SQRT2 * crowd.cellSize * 0.5 + crowd.asset.instanceRadius * 1.3,
    );
    this.array = null;
    this.buffer = null;
    this.attrs = null;
    this.geometries = [];
    this.meshes = [];
    this.attachmentMeshes = [];
    this.activeLod = -1;
    // Size from what the chunk actually holds on its first frame. Growing means
    // disposing and rebuilding this chunk's geometries, because three only
    // releases an attribute's GL buffer through BufferGeometry.dispose(), so the
    // aim is to grow approximately never rather than to grow cheaply.
    this.grow(Math.max(crowd.options.initialChunkCapacity, expected * 2));
  }

  grow(capacity) {
    if (capacity <= this.capacity) return;
    const cap = Math.max(capacity, this.capacity * 2);
    const old = this.array;
    this.array = new Float32Array(cap * STRIDE);
    if (old) this.array.set(old);
    this.capacity = cap;

    this.buffer = new InstancedInterleavedBuffer(this.array, STRIDE);
    this.buffer.setUsage(DynamicDrawUsage);
    this.attrs = {
      iXform: new InterleavedBufferAttribute(this.buffer, 4, 0),
      iAnimA: new InterleavedBufferAttribute(this.buffer, 4, 4),
      iAnimB: new InterleavedBufferAttribute(this.buffer, 4, 8),
      iTint: new InterleavedBufferAttribute(this.buffer, 4, 12),
    };

    this.rebuildGeometries();
    this.staticDirty = true;
  }

  rebuildGeometries() {
    const { crowd } = this;
    for (const g of this.geometries) disposeKeepingShared(g);
    this.geometries.length = 0;
    for (const m of this.meshes) crowd.remove(m);
    this.meshes.length = 0;
    for (const m of this.attachmentMeshes) { crowd.remove(m); disposeKeepingShared(m.geometry); }
    this.attachmentMeshes.length = 0;

    crowd.asset.lods.forEach((lod, i) => {
      const geom = new InstancedBufferGeometry();
      for (const [name, attr] of Object.entries(lod.attributes)) geom.setAttribute(name, attr);
      geom.setIndex(lod.index);
      for (const [name, attr] of Object.entries(this.attrs)) geom.setAttribute(name, attr);
      geom.instanceCount = 0;
      geom.boundingSphere = this.sphere;

      const mesh = new Mesh(geom, crowd.materials[i]);
      mesh.customDepthMaterial = crowd.depthMaterials[i];
      mesh.castShadow = crowd.options.castShadow;
      mesh.receiveShadow = crowd.options.receiveShadow;
      mesh.matrixAutoUpdate = false;
      mesh.visible = false;
      mesh.frustumCulled = true;
      mesh.userData.chunk = this;
      mesh.userData.lod = i;
      // fires once per colour pass, after three's frustum test -- the only place
      // the actual post-cull instance count can be observed
      mesh.onBeforeRender = () => {
        const st = crowd.stats;
        st.renderedInstances += geom.instanceCount;
        st.renderedChunks++;
      };

      this.geometries.push(geom);
      this.meshes.push(mesh);
      crowd.add(mesh);
    });

    for (const att of crowd.attachments) att.attachChunk(this);
  }

  dispose() {
    for (const g of this.geometries) disposeKeepingShared(g);
    for (const m of this.meshes) this.crowd.remove(m);
    for (const m of this.attachmentMeshes) {
      this.crowd.remove(m);
      disposeKeepingShared(m.geometry);
    }
    this.geometries.length = 0;
    this.meshes.length = 0;
    this.attachmentMeshes.length = 0;
  }
}

export class VATCrowd extends Object3D {
  constructor(asset, options = {}) {
    super();
    this.asset = asset;
    this.options = { ...DEFAULTS, ...options };
    this.name = 'VATCrowd';

    const cap = this.options.capacity;
    this.capacity = cap;
    this._count = 0;

    // authoritative per-instance state, interleaved
    this.data = new Float32Array(cap * STRIDE);
    this.animDirty = new Uint8Array(cap);
    // set per instance when its transform changes; consumed into per-cell dirt,
    // so cells where nothing moved skip their gather and upload entirely
    this.posDirty = new Uint8Array(cap);
    this.prevChunk = new Int32Array(cap).fill(-1);
    this.transformsDirty = true;   // force-everything flag (count change, reconfigure)
    // instances currently cross-fading, promoted to slot A once the fade ends.
    // Without this, slot A keeps the OLD clip forever, and LODs compiled without
    // cross-fade (crossFadeMaxLod and the far pool) render that stale clip.
    this._fades = [];
    this._fadeHead = 0;

    for (let i = 0; i < cap; i++) {
      const o = i * STRIDE;
      this.data[o + OFF.scale] = 1;
      this.data[o + OFF.clipB] = -1;
      this.data[o + OFF.tintR] = 1;
      this.data[o + OFF.tintG] = 1;
      this.data[o + OFF.tintB] = 1;
    }

    this.time = 0;
    this._rateQ = 1 / this.options.rateQuantum;
    this._rateDead = this.options.rateQuantum * 0.75;
    this.uniforms = createVATUniforms(asset);
    this.uniforms.vatInvFade.value = 1 / this.options.fadeDuration;

    this.attachments = [];
    this.materials = [];
    this.depthMaterials = [];
    this._buildMaterials();
    this._configureGrid();

    this._sorted = new Int32Array(cap);
    this._cellOf = new Int32Array(cap);

    this.stats = {
      instances: 0,          // simulated
      drawnInstances: 0,     // submitted (post chunk-emptiness, pre frustum)
      renderedInstances: 0,  // actually rasterised (post frustum cull)
      chunks: 0,             // chunks holding at least one instance
      activeChunks: 0,       // chunks submitted this frame
      renderedChunks: 0,     // chunks that survived the frustum
      dirtyChunks: 0,        // chunks whose animation state was re-uploaded
      triangles: 0,
      lodCounts: new Array(asset.lods.length).fill(0),
      uploadBytes: 0,
      farPooled: 0,          // instances rendered through the merged far-pool draw
      bucketMs: 0,
      copyMs: 0,
    };

    this._cameraPos = new Vector3();
    this._lastCamPos = new Vector3(Infinity, Infinity, Infinity);

    // far pool buffers (geometry/mesh built in _buildMaterials)
    this._poolArray = new Float32Array(cap * STRIDE);
    this._poolBuffer = null;
    this._poolGeometry = null;
    this._poolMesh = null;
    this._poolStale = true;
    this._buildPool();
  }

  _buildPool() {
    if (this._poolGeometry) {
      this.remove(this._poolMesh);
      disposeKeepingShared(this._poolGeometry);
    }
    this._poolBuffer = new InstancedInterleavedBuffer(this._poolArray, STRIDE);
    this._poolBuffer.setUsage(DynamicDrawUsage);
    const lodIdx = this.asset.lods.length - 1;
    const lod = this.asset.lods[lodIdx];
    const geom = new InstancedBufferGeometry();
    for (const [name, attr] of Object.entries(lod.attributes)) geom.setAttribute(name, attr);
    geom.setIndex(lod.index);
    geom.setAttribute('iXform', new InterleavedBufferAttribute(this._poolBuffer, 4, 0));
    geom.setAttribute('iAnimA', new InterleavedBufferAttribute(this._poolBuffer, 4, 4));
    geom.setAttribute('iAnimB', new InterleavedBufferAttribute(this._poolBuffer, 4, 8));
    geom.setAttribute('iTint', new InterleavedBufferAttribute(this._poolBuffer, 4, 12));
    geom.instanceCount = 0;
    const mesh = new Mesh(geom, this.materials[lodIdx]);
    mesh.castShadow = false;             // far instances stopped casting anyway
    mesh.receiveShadow = this.options.receiveShadow;
    mesh.frustumCulled = false;          // spans the world by definition
    mesh.matrixAutoUpdate = false;
    mesh.visible = false;
    mesh.onBeforeRender = () => {
      this.stats.renderedInstances += geom.instanceCount;
      this.stats.renderedChunks++;
    };
    this._poolGeometry = geom;
    this._poolMesh = mesh;
    this.add(mesh);
    this._poolStale = true;
  }

  // -------------------------------------------------------------- setup ---

  _buildMaterials() {
    for (const m of this.materials) m.dispose();
    for (const m of this.depthMaterials) m.dispose();
    this.materials = [];
    this.depthMaterials = [];
    const boneMode = this.options.mode === 'bone';
    if (boneMode && !this.asset.boneTex) {
      throw new Error('VATCrowd: mode "bone" requires a bake made with --bones');
    }
    this.asset.lods.forEach((lod, i) => {
      const crossFade = this.options.crossFade && i <= this.options.crossFadeMaxLod;
      const common = {
        boneMode,
        crossFade,
        tangents: this.asset.texture.hasTangents,
        lodLevel: i,
        quality: this.options.quality,
        ...this.options.materialOptions,
      };
      this.materials.push(createVATMaterial(this.asset, this.uniforms, common));
      this.depthMaterials.push(createVATDepthMaterial(this.asset, this.uniforms, common));
    });
    if (this._poolArray) this._buildPool();
  }

  _configureGrid() {
    const { chunkSize, worldSize } = this.options;
    const single = !chunkSize || !isFinite(chunkSize) || chunkSize >= worldSize;
    this.gridN = single ? 1 : Math.max(1, Math.ceil(worldSize / chunkSize));
    this.cellSize = single ? worldSize : chunkSize;
    this.gridOrigin = -worldSize / 2;
    const n = this.gridN * this.gridN;
    if (this.chunks) for (const c of this.chunks) if (c) c.dispose();
    this.chunks = new Array(n).fill(null);
    this._counts = new Int32Array(n);
    this._offsets = new Int32Array(n + 1);
    this._cursor = new Int32Array(n);
    this._dirtyCells = new Uint8Array(n);
    this.prevChunk.fill(-1);
  }

  /** Change grid, playback mode or blending at runtime. */
  reconfigure(patch) {
    Object.assign(this.options, patch);
    if (patch.fadeDuration) this.uniforms.vatInvFade.value = 1 / patch.fadeDuration;
    if (patch.rateQuantum) {
      this._rateQ = 1 / patch.rateQuantum;
      this._rateDead = patch.rateQuantum * 0.75;
    }
    if (patch.mode !== undefined || patch.crossFade !== undefined
      || patch.crossFadeMaxLod !== undefined || patch.quality !== undefined) {
      this._buildMaterials();
      for (const c of this.chunks) if (c) c.rebuildGeometries();
    }
    if (patch.chunkSize !== undefined || patch.worldSize !== undefined) this._configureGrid();
    if (patch.farPool !== undefined) this._poolStale = true;
    this.transformsDirty = true;
  }

  // --------------------------------------------------------- instance API --

  get count() { return this._count; }

  set count(n) {
    const next = Math.max(0, Math.min(this.capacity, n | 0));
    if (next !== this._count) this.transformsDirty = true;
    this._count = next;
    this.stats.instances = this._count;
  }

  setTransform(i, x, y, z, yaw) {
    const o = i * STRIDE;
    this.data[o] = x; this.data[o + 1] = y; this.data[o + 2] = z; this.data[o + 3] = yaw;
    this.posDirty[i] = 1;
  }

  setScale(i, s) {
    this.data[i * STRIDE + OFF.scale] = s;
    this.animDirty[i] = 1;
  }

  getScale(i) { return this.data[i * STRIDE + OFF.scale]; }

  /** @param variant {shirt:[r,g,b], pants:0..7, skin:0..7, accessory:0..7} */
  setVariant(i, variant) {
    const o = i * STRIDE;
    const shirt = variant.shirt || [1, 1, 1];
    this.data[o + OFF.tintR] = shirt[0];
    this.data[o + OFF.tintG] = shirt[1];
    this.data[o + OFF.tintB] = shirt[2];
    this.data[o + OFF.variant] = ((variant.pants || 0) & 7)
      + ((variant.skin || 0) & 7) * 8
      + ((variant.accessory || 0) & 7) * 64;
    this.animDirty[i] = 1;
  }

  getAccessory(i) { return Math.floor(this.data[i * STRIDE + OFF.variant] / 64) & 7; }

  currentClip(i) { return this.data[i * STRIDE + OFF.clipA] | 0; }

  /** Snap to a clip with no blend. */
  setClip(i, clipIndex, rate = null, phase = null) {
    const o = i * STRIDE;
    const r = rate === null ? this.asset.naturalRate(clipIndex) : rate;
    const p = phase === null ? Math.random() : phase;
    this.data[o + OFF.clipA] = clipIndex;
    this.data[o + OFF.phaseA] = wrap01(p - this.time * r);
    this.data[o + OFF.rateA] = r;
    this.data[o + OFF.clipB] = -1;
    this.animDirty[i] = 1;
  }

  /** Cross-fade into a clip; falls back to a snap when cross-fade is off. */
  play(i, clipIndex, opts = {}) {
    if (!this.options.crossFade) return this.setClip(i, clipIndex, opts.rate, opts.phase);
    const o = i * STRIDE;
    const d = this.data;
    this._promote(i);
    if ((d[o + OFF.clipA] | 0) === clipIndex && d[o + OFF.clipB] < 0) return;

    const r = opts.rate != null ? opts.rate : this.asset.naturalRate(clipIndex);
    // start the incoming clip in step with the outgoing one so feet stay in sync
    const phase = opts.phase != null
      ? opts.phase
      : wrap01(d[o + OFF.phaseA] + this.time * d[o + OFF.rateA]);
    d[o + OFF.clipB] = clipIndex;
    d[o + OFF.phaseB] = wrap01(phase - this.time * r);
    d[o + OFF.rateB] = r;
    d[o + OFF.fadeStart] = this.time;
    this.animDirty[i] = 1;
    this._fades.push(i);
  }

  /** Drive playback rate from a desired ground speed, keeping phase continuous. */
  setSpeed(i, speed) {
    const o = i * STRIDE;
    const d = this.data;
    const q = this._rateQ, dead = this._rateDead;
    const scale = this.asset.rateScale, bias = this.asset.rateBase;

    const ca = d[o + OFF.clipA] | 0;
    const cur = d[o + OFF.rateA];
    const rawA = speed * scale[ca] + bias[ca];
    if (rawA > cur + dead || rawA < cur - dead) {
      const snapped = Math.round(rawA * q) / q;
      if (snapped !== cur) this._setRate(i, o + OFF.phaseA, snapped);
    }
    if (d[o + OFF.clipB] >= 0) {
      const cb = d[o + OFF.clipB] | 0;
      const curB = d[o + OFF.rateB];
      const rawB = speed * scale[cb] + bias[cb];
      if (rawB > curB + dead || rawB < curB - dead) {
        const snapped = Math.round(rawB * q) / q;
        if (snapped !== curB) this._setRate(i, o + OFF.phaseB, snapped);
      }
    }
  }

  setRate(i, rate) {
    const po = i * STRIDE + OFF.phaseA;
    const snapped = Math.round(rate * this._rateQ) / this._rateQ;
    if (snapped !== this.data[po + 1]) this._setRate(i, po, snapped);
  }

  /**
   * Change a playback rate while preserving the current phase, so feet do not
   * teleport when an agent speeds up. phaseOffset lives at `po`, rate at `po+1`.
   */
  _setRate(i, po, rate) {
    const d = this.data;
    const t = this.time;
    let phase = d[po] + t * d[po + 1];
    phase -= Math.floor(phase);
    let off = phase - t * rate;
    off -= Math.floor(off);
    d[po] = off;
    d[po + 1] = rate;
    this.animDirty[i] = 1;
  }

  // Fold an in-flight blend target into A. There is only one blend slot, so a
  // clip change mid-fade adopts the newer intent rather than snapping back to
  // the clip we were already leaving.
  _promote(i) {
    const o = i * STRIDE;
    const d = this.data;
    if (d[o + OFF.clipB] < 0) return;
    d[o + OFF.clipA] = d[o + OFF.clipB];
    d[o + OFF.phaseA] = d[o + OFF.phaseB];
    d[o + OFF.rateA] = d[o + OFF.rateB];
    d[o + OFF.clipB] = -1;
    this.animDirty[i] = 1;
  }

  /** Call after writing transforms straight into `crowd.data`. */
  markTransformsDirty() { this.transformsDirty = true; }

  // -------------------------------------------------------------- update ---

  update(dt, camera) {
    const t0 = performance.now();
    this.time += dt;
    if (this.time > this.options.timeWrap) this._wrapTime();
    this.uniforms.vatTime.value = this.time;

    // fold finished cross-fades into slot A (FIFO by fade start time)
    {
      const fades = this._fades;
      const d = this.data;
      const dur = this.options.fadeDuration;
      while (this._fadeHead < fades.length) {
        const i = fades[this._fadeHead];
        const o = i * STRIDE;
        if (d[o + OFF.clipB] < 0) { this._fadeHead++; continue; }   // already promoted
        if (this.time - d[o + OFF.fadeStart] < dur) break;          // head not due yet
        this._promote(i);
        this._fadeHead++;
      }
      if (this._fadeHead > 1024 && this._fadeHead * 2 > fades.length) {
        fades.splice(0, this._fadeHead);
        this._fadeHead = 0;
      }
    }

    const n = this._count;
    const { gridN, cellSize, gridOrigin } = this;
    const counts = this._counts;
    counts.fill(0);
    const cellOf = this._cellOf;
    const data = this.data;
    const last = gridN - 1;

    for (let i = 0; i < n; i++) {
      const o = i * STRIDE;
      let cx = ((data[o] - gridOrigin) / cellSize) | 0;
      let cz = ((data[o + 2] - gridOrigin) / cellSize) | 0;
      if (cx < 0) cx = 0; else if (cx > last) cx = last;
      if (cz < 0) cz = 0; else if (cz > last) cz = last;
      const cell = cz * gridN + cx;
      cellOf[i] = cell;
      counts[cell]++;
    }

    const offsets = this._offsets;
    let acc = 0;
    for (let c = 0; c < counts.length; c++) { offsets[c] = acc; acc += counts[c]; }
    offsets[counts.length] = acc;

    const cursor = this._cursor;
    cursor.set(offsets.subarray(0, counts.length));
    const sorted = this._sorted;
    const dirtyCells = this._dirtyCells;
    dirtyCells.fill(0);
    const prevChunk = this.prevChunk;
    const animDirty = this.animDirty;
    const posDirty = this.posDirty;
    const chunks = this.chunks;
    let membershipChanged = false;
    let movedAny = false;

    for (let i = 0; i < n; i++) {
      const cell = cellOf[i];
      sorted[cursor[cell]++] = i;
      const prev = prevChunk[i];
      if (prev !== cell) {
        if (prev >= 0 && chunks[prev]) chunks[prev].staticDirty = true;
        dirtyCells[cell] = 1;
        prevChunk[i] = cell;
        membershipChanged = true;
      }
      if (posDirty[i]) {
        dirtyCells[cell] = 1;
        posDirty[i] = 0;
        movedAny = true;
      }
      if (animDirty[i]) {
        dirtyCells[cell] = 1;
        animDirty[i] = 0;
      }
    }

    this.stats.bucketMs = performance.now() - t0;
    const t1 = performance.now();

    if (camera) camera.getWorldPosition(this._cameraPos);
    const cameraMoved = this._lastCamPos.distanceToSquared(this._cameraPos) > 1e-8;
    this._lastCamPos.copy(this._cameraPos);
    const lodDist = this.options.lodDistances;
    const lodBias = this.options.lodBias;
    const lodFromEdge = this.options.lodPivot === 'edge';
    const shadowMaxLod = this.options.shadowMaxLod;
    const castShadow = this.options.castShadow;
    const stats = this.stats;
    stats.drawnInstances = 0;
    stats.activeChunks = 0;
    stats.renderedInstances = 0;
    stats.renderedChunks = 0;
    stats.dirtyChunks = 0;
    stats.triangles = 0;
    stats.uploadBytes = 0;
    stats.farPooled = 0;
    stats.lodCounts.fill(0);
    let liveChunks = 0;

    // A cell where nothing moved needs no copy and no upload; GPU-side phase
    // keeps it animating regardless. transformsDirty is the force-everything
    // escape hatch (count change, reconfigure).
    const forceAll = this.transformsDirty;
    const lastLod = this.asset.lods.length - 1;
    const poolEnabled = this.options.farPool !== false && this.gridN > 1;
    // LOD classification only changes when the camera or a sphere moved, so the
    // pool only needs rebuilding then; an unchanged pool keeps last frame's
    // buffer and costs nothing.
    const poolRebuild = poolEnabled
      && (forceAll || movedAny || membershipChanged || cameraMoved || this._poolStale);
    const poolDst = this._poolArray;
    let poolCursor = 0;
    const instanceRadius = this.asset.instanceRadius;
    const instanceHeight = this.asset.instanceHeight;

    for (let cell = 0; cell < counts.length; cell++) {
      const cnt = counts[cell];
      let chunk = chunks[cell];
      if (cnt === 0) {
        if (chunk) {
          for (const m of chunk.meshes) m.visible = false;
          for (const m of chunk.attachmentMeshes) m.visible = false;
        }
        continue;
      }
      liveChunks++;
      if (!chunk) {
        chunk = new Chunk(this, cell, cell % gridN, (cell / gridN) | 0, cnt);
        chunks[cell] = chunk;
      }
      if (cnt > chunk.capacity) chunk.grow(cnt);
      if (dirtyCells[cell]) chunk.staticDirty = true;

      // ---- far classification. Transitions can only happen on frames where
      // poolRebuild is already true (they require camera or sphere movement),
      // so a stale pool never disagrees with the per-chunk visibility below.
      if (poolEnabled) {
        const cd = this._cameraPos.distanceTo(chunk.sphere.center);
        const dd = Math.max(0, cd - (lodFromEdge ? chunk.sphere.radius : instanceRadius)) * lodBias;
        if (dd >= lodDist[lodDist.length - 1]) {
          if (poolRebuild) {
            const base = offsets[cell];
            for (let k = 0; k < cnt; k++) {
              const so = sorted[base + k] * STRIDE;
              const dof = poolCursor * STRIDE;
              for (let c = 0; c < STRIDE; c++) poolDst[dof + c] = data[so + c];
              poolCursor++;
            }
          } else {
            poolCursor += cnt;
          }
          for (let l = 0; l < chunk.meshes.length; l++) {
            chunk.meshes[l].visible = false;
            chunk.geometries[l].instanceCount = 0;
          }
          for (const m of chunk.attachmentMeshes) { m.visible = false; m.geometry.instanceCount = 0; }
          chunk.wasFar = true;
          chunk.activeLod = lastLod;
          chunk.count = cnt;
          stats.drawnInstances += cnt;
          stats.lodCounts[lastLod] += cnt;
          stats.triangles += cnt * this.asset.lods[lastLod].triangleCount;
          stats.farPooled += cnt;
          continue;
        }
        if (chunk.wasFar) {
          // per-chunk buffer content is stale from before the chunk went far
          chunk.wasFar = false;
          chunk.staticDirty = true;
        }
      }

      if (forceAll || chunk.staticDirty) {
        const base = offsets[cell];
        const dst = chunk.array;
        let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
        let minY = Infinity, maxY = -Infinity, maxScale = 0;

        // one gather of one cache line per instance
        for (let k = 0; k < cnt; k++) {
          const so = sorted[base + k] * STRIDE;
          const dof = k * STRIDE;
          const x = data[so], y = data[so + 1], z = data[so + 2];
          dst[dof] = x; dst[dof + 1] = y; dst[dof + 2] = z; dst[dof + 3] = data[so + 3];
          dst[dof + 4] = data[so + 4]; dst[dof + 5] = data[so + 5];
          dst[dof + 6] = data[so + 6]; dst[dof + 7] = data[so + 7];
          dst[dof + 8] = data[so + 8]; dst[dof + 9] = data[so + 9];
          dst[dof + 10] = data[so + 10]; dst[dof + 11] = data[so + 11];
          dst[dof + 12] = data[so + 12]; dst[dof + 13] = data[so + 13];
          dst[dof + 14] = data[so + 14]; dst[dof + 15] = data[so + 15];
          if (x < minX) minX = x; if (x > maxX) maxX = x;
          if (y < minY) minY = y; if (y > maxY) maxY = y;
          if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
          const s = data[so + OFF.scale];
          if (s > maxScale) maxScale = s;
        }

        stats.uploadBytes += cnt * STRIDE * 4;
        if (chunk.staticDirty) stats.dirtyChunks++;
        chunk.buffer.clearUpdateRanges();
        chunk.buffer.addUpdateRange(0, cnt * STRIDE);
        chunk.buffer.needsUpdate = true;
        chunk.staticDirty = false;

        const r = instanceRadius * maxScale;
        const dx = maxX - minX, dy = maxY - minY, dz = maxZ - minZ;
        chunk.sphere.center.set(
          (minX + maxX) * 0.5,
          (minY + maxY) * 0.5 + instanceHeight * 0.5 * maxScale,
          (minZ + maxZ) * 0.5,
        );
        chunk.sphere.radius = Math.sqrt(dx * dx + dy * dy + dz * dz) * 0.5 + r;
      }

      const centreDist = this._cameraPos.distanceTo(chunk.sphere.center);
      const d = Math.max(0, centreDist - (lodFromEdge ? chunk.sphere.radius : instanceRadius)) * lodBias;
      let lod = lodDist.length;
      for (let l = 0; l < lodDist.length; l++) { if (d < lodDist[l]) { lod = l; break; } }
      if (lod >= chunk.meshes.length) lod = chunk.meshes.length - 1;

      for (let l = 0; l < chunk.meshes.length; l++) {
        const visible = l === lod;
        chunk.meshes[l].visible = visible;
        chunk.geometries[l].instanceCount = visible ? cnt : 0;
      }
      chunk.meshes[lod].castShadow = castShadow && lod <= shadowMaxLod;
      for (const m of chunk.attachmentMeshes) {
        m.visible = m.userData.maxLod >= lod;
        m.geometry.instanceCount = m.visible ? cnt : 0;
      }
      chunk.activeLod = lod;
      chunk.count = cnt;

      stats.activeChunks++;
      stats.drawnInstances += cnt;
      stats.lodCounts[lod] += cnt;
      stats.triangles += cnt * this.asset.lods[lod].triangleCount;
    }

    if (poolEnabled) {
      this._poolGeometry.instanceCount = poolCursor;
      this._poolMesh.visible = poolCursor > 0;
      if (poolRebuild && poolCursor > 0) {
        this._poolBuffer.clearUpdateRanges();
        this._poolBuffer.addUpdateRange(0, poolCursor * STRIDE);
        this._poolBuffer.needsUpdate = true;
        stats.uploadBytes += poolCursor * STRIDE * 4;
      }
      this._poolStale = false;
      if (poolCursor > 0) stats.activeChunks++;
    } else if (this._poolMesh) {
      this._poolMesh.visible = false;
      this._poolGeometry.instanceCount = 0;
    }

    this.transformsDirty = false;
    stats.chunks = liveChunks;
    stats.instances = n;
    stats.copyMs = performance.now() - t1;
  }

  _wrapTime() {
    const shift = this.options.timeWrap;
    const d = this.data;
    for (let i = 0; i < this.capacity; i++) {
      const o = i * STRIDE;
      d[o + OFF.phaseA] = wrap01(d[o + OFF.phaseA] + shift * d[o + OFF.rateA]);
      if (d[o + OFF.clipB] >= 0) {
        d[o + OFF.phaseB] = wrap01(d[o + OFF.phaseB] + shift * d[o + OFF.rateB]);
        d[o + OFF.fadeStart] -= shift;
      }
      this.animDirty[i] = 1;
    }
    this.time -= shift;
  }

  // --------------------------------------------------------- attachments ---

  addAttachment(attachment) {
    this.attachments.push(attachment);
    attachment.bind(this);
    for (const c of this.chunks) if (c) attachment.attachChunk(c);
    return attachment;
  }

  setDebugLod(on) { this.uniforms.uDebugLod.value = on ? 1 : 0; }

  dispose() {
    if (this._poolGeometry) { this.remove(this._poolMesh); disposeKeepingShared(this._poolGeometry); }
    for (const c of this.chunks) if (c) c.dispose();
    for (const m of this.materials) m.dispose();
    for (const m of this.depthMaterials) m.dispose();
    for (const a of this.attachments) a.dispose();
  }
}

function wrap01(v) { return v - Math.floor(v); }
