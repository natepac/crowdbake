// LOD generation by vertex clustering.
//
// Why clustering and not quadric-error collapse: a VAT LOD mesh must address the
// SAME animation texture as LOD0, so every surviving vertex has to keep its
// original vertex id (its texel column). Clustering picks representatives from
// the existing vertex set, so an LOD is just a different index buffer plus a
// list of which original vertex ids it uses -- no extra texture memory at all.
//
// Cluster keys include the dominant bone and the material id, which stops the
// grid from welding the two legs together or smearing the shirt into the skin.

export function buildLOD(model, ratio) {
  const { vertexCount, positions, indices, skinIndex, skinWeight } = model;
  const materialIds = model.materialIds || null;

  if (ratio >= 0.999) {
    const vids = new Uint32Array(vertexCount);
    for (let i = 0; i < vertexCount; i++) vids[i] = i;
    return { vids, indices: Uint32Array.from(indices), ratio: 1 };
  }

  let min = [Infinity, Infinity, Infinity];
  let max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < vertexCount; i++) {
    for (let c = 0; c < 3; c++) {
      const v = positions[i * 3 + c];
      if (v < min[c]) min[c] = v;
      if (v > max[c]) max[c] = v;
    }
  }
  const size = [max[0] - min[0] || 1, max[1] - min[1] || 1, max[2] - min[2] || 1];
  const target = Math.max(24, Math.round(vertexCount * ratio));

  const dominantBone = new Uint16Array(vertexCount);
  for (let v = 0; v < vertexCount; v++) {
    let best = 0, bw = -1;
    for (let k = 0; k < 4; k++) {
      const w = skinWeight[v * 4 + k];
      if (w > bw) { bw = w; best = skinIndex[v * 4 + k]; }
    }
    dominantBone[v] = best;
  }

  const cluster = (grid) => {
    const map = new Map();
    const inv = [grid / size[0], grid / size[1], grid / size[2]];
    for (let v = 0; v < vertexCount; v++) {
      const cx = Math.min(grid - 1, Math.floor((positions[v * 3] - min[0]) * inv[0]));
      const cy = Math.min(grid - 1, Math.floor((positions[v * 3 + 1] - min[1]) * inv[1]));
      const cz = Math.min(grid - 1, Math.floor((positions[v * 3 + 2] - min[2]) * inv[2]));
      const mat = materialIds ? materialIds[v] | 0 : 0;
      const key = ((((cz * grid + cy) * grid + cx) * 256 + dominantBone[v]) * 8) + mat;
      let bucket = map.get(key);
      if (!bucket) { bucket = []; map.set(key, bucket); }
      bucket.push(v);
    }
    return map;
  };

  // binary search a grid resolution that lands near the target vertex count
  let lo = 2, hi = 96, map = cluster(24);
  for (let it = 0; it < 9 && lo < hi; it++) {
    const mid = (lo + hi) >> 1;
    map = cluster(mid);
    if (map.size > target) hi = mid; else lo = mid + 1;
  }
  map = cluster(Math.max(2, Math.min(96, lo)));

  // representative = vertex closest to its cluster centroid
  const repOf = new Uint32Array(vertexCount);
  const reps = [];
  for (const bucket of map.values()) {
    let cx = 0, cy = 0, cz = 0;
    for (const v of bucket) { cx += positions[v * 3]; cy += positions[v * 3 + 1]; cz += positions[v * 3 + 2]; }
    cx /= bucket.length; cy /= bucket.length; cz /= bucket.length;
    let best = bucket[0], bd = Infinity;
    for (const v of bucket) {
      const d = (positions[v * 3] - cx) ** 2 + (positions[v * 3 + 1] - cy) ** 2 + (positions[v * 3 + 2] - cz) ** 2;
      if (d < bd) { bd = d; best = v; }
    }
    for (const v of bucket) repOf[v] = best;
    reps.push(best);
  }

  // rebuild triangles, dropping degenerate + duplicate faces
  const localOf = new Map();
  const vids = [];
  const out = [];
  const seenTri = new Set();
  for (let i = 0; i < indices.length; i += 3) {
    const a = repOf[indices[i]], b = repOf[indices[i + 1]], c = repOf[indices[i + 2]];
    if (a === b || b === c || a === c) continue;
    const k1 = Math.min(a, b, c), k3 = Math.max(a, b, c), k2 = a + b + c - k1 - k3;
    const key = k1 + ',' + k2 + ',' + k3;
    if (seenTri.has(key)) continue;
    seenTri.add(key);
    for (const v of [a, b, c]) {
      let l = localOf.get(v);
      if (l === undefined) { l = vids.length; localOf.set(v, l); vids.push(v); }
      out.push(l);
    }
  }
  void reps;

  return {
    vids: Uint32Array.from(vids),
    indices: Uint32Array.from(out),
    ratio: vids.length / vertexCount,
  };
}
