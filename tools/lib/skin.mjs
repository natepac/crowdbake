// CPU linear-blend skinning. Hot loop of the baker, so it is written flat:
// no Vector3/Matrix4 allocation, direct indexing into typed arrays.

/**
 * @param {Float32Array} skinMats  boneCount*16, column-major (three's toArray order)
 */
export function skinFrame(model, skinMats, outPos, outNrm, inTan, outTan) {
  const { vertexCount, positions, normals, skinIndex, skinWeight } = model;
  for (let v = 0; v < vertexCount; v++) {
    const o4 = v * 4;
    let m0 = 0, m1 = 0, m2 = 0, m4 = 0, m5 = 0, m6 = 0, m8 = 0, m9 = 0, m10 = 0, m12 = 0, m13 = 0, m14 = 0;
    for (let k = 0; k < 4; k++) {
      const w = skinWeight[o4 + k];
      if (w === 0) continue;
      const b = skinIndex[o4 + k] * 16;
      m0 += skinMats[b] * w;      m1 += skinMats[b + 1] * w;   m2 += skinMats[b + 2] * w;
      m4 += skinMats[b + 4] * w;  m5 += skinMats[b + 5] * w;   m6 += skinMats[b + 6] * w;
      m8 += skinMats[b + 8] * w;  m9 += skinMats[b + 9] * w;   m10 += skinMats[b + 10] * w;
      m12 += skinMats[b + 12] * w; m13 += skinMats[b + 13] * w; m14 += skinMats[b + 14] * w;
    }
    const o3 = v * 3;
    const px = positions[o3], py = positions[o3 + 1], pz = positions[o3 + 2];
    outPos[o3] = m0 * px + m4 * py + m8 * pz + m12;
    outPos[o3 + 1] = m1 * px + m5 * py + m9 * pz + m13;
    outPos[o3 + 2] = m2 * px + m6 * py + m10 * pz + m14;

    const nx = normals[o3], ny = normals[o3 + 1], nz = normals[o3 + 2];
    let ox = m0 * nx + m4 * ny + m8 * nz;
    let oy = m1 * nx + m5 * ny + m9 * nz;
    let oz = m2 * nx + m6 * ny + m10 * nz;
    let l = Math.hypot(ox, oy, oz) || 1;
    ox /= l; oy /= l; oz /= l;
    outNrm[o3] = ox; outNrm[o3 + 1] = oy; outNrm[o3 + 2] = oz;

    if (inTan && outTan) {
      const t4 = v * 4;
      const tx = inTan[t4], ty = inTan[t4 + 1], tz = inTan[t4 + 2];
      let sx = m0 * tx + m4 * ty + m8 * tz;
      let sy = m1 * tx + m5 * ty + m9 * tz;
      let sz = m2 * tx + m6 * ty + m10 * tz;
      // Gram-Schmidt against the animated normal so the encoded pair stays orthonormal
      const d = sx * ox + sy * oy + sz * oz;
      sx -= ox * d; sy -= oy * d; sz -= oz * d;
      l = Math.hypot(sx, sy, sz);
      if (l < 1e-6) { sx = Math.abs(ox) < 0.9 ? 1 : 0; sy = Math.abs(ox) < 0.9 ? 0 : 1; sz = 0; l = 1; }
      outTan[o3] = sx / l; outTan[o3 + 1] = sy / l; outTan[o3 + 2] = sz / l;
    }
  }
}

/** Per-vertex tangents from bind-pose UVs. Returns Float32Array(n*4), w = handedness. */
export function computeTangents(model) {
  const { vertexCount, positions, normals, uvs, indices } = model;
  const tan = new Float32Array(vertexCount * 3);
  const bit = new Float32Array(vertexCount * 3);
  const out = new Float32Array(vertexCount * 4);

  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i], b = indices[i + 1], c = indices[i + 2];
    const a3 = a * 3, b3 = b * 3, c3 = c * 3;
    const e1x = positions[b3] - positions[a3], e1y = positions[b3 + 1] - positions[a3 + 1], e1z = positions[b3 + 2] - positions[a3 + 2];
    const e2x = positions[c3] - positions[a3], e2y = positions[c3 + 1] - positions[a3 + 1], e2z = positions[c3 + 2] - positions[a3 + 2];
    const du1 = uvs[b * 2] - uvs[a * 2], dv1 = uvs[b * 2 + 1] - uvs[a * 2 + 1];
    const du2 = uvs[c * 2] - uvs[a * 2], dv2 = uvs[c * 2 + 1] - uvs[a * 2 + 1];
    const det = du1 * dv2 - du2 * dv1;
    if (Math.abs(det) < 1e-12) continue;
    const r = 1 / det;
    const tx = (e1x * dv2 - e2x * dv1) * r, ty = (e1y * dv2 - e2y * dv1) * r, tz = (e1z * dv2 - e2z * dv1) * r;
    const bx = (e2x * du1 - e1x * du2) * r, by = (e2y * du1 - e1y * du2) * r, bz = (e2z * du1 - e1z * du2) * r;
    for (const v3 of [a3, b3, c3]) {
      tan[v3] += tx; tan[v3 + 1] += ty; tan[v3 + 2] += tz;
      bit[v3] += bx; bit[v3 + 1] += by; bit[v3 + 2] += bz;
    }
  }

  for (let v = 0; v < vertexCount; v++) {
    const o = v * 3;
    const nx = normals[o], ny = normals[o + 1], nz = normals[o + 2];
    let tx = tan[o], ty = tan[o + 1], tz = tan[o + 2];
    const d = tx * nx + ty * ny + tz * nz;
    tx -= nx * d; ty -= ny * d; tz -= nz * d;
    let l = Math.hypot(tx, ty, tz);
    if (l < 1e-8) {
      // degenerate UVs: pick any vector orthogonal to the normal
      if (Math.abs(nx) < 0.9) { tx = 1 - nx * nx; ty = -nx * ny; tz = -nx * nz; }
      else { tx = -ny * nx; ty = 1 - ny * ny; tz = -ny * nz; }
      l = Math.hypot(tx, ty, tz) || 1;
    }
    const cx = ny * tz / l - nz * ty / l;
    const cy = nz * tx / l - nx * tz / l;
    const cz = nx * ty / l - ny * tx / l;
    const w = (cx * bit[o] + cy * bit[o + 1] + cz * bit[o + 2]) < 0 ? -1 : 1;
    out[v * 4] = tx / l; out[v * 4 + 1] = ty / l; out[v * 4 + 2] = tz / l; out[v * 4 + 3] = w;
  }
  return out;
}
