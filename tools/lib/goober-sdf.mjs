// CPU port of the goobers "SDF blend-shell" vertex shader (goobers.html VERT +
// SDF_CHUNK). Given a frame's prim list and the template mesh, produces the
// same snapped vertex positions / normals / colors the GPU would.
//
// Ported faithfully, including the ORDER-DEPENDENT smooth-min chain (kc carries
// the previous blend radius) and central-difference gradients. Any change here
// diverges from what the original app renders.

/**
 * prims: flat Float32Array, 12 floats per prim:
 *   [ax, ay, az, ra,  bx, by, bz, rb,  cr, cg, cb, k]
 */
export function makeField(prims, count) {
  const P = prims;

  function sdPrim(px, py, pz, o) {
    const ax = P[o], ay = P[o + 1], az = P[o + 2], r1 = P[o + 3];
    const bx = P[o + 4], by = P[o + 5], bz = P[o + 6], r2 = P[o + 7];
    const bax = bx - ax, bay = by - ay, baz = bz - az;
    const l2 = bax * bax + bay * bay + baz * baz;
    const pax = px - ax, pay = py - ay, paz = pz - az;
    if (l2 < 1e-8) return Math.sqrt(pax * pax + pay * pay + paz * paz) - r1;
    const rr = r1 - r2;
    const a2 = l2 - rr * rr;
    const il2 = 1 / l2;
    const y = pax * bax + pay * bay + paz * baz;
    const z = y - l2;
    const qx = pax * l2 - bax * y, qy = pay * l2 - bay * y, qz = paz * l2 - baz * y;
    const x2 = qx * qx + qy * qy + qz * qz;
    const y2 = y * y * l2, z2 = z * z * l2;
    const k = Math.sign(rr) * rr * rr * x2;
    if (Math.sign(z) * a2 * z2 > k) return Math.sqrt(x2 + z2) * il2 - r2;
    if (Math.sign(y) * a2 * y2 < k) return Math.sqrt(x2 + y2) * il2 - r1;
    return (Math.sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
  }

  function mapDist(px, py, pz) {
    let d = 1e9, kc = 0.2;
    for (let i = 0; i < count; i++) {
      const o = i * 12;
      const di = sdPrim(px, py, pz, o);
      const ki = P[o + 11];
      const kp = Math.max(Math.min(kc, ki), 1e-3);
      let h = 0.5 + 0.5 * (d - di) / kp;
      if (h < 0) h = 0; else if (h > 1) h = 1;
      d = d + (di - d) * h - kp * h * (1 - h);
      kc = kc + (ki - kc) * h;
    }
    return d;
  }

  const colOut = new Float32Array(3);
  function mapColor(px, py, pz) {
    let d = 1e9, kc = 0.2;
    let cr = 1, cg = 1, cb = 1;
    for (let i = 0; i < count; i++) {
      const o = i * 12;
      const di = sdPrim(px, py, pz, o);
      const ki = P[o + 11];
      const kp = Math.max(Math.min(kc, ki), 1e-3);
      let h = 0.5 + 0.5 * (d - di) / kp;
      if (h < 0) h = 0; else if (h > 1) h = 1;
      d = d + (di - d) * h - kp * h * (1 - h);
      cr = cr + (P[o + 8] - cr) * h;
      cg = cg + (P[o + 9] - cg) * h;
      cb = cb + (P[o + 10] - cb) * h;
      kc = kc + (ki - kc) * h;
    }
    colOut[0] = cr; colOut[1] = cg; colOut[2] = cb;
    return colOut;
  }

  function mapMin(px, py, pz) {
    let d = 1e9;
    for (let i = 0; i < count; i++) {
      const di = sdPrim(px, py, pz, i * 12);
      if (di < d) d = di;
    }
    return d;
  }

  const grad = new Float32Array(3);
  function mapGrad(px, py, pz, e) {
    const gx = mapDist(px + e, py, pz) - mapDist(px - e, py, pz);
    const gy = mapDist(px, py + e, pz) - mapDist(px, py - e, pz);
    const gz = mapDist(px, py, pz + e) - mapDist(px, py, pz - e);
    const l = Math.sqrt(gx * gx + gy * gy + gz * gz);
    if (l > 1e-6) { grad[0] = gx / l; grad[1] = gy / l; grad[2] = gz / l; }
    else { grad[0] = 0; grad[1] = 1; grad[2] = 0; }
    return grad;
  }

  return { sdPrim, mapDist, mapColor, mapMin, mapGrad };
}

function smoothstep(a, b, x) {
  let t = (x - a) / (b - a || 1e-9);
  if (t < 0) t = 0; else if (t > 1) t = 1;
  return t * t * (3 - 2 * t);
}

const clampv = (v, lim) => (v > lim ? lim : v < -lim ? -lim : v);

/**
 * Snap every template vertex onto the blend-shell surface for one frame.
 *
 * template: { pos: Float32Array(v*3), aPrim: Uint16Array(v) }
 * consts:   { tuckDepth, tuck0, tuck1, gradEps, stepMax, offset }
 * outColor may be null after the first frame (colors are effectively static).
 */
export function snapFrame(template, prims, count, consts, outPos, outNrm, outColor) {
  const { mapDist, mapColor, mapMin, mapGrad, sdPrim } = makeField(prims, count);
  const { tuckDepth, tuck0, tuck1, gradEps, stepMax, offset } = consts;
  const tp = template.pos;
  const ap = template.aPrim;
  const n = ap.length;

  for (let v = 0; v < n; v++) {
    const pi = ap[v];
    const o = pi * 12;
    const A0 = prims[o], A1 = prims[o + 1], A2 = prims[o + 2], Aw = prims[o + 3];
    const B0 = prims[o + 4], B1 = prims[o + 5], B2 = prims[o + 6], Bw = prims[o + 7];

    // frame around the prim axis (shader's `basis` + right-handed rebuild)
    let axx = B0 - A0, axy = B1 - A1, axz = B2 - A2;
    const len = Math.sqrt(axx * axx + axy * axy + axz * axz);
    if (len > 1e-6) { axx /= len; axy /= len; axz /= len; }
    else { axx = 0; axy = 1; axz = 0; }
    const s = axz >= 0 ? 1 : -1;
    const a = -1 / (s + axz);
    const k = axx * axy * a;
    const n1x = 1 + s * axx * axx * a, n1y = s * k, n1z = -s * axx;
    // n2 = cross(n1, ax)
    const n2x = n1y * axz - n1z * axy;
    const n2y = n1z * axx - n1x * axz;
    const n2z = n1x * axy - n1y * axx;

    const lx = tp[v * 3], ly = tp[v * 3 + 1], lz = tp[v * 3 + 2];
    const yc = ly < -1 ? -1 : ly > 1 ? 1 : ly;
    const t = yc * 0.5 + 0.5;
    const r = Aw + (Bw - Aw) * t;
    let px = A0 + (B0 - A0) * t + axx * ((ly - yc) * r) + (lx * n1x + lz * n2x) * r;
    let py = A1 + (B1 - A1) * t + axy * ((ly - yc) * r) + (lx * n1y + lz * n2y) * r;
    let pz = A2 + (B2 - A2) * t + axz * ((ly - yc) * r) + (lx * n1z + lz * n2z) * r;

    const dOwn = sdPrim(px, py, pz, o);
    const dMin = mapMin(px, py, pz);
    const tuck = tuckDepth * smoothstep(tuck0, tuck1, dOwn - dMin);
    const target = offset - tuck;

    let gx = 0, gy = 1, gz = 0;
    for (let it = 0; it < 3; it++) {
      const d = mapDist(px, py, pz);
      const g = mapGrad(px, py, pz, gradEps);
      gx = g[0]; gy = g[1]; gz = g[2];
      const step = clampv((d - target) * 0.9, stepMax);
      px -= gx * step; py -= gy * step; pz -= gz * step;
    }
    const fstep = clampv(mapDist(px, py, pz) - target, stepMax);
    px -= gx * fstep; py -= gy * fstep; pz -= gz * fstep;

    outPos[v * 3] = px; outPos[v * 3 + 1] = py; outPos[v * 3 + 2] = pz;

    const g = mapGrad(px, py, pz, gradEps * 0.75);
    outNrm[v * 3] = g[0]; outNrm[v * 3 + 1] = g[1]; outNrm[v * 3 + 2] = g[2];

    if (outColor) {
      const c = mapColor(px, py, pz);
      outColor[v * 3] = c[0]; outColor[v * 3 + 1] = c[1]; outColor[v * 3 + 2] = c[2];
    }
  }
}

/** Same capsule template the app builds (goobers.html capsuleTemplate). */
export function capsuleTemplate(radial, hSeg, capSeg) {
  const TAU = Math.PI * 2;
  const rings = [];
  for (let i = 0; i <= capSeg; i++) { const ph = -Math.PI / 2 + (i / capSeg) * (Math.PI / 2); rings.push([Math.cos(ph), -1 + Math.sin(ph)]); }
  for (let i = 1; i <= hSeg; i++) rings.push([1, -1 + 2 * i / hSeg]);
  for (let i = 1; i <= capSeg; i++) { const ph = (i / capSeg) * (Math.PI / 2); rings.push([Math.cos(ph), 1 + Math.sin(ph)]); }
  const pos = [], idx = [];
  for (const [r, y] of rings) for (let s2 = 0; s2 < radial; s2++) { const th = s2 / radial * TAU; pos.push(Math.cos(th) * r, y, Math.sin(th) * r); }
  const R = rings.length;
  for (let j = 0; j < R - 1; j++) for (let s2 = 0; s2 < radial; s2++) {
    const a = j * radial + s2, b = j * radial + (s2 + 1) % radial, c = (j + 1) * radial + s2, d = (j + 1) * radial + (s2 + 1) % radial;
    idx.push(a, c, b, b, c, d);
  }
  return { pos, idx };
}
