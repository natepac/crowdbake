// Octahedral encoding of unit vectors into two components in [-1, 1].
// Halves normal storage vs. xyz and survives fp16 easily (worst case ~0.5 degrees).

export function octEncode(x, y, z, out, o = 0) {
  const l = Math.abs(x) + Math.abs(y) + Math.abs(z);
  if (l === 0) { out[o] = 0; out[o + 1] = 0; return out; }
  const inv = 1 / l;
  let px = x * inv, py = y * inv;
  const pz = z * inv;
  if (pz < 0) {
    const sx = px >= 0 ? 1 : -1;
    const sy = py >= 0 ? 1 : -1;
    const nx = (1 - Math.abs(py)) * sx;
    const ny = (1 - Math.abs(px)) * sy;
    px = nx; py = ny;
  }
  out[o] = px; out[o + 1] = py;
  return out;
}

export function octDecode(ex, ey, out, o = 0) {
  let x = ex, y = ey;
  let z = 1 - Math.abs(x) - Math.abs(y);
  const t = Math.max(-z, 0);
  x += x >= 0 ? -t : t;
  y += y >= 0 ? -t : t;
  const l = Math.hypot(x, y, z) || 1;
  out[o] = x / l; out[o + 1] = y / l; out[o + 2] = z / l;
  return out;
}
