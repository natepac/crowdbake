// float32 -> float16 (IEEE binary16) packing helpers.
// Uses the native Float16Array when the runtime has it (V8 13+/Node 24), otherwise
// falls back to the classic bit-twiddling routine (same math as three's DataUtils).

const HAS_NATIVE = typeof Float16Array !== 'undefined';

// Caveat on the fallback: toHalf() goes double -> float32 -> float16, so a value
// sitting a hair below the midpoint between two halves can be rounded up to the
// midpoint by the first step and then up again by the second. Native
// Float16Array rounds once and gets those cases right. The disagreement is one
// ULP on roughly 1 in 60,000 random values -- around 1e-7 absolute at unit
// magnitude, i.e. sub-micron on a baked character -- but it is why
// tools/test.mjs asserts the two paths agree to 1 ULP rather than bit-exactly.

const _f32 = new Float32Array(1);
const _i32 = new Int32Array(_f32.buffer);

export function toHalf(val) {
  _f32[0] = val;
  const x = _i32[0];
  let bits = (x >> 16) & 0x8000;
  let m = (x >> 12) & 0x07ff;
  const e = (x >> 23) & 0xff;
  if (e < 103) return bits;                       // underflow -> signed zero
  if (e > 142) {                                  // overflow -> inf / nan
    bits |= 0x7c00;
    bits |= (e === 255 ? 0 : 1) && (x & 0x007fffff);
    return bits;
  }
  if (e < 113) {                                  // subnormal
    m |= 0x0800;
    bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1);
    return bits;
  }
  bits |= ((e - 112) << 10) | (m >> 1);
  bits += m & 1;
  return bits;
}

export function fromHalf(h) {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

/** Float32Array (or plain array) -> Uint16Array of half floats. */
export function packHalfArray(src) {
  const n = src.length;
  const out = new Uint16Array(n);
  if (HAS_NATIVE) {
    const tmp = new Float16Array(n);
    tmp.set(src);
    out.set(new Uint16Array(tmp.buffer, tmp.byteOffset, n));
    return out;
  }
  for (let i = 0; i < n; i++) out[i] = toHalf(src[i]);
  return out;
}

/** Uint16Array of half floats -> Float32Array. Used by the verifier. */
export function unpackHalfArray(src) {
  const n = src.length;
  const out = new Float32Array(n);
  if (HAS_NATIVE) {
    const tmp = new Float16Array(src.buffer.slice(src.byteOffset, src.byteOffset + n * 2));
    out.set(tmp);
    return out;
  }
  for (let i = 0; i < n; i++) out[i] = fromHalf(src[i]);
  return out;
}

export const HAS_NATIVE_FLOAT16 = HAS_NATIVE;
