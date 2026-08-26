// Small rigid props for the socket-attachment demo. Authored around the origin;
// VATAttachment's socket offset puts them where the bone is.

import {
  BufferGeometry,
  ConeGeometry,
  CylinderGeometry,
  Float32BufferAttribute,
  Matrix4,
  SphereGeometry,
  TorusGeometry,
} from 'three';

function mergeSimple(parts) {
  let vCount = 0, iCount = 0;
  for (const { geo } of parts) {
    vCount += geo.attributes.position.count;
    iCount += geo.index ? geo.index.count : geo.attributes.position.count;
  }
  const position = new Float32Array(vCount * 3);
  const normal = new Float32Array(vCount * 3);
  const uv = new Float32Array(vCount * 2);
  const index = new Uint16Array(iCount);

  let vo = 0, io = 0;
  for (const { geo, matrix } of parts) {
    const g = geo.clone();
    if (matrix) g.applyMatrix4(matrix);
    const p = g.attributes.position.array;
    const n = g.attributes.normal.array;
    const t = g.attributes.uv ? g.attributes.uv.array : null;
    const c = g.attributes.position.count;
    position.set(p, vo * 3);
    normal.set(n, vo * 3);
    if (t) uv.set(t, vo * 2);
    if (g.index) {
      const gi = g.index.array;
      for (let k = 0; k < gi.length; k++) index[io + k] = gi[k] + vo;
      io += gi.length;
    } else {
      for (let k = 0; k < c; k++) index[io + k] = vo + k;
      io += c;
    }
    vo += c;
    g.dispose();
  }

  const out = new BufferGeometry();
  out.setAttribute('position', new Float32BufferAttribute(position, 3));
  out.setAttribute('normal', new Float32BufferAttribute(normal, 3));
  out.setAttribute('uv', new Float32BufferAttribute(uv, 2));
  out.setIndex(Array.from(index));
  return out;
}

/** A flat cap. ~120 triangles. */
export function makeCapGeometry() {
  const crown = new SphereGeometry(0.115, 12, 6, 0, Math.PI * 2, 0, Math.PI * 0.5);
  const brim = new CylinderGeometry(0.175, 0.175, 0.018, 14, 1);
  return mergeSimple([
    { geo: crown, matrix: new Matrix4().makeTranslation(0, 0.012, 0) },
    { geo: brim, matrix: new Matrix4().makeTranslation(0, 0.006, 0.055) },
  ]);
}

/** A balloon on a short string, meant for a hand socket. */
export function makeBalloonGeometry() {
  const body = new SphereGeometry(0.16, 12, 9);
  const knot = new ConeGeometry(0.035, 0.06, 6);
  const string = new CylinderGeometry(0.006, 0.006, 0.42, 4);
  return mergeSimple([
    { geo: body, matrix: new Matrix4().makeTranslation(0, 0.62, 0) },
    { geo: knot, matrix: new Matrix4().makeTranslation(0, 0.45, 0) },
    { geo: string, matrix: new Matrix4().makeTranslation(0, 0.21, 0) },
  ]);
}

/** A ring you can hang off anything; handy for checking socket alignment. */
export function makeRingGeometry() {
  return new TorusGeometry(0.09, 0.018, 6, 12);
}
