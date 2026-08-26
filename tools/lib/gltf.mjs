// Minimal, dependency-free glTF 2.0 / GLB reader.
// Only reads what the VAT baker needs: node hierarchy, skins, skinned mesh
// primitives and animation samplers. No textures, no materials, no Draco.

import fs from 'node:fs';
import path from 'node:path';

const COMP = {
  5120: { array: Int8Array, size: 1, norm: 127 },
  5121: { array: Uint8Array, size: 1, norm: 255 },
  5122: { array: Int16Array, size: 2, norm: 32767 },
  5123: { array: Uint16Array, size: 2, norm: 65535 },
  5125: { array: Uint32Array, size: 4, norm: 4294967295 },
  5126: { array: Float32Array, size: 4, norm: 1 },
};

const TYPE_COUNT = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16 };

function decodeDataURI(uri) {
  const comma = uri.indexOf(',');
  const meta = uri.slice(5, comma);
  const data = uri.slice(comma + 1);
  if (meta.endsWith(';base64')) {
    const buf = Buffer.from(data, 'base64');
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  }
  const buf = Buffer.from(decodeURIComponent(data), 'binary');
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

export function readGLTF(file) {
  const dir = path.dirname(file);
  const raw = fs.readFileSync(file);
  let json, glbBin = null;

  if (raw.length >= 4 && raw.readUInt32LE(0) === 0x46546c67) {
    // GLB container
    const total = raw.readUInt32LE(8);
    let off = 12;
    while (off < Math.min(total, raw.length)) {
      const len = raw.readUInt32LE(off);
      const type = raw.readUInt32LE(off + 4);
      const body = raw.subarray(off + 8, off + 8 + len);
      if (type === 0x4e4f534a) json = JSON.parse(body.toString('utf8'));
      else if (type === 0x004e4942) glbBin = body.buffer.slice(body.byteOffset, body.byteOffset + body.byteLength);
      off += 8 + len + ((4 - (len % 4)) % 4 === 4 ? 0 : 0);
      off += (4 - (len % 4)) % 4;
    }
  } else {
    json = JSON.parse(raw.toString('utf8'));
  }
  if (!json) throw new Error(`could not parse glTF JSON from ${file}`);

  const buffers = (json.buffers || []).map((b, i) => {
    if (b.uri == null) {
      if (!glbBin) throw new Error('glTF buffer without uri and no GLB chunk');
      return glbBin;
    }
    if (b.uri.startsWith('data:')) return decodeDataURI(b.uri);
    const p = path.resolve(dir, decodeURIComponent(b.uri));
    const buf = fs.readFileSync(p);
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  });

  const readAccessor = (index, { asInt = false } = {}) => {
    if (index == null) return null;
    const acc = json.accessors[index];
    const comps = TYPE_COUNT[acc.type];
    const count = acc.count;
    const info = COMP[acc.componentType];
    const OutArray = asInt ? Uint32Array : Float32Array;
    const out = new OutArray(count * comps);

    if (acc.bufferView != null) {
      const view = json.bufferViews[acc.bufferView];
      const bufAB = buffers[view.buffer];
      const base = (view.byteOffset || 0) + (acc.byteOffset || 0);
      const stride = view.byteStride || comps * info.size;
      const dv = new DataView(bufAB);
      const getters = {
        5120: (o) => dv.getInt8(o), 5121: (o) => dv.getUint8(o),
        5122: (o) => dv.getInt16(o, true), 5123: (o) => dv.getUint16(o, true),
        5125: (o) => dv.getUint32(o, true), 5126: (o) => dv.getFloat32(o, true),
      };
      const get = getters[acc.componentType];
      const denom = acc.normalized ? info.norm : 1;
      for (let i = 0; i < count; i++) {
        const o = base + i * stride;
        for (let c = 0; c < comps; c++) {
          const v = get(o + c * info.size);
          out[i * comps + c] = acc.normalized && !asInt ? Math.max(v / denom, -1) : v;
        }
      }
    }

    if (acc.sparse) {
      const s = acc.sparse;
      const idxView = json.bufferViews[s.indices.bufferView];
      const idxInfo = COMP[s.indices.componentType];
      const idxDV = new DataView(buffers[idxView.buffer]);
      const idxBase = (idxView.byteOffset || 0) + (s.indices.byteOffset || 0);
      const valView = json.bufferViews[s.values.bufferView];
      const valDV = new DataView(buffers[valView.buffer]);
      const valBase = (valView.byteOffset || 0) + (s.values.byteOffset || 0);
      const readIdx = (i) => s.indices.componentType === 5125 ? idxDV.getUint32(idxBase + i * 4, true)
        : s.indices.componentType === 5123 ? idxDV.getUint16(idxBase + i * 2, true)
        : idxDV.getUint8(idxBase + i);
      const readVal = (i) => acc.componentType === 5126 ? valDV.getFloat32(valBase + i * 4, true)
        : acc.componentType === 5123 ? valDV.getUint16(valBase + i * 2, true)
        : acc.componentType === 5125 ? valDV.getUint32(valBase + i * 4, true)
        : acc.componentType === 5122 ? valDV.getInt16(valBase + i * 2, true)
        : acc.componentType === 5121 ? valDV.getUint8(valBase + i)
        : valDV.getInt8(valBase + i);
      for (let i = 0; i < s.count; i++) {
        const target = readIdx(i);
        for (let c = 0; c < comps; c++) {
          const v = readVal(i * comps + c);
          out[target * comps + c] = acc.normalized && !asInt ? Math.max(v / info.norm, -1) : v;
        }
      }
      void idxInfo;
    }
    return out;
  };

  return { json, buffers, readAccessor };
}
