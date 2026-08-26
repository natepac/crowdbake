// Tiny binary container: one .bin with 4-byte-aligned sections, described by the
// sibling .json manifest. One fetch for geometry + animation textures.

const TYPE_OF = new Map([
  [Float32Array, 'f32'],
  [Uint32Array, 'u32'],
  [Uint16Array, 'u16'],
  [Uint8Array, 'u8'],
  [Int16Array, 'i16'],
]);

export class BinaryPacker {
  constructor() {
    this.sections = {};
    this.chunks = [];
    this.offset = 0;
  }

  add(name, array, extra = {}) {
    const type = extra.type || TYPE_OF.get(array.constructor);
    if (!type) throw new Error('unsupported array type for section ' + name);
    const bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
    const pad = (4 - (this.offset % 4)) % 4;
    if (pad) { this.chunks.push(new Uint8Array(pad)); this.offset += pad; }
    this.sections[name] = {
      offset: this.offset,
      byteLength: bytes.byteLength,
      count: array.length,
      type,
      ...extra,
    };
    this.chunks.push(bytes);
    this.offset += bytes.byteLength;
    return this.sections[name];
  }

  has(name) { return name in this.sections; }

  build() {
    const out = new Uint8Array(this.offset);
    let o = 0;
    for (const c of this.chunks) { out.set(c, o); o += c.byteLength; }
    return out;
  }
}

export function humanBytes(n) {
  if (n < 1024) return n + ' B';
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
  return (n / 1024 / 1024).toFixed(2) + ' MB';
}
