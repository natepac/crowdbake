// Socket attachments -- the thing pure VAT cannot do.
//
// A VAT vertex is baked geometry, so there is nothing to hang a hat off. The
// bone-matrix texture (bake with --bones) fixes that: one socket bone costs
// 3 texels per frame, so an attachment shader is 6 fetches and a matrix apply.
// The crowd body can stay in VAT mode while attachments read the bone texture,
// which is the cheapest combination -- the bone texture is only kilobytes.
//
// Attachments are instanced over the SAME per-chunk instance attributes as the
// bodies, so they cull, LOD and cross-fade with the crowd for free. Instances
// whose packed accessory id does not match collapse to a degenerate triangle.

import {
  InstancedBufferGeometry,
  Matrix4,
  Mesh,
  MeshDepthMaterial,
  MeshStandardMaterial,
  RGBADepthPacking,
} from 'three';

const ATTACH_GLSL = /* glsl */`
precision highp sampler2D;

uniform highp sampler2D vatBoneTex;
uniform vec4  vatClips[VAT_MAX_CLIPS];
uniform float vatTime;
uniform float vatInvFade;
uniform int   uSocketBone;
uniform mat4  uSocketOffset;
uniform float uAccessory;

attribute vec4 iXform;
attribute vec4 iAnimA;
attribute vec4 iAnimB;
attribute vec4 iTint;

void attFrames(int clipIdx, float phase, out int f0, out int f1, out float alpha) {
  vec4 c = vatClips[clipIdx];
  float fc = max(c.y, 1.0);
  float t = fract(phase) * fc;
  float i0 = floor(t);
  alpha = t - i0;
  float i1 = c.z > 0.5 ? mod(i0 + 1.0, fc) : min(i0 + 1.0, fc - 1.0);
  f0 = int(c.x + i0);
  f1 = int(c.x + i1);
}

void attRows(int frame, out vec4 r0, out vec4 r1, out vec4 r2) {
  int x = uSocketBone * 3;
  r0 = texelFetch(vatBoneTex, ivec2(x, frame), 0);
  r1 = texelFetch(vatBoneTex, ivec2(x + 1, frame), 0);
  r2 = texelFetch(vatBoneTex, ivec2(x + 2, frame), 0);
}

void attSocket(int clipIdx, float phase, out vec4 r0, out vec4 r1, out vec4 r2) {
  int f0, f1; float a;
  attFrames(clipIdx, phase, f0, f1, a);
  vec4 a0, a1, a2, b0, b1, b2;
  attRows(f0, a0, a1, a2);
  attRows(f1, b0, b1, b2);
  r0 = mix(a0, b0, a); r1 = mix(a1, b1, a); r2 = mix(a2, b2, a);
}

void attEvaluate(out vec3 outPos, out vec3 outNrm) {
  vec4 r0, r1, r2;
  attSocket(int(iAnimA.x), fract(iAnimA.y + vatTime * iAnimA.z), r0, r1, r2);

  #ifdef VAT_CROSSFADE
  if (iAnimB.x >= 0.0) {
    float blend = clamp((vatTime - iAnimB.w) * vatInvFade, 0.0, 1.0);
    if (blend > 0.001) {
      vec4 s0, s1, s2;
      attSocket(int(iAnimB.x), fract(iAnimB.y + vatTime * iAnimB.z), s0, s1, s2);
      r0 = mix(r0, s0, blend); r1 = mix(r1, s1, blend); r2 = mix(r2, s2, blend);
    }
  }
  #endif

  vec4 lp = uSocketOffset * vec4(position, 1.0);
  vec3 ln = mat3(uSocketOffset) * normal;
  outPos = vec3(dot(r0, lp), dot(r1, lp), dot(r2, lp));
  outNrm = normalize(vec3(dot(r0.xyz, ln), dot(r1.xyz, ln), dot(r2.xyz, ln)));

  // instances that are not wearing this accessory collapse to zero area
  float acc = floor(mod(iTint.w / 64.0, 8.0));
  outPos *= step(abs(acc - uAccessory), 0.5);

  outPos *= iAnimA.w;
  float s = sin(iXform.w), c = cos(iXform.w);
  outPos = vec3(c * outPos.x + s * outPos.z, outPos.y, -s * outPos.x + c * outPos.z);
  outNrm = vec3(c * outNrm.x + s * outNrm.z, outNrm.y, -s * outNrm.x + c * outNrm.z);
  outPos += iXform.xyz;
}
`;

function inject(shader, uniforms, own, opts) {
  Object.assign(shader.uniforms, uniforms, own);
  shader.vertexShader = ATTACH_GLSL + shader.vertexShader;
  const compute = '\n  vec3 attPos, attNrm;\n  attEvaluate(attPos, attNrm);\n';
  if (opts.normalStage) {
    shader.vertexShader = shader.vertexShader
      .replace('#include <beginnormal_vertex>', compute + '  vec3 objectNormal = attNrm;')
      .replace('#include <begin_vertex>', '  vec3 transformed = attPos;');
  } else {
    shader.vertexShader = shader.vertexShader
      .replace('#include <begin_vertex>', compute + '  vec3 transformed = attPos;');
  }
}

export class VATAttachment {
  /**
   * @param {BufferGeometry} geometry small rigid prop, authored around the socket bone
   * @param {object} opts { socket, offset, accessory, maxLod, color, roughness, metalness }
   */
  constructor(geometry, opts = {}) {
    this.source = geometry;
    this.options = {
      socket: 'head', offset: null, accessory: 1, maxLod: 1,
      color: 0xffffff, roughness: 0.8, metalness: 0, castShadow: true,
      ...opts,
    };
    this.crowd = null;
    this.material = null;
    this.depthMaterial = null;
  }

  bind(crowd) {
    this.crowd = crowd;
    const asset = crowd.asset;
    if (!asset.boneTex) {
      throw new Error('VATAttachment needs a bake made with --bones (no bone texture in this asset)');
    }
    let bone = this.options.socket;
    if (typeof bone === 'string') {
      const fromSockets = asset.sockets ? asset.sockets[bone] : undefined;
      bone = fromSockets !== undefined ? fromSockets : asset.boneNames.indexOf(bone);
    }
    if (bone === undefined || bone < 0) {
      throw new Error(`VATAttachment: unknown socket "${this.options.socket}"`);
    }
    this.boneIndex = bone;

    const own = {
      uSocketBone: { value: bone },
      uSocketOffset: { value: this.options.offset || new Matrix4() },
      uAccessory: { value: this.options.accessory },
    };
    this.own = own;

    const defines = {
      VAT_MAX_CLIPS: Math.max(1, asset.clips.length),
    };
    if (crowd.options.crossFade) defines.VAT_CROSSFADE = '';

    this.material = new MeshStandardMaterial({
      color: this.options.color,
      roughness: this.options.roughness,
      metalness: this.options.metalness,
    });
    this.material.defines = defines;
    this.material.onBeforeCompile = (s) => inject(s, crowd.uniforms, own, { normalStage: true });
    this.material.customProgramCacheKey = () => 'vatatt|' + JSON.stringify(defines);

    this.depthMaterial = new MeshDepthMaterial({ depthPacking: RGBADepthPacking });
    this.depthMaterial.defines = defines;
    this.depthMaterial.onBeforeCompile = (s) => inject(s, crowd.uniforms, own, { normalStage: false });
    this.depthMaterial.customProgramCacheKey = () => 'vatattdepth|' + JSON.stringify(defines);
    return this;
  }

  attachChunk(chunk) {
    const geom = new InstancedBufferGeometry();
    for (const [name, attr] of Object.entries(this.source.attributes)) geom.setAttribute(name, attr);
    if (this.source.index) geom.setIndex(this.source.index);
    for (const [name, attr] of Object.entries(chunk.attrs)) geom.setAttribute(name, attr);
    geom.instanceCount = 0;
    geom.boundingSphere = chunk.sphere;

    const mesh = new Mesh(geom, this.material);
    mesh.customDepthMaterial = this.depthMaterial;
    mesh.castShadow = this.options.castShadow;
    mesh.receiveShadow = false;
    mesh.matrixAutoUpdate = false;
    mesh.visible = false;
    mesh.userData.maxLod = this.options.maxLod;
    mesh.userData.attachment = this;

    // ownership lives on the chunk, which disposes these when it is rebuilt or
    // dropped -- keeping a second list here would leak references on every
    // chunk-size change
    chunk.attachmentMeshes.push(mesh);
    this.crowd.add(mesh);
    return mesh;
  }

  dispose() {
    if (this.material) this.material.dispose();
    if (this.depthMaterial) this.depthMaterial.dispose();
  }
}
