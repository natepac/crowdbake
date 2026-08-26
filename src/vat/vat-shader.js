// Shader plumbing for both playback modes.
//
// VAT mode   : two texelFetches into a position map + two into a normal map,
//              lerped. Cost is independent of bone count.
// BONE mode  : 4 skin influences x 2 frames x 3 texels of matrix rows. ~6x the
//              fetches, but the texture is kilobytes instead of megabytes and
//              you get real attachment sockets.
//
// Both are injected into a stock MeshStandardMaterial so three's lighting,
// shadows, fog and tone mapping keep working, and the same displacement is
// duplicated into a MeshDepthMaterial so shadows match the animation.

import { MeshDepthMaterial, MeshLambertMaterial, MeshStandardMaterial, RGBADepthPacking, Vector3 } from 'three';

const SKIN_TONES = [
  [0.96, 0.80, 0.69], [0.90, 0.72, 0.58], [0.80, 0.60, 0.46],
  [0.66, 0.47, 0.35], [0.48, 0.33, 0.24], [0.34, 0.23, 0.17],
];
const PANTS_COLORS = [
  [0.16, 0.18, 0.24], [0.30, 0.30, 0.33], [0.20, 0.24, 0.20],
  [0.36, 0.28, 0.22], [0.13, 0.13, 0.15], [0.25, 0.21, 0.30],
  [0.42, 0.38, 0.32], [0.10, 0.20, 0.28],
];

const COMMON_GLSL = /* glsl */`
precision highp sampler2D;

uniform highp sampler2D vatPosTex;
uniform highp sampler2D vatNrmTex;
uniform int   vatTexWidth;
uniform int   vatVertexCount;
uniform vec3  vatBoundsMin;
uniform vec3  vatBoundsExtent;
uniform vec4  vatClips[VAT_MAX_CLIPS];
uniform float vatTime;
uniform float vatInvFade;

attribute float aVid;
attribute vec4  iXform;   // world x, y, z, yaw
attribute vec4  iAnimA;   // clip, phaseOffset, rate, scale
attribute vec4  iAnimB;   // clip, phaseOffset, rate, fadeStart
attribute vec4  iTint;    // shirt rgb, packed variant

#ifdef VAT_BONE_MODE
uniform highp sampler2D vatBoneTex;
uniform int vatBoneTexWidth;
attribute vec4 aSkinIndex;
attribute vec4 aSkinWeight;
#endif

vec4 vatFetch(highp sampler2D tex, int index) {
  int y = index / vatTexWidth;
  return texelFetch(tex, ivec2(index - y * vatTexWidth, y), 0);
}

vec3 vatOctDecode(vec2 e) {
  vec3 v = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
  float t = max(-v.z, 0.0);
  v.x += v.x >= 0.0 ? -t : t;
  v.y += v.y >= 0.0 ? -t : t;
  return normalize(v);
}

// resolve a clip + phase into two frame rows and the blend between them
void vatFrames(int clipIdx, float phase, out int f0, out int f1, out float alpha) {
  vec4 c = vatClips[clipIdx];
  float fc = max(c.y, 1.0);
  float t = fract(phase) * fc;
  float i0 = floor(t);
  alpha = t - i0;
  float i1 = c.z > 0.5 ? mod(i0 + 1.0, fc) : min(i0 + 1.0, fc - 1.0);
  f0 = int(c.x + i0);
  f1 = int(c.x + i1);
}

#ifdef VAT_BONE_MODE
void vatBoneRows(int frame, int bone, out vec4 r0, out vec4 r1, out vec4 r2) {
  int x = bone * 3;
  r0 = texelFetch(vatBoneTex, ivec2(x, frame), 0);
  r1 = texelFetch(vatBoneTex, ivec2(x + 1, frame), 0);
  r2 = texelFetch(vatBoneTex, ivec2(x + 2, frame), 0);
}

void vatSkin(int frame, vec3 bindPos, vec3 bindNrm, out vec3 pos, out vec3 nrm) {
  vec4 m0 = vec4(0.0), m1 = vec4(0.0), m2 = vec4(0.0);
  for (int k = 0; k < 4; k++) {
    float w = aSkinWeight[k];
    if (w <= 0.0) continue;
    vec4 r0, r1, r2;
    vatBoneRows(frame, int(aSkinIndex[k]), r0, r1, r2);
    m0 += r0 * w; m1 += r1 * w; m2 += r2 * w;
  }
  vec4 p4 = vec4(bindPos, 1.0);
  pos = vec3(dot(m0, p4), dot(m1, p4), dot(m2, p4));
  nrm = vec3(dot(m0.xyz, bindNrm), dot(m1.xyz, bindNrm), dot(m2.xyz, bindNrm));
}

void vatSampleClip(int clipIdx, float phase, out vec3 pos, out vec3 nrm, out vec3 tan) {
  int f0, f1; float a;
  vatFrames(clipIdx, phase, f0, f1, a);
  vec3 bindPos = position;
  vec3 bindNrm = normal;
  vec3 p0, n0, p1, n1;
  vatSkin(f0, bindPos, bindNrm, p0, n0);
  vatSkin(f1, bindPos, bindNrm, p1, n1);
  pos = mix(p0, p1, a);
  nrm = normalize(mix(normalize(n0), normalize(n1), a));
  tan = vec3(1.0, 0.0, 0.0);
}
#else
void vatSampleClip(int clipIdx, float phase, out vec3 pos, out vec3 nrm, out vec3 tan) {
  int f0, f1; float a;
  vatFrames(clipIdx, phase, f0, f1, a);
  int vid = int(aVid);
  int i0 = f0 * vatVertexCount + vid;
  int i1 = f1 * vatVertexCount + vid;
  vec4 p0 = vatFetch(vatPosTex, i0);
  vec4 p1 = vatFetch(vatPosTex, i1);
  pos = vatBoundsMin + mix(p0.xyz, p1.xyz, a) * vatBoundsExtent;
  vec4 n0 = vatFetch(vatNrmTex, i0);
  vec4 n1 = vatFetch(vatNrmTex, i1);
  nrm = normalize(mix(vatOctDecode(n0.xy), vatOctDecode(n1.xy), a));
  #ifdef VAT_TANGENTS
  tan = normalize(mix(vatOctDecode(n0.zw), vatOctDecode(n1.zw), a));
  #else
  tan = vec3(1.0, 0.0, 0.0);
  #endif
}
#endif

void vatEvaluate(out vec3 pos, out vec3 nrm, out vec3 tan) {
  float phaseA = fract(iAnimA.y + vatTime * iAnimA.z);
  vatSampleClip(int(iAnimA.x), phaseA, pos, nrm, tan);

  #ifdef VAT_CROSSFADE
  if (iAnimB.x >= 0.0) {
    float blend = clamp((vatTime - iAnimB.w) * vatInvFade, 0.0, 1.0);
    if (blend > 0.001) {
      vec3 pB, nB, tB;
      float phaseB = fract(iAnimB.y + vatTime * iAnimB.z);
      vatSampleClip(int(iAnimB.x), phaseB, pB, nB, tB);
      pos = mix(pos, pB, blend);
      nrm = normalize(mix(nrm, nB, blend));
      tan = normalize(mix(tan, tB, blend));
    }
  }
  #endif

  // per-instance scale, yaw, translate
  pos *= iAnimA.w;
  float s = sin(iXform.w), c = cos(iXform.w);
  pos = vec3(c * pos.x + s * pos.z, pos.y, -s * pos.x + c * pos.z);
  nrm = vec3(c * nrm.x + s * nrm.z, nrm.y, -s * nrm.x + c * nrm.z);
  tan = vec3(c * tan.x + s * tan.z, tan.y, -s * tan.x + c * tan.z);
  pos += iXform.xyz;
}
`;

const VARYING_VERT = /* glsl */`
uniform float uLodLevel;
attribute float aMaterialId;
flat varying float vVatMat;
flat varying vec2 vVatVariant;
varying vec3 vVatShirt;
varying float vVatLod;
`;

const VARYING_FRAG = /* glsl */`
flat varying float vVatMat;
flat varying vec2 vVatVariant;
varying vec3 vVatShirt;
varying float vVatLod;
uniform vec3 uSkinPalette[6];
uniform vec3 uPantsPalette[8];
uniform float uDebugLod;
uniform float uLodLevel;
`;

const PALETTE_FRAG = /* glsl */`
{
  vec3 vatTint;
  int skinI = int(vVatVariant.y);
  int pantsI = int(vVatVariant.x);
  if (vVatMat < 0.5)      vatTint = uSkinPalette[skinI];
  else if (vVatMat < 1.5) vatTint = vVatShirt;
  else if (vVatMat < 2.5) vatTint = uPantsPalette[pantsI];
  else if (vVatMat < 3.5) vatTint = vec3(0.09, 0.09, 0.11);
  else                    vatTint = uSkinPalette[skinI] * 0.22 + vec3(0.02);
  diffuseColor.rgb *= vatTint;

  if (uDebugLod > 0.5) {
    vec3 lodCols[5];
    lodCols[0] = vec3(0.25, 0.85, 0.45);
    lodCols[1] = vec3(0.95, 0.80, 0.25);
    lodCols[2] = vec3(0.95, 0.45, 0.20);
    lodCols[3] = vec3(0.85, 0.25, 0.35);
    lodCols[4] = vec3(0.55, 0.35, 0.85);
    diffuseColor.rgb = mix(diffuseColor.rgb, lodCols[int(min(vVatLod, 4.0))], 0.75);
  }
}
`;

function flat3(list) {
  const out = [];
  for (const c of list) out.push(new Vector3(c[0], c[1], c[2]));
  return out;
}

/**
 * Shared uniform bag. One object, referenced by every material and the depth
 * material, so a single write per frame updates every draw.
 */
export function createVATUniforms(asset) {
  const clips = new Float32Array(Math.max(1, asset.clips.length) * 4);
  asset.clips.forEach((c, i) => {
    clips[i * 4] = c.frameStart;
    clips[i * 4 + 1] = c.frameCount;
    clips[i * 4 + 2] = c.loop ? 1 : 0;
    clips[i * 4 + 3] = 1 / c.duration;
  });
  return {
    vatPosTex: { value: asset.posTex },
    vatNrmTex: { value: asset.nrmTex },
    vatBoneTex: { value: asset.boneTex },
    vatBoneTexWidth: { value: asset.bone ? asset.bone.width : 1 },
    vatTexWidth: { value: asset.texture.width },
    vatVertexCount: { value: asset.vertexCount },
    vatBoundsMin: { value: new Vector3().fromArray(asset.bounds.min) },
    vatBoundsExtent: { value: new Vector3().fromArray(asset.bounds.extent) },
    vatClips: { value: clips },
    vatTime: { value: 0 },
    vatInvFade: { value: 1 / 0.25 },
    uSkinPalette: { value: flat3(SKIN_TONES) },
    uPantsPalette: { value: flat3(PANTS_COLORS) },
    uDebugLod: { value: 0 },
    uLodLevel: { value: 0 },
  };
}

function injectVertex(shader, uniforms, opts) {
  Object.assign(shader.uniforms, uniforms);
  if (opts.lodLevel !== undefined) shader.uniforms.uLodLevel = { value: opts.lodLevel };

  const prelude = COMMON_GLSL + (opts.withVaryings ? VARYING_VERT : '')
    + (opts.boneMode ? '\nattribute vec3 aBindNormal;\n' : '');

  const compute = /* glsl */`
  vec3 vatPos, vatNrm, vatTan;
  vatEvaluate(vatPos, vatNrm, vatTan);
`;

  const varyingWrite = opts.withVaryings ? /* glsl */`
  vVatMat = aMaterialId;
  float variant = iTint.w;
  vVatVariant = vec2(mod(variant, 8.0), mod(floor(variant / 8.0), 8.0));
  vVatShirt = iTint.rgb;
  vVatLod = uLodLevel;
` : '';

  shader.vertexShader = prelude + shader.vertexShader;

  if (opts.normalStage) {
    shader.vertexShader = shader.vertexShader
      .replace('#include <beginnormal_vertex>', compute + varyingWrite + `
  vec3 objectNormal = vatNrm;
  #ifdef USE_TANGENT
  vec3 objectTangent = vatTan;
  #endif
`)
      .replace('#include <begin_vertex>', '  vec3 transformed = vatPos;');
  } else {
    // depth / distance materials have no normal stage
    shader.vertexShader = shader.vertexShader
      .replace('#include <begin_vertex>', compute + varyingWrite + '  vec3 transformed = vatPos;');
  }
  return shader;
}

function defines(asset, opts) {
  const d = {
    VAT_MAX_CLIPS: Math.max(1, asset.clips.length),
  };
  if (opts.crossFade) d.VAT_CROSSFADE = '';
  if (opts.boneMode) d.VAT_BONE_MODE = '';
  else if (asset.texture.hasTangents && opts.tangents) d.VAT_TANGENTS = '';
  return d;
}

export function createVATMaterial(asset, uniforms, opts = {}) {
  const o = {
    crossFade: true, boneMode: false, tangents: false, lodLevel: 0,
    color: 0xffffff, roughness: 0.85, metalness: 0.0, quality: 'lambert', ...opts,
  };
  // 'lambert' = diffuse-only lighting, no GGX specular / env BRDF per pixel.
  // A matte crowd looks near-identical and the fragment shader is a fraction of
  // the cost -- on integrated GPUs the crowd's pixels are the frame budget.
  const mat = o.quality === 'pbr'
    ? new MeshStandardMaterial({
      color: o.color,
      roughness: o.roughness,
      metalness: o.metalness,
      map: o.map || null,
      normalMap: o.normalMap || null,
      vertexColors: !!o.vertexColors,
      dithering: true,
    })
    : new MeshLambertMaterial({
      color: o.color,
      map: o.map || null,
      vertexColors: !!o.vertexColors,
      dithering: true,
    });
  mat.defines = defines(asset, o);
  mat.userData.vat = o;
  mat.onBeforeCompile = (shader) => {
    injectVertex(shader, uniforms, { ...o, withVaryings: true, normalStage: true });
    shader.fragmentShader = VARYING_FRAG + shader.fragmentShader;
    shader.fragmentShader = shader.fragmentShader.replace('#include <color_fragment>',
      '#include <color_fragment>\n' + PALETTE_FRAG);
  };
  mat.customProgramCacheKey = () => 'vat|' + o.quality + '|' + JSON.stringify(mat.defines) + '|' + o.lodLevel;
  return mat;
}

export function createVATDepthMaterial(asset, uniforms, opts = {}) {
  const o = { crossFade: true, boneMode: false, lodLevel: 0, ...opts };
  const mat = new MeshDepthMaterial({ depthPacking: RGBADepthPacking });
  mat.defines = defines(asset, o);
  mat.onBeforeCompile = (shader) => injectVertex(shader, uniforms, { ...o, withVaryings: false, normalStage: false });
  mat.customProgramCacheKey = () => 'vatdepth|' + JSON.stringify(mat.defines);
  return mat;
}

export { SKIN_TONES, PANTS_COLORS };
