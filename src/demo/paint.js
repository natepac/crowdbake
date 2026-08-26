// Persistent floor paint.
//
// One RGBA render target covers the world in plan view and accumulates splats
// forever -- painting is "render a few blob quads into the target", which makes
// a thousand splats exactly as cheap to keep as one. The ground material is
// patched to sample the target by world XZ and lay the paint over its albedo,
// so paint receives lighting, shadows and fog like any other ground detail.

import {
  ClampToEdgeWrapping,
  CanvasTexture,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  Float32BufferAttribute,
  LinearFilter,
  Mesh,
  NormalBlending,
  OrthographicCamera,
  Scene,
  ShaderMaterial,
  WebGLRenderTarget,
} from 'three';

function makeBlobAtlas() {
  // 2x2 atlas of irregular splat silhouettes, white-on-transparent
  const S = 512, half = S / 2;
  const c = document.createElement('canvas');
  c.width = c.height = S;
  const g = c.getContext('2d');
  g.fillStyle = '#fff';
  let seed = 7;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return (seed & 0xffff) / 0xffff; };
  for (let cell = 0; cell < 4; cell++) {
    const cx = (cell % 2) * half + half / 2;
    const cy = ((cell / 2) | 0) * half + half / 2;
    const R = half * 0.30;
    // wobbly core
    g.beginPath();
    const lobes = 9 + cell * 2;
    for (let i = 0; i <= lobes; i++) {
      const a = (i / lobes) * Math.PI * 2;
      const r = R * (0.75 + 0.45 * rnd());
      const x = cx + Math.cos(a) * r, y = cy + Math.sin(a) * r;
      if (i === 0) g.moveTo(x, y); else g.quadraticCurveTo(
        cx + Math.cos(a - Math.PI / lobes) * R * 1.25,
        cy + Math.sin(a - Math.PI / lobes) * R * 1.25, x, y);
    }
    g.closePath(); g.fill();
    // satellite droplets
    for (let i = 0; i < 14; i++) {
      const a = rnd() * Math.PI * 2;
      const d = R * (1.1 + rnd() * 0.9);
      g.beginPath();
      g.arc(cx + Math.cos(a) * d, cy + Math.sin(a) * d, R * (0.05 + rnd() * 0.14), 0, Math.PI * 2);
      g.fill();
    }
  }
  const tex = new CanvasTexture(c);
  tex.minFilter = LinearFilter;
  tex.magFilter = LinearFilter;
  return tex;
}

export class PaintSystem {
  constructor(renderer, ground, { worldSize = 420, size = 4096 } = {}) {
    this.renderer = renderer;
    this.worldSize = worldSize;
    this.half = worldSize / 2;

    this.target = new WebGLRenderTarget(size, size, { depthBuffer: false, stencilBuffer: false });
    this.target.texture.wrapS = ClampToEdgeWrapping;
    this.target.texture.wrapT = ClampToEdgeWrapping;
    this.target.texture.generateMipmaps = false;
    this.target.texture.minFilter = LinearFilter;

    this.scene = new Scene();
    this.camera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.camera.position.z = 0.5;

    this.atlas = makeBlobAtlas();
    this.queue = [];
    this.splatCount = 0;

    this.material = new ShaderMaterial({
      transparent: true,
      depthTest: false,
      depthWrite: false,
      blending: NormalBlending,
      uniforms: { uAtlas: { value: this.atlas } },
      vertexShader: /* glsl */`
        attribute vec4 iSplat;    // world x, z, radius, rotation
        attribute vec4 iColor;    // rgb, atlas cell
        varying vec2 vUv;
        varying vec3 vColor;
        void main() {
          float cs = cos(iSplat.w), sn = sin(iSplat.w);
          vec2 local = vec2(cs * position.x - sn * position.y, sn * position.x + cs * position.y) * iSplat.z;
          vec2 world = iSplat.xy + local;
          // world XZ -> paint NDC over the covered region
          vec2 ndc = world / WORLD_HALF_TOKEN;
          float cell = iColor.w;
          vUv = (position.xy * 0.5 + 0.5) * 0.5 + vec2(mod(cell, 2.0), floor(cell / 2.0)) * 0.5;
          vColor = iColor.rgb;
          gl_Position = vec4(ndc, 0.0, 1.0);
        }`.replace('WORLD_HALF_TOKEN', (worldSize / 2).toFixed(1)),
      fragmentShader: /* glsl */`
        uniform sampler2D uAtlas;
        varying vec2 vUv;
        varying vec3 vColor;
        void main() {
          float a = texture2D(uAtlas, vUv).a;
          if (a < 0.35) discard;
          gl_FragColor = vec4(vColor, min(a * 1.6, 1.0));
        }`,
    });

    this.hookGround(ground);
    this.cleared = false;
  }

  hookGround(ground) {
    const mat = ground.material;
    const half = this.half;
    mat.onBeforeCompile = (shader) => {
      shader.uniforms.uPaint = { value: this.target.texture };
      shader.vertexShader = 'varying vec2 vPaintUv;\n' + shader.vertexShader.replace(
        '#include <begin_vertex>',
        `#include <begin_vertex>
  vec4 paintWorld = modelMatrix * vec4(position, 1.0);
  vPaintUv = paintWorld.xz / ${(half * 2).toFixed(1)} + 0.5;`,
      );
      shader.fragmentShader = 'uniform sampler2D uPaint;\nvarying vec2 vPaintUv;\n' + shader.fragmentShader.replace(
        '#include <map_fragment>',
        `#include <map_fragment>
  if (vPaintUv.x > 0.0 && vPaintUv.x < 1.0 && vPaintUv.y > 0.0 && vPaintUv.y < 1.0) {
    vec4 paint = texture2D(uPaint, vPaintUv);
    diffuseColor.rgb = mix(diffuseColor.rgb, paint.rgb, paint.a * 0.92);
  }`,
      );
    };
    mat.customProgramCacheKey = () => 'ground-paint';
    mat.needsUpdate = true;
  }

  /** Queue one splat; drawn into the target on the next flush(). */
  add(x, z, radius, color, rot = Math.random() * Math.PI * 2) {
    this.queue.push(x, z, radius, rot, color[0], color[1], color[2], (Math.random() * 4) | 0);
  }

  /** Render queued splats into the accumulation target. Call once per frame. */
  flush() {
    if (!this.queue.length) return;
    const n = this.queue.length / 8;
    const geom = new InstancedBufferGeometry();
    geom.setAttribute('position', new Float32BufferAttribute([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0], 3));
    geom.setIndex([0, 1, 2, 0, 2, 3]);
    const splat = new Float32Array(n * 4);
    const color = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      splat[i * 4] = this.queue[i * 8];
      splat[i * 4 + 1] = this.queue[i * 8 + 1];
      splat[i * 4 + 2] = this.queue[i * 8 + 2];
      splat[i * 4 + 3] = this.queue[i * 8 + 3];
      color[i * 4] = this.queue[i * 8 + 4];
      color[i * 4 + 1] = this.queue[i * 8 + 5];
      color[i * 4 + 2] = this.queue[i * 8 + 6];
      color[i * 4 + 3] = this.queue[i * 8 + 7];
    }
    geom.setAttribute('iSplat', new InstancedBufferAttribute(splat, 4));
    geom.setAttribute('iColor', new InstancedBufferAttribute(color, 4));
    geom.instanceCount = n;
    this.splatCount += n;
    this.queue.length = 0;

    const mesh = new Mesh(geom, this.material);
    mesh.frustumCulled = false;
    this.scene.add(mesh);

    const r = this.renderer;
    const prevTarget = r.getRenderTarget();
    const prevAutoClear = r.autoClear;
    r.setRenderTarget(this.target);
    if (!this.cleared) { r.setClearColor(0x000000, 0); r.clear(true, false, false); this.cleared = true; }
    r.autoClear = false;
    r.render(this.scene, this.camera);
    r.autoClear = prevAutoClear;
    r.setRenderTarget(prevTarget);

    this.scene.remove(mesh);
    geom.dispose();
  }

  clear() {
    const r = this.renderer;
    const prev = r.getRenderTarget();
    r.setRenderTarget(this.target);
    r.setClearColor(0x000000, 0);
    r.clear(true, false, false);
    r.setRenderTarget(prev);
    this.splatCount = 0;
  }
}
