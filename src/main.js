import {
  ACESFilmicToneMapping,
  Matrix4,
  PCFShadowMap,
  PCFSoftShadowMap,
  PerspectiveCamera,
  Scene,
  WebGLRenderer,
} from 'three';
import GUI from 'lil-gui';

import { VATAsset } from './vat/VATAsset.js';
import { VATCrowd } from './vat/VATCrowd.js';
import { VATAttachment } from './vat/VATAttachment.js';
import { buildEnvironment } from './demo/env.js';
import { CrowdSim } from './demo/crowd-sim.js';
import { CameraRig } from './demo/camera-rig.js';
import { HUD } from './demo/hud.js';
import { Benchmark } from './demo/benchmark.js';
import { makeCapGeometry, makeBalloonGeometry } from './demo/props.js';

const WORLD_SIZE = 420;
const MAX_INSTANCES = 100000;

const params = new URLSearchParams(location.search);
const startCount = clamp(parseInt(params.get('count') || '20000', 10) || 20000, 1, MAX_INSTANCES);
const ASSETS = ['crowd', 'goober-biped', 'goober-quad', 'cat'];
const assetName = ASSETS.includes(params.get('asset')) ? params.get('asset') : 'crowd';

function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

// Read the framebuffer while it is still valid (i.e. inside the animation
// frame). Outside it the drawing buffer is already cleared and comes back flat.
function samplePixels(renderer) {
  const src = renderer.domElement;
  const c = document.createElement('canvas');
  c.width = 160; c.height = 100;
  const ctx = c.getContext('2d');
  ctx.drawImage(src, 0, 0, c.width, c.height);
  const d = ctx.getImageData(0, 0, c.width, c.height).data;
  const buckets = new Set();
  let sum = 0;
  for (let i = 0; i < d.length; i += 4) {
    buckets.add(((d[i] >> 3) << 10) | ((d[i + 1] >> 3) << 5) | (d[i + 2] >> 3));
    sum += d[i] + d[i + 1] + d[i + 2];
  }
  return { uniqueColours: buckets.size, meanLuma: sum / (d.length / 4) / 3 };
}

const loading = document.getElementById('loading');
const loadingText = document.getElementById('loading-text');

async function main() {
  // ------------------------------------------------------------ renderer --
  // MSAA is off by default: a dense crowd makes nearly every pixel a geometry
  // edge, so 4x MSAA multiplies exactly the pixels that are already the budget.
  // ?aa=1 turns it back on (context-level, needs the reload).
  const wantAA = params.get('aa') === '1';
  const renderer = new WebGLRenderer({ antialias: wantAA, powerPreference: 'high-performance' });
  // A crowd this dense is fragment-heavy; 2x DPR quadruples that for very
  // little visible gain. Start conservative and let the slider raise it.
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.25));
  renderer.setSize(innerWidth, innerHeight);
  renderer.toneMapping = ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = PCFShadowMap;
  // three resets info *after* the shadow pass, so the default counters hide
  // every shadow draw. Take over the reset to report both passes honestly.
  renderer.info.autoReset = false;
  document.body.appendChild(renderer.domElement);

  const scene = new Scene();
  const camera = new PerspectiveCamera(58, innerWidth / innerHeight, 0.3, WORLD_SIZE * 2.2);
  camera.position.set(28, 16, 44);

  // which adapter did the browser actually give us? On dual-GPU laptops Chrome
  // regularly lands on the integrated one despite powerPreference.
  let gpuName = 'unknown';
  try {
    const gl = renderer.getContext();
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    gpuName = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  } catch (e) { void e; }
  console.log('WebGL adapter:', gpuName);

  loadingText.textContent = 'loading bake...';
  // Resolve against this module's own URL rather than the document URL, so the
  // bake is found whether the page is served from the site root, from a
  // subdirectory, or from a URL that is missing its trailing slash.
  // note: import.meta.url goes through a variable so Vite's static new URL()
  // analysis does not rewrite this into an (empty) build-time asset glob
  const metaUrl = import.meta.url;
  const bakeURL = new URL('../baked/' + assetName + '.json', metaUrl).href;
  const asset = await VATAsset.load(bakeURL);
  const vertexColorMode = !!asset.manifest.vertexColorMode;

  loadingText.textContent = 'building crowd...';
  const env = buildEnvironment(scene, { worldSize: WORLD_SIZE, shadowMapSize: 2048 });

  const crowd = new VATCrowd(asset, {
    capacity: MAX_INSTANCES,
    materialOptions: vertexColorMode ? { vertexColors: true } : {},
    mode: 'vat',
    chunkSize: 24,
    worldSize: WORLD_SIZE,
    lodDistances: [6, 14, 34, 80],
    crossFade: true,
    crossFadeMaxLod: 1,
    fadeDuration: 0.28,
    castShadow: true,
    receiveShadow: true,
    initialChunkCapacity: 256,
  });
  scene.add(crowd);

  // attachments read the bone texture even though the bodies are pure VAT;
  // frame-baked assets (goobers) have no bones and therefore no sockets
  if (asset.boneTex && asset.sockets && asset.sockets.head !== undefined) {
    const headRest = asset.manifest.boneRest[asset.sockets.head];
    const handRest = asset.manifest.boneRest[asset.sockets.handR];
    crowd.addAttachment(new VATAttachment(makeCapGeometry(), {
      socket: 'head',
      offset: new Matrix4().makeTranslation(headRest[0], headRest[1] + 0.215, headRest[2] + 0.004),
      accessory: 1,
      maxLod: 1,
      color: 0xd8514a,
      roughness: 0.75,
    }));
    crowd.addAttachment(new VATAttachment(makeBalloonGeometry(), {
      socket: 'handR',
      offset: new Matrix4().makeTranslation(handRest[0], handRest[1] - 0.06, handRest[2]),
      accessory: 2,
      maxLod: 1,
      color: 0xf2c14e,
      roughness: 0.35,
      metalness: 0.05,
    }));
  }

  const sim = new CrowdSim(crowd, {
    worldSize: WORLD_SIZE,
    obstacles: env.obstacles,
    separation: true,
    separationMaxAgents: 40000,
    scenario: 'wander',
    matchStride: true,
    // vertex-colored assets carry their identity in the mesh; the instance tint
    // is only a subtle hue shift, not the full apparel palette
    tintMode: vertexColorMode ? 'subtle' : 'palette',
  });
  sim.spawn(MAX_INSTANCES);
  crowd.count = startCount;

  const rig = new CameraRig(camera, renderer.domElement, { worldSize: WORLD_SIZE });
  const hud = new HUD();
  const bench = new Benchmark();

  // ----------------------------------------------------------------- gui --
  const state = {
    count: startCount,
    scenario: 'wander',
    speedScale: 1,
    separation: true,
    matchStride: true,
    mode: 'vat',
    crossFade: true,
    fadeDuration: 0.28,
    chunkSize: 24,
    lodBias: 1,
    lod0: 6, lod1: 14, lod2: 34, lod3: 80,
    debugLod: false,
    lodPivot: 'centre',
    shadowMaxLod: 2,
    shading: 'lambert',
    antialias: wantAA,
    autoRes: true,
    targetFps: 30,
    shadows: true,
    shadowAuto: true,
    shadowRadius: 70,
    fogDensity: 0.0011,
    shadowMapSize: 2048,
    sunAzimuth: 38,
    sunElevation: 46,
    exposure: 1.0,
    camera: 'orbit',
    fov: 58,
    pixelRatio: Math.min(devicePixelRatio, 1.25),
    softShadows: false,
    runBenchmark: () => startBenchmark(),
    resetCamera: () => { camera.position.set(28, 16, 44); rig.orbit.target.set(0, 1.2, 0); rig.setMode('orbit'); state.camera = 'orbit'; },
  };

  const gui = new GUI({ title: 'VAT crowd', width: 300 });
  state.asset = assetName;
  gui.add(state, 'asset', ASSETS).name('asset (reloads)').onChange((v) => {
    const u = new URL(location.href);
    u.searchParams.set('asset', v);
    location.href = u.href;
  });

  const fCrowd = gui.addFolder('crowd');
  fCrowd.add(state, 'count', 1, MAX_INSTANCES, 1).name('instances').onChange((v) => { crowd.count = v | 0; });
  fCrowd.add(state, 'scenario', CrowdSim.scenarios).onChange((v) => sim.setScenario(v));
  fCrowd.add(state, 'speedScale', 0, 2.5, 0.01).name('speed x').onChange((v) => { sim.options.speedScale = v; });
  fCrowd.add(state, 'separation').name('separation').onChange((v) => { sim.options.separation = v; });

  const fAnim = gui.addFolder('animation');
  fAnim.add(state, 'matchStride').name('stride matching').onChange((v) => {
    sim.options.matchStride = v;
    sim.refreshRates();
    hud.setNote(v ? '' : '<b>stride matching off</b> - clips play at their authored rate, so feet slide');
  });
  fAnim.add(state, 'mode', ['vat', 'bone']).name('playback mode').onChange((v) => {
    if (v === 'bone' && !asset.boneTex) {
      state.mode = 'vat';
      gui.controllersRecursive().forEach((ctl) => ctl.updateDisplay());
      hud.setNote('this asset has no bone texture (frame-baked) - vat mode only');
      return;
    }
    crowd.reconfigure({ mode: v });
  });
  fAnim.add(state, 'crossFade').name('cross-fade').onChange((v) => crowd.reconfigure({ crossFade: v }));
  fAnim.add(state, 'fadeDuration', 0.05, 1, 0.01).name('fade seconds').onChange((v) => crowd.reconfigure({ fadeDuration: v }));

  state.farPool = true;
  const fPerf = gui.addFolder('culling + lod');
  fPerf.add(state, 'farPool').name('merge far chunks').onChange((v) => crowd.reconfigure({ farPool: v }));
  fPerf.add(state, 'chunkSize', 12, WORLD_SIZE, 1).name('chunk size (m)').onChange((v) => crowd.reconfigure({ chunkSize: v }));
  fPerf.add(state, 'lodBias', 0.25, 4, 0.05).name('lod bias').onChange((v) => { crowd.options.lodBias = v; });
  fPerf.add(state, 'lod0', 2, 120, 1).name('lod 0 -> 1 (m)').onChange(applyLod);
  fPerf.add(state, 'lod1', 4, 200, 1).name('lod 1 -> 2 (m)').onChange(applyLod);
  fPerf.add(state, 'lod2', 8, 300, 1).name('lod 2 -> 3 (m)').onChange(applyLod);
  fPerf.add(state, 'lod3', 16, 500, 1).name('lod 3 -> 4 (m)').onChange(applyLod);
  fPerf.add(state, 'lodPivot', ['centre', 'edge']).name('lod measured from').onChange((v) => { crowd.options.lodPivot = v; });
  fPerf.add(state, 'shadowMaxLod', 0, 4, 1).name('shadow casts to lod').onChange((v) => { crowd.options.shadowMaxLod = v; });
  fPerf.add(state, 'debugLod').name('colour by lod').onChange((v) => crowd.setDebugLod(v));

  const fRender = gui.addFolder('rendering');
  fRender.add(state, 'shadows').onChange((v) => {
    renderer.shadowMap.enabled = v;
    env.sun.castShadow = v;
    scene.traverse((o) => { if (o.material) o.material.needsUpdate = true; });
  });
  fRender.add(state, 'shading', ['lambert', 'pbr']).name('shading').onChange((v) => {
    crowd.reconfigure({ quality: v });
  });
  fRender.add(state, 'antialias').name('msaa (reloads)').onChange((v) => {
    const u = new URL(location.href);
    u.searchParams.set('aa', v ? '1' : '0');
    location.href = u.href;
  });
  fRender.add(state, 'autoRes').name('auto resolution');
  fRender.add(state, 'targetFps', [30, 45, 60]).name('target fps');
  fRender.add(state, 'fogDensity', 0, 0.005, 0.0001).name('fog density').onChange((v) => {
    env.fog.density = v;
  });
  fRender.add(state, 'shadowAuto').name('shadow range: auto');
  fRender.add(state, 'softShadows').name('soft shadows (slow)').onChange((v) => {
    renderer.shadowMap.type = v ? PCFSoftShadowMap : PCFShadowMap;
    scene.traverse((o) => { if (o.material) o.material.needsUpdate = true; });
  });
  fRender.add(state, 'shadowRadius', 20, 220, 1).name('shadow range (m)');
  fRender.add(state, 'shadowMapSize', [1024, 2048, 4096]).name('shadow map').onChange((v) => {
    env.sun.shadow.mapSize.set(v, v);
    if (env.sun.shadow.map) { env.sun.shadow.map.dispose(); env.sun.shadow.map = null; }
  });
  fRender.add(state, 'sunAzimuth', 0, 360, 1).name('sun azimuth').onChange(applySun);
  fRender.add(state, 'sunElevation', 8, 88, 1).name('sun elevation').onChange(applySun);
  fRender.add(state, 'exposure', 0.3, 2, 0.01).onChange((v) => { renderer.toneMappingExposure = v; });
  fRender.add(state, 'pixelRatio', 0.5, 2, 0.05).name('pixel ratio').onChange((v) => renderer.setPixelRatio(v));

  const fCam = gui.addFolder('camera');
  fCam.add(state, 'camera', ['orbit', 'walk', 'tour']).onChange((v) => rig.setMode(v));
  fCam.add(state, 'fov', 30, 100, 1).onChange((v) => { camera.fov = v; camera.updateProjectionMatrix(); });
  fCam.add(state, 'resetCamera').name('reset camera');

  gui.add(state, 'runBenchmark').name('run benchmark');

  function applyLod() {
    crowd.options.lodDistances = [state.lod0, state.lod1, state.lod2, state.lod3];
  }
  function applySun() {
    env.setSunAngle(state.sunAzimuth, state.sunElevation);
  }
  applySun();

  function startBenchmark() {
    hud.setNote('benchmark starting...');
    bench.start(MAX_INSTANCES, (c) => {
      crowd.count = c;
      state.count = c;
      gui.controllersRecursive().forEach((ctl) => ctl.updateDisplay());
    }, (results) => {
      hud.setNote('<b>benchmark</b>' + bench.toTable());
      console.table(results);
      crowd.count = state.count;
    });
  }

  // --------------------------------------------------------------- loop ---
  addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });

  addEventListener('keydown', (e) => {
    if (e.code === 'KeyH') gui.show(gui._hidden);
    if (e.code === 'KeyB') startBenchmark();
  });

  let last = performance.now();
  let frameMs = 16;
  let resTimer = 0;
  let resFrames = 0;
  let frameNo = 0;
  window.__vat = { ready: false, frames: 0, errors: [] };

  // CPU-side scaling probe: how long do the sim and the chunk/LOD pass take at
  // a given instance count, with no rendering involved. Useful for spotting an
  // accidental O(n^2) without needing a fast GPU.
  window.__vat.cpuBench = (n, iters = 40, warmup = 12) => {
    const restore = crowd.count;
    crowd.count = Math.min(n, crowd.capacity);
    // Warm up first: the frames right after a count change pay for chunk
    // growth and a full attribute re-upload, which is not steady-state cost.
    for (let i = 0; i < warmup; i++) {
      sim.update(1 / 60, WORLD_SIZE);
      crowd.update(1 / 60, camera);
    }
    let sim0 = 0, crowd0 = 0;
    for (let i = 0; i < iters; i++) {
      let t = performance.now();
      sim.update(1 / 60, WORLD_SIZE);
      sim0 += performance.now() - t;
      t = performance.now();
      crowd.update(1 / 60, camera);
      crowd0 += performance.now() - t;
    }
    const out = {
      count: crowd.count,
      simMs: +(sim0 / iters).toFixed(3),
      crowdMs: +(crowd0 / iters).toFixed(3),
      totalMs: +((sim0 + crowd0) / iters).toFixed(3),
      chunks: crowd.stats.chunks,
      dirtyChunks: crowd.stats.dirtyChunks,
      bucketMs: +crowd.stats.bucketMs.toFixed(3),
      copyMs: +crowd.stats.copyMs.toFixed(3),
      uploadMB: +(crowd.stats.uploadBytes / 1048576).toFixed(2),
    };
    crowd.count = restore;
    return out;
  };

  renderer.setAnimationLoop(() => {
    const now = performance.now();
    // realDt is wall clock and drives reporting; dt is clamped so a long frame
    // cannot teleport the crowd. Never report frame rate from the clamped one.
    const realDt = (now - last) / 1000;
    const dt = Math.min(realDt, 0.05);
    last = now;

    rig.update(dt);
    sim.update(dt, WORLD_SIZE);
    crowd.update(dt, camera);
    // widen the shadow window as the camera pulls back, so a zoomed-out crowd
    // keeps its grounding shadows instead of losing them at the 70 m edge
    let shadowR = state.shadowRadius;
    if (state.shadowAuto) {
      const h = camera.position.y;
      const horiz = Math.hypot(camera.position.x, camera.position.z);
      shadowR = clamp(Math.max(70, (h + horiz * 0.5) * 1.1), 70, 260);
    }
    env.updateShadow(camera, shadowR);

    renderer.info.reset();
    renderer.render(scene, camera);
    if (window.__vat.wantSample) {
      window.__vat.sample = samplePixels(renderer);
      window.__vat.wantSample = false;
    }

    frameMs = performance.now() - now;
    bench.update(realDt, frameMs, {
      drawCalls: renderer.info.render.calls,
      triangles: renderer.info.render.triangles,
    });
    if (bench.active) hud.setNote(bench.progressText);
    hud.update(realDt, frameMs, renderer, crowd, sim, asset);

    // dynamic resolution: walk the pixel ratio toward the target frame rate.
    // The manual slider is the ceiling. Paused during benchmarks.
    if (state.autoRes && !bench.active) {
      resTimer += realDt; resFrames++;
      if (resTimer > 0.75) {
        const fps = resFrames / resTimer;
        const cur = renderer.getPixelRatio();
        let next = cur;
        if (fps < state.targetFps * 0.92) next = Math.max(0.6, cur * 0.92);
        else if (fps > state.targetFps * 1.25) next = Math.min(state.pixelRatio, cur * 1.05);
        if (Math.abs(next - cur) > 0.005) renderer.setPixelRatio(next);
        resTimer = 0; resFrames = 0;
      }
    }

    // smoke-test hooks (tools/smoke.mjs reads these)
    frameNo++;
    window.__vat.frames = frameNo;
    window.__vat.ms = frameMs;          // js submission cost only
    window.__vat.realMs = realDt * 1000; // wall clock, includes GPU
    window.__vat.stats = {
      drawCalls: renderer.info.render.calls,
      triangles: renderer.info.render.triangles,
      instances: crowd.stats.instances,
      drawn: crowd.stats.drawnInstances,
      rendered: crowd.stats.renderedInstances,
      farPooled: crowd.stats.farPooled,
      chunks: crowd.stats.chunks,
      activeChunks: crowd.stats.activeChunks,
      renderedChunks: crowd.stats.renderedChunks,
      lodCounts: Array.from(crowd.stats.lodCounts),
      programs: renderer.info.programs ? renderer.info.programs.length : 0,
    };
    if (frameNo > 4) window.__vat.ready = true;
  });

  loading.classList.add('done');
  setTimeout(() => loading.remove(), 700);

  Object.assign(window, { renderer, scene, camera, crowd, sim, asset, env, rig, gui });
  console.log('%cVAT crowd', 'font-weight:bold', {
    clips: asset.clips.map((c) => `${c.name} stride=${c.stride.toFixed(3)}m`),
    animationTextureMB: +((asset.memory.positions + asset.memory.normals) / 1048576).toFixed(2),
    boneTextureKB: +(asset.memory.bones / 1024).toFixed(1),
    lods: asset.lods.map((l) => `${l.vertexCount}v / ${l.triangleCount}t`),
  });
}

main().catch((err) => {
  console.error(err);
  loadingText.innerHTML = `<b>failed to start</b><br><span style="opacity:.7">${err.message}</span>`
    + '<br><br>run <code>npm run bake:demo</code> first if <code>public/baked/crowd.json</code> is missing.';
});
