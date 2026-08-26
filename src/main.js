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
import { PaintSystem } from './demo/paint.js';
import { GunSystem } from './demo/gun.js';

const WORLD_SIZE = 420;
const MAX_INSTANCES = 100000;

const params = new URLSearchParams(location.search);
const startCount = clamp(parseInt(params.get('count') || '20000', 10) || 20000, 1, MAX_INSTANCES);
// shipped bakes; in dev the list is replaced by whatever exists in public/baked,
// and a single-file build lists exactly what it embeds
const INLINE_BAKES = globalThis.__CROWDBAKE_INLINE || null;
const ASSETS = INLINE_BAKES
  ? Object.keys(INLINE_BAKES)
  : ['crowd', 'menagerie', 'goober-biped', 'goober-quad', 'goober-cat'];
const assetName = (params.get('asset') || 'crowd').replace(/[^a-z0-9-]/g, '') || 'crowd';

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

  // --------------------------------------------------------------- assets --
  // Resolve against this module's own URL rather than the document URL, so
  // bakes are found whether the page is served from the site root, from a
  // subdirectory, or from a URL that is missing its trailing slash.
  // note: import.meta.url goes through a variable so Vite's static new URL()
  // analysis does not rewrite this into an (empty) build-time asset glob
  const metaUrl = import.meta.url;
  const bakedURL = (name) => new URL('../baked/' + name + '.json', metaUrl).href;

  const menagerie = assetName === 'menagerie';
  let assetSpecs;      // [{ asset (name), label }]
  if (menagerie) {
    loadingText.textContent = 'loading menagerie manifest...';
    const res = await fetch(bakedURL('menagerie'));
    if (!res.ok) throw new Error('no menagerie manifest - run: npm run bake:menagerie');
    const man = await res.json();
    if (!man.assets || !man.assets.length) throw new Error('menagerie manifest is empty - run: npm run bake:menagerie');
    assetSpecs = man.assets.map((a) => ({ name: a.asset, label: a.kind }));
  } else {
    assetSpecs = [{ name: assetName, label: assetName }];
  }

  const assets = [];
  for (let i = 0; i < assetSpecs.length; i++) {
    loadingText.textContent = `loading ${assetSpecs[i].label} (${i + 1}/${assetSpecs.length})...`;
    assets.push(await VATAsset.load(bakedURL(assetSpecs[i].name)));
  }

  loadingText.textContent = 'building crowd...';
  const env = buildEnvironment(scene, { worldSize: WORLD_SIZE, shadowMapSize: 2048 });

  // --------------------------------------------------------------- crowds --
  // One VATCrowd + one CrowdSim per asset. The single-asset demo is just the
  // K = 1 case; the menagerie runs one small crowd per goober kind. Known
  // limitation: separation only acts within a kind, so different kinds can
  // overlap in dense clusters.
  const K = assets.length;
  const perCapacity = Math.ceil(MAX_INSTANCES / K);
  const crowds = [];
  const sims = [];
  for (const a of assets) {
    const vcm = !!a.manifest.vertexColorMode;
    const c = new VATCrowd(a, {
      capacity: perCapacity,
      materialOptions: vcm ? { vertexColors: true } : {},
      mode: 'vat',
      // 40 sparse crowds want bigger cells: same coverage, far fewer chunk
      // objects, and the per-kind far pool still bounds the draw count
      chunkSize: menagerie ? 84 : 24,
      worldSize: WORLD_SIZE,
      lodDistances: [6, 14, 34, 80],
      crossFade: true,
      crossFadeMaxLod: 1,
      fadeDuration: 0.28,
      castShadow: true,
      receiveShadow: true,
      initialChunkCapacity: K > 1 ? 64 : 256,
    });
    scene.add(c);
    const s = new CrowdSim(c, {
      worldSize: WORLD_SIZE,
      obstacles: env.obstacles,
      separation: true,
      separationMaxAgents: 40000,
      scenario: 'wander',
      matchStride: true,
      // vertex-colored assets carry their identity in the mesh; the instance
      // tint is only a subtle hue shift, not the full apparel palette
      tintMode: vcm ? 'subtle' : 'palette',
    });
    s.spawn(perCapacity);
    crowds.push(c);
    sims.push(s);
  }
  const crowd = crowds[0];
  const sim = sims[0];
  const asset = assets[0];

  const eachCrowd = (fn) => crowds.forEach(fn);
  const eachSim = (fn) => sims.forEach(fn);

  function setTotalCount(total) {
    const per = Math.floor(total / K);
    let extra = total - per * K;
    for (const c of crowds) {
      c.count = per + (extra > 0 ? 1 : 0);
      if (extra > 0) extra--;
    }
  }
  setTotalCount(startCount);

  // aggregate facades for the HUD and test hooks (K = 1 passes through cheaply)
  const aggStats = {
    instances: 0, drawnInstances: 0, renderedInstances: 0, chunks: 0,
    activeChunks: 0, renderedChunks: 0, dirtyChunks: 0, farPooled: 0,
    triangles: 0, lodCounts: [], uploadBytes: 0, bucketMs: 0, copyMs: 0,
  };
  const aggSimStats = { simMs: 0, sepMs: 0, clipSwitches: 0 };
  const aggMemory = { positions: 0, normals: 0, bones: 0 };
  for (const a of assets) {
    aggMemory.positions += a.memory.positions;
    aggMemory.normals += a.memory.normals;
    aggMemory.bones += a.memory.bones;
  }
  const hudCrowd = K === 1 ? crowd : { stats: aggStats };
  const hudSim = K === 1 ? sim : { stats: aggSimStats };
  const hudAsset = K === 1 ? asset : { memory: aggMemory };

  function aggregate() {
    if (K === 1) return;
    for (const k of Object.keys(aggStats)) if (typeof aggStats[k] === 'number') aggStats[k] = 0;
    aggStats.lodCounts = new Array(assets[0].lods.length).fill(0);
    for (const c of crowds) {
      const s = c.stats;
      aggStats.instances += s.instances;
      aggStats.drawnInstances += s.drawnInstances;
      aggStats.renderedInstances += s.renderedInstances;
      aggStats.chunks += s.chunks;
      aggStats.activeChunks += s.activeChunks;
      aggStats.renderedChunks += s.renderedChunks;
      aggStats.dirtyChunks += s.dirtyChunks;
      aggStats.farPooled += s.farPooled;
      aggStats.triangles += s.triangles;
      aggStats.uploadBytes += s.uploadBytes;
      aggStats.bucketMs += s.bucketMs;
      aggStats.copyMs += s.copyMs;
      for (let l = 0; l < s.lodCounts.length && l < aggStats.lodCounts.length; l++) {
        aggStats.lodCounts[l] += s.lodCounts[l];
      }
    }
    aggSimStats.simMs = 0; aggSimStats.sepMs = 0; aggSimStats.clipSwitches = 0;
    for (const s of sims) {
      aggSimStats.simMs += s.stats.simMs;
      aggSimStats.sepMs += s.stats.sepMs;
      aggSimStats.clipSwitches += s.stats.clipSwitches;
    }
  }

  // ------------------------------------------------------------ humanoid ---
  // attachments read the bone texture even though the bodies are pure VAT;
  // frame-baked assets (goobers) have no bones and therefore no sockets
  if (K === 1 && asset.boneTex && asset.sockets && asset.sockets.head !== undefined) {
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

  const rig = new CameraRig(camera, renderer.domElement, { worldSize: WORLD_SIZE });
  const hud = new HUD();
  const bench = new Benchmark();

  // ------------------------------------------------------ paint minigun ---
  const paint = new PaintSystem(renderer, env.ground, { worldSize: WORLD_SIZE });
  const gun = new GunSystem(scene, camera, {
    crowds, sims, obstacles: env.obstacles, paint, worldSize: WORLD_SIZE,
  });
  const crosshair = document.getElementById('crosshair');

  function setGunMode(on) {
    state.camera = on ? 'gun' : 'orbit';
    rig.setMode(state.camera);
    gun.setEnabled(on);
    crosshair.style.display = on ? 'block' : 'none';
    gui.controllersRecursive().forEach((ctl) => ctl.updateDisplay());
    hud.setNote(on
      ? '<b>paint minigun</b> - click to lock, hold LMB to fire, WASD to move, E/Q up/down (fly!), R respawns'
      : '');
  }
  addEventListener('mousedown', (e) => { if (e.button === 0 && gun.enabled && document.pointerLockElement) gun.triggerDown = true; });
  addEventListener('mouseup', (e) => { if (e.button === 0) gun.triggerDown = false; });
  document.addEventListener('pointerlockchange', () => { if (!document.pointerLockElement) gun.triggerDown = false; });

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
    chunkSize: menagerie ? 84 : 24,
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

  const gui = new GUI({ title: menagerie ? `VAT menagerie (${K} kinds)` : 'VAT crowd', width: 300 });
  state.asset = assetName;
  let assetCtl = gui.add(state, 'asset', ASSETS.includes(assetName) ? ASSETS : [assetName, ...ASSETS])
    .name('asset (reloads)')
    .onChange(reloadWithAsset);

  const fCrowd = gui.addFolder('crowd');
  fCrowd.add(state, 'count', 1, MAX_INSTANCES, 1).name(menagerie ? 'instances (total)' : 'instances')
    .onChange((v) => setTotalCount(v | 0));
  fCrowd.add(state, 'scenario', CrowdSim.scenarios).onChange((v) => eachSim((s) => s.setScenario(v)));
  fCrowd.add(state, 'speedScale', 0, 2.5, 0.01).name('speed x').onChange((v) => eachSim((s) => { s.options.speedScale = v; }));
  fCrowd.add(state, 'separation').name('separation').onChange((v) => eachSim((s) => { s.options.separation = v; }));

  const fAnim = gui.addFolder('animation');
  fAnim.add(state, 'matchStride').name('stride matching').onChange((v) => {
    eachSim((s) => { s.options.matchStride = v; s.refreshRates(); });
    hud.setNote(v ? '' : '<b>stride matching off</b> - clips play at their authored rate, so feet slide');
  });
  fAnim.add(state, 'mode', ['vat', 'bone']).name('playback mode').onChange((v) => {
    if (v === 'bone' && !crowds.every((c) => c.asset.boneTex)) {
      state.mode = 'vat';
      gui.controllersRecursive().forEach((ctl) => ctl.updateDisplay());
      hud.setNote('this asset has no bone texture (frame-baked) - vat mode only');
      return;
    }
    eachCrowd((c) => c.reconfigure({ mode: v }));
  });
  fAnim.add(state, 'crossFade').name('cross-fade').onChange((v) => eachCrowd((c) => c.reconfigure({ crossFade: v })));
  fAnim.add(state, 'fadeDuration', 0.05, 1, 0.01).name('fade seconds').onChange((v) => eachCrowd((c) => c.reconfigure({ fadeDuration: v })));

  state.farPool = true;
  const fPerf = gui.addFolder('culling + lod');
  fPerf.add(state, 'farPool').name('merge far chunks').onChange((v) => eachCrowd((c) => c.reconfigure({ farPool: v })));
  fPerf.add(state, 'chunkSize', 12, WORLD_SIZE, 1).name('chunk size (m)').onChange((v) => eachCrowd((c) => c.reconfigure({ chunkSize: v })));
  fPerf.add(state, 'lodBias', 0.25, 4, 0.05).name('lod bias').onChange((v) => eachCrowd((c) => { c.options.lodBias = v; }));
  fPerf.add(state, 'lod0', 2, 120, 1).name('lod 0 -> 1 (m)').onChange(applyLod);
  fPerf.add(state, 'lod1', 4, 200, 1).name('lod 1 -> 2 (m)').onChange(applyLod);
  fPerf.add(state, 'lod2', 8, 300, 1).name('lod 2 -> 3 (m)').onChange(applyLod);
  fPerf.add(state, 'lod3', 16, 500, 1).name('lod 3 -> 4 (m)').onChange(applyLod);
  fPerf.add(state, 'lodPivot', ['centre', 'edge']).name('lod measured from').onChange((v) => eachCrowd((c) => { c.options.lodPivot = v; }));
  fPerf.add(state, 'shadowMaxLod', 0, 4, 1).name('shadow casts to lod').onChange((v) => eachCrowd((c) => { c.options.shadowMaxLod = v; }));
  fPerf.add(state, 'debugLod').name('colour by lod').onChange((v) => eachCrowd((c) => c.setDebugLod(v)));

  const fRender = gui.addFolder('rendering');
  fRender.add(state, 'shadows').onChange((v) => {
    renderer.shadowMap.enabled = v;
    env.sun.castShadow = v;
    scene.traverse((o) => { if (o.material) o.material.needsUpdate = true; });
  });
  fRender.add(state, 'shading', ['lambert', 'pbr']).name('shading').onChange((v) => {
    eachCrowd((c) => c.reconfigure({ quality: v }));
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
  fCam.add(state, 'camera', ['orbit', 'walk', 'gun', 'tour']).onChange((v) => {
    if (v === 'gun') setGunMode(true);
    else { rig.setMode(v); gun.setEnabled(false); crosshair.style.display = 'none'; }
  });
  fCam.add(state, 'fov', 30, 100, 1).onChange((v) => { camera.fov = v; camera.updateProjectionMatrix(); });
  fCam.add(state, 'resetCamera').name('reset camera');

  const fGun = gui.addFolder('paint minigun');
  state.gunFireRate = 18;
  state.clearPaint = () => paint.clear();
  fGun.add(state, 'gunFireRate', 4, 40, 1).name('rounds / second').onChange((v) => { gun.fireRate = v; });
  fGun.add(state, 'clearPaint').name('clear paint');

  gui.add(state, 'runBenchmark').name('run benchmark');

  // ---- goober lab (dev server only: pick any critter kind, reroll its look) --
  try {
    // single-file and file:// contexts have no dev API by definition
    if (INLINE_BAKES || location.protocol === 'file:') throw new Error('no dev api');
    const kindsRes = await fetch('__goobers/kinds');
    if (kindsRes.ok) {
      const kinds = await kindsRes.json();
      const assetsRes = await fetch('__goobers/assets');
      if (assetsRes.ok) {
        const baked = await assetsRes.json();
        if (baked.length) {
          assetCtl = assetCtl.options(baked.sort()).name('asset (reloads)').onChange(reloadWithAsset);
        }
      }
      const lab = gui.addFolder('goober lab (dev)');
      const gs = {
        kind: assetName.startsWith('goober-') ? assetName.replace(/^goober-/, '') : kinds[0].kind,
        bake: () => bakeGoober(false),
        regenerate: () => bakeGoober(true),
      };
      // preset names double as kinds; goober-* presets are stored without prefix
      const kindNames = kinds.map((k) => k.kind);
      if (!kindNames.includes(gs.kind) && kindNames.includes('goober-' + gs.kind)) gs.kind = 'goober-' + gs.kind;
      lab.add(gs, 'kind', kindNames);
      lab.add(gs, 'bake').name('bake this kind');
      lab.add(gs, 'regenerate').name('regenerate (new look)');
      let baking = false;
      async function bakeGoober(reroll) {
        if (baking) return;
        baking = true;
        const seed = reroll ? (1 + Math.floor(Math.random() * 0xffffff)) : 7;
        hud.setNote(`<b>baking ${gs.kind}</b> (seed ${seed})... 15-60 s, the page reloads when done`);
        try {
          const r = await fetch(`__goobers/bake?kind=${encodeURIComponent(gs.kind)}&seed=${seed}`);
          const body = await r.json();
          if (!r.ok) { hud.setNote(`<b>bake failed</b><br>${(body.error || r.status)}`); return; }
          // regenerating from inside the menagerie keeps you in the menagerie
          reloadWithAsset(menagerie ? 'menagerie' : body.asset);
        } catch (e) {
          hud.setNote('<b>bake failed</b><br>' + e.message);
        } finally {
          baking = false;
        }
      }
    }
  } catch (e) { void e; /* static hosting: no dev API, no panel */ }

  function reloadWithAsset(name) {
    const u = new URL(location.href);
    u.searchParams.set('asset', name);
    location.href = u.href;
  }

  function applyLod() {
    eachCrowd((c) => { c.options.lodDistances = [state.lod0, state.lod1, state.lod2, state.lod3]; });
  }
  function applySun() {
    env.setSunAngle(state.sunAzimuth, state.sunElevation);
  }
  applySun();

  function startBenchmark() {
    hud.setNote('benchmark starting...');
    bench.start(MAX_INSTANCES, (c) => {
      setTotalCount(c);
      state.count = c;
      gui.controllersRecursive().forEach((ctl) => ctl.updateDisplay());
    }, (results) => {
      hud.setNote('<b>benchmark</b>' + bench.toTable());
      console.table(results);
      setTotalCount(state.count);
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
    if (e.code === 'KeyG') setGunMode(!gun.enabled);
    if (e.code === 'KeyR') {
      // respawn to the editor's instance count; dead slots come back where they fell
      setTotalCount(state.count);
      if (gun.enabled) hud.setNote(`respawned to ${state.count.toLocaleString()} - ${gun.kills.toLocaleString()} kills so far`);
    }
  });

  let last = performance.now();
  let frameMs = 16;
  let resTimer = 0;
  let resFrames = 0;
  let frameNo = 0;
  let lastKills = -1, lastShots = -1;
  window.__vat = { ready: false, frames: 0, errors: [] };

  // CPU-side scaling probe: how long do the sim and the chunk/LOD pass take at
  // a given instance count, with no rendering involved. Useful for spotting an
  // accidental O(n^2) without needing a fast GPU.
  window.__vat.cpuBench = (n, iters = 40, warmup = 12) => {
    const restore = crowds.map((c) => c.count);
    setTotalCount(Math.min(n, MAX_INSTANCES));
    for (let i = 0; i < warmup; i++) {
      eachSim((s) => s.update(1 / 60, WORLD_SIZE));
      eachCrowd((c) => c.update(1 / 60, camera));
    }
    let sim0 = 0, crowd0 = 0;
    for (let i = 0; i < iters; i++) {
      let t = performance.now();
      eachSim((s) => s.update(1 / 60, WORLD_SIZE));
      sim0 += performance.now() - t;
      t = performance.now();
      eachCrowd((c) => c.update(1 / 60, camera));
      crowd0 += performance.now() - t;
    }
    const out = {
      count: crowds.reduce((s, c) => s + c.count, 0),
      simMs: +(sim0 / iters).toFixed(3),
      crowdMs: +(crowd0 / iters).toFixed(3),
      totalMs: +((sim0 + crowd0) / iters).toFixed(3),
      chunks: crowds.reduce((s, c) => s + c.stats.chunks, 0),
      dirtyChunks: crowds.reduce((s, c) => s + c.stats.dirtyChunks, 0),
      bucketMs: +crowds.reduce((s, c) => s + c.stats.bucketMs, 0).toFixed(3),
      copyMs: +crowds.reduce((s, c) => s + c.stats.copyMs, 0).toFixed(3),
      uploadMB: +(crowds.reduce((s, c) => s + c.stats.uploadBytes, 0) / 1048576).toFixed(2),
    };
    crowds.forEach((c, i) => { c.count = restore[i]; });
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
    for (const s of sims) s.update(dt, WORLD_SIZE);
    for (const c of crowds) c.update(dt, camera);
    gun.update(dt);
    paint.flush();
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

    aggregate();
    frameMs = performance.now() - now;
    bench.update(realDt, frameMs, {
      drawCalls: renderer.info.render.calls,
      triangles: renderer.info.render.triangles,
    });
    if (bench.active) hud.setNote(bench.progressText);
    else if (gun.enabled && (gun.kills !== lastKills || gun.shots !== lastShots)) {
      lastKills = gun.kills; lastShots = gun.shots;
      hud.setNote(`<b>minigun</b> ${gun.kills.toLocaleString()} kills / ${gun.shots.toLocaleString()} rounds / ${paint.splatCount.toLocaleString()} splats - R respawns`);
    }
    hud.update(realDt, frameMs, renderer, hudCrowd, hudSim, hudAsset);

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
    const st = hudCrowd.stats;
    window.__vat.frames = frameNo;
    window.__vat.ms = frameMs;          // js submission cost only
    window.__vat.realMs = realDt * 1000; // wall clock, includes GPU
    window.__vat.stats = {
      drawCalls: renderer.info.render.calls,
      triangles: renderer.info.render.triangles,
      instances: st.instances,
      drawn: st.drawnInstances,
      rendered: st.renderedInstances,
      farPooled: st.farPooled,
      chunks: st.chunks,
      activeChunks: st.activeChunks,
      renderedChunks: st.renderedChunks,
      lodCounts: Array.from(st.lodCounts),
      programs: renderer.info.programs ? renderer.info.programs.length : 0,
    };
    if (frameNo > 4) window.__vat.ready = true;
  });

  loading.classList.add('done');
  setTimeout(() => loading.remove(), 700);

  Object.assign(window, { renderer, scene, camera, crowd, sim, asset, crowds, sims, assets, env, rig, gui, gun, paint, setGunMode });
  console.log('%cVAT ' + (menagerie ? `menagerie: ${K} kinds` : assetName), 'font-weight:bold', {
    kinds: assetSpecs.map((a) => a.label),
    animationTextureMB: +((aggMemory.positions + aggMemory.normals || asset.memory.positions + asset.memory.normals) / 1048576).toFixed(2),
  });
}

main().catch((err) => {
  console.error(err);
  loadingText.innerHTML = `<b>failed to start</b><br><span style="opacity:.7">${err.message}</span>`
    + '<br><br><code>npm run bake:demo</code> bakes the humanoid, <code>npm run bake:menagerie</code> bakes every goober.';
});
