// Crowd behaviour. Deliberately plain JS over flat typed arrays: at 100k agents
// the sim, not the GPU, is the thing most likely to blow the frame budget.
//
// The part that matters for the demo is speedToClip(): the agent's *actual*
// ground speed picks a clip and then drives that clip's playback rate through
// the baked stride length, which is what stops the feet sliding.

import { STRIDE } from '../vat/VATCrowd.js';

const SCENARIOS = ['wander', 'stampede', 'parade', 'mill', 'gather'];
const SC_WANDER = 0, SC_STAMPEDE = 1, SC_PARADE = 2, SC_MILL = 3, SC_GATHER = 4;

export class CrowdSim {
  constructor(crowd, opts = {}) {
    this.crowd = crowd;
    this.asset = crowd.asset;
    this.options = {
      worldSize: 512,
      obstacles: [],
      separation: true,
      separationMaxAgents: 40000,
      separationRadius: 0.85,
      maxNeighbours: 6,
      scenario: 'wander',
      matchStride: true,
      speedScale: 1,
      tintMode: 'palette',
      ...opts,
    };

    const cap = crowd.capacity;
    this.cap = cap;
    this.px = new Float32Array(cap);
    this.pz = new Float32Array(cap);
    this.vx = new Float32Array(cap);
    this.vz = new Float32Array(cap);
    this.tx = new Float32Array(cap);
    this.tz = new Float32Array(cap);
    this.desired = new Float32Array(cap);
    this.yaw = new Float32Array(cap);
    this.clip = new Int8Array(cap).fill(-1);
    this.nextSwitch = new Float32Array(cap);
    this.personal = new Float32Array(cap);
    // Neighbour lookup pack: [x, z, personalRadius, _] per agent. Separation
    // reads neighbours in scattered order, so keeping the three values it needs
    // on one cache line turns three misses per neighbour into one.
    this.nb = new Float32Array(cap * 4);

    this.clips = {
      idle: this.asset.clipByName('idle'),
      walk: this.asset.clipByName('walk'),
      jog: this.asset.clipByName('jog'),
      run: this.asset.clipByName('run'),
      wave: this.asset.clipByName('wave'),
      cheer: this.asset.clipByName('cheer'),
    };
    // fall back down the gait chain when a bake has fewer clips: a missing run
    // becomes jog, a missing jog becomes walk. Falling back to clip 0 would map
    // fast agents onto idle.
    const any = this.asset.clips.length ? 0 : -1;
    if (this.clips.idle < 0) this.clips.idle = any;
    if (this.clips.walk < 0) this.clips.walk = this.clips.idle;
    if (this.clips.jog < 0) this.clips.jog = this.clips.walk;
    if (this.clips.run < 0) this.clips.run = this.clips.jog;

    // Band thresholds are authored for a 1.4 m/s walk. An asset whose walk clip
    // covers ground at a different rate (a goober, a dog) scales the whole gait
    // ladder so agents pick sensible clips for their body.
    const walkClip = this.asset.clips[this.clips.walk];
    const norm = walkClip && walkClip.groundSpeed > 0.05 ? walkClip.groundSpeed / 1.4 : 1;
    this.speedNorm = norm;

    this.locomotion = [
      { clip: this.clips.idle, upTo: 0.22 * norm },
      { clip: this.clips.walk, upTo: 2.05 * norm },
      { clip: this.clips.jog, upTo: 3.15 * norm },
      { clip: this.clips.run, upTo: Infinity },
    ];
    // flat mirrors of the band table, so the per-agent loop touches no objects
    this.bandUpTo = Float64Array.from(this.locomotion.map((b) => b.upTo));
    this.bandClip = Int32Array.from(this.locomotion.map((b) => b.clip));

    this.time = 0;
    this.attractor = { x: 0, z: 0 };
    this.scenarioId = SCENARIOS.indexOf(this.options.scenario);
    this.buildObstacleGrid();

    // separation grid
    this.cellSize = 1.7;
    this.gridN = Math.ceil(this.options.worldSize / this.cellSize) + 2;
    this.cellCount = this.gridN * this.gridN;
    this.gCount = new Int32Array(this.cellCount);
    this.gStart = new Int32Array(this.cellCount + 1);
    this.gItems = new Int32Array(cap);
    this.gCursor = new Int32Array(this.cellCount);
    this.gCellOf = new Int32Array(cap);

    this.stats = { simMs: 0, sepMs: 0, clipSwitches: 0 };
  }

  static get scenarios() { return SCENARIOS; }

  /**
   * Obstacles are static, so bucket them once into a coarse grid. Without this
   * every agent tests every pillar -- 100k x 26 distance checks per frame, which
   * on its own costs more than the whole rest of the sim.
   */
  buildObstacleGrid() {
    const obs = this.options.obstacles || [];
    this.obCell = 32;
    this.obN = Math.ceil(this.options.worldSize / this.obCell) + 2;
    this.obOrigin = (this.obN * this.obCell) / 2;
    this.obBuckets = new Array(this.obN * this.obN);
    for (const ob of obs) {
      const reach = ob.radius + 1.2;
      const x0 = Math.max(0, Math.floor((ob.x - reach + this.obOrigin) / this.obCell));
      const x1 = Math.min(this.obN - 1, Math.floor((ob.x + reach + this.obOrigin) / this.obCell));
      const z0 = Math.max(0, Math.floor((ob.z - reach + this.obOrigin) / this.obCell));
      const z1 = Math.min(this.obN - 1, Math.floor((ob.z + reach + this.obOrigin) / this.obCell));
      for (let cz = z0; cz <= z1; cz++) {
        for (let cx = x0; cx <= x1; cx++) {
          const k = cz * this.obN + cx;
          (this.obBuckets[k] || (this.obBuckets[k] = [])).push(ob);
        }
      }
    }
  }

  /** Populate n agents with randomised appearance and starting state. */
  spawn(n) {
    const crowd = this.crowd;
    const half = this.options.worldSize * 0.46;
    for (let i = 0; i < n; i++) {
      const x = (Math.random() * 2 - 1) * half;
      const z = (Math.random() * 2 - 1) * half;
      this.px[i] = x; this.pz[i] = z;
      this.vx[i] = 0; this.vz[i] = 0;
      this.nb[i * 4] = x; this.nb[i * 4 + 1] = z;
      this.yaw[i] = Math.random() * Math.PI * 2;
      this.desired[i] = (0.7 + Math.random() * Math.random() * 3.2) * this.speedNorm;
      this.personal[i] = 0.55 + Math.random() * 0.35;
      this.nb[i * 4 + 2] = this.personal[i];
      this.pickTarget(i);
      this.clip[i] = -1;
      this.nextSwitch[i] = 0;

      const scale = 0.9 + Math.random() * 0.22;
      crowd.setScale(i, scale);
      crowd.setVariant(i, {
        shirt: this.options.tintMode === 'subtle' ? subtleTint() : hsvShirt(),
        pants: (Math.random() * 8) | 0,
        skin: (Math.random() * 6) | 0,
        accessory: Math.random() < 0.18 ? 1 : (Math.random() < 0.06 ? 2 : 0),
      });
      crowd.setClip(i, this.clips.idle, this.asset.naturalRate(this.clips.idle), Math.random());
      crowd.setTransform(i, x, 0, z, this.yaw[i]);
    }
    crowd.count = n;
  }

  pickTarget(i) {
    const half = this.options.worldSize * 0.45;
    this.tx[i] = (Math.random() * 2 - 1) * half;
    this.tz[i] = (Math.random() * 2 - 1) * half;
  }

  setScenario(name) {
    this.options.scenario = name;
    this.scenarioId = SCENARIOS.indexOf(name);
    const n = this.crowd.count;
    if (name === 'parade') {
      const cols = Math.ceil(Math.sqrt(n));
      const spacing = 1.6;
      for (let i = 0; i < n; i++) {
        const c = i % cols, r = (i / cols) | 0;
        this.px[i] = (c - cols / 2) * spacing + (Math.random() - 0.5) * 0.3;
        this.pz[i] = (r - cols / 2) * spacing + (Math.random() - 0.5) * 0.3;
        this.desired[i] = 1.45 * this.speedNorm;
        this.yaw[i] = 0;
      }
    } else if (name === 'wander') {
      for (let i = 0; i < n; i++) {
        this.desired[i] = (0.7 + Math.random() * Math.random() * 3.2) * this.speedNorm;
        this.pickTarget(i);
      }
    } else if (name === 'stampede') {
      for (let i = 0; i < n; i++) this.desired[i] = (3.0 + Math.random() * 1.6) * this.speedNorm;
    } else if (name === 'mill') {
      for (let i = 0; i < n; i++) this.desired[i] = (0.9 + Math.random() * 1.4) * this.speedNorm;
    } else if (name === 'gather') {
      for (let i = 0; i < n; i++) this.desired[i] = (0.5 + Math.random() * 2.4) * this.speedNorm;
    }
  }

  update(dt, worldSize) {
    const t0 = performance.now();
    this.time += dt;
    const n = this.crowd.count;
    if (n === 0) return;

    const half = (worldSize || this.options.worldSize) * 0.48;
    const scenario = this.scenarioId;
    const speedScale = this.options.speedScale;
    const px = this.px, pz = this.pz, vx = this.vx, vz = this.vz;
    const tx = this.tx, tz = this.tz, desired = this.desired;
    const obBuckets = this.obBuckets, obN = this.obN, obCell = this.obCell, obOrigin = this.obOrigin;
    const accelK = Math.min(1, 5.5 * dt);

    // moving attractor for stampede / gather
    this.attractor.x = Math.cos(this.time * 0.18) * half * 0.55;
    this.attractor.z = Math.sin(this.time * 0.23) * half * 0.55;

    for (let i = 0; i < n; i++) {
      const cx = px[i], cz = pz[i];
      let gx, gz;
      if (scenario === SC_STAMPEDE || scenario === SC_GATHER) {
        gx = this.attractor.x; gz = this.attractor.z;
      } else if (scenario === SC_PARADE) {
        gx = cx; gz = cz + 40;
      } else if (scenario === SC_MILL) {
        const r = Math.sqrt(cx * cx + cz * cz) || 1;
        const inward = (half * 0.5 - r) * 0.35;
        // tangential goal produces a slowly rotating crowd
        gx = cx + (-cz / r) * 18 + (cx / r) * inward;
        gz = cz + (cx / r) * 18 + (cz / r) * inward;
      } else {
        gx = tx[i]; gz = tz[i];
        const dx0 = gx - cx, dz0 = gz - cz;
        if (dx0 * dx0 + dz0 * dz0 < 9) { this.pickTarget(i); gx = tx[i]; gz = tz[i]; }
      }

      let dx = gx - cx;
      let dz = gz - cz;
      const d = Math.sqrt(dx * dx + dz * dz) || 1;
      const inv = 1 / d;
      dx *= inv; dz *= inv;

      let want = desired[i] * speedScale;
      if (scenario === SC_GATHER && d < 12) want *= d / 12;    // ease in at the rally point

      let ax = dx * want - vx[i];
      let az = dz * want - vz[i];

      // obstacle avoidance, but only against pillars in this agent's cell
      let ocx = ((cx + obOrigin) / obCell) | 0;
      let ocz = ((cz + obOrigin) / obCell) | 0;
      if (ocx >= 0 && ocz >= 0 && ocx < obN && ocz < obN) {
        const bucket = obBuckets[ocz * obN + ocx];
        if (bucket !== undefined) {
          for (let o = 0; o < bucket.length; o++) {
            const ob = bucket[o];
            const ox = cx - ob.x, oz = cz - ob.z;
            const od2 = ox * ox + oz * oz;
            const rr = ob.radius + 1.2;
            if (od2 < rr * rr) {
              const od = Math.sqrt(od2) || 0.001;
              const push = ((rr - od) / rr) * 26 / od;
              ax += ox * push;
              az += oz * push;
            }
          }
        }
      }

      // soft world boundary
      if (cx > half) ax -= (cx - half) * 4;
      else if (cx < -half) ax -= (cx + half) * 4;
      if (cz > half) az -= (cz - half) * 4;
      else if (cz < -half) az -= (cz + half) * 4;

      let nvx = vx[i] + ax * accelK;
      let nvz = vz[i] + az * accelK;

      const maxS = want * 1.25 + 0.2;
      const sp2 = nvx * nvx + nvz * nvz;
      if (sp2 > maxS * maxS) {
        const k = maxS / Math.sqrt(sp2);
        nvx *= k; nvz *= k;
      }
      vx[i] = nvx; vz[i] = nvz;
    }
    this.stats.simMs = performance.now() - t0;

    if (this.options.separation && n <= this.options.separationMaxAgents) {
      const t1 = performance.now();
      this.applySeparation(n, dt);
      this.stats.sepMs = performance.now() - t1;
    } else {
      this.stats.sepMs = 0;
    }

    this.integrate(n, dt);
  }

  applySeparation(n, dt) {
    const { cellSize, gridN, gCount, gStart, gItems, gCursor, gCellOf } = this;
    const originShift = (gridN * cellSize) / 2;
    const pack = this.nb;
    gCount.fill(0);
    for (let i = 0; i < n; i++) {
      let cx = ((pack[i * 4] + originShift) / cellSize) | 0;
      let cz = ((pack[i * 4 + 1] + originShift) / cellSize) | 0;
      if (cx < 0) cx = 0; else if (cx >= gridN) cx = gridN - 1;
      if (cz < 0) cz = 0; else if (cz >= gridN) cz = gridN - 1;
      const c = cz * gridN + cx;
      gCellOf[i] = c;
      gCount[c]++;
    }
    let acc = 0;
    for (let c = 0; c < gCount.length; c++) { gStart[c] = acc; acc += gCount[c]; }
    gStart[gCount.length] = acc;
    gCursor.set(gStart.subarray(0, gCount.length));
    for (let i = 0; i < n; i++) gItems[gCursor[gCellOf[i]]++] = i;

    const maxN = this.options.maxNeighbours;
    const strength = 14 * Math.min(dt, 0.05);
    const nb = this.nb;

    for (let i = 0; i < n; i++) {
      const cell = gCellOf[i];
      const cz = (cell / gridN) | 0;
      const cx = cell - cz * gridN;
      const io = i * 4;
      const ix = nb[io], iz = nb[io + 1], ri = nb[io + 2];
      let sx = 0, sz = 0, checked = 0;

      for (let oz = -1; oz <= 1 && checked < maxN; oz++) {
        const zz = cz + oz;
        if (zz < 0 || zz >= gridN) continue;
        for (let ox = -1; ox <= 1 && checked < maxN; ox++) {
          const xx = cx + ox;
          if (xx < 0 || xx >= gridN) continue;
          const c = zz * gridN + xx;
          const s = gStart[c], e = gStart[c + 1];
          for (let k = s; k < e && checked < maxN; k++) {
            const j = gItems[k];
            if (j === i) continue;
            checked++;
            const jo = j * 4;
            const dx = ix - nb[jo];
            const dz = iz - nb[jo + 1];
            const d2 = dx * dx + dz * dz;
            const rr = ri + nb[jo + 2];
            if (d2 > rr * rr || d2 < 1e-8) continue;
            const d = Math.sqrt(d2);
            const w = (rr - d) / rr / d;
            sx += dx * w; sz += dz * w;
          }
        }
      }
      if (sx !== 0 || sz !== 0) {
        this.vx[i] += sx * strength;
        this.vz[i] += sz * strength;
      }
    }
  }

  integrate(n, dt) {
    const crowd = this.crowd;
    const data = crowd.data;
    const stride = STRIDE;
    const matchStride = this.options.matchStride;
    const turnRate = Math.min(1, dt * 7);
    const px = this.px, pz = this.pz, vx = this.vx, vz = this.vz, yaw = this.yaw;
    const nb = this.nb;
    const clip = this.clip, nextSwitch = this.nextSwitch;
    const bands = this.bandUpTo, bandClip = this.bandClip, bandCount = bands.length;
    const time = this.time;
    const TAU = Math.PI * 2;
    let switches = 0;

    const posDirty = crowd.posDirty;
    for (let i = 0; i < n; i++) {
      const speed = Math.sqrt(vx[i] * vx[i] + vz[i] * vz[i]);

      // A stationary agent writes nothing: no position update, no dirty flag,
      // so its cell skips the gather and upload entirely. 1 mm/s is the floor.
      if (speed > 0.001) {
        const nx = px[i] + vx[i] * dt;
        const nz = pz[i] + vz[i] * dt;
        px[i] = nx; pz[i] = nz;
        nb[i * 4] = nx; nb[i * 4 + 1] = nz;

        let y = yaw[i];
        if (speed > 0.06) {
          let diff = Math.atan2(vx[i], vz[i]) - y;
          if (diff > Math.PI) diff -= TAU;
          else if (diff < -Math.PI) diff += TAU;
          y += diff * turnRate;
          yaw[i] = y;
        }

        const o = i * stride;
        data[o] = nx;
        data[o + 1] = 0;
        data[o + 2] = nz;
        data[o + 3] = y;
        posDirty[i] = 1;
      }

      // clip selection, throttled so agents cannot chatter at a band edge
      let want = bandClip[bandCount - 1];
      for (let b = 0; b < bandCount; b++) {
        if (speed < bands[b]) { want = bandClip[b]; break; }
      }
      if (want !== clip[i] && time >= nextSwitch[i]) {
        crowd.play(i, want, { rate: this.rateFor(want, speed, matchStride) });
        clip[i] = want;
        nextSwitch[i] = time + 0.35 + Math.random() * 0.3;
        switches++;
      } else if (matchStride) {
        crowd.setSpeed(i, speed);
      }
    }
    this.stats.clipSwitches = switches;
  }

  rateFor(clipIndex, speed, matchStride) {
    if (!matchStride) return this.asset.naturalRate(clipIndex);
    return this.asset.rateForSpeed(clipIndex, speed);
  }

  /** Re-derive every instance's playback rate (used when toggling stride matching). */
  refreshRates() {
    const n = this.crowd.count;
    const matchStride = this.options.matchStride;
    for (let i = 0; i < n; i++) {
      const speed = Math.sqrt(this.vx[i] * this.vx[i] + this.vz[i] * this.vz[i]);
      const clip = this.crowd.currentClip(i);
      this.crowd.setRate(i, this.rateFor(clip, speed, matchStride));
    }
  }
}

// gentle per-instance hue shift for assets whose color lives in the mesh
function subtleTint() {
  const h = Math.random();
  const amt = 0.18 * Math.random();
  const base = hsv(h, amt + 0.08, 1);
  return [0.86 + base[0] * 0.14, 0.86 + base[1] * 0.14, 0.86 + base[2] * 0.14];
}

function hsv(h, s, v) {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0: return [v, t, p];
    case 1: return [q, v, p];
    case 2: return [p, v, t];
    case 3: return [p, q, v];
    case 4: return [t, p, v];
    default: return [v, p, q];
  }
}

function hsvShirt() {
  // pleasant, desaturated apparel colours
  const h = Math.random();
  const s = 0.25 + Math.random() * 0.45;
  const v = 0.35 + Math.random() * 0.5;
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0: return [v, t, p];
    case 1: return [q, v, p];
    case 2: return [p, v, t];
    case 3: return [p, q, v];
    case 4: return [t, p, v];
    default: return [v, p, q];
  }
}
