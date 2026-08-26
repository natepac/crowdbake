// The paint minigun.
//
// Hitscan collision against the crowd is cheap because every instance is an
// analytic capsule: position + yaw + scale live in the interleaved instance
// record, and the crowd's chunk grid (rebuilt every frame for culling anyway)
// doubles as the broadphase. A shot samples grid cells along the ray's XZ
// track and tests only the instances in and around them -- a few dozen
// cylinder tests per round instead of 100k.
//
// Tracers and paint droplets are fire-and-forget GPU ring buffers: the CPU
// writes one record per round / per droplet at spawn, and the vertex shader
// evaluates flight from uTime alone, so a screenful of paint costs no per-frame
// CPU at all. Droplet landing times are solved analytically at spawn and
// scheduled as floor splats, which is why the speckle ring appears exactly
// where the droplets visually land.

import {
  AdditiveBlending,
  BufferGeometry,
  Float32BufferAttribute,
  LineSegments,
  Points,
  ShaderMaterial,
  Vector3,
} from 'three';

const TRACER_SPEED = 240;      // m/s, fast but visible
const TRACER_LEN = 7;          // metres of glowing streak
const MAX_TRACERS = 256;
const MAX_DROPLETS = 24576;
const GRAVITY = 14;            // heavier than earth reads better for paint

function paintColor() {
  // vivid, saturated paint hues
  const h = Math.random();
  const i = (h * 6) | 0, f = h * 6 - i;
  const q = 1 - f;
  const rgb = [[1, f, 0], [q, 1, 0], [0, 1, f], [0, q, 1], [f, 0, 1], [1, 0, q]][i % 6];
  const lift = 0.12;
  return [rgb[0] * (1 - lift) + lift, rgb[1] * (1 - lift) + lift, rgb[2] * (1 - lift) + lift];
}

export class GunSystem {
  constructor(scene, camera, { crowds, sims, obstacles = [], paint, worldSize = 420 }) {
    this.camera = camera;
    this.crowds = crowds;
    this.sims = sims;
    this.paint = paint;
    this.worldSize = worldSize;
    this.enabled = false;
    this.triggerDown = false;
    this.fireRate = 18;               // rounds per second
    this.kills = 0;
    this.shots = 0;
    this.time = 0;
    this._cooldown = 0;
    this.events = [];                 // { t, type, ... } sorted-ish by push order

    // pillars as blocking cylinders (height reconstructed from the demo's
    // pillar scale formula: radius = 1.35 * s + 0.6, mesh height 9 * s)
    this.pillars = obstacles.map((o) => {
      const s = Math.max((o.radius - 0.6) / 1.35, 0.3);
      return { x: o.x, z: o.z, r: o.radius, h: 9 * s };
    });

    // ---- tracer ring buffer -------------------------------------------
    {
      const geom = new BufferGeometry();
      const n = MAX_TRACERS * 2;
      geom.setAttribute('position', new Float32BufferAttribute(new Float32Array(n * 3), 3));
      geom.setAttribute('aStart', new Float32BufferAttribute(new Float32Array(n * 3), 3));
      geom.setAttribute('aDir', new Float32BufferAttribute(new Float32Array(n * 3), 3));
      geom.setAttribute('aInfo', new Float32BufferAttribute(new Float32Array(n * 3), 3)); // dist, t0, isHead
      this.tracerGeom = geom;
      this.tracerMat = new ShaderMaterial({
        transparent: true,
        blending: AdditiveBlending,
        depthWrite: false,
        uniforms: { uTime: { value: 0 } },
        vertexShader: /* glsl */`
          attribute vec3 aStart;
          attribute vec3 aDir;
          attribute vec3 aInfo;
          uniform float uTime;
          varying float vAlpha;
          void main() {
            float dist = aInfo.x;
            float dt = uTime - aInfo.y;
            float head = min(dt * ${TRACER_SPEED.toFixed(1)}, dist);
            float tail = max(head - ${TRACER_LEN.toFixed(1)}, 0.0);
            float d = aInfo.z > 0.5 ? head : tail;
            // dead tracers collapse to a degenerate segment far below ground
            bool dead = dt < 0.0 || tail >= dist;
            vec3 p = dead ? vec3(0.0, -1000.0, 0.0) : aStart + aDir * d;
            vAlpha = dead ? 0.0 : (1.0 - 0.35 * aInfo.z);
            gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
          }`,
        fragmentShader: /* glsl */`
          varying float vAlpha;
          void main() { gl_FragColor = vec4(1.0, 0.85, 0.35, vAlpha); }`,
      });
      this.tracers = new LineSegments(geom, this.tracerMat);
      this.tracers.frustumCulled = false;
      this.tracers.visible = false;
      scene.add(this.tracers);
      this._tracerCursor = 0;
    }

    // ---- droplet ring buffer ------------------------------------------
    {
      const geom = new BufferGeometry();
      geom.setAttribute('position', new Float32BufferAttribute(new Float32Array(MAX_DROPLETS * 3), 3));
      geom.setAttribute('aVel', new Float32BufferAttribute(new Float32Array(MAX_DROPLETS * 3), 3));
      geom.setAttribute('aInfo', new Float32BufferAttribute(new Float32Array(MAX_DROPLETS * 3), 3)); // t0, life, size
      geom.setAttribute('aColor', new Float32BufferAttribute(new Float32Array(MAX_DROPLETS * 3), 3));
      this.dropGeom = geom;
      this.dropMat = new ShaderMaterial({
        transparent: true,
        depthWrite: false,
        uniforms: { uTime: { value: 0 }, uScale: { value: 600 } },
        vertexShader: /* glsl */`
          attribute vec3 aVel;
          attribute vec3 aInfo;
          attribute vec3 aColor;
          uniform float uTime;
          uniform float uScale;
          varying vec3 vColor;
          varying float vAlive;
          void main() {
            float dt = uTime - aInfo.x;
            vAlive = (dt >= 0.0 && dt <= aInfo.y) ? 1.0 : 0.0;
            vec3 p = position + aVel * dt + vec3(0.0, -0.5 * ${GRAVITY.toFixed(1)} * dt * dt, 0.0);
            if (vAlive < 0.5) p = vec3(0.0, -1000.0, 0.0);
            vColor = aColor;
            vec4 mv = viewMatrix * vec4(p, 1.0);
            gl_PointSize = aInfo.z * uScale / max(-mv.z, 0.5);
            gl_Position = projectionMatrix * mv;
          }`,
        fragmentShader: /* glsl */`
          varying vec3 vColor;
          varying float vAlive;
          void main() {
            vec2 d = gl_PointCoord - 0.5;
            if (vAlive < 0.5 || dot(d, d) > 0.24) discard;
            gl_FragColor = vec4(vColor, 1.0);
          }`,
      });
      this.droplets = new Points(geom, this.dropMat);
      this.droplets.frustumCulled = false;
      this.droplets.visible = false;
      scene.add(this.droplets);
      this._dropCursor = 0;
    }

    this._origin = new Vector3();
    this._dir = new Vector3();
    this._right = new Vector3();
    this._up = new Vector3();
  }

  setEnabled(on) {
    this.enabled = on;
    this.tracers.visible = on || this._tracerCursor > 0;
    this.droplets.visible = true;
    if (!on) this.triggerDown = false;
  }

  // ------------------------------------------------------------- firing ---

  update(dt) {
    this.time += dt;
    this.tracerMat.uniforms.uTime.value = this.time;
    this.dropMat.uniforms.uTime.value = this.time;
    this.dropMat.uniforms.uScale.value = innerHeight * 0.55;

    if (this.enabled && this.triggerDown) {
      this._cooldown -= dt;
      const step = 1 / this.fireRate;
      // fire every round the cooldown owes us, so low fps still means full dakka
      let guard = 8;
      while (this._cooldown <= 0 && guard-- > 0) {
        this.fireFromCamera();
        this._cooldown += step;
      }
    } else {
      this._cooldown = Math.min(this._cooldown, 0);
    }

    // due events: paint bursts and droplet landings
    const due = [];
    for (let i = this.events.length - 1; i >= 0; i--) {
      if (this.events[i].t <= this.time) { due.push(this.events[i]); this.events[i] = this.events[this.events.length - 1]; this.events.pop(); }
    }
    for (const e of due) {
      if (e.type === 'burst') this.burst(e.x, e.y, e.z, e.big);
      else this.paint.add(e.x, e.z, e.r, e.color);
    }
  }

  fireFromCamera() {
    const cam = this.camera;
    cam.getWorldDirection(this._dir);
    this._right.set(1, 0, 0).applyQuaternion(cam.quaternion);
    this._up.set(0, 1, 0).applyQuaternion(cam.quaternion);
    // muzzle sits low-right of the eye so tracers visibly streak to the aim point
    this._origin.copy(cam.position)
      .addScaledVector(this._dir, 0.55)
      .addScaledVector(this._right, 0.22)
      .addScaledVector(this._up, -0.16);
    this.fire(this._origin, this._dir);
  }

  /** Hitscan one round; no spread, no drop, infinite ammo. */
  fire(origin, dir) {
    this.shots++;
    const maxRange = 420;

    // ground plane (y = 0)
    let bestT = Infinity;
    let hit = null;
    if (dir.y < -1e-5) {
      const t = -origin.y / dir.y;
      if (t > 0 && t < maxRange) { bestT = t; hit = { type: 'ground' }; }
    }

    // pillars
    for (const p of this.pillars) {
      const t = rayCylinder(origin, dir, p.x, p.z, p.r, 0, p.h);
      if (t !== null && t < bestT) { bestT = t; hit = { type: 'pillar' }; }
    }

    // crowd instances via each crowd's chunk grid
    for (let k = 0; k < this.crowds.length; k++) {
      const res = this.hitCrowd(this.crowds[k], origin, dir, Math.min(bestT, maxRange));
      if (res && res.t < bestT) { bestT = res.t; hit = { type: 'crowd', crowd: k, index: res.index }; }
    }

    const dist = Math.min(bestT, maxRange);
    this.spawnTracer(origin, dir, dist);
    if (!hit) return;

    const hx = origin.x + dir.x * dist;
    const hy = origin.y + dir.y * dist;
    const hz = origin.z + dir.z * dist;
    const arrive = this.time + dist / TRACER_SPEED;

    if (hit.type === 'crowd') {
      // the character dies now (hitscan); the eruption arrives with the tracer
      this.sims[hit.crowd].removeAgent(hit.index);
      this.kills++;
      this.events.push({ t: arrive, type: 'burst', x: hx, y: Math.max(hy, 0.4), z: hz, big: true });
    } else if (hit.type === 'pillar') {
      this.events.push({ t: arrive, type: 'burst', x: hx, y: Math.max(hy, 0.3), z: hz, big: false });
    } else {
      // stray rounds paint the floor where they land
      this.events.push({ t: arrive, type: 'splat', x: hx, z: hz, r: 0.22 + Math.random() * 0.3, color: paintColor() });
    }
  }

  hitCrowd(crowd, origin, dir, maxT) {
    const n = crowd.count;
    if (n === 0) return null;
    const data = crowd.data;
    const radius = crowd.asset.bounds.maxRadiusXZ * 0.6 + 0.12;
    const height = crowd.asset.instanceHeight;
    const { gridN, cellSize, gridOrigin } = crowd;
    const counts = crowd._counts, offsets = crowd._offsets, sorted = crowd._sorted;

    let best = null;
    const visited = new Set();
    const step = Math.max(cellSize * 0.45, 2);
    const steps = Math.ceil(maxT / step) + 1;
    const last = gridN - 1;

    for (let si = 0; si <= steps; si++) {
      const t = Math.min(si * step, maxT);
      const sx = origin.x + dir.x * t;
      const sz = origin.z + dir.z * t;
      const cx = Math.max(0, Math.min(last, ((sx - gridOrigin) / cellSize) | 0));
      const cz = Math.max(0, Math.min(last, ((sz - gridOrigin) / cellSize) | 0));
      for (let oz = -1; oz <= 1; oz++) {
        const zz = cz + oz;
        if (zz < 0 || zz > last) continue;
        for (let ox = -1; ox <= 1; ox++) {
          const xx = cx + ox;
          if (xx < 0 || xx > last) continue;
          const cell = zz * gridN + xx;
          if (visited.has(cell)) continue;
          visited.add(cell);
          const s0 = offsets[cell], s1 = s0 + counts[cell];
          for (let si2 = s0; si2 < s1; si2++) {
            const i = sorted[si2];
            if (i >= n) continue;
            const o = i * 16;
            const scale = data[o + 7];
            const tHit = rayCylinder(origin, dir, data[o], data[o + 2], radius * scale, data[o + 1], data[o + 1] + height * scale);
            if (tHit !== null && tHit < (best ? best.t : maxT)) best = { t: tHit, index: i };
          }
        }
      }
      if (t >= maxT) break;
    }
    return best;
  }

  // ------------------------------------------------------------ visuals ---

  spawnTracer(origin, dir, dist) {
    const i = this._tracerCursor % MAX_TRACERS;
    this._tracerCursor++;
    const g = this.tracerGeom;
    const start = g.attributes.aStart.array;
    const adir = g.attributes.aDir.array;
    const info = g.attributes.aInfo.array;
    for (let v = 0; v < 2; v++) {
      const o = (i * 2 + v) * 3;
      start[o] = origin.x; start[o + 1] = origin.y; start[o + 2] = origin.z;
      adir[o] = dir.x; adir[o + 1] = dir.y; adir[o + 2] = dir.z;
      info[o] = dist; info[o + 1] = this.time; info[o + 2] = v === 0 ? 1 : 0;
    }
    for (const name of ['aStart', 'aDir', 'aInfo']) {
      const a = g.attributes[name];
      a.clearUpdateRanges();
      a.addUpdateRange(i * 2 * 3, 6);
      a.needsUpdate = true;
    }
    this.tracers.visible = true;
  }

  /** Fountain of paint at (x, y, z): droplets fly, land, and speckle the floor. */
  burst(x, y, z, big) {
    const color = paintColor();
    // the main splat under the victim, plus one or two offset overlaps
    const mainR = big ? 1.5 + Math.random() * 0.9 : 0.5 + Math.random() * 0.3;
    this.paint.add(x, z, mainR, color);
    if (big) {
      this.paint.add(x + (Math.random() - 0.5) * 1.2, z + (Math.random() - 0.5) * 1.2, mainR * 0.55, paintColor());
    }

    const count = big ? 130 : 30;
    const g = this.dropGeom;
    const pos = g.attributes.position.array;
    const vel = g.attributes.aVel.array;
    const info = g.attributes.aInfo.array;
    const col = g.attributes.aColor.array;
    const lo = this._dropCursor % MAX_DROPLETS;

    for (let d = 0; d < count; d++) {
      const i = (this._dropCursor++) % MAX_DROPLETS;
      const o = i * 3;
      // each droplet gets its own hue -- the multi-color part of the fountain
      const c = Math.random() < 0.6 ? color : paintColor();
      const ang = Math.random() * Math.PI * 2;
      const horiz = (big ? 1.4 : 0.8) * (0.4 + Math.random() * Math.random() * 2.4);
      const vy = (big ? 5.2 : 3.2) + Math.random() * (big ? 5.5 : 2.4);
      const vx = Math.cos(ang) * horiz;
      const vz = Math.sin(ang) * horiz;

      pos[o] = x; pos[o + 1] = y; pos[o + 2] = z;
      vel[o] = vx; vel[o + 1] = vy; vel[o + 2] = vz;
      col[o] = c[0]; col[o + 1] = c[1]; col[o + 2] = c[2];

      // land when y + vy t - g/2 t^2 = 0
      const tLand = (vy + Math.sqrt(vy * vy + 2 * GRAVITY * y)) / GRAVITY;
      info[o] = this.time;
      info[o + 1] = tLand;
      info[o + 2] = big ? 0.05 + Math.random() * 0.07 : 0.035 + Math.random() * 0.04;

      // the landing speckle, scheduled for the moment the droplet visually lands
      if (this.events.length < 4000) {
        this.events.push({
          t: this.time + tLand,
          type: 'splat',
          x: x + vx * tLand,
          z: z + vz * tLand,
          r: 0.1 + Math.random() * 0.28,
          color: c,
        });
      }
    }

    // droplet writes may wrap the ring; update the whole touched span cheaply
    for (const name of ['position', 'aVel', 'aInfo', 'aColor']) {
      const a = g.attributes[name];
      a.clearUpdateRanges();
      if (lo + count <= MAX_DROPLETS) a.addUpdateRange(lo * 3, count * 3);
      else { a.addUpdateRange(lo * 3, (MAX_DROPLETS - lo) * 3); a.addUpdateRange(0, (lo + count - MAX_DROPLETS) * 3); }
      a.needsUpdate = true;
    }
    this.droplets.visible = true;
  }
}

/**
 * Ray vs vertical cylinder (axis-aligned, finite). Returns nearest positive t
 * or null. Cap hits count as body hits, close enough for a minigun.
 */
function rayCylinder(origin, dir, cx, cz, r, y0, y1) {
  const ox = origin.x - cx, oz = origin.z - cz;
  const a = dir.x * dir.x + dir.z * dir.z;
  if (a < 1e-9) {
    // vertical ray: inside the circle?
    if (ox * ox + oz * oz > r * r) return null;
    const t = dir.y > 0 ? (y0 - origin.y) / dir.y : (y1 - origin.y) / dir.y;
    return t > 0 ? t : null;
  }
  const b = 2 * (ox * dir.x + oz * dir.z);
  const c = ox * ox + oz * oz - r * r;
  const disc = b * b - 4 * a * c;
  if (disc < 0) return null;
  const sq = Math.sqrt(disc);
  let t = (-b - sq) / (2 * a);
  if (t < 0) t = (-b + sq) / (2 * a);
  if (t < 0) return null;
  const y = origin.y + dir.y * t;
  if (y < y0 || y > y1) return null;
  return t;
}
