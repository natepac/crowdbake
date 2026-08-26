// Procedural rigged humanoid + procedural clips.
//
// Exists so the whole pipeline runs with zero external assets. The gaits are
// built from an explicit foot trajectory solved with 2-bone IK, which means the
// contact foot is *exactly* planted and the stride length is known analytically
// -- that gives the stride extractor in stride.mjs a ground truth to be checked
// against (see tools/verify.mjs).

import { Matrix4, Quaternion, Vector3 } from 'three';

const DEG = Math.PI / 180;

// ---------------------------------------------------------------- skeleton --

// rest pose, world space, character faces +Z, up is +Y, units are metres
const SKELETON = [
  ['root',       -1, [0, 0.00, 0]],
  ['hips',        0, [0, 0.95, 0]],
  ['spine',       1, [0, 1.10, 0]],
  ['chest',       2, [0, 1.28, 0]],
  ['neck',        3, [0, 1.46, 0]],
  ['head',        4, [0, 1.56, 0]],
  ['headTop',     5, [0, 1.78, 0]],
  ['shoulderL',   3, [0.06, 1.42, 0]],
  ['upperArmL',   7, [0.19, 1.42, 0]],
  ['lowerArmL',   8, [0.19, 1.14, 0]],
  ['handL',       9, [0.19, 0.90, 0]],
  ['handLEnd',   10, [0.19, 0.78, 0]],
  ['shoulderR',   3, [-0.06, 1.42, 0]],
  ['upperArmR',  12, [-0.19, 1.42, 0]],
  ['lowerArmR',  13, [-0.19, 1.14, 0]],
  ['handR',      14, [-0.19, 0.90, 0]],
  ['handREnd',   15, [-0.19, 0.78, 0]],
  ['thighL',      1, [0.09, 0.92, 0]],
  ['shinL',      17, [0.09, 0.52, 0]],
  ['footL',      18, [0.09, 0.10, 0]],
  ['toeL',       19, [0.09, 0.03, 0.11]],
  ['thighR',      1, [-0.09, 0.92, 0]],
  ['shinR',      21, [-0.09, 0.52, 0]],
  ['footR',      22, [-0.09, 0.10, 0]],
  ['toeR',       23, [-0.09, 0.03, 0.11]],
];

const BONE = {};
SKELETON.forEach(([name], i) => { BONE[name] = i; });

// tube segments: [startBone, endBone, radiusStart, radiusEnd, material]
const LIMBS = [
  ['hips', 'spine', 0.15, 0.16, 'pants'],
  ['spine', 'chest', 0.16, 0.19, 'shirt'],
  ['chest', 'neck', 0.19, 0.07, 'shirt'],
  ['neck', 'head', 0.055, 0.075, 'skin'],
  ['shoulderL', 'upperArmL', 0.085, 0.062, 'shirt'],
  ['upperArmL', 'lowerArmL', 0.062, 0.048, 'shirt'],
  ['lowerArmL', 'handL', 0.048, 0.036, 'skin'],
  ['handL', 'handLEnd', 0.045, 0.028, 'skin'],
  ['shoulderR', 'upperArmR', 0.085, 0.062, 'shirt'],
  ['upperArmR', 'lowerArmR', 0.062, 0.048, 'shirt'],
  ['lowerArmR', 'handR', 0.048, 0.036, 'skin'],
  ['handR', 'handREnd', 0.045, 0.028, 'skin'],
  ['thighL', 'shinL', 0.098, 0.070, 'pants'],
  ['shinL', 'footL', 0.070, 0.048, 'pants'],
  ['footL', 'toeL', 0.058, 0.045, 'shoe'],
  ['thighR', 'shinR', 0.098, 0.070, 'pants'],
  ['shinR', 'footR', 0.070, 0.048, 'pants'],
  ['footR', 'toeR', 0.058, 0.045, 'shoe'],
];

const PALETTE = {
  skin: [0.86, 0.68, 0.56],
  shirt: [0.30, 0.42, 0.72],
  pants: [0.24, 0.26, 0.32],
  shoe: [0.12, 0.12, 0.14],
  hair: [0.24, 0.18, 0.14],
};

// materialId is handed to the runtime as a vertex attribute so per-instance
// palettes can recolour shirt/pants/skin independently.
const MATERIAL_ID = { skin: 0, shirt: 1, pants: 2, shoe: 3, hair: 4 };

// ------------------------------------------------------------------- rig ----

function buildRestPose() {
  const n = SKELETON.length;
  const parent = new Int32Array(n);
  const restWorld = new Float32Array(n * 3);
  const restLocal = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const [, p, pos] = SKELETON[i];
    parent[i] = p;
    restWorld[i * 3] = pos[0]; restWorld[i * 3 + 1] = pos[1]; restWorld[i * 3 + 2] = pos[2];
  }
  for (let i = 0; i < n; i++) {
    const p = parent[i];
    for (let c = 0; c < 3; c++) {
      restLocal[i * 3 + c] = restWorld[i * 3 + c] - (p >= 0 ? restWorld[p * 3 + c] : 0);
    }
  }
  return { parent, restWorld, restLocal, boneCount: n };
}

// ------------------------------------------------------------------ mesh ----

function generateMesh(rest, density) {
  const radial = Math.max(6, Math.round(10 * density));
  const rings = Math.max(3, Math.round(5 * density));

  const pos = [], nrm = [], uv = [], col = [], mat = [], si = [], sw = [], idx = [];
  const up = new Vector3(), axis = new Vector3(), tanA = new Vector3(), tanB = new Vector3();
  const P = (b) => new Vector3(rest.restWorld[b * 3], rest.restWorld[b * 3 + 1], rest.restWorld[b * 3 + 2]);

  const pushVertex = (p, n, u, v, matName, boneA, boneB, wB) => {
    pos.push(p.x, p.y, p.z);
    nrm.push(n.x, n.y, n.z);
    uv.push(u, v);
    const c = PALETTE[matName];
    col.push(c[0], c[1], c[2]);
    mat.push(MATERIAL_ID[matName]);
    si.push(boneA, boneB, 0, 0);
    sw.push(1 - wB, wB, 0, 0);
  };

  for (const [aName, bName, r0, r1, matName] of LIMBS) {
    const a = BONE[aName], b = BONE[bName];
    const pa = P(a), pb = P(b);
    axis.copy(pb).sub(pa);
    const len = axis.length();
    if (len < 1e-6) continue;
    axis.divideScalar(len);
    up.set(0, 0, 1);
    if (Math.abs(axis.dot(up)) > 0.9) up.set(1, 0, 0);
    tanA.copy(up).cross(axis).normalize();
    tanB.copy(axis).cross(tanA).normalize();

    const parentBone = rest.parent[a] >= 0 ? rest.parent[a] : a;
    const base = pos.length / 3;

    for (let r = 0; r <= rings; r++) {
      const t = r / rings;
      const radius = r0 + (r1 - r0) * t;
      // blend to the child bone near the far cap and to the parent near the near cap
      let wB = smoothstep(0.55, 1.0, t) * 0.5;
      const wParent = smoothstep(0.35, 0.0, t) * 0.4;
      let boneA = a, boneB = b;
      if (wParent > wB) { boneA = a; boneB = parentBone; wB = wParent; }
      const center = pa.clone().addScaledVector(axis, len * t);
      for (let s = 0; s < radial; s++) {
        const ang = (s / radial) * Math.PI * 2;
        const cs = Math.cos(ang), sn = Math.sin(ang);
        const nx = tanA.x * cs + tanB.x * sn;
        const ny = tanA.y * cs + tanB.y * sn;
        const nz = tanA.z * cs + tanB.z * sn;
        const nv = new Vector3(nx, ny, nz);
        const pv = center.clone().addScaledVector(nv, radius);
        pushVertex(pv, nv, s / radial, t, matName, boneA, boneB, wB);
      }
    }
    for (let r = 0; r < rings; r++) {
      for (let s = 0; s < radial; s++) {
        const s1 = (s + 1) % radial;
        const i0 = base + r * radial + s;
        const i1 = base + r * radial + s1;
        const i2 = base + (r + 1) * radial + s;
        const i3 = base + (r + 1) * radial + s1;
        idx.push(i0, i2, i1, i1, i2, i3);
      }
    }
    // caps
    for (const [ring, dir, capBone] of [[0, -1, a], [rings, 1, b]]) {
      const centerP = pa.clone().addScaledVector(axis, len * (ring / rings));
      const nv = axis.clone().multiplyScalar(dir);
      const ci = pos.length / 3;
      const wB = ring === 0 ? 0 : 0.5;
      pushVertex(centerP, nv, 0.5, ring === 0 ? 0 : 1, matName, ring === 0 ? a : a, ring === 0 ? a : b, wB);
      for (let s = 0; s < radial; s++) {
        const s1 = (s + 1) % radial;
        const rim = base + ring * radial;
        if (dir < 0) idx.push(ci, rim + s1, rim + s);
        else idx.push(ci, rim + s, rim + s1);
        void capBone;
      }
    }
  }

  // head: squashed uv-sphere on the head bone
  {
    const hb = BONE.head;
    const c = P(hb).add(new Vector3(0, 0.10, 0.005));
    const rx = 0.098, ry = 0.125, rz = 0.108;
    const segU = Math.max(8, Math.round(14 * density));
    const segV = Math.max(6, Math.round(10 * density));
    const base = pos.length / 3;
    for (let v = 0; v <= segV; v++) {
      const phi = (v / segV) * Math.PI;
      for (let u = 0; u <= segU; u++) {
        const th = (u / segU) * Math.PI * 2;
        const sx = Math.sin(phi) * Math.cos(th);
        const sy = Math.cos(phi);
        const sz = Math.sin(phi) * Math.sin(th);
        const p = new Vector3(c.x + sx * rx, c.y + sy * ry, c.z + sz * rz);
        const n = new Vector3(sx / rx, sy / ry, sz / rz).normalize();
        const isHair = sy > 0.15 || (sy > -0.25 && sz < -0.25);
        pushVertex(p, n, u / segU, v / segV, isHair ? 'hair' : 'skin', hb, hb, 0);
      }
    }
    for (let v = 0; v < segV; v++) {
      for (let u = 0; u < segU; u++) {
        const i0 = base + v * (segU + 1) + u;
        const i1 = i0 + 1;
        const i2 = i0 + (segU + 1);
        const i3 = i2 + 1;
        idx.push(i0, i2, i1, i1, i2, i3);
      }
    }
  }

  return {
    positions: new Float32Array(pos),
    normals: new Float32Array(nrm),
    uvs: new Float32Array(uv),
    colors: new Float32Array(col),
    materialIds: new Float32Array(mat),
    skinIndex: new Uint16Array(si),
    skinWeight: new Float32Array(sw),
    indices: new Uint32Array(idx),
  };
}

function smoothstep(a, b, x) {
  const t = Math.min(Math.max((x - a) / (b - a || 1e-6), 0), 1);
  return t * t * (3 - 2 * t);
}

// ------------------------------------------------------------- procedural ---
// Every clip writes local rotations (quaternions) per bone plus a hips offset.

const AXIS_X = new Vector3(1, 0, 0);
const AXIS_Y = new Vector3(0, 1, 0);
const AXIS_Z = new Vector3(0, 0, 1);

function makePose(boneCount) {
  const rot = new Float32Array(boneCount * 4);
  for (let i = 0; i < boneCount; i++) rot[i * 4 + 3] = 1;
  return { rot, hips: new Float32Array(3) };
}

const _axisQ = new Quaternion();
function _setAxis(axis, angle) { return _axisQ.setFromAxisAngle(axis, angle); }

function setQuat(pose, bone, q) {
  pose.rot[bone * 4] = q.x; pose.rot[bone * 4 + 1] = q.y;
  pose.rot[bone * 4 + 2] = q.z; pose.rot[bone * 4 + 3] = q.w;
}

function setRot(pose, bone, x, y, z) {
  const q = new Quaternion();
  if (x) q.multiply(new Quaternion().setFromAxisAngle(AXIS_X, x));
  if (y) q.multiply(new Quaternion().setFromAxisAngle(AXIS_Y, y));
  if (z) q.multiply(new Quaternion().setFromAxisAngle(AXIS_Z, z));
  pose.rot[bone * 4] = q.x; pose.rot[bone * 4 + 1] = q.y; pose.rot[bone * 4 + 2] = q.z; pose.rot[bone * 4 + 3] = q.w;
}

const V_DOWN = new Vector3(0, -1, 0);
const _ikD = new Vector3();
const _ikU = new Vector3();
const _ikPole = new Vector3();
const _ikKnee = new Vector3();
const _ikTmp = new Vector3();
const _ikQ = new Quaternion();

/**
 * 3D two-bone IK, solved in the parent bone's frame.
 *
 * The naive version of this solves in the sagittal plane and assumes the pelvis
 * is unrotated. That is wrong the moment the clip adds pelvic twist or lean: the
 * leg chain inherits the parent rotation, the ankle misses its target, and the
 * "planted" foot slides by a few millimetres per frame -- exactly the artefact
 * the whole stride-matching feature exists to remove. Solving in the parent
 * frame keeps the contact foot nailed to its world-space target whatever the
 * pelvis is doing.
 *
 * All bones rest pointing -Y, so a bone's local rotation is just the quaternion
 * taking -Y to the direction it should point.
 */
function solveLeg3D(qParent, thighWorld, targetWorld, l1, l2, poleWorld, outThigh, outShin) {
  const invParent = _ikQ.copy(qParent).invert();
  _ikD.copy(targetWorld).sub(thighWorld).applyQuaternion(invParent);

  let dist = _ikD.length();
  const maxD = (l1 + l2) * 0.9995;
  const minD = Math.abs(l1 - l2) + 1e-4;
  if (dist > maxD) { _ikD.multiplyScalar(maxD / dist); dist = maxD; }
  else if (dist < minD) { _ikD.multiplyScalar(minD / (dist || 1e-6)); dist = minD; }

  _ikU.copy(_ikD).divideScalar(dist);

  // knee points along the pole direction, projected perpendicular to the limb
  _ikPole.copy(poleWorld).applyQuaternion(invParent);
  _ikPole.addScaledVector(_ikU, -_ikPole.dot(_ikU));
  if (_ikPole.lengthSq() < 1e-10) {
    _ikPole.set(0, 0, 1).addScaledVector(_ikU, -_ikU.z);
    if (_ikPole.lengthSq() < 1e-10) _ikPole.set(1, 0, 0).addScaledVector(_ikU, -_ikU.x);
  }
  _ikPole.normalize();

  const cosA = Math.min(Math.max((l1 * l1 + dist * dist - l2 * l2) / (2 * l1 * dist), -1), 1);
  const alpha = Math.acos(cosA);
  _ikKnee.copy(_ikU).multiplyScalar(l1 * Math.cos(alpha)).addScaledVector(_ikPole, l1 * Math.sin(alpha));

  outThigh.setFromUnitVectors(V_DOWN, _ikTmp.copy(_ikKnee).normalize());
  _ikTmp.copy(_ikD).sub(_ikKnee).normalize().applyQuaternion(_ikQ.copy(outThigh).invert());
  outShin.setFromUnitVectors(V_DOWN, _ikTmp);
}

/**
 * Locomotion clip generator.
 * stride  - metres travelled per full cycle (two steps)
 * period  - seconds per full cycle
 * stanceF - fraction of the cycle each foot spends planted (>0.5 = walk, <0.5 = run)
 */
function makeLocomotion(rest, cfg) {
  const { stride, period, stanceF, bob, lean, armSwing, stepHeight, hipDrop } = cfg;
  const W = (b) => new Vector3(rest.restWorld[b * 3], rest.restWorld[b * 3 + 1], rest.restWorld[b * 3 + 2]);
  const hipsRest = W(BONE.hips);
  const thighRest = [W(BONE.thighL), W(BONE.thighR)];
  const l1 = thighRest[0].distanceTo(W(BONE.shinL));
  const l2 = W(BONE.shinL).distanceTo(W(BONE.footL));
  const restAnkleY = W(BONE.footL).y;

  // relative travel of a planted foot during its stance phase
  const travel = stride * stanceF;

  const qHips = new Quaternion();
  const qThigh = new Quaternion();
  const qShin = new Quaternion();
  const qChain = new Quaternion();
  const qFoot = new Quaternion();
  const hipsWorld = new Vector3();
  const thighWorld = new Vector3();
  const target = new Vector3();
  const pole = new Vector3(0, 0, 1);
  const offset = new Vector3();

  return function pose(t, out) {
    const cyc = ((t / period) % 1 + 1) % 1;
    out.hips[0] = 0;
    out.hips[1] = bob * Math.cos(4 * Math.PI * cyc) - hipDrop;
    out.hips[2] = 0;

    // pelvis first -- the legs are solved underneath whatever it is doing
    const twist = 0.055 * Math.sin(2 * Math.PI * cyc);
    qHips.setFromAxisAngle(AXIS_X, lean * 0.35)
      .multiply(_setAxis(AXIS_Y, -twist))
      .multiply(_setAxis(AXIS_Z, lean * 0.06 * Math.sin(2 * Math.PI * cyc)));
    setQuat(out, BONE.hips, qHips);
    hipsWorld.copy(hipsRest).add(offset.set(out.hips[0], out.hips[1], out.hips[2]));

    for (const side of [0, 1]) {
      const ph = ((cyc + (side ? 0.5 : 0)) % 1);
      const thigh = side ? BONE.thighR : BONE.thighL;
      const shin = side ? BONE.shinR : BONE.shinL;
      const foot = side ? BONE.footR : BONE.footL;

      let footZ, footY, footPitch;
      if (ph < stanceF) {
        const u = ph / stanceF;
        footZ = travel * (0.5 - u);
        footY = restAnkleY;
        // heel strike -> roll through -> toe off
        footPitch = -0.30 * Math.pow(1 - u, 2) + 0.55 * Math.pow(u, 3);
      } else {
        const u = (ph - stanceF) / (1 - stanceF);
        footZ = travel * (u - 0.5);
        footY = restAnkleY + stepHeight * Math.sin(Math.PI * u);
        footPitch = 0.55 * Math.pow(1 - u, 3) - 0.22 * smoothstep(0.4, 1.0, u);
      }

      // thigh joint follows the pelvis; the foot target does not
      thighWorld.copy(thighRest[side]).sub(hipsRest).applyQuaternion(qHips).add(hipsWorld);
      target.set(thighRest[side].x, footY, footZ);

      solveLeg3D(qHips, thighWorld, target, l1, l2, pole, qThigh, qShin);
      setQuat(out, thigh, qThigh);
      setQuat(out, shin, qShin);

      // hold the foot at the authored world pitch regardless of the leg chain
      qChain.copy(qHips).multiply(qThigh).multiply(qShin).invert();
      qFoot.copy(qChain).multiply(_setAxis(AXIS_X, footPitch));
      setQuat(out, foot, qFoot);

      // arms swing opposite the same-side leg
      const arm = side ? BONE.upperArmR : BONE.upperArmL;
      const fore = side ? BONE.lowerArmR : BONE.lowerArmL;
      const swing = armSwing * Math.cos(2 * Math.PI * ph);
      setRot(out, arm, swing, 0, (side ? -1 : 1) * (0.08 + 0.05 * Math.abs(swing)));
      setRot(out, fore, -Math.abs(swing) * 0.55 - 0.15, 0, 0);
    }

    setRot(out, BONE.spine, lean * 0.35, twist * 0.6, 0.02 * Math.sin(2 * Math.PI * cyc));
    setRot(out, BONE.chest, lean * 0.3, twist * 0.9, 0);
    setRot(out, BONE.neck, -lean * 0.6, -twist * 0.5, 0);
    setRot(out, BONE.head, -lean * 0.4, 0, 0.03 * Math.sin(2 * Math.PI * cyc));
  };
}

function makeIdle(rest, cfg) {
  const { period, sway } = cfg;
  return function pose(t, out) {
    const c = (t / period) * Math.PI * 2;
    out.hips[0] = sway * 0.35 * Math.sin(c * 0.5);
    out.hips[1] = -0.012 + 0.012 * Math.cos(c);
    out.hips[2] = 0;
    setRot(out, BONE.hips, 0.02, 0.06 * Math.sin(c * 0.5), 0.03 * Math.sin(c * 0.5));
    setRot(out, BONE.spine, 0.015 + 0.012 * Math.cos(c), -0.04 * Math.sin(c * 0.5), 0);
    setRot(out, BONE.chest, 0.02 * Math.cos(c), -0.03 * Math.sin(c * 0.5), 0);
    setRot(out, BONE.neck, -0.03 * Math.cos(c), 0.10 * Math.sin(c * 0.33), 0);
    setRot(out, BONE.head, -0.02, 0.14 * Math.sin(c * 0.33 + 0.7), 0);
    for (const side of [0, 1]) {
      const s = side ? -1 : 1;
      setRot(out, side ? BONE.upperArmR : BONE.upperArmL, 0.05 * Math.sin(c + side), 0, s * 0.11);
      setRot(out, side ? BONE.lowerArmR : BONE.lowerArmL, -0.30 - 0.06 * Math.cos(c + side), 0, 0);
      setRot(out, side ? BONE.thighR : BONE.thighL, 0.02, 0, s * 0.03);
      setRot(out, side ? BONE.shinR : BONE.shinL, 0.05, 0, 0);
      setRot(out, side ? BONE.footR : BONE.footL, -0.07, 0, 0);
    }
    void rest;
  };
}

function makeWave(rest, cfg) {
  const idle = makeIdle(rest, { period: cfg.period * 1.5, sway: 0.01 });
  return function pose(t, out) {
    idle(t, out);
    const c = (t / cfg.period) * Math.PI * 2;
    const raise = smoothstep(0, 0.25, (t / cfg.period) % 1) * smoothstep(1, 0.75, (t / cfg.period) % 1);
    setRot(out, BONE.upperArmR, -0.15, 0, -(0.15 + 2.35 * raise));
    setRot(out, BONE.lowerArmR, -0.2, 0, -0.5 * raise + 0.55 * raise * Math.sin(c * 3));
    setRot(out, BONE.handR, 0, 0, 0.35 * raise * Math.sin(c * 3 + 0.6));
    setRot(out, BONE.head, -0.05, -0.12, 0.04);
  };
}

function makeCheer(rest, cfg) {
  return function pose(t, out) {
    const c = (t / cfg.period) * Math.PI * 2;
    const jump = Math.max(0, Math.sin(c));
    out.hips[0] = 0;
    out.hips[1] = -0.10 * (1 - jump) + 0.16 * Math.pow(jump, 1.5);
    out.hips[2] = 0;
    setRot(out, BONE.hips, 0.05 * (1 - jump), 0.10 * Math.sin(c * 0.5), 0);
    setRot(out, BONE.spine, -0.08 * jump + 0.10 * (1 - jump), 0, 0);
    setRot(out, BONE.chest, -0.10 * jump, 0, 0);
    setRot(out, BONE.head, -0.15 * jump, 0, 0);
    for (const side of [0, 1]) {
      const s = side ? -1 : 1;
      setRot(out, side ? BONE.upperArmR : BONE.upperArmL, -0.1, 0, s * (0.3 + 2.4 * jump));
      setRot(out, side ? BONE.lowerArmR : BONE.lowerArmL, -0.4 + 0.3 * jump, 0, 0);
      setRot(out, side ? BONE.thighR : BONE.thighL, 0.55 * (1 - jump), 0, s * 0.05);
      setRot(out, side ? BONE.shinR : BONE.shinL, 0.9 * (1 - jump), 0, 0);
      setRot(out, side ? BONE.footR : BONE.footL, -0.5 * (1 - jump) + 0.4 * jump, 0, 0);
    }
    void rest;
  };
}

// ------------------------------------------------------------------ build ---

export function makeProceduralRig(opts = {}) {
  const density = opts.density != null ? opts.density : 1.4;
  const rest = buildRestPose();
  const mesh = generateMesh(rest, density);
  const boneCount = rest.boneCount;

  const clipDefs = [
    { name: 'idle', duration: 4.0, loop: true, pose: makeIdle(rest, { period: 4.0, sway: 0.02 }), stride: 0 },
    { name: 'walk', duration: 1.06, loop: true, stride: 1.48, pose: makeLocomotion(rest, { stride: 1.48, period: 1.06, stanceF: 0.62, bob: 0.028, lean: 0.05, armSwing: 0.42, stepHeight: 0.10, hipDrop: 0.025 }) },
    { name: 'jog', duration: 0.80, loop: true, stride: 2.28, pose: makeLocomotion(rest, { stride: 2.28, period: 0.80, stanceF: 0.46, bob: 0.045, lean: 0.14, armSwing: 0.72, stepHeight: 0.17, hipDrop: 0.05 }) },
    { name: 'run', duration: 0.66, loop: true, stride: 3.30, pose: makeLocomotion(rest, { stride: 3.30, period: 0.66, stanceF: 0.36, bob: 0.060, lean: 0.26, armSwing: 1.02, stepHeight: 0.26, hipDrop: 0.075 }) },
    { name: 'wave', duration: 2.4, loop: true, pose: makeWave(rest, { period: 2.4 }), stride: 0 },
    { name: 'cheer', duration: 1.2, loop: true, pose: makeCheer(rest, { period: 1.2 }), stride: 0 },
  ];

  // rest world matrices -> inverse bind
  const inverseBind = new Float32Array(boneCount * 16);
  const m = new Matrix4();
  for (let b = 0; b < boneCount; b++) {
    m.makeTranslation(rest.restWorld[b * 3], rest.restWorld[b * 3 + 1], rest.restWorld[b * 3 + 2]);
    m.invert().toArray(inverseBind, b * 16);
  }

  const poseBuf = makePose(boneCount);
  const world = new Float32Array(boneCount * 16);
  const mA = new Matrix4(), mB = new Matrix4();
  const vp = new Vector3(), vq = new Quaternion(), vs = new Vector3(1, 1, 1);
  const hipsOffset = new Vector3();

  function sampleJointWorld(clipIndex, time, outWorld) {
    const def = clipDefs[clipIndex];
    for (let i = 0; i < boneCount; i++) {
      poseBuf.rot[i * 4] = 0; poseBuf.rot[i * 4 + 1] = 0; poseBuf.rot[i * 4 + 2] = 0; poseBuf.rot[i * 4 + 3] = 1;
    }
    poseBuf.hips[0] = poseBuf.hips[1] = poseBuf.hips[2] = 0;
    if (def) def.pose(time, poseBuf);

    for (let b = 0; b < boneCount; b++) {
      vp.set(rest.restLocal[b * 3], rest.restLocal[b * 3 + 1], rest.restLocal[b * 3 + 2]);
      if (b === BONE.hips) vp.add(hipsOffset.set(poseBuf.hips[0], poseBuf.hips[1], poseBuf.hips[2]));
      vq.set(poseBuf.rot[b * 4], poseBuf.rot[b * 4 + 1], poseBuf.rot[b * 4 + 2], poseBuf.rot[b * 4 + 3]);
      mA.compose(vp, vq, vs);
      const p = rest.parent[b];
      if (p >= 0) { mB.fromArray(world, p * 16); mA.premultiply(mB); }
      mA.toArray(world, b * 16);
    }
    outWorld.set(world);
    return outWorld;
  }

  // Local TRS for one frame. Used by the glTF exporter so the procedural clips
  // can be written out as real animation tracks.
  function sampleLocalTRS(clipIndex, time, outT, outR, outS) {
    const def = clipDefs[clipIndex];
    for (let i = 0; i < boneCount; i++) {
      poseBuf.rot[i * 4] = 0; poseBuf.rot[i * 4 + 1] = 0; poseBuf.rot[i * 4 + 2] = 0; poseBuf.rot[i * 4 + 3] = 1;
    }
    poseBuf.hips[0] = poseBuf.hips[1] = poseBuf.hips[2] = 0;
    if (def) def.pose(time, poseBuf);
    for (let b = 0; b < boneCount; b++) {
      outT[b * 3] = rest.restLocal[b * 3] + (b === BONE.hips ? poseBuf.hips[0] : 0);
      outT[b * 3 + 1] = rest.restLocal[b * 3 + 1] + (b === BONE.hips ? poseBuf.hips[1] : 0);
      outT[b * 3 + 2] = rest.restLocal[b * 3 + 2] + (b === BONE.hips ? poseBuf.hips[2] : 0);
      for (let c = 0; c < 4; c++) outR[b * 4 + c] = poseBuf.rot[b * 4 + c];
      outS[b * 3] = 1; outS[b * 3 + 1] = 1; outS[b * 3 + 2] = 1;
    }
  }

  const scratch = new Float32Array(boneCount * 16);
  function sampleSkinMatrices(clipIndex, time, outSkin) {
    sampleJointWorld(clipIndex, time, scratch);
    for (let b = 0; b < boneCount; b++) {
      mA.fromArray(scratch, b * 16);
      mB.fromArray(inverseBind, b * 16);
      mA.multiply(mB);
      mA.toArray(outSkin, b * 16);
    }
    return outSkin;
  }

  return {
    name: 'procedural-humanoid',
    source: 'procgen',
    vertexCount: mesh.positions.length / 3,
    boneCount,
    boneNames: SKELETON.map((s) => s[0]),
    positions: mesh.positions,
    normals: mesh.normals,
    uvs: mesh.uvs,
    colors: mesh.colors,
    materialIds: mesh.materialIds,
    skinIndex: mesh.skinIndex,
    skinWeight: mesh.skinWeight,
    indices: mesh.indices,
    inverseBind,
    clips: clipDefs.map((c) => ({ name: c.name, duration: c.duration, loop: c.loop, groundTruthStride: c.stride })),
    sockets: { head: BONE.head, handR: BONE.handR, handL: BONE.handL, chest: BONE.chest },
    restLocal: rest.restLocal,
    restParent: rest.parent,
    sampleJointWorld,
    sampleSkinMatrices,
    sampleLocalTRS,
  };
}

export { BONE, DEG };
