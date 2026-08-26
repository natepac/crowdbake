// Three camera behaviours: orbit the crowd, walk inside it, or run a slow
// automatic tour. Walking inside is the honest test -- that is where a single
// bounding sphere stops saving you and chunked culling starts earning its keep.

import { Vector3 } from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const EYE_HEIGHT = 1.72;

export class CameraRig {
  constructor(camera, domElement, { worldSize = 512 } = {}) {
    this.camera = camera;
    this.dom = domElement;
    this.worldSize = worldSize;
    this.mode = 'orbit';

    this.orbit = new OrbitControls(camera, domElement);
    this.orbit.enableDamping = true;
    this.orbit.dampingFactor = 0.08;
    this.orbit.maxPolarAngle = Math.PI * 0.495;
    this.orbit.minDistance = 3;
    this.orbit.maxDistance = worldSize * 0.9;
    this.orbit.target.set(0, 1.2, 0);

    this.keys = new Set();
    this.walkYaw = 0;
    this.walkPitch = -0.05;
    this.walkPos = new Vector3(0, EYE_HEIGHT, worldSize * 0.2);
    this.walkSpeed = 6;
    this.tourTime = 0;
    this._tmp = new Vector3();

    this._onKeyDown = (e) => {
      this.keys.add(e.code);
      if (e.code === 'ShiftLeft' || e.code === 'ShiftRight') this.walkSpeed = 16;
    };
    this._onKeyUp = (e) => {
      this.keys.delete(e.code);
      if (e.code === 'ShiftLeft' || e.code === 'ShiftRight') this.walkSpeed = 6;
    };
    this._onMouseMove = (e) => {
      if (this.mode !== 'walk' || document.pointerLockElement !== this.dom) return;
      this.walkYaw -= e.movementX * 0.0022;
      this.walkPitch = Math.max(-1.4, Math.min(1.4, this.walkPitch - e.movementY * 0.0022));
    };
    this._onClick = () => {
      if (this.mode === 'walk' && document.pointerLockElement !== this.dom) this.dom.requestPointerLock();
    };

    window.addEventListener('keydown', this._onKeyDown);
    window.addEventListener('keyup', this._onKeyUp);
    window.addEventListener('mousemove', this._onMouseMove);
    this.dom.addEventListener('click', this._onClick);
  }

  setMode(mode) {
    if (mode === this.mode) return;
    if (this.mode === 'walk' && document.pointerLockElement === this.dom) document.exitPointerLock();
    this.mode = mode;
    this.orbit.enabled = mode === 'orbit';
    if (mode === 'walk') {
      this.walkPos.copy(this.camera.position);
      this.walkPos.y = EYE_HEIGHT;
      const dir = this.camera.getWorldDirection(this._tmp);
      this.walkYaw = Math.atan2(-dir.x, -dir.z);
      this.walkPitch = Math.asin(Math.max(-1, Math.min(1, dir.y)));
    }
    if (mode === 'orbit') {
      this.orbit.target.copy(this.camera.position).add(this.camera.getWorldDirection(this._tmp).multiplyScalar(15));
      this.orbit.target.y = 1.2;
      this.orbit.update();
    }
  }

  update(dt) {
    if (this.mode === 'orbit') {
      this.orbit.update();
      return;
    }
    if (this.mode === 'tour') {
      this.tourTime += dt;
      const t = this.tourTime * 0.045;
      const r = this.worldSize * (0.16 + 0.13 * (0.5 + 0.5 * Math.sin(t * 0.53)));
      const h = 3 + 26 * (0.5 + 0.5 * Math.sin(t * 0.31));
      this.camera.position.set(Math.cos(t) * r, h, Math.sin(t) * r);
      this.camera.lookAt(Math.cos(t + 1.3) * r * 0.2, 1.4, Math.sin(t + 1.3) * r * 0.2);
      return;
    }

    // walk
    const fwdX = -Math.sin(this.walkYaw), fwdZ = -Math.cos(this.walkYaw);
    const rightX = -fwdZ, rightZ = fwdX;
    let mx = 0, mz = 0;
    if (this.keys.has('KeyW') || this.keys.has('ArrowUp')) { mx += fwdX; mz += fwdZ; }
    if (this.keys.has('KeyS') || this.keys.has('ArrowDown')) { mx -= fwdX; mz -= fwdZ; }
    if (this.keys.has('KeyD') || this.keys.has('ArrowRight')) { mx += rightX; mz += rightZ; }
    if (this.keys.has('KeyA') || this.keys.has('ArrowLeft')) { mx -= rightX; mz -= rightZ; }
    const len = Math.hypot(mx, mz);
    if (len > 0) {
      const k = (this.walkSpeed * dt) / len;
      this.walkPos.x += mx * k;
      this.walkPos.z += mz * k;
    }
    if (this.keys.has('KeyQ')) this.walkPos.y = Math.max(0.4, this.walkPos.y - this.walkSpeed * dt);
    if (this.keys.has('KeyE')) this.walkPos.y += this.walkSpeed * dt;

    const half = this.worldSize * 0.55;
    this.walkPos.x = Math.max(-half, Math.min(half, this.walkPos.x));
    this.walkPos.z = Math.max(-half, Math.min(half, this.walkPos.z));

    this.camera.position.copy(this.walkPos);
    const cp = Math.cos(this.walkPitch);
    this.camera.lookAt(
      this.walkPos.x - Math.sin(this.walkYaw) * cp,
      this.walkPos.y + Math.sin(this.walkPitch),
      this.walkPos.z - Math.cos(this.walkYaw) * cp,
    );
  }

  dispose() {
    window.removeEventListener('keydown', this._onKeyDown);
    window.removeEventListener('keyup', this._onKeyUp);
    window.removeEventListener('mousemove', this._onMouseMove);
    this.dom.removeEventListener('click', this._onClick);
    this.orbit.dispose();
  }
}
