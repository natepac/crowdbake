// Scene furniture: sky, ground, sun with a camera-following shadow map, and the
// pillars the crowd steers around. Nothing here is VAT-specific.

import {
  BackSide,
  BoxGeometry,
  CanvasTexture,
  Color,
  CylinderGeometry,
  DirectionalLight,
  FogExp2,
  HemisphereLight,
  InstancedMesh,
  Matrix4,
  Mesh,
  MeshStandardMaterial,
  PlaneGeometry,
  Quaternion,
  RepeatWrapping,
  ShaderMaterial,
  SphereGeometry,
  SRGBColorSpace,
  Vector3,
} from 'three';

const SKY_TOP = new Color(0x2b4a7a);
const SKY_BOTTOM = new Color(0xbcd0e4);

function groundTexture(size = 512) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d');
  g.fillStyle = '#8f9384';
  g.fillRect(0, 0, size, size);
  // speckle
  const img = g.getImageData(0, 0, size, size);
  for (let i = 0; i < img.data.length; i += 4) {
    const n = (Math.random() - 0.5) * 26;
    img.data[i] += n; img.data[i + 1] += n; img.data[i + 2] += n * 0.7;
  }
  g.putImageData(img, 0, 0);
  // faint grid
  g.strokeStyle = 'rgba(0,0,0,0.10)';
  g.lineWidth = 2;
  g.beginPath();
  g.moveTo(0, 0); g.lineTo(size, 0); g.moveTo(0, 0); g.lineTo(0, size);
  g.stroke();
  const tex = new CanvasTexture(c);
  tex.wrapS = tex.wrapT = RepeatWrapping;
  tex.colorSpace = SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

export function buildEnvironment(scene, { worldSize = 512, shadowMapSize = 2048, pillarCount = 26 } = {}) {
  // Fog density is the thing that makes a zoomed-out crowd "disappear": at
  // 0.0042, a character 250 m away is 67% fog and 91% at 370 m. Tune so the far
  // corner of the world (~600 m) is still only ~1/3 fogged.
  const fog = new FogExp2(SKY_BOTTOM.getHex(), 0.0011);
  scene.fog = fog;
  scene.background = SKY_BOTTOM.clone();

  // --- sky dome ---
  const sky = new Mesh(
    new SphereGeometry(worldSize * 1.4, 24, 16),
    new ShaderMaterial({
      side: BackSide,
      depthWrite: false,
      fog: false,
      uniforms: {
        topColor: { value: SKY_TOP.clone() },
        bottomColor: { value: SKY_BOTTOM.clone() },
        offset: { value: 12 },
        exponent: { value: 0.65 },
      },
      vertexShader: `
        varying vec3 vWorld;
        void main() {
          vWorld = (modelMatrix * vec4(position, 1.0)).xyz;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
      fragmentShader: `
        uniform vec3 topColor; uniform vec3 bottomColor;
        uniform float offset; uniform float exponent;
        varying vec3 vWorld;
        void main() {
          float h = normalize(vWorld + vec3(0.0, offset, 0.0)).y;
          gl_FragColor = vec4(mix(bottomColor, topColor, pow(max(h, 0.0), exponent)), 1.0);
        }`,
    }),
  );
  sky.frustumCulled = false;
  scene.add(sky);

  // --- lights ---
  const hemi = new HemisphereLight(0xbcd4f0, 0x6b6a5c, 1.05);
  scene.add(hemi);

  const sun = new DirectionalLight(0xfff2dd, 2.6);
  sun.position.set(60, 90, 40);
  sun.castShadow = true;
  sun.shadow.mapSize.set(shadowMapSize, shadowMapSize);
  sun.shadow.camera.near = 1;
  sun.shadow.camera.far = 400;
  sun.shadow.bias = -0.0008;
  sun.shadow.normalBias = 0.035;
  scene.add(sun);
  scene.add(sun.target);

  // --- ground ---
  const groundMat = new MeshStandardMaterial({ map: groundTexture(), roughness: 0.97, metalness: 0 });
  groundMat.map.repeat.set(worldSize / 4, worldSize / 4);
  const ground = new Mesh(new PlaneGeometry(worldSize * 2, worldSize * 2), groundMat);
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);

  // --- pillars (also the crowd's obstacles) ---
  const obstacles = [];
  const pillarGeo = new CylinderGeometry(1, 1.25, 9, 12);
  const pillarMat = new MeshStandardMaterial({ color: 0xb9b3a5, roughness: 0.9 });
  const pillars = new InstancedMesh(pillarGeo, pillarMat, pillarCount);
  pillars.castShadow = true;
  pillars.receiveShadow = true;
  const m = new Matrix4();
  const q = new Quaternion();
  const s = new Vector3();
  const p = new Vector3();
  const half = worldSize * 0.42;
  for (let i = 0; i < pillarCount; i++) {
    const a = (i / pillarCount) * Math.PI * 2 * 3.3;
    const r = half * Math.sqrt((i + 0.5) / pillarCount);
    const x = Math.cos(a) * r;
    const z = Math.sin(a) * r;
    const scale = 0.7 + (i % 5) * 0.22;
    p.set(x, 4.5 * scale, z);
    s.set(scale, scale, scale);
    m.compose(p, q, s);
    pillars.setMatrixAt(i, m);
    obstacles.push({ x, z, radius: 1.35 * scale + 0.6 });
  }
  pillars.instanceMatrix.needsUpdate = true;
  scene.add(pillars);

  // plinth under each pillar so they read as placed, not floating
  const plinth = new InstancedMesh(new BoxGeometry(3.4, 0.35, 3.4), pillarMat, pillarCount);
  plinth.receiveShadow = true;
  plinth.castShadow = true;
  for (let i = 0; i < pillarCount; i++) {
    const o = obstacles[i];
    p.set(o.x, 0.17, o.z);
    s.set(1, 1, 1);
    m.compose(p, q, s);
    plinth.setMatrixAt(i, m);
  }
  plinth.instanceMatrix.needsUpdate = true;
  scene.add(plinth);

  const shadowTarget = new Vector3();
  const lightDir = new Vector3();
  const lightRight = new Vector3();
  const lightUp = new Vector3();
  const camFwd = new Vector3();
  const UP = new Vector3(0, 1, 0);
  // authoritative sun direction (target -> light); sun.position is derived from
  // it every frame, so it must not be read back as the source of truth
  const sunDir = new Vector3(60, 90, 40).normalize();

  /**
   * Keep the shadow frustum around the viewer and snap it to whole shadow-map
   * texels, otherwise the crowd's shadows crawl as the camera moves.
   */
  function updateShadow(camera, radius) {
    const cam = camera.position;
    camera.getWorldDirection(camFwd);
    camFwd.y = 0;
    if (camFwd.lengthSq() < 1e-6) camFwd.set(0, 0, 1);
    camFwd.normalize();
    shadowTarget.set(cam.x + camFwd.x * radius * 0.45, 0, cam.z + camFwd.z * radius * 0.45);

    lightDir.copy(sunDir);
    lightRight.crossVectors(UP, lightDir).normalize();
    lightUp.crossVectors(lightDir, lightRight).normalize();

    const texel = (radius * 2) / sun.shadow.mapSize.x;
    const rx = Math.round(shadowTarget.dot(lightRight) / texel) * texel;
    const ry = Math.round(shadowTarget.dot(lightUp) / texel) * texel;
    const rz = shadowTarget.dot(lightDir);
    shadowTarget.set(0, 0, 0)
      .addScaledVector(lightRight, rx)
      .addScaledVector(lightUp, ry)
      .addScaledVector(lightDir, rz);

    sun.target.position.copy(shadowTarget);
    sun.position.copy(shadowTarget).addScaledVector(lightDir, radius * 2.2);
    sun.target.updateMatrixWorld();

    const c = sun.shadow.camera;
    if (c.left !== -radius) {
      c.left = -radius; c.right = radius; c.top = radius; c.bottom = -radius;
      c.near = 1; c.far = radius * 4.5;
      c.updateProjectionMatrix();
    }
  }

  function setSunAngle(azimuthDeg, elevationDeg) {
    const az = azimuthDeg * Math.PI / 180;
    const el = Math.max(4, elevationDeg) * Math.PI / 180;
    sunDir.set(Math.cos(az) * Math.cos(el), Math.sin(el), Math.sin(az) * Math.cos(el)).normalize();
  }

  return { sun, hemi, ground, sky, fog, pillars, plinth, obstacles, updateShadow, setSunAngle };
}
