// Stride extraction.
//
// This is the bit that kills foot-slide. For a locomotion clip we need to know
// how far the character travels per animation cycle; at runtime the playback
// rate is then simply  desiredSpeed / (strideLength / clipDuration).
//
// Method: the planted foot is fixed to the ground, so in root space it slides
// backwards at exactly the character's ground speed. Sample every frame, decide
// which feet are planted (low + not rising), take the MEDIAN backwards velocity
// across all planted samples (robust to heel roll, toe pivot and pop), and
// multiply by the clip duration. Using the median rather than integrating over
// the whole cycle means running clips with an airborne phase come out right too.

const RE_FOOT = /foot|ankle/i;
const RE_TOE = /toe|ball/i;
const RE_ROOT = /^(root|hips|pelvis|armature)$/i;

function median(arr) {
  if (!arr.length) return 0;
  const a = Float64Array.from(arr).sort();
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) * 0.5;
}

function pickFootBones(model, samples, frames) {
  const names = model.boneNames || [];
  let cands = names.map((n, i) => i).filter((i) => RE_FOOT.test(names[i]));
  if (cands.length < 2) cands = names.map((n, i) => i).filter((i) => RE_TOE.test(names[i]));

  if (cands.length < 2) {
    // no naming convention to lean on: use the two lowest bones that are
    // horizontally separated (i.e. a left/right pair rather than a chain)
    const meanY = [], meanX = [], meanZ = [];
    for (let b = 0; b < model.boneCount; b++) {
      let sy = 0, sx = 0, sz = 0;
      for (let f = 0; f < frames; f++) {
        sx += samples[f][b * 16 + 12]; sy += samples[f][b * 16 + 13]; sz += samples[f][b * 16 + 14];
      }
      meanX[b] = sx / frames; meanY[b] = sy / frames; meanZ[b] = sz / frames;
    }
    const byY = meanY.map((y, b) => [y, b]).sort((a, c) => a[0] - c[0]).map((p) => p[1]);
    const first = byY[0];
    let second = -1;
    for (const b of byY) {
      if (b === first) continue;
      const d = Math.hypot(meanX[b] - meanX[first], meanZ[b] - meanZ[first]);
      if (d > 0.04) { second = b; break; }
    }
    cands = second >= 0 ? [first, second] : [first];
  }
  if (cands.length > 2) {
    // keep the two with the lowest average height (ankles, not knees)
    cands = cands
      .map((b) => {
        let s = 0;
        for (let f = 0; f < frames; f++) s += samples[f][b * 16 + 13];
        return [s / frames, b];
      })
      .sort((a, c) => a[0] - c[0])
      .slice(0, 2)
      .map((p) => p[1]);
  }
  return cands;
}

/**
 * @returns {{strideLength:number, groundSpeed:number, method:string, facing:number,
 *            forwardAxis:string, plantedSamples:number, feet:string[], rootTravel:number,
 *            slideRms:number, confidence:number}}
 */
export function measureStride(model, clipIndex, frames) {
  const clip = model.clips[clipIndex];
  const duration = clip.duration || 1;
  const boneCount = model.boneCount;

  const samples = [];
  for (let f = 0; f < frames; f++) {
    const t = (f / frames) * duration;
    samples.push(model.sampleJointWorld(clipIndex, t, new Float32Array(boneCount * 16)).slice());
  }

  // root motion across a full cycle
  const rootBone = (model.boneNames || []).findIndex((n) => RE_ROOT.test(n));
  const rb = rootBone >= 0 ? rootBone : 0;
  const at0 = model.sampleJointWorld(clipIndex, 0, new Float32Array(boneCount * 16)).slice();
  const atEnd = model.sampleJointWorld(clipIndex, duration, new Float32Array(boneCount * 16)).slice();
  const rootTravel = Math.hypot(
    atEnd[rb * 16 + 12] - at0[rb * 16 + 12],
    atEnd[rb * 16 + 14] - at0[rb * 16 + 14],
  );

  const feet = pickFootBones(model, samples, frames);
  const dt = duration / frames;

  const dxs = [], dzs = [];
  let lowSamples = 0;
  const slide = [];

  for (const b of feet) {
    let minY = Infinity, maxY = -Infinity;
    for (let f = 0; f < frames; f++) {
      const y = samples[f][b * 16 + 13];
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const thresh = minY + Math.max(0.2 * (maxY - minY), 0.005);
    for (let f = 0; f < frames; f++) {
      const g = (f + 1) % frames;
      const y0 = samples[f][b * 16 + 13], y1 = samples[g][b * 16 + 13];
      if (y0 > thresh || y1 > thresh) continue;
      lowSamples++;
      dxs.push(samples[g][b * 16 + 12] - samples[f][b * 16 + 12]);
      dzs.push(samples[g][b * 16 + 14] - samples[f][b * 16 + 14]);
    }
  }

  const mdx = median(dxs), mdz = median(dzs);
  const forwardAxis = Math.abs(mdz) >= Math.abs(mdx) ? 'z' : 'x';
  const mForward = forwardAxis === 'z' ? mdz : mdx;
  const perFrame = -mForward;                 // foot slides back -> body moves forward
  let groundSpeed = perFrame / dt;
  let strideLength = groundSpeed * duration;
  let method = 'contact-foot';

  if (rootTravel > 0.05) {
    strideLength = rootTravel;
    groundSpeed = rootTravel / duration;
    method = 'root-motion';
  }

  const facing = strideLength >= 0 ? 1 : -1;
  strideLength = Math.abs(strideLength);
  groundSpeed = Math.abs(groundSpeed);

  // how consistent were the planted samples? tight spread == trustworthy stride
  const src = forwardAxis === 'z' ? dzs : dxs;
  for (const d of src) slide.push(d - mForward);
  const slideRms = slide.length ? Math.sqrt(slide.reduce((s, v) => s + v * v, 0) / slide.length) : 0;
  const confidence = strideLength < 1e-4
    ? 0
    : Math.max(0, Math.min(1, 1 - (slideRms / Math.max(Math.abs(mForward), 1e-6)) * 0.5));

  return {
    strideLength,
    groundSpeed,
    method,
    facing,
    forwardAxis,
    plantedSamples: lowSamples,
    feet: feet.map((b) => (model.boneNames || [])[b] || ('bone' + b)),
    rootTravel,
    slideRms,
    confidence,
  };
}
