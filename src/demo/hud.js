// Minimal stats overlay. Deliberately not stats.js: the interesting numbers
// here are draw calls, drawn instances and the LOD split, not just fps.

const fmt = (n) => n.toLocaleString('en-US');

function short(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'k';
  return String(Math.round(n));
}

export class HUD {
  constructor(parent = document.body) {
    this.el = document.createElement('div');
    this.el.className = 'hud';
    parent.appendChild(this.el);

    this.frames = 0;
    this.accum = 0;
    this.fps = 0;
    this.msAccum = 0;
    this.ms = 0;
    this.cpuMs = 0;
    this.history = new Float32Array(120);
    this.hIndex = 0;

    this.canvas = document.createElement('canvas');
    this.canvas.width = 240;
    this.canvas.height = 38;
    this.canvas.className = 'hud-graph';
    this.el.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');

    this.text = document.createElement('div');
    this.text.className = 'hud-text';
    this.el.appendChild(this.text);

    this.note = document.createElement('div');
    this.note.className = 'hud-note';
    this.el.appendChild(this.note);
  }

  setNote(html) { this.note.innerHTML = html || ''; }

  /**
   * @param realDt seconds of WALL CLOCK time since the last frame. This must not
   *   be the simulation's dt: that is clamped for stability, so feeding it here
   *   silently caps the reported frame rate at 1/clamp (the reason this used to
   *   insist it was running at exactly 20 fps while the browser crawled).
   */
  update(realDt, frameMs, renderer, crowd, sim, asset) {
    this.frames++;
    this.accum += realDt;
    this.msAccum += frameMs;
    this.history[this.hIndex] = Math.min(realDt * 1000, 200);
    this.hIndex = (this.hIndex + 1) % this.history.length;

    if (this.accum < 0.25) return;
    this.fps = this.frames / this.accum;
    this.ms = (this.accum / this.frames) * 1000;      // true frame time
    this.cpuMs = this.msAccum / this.frames;          // what our JS accounts for
    this.frames = 0; this.accum = 0; this.msAccum = 0;

    const info = renderer.info;
    const s = crowd.stats;
    const lodBits = s.lodCounts.map((c, i) => `<b>L${i}</b> ${short(c)}`).join('<span class="sep">/</span>');
    const anim = asset.memory;

    this.text.innerHTML = `
<div class="hud-row hud-big"><span>${this.fps.toFixed(1)} fps</span><span>${this.ms.toFixed(1)} ms</span></div>
<div class="hud-row"><span>cpu (js) / frame</span><span>${this.cpuMs.toFixed(1)} / ${this.ms.toFixed(1)} ms${this.cpuMs < this.ms * 0.6 ? '  <b>gpu-bound</b>' : ''}</span></div>
<div class="hud-row"><span>draw calls</span><span>${fmt(info.render.calls)}</span></div>
<div class="hud-row hud-dim"><span>resolution</span><span>${renderer.domElement.width}x${renderer.domElement.height}</span></div>
<div class="hud-row"><span>triangles</span><span>${short(info.render.triangles)}</span></div>
<div class="hud-row"><span>instances</span><span>${fmt(s.instances)}</span></div>
<div class="hud-row"><span>submitted</span><span>${fmt(s.drawnInstances)}</span></div>
<div class="hud-row"><span>rasterised (post-cull)</span><span>${fmt(s.renderedInstances)}</span></div>
<div class="hud-row"><span>chunks live / drawn</span><span>${s.chunks} / ${s.renderedChunks}</span></div>
<div class="hud-row"><span>far pool (1 draw)</span><span>${fmt(s.farPooled)}</span></div>
<div class="hud-lods hud-dim">${lodBits}</div>
<div class="hud-row"><span>sim / separation</span><span>${sim.stats.simMs.toFixed(1)} / ${sim.stats.sepMs.toFixed(1)} ms</span></div>
<div class="hud-row"><span>instance upload</span><span>${(s.uploadBytes / 1048576).toFixed(2)} MB/f</span></div>
<div class="hud-row hud-dim"><span>anim texture</span><span>${((anim.positions + anim.normals) / 1048576).toFixed(2)} MB</span></div>
<div class="hud-row hud-dim"><span>bone texture</span><span>${anim.bones ? (anim.bones / 1024).toFixed(0) + ' KB' : '-'}</span></div>
`;

    const ctx = this.ctx;
    const { width, height } = this.canvas;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = 'rgba(255,255,255,0.06)';
    ctx.fillRect(0, 0, width, height);
    // 16.7ms line
    ctx.strokeStyle = 'rgba(255,255,255,0.22)';
    ctx.beginPath();
    const y167 = height - (16.7 / 200) * height;
    ctx.moveTo(0, y167); ctx.lineTo(width, y167); ctx.stroke();
    ctx.beginPath();
    for (let i = 0; i < this.history.length; i++) {
      const idx = (this.hIndex + i) % this.history.length;
      const v = this.history[idx];
      const x = (i / (this.history.length - 1)) * width;
      const y = height - (v / 200) * height;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = this.ms < 17 ? '#7ee787' : this.ms < 34 ? '#e3b341' : '#f85149';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}
