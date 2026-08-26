// Count-ramp benchmark. Holds each step for a warm-up window, then averages the
// frame time over a measurement window, so the numbers are not polluted by the
// shader compiles and buffer growth that happen right after a count change.

export class Benchmark {
  constructor({ counts = [1000, 5000, 10000, 20000, 50000, 100000], warmup = 1.0, measure = 2.5 } = {}) {
    this.counts = counts;
    this.warmup = warmup;
    this.measure = measure;
    this.reset();
  }

  reset() {
    this.active = false;
    this.step = -1;
    this.phase = 'idle';
    this.timer = 0;
    this.frames = 0;
    this.msAccum = 0;
    this.results = [];
    this.onCount = null;
    this.onDone = null;
  }

  start(maxCount, onCount, onDone) {
    this.reset();
    this.counts = this.counts.filter((c) => c <= maxCount);
    this.active = true;
    this.step = -1;
    this.onCount = onCount;
    this.onDone = onDone;
    this.next();
  }

  next() {
    this.step++;
    if (this.step >= this.counts.length) {
      this.active = false;
      this.phase = 'done';
      if (this.onDone) this.onDone(this.results);
      return;
    }
    this.phase = 'warmup';
    this.timer = 0;
    this.frames = 0;
    this.msAccum = 0;
    if (this.onCount) this.onCount(this.counts[this.step]);
  }

  update(dt, frameMs, extra = {}) {
    if (!this.active) return;
    this.timer += dt;
    if (this.phase === 'warmup') {
      if (this.timer >= this.warmup) { this.phase = 'measure'; this.timer = 0; }
      return;
    }
    this.frames++;
    this.msAccum += frameMs;
    if (this.timer >= this.measure) {
      const ms = this.msAccum / Math.max(1, this.frames);
      this.results.push({
        count: this.counts[this.step],
        ms: +ms.toFixed(2),
        fps: +(1000 / ms).toFixed(1),
        drawCalls: extra.drawCalls || 0,
        triangles: extra.triangles || 0,
      });
      this.next();
    }
  }

  get progressText() {
    if (!this.active) return '';
    const c = this.counts[this.step];
    return `benchmark ${this.step + 1}/${this.counts.length} - ${c.toLocaleString()} instances (${this.phase})`;
  }

  toTable() {
    if (!this.results.length) return '';
    const rows = this.results.map((r) =>
      `<tr><td>${r.count.toLocaleString()}</td><td>${r.fps.toFixed(1)}</td><td>${r.ms.toFixed(2)}</td><td>${r.drawCalls}</td></tr>`);
    return `<table class="bench"><thead><tr><th>instances</th><th>fps</th><th>ms</th><th>draws</th></tr></thead><tbody>${rows.join('')}</tbody></table>`;
  }
}
