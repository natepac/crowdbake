import { defineConfig } from 'vite';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

/**
 * Dev-only goober API. Lets the demo list every bakeable critter kind, bake or
 * regenerate one on demand (capture in headless Chrome -> VAT bake), and list
 * which baked assets exist. Bakes land in public/baked/, which Vite serves
 * directly, so a reload after baking picks the new asset up. None of this
 * exists in a production build -- the demo probes for it and hides the panel.
 */
function goobersDevApi() {
  let kindsCache = null;
  let busy = null;

  const run = (args) => new Promise((resolve) => {
    const p = spawn(process.execPath, args, { cwd: process.cwd() });
    let out = '', err = '';
    p.stdout.on('data', (d) => { out += d; });
    p.stderr.on('data', (d) => { err += d; });
    p.on('close', (code) => resolve({ code, out, err }));
  });
  const json = (res, status, body) => {
    res.statusCode = status;
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify(body));
  };

  return {
    name: 'goobers-dev-api',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const url = new URL(req.url, 'http://localhost');
        if (!url.pathname.startsWith('/__goobers/')) return next();

        if (url.pathname === '/__goobers/kinds') {
          if (!kindsCache) {
            const r = await run(['tools/bake-goober.mjs', '--list', '--quiet']);
            if (r.code !== 0) return json(res, 500, { error: (r.err || r.out).slice(-400) });
            try { kindsCache = JSON.parse(r.out.trim().split('\n').pop()); }
            catch { return json(res, 500, { error: 'could not parse kind list' }); }
          }
          return json(res, 200, kindsCache);
        }

        if (url.pathname === '/__goobers/assets') {
          const dir = path.resolve('public/baked');
          const assets = fs.existsSync(dir)
            ? fs.readdirSync(dir).filter((f) => f.endsWith('.json')).map((f) => f.replace(/\.json$/, ''))
            : [];
          return json(res, 200, assets);
        }

        if (url.pathname === '/__goobers/bake') {
          const kind = url.searchParams.get('kind') || '';
          const seed = (parseInt(url.searchParams.get('seed') || '1', 10) >>> 0) || 1;
          if (!/^[a-z0-9-]{1,40}$/.test(kind)) return json(res, 400, { error: 'bad kind' });
          if (busy) return json(res, 409, { error: 'a bake is already running (' + busy + ')' });
          busy = kind;
          try {
            const dump = 'tools/_out/goober-' + kind;
            const asset = 'goober-' + kind.replace(/^goober-/, '');
            const r1 = await run(['tools/bake-goober.mjs', '--kind', kind, '--seed', String(seed), '--out', dump, '--quiet']);
            if (r1.code !== 0) return json(res, 500, { error: 'capture failed: ' + (r1.err || r1.out).slice(-400) });
            const r2 = await run(['tools/bake.mjs', '--frames-dump', dump + '.frames.json',
              '--fps', '40', '--lods', '5', '--out', 'public/baked/' + asset, '--quiet']);
            if (r2.code !== 0) return json(res, 500, { error: 'bake failed: ' + (r2.err || r2.out).slice(-400) });
            return json(res, 200, { ok: true, asset, seed });
          } finally {
            busy = null;
          }
        }

        return next();
      });
    },
  };
}

export default defineConfig({
  base: './',
  plugins: [goobersDevApi()],
  server: { open: true },
  build: { target: 'es2022', outDir: 'dist', assetsInlineLimit: 0 },
});
