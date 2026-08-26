import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  server: { open: true },
  build: { target: 'es2022', outDir: 'dist', assetsInlineLimit: 0 },
});
