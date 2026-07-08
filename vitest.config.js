import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    // Transform the intra-suite deps from source instead of externalising them
    // to the native loader. Their published entry is ESM (dist/index.js); on
    // some CI Node versions vitest fails to externally resolve/load them
    // ("Failed to load url @tangent.to/..."), so inline them to be safe.
    server: { deps: { inline: [/@tangent\.to\//] } },
    watchExclude: ['**/node_modules/**', '**/.venv/**', '**/dist/**'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      include: ['src/**/*.js'],
      exclude: ['tests/**', 'node_modules/**']
    }
  }
});
