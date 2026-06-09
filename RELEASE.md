# One-command release

```bash
scripts/release.sh -m "feat: add new clustering plot"
```

The script will:

1. Stage and commit your current changes with the provided message.
2. Bump the patch version in `package.json`.
3. Commit the version bump, create the matching tag, and push both branch and tag.

Pushing a `v*` tag triggers `.github/workflows/publish.yml`, which runs the
tests, builds the browser bundle, verifies the package, publishes to npm and
JSR, and deploys the documentation site.

Requirements:

- Releases run from `main` with a clean, ready working tree.
- The `NPM_TOKEN` secret must be set in the repository settings (JSR and the
  docs deployment use the built-in `GITHUB_TOKEN`).
- For minor/major bumps, use `node scripts/bump-version.js` (which keeps
  `package.json`, `deno.json`, `jsr.json`, and `CITATION.cff` in sync) instead
  of the patch-only release script.

To inspect the current version, run `npm pkg get version`.
