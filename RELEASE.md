# One-command release

```bash
scripts/release.sh -m "feat: add new clustering plot"
```

The script will:

1. Stage and commit your current changes with the provided message.
2. Bump the patch version inside `tangent-ds/package.json`.
3. Commit the version bump, create the matching tag, and push both branch and tag.

Make sure your feature work is ready before running the script. To inspect the version it set, run `npm pkg get version` inside `tangent-ds/`.
