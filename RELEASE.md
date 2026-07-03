# Release process

Publishing is fully manual: pushing a tag does **not** trigger anything.
A release is (1) a version bump on `main`, then (2) manually running the
three `workflow_dispatch` workflows, then (3) creating the GitHub release.

## 1. Bump the version (from `main`, clean working tree)

```bash
node scripts/bump-version.js patch      # or minor, major, or an explicit "0.7.0"
```

This keeps `package.json`, `jsr.json`, `deno.json`, and `CITATION.cff` in
sync, commits the bump, and creates the `v<version>` tag (use `--no-git` to
skip the commit/tag, `--dry-run` to preview). Do **not** use `npm version`:
it only updates `package.json` and desyncs the JSR publish.

Then sync the lockfile and push:

```bash
npm install --package-lock-only --ignore-scripts
git add package-lock.json && git commit --amend --no-edit
git push origin main && git push origin "v$(npm pkg get version | tr -d '"')"
```

Shortcut for patch releases: `scripts/release.sh -m "commit message"` commits
the working tree, bumps the patch version, tags, and pushes — but it does not
publish anything; steps 2–3 below are still required.

## 2. Run the publish workflows

From the Actions tab ("Run workflow") or with the `gh` CLI:

```bash
gh workflow run publish-npm.yml     # tests + build + verify + npm publish
gh workflow run publish-jsr.yml     # deno publish (version from jsr.json/deno.json)
gh workflow run deploy-site.yml     # notebooks -> tutorials + TypeDoc API ref -> gh-pages
gh run watch
```

Notes:

- `publish-npm.yml` publishes the version currently in `package.json` on
  `main` and skips silently if that version is already on the registry, so
  re-running is safe. It needs the `NPM_TOKEN` repository secret. A
  `skip_tests` input exists for re-publishing only.
- `publish-jsr.yml` authenticates via OIDC (`id-token`); no secret needed.
- `deploy-site.yml` can also be run on its own whenever docs or the
  user-guide notebooks change.

## 3. Create the GitHub release

```bash
gh release create "v$(npm pkg get version | tr -d '"')" --generate-notes
```

(Edit the title/notes as needed; `--notes "..."` for hand-written notes.)

To inspect the current version, run `npm pkg get version`.
