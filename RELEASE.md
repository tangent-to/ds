# Release Checklist

Follow these steps in order. Run each command line separately so you can see if anything fails.

## 1. Sync the repository

```bash
git checkout main
git pull origin main
```

Make sure all your feature/docs changes (if any) are committed.

```bash
git status
```

If `git status` shows “nothing to commit, working tree clean”, you are ready for the next step.

## 2. Run tests and build

```bash
cd tangent-ds
npm run test:run
npm run build:browser
npm run verify
cd ..
```

Run `git status` again. It should still be clean. If something modified a file, commit or stash it before continuing.

## 3. Bump the version

Use one of the following inside `tangent-ds/`:

```bash
cd tangent-ds
npm version patch  # or npm version minor / major
cd ..
```

`npm version` updates package files and, because the tree was clean, also creates a commit and a tag for you.

## 4. Push code and tags

```bash
git push origin main
git push origin --tags
```

This push triggers the GitHub Actions release workflow.
