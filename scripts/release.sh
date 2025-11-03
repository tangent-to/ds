#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/release.sh -m "commit message"

Automates the release flow:
  1. Commit current working tree with the provided message.
  2. Bump tangent-ds/package.json patch version.
  3. Commit the release bump and create matching tag.
  4. Push branch and tag to origin.
EOF
}

commit_msg=""

while getopts ":m:h" opt; do
  case "$opt" in
    m) commit_msg="$OPTARG" ;;
    h) usage; exit 0 ;;
    :) echo "Error: -$OPTARG requires an argument." >&2; usage; exit 1 ;;
    \?) echo "Error: invalid option -$OPTARG" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$commit_msg" ]]; then
  echo "Error: commit message is required via -m." >&2
  usage
  exit 1
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

current_branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$current_branch" != "main" ]]; then
  echo "Error: releases must run from main (current: $current_branch)." >&2
  exit 1
fi

if ! git diff --quiet --ignore-submodules HEAD --; then
  echo "Staging current changes..."
else
  echo "Warning: no changes detected before release commit." >&2
fi

git add .
git commit -m "$commit_msg"

pushd tangent-ds >/dev/null
echo "Bumping patch version..."
npm version patch --no-git-tag-version >/dev/null
new_version="$(npm pkg get version | tr -d '"')"
popd >/dev/null

git add tangent-ds/package.json tangent-ds/package-lock.json
git commit -m "chore: release $new_version"

tag="v$new_version"
echo "Creating tag $tag"
git tag "$tag"

echo "Pushing branch and tag..."
git push origin main
git push origin "$tag"

echo "Release $new_version pushed successfully."
