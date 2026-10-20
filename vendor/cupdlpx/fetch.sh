#!/usr/bin/env bash
# Re-fetches the cuPDLPx upstream tree at the pinned SHA into upstream/, which
# is committed (Apache-2.0). Running it stages upstream/ in the git index.
#
# Upstream's nested upstream/.gitignore ignores *.txt, *.sh and test/*, which
# hides its own CMakeLists.txt files (root and python_bindings/, both needed
# by scikit-build-core) and its test suite from a plain `git add`. The tree is
# kept verbatim, nested .gitignore included, so this script force-adds
# upstream/ and fails if any upstream file is still hidden.
set -euo pipefail
cd "$(dirname "$0")"

PIN=$(tr -d '[:space:]' < PIN)
URL=https://github.com/MIT-Lu-Lab/cuPDLPx
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "vendor/cupdlpx: cloning $URL @ $PIN"
git clone --quiet "$URL" "$TMP/cuPDLPx"
git -C "$TMP/cuPDLPx" checkout --quiet --detach "$PIN"
ACTUAL=$(git -C "$TMP/cuPDLPx" rev-parse HEAD)
if [ "$ACTUAL" != "$PIN" ]; then
    echo "FATAL: checked-out SHA $ACTUAL != PIN $PIN" >&2
    exit 1
fi
rm -rf "$TMP/cuPDLPx/.git"

mkdir -p upstream
rsync -a --delete --exclude='/build/' "$TMP/cuPDLPx/" upstream/
echo "vendor/cupdlpx: upstream/ synced to $ACTUAL"

# build.sh leaves its build directory upstream/build/ (kept by the rsync
# exclude); git add -f would stage it, FetchContent clones included, so it is
# unstaged and left out of the hidden-file check.
git add -f upstream/
git rm -r -q -f --cached --ignore-unmatch upstream/build/
MISSED=$(git status --porcelain --ignored=matching upstream/ \
         | grep '^!!' | grep -v 'upstream/build/' || true)
if [ -n "$MISSED" ]; then
    echo "FATAL: files still hidden from git after force-add:" >&2
    echo "$MISSED" >&2
    exit 1
fi
echo "vendor/cupdlpx: upstream staged in full ($(git diff --cached --name-only upstream/ | wc -l) files incl. the gitignore-hidden CMakeLists/test suite)"
