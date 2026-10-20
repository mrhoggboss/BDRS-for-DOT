#!/usr/bin/env bash
# Fetches M3S (ucascnic/CudaOT) at the pinned SHA into upstream/, which is
# gitignored.
# No upstream file is edited; the first-party code lives in shim/.
set -euo pipefail
cd "$(dirname "$0")"

PIN=$(tr -d '[:space:]' < PIN)
URL=https://github.com/ucascnic/CudaOT
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "vendor/m3s: cloning $URL @ $PIN"
git clone --quiet "$URL" "$TMP/CudaOT"
git -C "$TMP/CudaOT" checkout --quiet --detach "$PIN"
ACTUAL=$(git -C "$TMP/CudaOT" rev-parse HEAD)
if [ "$ACTUAL" != "$PIN" ]; then
    echo "FATAL: checked-out SHA $ACTUAL != PIN $PIN" >&2
    exit 1
fi
rm -rf "$TMP/CudaOT/.git"

mkdir -p upstream
rsync -a --delete "$TMP/CudaOT/" upstream/
echo "vendor/m3s: upstream/ synced to $ACTUAL (local-only; gitignored)"
