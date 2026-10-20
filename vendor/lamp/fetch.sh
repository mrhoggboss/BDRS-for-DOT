#!/usr/bin/env bash
# Fetches LAMP (mxburns2022/CuLAMP.jl) at the pinned SHA into upstream/, which is
# gitignored.
# No upstream file is edited.
#
# Only src/, configurations/, Project.toml and README.md are kept: the full tree
# is about 364 MB, almost all of it figure data (data_archive/, data/). The
# OT-scOmics submodule is not fetched; no transport entry point reads it.
#
# upstream/.bdrs_pin is written last. build.sh, the shim and the runner compare it
# to PIN, so a partial sync is refused.
set -euo pipefail
cd "$(dirname "$0")"

PIN=$(tr -d '[:space:]' < PIN)
URL=https://github.com/mxburns2022/CuLAMP.jl
if ! [[ "$PIN" =~ ^[0-9a-f]{40}$ ]]; then
    echo "FATAL: PIN '$PIN' is not a full 40-hex commit SHA" >&2
    exit 1
fi
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "vendor/lamp: cloning $URL @ $PIN"
git clone --quiet "$URL" "$TMP/CuLAMP.jl"
git -C "$TMP/CuLAMP.jl" checkout --quiet --detach "$PIN"
ACTUAL=$(git -C "$TMP/CuLAMP.jl" rev-parse HEAD)
if [ "$ACTUAL" != "$PIN" ]; then
    echo "FATAL: checked-out SHA $ACTUAL != PIN $PIN" >&2
    exit 1
fi

# The files the shim reads. lamp_shim.jl also checks their sha256, so a moved or
# edited line fails at load instead of giving a wrong answer.
for f in src/Utilities.jl src/solvers/MirrorProx.jl src/solvers/MirrorProxKernel.jl \
         configurations/default_0.json; do
    [ -f "$TMP/CuLAMP.jl/$f" ] \
        || { echo "FATAL: $f missing at $PIN (pin bump moved it?)" >&2; exit 1; }
done

mkdir -p upstream
rm -f upstream/.bdrs_pin
rsync -a --delete \
    --include='/src/***' --include='/configurations/***' \
    --include='/Project.toml' --include='/README.md' --exclude='*' \
    "$TMP/CuLAMP.jl/" upstream/
printf '%s\n' "$ACTUAL" > upstream/.bdrs_pin
echo "vendor/lamp: upstream/ synced to $ACTUAL (subset: src, configurations, Project.toml, README.md; gitignored)"
