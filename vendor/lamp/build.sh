#!/usr/bin/env bash
# Instantiates and precompiles the Julia environment of Project.toml and the
# committed Manifest.toml, then loads the shim, which checks the fetched upstream
# (PIN stamp, per-file sha256) and applies its patches. Packages and CUDA
# artifacts go to $JULIA_DEPOT_PATH; nothing is built here.
#
# The Manifest is resolved under Julia 1.11 with PDOT's CUDA.jl stack (CUDA.jl
# 5.11.3, CUDA_Runtime_jll 0.21.0+1, GPUCompiler 1.17.1). CUDA.jl 5.11.3 does not
# support Julia 1.12.
set -euo pipefail
cd "$(dirname "$0")"

PIN=$(tr -d '[:space:]' < PIN)
FETCHED=$(tr -d '[:space:]' 2>/dev/null < upstream/.bdrs_pin || true)
if [ "$FETCHED" != "$PIN" ]; then
    echo "FATAL: upstream/ fetched at '${FETCHED:-nothing}' but PIN is $PIN; run vendor/lamp/fetch.sh" >&2
    exit 1
fi
JULIA=${BDRS_LAMP_JULIA:-julia}
VER=$($JULIA --version 2>/dev/null || true)
case "$VER" in
    *" 1.11."*) echo "vendor/lamp: $VER" ;;
    *) echo "FATAL: need Julia 1.11 (got '${VER:-no julia}'); juliaup add 1.11, or set BDRS_LAMP_JULIA" >&2
       exit 1 ;;
esac

sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }
MANIFEST_SHA_BEFORE=$(sha256 Manifest.toml)
$JULIA --project=. --startup-file=no -e 'import Pkg; Pkg.instantiate(); Pkg.precompile()'
MANIFEST_SHA_AFTER=$(sha256 Manifest.toml)
if [ "$MANIFEST_SHA_BEFORE" != "$MANIFEST_SHA_AFTER" ]; then
    echo "FATAL: instantiate rewrote the committed Manifest.toml; restore it (git checkout vendor/lamp/Manifest.toml)" >&2
    echo "       and resolve deliberately if a new dependency set is intended" >&2
    exit 1
fi

$JULIA --project=. --startup-file=no -e '
    include("lamp_shim.jl")
    info = LampShim.load!("upstream")
    using CUDA
    println("vendor/lamp: shim OK (pin=", info["pin"], ", ranges=", info["ranges"],
            ", patches=", info["patches"], "); CUDA.functional() = ", CUDA.functional())'
