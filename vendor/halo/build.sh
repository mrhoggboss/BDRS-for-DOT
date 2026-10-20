#!/usr/bin/env bash
# Builds HALO's pycupdlpx CUDA extension out of tree, in place of upstream's
# micromamba-bound lp_solvers/build_solvers/build_cupdlpx_micromamba.sh.
# The libraries land flat in _build/ (gitignored) with an $ORIGIN RPATH, so no
# LD_LIBRARY_PATH is needed; src/runners/run_halo.py puts _build/ on sys.path.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

if [ ! -d upstream/src/hierarchical_ot ]; then
    echo "FATAL: upstream/ missing or incomplete — run vendor/halo/fetch.sh first" >&2
    exit 1
fi

# nvidia-smi prints query errors to stdout and may report "[N/A]" (MIG,
# virtualized GPUs), so only digits are accepted.
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
       | head -1 | tr -d '. \r' || true)
case "$ARCH" in
    ''|*[!0-9]*) ARCH=80 ;;
esac
ARCH_LIST="${ARCH}-real"
echo "vendor/halo: building pycupdlpx for CUDA arch ${ARCH_LIST} with $($PYTHON --version 2>&1)"

SRC=upstream/src/hierarchical_ot/lp_solvers/cuPDLPx/pycupdlpx
cmake -S "$SRC" -B _cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_CUDA=ON \
    -DBUILD_PYTHON=ON \
    -DPython_EXECUTABLE="$(command -v "$PYTHON")" \
    -DCMAKE_CUDA_ARCHITECTURES="${ARCH_LIST}" \
    -DCMAKE_BUILD_RPATH='$ORIGIN' \
    -DCMAKE_INSTALL_RPATH='$ORIGIN' \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
cmake --build _cmake -j"$(nproc)"

mkdir -p _build
find _cmake -name 'pycupdlpx*.so*' -exec cp -P {} _build/ \;
find _cmake -name 'libcupdlpx*.so*' -exec cp -P {} _build/ \;
ls -la _build/

EXT=$(ls _build/pycupdlpx*.so 2>/dev/null | head -1)
if [ -n "$EXT" ] && ! readelf -d "$EXT" | grep -q 'ORIGIN'; then
    if command -v patchelf >/dev/null; then
        patchelf --set-rpath '$ORIGIN' "$EXT"
        echo "vendor/halo: RPATH patched to \$ORIGIN via patchelf"
    else
        echo "FATAL: $EXT lacks an \$ORIGIN RPATH and patchelf is not installed" >&2
        echo "       apt-get install -y patchelf, then re-run build.sh" >&2
        exit 1
    fi
fi

PYTHONPATH="_build:upstream/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" -c "import pycupdlpx, hierarchical_ot; print('vendor/halo: build OK —', hierarchical_ot.__name__, 'and pycupdlpx import')"
