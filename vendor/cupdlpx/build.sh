#!/usr/bin/env bash
# Builds and installs the official cuPDLPx Python binding from the committed
# upstream/ tree into $PYTHON's environment, by upstream's `pip install .`
# route with two deviations:
#   - --no-build-isolation, so the extension builds against the environment's
#     own numpy, scipy, pybind11 and scikit-build-core (install
#     scikit-build-core and "pybind11>=2.11" first if missing);
#   - CMAKE_CUDA_ARCHITECTURES set to the visible device instead of
#     pyproject's "all", which compiles every architecture; override with
#     CUPDLPX_CUDA_ARCH (digits only, e.g. 90).
# Needs network: upstream's CMake FetchContent-clones the PSLP presolver
# (v0.0.8), and zlib v1.3 when no system zlib is found. For an offline build,
# pre-seed FETCHCONTENT_SOURCE_DIR_PSLP.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}

if [ ! -f upstream/pyproject.toml ]; then
    echo "FATAL: upstream/ missing or incomplete (vendor/cupdlpx/upstream is committed —" >&2
    echo "       a fresh checkout should already have it)" >&2
    exit 1
fi

# nvidia-smi prints query errors to stdout and may report "[N/A]" (MIG,
# virtualized GPUs), so only digits are accepted.
ARCH=${CUPDLPX_CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
       | head -1 | tr -d '. \r' || true)}
case "$ARCH" in
    ''|*[!0-9]*) ARCH=80 ;;
esac
echo "vendor/cupdlpx: pip-installing the official binding for CUDA arch ${ARCH}"

CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${ARCH}" \
    "$PYTHON" -m pip install --no-build-isolation --no-deps --force-reinstall ./upstream

"$PYTHON" -c "import cupdlpx; print('vendor/cupdlpx: build OK —', cupdlpx.__file__, cupdlpx.__version__)"
