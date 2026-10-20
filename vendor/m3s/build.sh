#!/usr/bin/env bash
# Builds the M3S shim (shim/m3s_shim.cu plus the upstream sources listed in
# shim/CMakeLists.txt) into _build/m3s_shim (gitignored). Upstream's top-level
# CMake is never used: it carries committed CMake caches and requires Eigen
# and OpenMP for targets the shim does not build.
#
# Needs the CUDA toolkit (nvcc, cuBLAS, cuSPARSE), CMake >= 3.18 and GCC (any
# version) as the host compiler of both the C++ and the CUDA sources (CXX and
# CUDAHOSTCXX select it; CUDAHOSTCXX defaults to $CXX, else g++). The CUDA
# arch is read from the visible device; override with M3S_CUDA_ARCH.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f upstream/SparseSinkhorn/solver.cu ]; then
    echo "FATAL: upstream/ missing or incomplete — run vendor/m3s/fetch.sh first" >&2
    exit 1
fi

# Upstream's host code falls off the end of int-returning functions, and only
# GCC's -O0 behaviour for that is the one upstream was published under, so
# the host compiler must be GCC (shim/CMakeLists.txt adds -fno-unreachable-traps
# when the GCC has it). CMake compiles the C++ sources with $CXX (default c++)
# and nvcc hands the host code to $CUDAHOSTCXX, which defaults to $CXX, else g++.
export CUDAHOSTCXX="${CUDAHOSTCXX:-${CXX:-g++}}"
for HOSTCXX in "${CXX:-c++}" "$CUDAHOSTCXX"; do
    HOSTVER=$("$HOSTCXX" --version 2>/dev/null || true)
    case "$HOSTVER" in
        *"Free Software Foundation"*) ;;
        *)
            HOSTVER=${HOSTVER%%$'\n'*}
            echo "FATAL: M3S needs GCC as host compiler; '$HOSTCXX' is '${HOSTVER:-not found}'." >&2
            echo "       Set CXX and CUDAHOSTCXX to a g++." >&2
            exit 1
            ;;
    esac
done

# nvidia-smi prints query errors to stdout and may report "[N/A]" (MIG,
# virtualized GPUs), so a detected value is used only when it is all digits.
# M3S_CUDA_ARCH reaches CMake as given.
ARCH=${M3S_CUDA_ARCH:-}
if [ -z "$ARCH" ]; then
    ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
           | head -1 | tr -d '. \r' || true)
    case "$ARCH" in
        ''|*[!0-9]*) ARCH=80 ;;
    esac
fi
echo "vendor/m3s: building m3s_shim for CUDA arch ${ARCH}"

cmake -S shim -B _build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${ARCH}"
cmake --build _build -j"$(nproc)"

test -x _build/m3s_shim || { echo "FATAL: _build/m3s_shim not produced" >&2; exit 1; }
echo "vendor/m3s: build OK — $(ls -la _build/m3s_shim | awk '{print $5, $9}')"
