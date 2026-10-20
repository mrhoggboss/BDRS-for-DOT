#!/usr/bin/env bash
# Fetches HALO at the pinned SHA into upstream/ and applies the source repairs
# below. upstream/ is gitignored. Run before build.sh. Fails on a SHA mismatch or on a repair
# whose anchor text is not found.
set -euo pipefail
cd "$(dirname "$0")"

PIN=$(tr -d '[:space:]' < PIN)
URL=https://github.com/WenzhouXia/HALO
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "vendor/halo: cloning $URL @ $PIN"
git clone --quiet "$URL" "$TMP/halo"
git -C "$TMP/halo" checkout --quiet --detach "$PIN"
ACTUAL=$(git -C "$TMP/halo" rev-parse HEAD)
if [ "$ACTUAL" != "$PIN" ]; then
    echo "FATAL: checked-out SHA $ACTUAL != PIN $PIN" >&2
    exit 1
fi
rm -rf "$TMP/halo/.git"

mkdir -p upstream
rsync -a --delete "$TMP/halo/" upstream/
echo "vendor/halo: upstream/ synced to $ACTUAL"

# Upstream's lp_solvers/cuPDLPx/.gitignore ignores *.cmake, so the published
# tree lacks pybind11's tools/*.cmake and add_subdirectory(pybind11) fails.
# Overlay the full pybind11 release of the vendored version (BSD-3-Clause).
PB11_DIR=upstream/src/hierarchical_ot/lp_solvers/cuPDLPx/pycupdlpx/pybind11
PB11_VER=v2.12.0
if [ ! -f "$PB11_DIR/tools/pybind11Common.cmake" ]; then
    echo "vendor/halo: vendored pybind11 lacks tools/*.cmake; overlaying full $PB11_VER"
    curl -fsSL "https://github.com/pybind/pybind11/archive/refs/tags/${PB11_VER}.tar.gz" \
        -o "$TMP/pb11.tgz"
    tar -xzf "$TMP/pb11.tgz" -C "$TMP"
    rsync -a "$TMP/pybind11-${PB11_VER#v}/" "$PB11_DIR/"
    test -f "$PB11_DIR/tools/pybind11Common.cmake" \
        || { echo "FATAL: pybind11 repair failed" >&2; exit 1; }
    echo "vendor/halo: pybind11 $PB11_VER overlay complete"
fi

# Source repairs, each an exact text match; a missing anchor means the pin
# moved, and the script fails rather than ship a bug unrepaired. The bugs:
# numpy >= 2 rejects int() of a shape-(1,) counter, and a round with no
# violations raises IndexError (both in find_violations_local_dt_gpu, used on
# grid levels of side >= 512); safe_malloc/safe_calloc take int, which truncates host
# allocation sizes of 2 GiB and more, and a failed allocation calls exit(),
# which Python cannot catch. Details: vendor/halo/README.md.
python3 - <<'REPAIRS'
import sys
from pathlib import Path

def repair(path, broken, fixed, name):
    p = Path(path)
    if not p.is_file():
        sys.exit(f"FATAL: {path} missing (pin bump moved it?) — repair {name!r} cannot apply")
    s = p.read_text()
    if fixed in s:
        return False
    if broken not in s:
        sys.exit(f"FATAL: repair {name!r} anchor not found in {path} — "
                 f"upstream drifted at this pin; re-verify the repair")
    if s.count(broken) != 1:
        sys.exit(f"FATAL: repair {name!r} anchor not unique in {path}")
    p.write_text(s.replace(broken, fixed))
    print(f"vendor/halo: repair {name!r} applied to {path}")
    return True

VK = "upstream/src/hierarchical_ot/modes/grid/violation_kernels.py"
CU = "upstream/src/hierarchical_ot/lp_solvers/cuPDLPx/cupdlpx/utils.cu"
CM = "upstream/src/hierarchical_ot/lp_solvers/cuPDLPx/pycupdlpx/CMakeLists.txt"
UH = "upstream/src/hierarchical_ot/lp_solvers/cuPDLPx/cupdlpx/utils.h"

repair(VK,
    "        tot = int(ctr1.get())\n",
    "        tot = int(ctr1.get()[0])   # repo repair: numpy>=2 rejects int(shape-(1,) array); see vendor/halo/README.md\n",
    "numpy2-counter-read")

repair(VK,
    "    order = cp.lexsort(cp.stack((g_idx, -g_slk)))\n"
    "    g_idx, g_slk = g_idx[order], g_slk[order]\n",
    "    if g_slk.size == 0:   # repo repair: mirror find_violations_global_gpu's empty guard; see vendor/halo/README.md\n"
    "        empty = cp.empty(0, cp.int64) if return_gpu else np.empty(0, np.int64)\n"
    "        return empty, np.float32(0.0)\n"
    "\n"
    "    order = cp.lexsort(cp.stack((g_idx, -g_slk)))\n"
    "    g_idx, g_slk = g_idx[order], g_slk[order]\n",
    "local-dt-empty-guard")

repair(CU,
    "void *safe_malloc(int size)",
    "void *safe_malloc(size_t size)   /* repo repair: int truncated >2GiB host allocs; see vendor/halo/README.md */",
    "size_t-malloc-def")
repair(CU,
    "void *safe_calloc(int num, int size)",
    "void *safe_calloc(size_t num, size_t size)   /* repo repair: see vendor/halo/README.md */",
    "size_t-calloc-def")
repair(UH,
    "void *safe_malloc(int size);",
    "void *safe_malloc(size_t size);   /* repo repair: see vendor/halo/README.md */",
    "size_t-malloc-decl")
repair(UH,
    "void *safe_calloc(int num, int size);",
    "void *safe_calloc(size_t num, size_t size);   /* repo repair: see vendor/halo/README.md */",
    "size_t-calloc-decl")
repair(UH,
    "#include <stdbool.h>\n",
    "#include <stdbool.h>\n#include <stddef.h>   /* repo repair: size_t for safe_malloc/safe_calloc */\n",
    "stddef-include")

# CUDA 13's headers (CCCL) require C++17; upstream builds the extension as C++14.
repair(CM,
    "set(CMAKE_CXX_STANDARD 14)\nset(CMAKE_CUDA_STANDARD 14)\n",
    "set(CMAKE_CXX_STANDARD 17)\nset(CMAKE_CUDA_STANDARD 17)\n",
    "cxx17-standard")

print("vendor/halo: all source repairs verified")
REPAIRS
