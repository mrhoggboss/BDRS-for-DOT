"""Build the BDRS CUDA extension (``src/methods/cuda/``) once per process.

torch caches the built library on disk (``TORCH_EXTENSIONS_DIR`` when set), so
a later process with unchanged sources and flags loads it without recompiling.
"""

import os
import threading

import torch

_EXTENSION = None
_EXTENSION_LOCK = threading.Lock()


def load():
    """Return the extension module, compiling it on the first call.

    After a failed build torch has already recorded the hash of the sources and
    flags, so a retry in the same process does not rebuild: it imports whatever
    library the build directory holds.
    """
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    with _EXTENSION_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        if not torch.cuda.is_available():
            raise RuntimeError(
                "the BDRS CUDA extension requires a CUDA device, but torch.cuda.is_available() is False."
            )
        from torch.utils.cpp_extension import load as _jit_load

        # Compile for this device only: with the list unset, torch compiles for
        # every visible device's architecture plus PTX.
        if "TORCH_CUDA_ARCH_LIST" not in os.environ:
            cap = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cap[0]}.{cap[1]}"

        here = os.path.dirname(os.path.abspath(__file__))
        cuda_dir = os.path.join(here, "cuda")
        sources = [
            os.path.join(cuda_dir, "bdrs_solver.cpp"),
            os.path.join(cuda_dir, "bdrs_kernels.cu"),
        ]
        _EXTENSION = _jit_load(
            name="bdrs_cuda_ext",
            sources=sources,
            extra_cuda_cflags=["-O3", "-lineinfo"],
            extra_cflags=["-O3"],
            verbose=False,
        )
        return _EXTENSION


def loaded():
    """Return the extension module if ``load()`` has succeeded in this process, else None."""
    return _EXTENSION
