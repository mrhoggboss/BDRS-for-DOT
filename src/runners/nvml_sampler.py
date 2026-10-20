"""
Peak GPU memory of a solver that allocates outside torch's caching allocator
(raw ``cudaMalloc``, CuPy, JAX), which ``torch.cuda.max_memory_allocated``
does not see.

NVML reports the used memory of the whole device, all processes included, so
the peak is one solve's only because GPU work is serialized. The counter is
polled, so an allocation freed between two samples is missed. No torch
import: the JAX runner uses this module.
"""

import threading
from typing import Optional, Tuple

from src.runners.gpu_state import device_uuid


def _handle(pynvml, idx: int):
    """NVML handle of this process's CUDA device ``idx``. NVML ignores
    CUDA_VISIBLE_DEVICES, so the device is found by its UUID; NVML index
    ``idx``, which may name another GPU, is the fallback."""
    uuid_s = device_uuid(idx)
    if uuid_s:
        try:
            return pynvml.nvmlDeviceGetHandleByUUID(uuid_s)
        except Exception:                                   # noqa: BLE001
            pass
    return pynvml.nvmlDeviceGetHandleByIndex(idx)


class NvmlPeakSampler:
    """Polls the device's used memory on a daemon thread from ``start`` to ``stop``.

    ``stop`` returns ``(peak_bytes, baseline_bytes)``, the largest sample and
    the value read in ``start``; both are None when pynvml is missing or NVML
    fails in ``start``. One window per sampler: ``stop`` leaves the stop event
    set.
    """

    def __init__(self, device_index: int, interval_s: float = 0.05):
        self._idx = device_index
        self._interval = interval_s
        self._peak: Optional[int] = None
        self._baseline: Optional[int] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvml = None

    def start(self) -> None:
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = _handle(pynvml, self._idx)
            self._baseline = int(pynvml.nvmlDeviceGetMemoryInfo(self._handle).used)
            self._peak = self._baseline
        except Exception:                                   # noqa: BLE001
            self._nvml = None
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                used = int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used)
                if self._peak is None or used > self._peak:
                    self._peak = used
            except Exception:                               # noqa: BLE001
                break
            self._stop.wait(self._interval)

    def stop(self) -> Tuple[Optional[int], Optional[int]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:                               # noqa: BLE001
                pass
        return self._peak, self._baseline
