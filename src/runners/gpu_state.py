"""Which GPU a result row names.

A process's CUDA device ``i`` is the ``i``-th entry of CUDA_VISIBLE_DEVICES when
that variable is set, and nvidia-smi ignores the variable, so ``nvidia-smi -i i``
can describe another job's GPU. Every query here maps the index through the
variable first.
"""

import os
import subprocess
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple


def visible_selector(idx: int) -> str:
    """The nvidia-smi selector of this process's CUDA device ``idx``."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        parts = [p.strip() for p in cvd.split(",") if p.strip()]
        if idx < len(parts):
            return parts[idx]
    return str(idx)


@lru_cache(maxsize=None)
def _smi_query(selector: str, field: str) -> Optional[str]:
    """One ``nvidia-smi --query-gpu`` field of one device, or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader", "-i", selector],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    s = out.stdout.strip()
    return s if out.returncode == 0 and s else None


def device_uuid(idx: int = 0) -> Optional[str]:
    """UUID of this process's CUDA device ``idx``, or None when nvidia-smi cannot answer."""
    sel = visible_selector(idx)
    if sel.startswith(("GPU-", "MIG-")):
        return sel
    return _smi_query(sel, "uuid")


def device_identity(idx: int = 0) -> Tuple[Optional[str], Optional[str]]:
    """``(name, UUID)`` of this process's CUDA device ``idx``, without torch."""
    return _smi_query(visible_selector(idx), "name"), device_uuid(idx)


def device_fingerprint(idx: int = 0) -> Dict[str, Any]:
    """Torch's properties of CUDA device ``idx``, with its UUID when known; {} without torch CUDA."""
    try:
        import torch
    except ImportError:
        return {}
    if not torch.cuda.is_available():
        return {}
    if idx >= torch.cuda.device_count():
        return {"error": f"device {idx} out of range; have {torch.cuda.device_count()}"}

    props = torch.cuda.get_device_properties(idx)
    out: Dict[str, Any] = {
        "device_index": idx,
        "device_name": props.name,
        "total_memory_bytes": int(props.total_memory),
        "multi_processor_count": int(props.multi_processor_count),
        "compute_capability": f"{props.major}.{props.minor}",
    }
    uuid_str = device_uuid(idx)
    if uuid_str:
        out["device_uuid"] = uuid_str
    return out
