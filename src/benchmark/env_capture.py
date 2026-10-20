"""
The environment fingerprint of a benchmark run, written as ``header.json``.

Every probe is best-effort: a command that is missing, times out or fails is
omitted, and a torch failure is recorded as ``torch_capture_error``.
``fingerprint_sha256`` hashes the capture; the orchestrator's log lines carry
it as ``header_sha``.
"""

import hashlib
import json
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from src.protocol.instance_record import utc_now_iso
from src.runners.gpu_state import device_uuid


def _safe_cmd(cmd: List[str], timeout: float = 10.0, *,
              allow_empty: bool = False) -> Optional[str]:
    """Stripped stdout of ``cmd``; None if it is missing, times out, fails or,
    unless ``allow_empty``, prints nothing."""
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    text = out.stdout.strip()
    return text if (text or allow_empty) else None


def capture(repo_root: Union[str, Path]) -> Dict[str, Any]:
    """Capture an environment fingerprint as a plain dict."""
    repo_root = str(Path(repo_root).resolve())

    info: Dict[str, Any] = {
        "captured_utc": utc_now_iso(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "repo_root": repo_root,
    }

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["torch_cuda_version"] = torch.version.cuda
        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["torch_device_name"] = props.name
            info["torch_device_total_mem_bytes"] = int(props.total_memory)
            info["torch_device_compute_capability"] = f"{props.major}.{props.minor}"
    except Exception as e:    # noqa: BLE001
        info["torch_capture_error"] = repr(e)

    nvsmi = _safe_cmd([
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if nvsmi:
        info["nvidia_smi_query_gpu"] = nvsmi

    nvsmi_clocks = _safe_cmd([
        "nvidia-smi",
        "--query-gpu=clocks.gr,clocks.sm,clocks.mem,clocks.applications.gr",
        "--format=csv,noheader,nounits",
    ])
    if nvsmi_clocks:
        info["nvidia_smi_clocks"] = nvsmi_clocks

    nvcc = _safe_cmd(["nvcc", "--version"])
    if nvcc:
        # The last line is the build tag, which names the CUDA version.
        info["nvcc_version"] = nvcc.splitlines()[-1]
        info["nvcc_full"] = nvcc

    git_sha = _safe_cmd(["git", "-C", repo_root, "rev-parse", "HEAD"])
    if git_sha:
        info["git_sha"] = git_sha
    # A clean tree prints nothing and is recorded as git_dirty = False.
    git_status = _safe_cmd(["git", "-C", repo_root, "status", "--porcelain"],
                           allow_empty=True)
    if git_status is not None:
        info["git_dirty"] = bool(git_status)
        if git_status:
            info["git_status_porcelain"] = git_status[:8000]

    digest_payload = json.dumps(info, sort_keys=True).encode("utf-8")
    info["fingerprint_sha256"] = hashlib.sha256(digest_payload).hexdigest()
    return info


def write(path: Union[str, Path], repo_root: Union[str, Path]) -> Dict[str, Any]:
    """Capture the fingerprint, write it to ``path`` as JSON and return it."""
    info = capture(repo_root)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, sort_keys=True, indent=2))
    return info


def cached_device_uuid(device_index: Optional[int] = None) -> Optional[str]:
    """UUID of torch CUDA device ``device_index`` (default: the current device),
    or None without torch, CUDA or nvidia-smi. The nvidia-smi answer is cached."""
    if device_index is None:
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            device_index = torch.cuda.current_device()
        except Exception:      # noqa: BLE001
            return None
    return device_uuid(device_index)
