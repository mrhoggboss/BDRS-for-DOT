#!/usr/bin/env python3
"""Record the software stack a benchmark ran on, as JSON.

Package versions and Julia are read from the running environment, so
run this inside the environment that ran the benchmark; the toolchain fields
come from the run's ``header.json``; ``--repo`` is only read. Run it in the
torch environment, then with the OTT interpreter and ``--merge``.

    python scripts/capture_software_versions.py \
        --repo ~/bdrs --header <output-dir>/header.json --out software_versions.json
"""
import argparse
import datetime as _dt
import json
import os
import re
import shutil
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

PY_PACKAGES = [
    "torch", "numpy", "scipy", "POT", "ott-jax", "jax", "jaxlib",
    "trimesh", "matplotlib", "pandas",
]


def _run(cmd: List[str], timeout: float = 20.0) -> Optional[str]:
    if not shutil.which(cmd[0]):
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    return (out.stdout or out.stderr).strip() or None


def python_packages() -> Dict[str, Optional[str]]:
    from importlib.metadata import PackageNotFoundError, version
    out: Dict[str, Optional[str]] = {}
    for name in PY_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def vendor_pins(repo: Path) -> Dict[str, Dict[str, Any]]:
    """Per vendored baseline: the upstream commit in its ``PIN`` file (``None``
    without one) and the last repository commit touching its ``src/``, or the
    whole subtree when it has no ``src/``."""
    out: Dict[str, Dict[str, Any]] = {}
    vend = repo / "vendor"
    if not vend.is_dir():
        return out
    for d in sorted(p for p in vend.iterdir() if p.is_dir()):
        pin = d / "PIN"
        rec: Dict[str, Any] = {"upstream_commit": None, "kind": "vendored snapshot"}
        if pin.is_file():
            m = re.search(r"[0-9a-f]{40}", pin.read_text())
            if m:
                rec = {"upstream_commit": m.group(0), "kind": "pinned upstream"}
        # src/ when it exists: the rest of the subtree holds our harness (PDOT's
        # bench_driver.jl, DROT's shim/), whose edits leave the solver unchanged.
        sub = f"vendor/{d.name}/src" if (d / "src").is_dir() else f"vendor/{d.name}"
        sha = _run(["git", "-C", str(repo), "log", "-1", "--format=%H", "--", sub])
        rec["repo_commit"] = sha
        rec["repo_commit_path"] = sub
        out[d.name] = rec
    return out


def julia_stack(repo: Path) -> Dict[str, Optional[str]]:
    """The ``julia`` on PATH and the CUDA.jl version in ``vendor/PDOT.jl/Manifest.toml``."""
    out: Dict[str, Optional[str]] = {"julia": None, "CUDA.jl": None}
    v = _run(["julia", "--version"])
    if v:
        out["julia"] = v.replace("julia version", "").strip()
    man = repo / "vendor" / "PDOT.jl" / "Manifest.toml"
    if man.is_file():
        txt = man.read_text()
        blk = re.search(r'^\[\[deps\.CUDA\]\]$(.*?)^\[', txt, re.S | re.M)
        if blk:
            m = re.search(r'^version\s*=\s*"([^"]+)"', blk.group(1), re.M)
            if m:
                out["CUDA.jl"] = m.group(1)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", type=Path, default=Path("."),
                   help="the repository checkout that ran the benchmark, read "
                        "only (for vendor pins)")
    p.add_argument("--header", type=Path, default=None,
                   help="the benchmark's header.json; supplies the device and "
                        "toolchain")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--merge", action="store_true",
                   help="Union into an existing --out instead of overwriting "
                        "it. No single environment holds the whole stack: "
                        "SinkhornGrid runs in its own JAX interpreter that "
                        "never imports torch, so a second pass with that "
                        "interpreter is required and, without this flag, "
                        "silently nulls torch/POT/trimesh from the "
                        "first. Run the benchmark's torch environment first "
                        "(the header describes it, so it must be the base), "
                        "then the OTT interpreter with --merge.")
    a = p.parse_args(argv)

    header: Dict[str, Any] = {}
    if a.header and a.header.is_file():
        header = json.loads(a.header.read_text())

    rec = {
        "captured_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "captured_on": os.uname().nodename,
        # sys.prefix, not $CONDA_PREFIX: the --merge pass runs an interpreter by
        # path without activating its env, so CONDA_PREFIX names the shell's env.
        "env_prefix": sys.prefix,
        "env_executable": sys.executable,
        "header_source": str(a.header) if a.header else None,
        "platform": header.get("platform"),
        "python": header.get("python_version"),
        "torch": header.get("torch_version"),
        "torch_cuda": header.get("torch_cuda_version"),
        "nvcc": header.get("nvcc_version"),
        "device": header.get("torch_device_name"),
        "nvidia_smi_query_gpu": header.get("nvidia_smi_query_gpu"),
        "campaign_git_sha": header.get("git_sha"),
        "python_packages": python_packages(),
        "julia": julia_stack(a.repo),
        "vendored_baselines": vendor_pins(a.repo),
    }
    # A torch version other than the header's means this is not the environment
    # that ran the benchmark, so the live fields describe the wrong stack.
    live_torch = rec["python_packages"].get("torch")
    hdr_torch = rec.get("torch")
    rec["environment_matches_header"] = (
        None if not (live_torch and hdr_torch) else live_torch == hdr_torch)

    if a.merge and a.out.is_file():
        base = json.loads(a.out.read_text())
        note = base.get("python_packages_env_note") or {}
        adopted = []
        for k, v in rec["python_packages"].items():
            # Fill gaps only: the base file comes from the environment the
            # header describes, so its values win.
            if v is not None and base["python_packages"].get(k) is None:
                base["python_packages"][k] = v
                adopted.append(k)
        note.update({
            "keys_from_secondary_env": sorted(set(note.get(
                "keys_from_secondary_env", [])) | set(adopted)),
            "primary_env": base.get("env_prefix", "unknown"),
            "secondary_env": rec.get("env_prefix", "unknown"),
            "why": ("No single environment holds the whole stack: SinkhornGrid "
                    "runs in its own JAX interpreter that never imports torch. "
                    "This file is the union, base taken from the environment "
                    "the run header pins."),
        })
        base["python_packages_env_note"] = note
        base["merged_utc"] = rec["captured_utc"]
        rec = base
        print(f"  merged into {a.out}: adopted {adopted or 'nothing new'}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")

    print(f"capture_software_versions: wrote {a.out}")
    if rec["environment_matches_header"] is False:
        print(f"  *** WRONG ENVIRONMENT: header says torch {hdr_torch}, this "
              f"interpreter has {live_torch}. The live-only fields (POT, JAX, "
              f"Julia) describe a machine that did not run the "
              f"benchmark. Re-run inside the benchmark's environment.")
    elif rec["environment_matches_header"] is None:
        print("  (could not confirm the environment against the header)")
    missing = [k for k, v in rec["python_packages"].items() if v is None]
    if missing:
        print(f"  not installed in THIS env (recorded as null): {missing}")
        print("  NOTE: SinkhornGrid runs in a separate OTT-JAX env. Re-run this")
        print("        with that interpreter and merge, or ott-jax/jax stay null.")
    unpinned = [k for k, v in rec["vendored_baselines"].items()
                if v.get("upstream_commit") is None]
    if unpinned:
        print(f"  vendored with no upstream pin: {unpinned} (repo commit only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
