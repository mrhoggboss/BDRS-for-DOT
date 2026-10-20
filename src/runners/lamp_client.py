"""Client of the persistent LAMP solve server (``vendor/lamp/lamp_server.jl``).

A fresh Julia process pays start-up, the CUDA.jl load and kernel compilation on
every call, so one server compiles once per job and serves every LAMP solve of
it. It keeps code only: before every solve it frees the device and the heap, so
nothing instance-specific crosses solves.

The first runner of a job starts the server in its own session, so it outlives
the runner; later runners find it by its socket. It exits after
``BDRS_LAMP_IDLE_TIMEOUT`` seconds idle. The socket name carries SLURM_JOB_ID,
the Slurm step, CUDA_VISIBLE_DEVICES, the device index and a hash of this
checkout, and a ping returns a fingerprint of the shim, the server, PIN and the
Julia environment, so a runner never uses a server built from other code or
bound to another GPU. The orchestrator is serial, so a server that is busy when
a runner arrives is running an orphaned solve (its runner was killed) and is
restarted. Since an idle server holds a CUDA context and an orphaned solve
keeps the GPU busy, the orchestrator runs LAMP only in a job of its own.

Environment: BDRS_LAMP_JULIA (the Julia command, default "julia"; e.g.
"julia +1.11"), BDRS_LAMP_PROJECT, BDRS_LAMP_UPSTREAM, BDRS_LAMP_SOCKET,
BDRS_LAMP_IDLE_TIMEOUT (s, default 300), BDRS_LAMP_STARTUP_TIMEOUT (s, default
1200).

    python -m src.runners.lamp_client            # start or ping the server
    python -m src.runners.lamp_client shutdown   # stop it
"""

import contextlib
import fcntl
import hashlib
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.protocol.paths import REPO_ROOT

VENDOR = REPO_ROOT / "vendor" / "lamp"
SERVER_JL = VENDOR / "lamp_server.jl"
SHIM_JL = VENDOR / "lamp_shim.jl"
PIN_FILE = VENDOR / "PIN"


class LampServerError(RuntimeError):
    """The server could not be started or reached."""


class LampTimeout(LampServerError):
    """A solve did not answer within the runner's remaining wall time."""


class LampServerDied(LampServerError):
    """The connection closed without a response (a crash or a kill mid-solve)."""


class _ConnectFailed(OSError):
    """Nothing accepted the connection: the request reached no server."""


def project_dir() -> Path:
    return Path(os.environ.get("BDRS_LAMP_PROJECT", VENDOR))


def upstream_dir() -> Path:
    return Path(os.environ.get("BDRS_LAMP_UPSTREAM", VENDOR / "upstream"))


def julia_cmd() -> list:
    return shlex.split(os.environ.get("BDRS_LAMP_JULIA", "julia"))


def idle_timeout_s() -> int:
    return int(os.environ.get("BDRS_LAMP_IDLE_TIMEOUT", "300"))


def startup_timeout_s() -> float:
    return float(os.environ.get("BDRS_LAMP_STARTUP_TIMEOUT", "1200"))


def fingerprint() -> str:
    """Identity of the code a server runs: shim, server, PIN and Julia environment."""
    h = hashlib.sha256()
    for p in (SHIM_JL, SERVER_JL, PIN_FILE, VENDOR / "Project.toml", VENDOR / "Manifest.toml"):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def upstream_ready() -> Optional[str]:
    """None if ``upstream/`` is fetched at the PIN, else why it is not."""
    stamp = upstream_dir() / ".bdrs_pin"
    if not stamp.is_file():
        return f"{stamp} missing; run vendor/lamp/fetch.sh"
    pin = PIN_FILE.read_text().strip()
    got = stamp.read_text().strip()
    if got != pin:
        return f"upstream fetched at {got} but PIN is {pin}; re-run vendor/lamp/fetch.sh"
    return None


def socket_path(device: int = 0) -> Path:
    override = os.environ.get("BDRS_LAMP_SOCKET")
    if override:
        return Path(override)
    tag = os.environ.get("SLURM_JOB_ID", "local")
    key = "|".join([str(VENDOR.resolve()), os.environ.get("SLURM_STEP_ID", ""),
                    os.environ.get("CUDA_VISIBLE_DEVICES", ""), str(device)])
    h = hashlib.sha256(key.encode()).hexdigest()[:8]
    # Under /tmp, the path stays within the 104-byte AF_UNIX limit of macOS.
    return Path("/tmp") / f"bdrs-lamp-{os.getuid()}-{tag}-{h}.sock"


def _side(sp: Path, suffix: str) -> Path:
    return sp.with_name(sp.name + suffix)


def log_path(device: int = 0) -> Path:
    return _side(socket_path(device), ".log")


def _rpc(sp: Path, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout_s)
    try:
        try:
            s.connect(str(sp))
        except (ConnectionRefusedError, FileNotFoundError) as e:
            raise _ConnectFailed(str(e)) from e
        s.sendall((json.dumps(payload) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(1 << 16)
            if not chunk:
                break
            buf += chunk
    finally:
        s.close()
    if not buf.strip():
        raise LampServerDied("connection closed without a response")
    return json.loads(buf.decode())          # accepts the Infinity and NaN tokens


def _ping(sp: Path, timeout_s: float = 10.0) -> Tuple[Optional[Dict[str, Any]], str]:
    """(pong, state), state one of absent, refused, busy, dead and ok."""
    if not sp.exists():
        return None, "absent"
    try:
        pong = _rpc(sp, {"op": "ping"}, timeout_s)
    except _ConnectFailed:
        return None, "refused"
    except socket.timeout:
        return None, "busy"
    except (LampServerDied, OSError, ValueError):
        return None, "dead"
    return (pong, "ok") if pong.get("status") == "pong" else (None, "dead")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _is_our_server(pid: int, sp: Path) -> bool:
    """Whether ``pid`` is a LAMP server bound to this socket, checked before any
    signal, since a reused PID can belong to another process."""
    if not _pid_alive(pid):
        return False
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        try:
            cmd = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                                 capture_output=True, text=True, timeout=5).stdout
        except Exception:                                   # noqa: BLE001
            return False
    return "lamp_server.jl" in cmd and f"--socket {sp}" in cmd


def _read_pidfile(sp: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(_side(sp, ".pid").read_text())
    except (OSError, ValueError):
        return None


def _kill(pid: int, sp: Path) -> None:
    if not _is_our_server(pid, sp):
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    for _ in range(50):
        with contextlib.suppress(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)      # reaps it when this process launched it
        if not _pid_alive(pid):
            return
        time.sleep(0.1)


def _cleanup(sp: Path) -> None:
    for p in (sp, _side(sp, ".pid")):
        with contextlib.suppress(FileNotFoundError):
            p.unlink()


@contextlib.contextmanager
def _locked(sp: Path):
    with open(_side(sp, ".lock"), "a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _launch(sp: Path, fp: str, device: int) -> int:
    cmd = julia_cmd() + [f"--project={project_dir()}", "--startup-file=no", str(SERVER_JL),
                         "--socket", str(sp), "--idle-timeout", str(idle_timeout_s()),
                         "--upstream", str(upstream_dir()), "--fingerprint", fp,
                         "--warmup", "1", "--device", str(device)]
    log = open(_side(sp, ".log"), "ab")
    log.write(f"\n=== launch {time.strftime('%Y-%m-%dT%H:%M:%S')} {' '.join(cmd)}\n".encode())
    log.flush()
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                            start_new_session=True, close_fds=True)
    log.close()
    _side(sp, ".pid").write_text(json.dumps(
        {"pid": proc.pid, "started": time.time(), "fingerprint": fp}))
    return proc.pid


def _await_ready(sp: Path, fp: str, pid: int, deadline: float) -> Dict[str, Any]:
    while time.time() < deadline:
        if not _pid_alive(pid):
            raise LampServerError(f"LAMP server (pid {pid}) exited during start-up.\n{_tail(sp)}")
        pong, _ = _ping(sp, timeout_s=5.0)
        if pong is not None:
            if pong.get("fingerprint") != fp:
                _kill(pid, sp)
                raise LampServerError("LAMP server answered with a different code fingerprint")
            return pong
        time.sleep(1.0)
    _kill(pid, sp)
    raise LampServerError(f"LAMP server not ready after {startup_timeout_s():.0f} s.\n{_tail(sp)}")


def _tail(sp: Path, since: int = 0, n: int = 60) -> str:
    p = _side(sp, ".log")
    if not p.exists():
        return ""
    with p.open("rb") as f:
        f.seek(0, 2)
        f.seek(max(since, f.tell() - 16384))
        return "\n".join(f.read().decode("utf-8", "replace").splitlines()[-n:])


def ensure_server(device: int = 0) -> Dict[str, Any]:
    """A pong from a warm server running this checkout's code on ``device``,
    starting one (compilation and warm-up) when there is none."""
    sp = socket_path(device)
    fp = fingerprint()
    pong, _ = _ping(sp)
    if pong is not None and pong.get("fingerprint") == fp:
        return pong
    with _locked(sp):
        pong, state = _ping(sp)
        if pong is not None and pong.get("fingerprint") == fp:
            return pong
        info = _read_pidfile(sp)
        if info and _is_our_server(int(info["pid"]), sp):
            starting = (state in ("absent", "refused") and info.get("fingerprint") == fp
                        and time.time() - float(info.get("started", 0)) < startup_timeout_s())
            if starting:                       # same code, still compiling or warming up
                return _await_ready(sp, fp, int(info["pid"]),
                                    float(info["started"]) + startup_timeout_s())
            _kill(int(info["pid"]), sp)        # other code, an orphaned solve, or a hung start
        _cleanup(sp)
        pid = _launch(sp, fp, device)
        return _await_ready(sp, fp, pid, time.time() + startup_timeout_s())


def solve(payload: Dict[str, Any], *, timeout_s: float, device: int = 0) -> Dict[str, Any]:
    """One solve on the server, after ``ensure_server``; ``timeout_s`` bounds the
    request. Raises LampTimeout, LampServerDied or LampServerError. A request
    that reaches no server (it exited between the ping and the request) is sent
    once more to a relaunched server, within the remaining budget."""
    sp = socket_path(device)
    since = _side(sp, ".log").stat().st_size if _side(sp, ".log").exists() else 0
    deadline = time.time() + timeout_s
    for attempt in (0, 1):
        try:
            return _rpc(sp, {"op": "solve", **payload}, timeout_s=max(1.0, deadline - time.time()))
        except _ConnectFailed as e:
            if attempt == 1:
                raise LampServerDied(f"{e!r}: no server accepted the request after a relaunch.\n"
                                     f"{_tail(sp, since)}") from e
            ensure_server(device)
            if deadline - time.time() <= 1.0:
                raise LampTimeout("no wall time left after relaunching the LAMP server") from e
        except socket.timeout as e:
            info = _read_pidfile(sp)
            if info:
                _kill(int(info["pid"]), sp)
            _cleanup(sp)
            raise LampTimeout(f"no response within {timeout_s:.0f} s") from e
        except (LampServerDied, ConnectionError, OSError, ValueError) as e:
            info = _read_pidfile(sp)
            alive = bool(info and _is_our_server(int(info["pid"]), sp))
            # Only this solve's part of the shared log, so that an earlier
            # solve's lines cannot decide how this failure is classified.
            raise LampServerDied(f"{e!r}; server alive={alive}.\n{_tail(sp, since)}") from e
    raise AssertionError("unreachable")


def shutdown(device: int = 0) -> None:
    """Stop the server of this checkout, job and device, if one is running."""
    sp = socket_path(device)
    with contextlib.suppress(Exception):
        _rpc(sp, {"op": "shutdown"}, timeout_s=10.0)
    info = _read_pidfile(sp)
    if info:
        _kill(int(info["pid"]), sp)
    _cleanup(sp)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "shutdown":
        shutdown()
        print("shut down")
    else:
        print(json.dumps(ensure_server()))
