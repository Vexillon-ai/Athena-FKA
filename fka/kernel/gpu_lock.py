"""One GPU job at a time on this box, enforced loudly.

Two concurrent compute contexts on the gfx1151 iGPU crash the process **natively** — a C++
stack dump out of `mm` backward, not a Python exception (observed 2026-08-02 when the M2 Stage A
fit was launched against a running 50M training job). A native abort cannot be caught, retried, or
attributed: it looks like a model bug and it is a hardware/driver limit.

So the guard has to be cooperative and *before* any device work: take a lockfile, and refuse to
start if another live process holds it.

    with gpu_lock():
        ...  # anything that touches the device
"""

from __future__ import annotations

import atexit
import os
from contextlib import contextmanager
from pathlib import Path

LOCK_PATH = Path(os.environ.get("FKA_GPU_LOCK", Path.home() / ".fka-gpu.lock"))


def _alive(pid: int) -> bool:
    """Is that pid still running? Windows and POSIX."""
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            import subprocess

            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            return str(pid) in out
        os.kill(pid, 0)
        return True
    except Exception:  # noqa: BLE001 - a failed liveness check must not block a legitimate run
        return False


@contextmanager
def gpu_lock(*, path: Path | None = None, force: bool = False):
    """Hold the box's single GPU slot. Raises SystemExit if another live job holds it."""
    path = Path(path or LOCK_PATH)
    if path.exists() and not force:
        try:
            holder = int(path.read_text(encoding="utf-8").split()[0])
        except Exception:  # noqa: BLE001 - a corrupt lock is a stale lock
            holder = -1
        if _alive(holder):
            raise SystemExit(
                f"GPU is already in use by pid {holder} (lock: {path}).\n"
                f"This box crashes NATIVELY in mm backward under two compute contexts, so this "
                f"run is refusing to start rather than taking the machine down.\n"
                f"Wait for it, run with --device cpu, or pass --force-gpu-lock if you are certain "
                f"that pid is not using the GPU."
            )
        print(f"!! removing a stale GPU lock from pid {holder}")
        path.unlink(missing_ok=True)

    path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    released = False

    def release() -> None:
        nonlocal released
        if released:
            return
        released = True
        try:
            if path.exists() and path.read_text(encoding="utf-8").split()[0] == str(os.getpid()):
                path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass

    atexit.register(release)
    try:
        yield path
    finally:
        release()
