"""The GPU lock must refuse a live holder and clear a dead one.

Regression tier for a hazard that cannot be caught any other way: two compute contexts on this
box abort the process natively out of `mm` backward, so there is no exception to handle. The only
defence is refusing to start, which makes this guard's correctness load-bearing.
"""

from __future__ import annotations

import os

import pytest

from fka.kernel.gpu_lock import gpu_lock


def test_refuses_when_a_live_process_holds_the_lock(tmp_path):
    lock = tmp_path / "gpu.lock"
    lock.write_text(f"{os.getpid()}\n", encoding="utf-8")  # this process is definitely alive
    with pytest.raises(SystemExit, match="already in use"), gpu_lock(path=lock):
        pass


def test_clears_a_stale_lock_from_a_dead_pid(tmp_path, capsys):
    lock = tmp_path / "gpu.lock"
    lock.write_text("999999999\n", encoding="utf-8")  # not a live pid
    with gpu_lock(path=lock):
        assert lock.read_text(encoding="utf-8").strip() == str(os.getpid())
    assert "stale" in capsys.readouterr().out


def test_releases_on_exit_including_on_error(tmp_path):
    lock = tmp_path / "gpu.lock"
    with gpu_lock(path=lock):
        assert lock.exists()
    assert not lock.exists()

    with pytest.raises(RuntimeError), gpu_lock(path=lock):
        raise RuntimeError("boom")
    assert not lock.exists(), "a crashed run must not leave the box locked"


def test_force_overrides_a_live_holder(tmp_path):
    lock = tmp_path / "gpu.lock"
    lock.write_text(f"{os.getpid()}\n", encoding="utf-8")
    with gpu_lock(path=lock, force=True):
        assert lock.exists()
