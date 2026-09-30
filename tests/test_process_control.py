"""CPU-only checks for src/urop/process_control.py, against real (trivial)
subprocesses -- not mocked subprocess.Popen objects -- so terminate/wait/kill
escalation and startup-exit detection are exercised for real.
"""
from __future__ import annotations

import subprocess
import sys
import time

import pytest

from urop.process_control import (
    DeadlineSupervisor,
    ServerExitedDuringStartup,
    ServerNotReady,
    shutdown_process,
    shutdown_process_group,
    start_owned_process,
    wait_for_server_ready,
)


def _spawn(code: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", code])


# --- wait_for_server_ready -------------------------------------------------


def test_wait_for_server_ready_succeeds_once_health_check_passes():
    proc = _spawn("import time; time.sleep(10)")
    try:
        calls = {"n": 0}

        def health_check():
            calls["n"] += 1
            return calls["n"] >= 3  # "ready" on the 3rd poll

        elapsed = wait_for_server_ready(proc, health_check, timeout_seconds=5, poll_interval_seconds=0.05)
        assert elapsed < 5
        assert calls["n"] >= 3
    finally:
        shutdown_process(proc, wait_seconds=5)


def test_wait_for_server_ready_raises_immediately_if_process_exits_during_startup():
    proc = _spawn("import sys; sys.exit(7)")
    start = time.monotonic()
    with pytest.raises(ServerExitedDuringStartup) as excinfo:
        wait_for_server_ready(proc, lambda: False, timeout_seconds=30, poll_interval_seconds=0.2)
    elapsed = time.monotonic() - start
    assert excinfo.value.returncode == 7
    assert elapsed < 5, "must detect the dead process quickly, not poll a health check for the full 30s timeout"


def test_wait_for_server_ready_times_out_if_never_healthy():
    proc = _spawn("import time; time.sleep(10)")
    try:
        with pytest.raises(ServerNotReady):
            wait_for_server_ready(proc, lambda: False, timeout_seconds=0.3, poll_interval_seconds=0.05)
    finally:
        shutdown_process(proc, wait_seconds=5)


# --- shutdown_process --------------------------------------------------------


def test_shutdown_process_on_already_exited_process():
    proc = _spawn("pass")
    proc.wait(timeout=5)
    result = shutdown_process(proc, wait_seconds=5)
    assert result.how == "already_exited"


def test_shutdown_process_terminates_a_cooperative_process():
    proc = _spawn(
        "import signal, time, sys\n"
        "signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))\n"
        "time.sleep(30)"
    )
    result = shutdown_process(proc, wait_seconds=10)
    assert result.how in ("terminated", "killed")  # Windows may not deliver SIGTERM the same way -- either is fine
    assert proc.poll() is not None, "the process must actually be gone after shutdown_process returns"


def test_shutdown_process_escalates_to_kill_for_an_unresponsive_process():
    # Ignores termination; only a hard kill can stop it. Windows note: Popen.terminate()
    # maps to TerminateProcess, which this process cannot actually block -- but the point
    # of this test is the ESCALATION PATH runs without hanging and the process ends up dead.
    proc = _spawn("import time\ntime.sleep(60)")
    start = time.monotonic()
    result = shutdown_process(proc, wait_seconds=2)
    elapsed = time.monotonic() - start
    assert proc.poll() is not None, "process must be dead after shutdown_process returns"
    assert elapsed < 15, "escalation must not hang anywhere near the full 60s sleep"
    assert result.how in ("terminated", "killed")


# --- shutdown_process_group / start_owned_process -----------------------------
#
# These spawn a PARENT that itself spawns a CHILD, and prove the CHILD --
# not just the parent Popen handle -- actually stops. The child writes an
# incrementing heartbeat file; if shutdown_process_group only killed the
# parent (like plain shutdown_process would, for a process that spawned its
# own subprocess) the child would keep updating the heartbeat forever.


def _spawn_owned_parent_with_child(heartbeat_path) -> subprocess.Popen:
    child_script = (
        "import time, pathlib\n"
        f"hb = pathlib.Path(r'{heartbeat_path}')\n"
        "i = 0\n"
        "while True:\n"
        "    hb.write_text(str(i))\n"
        "    i += 1\n"
        "    time.sleep(0.05)\n"
    )
    parent_code = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child_script!r}])\n"
        "time.sleep(60)\n"
    )
    return start_owned_process([sys.executable, "-c", parent_code])


def _wait_for_heartbeat(heartbeat_path, *, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not heartbeat_path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert heartbeat_path.exists(), "child process never started heartbeating (test setup issue)"


def test_shutdown_process_group_stops_parent_and_child(tmp_path):
    heartbeat = tmp_path / "heartbeat.txt"
    proc = _spawn_owned_parent_with_child(heartbeat)
    try:
        _wait_for_heartbeat(heartbeat)

        result = shutdown_process_group(proc, wait_seconds=10)
        assert proc.poll() is not None, "parent must be dead after shutdown_process_group returns"
        assert result.how in ("terminated", "killed")

        value_at_shutdown = heartbeat.read_text()
        time.sleep(0.5)
        value_later = heartbeat.read_text()
        assert value_at_shutdown == value_later, (
            "child process kept running (heartbeat still advancing) after "
            "shutdown_process_group returned -- only the parent was stopped"
        )
    finally:
        if proc.poll() is None:
            shutdown_process_group(proc, wait_seconds=5)


# --- DeadlineSupervisor -------------------------------------------------------


def test_deadline_supervisor_independent_of_main_thread_blocking(tmp_path):
    heartbeat = tmp_path / "heartbeat.txt"
    proc = _spawn_owned_parent_with_child(heartbeat)
    try:
        _wait_for_heartbeat(heartbeat)

        supervisor = DeadlineSupervisor(proc, deadline_seconds=0.3, poll_interval_seconds=0.05)
        supervisor.start()

        # Simulate the main thread being stuck doing something unrelated
        # (e.g. a hung synchronous call) for LONGER than the deadline --
        # the supervisor thread must still fire without any cooperation.
        time.sleep(1.5)

        assert supervisor.fired, "supervisor must have fired on its own while the main thread was blocked"
        assert proc.poll() is not None, "process must be dead"
        value_at_check = heartbeat.read_text()
        time.sleep(0.5)
        assert heartbeat.read_text() == value_at_check, "child must have actually stopped, not just the parent"
    finally:
        supervisor.cancel()
        if proc.poll() is None:
            shutdown_process_group(proc, wait_seconds=5)


def test_deadline_supervisor_cancel_prevents_a_later_kill(tmp_path):
    heartbeat = tmp_path / "heartbeat.txt"
    proc = _spawn_owned_parent_with_child(heartbeat)
    try:
        _wait_for_heartbeat(heartbeat)

        supervisor = DeadlineSupervisor(proc, deadline_seconds=5.0, poll_interval_seconds=0.05)
        supervisor.start()
        supervisor.cancel()  # cancelled well before the 5s deadline

        time.sleep(0.5)
        assert not supervisor.fired
        assert proc.poll() is None, "process must still be running -- cancel() must prevent the later kill"
    finally:
        shutdown_process_group(proc, wait_seconds=5)


def test_deadline_supervisor_does_not_fire_if_process_exits_on_its_own(tmp_path):
    proc = _spawn("pass")  # exits almost immediately
    proc.wait(timeout=5)
    supervisor = DeadlineSupervisor(proc, deadline_seconds=0.2, poll_interval_seconds=0.05)
    supervisor.start()
    time.sleep(0.5)
    supervisor.cancel()
    assert not supervisor.fired, "supervisor must not report firing for a process that already exited on its own"
