"""Owned-subprocess lifecycle: startup readiness (fails fast if the process
dies during startup, instead of polling a health check until a timeout that
will never succeed), terminate/wait/kill shutdown escalation, process-GROUP
shutdown (a served model can spawn worker subprocesses; killing only the
direct Popen handle can leave those running), and an independent wall-clock
deadline supervisor.

Split out from budgets.py because it is about controlling a real OS
process (can be forcibly killed), not a Python-level call (which cannot).
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable


class ServerExitedDuringStartup(RuntimeError):
    def __init__(self, returncode: int | None):
        super().__init__(f"server process exited during startup (returncode={returncode})")
        self.returncode = returncode


class ServerNotReady(TimeoutError):
    pass


def wait_for_server_ready(
    proc: subprocess.Popen,
    health_check: Callable[[], bool],
    *,
    timeout_seconds: float,
    poll_interval_seconds: float = 2.0,
) -> float:
    """Poll `health_check()` until it returns True, `timeout_seconds` elapses,
    or `proc` exits first.

    Checking `proc.poll()` on every iteration means a process that crashes
    during startup is reported immediately (`ServerExitedDuringStartup`),
    instead of silently polling a health endpoint that will never come up
    until the full timeout is burned uselessly.

    Returns elapsed seconds on success.
    """
    start = time.monotonic()
    deadline = start + timeout_seconds
    while time.monotonic() < deadline:
        returncode = proc.poll()
        if returncode is not None:
            raise ServerExitedDuringStartup(returncode)
        if health_check():
            return time.monotonic() - start
        time.sleep(poll_interval_seconds)
    raise ServerNotReady(f"server not ready within {timeout_seconds}s")


@dataclass
class ShutdownResult:
    how: str  # "already_exited" | "terminated" | "killed" | "kill_failed"
    returncode: int | None


def shutdown_process(proc: subprocess.Popen, *, wait_seconds: float = 30.0) -> ShutdownResult:
    """terminate() -> wait() -> escalate to kill() -> wait() if still alive.

    Always returns (never raises) so this is safe to call from a `finally`
    block even if the process is already dead or misbehaving.
    """
    if proc.poll() is not None:
        return ShutdownResult(how="already_exited", returncode=proc.returncode)

    proc.terminate()
    try:
        proc.wait(timeout=wait_seconds)
        return ShutdownResult(how="terminated", returncode=proc.returncode)
    except subprocess.TimeoutExpired:
        pass

    proc.kill()
    try:
        proc.wait(timeout=wait_seconds)
        return ShutdownResult(how="killed", returncode=proc.returncode)
    except subprocess.TimeoutExpired:
        return ShutdownResult(how="kill_failed", returncode=None)


def start_owned_process(cmd: list[str], **popen_kwargs) -> subprocess.Popen:
    """Launch `cmd` as the root of its own process group (POSIX) or process
    group (Windows `CREATE_NEW_PROCESS_GROUP`), so `shutdown_process_group`
    can later stop it AND every child it spawns -- not just the one PID this
    Popen object tracks. A served model (e.g. vLLM) commonly forks worker
    subprocesses; terminating only the direct handle can leave those running
    and still holding the GPU.
    """
    if sys.platform == "win32":
        popen_kwargs.setdefault("creationflags", subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        popen_kwargs.setdefault("start_new_session", True)  # setsid -- own process group
    return subprocess.Popen(cmd, **popen_kwargs)


def shutdown_process_group(proc: subprocess.Popen, *, wait_seconds: float = 30.0) -> ShutdownResult:
    """Like `shutdown_process`, but stops the WHOLE process group/tree
    started via `start_owned_process` -- terminate escalating to kill, of
    every process in the group, not just `proc` itself.

    POSIX: signals the process group (`os.killpg`) via SIGTERM then SIGKILL
    -- a real graceful-then-forced escalation.
    Windows: `taskkill /F /T` (forced tree kill) directly, with no separate
    graceful stage. A non-forced `taskkill /T` sends WM_CLOSE, which only a
    process with a GUI message loop ever responds to -- a plain console
    subprocess (e.g. a served model's Python process) silently ignores it
    and keeps running, so a graceful-first attempt here would not be a real
    escalation step, just a wasted wait. `Popen.terminate()`/`.kill()` alone
    only reach the one tracked PID, not children it spawned, which is the
    reason this function exists instead of plain `shutdown_process`.

    Always returns (never raises); safe to call from a `finally` block even
    if the process/group is already gone.
    """
    if proc.poll() is not None:
        return ShutdownResult(how="already_exited", returncode=proc.returncode)

    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        try:
            proc.wait(timeout=wait_seconds)
            return ShutdownResult(how="killed", returncode=proc.returncode)
        except subprocess.TimeoutExpired:
            return ShutdownResult(how="kill_failed", returncode=None)

    import signal

    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return ShutdownResult(how="already_exited", returncode=proc.returncode)

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return ShutdownResult(how="already_exited", returncode=proc.returncode)
    try:
        proc.wait(timeout=wait_seconds)
        return ShutdownResult(how="terminated", returncode=proc.returncode)
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=wait_seconds)
        return ShutdownResult(how="killed", returncode=proc.returncode)
    except subprocess.TimeoutExpired:
        return ShutdownResult(how="kill_failed", returncode=None)


class DeadlineSupervisor:
    """An independent wall-clock watchdog over one owned process group.

    Runs in its own daemon thread, polling only `time.monotonic()` and
    `proc.poll()` -- nothing about it depends on, or can be blocked by, an
    HTTP client's own `timeout=`, the main thread being stuck in a
    synchronous call, or any cooperation from the code making requests.
    Once `deadline_seconds` elapses with the process still alive, this
    thread stops the WHOLE process group itself, on its own, via
    `shutdown_process_group` -- it does not wait for the main thread to
    notice or call anything.

    This exists because `bounded_llm_call`'s `timeout=` only bounds ONE
    request at a time; nothing previously enforced a hard ceiling on total
    wall-clock runtime if the main thread ever got stuck outside of an HTTP
    call entirely (e.g. blocked loading a tokenizer, or waiting on a hung
    subprocess pipe read) -- an HTTP timeout guarantee is not a total-
    runtime guarantee.
    """

    def __init__(self, proc: subprocess.Popen, deadline_seconds: float, *, poll_interval_seconds: float = 1.0):
        self._proc = proc
        self._deadline = time.monotonic() + deadline_seconds
        self._poll_interval = poll_interval_seconds
        self._stop_event = threading.Event()
        self._fired = threading.Event()
        self.result: ShutdownResult | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="deadline-supervisor")

    def start(self) -> "DeadlineSupervisor":
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if self._proc.poll() is not None:
                return  # process already exited on its own -- nothing left to supervise
            if time.monotonic() >= self._deadline:
                self._fired.set()
                self.result = shutdown_process_group(self._proc)
                return
            self._stop_event.wait(self._poll_interval)

    def cancel(self) -> None:
        """Stop supervising -- call once normal cleanup has already run so
        this thread doesn't fire a redundant kill afterwards. Blocks briefly
        until the watchdog thread has actually exited.
        """
        self._stop_event.set()
        self._thread.join(timeout=5)

    @property
    def fired(self) -> bool:
        """True once the deadline actually elapsed and this supervisor, not
        the caller, force-stopped the process group."""
        return self._fired.is_set()
