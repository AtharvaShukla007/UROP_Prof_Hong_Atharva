"""Budgets, enforceable per-call cancellation, and a resume cache.

An earlier version of this module wrapped calls in a `ThreadPoolExecutor`
and abandoned the worker thread on timeout. That is not real cancellation:
the caller returns early, but the network call the thread was blocked on
keeps running to completion in the background, on a server we've already
decided to stop paying for -- a review caught this. Python genuinely cannot
force-kill a thread; the only real way to bound a blocking network call is
to make the call itself bounded, i.e. pass a `timeout` into the request so
the HTTP/SDK layer aborts it at the deadline (this is a standard, real
capability of `httpx`/`openai`-style clients, not something faked here).
`bounded_llm_call` does that: it calls `fn(timeout=..., **kwargs)` directly,
synchronously, in the caller's own thread -- no thread pool, nothing left
running in the background once it returns or raises.

The vendored upstream pipeline's own retry policy
(external/autotracegt_upstream's llm.endpoints.chat_with_retry) retries up
to 15 times with exponential backoff -- enough to consume a whole session
unattended, and separate from (stacked on top of) the openai SDK's own
default internal retries. `src/urop/autotrace_adapter.install_budget_guard`
routes every actual outbound request through `bounded_llm_call`, so retries
are bounded and counted as real attempts, not just topline "completed
trajectory" counts.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class Budget:
    max_seconds: float
    max_calls: int | None = None
    started_at: float = field(default_factory=time.monotonic)
    calls_made: int = 0
    """Count of real attempts (bounded_llm_call invocations), NOT completed
    trajectories or successful calls -- a retry/repair attempt or a failed
    call still counts. See bounded_llm_call."""
    stop_reason: str | None = None

    def remaining_seconds(self) -> float:
        return max(0.0, self.max_seconds - (time.monotonic() - self.started_at))

    def exhausted(self) -> bool:
        if self.remaining_seconds() <= 0:
            return True
        if self.max_calls is not None and self.calls_made >= self.max_calls:
            return True
        return False

    def record_call(self) -> None:
        self.calls_made += 1


class BudgetExhausted(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def bounded_llm_call(
    fn: Callable[..., Any], *, budget: Budget, timeout_override: float | None = None, **kwargs: Any
) -> Any:
    """Call fn(timeout=<bounded>, **kwargs) synchronously -- no thread, nothing
    abandoned. `fn` must itself honour its `timeout` kwarg by aborting the
    underlying request at that deadline (true of an OpenAI-SDK-style client's
    `.create(timeout=...)`, and of `MockCoder`/`FakeLocalChatServer` in this
    project's tests, which poll their own simulated deadline cooperatively).

    Raises BudgetExhausted immediately, without attempting a call, once the
    remaining budget is already zero. Counts this attempt (`budget.record_call()`)
    before the outcome is known, so a failed or timed-out attempt still counts
    -- matching "count actual inference attempts, including failures and
    retries, rather than completed trajectories."

    A failure converts to `BudgetExhausted` only when BOTH: (a) this call's
    timeout was bound by the overall remaining budget, not a tighter
    `timeout_override`, and (b) the call actually ran for approximately that
    long before failing -- i.e. it was plausibly the timeout that caused the
    failure, not something else (a malformed-response error, an auth
    failure, ...) that happened to occur on a call whose timeout parameter
    was budget-bound. (a) alone is not enough: an immediate `ValueError`
    from a call given a 60s budget-bound timeout is not a budget problem.
    Checking the call's own elapsed time (rather than re-reading the
    budget's absolute clock after the fact) also avoids a race against
    exactly when the underlying HTTP client's own timeout fires, which was
    observed to occasionally misclassify a genuine budget-caused failure
    under load.
    """
    if budget.exhausted():
        budget.stop_reason = budget.stop_reason or "budget_exhausted_before_call"
        raise BudgetExhausted(budget.stop_reason)

    remaining = budget.remaining_seconds()
    budget_bound = timeout_override is None or timeout_override >= remaining
    timeout = remaining if timeout_override is None else min(remaining, timeout_override)

    budget.record_call()
    call_start = time.monotonic()
    try:
        return fn(timeout=timeout, **kwargs)
    except Exception:
        call_elapsed = time.monotonic() - call_start
        ran_until_timeout = timeout <= 0 or call_elapsed >= timeout * 0.9
        if budget_bound and ran_until_timeout:
            budget.stop_reason = budget.stop_reason or "budget_exhausted_mid_call"
            raise BudgetExhausted(budget.stop_reason) from None
        raise


def cache_key(
    *, model: str, model_revision: str, prompt_hash: str, settings_hash: str, data_hash: str, stage: str
) -> str:
    """`model_revision` is a required, separate field (not folded into
    `model`) so every caller has to think about it explicitly -- an
    unpinned/floating model reference (e.g. "main") can silently resolve to
    different weights between one run and the next, which must invalidate
    the cache the same way a changed prompt or settings hash would.
    """
    import hashlib

    payload = json.dumps(
        {
            "model": model,
            "model_revision": model_revision,
            "prompt_hash": prompt_hash,
            "settings_hash": settings_hash,
            "data_hash": data_hash,
            "stage": stage,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class ResumeCache:
    """A tiny on-disk resume index: {cache_key: {completed, attempts,
    result_path, reason}}.

    Skips work only when the cache_key is marked completed AND its
    result_path still exists on disk -- an index entry alone is not enough
    (handoff §6: "Skip successful work, not merely existing files"). A
    changed model/revision/prompt/settings/data hash produces a different
    cache_key, so it naturally misses the cache rather than reusing stale
    results.

    A result must be validated by the CALLER before calling `mark_done` --
    this class has no opinion on what "valid" means (see
    `autotrace_adapter.classify_trajectory_coding`). An invalid/empty raw
    result is preserved via `mark_invalid` (never silently dropped, never
    marked done), with a bounded, explicit attempt counter
    (`record_attempt`/`is_exhausted`) so a call site can retry a SMALL,
    FIXED number of times and then stop -- not rerun a persistently failing
    record indefinitely on every future resume.
    """

    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path)
        self._index: dict[str, dict] = {}
        if self.index_path.exists():
            self._index = json.loads(self.index_path.read_text(encoding="utf-8"))

    def is_done(self, key: str) -> bool:
        entry = self._index.get(key)
        if not entry or not entry.get("completed"):
            return False
        return Path(entry["result_path"]).exists()

    def result_path_for(self, key: str) -> Path | None:
        """The completed result's path, but only when `is_done(key)` is
        true -- lets a resumed run actually LOAD a previously valid result
        (so it's included in this run's summary/report), not just skip past
        it silently.
        """
        if not self.is_done(key):
            return None
        return Path(self._index[key]["result_path"])

    def mark_done(self, key: str, *, result_path: str) -> None:
        self._index[key] = {"completed": True, "result_path": result_path}
        self._flush()

    def record_attempt(self, key: str) -> int:
        """Record one more attempt at `key` (called once per real try, valid
        or not) and return the new total attempt count. Preserves any prior
        attempts already recorded (e.g. from an earlier `mark_invalid`)."""
        entry = self._index.setdefault(key, {"completed": False, "attempts": 0})
        entry["attempts"] = entry.get("attempts", 0) + 1
        self._flush()
        return entry["attempts"]

    def is_exhausted(self, key: str, *, max_attempts: int) -> bool:
        """True once `key` has been attempted `max_attempts` times without
        ever completing -- the caller should stop retrying and report it as
        permanently invalid, not attempt it again on this or a future
        resume."""
        entry = self._index.get(key)
        if not entry or entry.get("completed"):
            return False
        return entry.get("attempts", 0) >= max_attempts

    def mark_invalid(self, key: str, *, reason: str, result_path: str | None = None) -> None:
        """Record an invalid/empty raw result. Never marks `completed`, so
        `is_done` keeps returning False -- this is preserved as a distinct,
        explicit status, not treated as either a success or a silent drop.
        `result_path`, if given, points at the raw output actually saved to
        disk (kept, never overwritten) for later inspection.
        """
        entry = self._index.setdefault(key, {"completed": False, "attempts": 0})
        entry["completed"] = False
        entry["reason"] = reason
        if result_path is not None:
            entry["result_path"] = result_path
        self._flush()

    def _flush(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(self._index, indent=2, sort_keys=True), encoding="utf-8")
