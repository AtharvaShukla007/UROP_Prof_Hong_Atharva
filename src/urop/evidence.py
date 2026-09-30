"""Per-request evidence capture for real coding-model calls.

A real Kaggle smoke run (2026-09-29, see docs/experiment_log.md) surfaced a
concrete gap: when a later chunk's JSON-repair request was blocked by the
context-length guard (`PromptTooLongError`), the EARLIER chunk's already-
received response for that same trajectory was never saved anywhere --
upstream's `OpenCodingAgent.code_record` only appends a chunk's result to
its local, in-progress `TrajectoryCoding` object, which is discarded
entirely when a later chunk raises. The raw model response existed for a
moment (it came back over the wire) and then was gone, with nothing to show
for it.

`EvidenceLog` fixes this by writing one record to disk at the RESPONSE
BOUNDARY -- immediately after each real HTTP attempt returns or fails,
inside `autotrace_adapter.install_budget_guard`'s wrapped `create` call --
independent of whether the trajectory-level call that triggered it ever
successfully returns. A later exception, anywhere upstream of that point,
cannot retroactively delete evidence already written to disk.

Evidence files are written under a caller-supplied directory (in this
project, always somewhere under the git-ignored `runs/` tree) and contain
no credentials: `RequestEvidence` never carries the client's API key or
auth headers, only the request body (model/messages/temperature/etc, which
is exactly what a wire sniffer would see going out) and the response body.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RequestEvidence:
    """Everything needed to reproduce or audit ONE real (or blocked) request.

    `request_seq` is unique and monotonically increasing within one
    `EvidenceLog` (i.e. within one coding run) -- evidence files sort
    chronologically by filename. `composite_id`/`chunk_seq`/`attempt_kind`
    attribute a request back to a specific trajectory/chunk/attempt even
    though upstream's own `create()` call carries no such identifiers --
    see `CodingContext.classify`.
    """

    request_seq: int
    composite_id: str | None
    chunk_seq: int | None
    attempt_kind: str  # "initial" | "repair"
    output_mode: str  # "upstream_codes_block" | "structured_json"
    model: str
    model_revision: str | None
    generation_settings_hash: str | None
    sent_at: str
    elapsed_seconds: float | None
    blocked_by_context_guard: bool
    request_payload: dict[str, Any]  # the exact kwargs sent (model/messages/temperature/... -- no credentials ever live here)
    finish_reason: str | None = None
    raw_content: str | None = None
    usage: dict[str, int] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceLog:
    """Writes one JSON file per `RequestEvidence`, immediately, to
    `out_dir` (created if missing). Never batches -- a crash right after
    `record()` returns still leaves that evidence file on disk.
    """

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._seq = 0

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def record(self, evidence: RequestEvidence) -> Path:
        kind_tag = "blocked" if evidence.blocked_by_context_guard else evidence.attempt_kind
        path = self.out_dir / f"evidence_{evidence.request_seq:04d}_{kind_tag}.json"
        path.write_text(json.dumps(evidence.to_dict(), indent=2, default=str), encoding="utf-8")
        return path


@dataclass
class CodingContext:
    """Mutable, caller-updated attribution context for the evidence logger.

    Upstream's `OpenCodingAgent.code_chunk`/`code_record` pass no
    trajectory/chunk identifiers into the `create()` kwargs it builds --
    only `model`/`messages`/`temperature`/etc. To attribute an evidence
    record to a specific (trajectory, chunk, attempt) without editing
    upstream, the CALLER sets `composite_id` via `begin_record()` right
    before invoking `agent.code_record(...)` for that trajectory, and
    `classify()` (called from inside the installed guard) infers the rest
    from the shape of the messages list actually being sent:

    - A fresh chunk's initial attempt sends exactly
      `[system, user]` (2 messages) -- `classify()` treats any 2-message
      request as the start of a NEW chunk and increments `chunk_seq`.
    - A JSON-repair retry appends the assistant's raw reply plus a repair
      instruction, making the message list longer (4 messages) --
      `classify()` reports that as `"repair"` for the CURRENT chunk_seq,
      without incrementing it.

    This mirrors `agents.open_coding.OpenCodingAgent.code_chunk`'s actual
    message-list construction exactly (verified by reading that source),
    so the inference is accurate for the real upstream agent, not a guess.
    """

    composite_id: str | None = None
    _chunk_seq: int = field(default=-1, repr=False)

    def begin_record(self, composite_id: str) -> None:
        self.composite_id = composite_id
        self._chunk_seq = -1

    def classify(self, messages: list[dict]) -> tuple[int | None, str]:
        is_initial = len(messages) <= 2
        if is_initial:
            self._chunk_seq += 1
        kind = "initial" if is_initial else "repair"
        chunk_seq = self._chunk_seq if self._chunk_seq >= 0 else None
        return chunk_seq, kind


def make_evidence(
    *,
    request_seq: int,
    composite_id: str | None,
    chunk_seq: int | None,
    attempt_kind: str,
    output_mode: str,
    model: str,
    model_revision: str | None,
    generation_settings_hash: str | None,
    request_payload: dict[str, Any],
    elapsed_seconds: float | None = None,
    blocked_by_context_guard: bool = False,
    finish_reason: str | None = None,
    raw_content: str | None = None,
    usage: dict[str, int] | None = None,
    error: str | None = None,
) -> RequestEvidence:
    return RequestEvidence(
        request_seq=request_seq,
        composite_id=composite_id,
        chunk_seq=chunk_seq,
        attempt_kind=attempt_kind,
        output_mode=output_mode,
        model=model,
        model_revision=model_revision,
        generation_settings_hash=generation_settings_hash,
        sent_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        elapsed_seconds=elapsed_seconds,
        blocked_by_context_guard=blocked_by_context_guard,
        request_payload=request_payload,
        finish_reason=finish_reason,
        raw_content=raw_content,
        usage=usage,
        error=error,
    )
