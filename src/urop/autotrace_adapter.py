"""Bridge between our pipeline and the vendored AutoTraceGT upstream code.

Upstream: external/autotracegt_upstream/autotracegt (see external/PROVENANCE.md).
Not imported at module load time -- only when a caller actually needs the
upstream agents/pipeline -- so CPU tests that exercise only our own adapter
logic don't require the upstream package or its dependencies to be installed.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from urop.budgets import Budget, bounded_llm_call

_UPSTREAM_ROOT = Path(__file__).resolve().parents[2] / "external" / "autotracegt_upstream" / "autotracegt"

LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0")

LEAKAGE_KEYS = {"resolved", "status", "reward", "outcome", "future_events"}


class UpstreamNotVendoredError(RuntimeError):
    pass


def ensure_upstream_on_path() -> Path:
    """Add the vendored autotracegt package directory to sys.path, if present."""
    if not _UPSTREAM_ROOT.exists():
        raise UpstreamNotVendoredError(
            f"{_UPSTREAM_ROOT} not found. Extract "
            "reference_materials/Qual-Agent-Behavior-Analysis-main.zip into "
            "external/autotracegt_upstream first (see external/PROVENANCE.md)."
        )
    root_str = str(_UPSTREAM_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return _UPSTREAM_ROOT


def register_local_endpoint(
    *,
    name: str = "local",
    base_url: str,
    api_key_env: str = "LOCAL_ENDPOINT_API_KEY",
    dummy_key_value: str = "local-no-auth",
) -> None:
    """Register a local OpenAI-compatible endpoint (e.g. a Kaggle-hosted vLLM
    server) with the vendored autotracegt.llm.endpoints registry.

    Refuses to register the no-auth dummy key against anything that is not an
    explicit localhost/loopback base_url (handoff §2: "a local dummy key must
    never be accepted for a remote endpoint"). A genuine remote/paid provider
    must go through llm.endpoints.register_endpoint directly with a real
    api_key_env, never through this helper.
    """
    if not base_url:
        raise ValueError("base_url is required")
    host = base_url.split("://", 1)[-1].split("/", 1)[0].split(":")[0]
    if host not in LOCAL_HOSTS:
        raise ValueError(
            f"register_local_endpoint refuses base_url={base_url!r} (host {host!r} is not "
            f"one of {LOCAL_HOSTS}). This helper is for a local no-auth server only."
        )
    ensure_upstream_on_path()
    from llm.endpoints import register_endpoint  # type: ignore[import-not-found]

    os.environ.setdefault(api_key_env, dummy_key_value)
    register_endpoint(name, api_key_env=api_key_env, base_url=base_url)


def assemble_holdout_prompt(trajectory: list[dict], *, up_to_index: int) -> list[dict]:
    """Prefix of a trajectory usable for held-out annotation.

    Returns messages[:up_to_index] with any leakage keys stripped from each
    message dict. Upstream message dicts shouldn't carry these, but this is a
    defensive strip against a dataset-specific adapter smuggling them in as
    extra keys (handoff §5: "No reward or future observations in judge input
    for decision-point labels").
    """
    prefix = trajectory[:up_to_index]
    return [{k: v for k, v in msg.items() if k not in LEAKAGE_KEYS} for msg in prefix]


def find_leaked_substrings(rendered_prompt: str, *, forbidden_substrings: list[str]) -> list[str]:
    """Return which forbidden substrings (outcome words, reward values, ...)
    appear verbatim in a rendered prompt string. Empty = clean. The caller
    decides what counts as forbidden per config; this only checks membership,
    so a real check must be run against the *actual rendered prompt*, not
    assumed from the code path alone (handoff §3: "Verify actual rendered
    prompts; do not repeat the paper's leakage statement as if default code
    guarantees it").
    """
    return [s for s in forbidden_substrings if s and s in rendered_prompt]


@dataclass
class QuoteCheckResult:
    valid: bool
    reason: str | None = None


def verify_quote(quote: str, *, source_text: str) -> QuoteCheckResult:
    """An open-coding (code, span, quote) record's quote must be verbatim
    present in the source trajectory text it claims to come from. A
    non-verbatim or empty quote is invalid -> the caller must record the
    code as invalid/abstain, never silently accept it as a clean label
    (handoff §5 point 5, §8 acceptance check 5).
    """
    if not quote or not quote.strip():
        return QuoteCheckResult(valid=False, reason="empty quote")
    if quote not in source_text:
        return QuoteCheckResult(valid=False, reason="quote not found verbatim in source text")
    return QuoteCheckResult(valid=True)


@dataclass
class MockChatResult:
    content: str
    label: str = "mock"


class MockCoder:
    """A deterministic, offline stand-in for a real chat-completion call.

    Used only by CPU tests and the text_smoke config to exercise batching,
    budget, and resume logic without network access or cost. Every result
    is tagged label="mock" so reporting.summarize_run refuses to treat it
    as research output.

    Cooperatively honours `timeout`: simulated work is done in small
    increments, checked against the deadline, and a `TimeoutError` is
    raised (work genuinely stops) if it would be exceeded -- the same
    contract `bounded_llm_call` requires of a real OpenAI-SDK-style client,
    so a test against MockCoder actually exercises cancellation, not just
    "the caller gave up waiting while the mock kept sleeping regardless."
    """

    def __init__(self, *, delay_seconds: float = 0.0, response: str = '{"codes": []}', fail_first_n: int = 0):
        self.delay_seconds = delay_seconds
        self.response = response
        self.fail_first_n = fail_first_n
        self.calls = 0
        self.completed_calls = 0  # only calls that actually finished their simulated work

    def __call__(self, *, messages: list[dict], timeout: float | None = None, **kwargs: Any) -> MockChatResult:
        self.calls += 1
        start = time.monotonic()
        remaining = self.delay_seconds
        step = 0.01
        while remaining > 0:
            if timeout is not None and (time.monotonic() - start) >= timeout:
                raise TimeoutError(f"MockCoder: simulated work stopped at timeout={timeout}s")
            time.sleep(min(step, remaining))
            remaining -= step
        if timeout is not None and (time.monotonic() - start) >= timeout:
            raise TimeoutError(f"MockCoder: simulated work stopped at timeout={timeout}s")
        self.completed_calls += 1
        if self.calls <= self.fail_first_n:
            raise ValueError(f"MockCoder: simulated failure on call {self.calls}/{self.fail_first_n}")
        return MockChatResult(content=self.response)


BackendCallable = Callable[..., Any]


class PromptTooLongError(RuntimeError):
    def __init__(self, n_tokens: int, budget_tokens: int):
        super().__init__(
            f"rendered prompt is {n_tokens} tokens, exceeds the {budget_tokens}-token budget -- "
            "refusing to send. Never silently truncated."
        )
        self.n_tokens = n_tokens
        self.budget_tokens = budget_tokens
        # Set by code_trajectory_preserving_partial_results (never by the
        # bare token-check raise inside install_budget_guard, which has no
        # per-record context) to whatever earlier chunks of the SAME
        # trajectory already completed before this block -- never silently
        # dropped just because a later chunk got blocked.
        self.partial_result: Any = None


def install_budget_guard(
    agent: Any,
    budget: Budget,
    *,
    tokenizer: Any = None,
    context_limit: int | None = None,
    reserved_output_tokens: int = 0,
    max_sdk_retries: int = 0,
    evidence_log: Any = None,
    coding_context: Any = None,
    output_mode: str = "upstream_codes_block",
    model_revision: str | None = None,
    generation_settings_hash: str | None = None,
) -> None:
    """Monkeypatch agent.client.chat.completions.create so every actual
    outbound request OpenCodingAgent makes -- the first attempt, every
    JSON-repair retry, and every chunk -- goes through one choke point:

    1. If `tokenizer`/`context_limit` are given: tokenize the EXACT
       `messages` about to be sent via `tokenizer.apply_chat_template`
       (the real, server-compatible template -- not a reimplementation),
       and raise `PromptTooLongError` rather than send an oversized
       request. This runs for every real request, including a JSON-repair
       round's grown message list and a later chunk's prompt (which
       depends on a generated memo we couldn't know ahead of time) --
       not just an ahead-of-time check on the first chunk.
    2. `bounded_llm_call`: counted as a real attempt (not just "one per
       completed trajectory"), given a per-call timeout bounded by the
       remaining budget, converted to BudgetExhausted once the overall
       deadline is gone.
    3. If `evidence_log` (an `urop.evidence.EvidenceLog`) is given: writes
       one evidence record to disk at the RESPONSE BOUNDARY -- right after
       a real attempt returns, fails, or is blocked by the token check --
       before returning control to whatever upstream code will try to
       parse it or build a repair prompt from it. This is what makes a
       response survive a LATER chunk's request being blocked or erroring:
       each record is written immediately, not accumulated and saved only
       if the whole trajectory-level call eventually succeeds. A request
       blocked by the token check is recorded with
       `blocked_by_context_guard=True` and is NOT counted as a real
       attempt (the block happens before `bounded_llm_call` /
       `budget.record_call()` ever runs). `coding_context`, if given (an
       `urop.evidence.CodingContext`), attributes each record to a
       trajectory/chunk/attempt -- see its docstring for how.

    Also caps the openai SDK's own internal retry count
    (`max_sdk_retries`, default 0) so a transient network error can't
    silently multiply attempts underneath bounded_llm_call's counting --
    "bound SDK and transport retries so they do not multiply unnoticed."

    Does not edit upstream source; wraps the already-constructed client
    object and method in place.
    """
    if hasattr(agent.client, "with_options"):
        agent.client = agent.client.with_options(max_retries=max_sdk_retries)

    original_create = agent.client.chat.completions.create

    def _log(
        *, chunk_seq, attempt_kind, request_payload, elapsed_seconds=None,
        blocked=False, finish_reason=None, raw_content=None, usage=None, error=None,
    ) -> None:
        if evidence_log is None:
            return
        from urop.evidence import make_evidence

        evidence = make_evidence(
            request_seq=evidence_log.next_seq(),
            composite_id=coding_context.composite_id if coding_context is not None else None,
            chunk_seq=chunk_seq,
            attempt_kind=attempt_kind,
            output_mode=output_mode,
            model=request_payload.get("model", ""),
            model_revision=model_revision,
            generation_settings_hash=generation_settings_hash,
            request_payload=request_payload,
            elapsed_seconds=elapsed_seconds,
            blocked_by_context_guard=blocked,
            finish_reason=finish_reason,
            raw_content=raw_content,
            usage=usage,
            error=error,
        )
        evidence_log.record(evidence)

    def guarded_create(**kwargs: Any) -> Any:
        messages = kwargs.get("messages", [])
        chunk_seq, attempt_kind = (
            coding_context.classify(messages) if coding_context is not None else (None, "initial")
        )

        if tokenizer is not None and context_limit is not None:
            token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            n_tokens = len(token_ids)
            budget_tokens = context_limit - reserved_output_tokens
            if n_tokens > budget_tokens:
                _log(
                    chunk_seq=chunk_seq, attempt_kind=attempt_kind, request_payload=kwargs,
                    blocked=True, error=f"PromptTooLongError: {n_tokens} tokens > {budget_tokens} budget",
                )
                raise PromptTooLongError(n_tokens, budget_tokens)

        start = time.monotonic()
        try:
            response = bounded_llm_call(original_create, budget=budget, **kwargs)
        except Exception as exc:
            _log(
                chunk_seq=chunk_seq, attempt_kind=attempt_kind, request_payload=kwargs,
                elapsed_seconds=time.monotonic() - start, error=repr(exc),
            )
            raise

        elapsed = time.monotonic() - start
        choice = response.choices[0] if getattr(response, "choices", None) else None
        usage_obj = getattr(response, "usage", None)
        usage = (
            {
                "prompt_tokens": usage_obj.prompt_tokens,
                "completion_tokens": usage_obj.completion_tokens,
                "total_tokens": usage_obj.total_tokens,
            }
            if usage_obj is not None
            else None
        )
        _log(
            chunk_seq=chunk_seq, attempt_kind=attempt_kind, request_payload=kwargs,
            elapsed_seconds=elapsed,
            finish_reason=getattr(choice, "finish_reason", None) if choice is not None else None,
            raw_content=getattr(choice.message, "content", None) if choice is not None else None,
            usage=usage,
        )
        return response

    agent.client.chat.completions.create = guarded_create


def code_trajectory_preserving_partial_results(agent: Any, record: dict) -> Any:
    """Like `OpenCodingAgent.code_record`, but calls `agent.code_chunk()`
    directly, chunk by chunk, so a `PromptTooLongError` raised while
    building a LATER chunk's repair prompt does not discard chunk codings
    already completed for EARLIER chunks in the same trajectory.

    Real bug this fixes (2026-09-29 Kaggle run, see
    docs/experiment_log.md): upstream's own `code_record` builds one local
    `TrajectoryCoding` and only returns it at the very end of the loop --
    if `code_chunk` raises on chunk 2, whatever `chunk_codings` were
    already appended for chunk 0/1 are thrown away with the exception, even
    though the model's chunk-0/1 responses were real and (via
    `install_budget_guard`'s evidence log) already saved to disk
    independently. This wrapper attaches whatever WAS completed to the
    raised exception as `exc.partial_result`, so the caller can still save/
    report it instead of losing it a second time at this layer too.

    Does not edit upstream source -- calls its public `code_chunk` method
    in the same sequence `code_record` itself uses (verified by reading
    that source: chunk via `chunk_trajectory`, threading `segment_memo`
    forward as `preceding_memo`, advancing `step_offset` by chunk length).
    """
    ensure_upstream_on_path()
    from agents.open_coding import TrajectoryCoding
    from utils.trajectory_processing import chunk_trajectory

    task = record["context"]["task_description"]
    tools = json.dumps([t["function"]["name"] for t in record["context"]["tools"]], indent=2)
    status = record["status"]
    trajectory = record["trajectory"]
    chunks = chunk_trajectory(trajectory, agent.chunk_size)

    result = TrajectoryCoding(instance_id=record["instance_id"], run_id=record["run_id"], status=status)
    preceding_memo = ""
    step_offset = 1
    for chunk in chunks:
        try:
            cc = agent.code_chunk(
                chunk, step_offset=step_offset, task=task, tools=tools, status=status,
                preceding_memo=preceding_memo,
            )
        except PromptTooLongError as exc:
            exc.partial_result = result  # whatever chunks completed before the block, preserved -- never silently dropped
            raise
        result.chunk_codings.append(cc)
        preceding_memo = cc.segment_memo
        step_offset += len(chunk)
    return result


def hash_prompt_templates(*names: str) -> str:
    """SHA-256 of the concatenated real prompt template file contents
    (e.g. "open_coding_system", "open_coding_user") under the vendored
    upstream prompts/ directory -- for cache_key's prompt_hash, so a cache
    entry invalidates automatically if the actual template text changes,
    rather than relying on a hand-maintained version string.
    """
    import hashlib

    root = ensure_upstream_on_path()
    prompts_dir = root / "prompts"
    parts = []
    for name in sorted(names):
        path = prompts_dir / f"{name}.md"
        parts.append(path.read_text(encoding="utf-8"))
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def render_open_coding_chunk_prompt(
    record: dict, *, chunk_index: int, preceding_memo: str, chunk_size: int
) -> tuple[str, str, int]:
    """Render the EXACT system+user prompt OpenCodingAgent.code_chunk would
    send for `record`'s chunk `chunk_index`, using the real upstream
    load_prompt/chunk_trajectory/format_chunk functions -- not a
    reimplementation that could silently drift from what's actually sent.

    `record` is the plain-dict shape OpenCodingAgent.code_record expects
    (context/trajectory/status). Requires ensure_upstream_on_path() to have
    already been called. Returns (system_prompt, user_prompt, n_chunks).
    """
    ensure_upstream_on_path()
    from agents.open_coding import format_chunk
    from utils.trajectory_processing import chunk_trajectory
    from utils.utils import load_prompt

    chunks = chunk_trajectory(record["trajectory"], chunk_size)
    step_offset = sum(len(c) for c in chunks[:chunk_index]) + 1
    chunk_text = format_chunk(chunks[chunk_index], step_offset=step_offset)
    tools_json = json.dumps([t["function"]["name"] for t in record["context"]["tools"]], indent=2)
    system_prompt = load_prompt("open_coding_system")
    user_prompt = load_prompt(
        "open_coding_user",
        task=record["context"]["task_description"],
        tools=tools_json,
        status=record["status"],
        preceding_memo=preceding_memo,
        trajectory=chunk_text,
    )
    return system_prompt, user_prompt, len(chunks)


def count_tokens_via_chat_template(
    tokenizer: Any, *, system_prompt: str, user_prompt: str, extra_messages: list[dict] | None = None
) -> int:
    """Token count via tokenizer.apply_chat_template (the real, server-
    compatible template) over [system, user, *extra_messages] -- not a
    naive per-string token sum, which would miss the role markers/special
    tokens the chat template itself adds.
    """
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    if extra_messages:
        messages.extend(extra_messages)
    token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return len(token_ids)


_EMPTY_CODES_BLOCK_RE = re.compile(r"===CODES===\s*\[\s*\]", re.DOTALL)


def classify_chunk_coding(cc: dict) -> str:
    """Classify one `ChunkCoding.to_dict()`-shaped dict as one of:

    - "coded": >=1 parsed code -- real output.
    - "empty_clean": the response was a legitimate, well-formed "nothing
      worth coding in this chunk" result -- either the upstream freeform
      mode's raw response contained an explicit, clean empty CODES list
      (`===CODES===` followed by `[]`), or (structured JSON mode) `raw` is
      itself the parsed response object with `"codes": []`.
    - "parse_failed": `codes == []` and neither of the above holds.
      `codes == []` ALONE never determines this either way -- it is
      produced just as often by a genuinely empty, valid coding as by
      unparseable output upstream's own JSON-repair retries already gave up
      on (handoff §5 point 5: "do not turn parse failures into clean/
      negative labels silently", and this project's own 2026-09-30 fix:
      "do not classify solely from whether the codes list is empty").

    Structured mode's `raw` is checked FIRST and directly (it is exactly
    the model's JSON object, already schema-validated by
    `structured_coding.parse_structured_coding_response` before a
    `ChunkCoding` is ever constructed for it -- so a "codes" key found here
    is authoritative, not a guess). Only if that check doesn't apply does
    this fall back to the freeform mode's raw-text marker heuristic.
    """
    if cc.get("codes"):
        return "coded"
    raw = cc.get("raw", "") or ""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        parsed = None
    if isinstance(parsed, dict) and "codes" in parsed:
        return "empty_clean" if parsed["codes"] == [] else "coded"
    if _EMPTY_CODES_BLOCK_RE.search(raw):
        return "empty_clean"
    return "parse_failed"


def classify_trajectory_coding(coding_dict: dict) -> str:
    """A `TrajectoryCoding.to_dict()`-shaped dict is "valid" only if every
    one of its chunks is "coded" or "empty_clean". A single "parse_failed"
    chunk -- or no chunks at all -- makes the whole result invalid: do not
    cache it via `ResumeCache.mark_done` or report it as a clean success;
    route it through `ResumeCache.mark_invalid` instead, with the raw output
    still saved to disk for inspection.
    """
    chunk_codings = coding_dict.get("chunk_codings", [])
    if not chunk_codings:
        return "parse_failed"
    classes = [classify_chunk_coding(cc) for cc in chunk_codings]
    if "parse_failed" in classes:
        return "parse_failed"
    return "valid"


@dataclass
class CodeValidationReport:
    total_codes: int = 0
    valid_codes: int = 0
    invalid_codes: list[dict] = field(default_factory=list)


_STEP_RANGE_RE = re.compile(r"\s*(\d+)\s*(?:-\s*(\d+))?\s*")


def validate_coding_quotes(coding_dict: dict, *, trajectory: list[dict], chunk_size: int) -> CodeValidationReport:
    """Verify every open-coding record's quote is verbatim within the
    SPECIFIC step range it cites (`code["steps"]`), not just present
    somewhere in the whole chunk. Invalid/empty quotes -- and now invalid
    or out-of-chunk step references -- are reported separately
    (`invalid_codes`), never silently dropped or counted as valid; the raw
    parsed code dict is preserved in the report for inspection, not
    discarded.

    A quote that IS verbatim in the chunk but from a DIFFERENT step than
    the one cited would have passed the old whole-chunk check -- this is
    the "cited-step quote validation" fix (2026-09-30, see
    docs/experiment_log.md): restricting the source text to just the cited
    step(s), reconstructed via the real upstream `format_chunk`, catches
    that case, at the cost of also being strict about incidental
    whitespace differences between how the model rendered a quote and how
    `format_chunk` renders the same step (e.g. a space where `format_chunk`
    puts a newline) -- a real, observed pattern in the 2026-09-29/30 Kaggle
    runs. That strictness is intentional: do not relax it to a whitespace-
    normalized comparison to make old "invalid" results look valid after
    the fact (see docs/experiment_log.md's note on this exact trade-off).

    `coding_dict` is a TrajectoryCoding.to_dict()-shaped dict. Requires
    ensure_upstream_on_path() to have already been called.
    """
    ensure_upstream_on_path()
    from agents.open_coding import format_chunk
    from utils.trajectory_processing import chunk_trajectory

    chunks = chunk_trajectory(trajectory, chunk_size)
    report = CodeValidationReport()
    for cc in coding_dict.get("chunk_codings", []):
        idx = cc["chunk_index"]
        for code in cc.get("codes", []):
            report.total_codes += 1

            step_text = str(code.get("steps", ""))
            numbers: list[int] = []
            step_error = None
            for part in step_text.split(","):
                match = _STEP_RANGE_RE.fullmatch(part)
                if not match:
                    step_error = "invalid step format (use 4 or 4-7 or comma-separated ranges)"
                    break
                first, last = int(match[1]), int(match[2] or match[1])
                offset = idx * chunk_size + 1
                end = offset + (len(chunks[idx]) if 0 <= idx < len(chunks) else 0) - 1
                if first > last or first < offset or last > end:
                    step_error = "step reference outside this chunk"
                    break
                numbers.extend(range(first, last + 1))

            if step_error or not numbers:
                report.invalid_codes.append(
                    {"chunk_index": idx, "code": code.get("code"), "quote": code.get("quote"),
                     "reason": step_error or "missing steps"}
                )
                continue

            offset = idx * chunk_size + 1
            selected_text = "\n".join(
                format_chunk([chunks[idx][n - offset]], step_offset=n) for n in sorted(set(numbers))
            )
            result = verify_quote(code.get("quote", ""), source_text=selected_text)
            if result.valid:
                report.valid_codes += 1
            else:
                report.invalid_codes.append(
                    {"chunk_index": idx, "code": code.get("code"), "quote": code.get("quote"), "reason": result.reason}
                )
    return report
