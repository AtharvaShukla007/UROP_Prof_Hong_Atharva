"""CPU-only regression tests built from the REAL 2026-09-29 Kaggle smoke
run's observed failure patterns (see docs/experiment_log.md and the
attached urop_smoke_debug.zip, kept private/git-ignored -- never committed).

These tests exercise `install_budget_guard`'s evidence logging and
`urop.reporting.build_outcome_report` against a minimal FAKE OpenAI-shaped
client (no real `openai` package import, no network, no GPU) -- see
`tests/test_open_coding_integration.py` for the two scenarios that
genuinely need the real vendored `agents.open_coding` module (upstream's
own JSON-repair loop behavior on unescaped quotes / incomplete JSON).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from urop.autotrace_adapter import PromptTooLongError, install_budget_guard, verify_quote
from urop.budgets import Budget
from urop.evidence import CodingContext, EvidenceLog
from urop.reporting import build_outcome_report
from urop.structured_coding import StructuredOutputError, parse_structured_coding_response


# --- a minimal fake OpenAI-shaped client, no `openai` package needed --------


@dataclass
class _FakeUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage
    finish_reason: str


@dataclass
class _FakeResponse:
    choices: list
    usage: _FakeUsage | None = None


class _ScriptedCreate:
    """Callable stand-in for `client.chat.completions.create` that returns
    (or raises) the next scripted outcome each call, mirroring the real
    signature (`**kwargs`, honours `timeout=` cooperatively -- not needed
    here since these tests don't exercise real timeouts)."""

    def __init__(self, outcomes: list):
        self._outcomes = list(outcomes)
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _fake_agent(create_fn) -> SimpleNamespace:
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_fn)))
    return SimpleNamespace(client=client)


def _load_evidence_files(out_dir) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("evidence_*.json"))]


# --- finish_reason and usage are recorded -------------------------------------


def test_finish_reason_length_is_recorded_appropriately(tmp_path):
    """A response cut off by the server's own max-token limit reports
    finish_reason="length" -- exactly the confirmed-vs-suspected distinction
    the real Kaggle run's trajectory 86 could not make, since finish_reason
    was never saved there."""
    response = _FakeResponse(
        choices=[_FakeChoice(message=_FakeMessage(content='{"codes": []'), finish_reason="length")],
        usage=_FakeUsage(prompt_tokens=4000, completion_tokens=1024, total_tokens=5024),
    )
    agent = _fake_agent(_ScriptedCreate([response]))
    evidence_log = EvidenceLog(tmp_path)
    install_budget_guard(agent, Budget(max_seconds=30), evidence_log=evidence_log)

    agent.client.chat.completions.create(model="m", messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])

    records = _load_evidence_files(tmp_path)
    assert len(records) == 1
    assert records[0]["finish_reason"] == "length"
    assert records[0]["usage"] == {"prompt_tokens": 4000, "completion_tokens": 1024, "total_tokens": 5024}
    assert records[0]["blocked_by_context_guard"] is False


# --- initial response survives a later repair-prompt context block -----------


def test_initial_response_survives_a_later_repair_prompt_context_block(tmp_path):
    """Real bug this fixes: trajectory 102's initial (successful) chunk
    response was not preserved after ITS OWN repair request was blocked by
    the context guard. Here: chunk 0's initial request succeeds and is
    evidenced; chunk 0's own repair round (a 4-message request) then gets
    blocked -- the FIRST evidence file must still exist and be untouched,
    and the blocked request must be recorded separately, not counted as a
    real attempt.
    """

    class _WordCountTokenizer:
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
            return " ".join(str(m.get("content") or "") for m in messages).split()

    good_response = _FakeResponse(choices=[_FakeChoice(message=_FakeMessage(content="not valid, triggers a repair"), finish_reason="stop")])
    agent = _fake_agent(_ScriptedCreate([good_response]))  # only ONE outcome scripted -- the repair must never reach here
    evidence_log = EvidenceLog(tmp_path)
    coding_context = CodingContext()
    coding_context.begin_record("tau_bench_retail:102:trial_0")
    budget = Budget(max_seconds=30)
    install_budget_guard(
        agent, budget, tokenizer=_WordCountTokenizer(), context_limit=10, reserved_output_tokens=0,
        evidence_log=evidence_log, coding_context=coding_context,
    )

    # Initial attempt (2 messages) -- small enough to pass the token guard, succeeds.
    agent.client.chat.completions.create(model="m", messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    assert budget.calls_made == 1

    # Repair round (4 messages) -- deliberately oversized relative to context_limit=10 -- must be BLOCKED.
    repair_messages = [
        {"role": "system", "content": "s"}, {"role": "user", "content": "u"},
        {"role": "assistant", "content": "not valid, triggers a repair"},
        {"role": "user", "content": "please fix your JSON and this padding makes it clearly too long for the tiny budget"},
    ]
    with pytest.raises(PromptTooLongError):
        agent.client.chat.completions.create(model="m", messages=repair_messages)

    assert budget.calls_made == 1, "the blocked repair request must NOT be counted as a real attempt"

    records = _load_evidence_files(tmp_path)
    assert len(records) == 2, "both the successful initial response AND the blocked repair must be recorded"
    initial, blocked = records[0], records[1]
    assert initial["attempt_kind"] == "initial" and initial["chunk_seq"] == 0
    assert initial["raw_content"] == "not valid, triggers a repair", "the initial response must survive the LATER block untouched"
    assert blocked["attempt_kind"] == "repair" and blocked["chunk_seq"] == 0
    assert blocked["blocked_by_context_guard"] is True
    assert blocked["raw_content"] is None
    assert all(r["composite_id"] == "tau_bench_retail:102:trial_0" for r in records)


# --- errors (not just blocks) are also recorded, never silently swallowed ----


def test_call_error_is_recorded_with_error_field_not_silently_dropped(tmp_path):
    agent = _fake_agent(_ScriptedCreate([ConnectionError("simulated network failure")]))
    evidence_log = EvidenceLog(tmp_path)
    install_budget_guard(agent, Budget(max_seconds=30), evidence_log=evidence_log)

    with pytest.raises(ConnectionError):
        agent.client.chat.completions.create(model="m", messages=[{"role": "user", "content": "u"}])

    records = _load_evidence_files(tmp_path)
    assert len(records) == 1
    assert "ConnectionError" in records[0]["error"]
    assert records[0]["raw_content"] is None
    assert records[0]["blocked_by_context_guard"] is False


# --- structured JSON with correctly escaped nested quotes parses -------------


def test_structured_json_with_escaped_nested_quotes_parses():
    """The exact failure pattern from the real run's trajectory 13: a quote
    that embeds a tool-call's own JSON. Freeform mode broke on this because
    the model emitted the inner quotes unescaped inside a hand-written
    string; a properly JSON-encoded payload (as schema-constrained decoding
    should produce) parses cleanly.
    """
    payload = json.dumps({
        "codes": [{
            "steps": "3-4",
            "quote": '<tool_call name=find_user_id_by_email>{"email":"mia.garcia2723@example.com"}',
            "code": "authenticate user",
            "memo": "Tool call to authenticate the user.",
        }],
        "skipped": "none",
        "segment_memo": "Authentication proceeds normally.",
    })
    codes, skipped, memo = parse_structured_coding_response(payload)
    assert len(codes) == 1
    assert codes[0]["quote"] == '<tool_call name=find_user_id_by_email>{"email":"mia.garcia2723@example.com"}'


def test_incomplete_structured_json_is_rejected_without_fabricating():
    """The exact failure pattern from trajectory 86: the response is cut
    off mid-object (no closing braces). Must raise, never fabricate a
    completed structure."""
    truncated = '{"codes": [{"steps": "1-2", "quote": "some text", "code": "list available options'
    with pytest.raises(StructuredOutputError):
        parse_structured_coding_response(truncated)


# --- parse success does not imply valid quotes or valid step references -----


def test_parse_success_does_not_imply_a_verbatim_quote():
    """A structurally valid (schema-passing) structured response can still
    contain a quote the model altered/paraphrased -- schema validity and
    quote verbatim-ness are two separate checks, and passing one must never
    be read as having passed the other."""
    payload = json.dumps({
        "codes": [{"steps": "1-2", "quote": "this text was never actually said", "code": "x", "memo": "m"}],
        "skipped": "none", "segment_memo": "m",
    })
    codes, _, _ = parse_structured_coding_response(payload)  # this succeeds -- schema is satisfied
    result = verify_quote(codes[0]["quote"], source_text="Agent said: something completely different.")
    assert not result.valid, "schema validity must not be mistaken for quote validity"


# --- a completed run with zero usable outputs reports that accurately -------


def test_completed_run_with_zero_usable_outputs_reports_accurately():
    """Reproduces the real 2026-09-29 run's exact shape: stop_reason was
    "completed" (the loop reached its natural end) while 0 of 3 records
    produced anything usable -- `RunOutcomeReport` must not let
    "completed" read as "succeeded"."""
    report = build_outcome_report(
        run_id="kaggle_smoke_20260929T193910Z",
        server_startup_success=True,
        stop_reason="completed",
        records_loaded=3,
        real_attempts_made=5,
        context_limit_blocks=1,
        coded_results_by_composite_id={
            "tau_bench_retail:13:trial_0": {"chunk_codings": [{"codes": [], "raw": "garbage"}]},
            "tau_bench_retail:86:trial_0": {"chunk_codings": [{"codes": [], "raw": "still garbage"}]},
        },
        quote_validation_reports={},
    )
    assert report.execution_completed is True
    assert report.usable_outputs == 0
    assert report.outcome_label == "completed_with_zero_usable_outputs"
    assert report.parse_or_schema_failures == 2


def test_run_with_usable_outputs_and_some_failures_reports_both_separately():
    class _FakeValidation:
        def __init__(self, total_codes, valid_codes, invalid_codes):
            self.total_codes = total_codes
            self.valid_codes = valid_codes
            self.invalid_codes = invalid_codes

    report = build_outcome_report(
        run_id="r1",
        server_startup_success=True,
        stop_reason="completed",
        records_loaded=2,
        real_attempts_made=2,
        context_limit_blocks=0,
        coded_results_by_composite_id={
            "d:1:t0": {"chunk_codings": [{"codes": [{"code": "x", "quote": "y", "steps": "1", "memo": "m"}], "raw": "ok"}]},
            "d:2:t0": {"chunk_codings": [{"codes": [], "raw": "garbage"}]},
        },
        quote_validation_reports={"d:1:t0": _FakeValidation(total_codes=1, valid_codes=1, invalid_codes=[])},
    )
    assert report.structurally_valid_outputs == 1
    assert report.parse_or_schema_failures == 1
    assert report.usable_outputs == 1
    assert report.outcome_label == "completed_with_usable_outputs"


def test_empty_but_structurally_valid_output_does_not_count_as_usable():
    """Real bug this fixes (2026-09-30, first structured-mode Kaggle
    diagnostic): a genuinely empty result (0 codes) trivially has
    invalid_codes == [] too, so the OLD check ("structurally valid AND
    invalid_codes == []") incorrectly counted it as passing automated
    evidence checks -- HANDOFF: "The old report incorrectly counted this
    empty result as usable." A result must have at least one code to count
    as usable; it is tracked separately via `empty_code_outputs` instead.
    """
    class _FakeValidation:
        def __init__(self, total_codes, valid_codes, invalid_codes):
            self.total_codes = total_codes
            self.valid_codes = valid_codes
            self.invalid_codes = invalid_codes

    report = build_outcome_report(
        run_id="r2",
        server_startup_success=True,
        stop_reason="completed",
        records_loaded=1,
        real_attempts_made=1,
        context_limit_blocks=0,
        coded_results_by_composite_id={
            "d:1:t0": {"chunk_codings": [{"codes": [], "raw": '{"codes": [], "skipped": "none", "segment_memo": "m"}'}]},
        },
        quote_validation_reports={"d:1:t0": _FakeValidation(total_codes=0, valid_codes=0, invalid_codes=[])},
    )
    assert report.structurally_valid_outputs == 1
    assert report.empty_code_outputs == 1
    assert report.usable_outputs == 0
    assert report.outputs_passing_automated_evidence_checks == 0
    assert report.outcome_label == "completed_with_zero_usable_outputs"


# --- failed/invalid results cannot masquerade as successful cached results --


def test_run_server_false_reports_not_run_not_a_failure():
    """RUN_SERVER=False is a deliberate no-op, not a server failure --
    outcome_label must distinguish 'never attempted' from 'attempted and
    failed' (a real gap found while testing the notebook's own default,
    RUN_SERVER=False, path: it initially reported "server_startup_failed")."""
    report = build_outcome_report(
        run_id="r0",
        server_startup_success=False,
        server_attempted=False,
        stop_reason="completed",
        records_loaded=1,
        real_attempts_made=0,
        context_limit_blocks=0,
        coded_results_by_composite_id={},
        quote_validation_reports={},
    )
    assert report.outcome_label == "not_run"


def test_structured_output_error_never_gets_cached_as_done(tmp_path):
    """Mirrors the real caller pattern: a StructuredOutputError must route
    through mark_invalid, never mark_done -- proven here directly against
    ResumeCache rather than assumed."""
    from urop.budgets import ResumeCache, cache_key

    cache = ResumeCache(tmp_path / "resume_index.json")
    key = cache_key(model="m", model_revision="rev1", prompt_hash="p", settings_hash="s", data_hash="d", stage="open_coding")

    try:
        parse_structured_coding_response("{not valid json")
        raise AssertionError("expected StructuredOutputError")
    except StructuredOutputError as exc:
        cache.record_attempt(key)
        cache.mark_invalid(key, reason=str(exc), result_path=str(tmp_path / "raw.json"))

    assert not cache.is_done(key), "a StructuredOutputError must never be cached as a successful result"
