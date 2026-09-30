"""Integration checks against the REAL external/autotracegt_upstream
OpenCodingAgent, talking over a real socket to a fake local HTTP server
(tests/fake_local_server.py) -- not a mock of our own code.

Requires `openai`, `python-dotenv`, and `huggingface_hub` installed (the
upstream package's own import-time dependencies) in addition to this
project's normal CPU test deps. Skipped entirely if the upstream code isn't
vendored (external/autotracegt_upstream, git-ignored) or if `openai` isn't
importable, so the rest of the suite stays runnable without them.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = REPO_ROOT / "external" / "autotracegt_upstream" / "autotracegt"

pytest.importorskip("openai", reason="openai not installed -- integration tests need the real upstream SDK dep")
pytest.importorskip("dotenv", reason="python-dotenv not installed -- upstream utils.utils imports it")

pytestmark = pytest.mark.skipif(
    not UPSTREAM_ROOT.exists(), reason="external/autotracegt_upstream not vendored in this checkout"
)

sys.path.insert(0, str(REPO_ROOT / "src"))
from fake_local_server import FakeLocalChatServer  # noqa: E402

from urop.autotrace_adapter import (  # noqa: E402
    PromptTooLongError,
    ensure_upstream_on_path,
    hash_prompt_templates,
    install_budget_guard,
    register_local_endpoint,
    validate_coding_quotes,
)
from urop.budgets import Budget, BudgetExhausted, ResumeCache, cache_key  # noqa: E402

# JSON_RETRIES must be set before the FIRST import of any upstream module
# this process ever does (module-level constant, read once). conftest-free
# module, so do it here, before ensure_upstream_on_path()/agents import.
os.environ["JSON_RETRIES"] = "2"  # upstream counts TOTAL attempts: 1 initial + 1 repair = 2
ensure_upstream_on_path()
from agents.open_coding import OpenCodingAgent, TrajectoryCoding  # noqa: E402

from urop.autotrace_adapter import classify_trajectory_coding  # noqa: E402
from urop.structured_coding import StructuredOutputError, code_chunk_structured  # noqa: E402


VALID_RESPONSE = (
    '===CODES===\n'
    '[{"steps": "1-2", "code": "checks order status", "quote": "checking the order", "memo": "verification step"}]\n'
    '===SKIPPED===\n'
    '===SEGMENT_MEMO===\n'
    'short memo\n'
    '===END==='
)

SYNTHETIC_RECORD = {
    "instance_id": "synthetic_0",
    "run_id": "trial_0",
    "context": {
        "system_prompt": "You are a synthetic retail agent for testing only.",
        "task_description": "Cancel order 42.",
        "tools": [{"type": "function", "function": {"name": "get_order", "parameters": {}}}],
    },
    "trajectory": [
        {"role": "assistant", "content": "checking the order first"},
        {"role": "tool", "content": '{"order_id": 42, "status": "shipped"}'},
    ],
    "status": "resolved",
}


def _make_agent(base_url: str, budget: Budget, **guard_kwargs) -> OpenCodingAgent:
    register_local_endpoint(base_url=base_url)
    agent = OpenCodingAgent(model="fake-model", endpoint="local", chunk_size=50, concurrency=1, max_tokens=64, temperature=1)
    install_budget_guard(agent, budget, **guard_kwargs)
    return agent


# --- real request path -------------------------------------------------------


def test_real_request_path_against_fake_server():
    with FakeLocalChatServer(responses=[VALID_RESPONSE]) as server:
        budget = Budget(max_seconds=30)
        agent = _make_agent(server.base_url, budget)

        coding = agent.code_record(dict(SYNTHETIC_RECORD))

        assert len(server.requests) == 1
        sent_messages = server.requests[0]["messages"]
        assert sent_messages[0]["role"] == "system"
        assert "Cancel order 42" in sent_messages[1]["content"]
        assert len(coding.all_codes) == 1
        assert coding.all_codes[0]["code"] == "checks order status"
        assert budget.calls_made == 1


# --- malformed-response repair count -----------------------------------------


def test_malformed_response_triggers_exactly_one_repair_attempt():
    """First response is malformed (no ===CODES=== block at all), second
    (the repair round) is valid. With JSON_RETRIES=2 (1 initial + 1 repair,
    the corrected semantics), this must succeed after exactly 2 requests --
    not 1 (old, wrong semantics that skip repair entirely) and not more.
    """
    responses = ["this is not a valid coding response at all", VALID_RESPONSE]
    with FakeLocalChatServer(responses=responses) as server:
        budget = Budget(max_seconds=30)
        agent = _make_agent(server.base_url, budget)

        coding = agent.code_record(dict(SYNTHETIC_RECORD))

        assert len(server.requests) == 2, "must have made exactly 2 attempts: 1 initial + 1 repair"
        assert budget.calls_made == 2, "both attempts (including the failed one) must be counted"
        assert len(coding.all_codes) == 1, "the repair round's valid response must be the one used"
        # the repair request must actually contain the repair instruction and the raw first response
        repair_request_messages = server.requests[1]["messages"]
        assert any("malformed or missing JSON" in (m.get("content") or "") for m in repair_request_messages)


def test_persistently_malformed_response_exhausts_retries_without_crashing():
    with FakeLocalChatServer(responses=["still not valid", "still not valid"]) as server:
        budget = Budget(max_seconds=30)
        agent = _make_agent(server.base_url, budget)

        coding = agent.code_record(dict(SYNTHETIC_RECORD))

        assert len(server.requests) == 2, "must stop at JSON_RETRIES=2, not retry forever"
        assert coding.all_codes == [], "no valid codes -- must not fabricate a clean label from unparsable output"
        assert coding.chunk_codings[0].raw == "still not valid", "the raw invalid output must be preserved, not discarded"


# --- real failure patterns from the 2026-09-29 Kaggle run (see docs/experiment_log.md) --


def test_unescaped_quotes_inside_tool_call_quotation_are_rejected():
    """The exact failure pattern from that run's trajectory 13: the model's
    "quote" value embeds a tool-call's own JSON with UNescaped inner double
    quotes (`"quote": "<tool_call ...>{"email":"x@example.com"}"`), which
    breaks `json.loads` on the surrounding CODES array. Upstream's own
    `parse_response` must reject this (codes stays empty) rather than
    crash or silently accept a mis-parsed structure; the repair round then
    runs (real bug-adjacent behavior, not fabrication) since the first
    response yields zero codes.
    """
    malformed = (
        '## BEHAVIORAL LOG\n\n===CODES===\n[\n'
        '  {\n    "steps": "1-2",\n'
        '    "quote": "<tool_call name=find_user_id_by_email>{"email":"mia.garcia2723@example.com"}",\n'
        '    "code": "authenticate user",\n    "memo": "m"\n  }\n]\n'
        '===SKIPPED===\nnone\n===SEGMENT_MEMO===\nm\n===END==='
    )
    with FakeLocalChatServer(responses=[malformed, malformed]) as server:
        budget = Budget(max_seconds=30)
        agent = _make_agent(server.base_url, budget)

        coding = agent.code_record(dict(SYNTHETIC_RECORD))

        assert coding.all_codes == [], "unescaped inner quotes must not silently parse into a fabricated code"
        assert len(server.requests) == 2, "a genuine repair attempt is expected, not a crash"
        assert classify_trajectory_coding(coding.to_dict()) == "parse_failed"


def test_incomplete_json_is_rejected_without_fabricating_a_repair():
    """The exact failure pattern from that run's trajectory 86: the
    response is cut off mid-object (`finish_reason="length"` suspected,
    unconfirmed in that run because finish_reason wasn't saved -- see
    test_evidence_and_reporting.py for the fix). Upstream's repair loop
    must not invent a closing bracket or missing fields."""
    truncated = (
        '```json\n===CODES===\n[\n  {\n    "steps": "1-2",\n    "quote": "some fragment",\n'
        '    "code": "list available options'  # cut off mid-string, no closing quote/brace/bracket anywhere
    )
    with FakeLocalChatServer(responses=[truncated, truncated]) as server:
        budget = Budget(max_seconds=30)
        agent = _make_agent(server.base_url, budget)

        coding = agent.code_record(dict(SYNTHETIC_RECORD))

        assert coding.all_codes == [], "truncated JSON must not be silently completed/fabricated"
        assert coding.chunk_codings[0].raw == truncated, "the raw truncated output must be preserved verbatim, not discarded or rewritten"
        assert classify_trajectory_coding(coding.to_dict()) == "parse_failed"


# --- per-request token check, including later chunks and repair rounds ------


class _WordCountTokenizer:
    """Test double: one 'token' per whitespace-separated word. Not a real
    tokenizer -- just needs apply_chat_template's real contract (token count
    grows with the actual message content) for exercising the guard."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        text = " ".join(str(m.get("content") or "") for m in messages)
        return text.split()


def test_token_check_runs_on_every_actual_request_including_repair():
    """A request whose exact rendered messages exceed budget must be refused
    (PromptTooLongError) even though an ahead-of-time check on the first
    attempt alone would have looked fine -- the repair round's grown
    message list (original + raw response + repair instruction) is what
    actually gets checked here, using apply_chat_template on the real sent
    messages, not a precomputed estimate.
    """
    long_first_response = "not valid json " + ("padding word " * 200)  # forces the repair round's prompt to grow a lot
    with FakeLocalChatServer(responses=[long_first_response, VALID_RESPONSE]) as server:
        budget = Budget(max_seconds=30)
        tokenizer = _WordCountTokenizer()
        # Real prompt (open_coding system+user templates are long, real instruction
        # text) is ~715 words for attempt 1 even with this tiny synthetic trajectory;
        # the repair round adds the ~200-word raw response + repair instruction on
        # top (~950 words). 800 sits strictly between the two.
        agent = _make_agent(
            server.base_url, budget, tokenizer=tokenizer, context_limit=800, reserved_output_tokens=0
        )

        with pytest.raises(PromptTooLongError):
            agent.code_record(dict(SYNTHETIC_RECORD))

        assert len(server.requests) == 1, "the oversized repair request must never actually be sent"


def test_token_check_allows_requests_within_budget():
    with FakeLocalChatServer(responses=[VALID_RESPONSE]) as server:
        budget = Budget(max_seconds=30)
        tokenizer = _WordCountTokenizer()
        agent = _make_agent(server.base_url, budget, tokenizer=tokenizer, context_limit=100000, reserved_output_tokens=0)

        coding = agent.code_record(dict(SYNTHETIC_RECORD))
        assert len(coding.all_codes) == 1


# --- resume caching -----------------------------------------------------------


def test_cache_reuse_skips_a_repeat_call(tmp_path):
    with FakeLocalChatServer(responses=[VALID_RESPONSE]) as server:
        budget = Budget(max_seconds=30)
        agent = _make_agent(server.base_url, budget)
        resume_cache = ResumeCache(tmp_path / "resume_index.json")

        prompt_hash = hash_prompt_templates("open_coding_system", "open_coding_user")
        key = cache_key(
            model="fake-model", model_revision="rev1", prompt_hash=prompt_hash,
            settings_hash="chunk_size=50;temperature=1",
            data_hash="synthetic-record-hash", stage="open_coding",
        )
        result_path = tmp_path / "result.json"

        assert not resume_cache.is_done(key)
        coding = agent.code_record(dict(SYNTHETIC_RECORD))
        result_path.write_text('{"codes": "saved"}', encoding="utf-8")
        resume_cache.mark_done(key, result_path=str(result_path))
        assert len(server.requests) == 1

        # Second "run": cache says done and the file exists -> must skip, no second request.
        if resume_cache.is_done(key):
            pass  # this is the real call site's branch; we just assert the request count below
        else:
            agent.code_record(dict(SYNTHETIC_RECORD))
        assert len(server.requests) == 1, "a cache hit must not trigger a second real request"


def test_cache_key_changes_with_generation_settings():
    prompt_hash = hash_prompt_templates("open_coding_system", "open_coding_user")
    k1 = cache_key(model="fake-model", model_revision="rev1", prompt_hash=prompt_hash, settings_hash="temperature=1", data_hash="d", stage="open_coding")
    k2 = cache_key(model="fake-model", model_revision="rev1", prompt_hash=prompt_hash, settings_hash="temperature=0.2", data_hash="d", stage="open_coding")
    assert k1 != k2


def test_cache_key_changes_with_model_revision():
    prompt_hash = hash_prompt_templates("open_coding_system", "open_coding_user")
    k1 = cache_key(model="fake-model", model_revision="rev1", prompt_hash=prompt_hash, settings_hash="s", data_hash="d", stage="open_coding")
    k2 = cache_key(model="fake-model", model_revision="rev2", prompt_hash=prompt_hash, settings_hash="s", data_hash="d", stage="open_coding")
    assert k1 != k2, "a different resolved model revision (different actual weights) must not share a cache entry"


def test_resume_cache_loads_a_valid_cached_result_back_into_a_real_trajectorycoding(tmp_path):
    """The 'load valid cached results on resume' requirement: a resumed run
    must reconstruct the SAME real TrajectoryCoding object from the cached
    file, not just note that a cache hit occurred, so it can be folded back
    into that run's summary/report."""
    with FakeLocalChatServer(responses=[VALID_RESPONSE]) as server:
        budget = Budget(max_seconds=30)
        agent = _make_agent(server.base_url, budget)
        coding = agent.code_record(dict(SYNTHETIC_RECORD))

    assert classify_trajectory_coding(coding.to_dict()) == "valid"

    result_path = tmp_path / "cached.json"
    result_path.write_text(json.dumps(coding.to_dict()), encoding="utf-8")

    resume_cache = ResumeCache(tmp_path / "resume_index.json")
    key = cache_key(model="fake-model", model_revision="rev1", prompt_hash="p", settings_hash="s", data_hash="d", stage="open_coding")
    resume_cache.mark_done(key, result_path=str(result_path))

    loaded_path = resume_cache.result_path_for(key)
    assert loaded_path is not None
    reloaded = TrajectoryCoding.from_dict(json.loads(loaded_path.read_text(encoding="utf-8")))
    assert reloaded.instance_id == coding.instance_id
    assert reloaded.all_codes == coding.all_codes, "a resumed run must recover the exact same codes, not an empty placeholder"


# --- invalid/parse-failed output: preserved, classified, never cached as done -


def test_persistently_malformed_output_is_classified_parse_failed_not_cached_as_done(tmp_path):
    """Item 3: a parse-failed result must never be marked done in the resume
    cache -- it must be classified explicitly and routed through
    mark_invalid so a future resume does not silently treat it as a clean
    success."""
    with FakeLocalChatServer(responses=["still not valid", "still not valid"]) as server:
        budget = Budget(max_seconds=30)
        agent = _make_agent(server.base_url, budget)
        coding = agent.code_record(dict(SYNTHETIC_RECORD))

    coding_dict = coding.to_dict()
    assert classify_trajectory_coding(coding_dict) == "parse_failed"

    resume_cache = ResumeCache(tmp_path / "resume_index.json")
    key = cache_key(model="fake-model", model_revision="rev1", prompt_hash="p", settings_hash="s", data_hash="d", stage="open_coding")
    result_path = tmp_path / "invalid_result.json"
    result_path.write_text('{}', encoding="utf-8")

    resume_cache.record_attempt(key)
    resume_cache.mark_invalid(key, reason="parse_failed", result_path=str(result_path))

    assert not resume_cache.is_done(key), "a parse-failed result must never read back as done"
    assert not resume_cache.is_exhausted(key, max_attempts=2), "1 attempt so far, must allow exactly 1 more before giving up"

    resume_cache.record_attempt(key)
    resume_cache.mark_invalid(key, reason="parse_failed", result_path=str(result_path))
    assert resume_cache.is_exhausted(key, max_attempts=2), "after the bounded retry budget, must stop, not rerun indefinitely"


# --- quote/span validation, wired to a real coding result -------------------


def test_quote_validation_separates_valid_from_invalid_codes():
    bad_quote_response = (
        '===CODES===\n'
        '[{"steps": "1-2", "code": "checks order", "quote": "this text is not in the trajectory", "memo": "m"}]\n'
        '===SKIPPED===\n===SEGMENT_MEMO===\n===END==='
    )
    with FakeLocalChatServer(responses=[bad_quote_response]) as server:
        budget = Budget(max_seconds=30)
        agent = _make_agent(server.base_url, budget)
        coding = agent.code_record(dict(SYNTHETIC_RECORD))

    report = validate_coding_quotes(coding.to_dict(), trajectory=SYNTHETIC_RECORD["trajectory"], chunk_size=50)
    assert report.total_codes == 1
    assert report.valid_codes == 0
    assert len(report.invalid_codes) == 1
    assert report.invalid_codes[0]["reason"] == "quote not found verbatim in source text"


def test_quote_validation_accepts_a_real_verbatim_quote():
    with FakeLocalChatServer(responses=[VALID_RESPONSE]) as server:
        budget = Budget(max_seconds=30)
        agent = _make_agent(server.base_url, budget)
        coding = agent.code_record(dict(SYNTHETIC_RECORD))

    report = validate_coding_quotes(coding.to_dict(), trajectory=SYNTHETIC_RECORD["trajectory"], chunk_size=50)
    assert report.total_codes == 1
    assert report.valid_codes == 1
    assert report.invalid_codes == []


# --- cited-step quote validation: a quote verbatim in the WRONG step is rejected ---


def test_quote_verbatim_in_chunk_but_cited_under_the_wrong_step_is_rejected():
    """Real fix (2026-09-30): the old check verified a quote against the
    WHOLE chunk, so a quote that is verbatim somewhere in the chunk but
    from a DIFFERENT step than the one the code cites would have passed.
    SYNTHETIC_RECORD step 2's tool content is '{"order_id": 42, "status":
    "shipped"}' -- citing steps="1" (step 1's content is "checking the
    order first") while quoting step 2's text must now be rejected.
    """
    wrong_step_response = (
        '===CODES===\n'
        '[{"steps": "1", "code": "checks order", "quote": "\\"order_id\\": 42", "memo": "m"}]\n'
        '===SKIPPED===\n===SEGMENT_MEMO===\n===END==='
    )
    with FakeLocalChatServer(responses=[wrong_step_response]) as server:
        budget = Budget(max_seconds=30)
        agent = _make_agent(server.base_url, budget)
        coding = agent.code_record(dict(SYNTHETIC_RECORD))

    report = validate_coding_quotes(coding.to_dict(), trajectory=SYNTHETIC_RECORD["trajectory"], chunk_size=50)
    assert report.total_codes == 1
    assert report.valid_codes == 0, "a quote from the wrong step must not be accepted just because it's verbatim somewhere in the chunk"
    assert report.invalid_codes[0]["reason"] == "quote not found verbatim in source text"


def test_out_of_range_step_citation_is_rejected_with_a_clear_reason():
    out_of_range_response = (
        '===CODES===\n'
        '[{"steps": "5-9", "code": "checks order", "quote": "checking the order", "memo": "m"}]\n'
        '===SKIPPED===\n===SEGMENT_MEMO===\n===END==='
    )
    with FakeLocalChatServer(responses=[out_of_range_response]) as server:
        budget = Budget(max_seconds=30)
        agent = _make_agent(server.base_url, budget)
        coding = agent.code_record(dict(SYNTHETIC_RECORD))  # only 2 real steps

    report = validate_coding_quotes(coding.to_dict(), trajectory=SYNTHETIC_RECORD["trajectory"], chunk_size=50)
    assert report.total_codes == 1
    assert report.valid_codes == 0
    assert report.invalid_codes[0]["reason"] == "step reference outside this chunk"


def test_malformed_step_format_is_rejected_with_a_clear_reason():
    malformed_response = (
        '===CODES===\n'
        '[{"steps": "not-a-range", "code": "checks order", "quote": "checking the order", "memo": "m"}]\n'
        '===SKIPPED===\n===SEGMENT_MEMO===\n===END==='
    )
    with FakeLocalChatServer(responses=[malformed_response]) as server:
        budget = Budget(max_seconds=30)
        agent = _make_agent(server.base_url, budget)
        coding = agent.code_record(dict(SYNTHETIC_RECORD))

    report = validate_coding_quotes(coding.to_dict(), trajectory=SYNTHETIC_RECORD["trajectory"], chunk_size=50)
    assert report.total_codes == 1
    assert report.valid_codes == 0
    assert report.invalid_codes[0]["reason"] == "invalid step format (use 4 or 4-7 or comma-separated ranges)"


# --- timeout cleanup: real work actually stops -------------------------------


def test_timeout_against_a_slow_fake_server_stops_promptly_and_is_recorded():
    """The fake server sleeps 5s before replying; the budget for the actual
    request only allows ~0.3s. The real openai client's own timeout must
    abort the request -- this proves the CLIENT side genuinely stops
    waiting quickly (a real, standard HTTP-client timeout, not a thread we
    merely stop watching) and that the attempt and stop reason are both
    recorded.

    Bug found and fixed here: the budget's clock starts at `Budget(...)`
    construction, but `_make_agent` -> `OpenCodingAgent(...)` construction
    (building the sync/async openai clients, loading the prompt template)
    measured at ~1s in a cold environment (see docs/experiment_log.md) --
    well over a 0.3s budget by itself. A budget created tight BEFORE
    `_make_agent` gets silently consumed by agent construction, so the
    first real HTTP attempt sees an ALREADY-exhausted budget
    (`budget_exhausted_before_call`) instead of timing out mid-request
    (`budget_exhausted_mid_call`) -- the assertion below was failing on
    construction time, not on the timeout behaviour it's meant to test.

    Fix: construct the agent under a generous budget so setup can't consume
    it, THEN reset that SAME Budget object's clock/counters to a fresh,
    short window immediately before the slow request. This has to mutate
    the existing object rather than construct-and-attach a brand new one:
    `install_budget_guard` closes over the `Budget` instance by reference
    when it wraps `agent.client.chat.completions.create`, and calling
    `install_budget_guard` a second time with a new Budget would wrap the
    ALREADY-guarded method as the new "original", double-applying the
    guard (confirmed by tracing it through: the outer guard's own
    `timeout=` kwarg collides with the inner guard forwarding its own
    `timeout=` to the same call, a hard `TypeError`). Budget is a plain,
    unfrozen dataclass, so resetting its fields in place is the correct
    "fresh budget, same wiring" move here, not a workaround.
    """
    with FakeLocalChatServer(responses=[VALID_RESPONSE], delay_seconds=5.0) as server:
        setup_budget = Budget(max_seconds=60)  # generous -- agent construction must not be able to eat this
        agent = _make_agent(server.base_url, setup_budget)

        # Fresh, short budget for the actual timed call -- same object
        # `install_budget_guard` already wrapped `create` around, reset in
        # place immediately before the slow request it's meant to bound.
        setup_budget.max_seconds = 0.3
        setup_budget.started_at = time.monotonic()
        setup_budget.calls_made = 0
        setup_budget.stop_reason = None
        budget = setup_budget

        start = time.monotonic()
        with pytest.raises(BudgetExhausted) as excinfo:
            agent.code_record(dict(SYNTHETIC_RECORD))
        elapsed = time.monotonic() - start

        assert elapsed < 3.0, "client must stop waiting near the budget deadline, not the server's full 5s delay"
        assert len(server.requests) == 1, (
            "the request must actually have reached the fake server before timing out -- proves this is a "
            "real client-side timeout on an in-flight request, not e.g. a connection that never went out"
        )
        assert excinfo.value.reason == "budget_exhausted_mid_call", (
            "must be a mid-call timeout against the fresh 0.3s budget, not budget already exhausted by setup"
        )
        assert budget.calls_made >= 1, "the timed-out attempt must still be counted"


# --- structured mode: confirmed length-limited responses fail explicitly ----


def test_structured_mode_raises_explicitly_on_finish_reason_length():
    """A response cut off by the server's own max-token limit must raise
    StructuredOutputError immediately, attributing the failure to
    truncation -- not silently attempt (and likely fail) JSON parsing on
    truncated text with a generic, unattributed error."""
    with FakeLocalChatServer(responses=['{"codes": [{"steps": "1'], finish_reason="length") as server:
        budget = Budget(max_seconds=30)
        agent = _make_agent(server.base_url, budget)

        with pytest.raises(StructuredOutputError, match="output limit"):
            code_chunk_structured(
                agent, SYNTHETIC_RECORD["trajectory"], step_offset=1,
                task=SYNTHETIC_RECORD["context"]["task_description"],
                tools=json.dumps([t["function"]["name"] for t in SYNTHETIC_RECORD["context"]["tools"]]),
                status=SYNTHETIC_RECORD["status"],
            )
        assert len(server.requests) == 1
