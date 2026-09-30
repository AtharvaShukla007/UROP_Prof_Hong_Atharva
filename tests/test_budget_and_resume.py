"""CPU-only checks for src/urop/budgets.py, autotrace_adapter.py, reporting.py.

Covers handoff §8 acceptance checks 4-7:
  4. Prompt assembly for held-out annotation excludes explicit outcomes and future events.
  5. Unsupported quotes and parse errors become invalid/abstain, not clean labels.
  6. Budget cancellation/resume works on a deliberately slow mock call; no
     overwriting raw outputs; changed prompt/model invalidates cache.
  7. Summary generation cannot report research AUROC from synthetic fixtures
     or missing model outputs.
"""
from __future__ import annotations

import json
import time

import pytest

from urop.autotrace_adapter import (
    MockCoder,
    assemble_holdout_prompt,
    classify_chunk_coding,
    classify_trajectory_coding,
    find_leaked_substrings,
    register_local_endpoint,
    verify_quote,
)
from urop.budgets import Budget, BudgetExhausted, ResumeCache, bounded_llm_call, cache_key
from urop.manifests import build_run_manifest, save_manifest
from urop.reporting import NotResearchDataError, summarize_run


# --- check 4: leakage-free held-out prompts ------------------------------------


def test_holdout_prompt_excludes_leakage_keys():
    trajectory = [
        {"role": "assistant", "content": "checking order", "reward": 1.0, "status": "resolved"},
        {"role": "tool", "content": "order found", "outcome": "success"},
        {"role": "assistant", "content": "acting now"},
        {"role": "tool", "content": "this step is AFTER the cut and must not appear"},
    ]
    prompt = assemble_holdout_prompt(trajectory, up_to_index=3)
    assert len(prompt) == 3
    assert all("reward" not in m and "status" not in m and "outcome" not in m for m in prompt)
    rendered = json.dumps(prompt)
    assert "must not appear" not in rendered


def test_find_leaked_substrings_flags_real_leakage():
    rendered = "The agent then confirmed the task was resolved=true with reward 1.0"
    leaked = find_leaked_substrings(rendered, forbidden_substrings=["resolved=true", "reward 1.0", "not present"])
    assert set(leaked) == {"resolved=true", "reward 1.0"}


def test_register_local_endpoint_refuses_non_local_host():
    with pytest.raises(ValueError):
        register_local_endpoint(base_url="https://api.openai.com/v1")


# --- check 5: quote verification -> invalid/abstain, not a clean label ---------


def test_verbatim_quote_is_valid():
    result = verify_quote("cancelled the order", source_text="Agent said: cancelled the order after confirming.")
    assert result.valid


def test_non_verbatim_quote_is_invalid_not_silently_accepted():
    result = verify_quote("cancelled the order immediately", source_text="Agent said: cancelled the order.")
    assert not result.valid
    assert result.reason == "quote not found verbatim in source text"


def test_empty_quote_is_invalid():
    result = verify_quote("", source_text="anything")
    assert not result.valid


# --- check 6: enforceable cancellation, resume cache, cache invalidation -------


def test_bounded_call_stops_real_work_not_just_the_caller():
    """The timeout must stop the WORK, not just let the caller give up while
    background work keeps running. MockCoder's own loop checks the deadline
    cooperatively (mirroring a real OpenAI-SDK client honouring `timeout=`),
    so `completed_calls` staying 0 proves the simulated work itself never
    finished -- not merely that we stopped waiting for it.
    """
    slow_coder = MockCoder(delay_seconds=2.0)
    budget = Budget(max_seconds=0.2)
    start = time.monotonic()
    with pytest.raises(BudgetExhausted) as excinfo:
        bounded_llm_call(slow_coder, budget=budget, messages=[])
    elapsed = time.monotonic() - start
    assert excinfo.value.reason == "budget_exhausted_mid_call"
    assert elapsed < 1.0, "must cut off near the budget deadline, not wait for the slow call"
    assert slow_coder.calls == 1, "the attempt must still be counted even though it failed"
    assert slow_coder.completed_calls == 0, "the simulated work itself must have actually stopped, not just the caller"


def test_bounded_call_refuses_new_work_once_budget_is_already_exhausted():
    fast_coder = MockCoder(delay_seconds=0.0)
    budget = Budget(max_seconds=0.01)
    time.sleep(0.05)
    with pytest.raises(BudgetExhausted) as excinfo:
        bounded_llm_call(fast_coder, budget=budget, messages=[])
    assert excinfo.value.reason == "budget_exhausted_before_call"
    assert fast_coder.calls == 0


def test_bounded_call_counts_attempts_not_just_successes():
    """A failing call still counts as a real inference attempt."""
    flaky_coder = MockCoder(fail_first_n=2)
    budget = Budget(max_seconds=60)
    for _ in range(2):
        with pytest.raises(ValueError):
            bounded_llm_call(flaky_coder, budget=budget, messages=[])
    result = bounded_llm_call(flaky_coder, budget=budget, messages=[])
    assert result.content
    assert budget.calls_made == 3, "2 failures + 1 success must all count as attempts"


def test_bounded_call_per_call_timeout_does_not_kill_the_whole_budget():
    """A tight timeout_override on one call is a normal failure, not an
    overall budget exhaustion, if the budget itself still has time left."""
    slow_coder = MockCoder(delay_seconds=1.0)
    budget = Budget(max_seconds=60)
    with pytest.raises(TimeoutError):
        bounded_llm_call(slow_coder, budget=budget, timeout_override=0.05, messages=[])
    assert budget.stop_reason is None, "one slow call should not poison the whole run's budget"
    assert not budget.exhausted()


def test_resume_cache_skips_only_when_completed_and_file_exists(tmp_path):
    cache = ResumeCache(tmp_path / "resume_index.json")
    key = cache_key(model="m", model_revision="rev1", prompt_hash="p", settings_hash="s", data_hash="d", stage="open_coding")
    assert not cache.is_done(key)

    result_file = tmp_path / "result.json"
    cache.mark_done(key, result_path=str(result_file))
    assert not cache.is_done(key), "marked done but the result file does not exist yet -> must not be skipped"

    result_file.write_text("{}", encoding="utf-8")
    assert cache.is_done(key)


def test_resume_cache_reloads_from_disk(tmp_path):
    index_path = tmp_path / "resume_index.json"
    result_file = tmp_path / "result.json"
    result_file.write_text("{}", encoding="utf-8")
    cache1 = ResumeCache(index_path)
    key = cache_key(model="m", model_revision="rev1", prompt_hash="p", settings_hash="s", data_hash="d", stage="open_coding")
    cache1.mark_done(key, result_path=str(result_file))

    cache2 = ResumeCache(index_path)
    assert cache2.is_done(key)


def test_changed_prompt_model_or_revision_invalidates_cache_key():
    base = dict(
        model="qwen2.5-3b", model_revision="rev1", prompt_hash="abc123",
        settings_hash="s", data_hash="d", stage="open_coding",
    )
    k1 = cache_key(**base)
    k2 = cache_key(**{**base, "prompt_hash": "different"})
    k3 = cache_key(**{**base, "model": "different-model"})
    k4 = cache_key(**{**base, "model_revision": "rev2"})
    assert len({k1, k2, k3, k4}) == 4


def test_raw_output_file_is_never_overwritten_by_resume_cache(tmp_path):
    raw = tmp_path / "raw_input.jsonl"
    raw.write_text('{"a": 1}\n', encoding="utf-8")
    original = raw.read_bytes()

    cache = ResumeCache(tmp_path / "resume_index.json")
    key = cache_key(model="m", model_revision="rev1", prompt_hash="p", settings_hash="s", data_hash="d", stage="open_coding")
    cache.mark_done(key, result_path=str(raw))  # even if pointed at the raw file, ResumeCache itself never writes to it

    assert raw.read_bytes() == original


# --- resume-cache attempt tracking, exhaustion, result loading -----------------


def test_resume_cache_result_path_for_only_returns_a_path_once_done(tmp_path):
    cache = ResumeCache(tmp_path / "resume_index.json")
    key = cache_key(model="m", model_revision="rev1", prompt_hash="p", settings_hash="s", data_hash="d", stage="open_coding")
    assert cache.result_path_for(key) is None

    result_file = tmp_path / "result.json"
    result_file.write_text("{}", encoding="utf-8")
    cache.mark_done(key, result_path=str(result_file))
    assert cache.result_path_for(key) == result_file


def test_resume_cache_attempts_accumulate_across_mark_invalid_calls(tmp_path):
    cache = ResumeCache(tmp_path / "resume_index.json")
    key = cache_key(model="m", model_revision="rev1", prompt_hash="p", settings_hash="s", data_hash="d", stage="open_coding")

    assert cache.record_attempt(key) == 1
    cache.mark_invalid(key, reason="parse_failed", result_path=str(tmp_path / "raw1.json"))
    assert not cache.is_exhausted(key, max_attempts=2), "only 1 attempt so far, must not be exhausted at max_attempts=2"

    assert cache.record_attempt(key) == 2, "attempts must accumulate, not reset when mark_invalid is called"
    cache.mark_invalid(key, reason="parse_failed", result_path=str(tmp_path / "raw2.json"))
    assert cache.is_exhausted(key, max_attempts=2)
    assert not cache.is_done(key), "an exhausted-invalid key must never be reported as done"


def test_resume_cache_is_exhausted_false_for_unknown_or_completed_key(tmp_path):
    cache = ResumeCache(tmp_path / "resume_index.json")
    key = cache_key(model="m", model_revision="rev1", prompt_hash="p", settings_hash="s", data_hash="d", stage="open_coding")
    assert not cache.is_exhausted(key, max_attempts=1), "an unattempted key is not exhausted"

    result_file = tmp_path / "result.json"
    result_file.write_text("{}", encoding="utf-8")
    cache.record_attempt(key)
    cache.mark_done(key, result_path=str(result_file))
    assert not cache.is_exhausted(key, max_attempts=1), "a completed key is never exhausted, regardless of attempt count"


# --- classify_chunk_coding / classify_trajectory_coding: parse failure -------
# must stay distinguishable from a legitimate empty result, never silently
# folded into either a clean success or a dropped record (handoff §5 point 5).


def test_classify_chunk_coding_with_codes_is_coded():
    assert classify_chunk_coding({"codes": [{"code": "x", "quote": "y"}], "raw": "anything"}) == "coded"


def test_classify_chunk_coding_explicit_empty_list_is_empty_clean():
    raw = "===CODES===\n[]\n===SKIPPED===\n===SEGMENT_MEMO===\nnothing notable\n===END==="
    assert classify_chunk_coding({"codes": [], "raw": raw}) == "empty_clean"


def test_classify_chunk_coding_unparseable_raw_is_parse_failed():
    raw = "the model rambled and never produced a CODES block at all"
    assert classify_chunk_coding({"codes": [], "raw": raw}) == "parse_failed"


def test_classify_trajectory_coding_any_parse_failed_chunk_invalidates_whole_result():
    coding_dict = {
        "chunk_codings": [
            {"codes": [{"code": "x", "quote": "y"}], "raw": "===CODES===[{...}]==="},
            {"codes": [], "raw": "garbage, no markers"},
        ]
    }
    assert classify_trajectory_coding(coding_dict) == "parse_failed"


def test_classify_trajectory_coding_all_coded_or_empty_clean_chunks_is_valid():
    coding_dict = {
        "chunk_codings": [
            {"codes": [{"code": "x", "quote": "y"}], "raw": "===CODES===[{...}]==="},
            {"codes": [], "raw": "===CODES===\n[]\n==="},
        ]
    }
    assert classify_trajectory_coding(coding_dict) == "valid"


def test_classify_trajectory_coding_no_chunks_at_all_is_parse_failed():
    assert classify_trajectory_coding({"chunk_codings": []}) == "parse_failed"


# --- check 7: reporting refuses to summarize mock/synthetic runs as research ---


def test_summarize_run_refuses_metrics_for_mock_backend(tmp_path):
    manifest = build_run_manifest(
        run_id="run_mock_1", stage="probe", config={}, input_files=[], backend_label="mock"
    )
    manifest_path = tmp_path / "manifest.json"
    save_manifest(manifest, manifest_path)

    with pytest.raises(NotResearchDataError):
        summarize_run(manifest_path, metrics={"auroc": 0.95})


def test_summarize_run_allows_metrics_for_a_real_backend_label(tmp_path):
    manifest = build_run_manifest(
        run_id="run_local_1", stage="probe", config={}, input_files=[], backend_label="local"
    )
    manifest_path = tmp_path / "manifest.json"
    save_manifest(manifest, manifest_path)

    summary = summarize_run(manifest_path, metrics={"auroc": 0.72})
    assert summary.metrics == {"auroc": 0.72}
    assert summary.warnings == []


def test_summarize_run_without_metrics_never_fabricates_them(tmp_path):
    manifest = build_run_manifest(
        run_id="run_mock_2", stage="probe", config={}, input_files=[], backend_label="mock"
    )
    manifest_path = tmp_path / "manifest.json"
    save_manifest(manifest, manifest_path)

    summary = summarize_run(manifest_path)
    assert summary.metrics is None
