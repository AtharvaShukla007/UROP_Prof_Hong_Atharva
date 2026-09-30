"""CPU-only checks for src/urop/tau_bench_adapter.py, against the hand-written
synthetic fixture tests/fixtures/tiny_tau_bench_raw.json -- never real data.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from urop.data import load_and_clean_jsonl, validate_raw_record
from urop.splits import make_split, task_key, verify_no_overlap
from urop.tau_bench_adapter import (
    convert_tau_bench_corpus,
    extract_retail_tool_schemas,
    load_tau_bench_raw_array,
    sample_train_records,
    validate_tau_bench_record,
)

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_tau_bench_raw.json"
REPO_ROOT = Path(__file__).resolve().parents[1]
TAU_BENCH_CLONE = REPO_ROOT / "external" / "tau-bench"

FAKE_TOOLS = [{"type": "function", "function": {"name": "fake_tool", "parameters": {}}}]


@pytest.fixture(scope="module")
def raw_records():
    return load_tau_bench_raw_array(FIXTURE)


@pytest.fixture(scope="module")
def converted(raw_records):
    return convert_tau_bench_corpus(raw_records, tool_schemas=FAKE_TOOLS)


def test_raw_array_has_expected_length(raw_records):
    assert len(raw_records) == 6


def test_conversion_counts(converted):
    _, report = converted
    assert report.total_raw == 6
    assert report.converted == 3
    assert report.rejected == 3
    assert report.reward_1_count == 2
    assert report.reward_0_count == 1
    assert report.distinct_task_ids == ["0", "1"]


def test_missing_reward_is_rejected(converted):
    _, report = converted
    assert any("missing 'reward'" in e for e in report.errors)


def test_non_binary_reward_is_rejected(converted):
    _, report = converted
    assert any("expected exactly 0.0 or 1.0" in e for e in report.errors)


def test_wrong_first_role_is_rejected(converted):
    _, report = converted
    assert any("traj[0] expected role 'system'" in e for e in report.errors)


def test_reward_maps_to_resolved_boolean(converted):
    records, _ = converted
    by_key = {(r["instance_id"], r["run_id"]): r for r in records}
    assert by_key[("0", "trial_0")]["resolved"] is True
    assert by_key[("1", "trial_0")]["resolved"] is False


def test_multi_trial_same_task_both_present(converted):
    records, _ = converted
    task_0_records = [r for r in records if r["instance_id"] == "0"]
    assert {r["run_id"] for r in task_0_records} == {"trial_0", "trial_1"}


def test_tool_schemas_attached_to_every_record(converted):
    records, _ = converted
    assert all(r["tools"] == FAKE_TOOLS for r in records)


def test_converted_output_validates_against_our_own_raw_schema(converted):
    """The whole point of the adapter: its output must satisfy
    src/urop/data.py's validator unchanged."""
    records, _ = converted
    for i, rec in enumerate(records):
        errors = validate_raw_record(rec, line_no=i)
        assert errors == [], f"converted record {i} failed our own raw-schema validator: {errors}"


def test_converted_records_flow_through_load_and_clean_and_split(converted, tmp_path):
    records, _ = converted
    jsonl_path = tmp_path / "converted.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        import json

        for r in records:
            f.write(json.dumps(r) + "\n")

    cleaned, load_report = load_and_clean_jsonl(jsonl_path, domain="tau_bench_retail")
    assert load_report.accepted == 3
    assert load_report.rejected == 0  # already-converted records are all valid raw shape

    manifest = make_split(cleaned, seed=42, train_frac=0.8)
    verify_no_overlap(manifest)
    # task "0" has two trials -- both must be on the same side (test_data_and_splits.py
    # already covers this generically; this just confirms real tau-bench-shaped IDs work too)
    task_0_keys = {task_key(r) for r in cleaned if r.instance_id == "0"}
    assert len(task_0_keys) == 1


def _make_train_pool(n_tasks: int, trials_per_task: int = 1):
    from urop.data import CleanedRecord

    records = []
    for t in range(n_tasks):
        for trial in range(trials_per_task):
            records.append(CleanedRecord(
                composite_id=f"tau_bench_retail:{t}:trial_{trial}",
                instance_id=str(t),
                run_id=f"trial_{trial}",
                domain="tau_bench_retail",
                context={"system_prompt": "p", "task_description": "d", "tools": []},
                trajectory=[],
                status="resolved" if t % 2 == 0 else "failed",
                resolved=(t % 2 == 0),
                source_file="synthetic",
                source_record_hash="synthetic",
            ))
    return records


def test_sample_train_records_is_random_not_ascending():
    pool = _make_train_pool(n_tasks=50)
    sample, manifest = sample_train_records(pool, n=5, seed=42)
    ids = sorted(int(r.instance_id) for r in sample)
    assert ids != sorted(range(5)), "seeded random sample landed on ascending task ids 0-4 by chance -- re-check"
    assert manifest.n_selected == 5
    assert manifest.train_side_task_pool_size == 50
    assert manifest.seed == 42


def test_sample_train_records_deterministic_given_same_seed():
    pool = _make_train_pool(n_tasks=50)
    sample1, _ = sample_train_records(pool, n=10, seed=7)
    sample2, _ = sample_train_records(pool, n=10, seed=7)
    assert [r.instance_id for r in sample1] == [r.instance_id for r in sample2]


def test_sample_train_records_different_seed_gives_different_sample():
    pool = _make_train_pool(n_tasks=50)
    sample1, _ = sample_train_records(pool, n=10, seed=1)
    sample2, _ = sample_train_records(pool, n=10, seed=2)
    assert {r.instance_id for r in sample1} != {r.instance_id for r in sample2}


def test_sample_train_records_includes_mixed_outcomes_not_filtered():
    pool = _make_train_pool(n_tasks=50)  # alternating resolved True/False by construction
    sample, _ = sample_train_records(pool, n=20, seed=42)
    resolved_values = {r.resolved for r in sample}
    assert resolved_values == {True, False}, "sample should not be filtered to a single outcome"


def test_sample_train_records_one_trial_per_task_even_with_multiple_trials():
    pool = _make_train_pool(n_tasks=5, trials_per_task=4)
    sample, manifest = sample_train_records(pool, n=5, seed=42)
    assert len(sample) == 5
    assert len({r.instance_id for r in sample}) == 5, "must not select two trials of the same task"
    assert all(r.run_id == "trial_0" for r in sample), "must pick the lowest-numbered trial per task"
    assert manifest.train_side_task_pool_size == 5  # tasks, not trials


def test_sample_train_records_manifest_matches_selected_records():
    pool = _make_train_pool(n_tasks=20)
    sample, manifest = sample_train_records(pool, n=6, seed=3)
    manifest_ids = {(s["instance_id"], s["run_id"]) for s in manifest.selected}
    sample_ids = {(r.instance_id, r.run_id) for r in sample}
    assert manifest_ids == sample_ids


def test_sample_train_records_n_greater_than_pool_returns_whole_pool():
    pool = _make_train_pool(n_tasks=3)
    sample, manifest = sample_train_records(pool, n=100, seed=42)
    assert len(sample) == 3
    assert manifest.n_selected == 3
    assert manifest.n_requested == 100


@pytest.mark.skipif(not TAU_BENCH_CLONE.exists(), reason="external/tau-bench not vendored in this checkout")
def test_extract_retail_tool_schemas_from_real_vendored_source():
    schemas = extract_retail_tool_schemas(TAU_BENCH_CLONE)
    assert len(schemas) == 16
    names = {s["function"]["name"] for s in schemas}
    assert "cancel_pending_order" in names
    assert "get_order_details" in names
    for s in schemas:
        assert s["type"] == "function"
        assert "name" in s["function"] and "parameters" in s["function"]
