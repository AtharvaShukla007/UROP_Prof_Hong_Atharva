"""CPU-only checks for src/urop/data.py and src/urop/splits.py.

Covers handoff §8 acceptance checks 1-3:
  1. Missing outcome rejected; malformed messages rejected; tool calls preserved.
  2. No task overlap between train and test, including multiple trials.
  3. Same seeds/config produce identical split manifests.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from urop.data import load_and_clean_jsonl
from urop.splits import assign_split, make_split, task_key, verify_no_overlap

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_trajectories.jsonl"


@pytest.fixture(scope="module")
def loaded():
    return load_and_clean_jsonl(FIXTURE, domain="synthetic")


def test_load_report_counts(loaded):
    records, report = loaded
    assert report.total_lines == 8
    assert report.accepted == 5
    assert report.rejected == 2
    assert report.duplicate_composite_ids == 1
    assert len(records) == 5


def test_missing_outcome_is_rejected(loaded):
    _, report = loaded
    assert any("missing 'resolved'" in e for e in report.errors)


def test_malformed_messages_are_rejected(loaded):
    _, report = loaded
    assert any("expected role" in e for e in report.errors)


def test_tool_calls_are_preserved_verbatim(loaded):
    records, _ = loaded
    rec = next(r for r in records if r.instance_id == "synth_task_001" and r.run_id == "run_1")
    assistant_step = next(m for m in rec.trajectory if m.get("role") == "assistant" and "tool_calls" in m)
    assert assistant_step["tool_calls"][0]["function"]["name"] == "get_order"
    tool_step = next(m for m in rec.trajectory if m.get("role") == "tool")
    assert tool_step["tool_call_id"] == "call_1"


def test_status_only_from_explicit_boolean(loaded):
    records, _ = loaded
    by_id = {(r.instance_id, r.run_id): r for r in records}
    assert by_id[("synth_task_001", "run_1")].status == "resolved"
    assert by_id[("synth_task_001", "run_2")].status == "failed"


def test_no_task_overlap_with_multi_trial_task(loaded):
    records, _ = loaded
    manifest = make_split(records, seed=42, train_frac=0.8)
    verify_no_overlap(manifest)  # must not raise

    assigned = assign_split(records, manifest)
    trial_1, trial_2 = (
        next(r for r in records if r.instance_id == "synth_task_001" and r.run_id == "run_1"),
        next(r for r in records if r.instance_id == "synth_task_001" and r.run_id == "run_2"),
    )
    side_1 = "train" if trial_1 in assigned["train"] else "test"
    side_2 = "train" if trial_2 in assigned["train"] else "test"
    assert side_1 == side_2, "both trials of one task must land on the same split side"


def test_same_seed_same_config_gives_identical_manifest(loaded):
    records, _ = loaded
    m1 = make_split(records, seed=42, train_frac=0.8)
    m2 = make_split(records, seed=42, train_frac=0.8)
    assert m1.config_hash == m2.config_hash
    assert m1.train_task_keys == m2.train_task_keys
    assert m1.test_task_keys == m2.test_task_keys


def test_same_seed_independent_of_input_order(loaded):
    records, _ = loaded
    m1 = make_split(records, seed=42, train_frac=0.8)
    m2 = make_split(list(reversed(records)), seed=42, train_frac=0.8)
    assert m1.train_task_keys == m2.train_task_keys
    assert m1.test_task_keys == m2.test_task_keys


def test_different_seed_can_change_split(loaded):
    records, _ = loaded
    m1 = make_split(records, seed=42, train_frac=0.8)
    m2 = make_split(records, seed=1, train_frac=0.8)
    assert m1.config_hash != m2.config_hash


def test_task_key_groups_by_domain_and_instance(loaded):
    records, _ = loaded
    rec = records[0]
    assert task_key(rec) == f"{rec.domain}:{rec.instance_id}"
