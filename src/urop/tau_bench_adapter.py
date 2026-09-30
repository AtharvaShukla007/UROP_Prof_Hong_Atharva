"""Convert tau-bench's raw historical-trajectory schema into the raw record
shape src/urop/data.py expects (instance_id, run_id, messages, tools, resolved).

tau-bench's own schema (verified against a real download -- see
data/tau_bench_raw/PROVENANCE.md and docs/source_audit.md) is NOT the schema
AutoTraceGT/src/urop/data.py expects:

    tau-bench raw:  {"task_id": int, "reward": float, "trial": int,
                      "traj": [{role, content, [tool_calls]}, ...], "info": {...}}
    our/AutoTraceGT: {"instance_id": str, "run_id": str, "messages": [...],
                       "tools": [...], "resolved": bool}

This module only does that mapping. It never touches src/urop/data.py --
its output is exactly data.py's expected *input* shape, so the existing,
already-CPU-tested validate/clean/split pipeline is reused unchanged on real
data once this conversion has run.
"""
from __future__ import annotations

import ast
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from urop.data import CleanedRecord
from urop.splits import task_key


def extract_retail_tool_schemas(tau_bench_repo_root: Path) -> list[dict[str, Any]]:
    """Statically read get_info() from every tau_bench/envs/retail/tools/*.py
    file via AST, instead of importing the tau_bench package (whose top-level
    __init__ pulls in openai/anthropic/mistralai/google-generativeai/litellm
    just to read a handful of static dicts).

    Faithful to source: parses the literal dict each file's get_info()
    returns via ast.literal_eval. Does not hand-transcribe or invent schema.
    """
    tools_dir = tau_bench_repo_root / "tau_bench" / "envs" / "retail" / "tools"
    if not tools_dir.exists():
        raise FileNotFoundError(
            f"{tools_dir} not found -- clone tau-bench into external/tau-bench first "
            "(see external/PROVENANCE.md)."
        )

    schemas = []
    for py_file in sorted(tools_dir.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        found = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "get_info":
                for stmt in node.body:
                    if isinstance(stmt, ast.Return):
                        found = ast.literal_eval(stmt.value)
        if found is None:
            raise RuntimeError(f"no get_info() return statement found in {py_file}")
        schemas.append(found)
    return schemas


def validate_tau_bench_record(record: dict, *, index: int) -> list[str]:
    """Return validation error strings; an empty list means the record is usable.

    Rejects (does not coerce) a missing/invalid task_id, trial, or reward --
    same "reject rather than silently mark as failure" rule as
    src/urop/data.py applies to AutoTraceGT's own `resolved` field.
    """
    errors: list[str] = []
    task_id = record.get("task_id")
    trial = record.get("trial")
    reward = record.get("reward")
    traj = record.get("traj")

    if not isinstance(task_id, int):
        errors.append(f"record {index}: missing/invalid task_id")
    if not isinstance(trial, int):
        errors.append(f"record {index} (task_id={task_id!r}): missing/invalid trial")
    if reward is None:
        errors.append(f"record {index} (task_id={task_id!r}): missing 'reward' outcome field")
    elif isinstance(reward, bool) or not isinstance(reward, (int, float)) or float(reward) not in (0.0, 1.0):
        errors.append(
            f"record {index} (task_id={task_id!r}): 'reward' is {reward!r}, expected exactly 0.0 or 1.0"
        )

    if not isinstance(traj, list) or len(traj) < 3:
        errors.append(f"record {index} (task_id={task_id!r}): 'traj' must be a list of >= 3 messages")
        return errors

    if not isinstance(traj[0], dict) or traj[0].get("role") != "system":
        errors.append(f"record {index} (task_id={task_id!r}): traj[0] expected role 'system'")
    if not isinstance(traj[1], dict) or traj[1].get("role") != "user":
        errors.append(f"record {index} (task_id={task_id!r}): traj[1] expected role 'user'")
    for i, msg in enumerate(traj[2:], start=2):
        if not isinstance(msg, dict) or "role" not in msg:
            errors.append(f"record {index} (task_id={task_id!r}): traj[{i}] is not a valid message dict")

    return errors


def convert_tau_bench_record(record: dict, *, tool_schemas: list[dict[str, Any]]) -> dict:
    """Caller must have already confirmed validate_tau_bench_record(record) == []."""
    return {
        "instance_id": str(record["task_id"]),
        "run_id": f"trial_{record['trial']}",
        "messages": record["traj"],
        "tools": tool_schemas,
        "resolved": float(record["reward"]) == 1.0,
    }


@dataclass
class TauBenchConversionReport:
    total_raw: int = 0
    converted: int = 0
    rejected: int = 0
    reward_1_count: int = 0
    reward_0_count: int = 0
    distinct_task_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "total_raw": self.total_raw,
            "converted": self.converted,
            "rejected": self.rejected,
            "reward_1_count": self.reward_1_count,
            "reward_0_count": self.reward_0_count,
            "distinct_task_count": len(self.distinct_task_ids),
            "errors": self.errors,
        }


def convert_tau_bench_corpus(
    raw_records: list[dict], *, tool_schemas: list[dict[str, Any]]
) -> tuple[list[dict], TauBenchConversionReport]:
    report = TauBenchConversionReport(total_raw=len(raw_records))
    converted: list[dict] = []
    seen_task_ids: set[str] = set()

    for i, raw in enumerate(raw_records):
        errors = validate_tau_bench_record(raw, index=i)
        if errors:
            report.rejected += 1
            report.errors.extend(errors)
            continue

        out = convert_tau_bench_record(raw, tool_schemas=tool_schemas)
        converted.append(out)
        report.converted += 1
        seen_task_ids.add(out["instance_id"])
        if out["resolved"]:
            report.reward_1_count += 1
        else:
            report.reward_0_count += 1

    report.distinct_task_ids = sorted(seen_task_ids, key=lambda s: int(s) if s.isdigit() else s)
    return converted, report


def load_tau_bench_raw_array(path: str | Path) -> list[dict]:
    """tau-bench's historical_trajectories files are a single JSON array, not
    JSONL -- distinct from the JSONL shape src/urop/data.py reads."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a top-level JSON array, got {type(data).__name__}")
    return data


@dataclass
class SampleManifest:
    method: str
    seed: int
    n_requested: int
    n_selected: int
    train_side_task_pool_size: int
    selected: list[dict[str, Any]]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "seed": self.seed,
            "n_requested": self.n_requested,
            "n_selected": self.n_selected,
            "train_side_task_pool_size": self.train_side_task_pool_size,
            "selected": self.selected,
        }


def sample_train_records(
    train_records: list[CleanedRecord], *, n: int, seed: int
) -> tuple[list[CleanedRecord], SampleManifest]:
    """Seeded random.Random(seed).sample() over the set of train-side task
    keys, one trial (the lowest-numbered run_id) per task, no filtering by
    outcome/category. Returns the sampled records plus a manifest recording
    exactly which (instance_id, run_id) pairs were selected and how.

    This is NOT the same claim as "sorted/ascending order is unbiased" --
    only an actual seeded random draw over the full train-side task pool
    supports an unbiased-selection claim, and even then only with respect
    to outcome (a random sample of tasks is not automatically
    representative of anything else about the corpus, e.g. task length or
    category).
    """
    by_task: dict[str, list[CleanedRecord]] = {}
    for r in train_records:
        by_task.setdefault(task_key(r), []).append(r)

    def trial_num(run_id: str) -> int:
        suffix = run_id.rsplit("_", 1)[-1]
        return int(suffix) if suffix.isdigit() else 0

    pool_by_task = {k: sorted(trials, key=lambda r: trial_num(r.run_id))[0] for k, trials in by_task.items()}
    task_keys_sorted = sorted(pool_by_task)  # deterministic input order for the RNG, regardless of arg order

    rng = random.Random(seed)
    k = min(n, len(task_keys_sorted))
    chosen = rng.sample(task_keys_sorted, k)
    selected = [pool_by_task[tk] for tk in chosen]

    manifest = SampleManifest(
        method="random.Random(seed).sample() over all train-side task keys, one trial "
               "(lowest-numbered) per task, no filtering by outcome/category",
        seed=seed,
        n_requested=n,
        n_selected=len(selected),
        train_side_task_pool_size=len(task_keys_sorted),
        selected=[
            {"instance_id": r.instance_id, "run_id": r.run_id, "resolved": r.resolved} for r in selected
        ],
    )
    return selected, manifest
