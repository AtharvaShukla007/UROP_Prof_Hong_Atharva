"""Raw trajectory loading, validation, and cleaning.

Deliberately stricter than external/autotracegt_upstream's
utils/trajectory_processing.py: that code treats a missing `resolved` field
as a failure label (`"resolved" if record.get("resolved") else "failed"`).
We treat a missing/invalid outcome as an invalid record instead -- a missing
label is not evidence of failure (see docs/methods_and_deviations.md).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REQUIRED_ROLES_PREFIX = ("system", "user")


@dataclass
class CleanedRecord:
    composite_id: str
    instance_id: str
    run_id: str
    domain: str
    context: dict[str, Any]
    trajectory: list[dict]
    status: str
    resolved: bool
    source_file: str
    source_record_hash: str


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_raw_record(record: dict, *, line_no: int) -> list[str]:
    """Return validation error strings; an empty list means the record is usable."""
    errors: list[str] = []
    instance_id = record.get("instance_id")
    run_id = record.get("run_id")
    if not instance_id or not isinstance(instance_id, str):
        errors.append(f"line {line_no}: missing/invalid instance_id")
    if not run_id or not isinstance(run_id, str):
        errors.append(f"line {line_no}: missing/invalid run_id")

    if "resolved" not in record:
        errors.append(f"line {line_no} ({instance_id}/{run_id}): missing 'resolved' outcome field")
    elif not isinstance(record["resolved"], bool):
        errors.append(
            f"line {line_no} ({instance_id}/{run_id}): 'resolved' is "
            f"{type(record['resolved']).__name__}, not bool"
        )

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        errors.append(
            f"line {line_no} ({instance_id}/{run_id}): 'messages' must be a list of >= 3 "
            "entries (system, user, and at least one trajectory step)"
        )
        return errors

    for i, expected_role in enumerate(REQUIRED_ROLES_PREFIX):
        msg = messages[i]
        role = msg.get("role") if isinstance(msg, dict) else None
        if role != expected_role:
            errors.append(
                f"line {line_no} ({instance_id}/{run_id}): messages[{i}] expected role "
                f"'{expected_role}', got {role!r}"
            )

    for i, msg in enumerate(messages[2:], start=2):
        if not isinstance(msg, dict) or "role" not in msg:
            errors.append(f"line {line_no} ({instance_id}/{run_id}): messages[{i}] is not a valid message dict")
            continue
        if msg["role"] == "assistant" and "tool_calls" in msg and not isinstance(msg["tool_calls"], list):
            errors.append(f"line {line_no} ({instance_id}/{run_id}): messages[{i}].tool_calls must be a list")

    return errors


def clean_record(record: dict, *, domain: str, source_file: str, source_record_hash: str) -> CleanedRecord:
    """Caller must have already confirmed validate_raw_record(record) == []."""
    messages = record["messages"]
    instance_id = record["instance_id"]
    run_id = record["run_id"]
    resolved = record["resolved"]
    return CleanedRecord(
        composite_id=f"{domain}:{instance_id}:{run_id}",
        instance_id=instance_id,
        run_id=run_id,
        domain=domain,
        context={
            "system_prompt": messages[0].get("content", ""),
            "task_description": messages[1].get("content", ""),
            "tools": record.get("tools") or [],
        },
        trajectory=messages[2:],
        status="resolved" if resolved else "failed",
        resolved=resolved,
        source_file=source_file,
        source_record_hash=source_record_hash,
    )


@dataclass
class LoadReport:
    total_lines: int = 0
    accepted: int = 0
    rejected: int = 0
    duplicate_composite_ids: int = 0
    errors: list[str] = field(default_factory=list)


def load_and_clean_jsonl(path: str | Path, *, domain: str) -> tuple[list[CleanedRecord], LoadReport]:
    """Load one raw JSONL file, validate every line, clean what passes.

    Never mutates or overwrites `path`. Rejects rather than silently coerces
    missing/invalid outcomes and malformed messages (see module docstring).
    Exact-duplicate (composite_id) records are dropped after the first
    occurrence and counted in the report, not silently merged.
    """
    path = Path(path)
    report = LoadReport()
    records: list[CleanedRecord] = []
    seen_ids: set[str] = set()

    with open(path, encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            report.total_lines += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                report.rejected += 1
                report.errors.append(f"line {line_no}: invalid JSON ({exc})")
                continue

            errors = validate_raw_record(raw, line_no=line_no)
            if errors:
                report.rejected += 1
                report.errors.extend(errors)
                continue

            cleaned = clean_record(
                raw, domain=domain, source_file=str(path), source_record_hash=_sha256_text(line)
            )
            if cleaned.composite_id in seen_ids:
                report.duplicate_composite_ids += 1
                continue
            seen_ids.add(cleaned.composite_id)
            records.append(cleaned)
            report.accepted += 1

    return records, report
