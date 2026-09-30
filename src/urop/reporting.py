"""Summaries built only from what actually exists on disk for a run.

Never computes or reports a metric (especially AUROC) attached to a run
whose backend_label marks it as mock/synthetic, and never lets a synthetic
fixture run masquerade as research data (handoff §8, acceptance check 7).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MOCK_LABELS = {"mock", "synthetic"}
REAL_LABELS = {"local", "openai", "deepseek", "gemini"}

AUTOMATED_EVIDENCE_DISCLAIMER = (
    "Every count below comes from automated parsing/schema/quote checks on the model's own "
    "raw output. None of it is human-validated coding quality -- valid JSON, a schema match, "
    "and a verbatim quote only mean the output is STRUCTURALLY trustworthy enough to read, not "
    "that the behavioral codes it contains are correct. No manual annotation exists for this "
    "project (see CLAUDE.md); do not describe these counts as accuracy or as validated labels."
)


@dataclass
class RunOutcomeReport:
    """Explicit, separated success/failure counts for one open-coding smoke
    run -- built so that "the loop finished" (`stop_reason`) can never be
    read as "the run produced usable output" (`usable_outputs`). A run can
    have `stop_reason == "completed"` and `usable_outputs == 0`
    simultaneously; `outcome_label` names that combination explicitly
    instead of leaving a reader to infer it from `stop_reason` alone (the
    real 2026-09-29 Kaggle run did exactly this -- see
    docs/experiment_log.md -- and `stop_reason="completed"` alone made it
    look like a successful smoke test until the per-record counts were
    actually read).
    """

    run_id: str
    server_startup_success: bool
    execution_completed: bool  # True iff the loop reached its natural end (stop_reason == "completed"), not budget-cut/errored
    records_loaded: int
    real_attempts_made: int
    context_limit_blocks: int  # requests the token guard refused to send -- NOT counted in real_attempts_made
    parse_or_schema_failures: int  # classify_trajectory_coding == "parse_failed"
    length_limited_responses: int  # finish_reason == "length" seen in the evidence log -- a confirmed fact, not a guess
    quote_or_step_validation_failures: int  # sum of invalid_codes across all quote_validation_reports
    structurally_valid_outputs: int  # classify_trajectory_coding == "valid" -- JSON/schema parsed, nothing more claimed
    outputs_passing_automated_evidence_checks: int  # structurally valid AND every code's quote verified verbatim
    empty_code_outputs: int  # Valid structure but no behaviour labels.
    usable_outputs: int  # Nonempty and passes automated evidence checks; named again at top level so it can't be missed
    outcome_label: str
    disclaimer: str = AUTOMATED_EVIDENCE_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def _outcome_label(
    *, server_attempted: bool, server_startup_success: bool, execution_completed: bool, usable_outputs: int
) -> str:
    if not server_attempted:
        return "not_run"  # RUN_SERVER was False -- a deliberate no-op, NOT a failure; must not read as one
    if not server_startup_success:
        return "server_startup_failed"
    if usable_outputs > 0:
        return "completed_with_usable_outputs" if execution_completed else "stopped_early_with_usable_outputs"
    return "completed_with_zero_usable_outputs" if execution_completed else "stopped_early_with_zero_usable_outputs"


def build_outcome_report(
    *,
    run_id: str,
    server_startup_success: bool,
    stop_reason: str | None,
    records_loaded: int,
    real_attempts_made: int,
    context_limit_blocks: int,
    coded_results_by_composite_id: dict[str, dict],
    quote_validation_reports: dict[str, Any],
    length_limited_response_count: int = 0,
    server_attempted: bool = True,
    failed_record_ids: set[str] | None = None,
) -> RunOutcomeReport:
    """Build a `RunOutcomeReport` from the raw pieces a coding run already
    tracks. `coded_results_by_composite_id` maps each attempted record's
    composite_id (the SAME key `quote_validation_reports` uses -- e.g.
    `data.CleanedRecord.composite_id`, `"{domain}:{instance_id}:{run_id}"`)
    to its `TrajectoryCoding.to_dict()`-shaped result. Classification is
    recomputed here rather than trusted from the caller, so this function
    is the single place that decides what counts as "structurally valid"
    vs "passing automated checks". `quote_validation_reports` maps the same
    composite_id -> `autotrace_adapter.CodeValidationReport`-like objects
    with `.valid_codes`/`.invalid_codes`/`.total_codes`.
    """
    from urop.autotrace_adapter import classify_trajectory_coding

    execution_completed = stop_reason == "completed"

    failed_ids = set(failed_record_ids or ())
    parse_or_schema_failures = 0
    structurally_valid_ids: set[str] = set()
    for composite_id, d in coded_results_by_composite_id.items():
        if classify_trajectory_coding(d) == "parse_failed":
            failed_ids.add(composite_id)
        else:
            structurally_valid_ids.add(composite_id)
    parse_or_schema_failures = len(failed_ids)
    structurally_valid_outputs = len(structurally_valid_ids)

    quote_or_step_validation_failures = sum(len(v.invalid_codes) for v in quote_validation_reports.values())
    empty_code_outputs = sum(
        not any(cc.get("codes") for cc in coded_results_by_composite_id[cid].get("chunk_codings", []))
        for cid in structurally_valid_ids
    )
    # Empty results remain structurally valid, but cannot provide an evidence example.
    # "passes automated evidence checks" = structurally valid AND has a quote-validation
    # report recorded (i.e. was actually validated) AND that report shows zero invalid codes.
    outputs_passing_automated_evidence_checks = sum(
        1 for composite_id in structurally_valid_ids
        if composite_id in quote_validation_reports
        and quote_validation_reports[composite_id].total_codes > 0
        and quote_validation_reports[composite_id].valid_codes == quote_validation_reports[composite_id].total_codes
        and quote_validation_reports[composite_id].invalid_codes == []
    )

    usable_outputs = outputs_passing_automated_evidence_checks
    outcome_label = _outcome_label(
        server_attempted=server_attempted,
        server_startup_success=server_startup_success,
        execution_completed=execution_completed,
        usable_outputs=usable_outputs,
    )

    return RunOutcomeReport(
        run_id=run_id,
        server_startup_success=server_startup_success,
        execution_completed=execution_completed,
        records_loaded=records_loaded,
        real_attempts_made=real_attempts_made,
        context_limit_blocks=context_limit_blocks,
        parse_or_schema_failures=parse_or_schema_failures,
        length_limited_responses=length_limited_response_count,
        quote_or_step_validation_failures=quote_or_step_validation_failures,
        structurally_valid_outputs=structurally_valid_outputs,
        outputs_passing_automated_evidence_checks=outputs_passing_automated_evidence_checks,
        usable_outputs=usable_outputs,
        empty_code_outputs=empty_code_outputs,
        outcome_label=outcome_label,
    )


class NotResearchDataError(RuntimeError):
    pass


@dataclass
class RunSummary:
    run_id: str
    backend_label: str
    stage: str
    counts: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None
    metrics: dict[str, float] | None = None
    warnings: list[str] = field(default_factory=list)


def summarize_run(manifest_path: str | Path, *, metrics: dict[str, float] | None = None) -> RunSummary:
    manifest: dict[str, Any] = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    backend_label = manifest.get("backend_label", "unset")

    summary = RunSummary(
        run_id=manifest["run_id"],
        backend_label=backend_label,
        stage=manifest["stage"],
        stop_reason=manifest.get("stop_reason"),
    )

    if metrics:
        if backend_label in MOCK_LABELS:
            raise NotResearchDataError(
                f"refusing to attach metrics {sorted(metrics)} to a run labeled "
                f"backend_label={backend_label!r} -- mock/synthetic runs are engineering "
                "checks, not research results."
            )
        if backend_label not in REAL_LABELS:
            summary.warnings.append(
                f"backend_label={backend_label!r} is not a recognized real-inference label "
                f"({sorted(REAL_LABELS)}); metrics attached but flagged as unverified."
            )
        summary.metrics = metrics

    return summary
