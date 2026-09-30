#!/usr/bin/env python
"""Print a run summary from an existing run directory's manifest.json.

    python scripts/summarize_run.py --run-dir runs/text_smoke/<timestamp>

Refuses (via urop.reporting.summarize_run) to attach metrics to a run whose
backend_label marks it mock/synthetic -- there are no metrics to attach yet
in this repo anyway (no probe stage has run), so today this only prints
counts/stop-reason bookkeeping. Kept as a separate script because Phase 4
(probe AUROC) will pass --metrics-file here once it exists, and the refusal
must already be in place before that day.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from urop.reporting import NotResearchDataError, summarize_run  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--metrics-file", type=Path, default=None, help="Optional JSON {metric_name: value}.")
    args = p.parse_args()

    manifest_path = args.run_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[summarize_run] no manifest.json under {args.run_dir}", file=sys.stderr)
        sys.exit(1)

    metrics = None
    if args.metrics_file:
        metrics = json.loads(args.metrics_file.read_text(encoding="utf-8"))

    try:
        summary = summarize_run(manifest_path, metrics=metrics)
    except NotResearchDataError as exc:
        print(f"[summarize_run] REFUSED: {exc}", file=sys.stderr)
        sys.exit(3)

    result_path = args.run_dir / "result.json"
    if result_path.exists():
        summary.counts = json.loads(result_path.read_text(encoding="utf-8"))

    print(json.dumps(summary.__dict__, indent=2))


if __name__ == "__main__":
    main()
