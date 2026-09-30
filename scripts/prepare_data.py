#!/usr/bin/env python
"""Load, validate, clean, and split a raw trajectory JSONL file. CPU-only, offline.

Defaults to the synthetic test fixture so this runs with zero setup:

    python scripts/prepare_data.py
    python scripts/prepare_data.py --input data/tau_bench_train_1000.jsonl --domain tau_bench

Writes, under --output-dir (default runs/prepare_data/<timestamp>/):
  load_report.json   -- accept/reject/duplicate counts and every rejection reason
  split_manifest.json
  manifest.json       -- provenance (input file hash, config hash, backend_label="n/a")

Never writes into --input; that file is read-only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from urop.data import load_and_clean_jsonl  # noqa: E402
from urop.manifests import build_run_manifest, save_manifest  # noqa: E402
from urop.splits import make_split, save_manifest as save_split_manifest, verify_no_overlap  # noqa: E402

DEFAULT_INPUT = REPO_ROOT / "tests" / "fixtures" / "tiny_trajectories.jsonl"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--domain", default="synthetic")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    if not args.input.exists():
        print(f"[prepare_data] input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.output_dir or (REPO_ROOT / "runs" / "prepare_data" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    out_dir.mkdir(parents=True, exist_ok=True)

    records, report = load_and_clean_jsonl(args.input, domain=args.domain)
    (out_dir / "load_report.json").write_text(json.dumps(report.__dict__, indent=2), encoding="utf-8")
    print(f"[prepare_data] {report.accepted} accepted / {report.rejected} rejected / "
          f"{report.duplicate_composite_ids} duplicate, out of {report.total_lines} lines")

    if not records:
        print("[prepare_data] no valid records -- nothing to split", file=sys.stderr)
        sys.exit(1)

    split_manifest = make_split(records, seed=args.seed, train_frac=args.train_frac)
    verify_no_overlap(split_manifest)
    save_split_manifest(split_manifest, out_dir / "split_manifest.json")
    print(f"[prepare_data] split: {len(split_manifest.train_task_keys)} train tasks / "
          f"{len(split_manifest.test_task_keys)} test tasks")

    run_manifest = build_run_manifest(
        run_id=out_dir.name,
        stage="prepare_data",
        config={"domain": args.domain, "seed": args.seed, "train_frac": args.train_frac, "input": str(args.input)},
        input_files=[args.input],
        backend_label="n/a",
    )
    save_manifest(run_manifest, out_dir / "manifest.json")
    print(f"[prepare_data] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
