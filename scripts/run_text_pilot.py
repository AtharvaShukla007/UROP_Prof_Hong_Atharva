#!/usr/bin/env python
"""Run a bounded, budgeted, resumable text-coding dry run from a config file.

    python scripts/run_text_pilot.py --config configs/text_smoke.yaml

`backend: mock` (the only backend actually wired up here) never calls a real
model -- every output is labeled backend_label="mock" so reporting.py refuses
to treat it as research data. `backend: local` and any paid backend
(openai/deepseek/gemini) are deliberately NOT implemented in this script yet:
wiring the real open-coding loop means reusing
external/autotracegt_upstream/autotracegt's own OpenCodingAgent (per the
handoff: reuse the authors' implementation, don't reimplement the method),
which is Phase 1/2 work gated on real tau-bench data and a running endpoint
-- out of scope for this CPU-only setup task. This script fails loudly
rather than silently degrading to a fake real run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from urop.autotrace_adapter import MockCoder  # noqa: E402
from urop.budgets import Budget, BudgetExhausted, ResumeCache, bounded_llm_call, cache_key  # noqa: E402
from urop.data import load_and_clean_jsonl  # noqa: E402
from urop.manifests import build_run_manifest, save_manifest  # noqa: E402

PAID_BACKENDS = {"openai", "deepseek", "gemini"}


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run_mock_pass(cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    input_path = REPO_ROOT / cfg["input_path"]
    records, load_report = load_and_clean_jsonl(input_path, domain=cfg["domain"])
    (out_dir / "load_report.json").write_text(json.dumps(load_report.__dict__, indent=2), encoding="utf-8")

    max_traj = cfg.get("max_trajectories")
    if max_traj is not None:
        records = records[:max_traj]

    budget_cfg = cfg.get("budget", {})
    budget = Budget(
        max_seconds=float(budget_cfg.get("max_seconds", 60)),
        max_calls=budget_cfg.get("max_calls"),
    )
    resume_cache = ResumeCache(out_dir / "resume_index.json")
    coder = MockCoder(response='{"codes": [{"code": "mock-code", "span": "0-1", "quote": "mock"}]}')

    open_coding_dir = out_dir / "open_coding"
    open_coding_dir.mkdir(parents=True, exist_ok=True)

    processed, skipped, stop_reason = 0, 0, None
    for record in records:
        key = cache_key(
            model="mock-model",
            prompt_hash="mock-prompt-v1",
            settings_hash=json.dumps(cfg.get("chunk_size", 50)),
            data_hash=record.source_record_hash,
            stage="open_coding",
        )
        result_path = open_coding_dir / f"{key}.json"
        if resume_cache.is_done(key):
            skipped += 1
            continue
        try:
            result = bounded_llm_call(coder, budget=budget, messages=[{"role": "user", "content": "mock"}])
        except BudgetExhausted as exc:
            stop_reason = exc.reason
            break
        result_path.write_text(
            json.dumps({"composite_id": record.composite_id, "raw_output": result.content, "label": result.label}, indent=2),
            encoding="utf-8",
        )
        resume_cache.mark_done(key, result_path=str(result_path))
        processed += 1

    return {
        "records_seen": len(records),
        "processed": processed,
        "skipped_via_resume_cache": skipped,
        "calls_made": budget.calls_made,
        "stop_reason": stop_reason,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--i-understand-this-costs-money",
        action="store_true",
        help="Required (and still not implemented) to select a paid backend.",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    backend = cfg.get("backend", "mock")

    if backend in PAID_BACKENDS:
        print(
            f"[run_text_pilot] backend={backend!r} is a paid endpoint. This script does not "
            "call any paid API -- default paid budget is zero and this path is not implemented. "
            "Refusing to run.",
            file=sys.stderr,
        )
        sys.exit(2)

    if backend == "local":
        print(
            "[run_text_pilot] backend='local' (a real local OpenAI-compatible server) is not "
            "implemented in this script yet -- wiring the real open-coding loop belongs to "
            "Phase 1/2 and needs external/autotracegt_upstream's OpenCodingAgent, real tau-bench "
            "data, and a running endpoint. Use backend='mock' for the CPU dry run.",
            file=sys.stderr,
        )
        sys.exit(2)

    if backend != "mock":
        print(f"[run_text_pilot] unknown backend {backend!r}", file=sys.stderr)
        sys.exit(2)

    out_dir = args.output_dir or (REPO_ROOT / "runs" / cfg["stage"] / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_mock_pass(cfg, out_dir)
    print(f"[run_text_pilot] {json.dumps(result)}")

    run_manifest = build_run_manifest(
        run_id=out_dir.name,
        stage=cfg["stage"],
        config=cfg,
        input_files=[args.config],
        backend_label="mock",
    )
    run_manifest.stop_reason = result["stop_reason"]
    save_manifest(run_manifest, out_dir / "manifest.json")
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[run_text_pilot] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
