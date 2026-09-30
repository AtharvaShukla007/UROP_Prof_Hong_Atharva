#!/usr/bin/env python
"""Convert real tau-bench historical retail trajectories into our raw JSONL
schema, then select smoke/pilot samples from the training side of a
task-grouped split by seeded random sampling over train task IDs. CPU-only,
offline (reads local files only).

    python scripts/convert_tau_bench.py

Reads data/tau_bench_raw/gpt-4o-retail.json (git-ignored, see its
PROVENANCE.md), writes under data/tau_bench/ (git-ignored):
    tau_bench_retail_gpt4o_all.jsonl   -- every valid converted record
    split_manifest.json                -- deterministic task-grouped split (seed 42, 80/20)
    tau_bench_train_3.jsonl            -- 3 randomly sampled train-side trajectories
    tau_bench_train_30.jsonl           -- <=30 randomly sampled train-side trajectories
    smoke_sample_manifest.json         -- the exact task/trial IDs selected for the smoke sample, + seed
    pilot_sample_manifest.json         -- same, for the pilot sample
    conversion_report.json             -- counts, exclusions, token-length stats

Selection: `random.Random(sample_seed).sample()` over the set of train-side
task IDs (one trial per task, the lowest-numbered available), never filtered
or reordered by outcome/category. This is NOT the same claim as "ascending
task-id order is unbiased" -- ascending order is an arbitrary, non-random
ordering and earlier revisions of this script mislabeled it as unbiased;
only an actual seeded random draw over the full train-side task-ID pool
supports that claim, and even then only with respect to outcome (a random
sample from ID order is not automatically representative of anything else
about the corpus). See docs/methods_and_deviations.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from urop.data import load_and_clean_jsonl  # noqa: E402
from urop.splits import assign_split, make_split, save_manifest as save_split_manifest, verify_no_overlap  # noqa: E402
from urop.tau_bench_adapter import (  # noqa: E402
    convert_tau_bench_corpus,
    extract_retail_tool_schemas,
    load_tau_bench_raw_array,
    sample_train_records,
)

DEFAULT_RAW = REPO_ROOT / "data" / "tau_bench_raw" / "gpt-4o-retail.json"
DEFAULT_TAU_BENCH_CLONE = REPO_ROOT / "external" / "tau-bench"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "tau_bench"


def approx_token_count(text: str) -> int:
    """tiktoken cl100k_base -- INFORMATIONAL ONLY. This is a GPT-family
    tokenizer, not the Qwen tokenizer the candidate agent model actually
    uses, and it tokenizes raw message content, not the exact rendered
    open-coding prompt (system + user template + tools listing + trajectory
    text, as `agents.open_coding.format_chunk`/`load_prompt` actually build
    it). The real, authoritative context-limit enforcement happens in
    `notebooks/01_kaggle_text_pilot.ipynb` using the real Qwen tokenizer
    against the real rendered prompt (see its Section 5) -- these numbers
    are only a rough, offline sanity check for repo work done without a GPU.
    """
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text, disallowed_special=()))


def trajectory_token_stats(records: list[dict]) -> dict:
    per_record_tokens = []
    for r in records:
        text = "\n".join(str(m.get("content") or "") for m in r["messages"])
        per_record_tokens.append(approx_token_count(text))
    per_record_tokens.sort()
    n = len(per_record_tokens)
    return {
        "n": n,
        "min": per_record_tokens[0] if n else 0,
        "max": per_record_tokens[-1] if n else 0,
        "median": per_record_tokens[n // 2] if n else 0,
        "mean": round(sum(per_record_tokens) / n, 1) if n else 0,
        "over_8192": sum(1 for t in per_record_tokens if t > 8192),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    p.add_argument("--tau-bench-clone", type=Path, default=DEFAULT_TAU_BENCH_CLONE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--seed", type=int, default=42, help="Seed for the task-grouped train/test split.")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--smoke-n", type=int, default=3)
    p.add_argument("--pilot-n", type=int, default=30)
    p.add_argument(
        "--sample-seed", type=int, default=42,
        help="Seed for the random draw of train-side task IDs into the smoke/pilot samples "
             "(independent of --seed, which only controls the train/test split).",
    )
    args = p.parse_args()

    if not args.raw.exists():
        print(f"[convert_tau_bench] raw file not found: {args.raw}", file=sys.stderr)
        print("See data/tau_bench_raw/PROVENANCE.md for the exact download command.", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    tool_schemas = extract_retail_tool_schemas(args.tau_bench_clone)
    print(f"[convert_tau_bench] extracted {len(tool_schemas)} real retail tool schemas "
          f"from {args.tau_bench_clone}")

    raw_records = load_tau_bench_raw_array(args.raw)
    converted, report = convert_tau_bench_corpus(raw_records, tool_schemas=tool_schemas)
    print(f"[convert_tau_bench] {report.converted} converted / {report.rejected} rejected "
          f"of {report.total_raw} raw records")
    print(f"[convert_tau_bench] {len(report.distinct_task_ids)} distinct tasks; "
          f"reward=1.0: {report.reward_1_count}  reward=0.0: {report.reward_0_count}")

    all_path = args.out_dir / "tau_bench_retail_gpt4o_all.jsonl"
    with open(all_path, "w", encoding="utf-8") as f:
        for rec in converted:
            f.write(json.dumps(rec) + "\n")
    print(f"[convert_tau_bench] wrote {len(converted)} records to {all_path}")

    cleaned, load_report = load_and_clean_jsonl(all_path, domain="tau_bench_retail")
    if load_report.rejected:
        print(f"[convert_tau_bench] WARNING: {load_report.rejected} records failed our own raw-schema "
              "validator after conversion -- this should not happen; see load_report below.", file=sys.stderr)

    split_manifest = make_split(cleaned, seed=args.seed, train_frac=args.train_frac)
    verify_no_overlap(split_manifest)
    save_split_manifest(split_manifest, args.out_dir / "split_manifest.json")
    assigned = assign_split(cleaned, split_manifest)
    print(f"[convert_tau_bench] split: {len(split_manifest.train_task_keys)} train tasks / "
          f"{len(split_manifest.test_task_keys)} test tasks "
          f"({len(assigned['train'])} / {len(assigned['test'])} trajectories)")

    def write_raw_sample(records: list, filename: str) -> None:
        path = args.out_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                messages = [
                    {"role": "system", "content": r.context["system_prompt"]},
                    {"role": "user", "content": r.context["task_description"]},
                    *r.trajectory,
                ]
                f.write(json.dumps({
                    "instance_id": r.instance_id,
                    "run_id": r.run_id,
                    "messages": messages,
                    "tools": r.context["tools"],
                    "resolved": r.resolved,
                }) + "\n")

    smoke_sample, smoke_manifest = sample_train_records(assigned["train"], n=args.smoke_n, seed=args.sample_seed)
    write_raw_sample(smoke_sample, f"tau_bench_train_{args.smoke_n}.jsonl")
    (args.out_dir / "smoke_sample_manifest.json").write_text(
        json.dumps(smoke_manifest.to_json_dict(), indent=2), encoding="utf-8"
    )

    pilot_sample, pilot_manifest = sample_train_records(assigned["train"], n=args.pilot_n, seed=args.sample_seed)
    pilot_n = pilot_manifest.n_selected
    write_raw_sample(pilot_sample, f"tau_bench_train_{pilot_n}.jsonl")
    (args.out_dir / "pilot_sample_manifest.json").write_text(
        json.dumps(pilot_manifest.to_json_dict(), indent=2), encoding="utf-8"
    )

    print(f"[convert_tau_bench] wrote smoke sample: {len(smoke_sample)} trajectories "
          f"(seed={args.sample_seed}, task_ids {sorted((r.instance_id for r in smoke_sample), key=int)})")
    print(f"[convert_tau_bench] wrote pilot sample: {len(pilot_sample)} trajectories "
          f"(seed={args.sample_seed}, {pilot_n} requested {args.pilot_n}, "
          f"{smoke_manifest.train_side_task_pool_size} available on the train side)")

    report_dict = report.to_json_dict()
    report_dict["split"] = {
        "train_tasks": len(split_manifest.train_task_keys),
        "test_tasks": len(split_manifest.test_task_keys),
        "train_trajectories": len(assigned["train"]),
        "test_trajectories": len(assigned["test"]),
    }
    report_dict["token_stats_all_converted"] = trajectory_token_stats(converted)
    report_dict["token_stats_smoke_sample"] = trajectory_token_stats(
        [{"messages": [{"role": "system", "content": r.context["system_prompt"]},
                        {"role": "user", "content": r.context["task_description"]}, *r.trajectory]}
         for r in smoke_sample]
    )
    (args.out_dir / "conversion_report.json").write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    print(f"[convert_tau_bench] wrote {args.out_dir / 'conversion_report.json'}")


if __name__ == "__main__":
    main()
