#!/usr/bin/env python
"""Phase 0 source-audit tool: report on what is actually present, offline.

CPU-only, no network, no model calls. Checks:
  - the four reference_materials/ files the handoff expects, with sha256
  - whether external/autotracegt_upstream/ has been extracted, and what's in it
  - the upstream package's declared saturation threshold / DATASETS keys, read
    directly from its source (not from memory)

Run:
    python scripts/inspect_sources.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from urop.manifests import sha256_file  # noqa: E402

EXPECTED_REFERENCE_FILES = [
    "urop_context_and_instructions_for_executing_ai.md",
    "2608.30391v1.pdf",
    "Qual-Agent-Behavior-Analysis-main.zip",
    "hidden_error_awareness_deep_dive (1).pdf",
    "CLAUDE_CODE_UROP_HANDOFF.md",
]

UPSTREAM_PKG = REPO_ROOT / "external" / "autotracegt_upstream" / "autotracegt"


def check_reference_materials() -> dict:
    ref_dir = REPO_ROOT / "reference_materials"
    result = {"dir_exists": ref_dir.exists(), "files": {}}
    for name in EXPECTED_REFERENCE_FILES:
        path = ref_dir / name
        if path.exists():
            result["files"][name] = {"present": True, "sha256": sha256_file(path), "bytes": path.stat().st_size}
        else:
            result["files"][name] = {"present": False}
    return result


def check_upstream_code() -> dict:
    if not UPSTREAM_PKG.exists():
        return {"extracted": False}

    result: dict = {"extracted": True, "path": str(UPSTREAM_PKG)}

    pipeline_py = UPSTREAM_PKG / "pipeline.py"
    if pipeline_py.exists():
        text = pipeline_py.read_text(encoding="utf-8", errors="replace")
        result["pipeline_py_bytes"] = len(text)
        result["has_add_rate_max_0.1"] = "_ADD_RATE_MAX    = 0.1" in text or "_ADD_RATE_MAX = 0.1" in text
        datasets_line_start = text.find("DATASETS: dict[str, str] = {")
        result["datasets_declared"] = datasets_line_start != -1

    endpoints_py = UPSTREAM_PKG / "llm" / "endpoints.py"
    result["endpoints_py_present"] = endpoints_py.exists()

    license_files = list((REPO_ROOT / "external" / "autotracegt_upstream").glob("LICEN*"))
    result["license_file_found"] = [str(p.relative_to(REPO_ROOT)) for p in license_files]

    return result


def main() -> None:
    report = {
        "reference_materials": check_reference_materials(),
        "external_autotracegt_upstream": check_upstream_code(),
    }
    print(json.dumps(report, indent=2))

    ref = report["reference_materials"]
    missing = [name for name, info in ref["files"].items() if not info["present"]]
    if missing:
        print(f"\n[inspect_sources] MISSING reference files: {missing}", file=sys.stderr)
    if not report["external_autotracegt_upstream"]["extracted"]:
        print(
            "\n[inspect_sources] external/autotracegt_upstream/ not found -- "
            "extract reference_materials/Qual-Agent-Behavior-Analysis-main.zip there first.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
