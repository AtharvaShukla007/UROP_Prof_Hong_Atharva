# Project rules for anyone (human or AI) working in this repo

This file captures stable rules. Details live in `docs/`; don't duplicate
them here, and don't copy the full handoff/brief into every file that
touches this project.

## Non-negotiables

- **Never fabricate** numbers, results, file contents, or claims about what
  ran. Say "not verified" when that's the truth. `src/urop/reporting.py`
  enforces the sharpest version of this: it raises if asked to attach a
  metric to a run whose `backend_label` isn't a real inference backend.
- **Default paid API budget is zero.** `scripts/run_text_pilot.py` hard-
  refuses `openai`/`deepseek`/`gemini` backends. Don't work around that
  refusal without an explicit, separate cost approval recorded in
  `docs/compute_budget.md`.
- **No model inference, no GPU allocation, no weight downloads, and no
  `git push`/commit happen automatically.** Every script that touches a real
  backend or spends money requires an explicit flag or gate; see
  `notebooks/01_kaggle_text_pilot.ipynb`'s Section 0 gate for the pattern.
- **Raw inputs are never overwritten.** `src/urop/data.py` only reads;
  derived outputs go under git-ignored `runs/`/`data/` or tracked `results/`
  for curated summaries only.
- **A local dummy credential must never reach a non-local endpoint.**
  `src/urop/autotrace_adapter.register_local_endpoint` enforces this; use
  `external/autotracegt_upstream`'s `register_endpoint` directly (with a
  real key) for any genuine remote provider.

## Where things live

- `src/urop/` -- this project's own code: data validation/cleaning
  (`data.py`), deterministic task-grouped splits (`splits.py`), the bridge
  to vendored AutoTraceGT (`autotrace_adapter.py`), the tau-bench raw-schema
  converter (`tau_bench_adapter.py`), budgets/watchdog/resume (`budgets.py`),
  provenance manifests (`manifests.py`), and a reporting layer that refuses
  to fabricate metrics (`reporting.py`).
- `external/autotracegt_upstream/` -- vendored AutoTraceGT author code
  (git-ignored, no confirmed license -- see `external/PROVENANCE.md`). Wrap
  or subclass it from `src/urop/`; don't edit it in place.
- `external/tau-bench/` -- vendored tau-bench source (git-ignored, MIT
  licensed, pinned commit -- see `external/PROVENANCE.md`). Used for its
  retail task/tool/policy definitions, not run as a live harness yet.
- `data/tau_bench_raw/`, `data/tau_bench/` -- git-ignored real data: the raw
  downloaded historical trajectories and their converted/split/sampled
  derivatives (`scripts/convert_tau_bench.py`). Never present in a fresh
  clone -- see `docs/kaggle_quickstart.md` for exact reproduction commands.
- `reference_materials/` -- user-supplied source documents (git-ignored).
  `scripts/inspect_sources.py` checks presence/hashes.
- `configs/` -- one YAML per run stage. `text_smoke.yaml` (mock backend,
  runs today) vs. `text_pilot.yaml`/`agent_smoke.yaml` (candidate designs,
  blocked on Phase 0 items -- see `docs/phase0_findings.md`).
- `docs/` -- `research_plan.md` (what/why), `phase0_findings.md`
  (verified vs. outstanding, go/no-go), `methods_and_deviations.md` (every
  deliberate deviation from AutoTraceGT, with reasons),
  `compute_budget.md` (limits + what's actually been measured),
  `source_audit.md` (evidence behind every verified claim),
  `experiment_log.md` (append-only run history),
  `presentation_outline.md`, `kaggle_quickstart.md`.
- `tests/fixtures/` -- hand-written, clearly-synthetic data only. Never
  treat it as, or let it silently become, research input.

## Working style

- Plain language, explain jargon on first use, short bullets over walls of
  text, bold the important numbers/decisions -- this needs to be explainable
  out loud to a professor.
- A null/negative result is a valid deliverable. Report it, don't paper over
  it or keep re-splitting/re-running until something looks better.
- Before citing any paper: check it has real authors/venue/code, and that
  its claimed numbers actually appear in its own tables (see
  `docs/source_audit.md` for the standard this repo already applied to
  AutoTraceGT and the prior Hidden Error Awareness deck).
