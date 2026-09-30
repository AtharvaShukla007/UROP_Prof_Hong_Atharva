# Kaggle quickstart

`notebooks/01_kaggle_text_pilot.ipynb` has been run for real three times.
**2026-09-29:** the server started (after a CUDA-linker workaround, now a
permanent cell) and made 5 real requests, but produced **0 usable coded
outputs**. **2026-09-30, run 1:** the new structured-JSON mode returned
valid, schema-conformant JSON with `codes: []` -- structurally valid,
still 0 usable output (and a real reporting bug that counted it as usable
anyway, since fixed). **2026-09-30, run 2** (after a prompt clarification
and withholding `status`): 3 real behavioral codes came back, though all 3
failed strict cited-step quote validation on a whitespace formatting
difference (content confirmed to genuinely match on informal inspection;
the strict result is kept as official). See `docs/experiment_log.md` for
the full, evidence-based account of all three runs and
`docs/methods_and_deviations.md` §15-16 for what changed in response. This
is the exact sequence to run the notebook's current state: a
**one-trajectory diagnostic** (instance_id `13`, run_id `trial_0` -- the
one already exercised above), not a fresh multi-trajectory smoke test.
**If this run also fails or returns invalid evidence, stop and preserve
the artifacts (Section 7 below) rather than launching another run.**

## 1. Before opening Kaggle at all

- Read `notebooks/01_kaggle_text_pilot.ipynb` top to bottom in this repo
  first. Know what every cell does before it runs on a rented clock.
- Make sure real input data exists locally -- none of this is in a fresh
  `git clone` (everything below is git-ignored). Reproduce it with:

  ```bash
  # 1. Vendor AutoTraceGT's code (no confirmed license -- stays local-only)
  #    Extract reference_materials/Qual-Agent-Behavior-Analysis-main.zip into
  #    external/autotracegt_upstream/ (see external/PROVENANCE.md).

  # 2. Vendor tau-bench's code (MIT licensed), pinned to the commit our
  #    downloaded trajectories were generated against:
  git clone https://github.com/sierra-research/tau-bench.git external/tau-bench
  cd external/tau-bench && git checkout 59a200c6d575d595120f1cb70fea53cef0632f6b && cd ../..

  # 3. Download the real historical retail trajectories (MIT licensed,
  #    10,813,408 bytes, sha256 df01707894836168ff0ec9616b0bf08f66c7e5afcf313e5fe4f7a2f5c2ec938b):
  mkdir -p data/tau_bench_raw
  curl -sS "https://raw.githubusercontent.com/sierra-research/tau-bench/59a200c6d575d595120f1cb70fea53cef0632f6b/historical_trajectories/gpt-4o-retail.json" \
    -o data/tau_bench_raw/gpt-4o-retail.json

  # 4. Convert to our schema, split by task, select the smoke/pilot samples:
  .venv/Scripts/python.exe scripts/convert_tau_bench.py
  ```

  This produces `data/tau_bench/tau_bench_train_3.jsonl` (smoke) and
  `tau_bench_train_30.jsonl` (pilot) -- see `docs/phase0_findings.md` and
  `data/tau_bench/conversion_report.json` for exact counts. There is no
  synthetic-data fallback: if `tau_bench_train_3.jsonl` is missing from the
  attached bundle, the notebook's Section 5 raises and stops rather than
  substitute `tests/fixtures/tiny_trajectories.jsonl` (see
  `docs/methods_and_deviations.md` §12).

## 2. Prepare a Kaggle Dataset for the git-ignored attachments

Do not hand-zip `external/`, `reference_materials/`, and `data/` wholesale --
that would drag in unrelated PDFs, the full (untrimmed) tau-bench clone, and
the full 460-record corpus, none of which the gated notebook actually reads.
`scripts/build_review_bundles.py` builds a minimal **private** bundle
instead, containing only what `notebooks/01_kaggle_text_pilot.ipynb`
Section 2's `REQUIRED_PATHS` check (and the cells that follow it) actually
touch:

```bash
.venv/Scripts/python.exe scripts/build_review_bundles.py
```

This writes `review_bundles/urop_kaggle_private_bundle.zip` (private-bundle
contents below) and, separately, `review_bundles/urop_review_bundle.zip` (a
human-reviewable snapshot of our own source/docs/tests -- no vendored code,
no raw data; see that script's `build_review_zip()`). Both paths are
git-ignored (`review_bundles/`) -- rebuild on demand, never committed.

The private bundle contains, at minimum, for the notebook to do real work
rather than refuse to start:

| Path | What | Why the notebook needs it |
|---|---|---|
| `src/urop/` | our corrected package | imported directly via `sys.path.insert` (Section 2) -- also what Section 2's bundle auto-detection cell searches `/kaggle/input` for (`src/urop/__init__.py`), so the dataset MUST preserve this exact relative path |
| `requirements/kaggle-candidate.txt` | candidate GPU deps list | read by Section 3's `pip install` when `INSTALL_EXTRA_DEPS=True` |
| `requirements/kaggle-verified-pins.txt` | the real, evidence-backed 67-package pin list from the 2026-09-29 install (torch, vllm, and real transitive deps) | used as a pip constraints file (`-c`) by Section 3's install cell, when present, so the resolver can't silently pick different transitive versions than what was actually tested |
| `external/autotracegt_upstream/` | AutoTraceGT source (no confirmed license -- private bundle only, never the public GitHub repo) | `agents.open_coding.OpenCodingAgent`, prompts |
| `external/tau-bench/tau_bench/envs/retail/tools/` | only the retail tool-schema source files (not the full tau-bench clone) | `urop.tau_bench_adapter.extract_retail_tool_schemas` AST-parses these `get_info()` returns; also satisfies Section 2's path check directly |
| `data/tau_bench/tau_bench_train_3.jsonl` | the 3 real, already-converted smoke trajectories (each record already carries its own `tools` schema -- the notebook does not need to re-run the extractor) | what the notebook actually loads (Section 5) |
| `data/tau_bench/smoke_sample_manifest.json`, `conversion_report.json` | provenance for that sample | audit trail, small |
| `external/PROVENANCE.md`, `data/tau_bench_raw/PROVENANCE.md` | source hashes/commits | provenance, no raw data itself |

Deliberately excluded: `reference_materials/` (unrelated PDFs/original
brief -- not read by this notebook), the full `external/tau-bench` clone and
its own nested `.git/`, the full 460-record corpus and the raw
`gpt-4o-retail.json` download, the 30-trajectory pilot sample, `.venv/`, and
anything under `runs/`.

Upload `urop_kaggle_private_bundle.zip` via kaggle.com -> Datasets -> New
Dataset, unzip on upload (Kaggle does this automatically for a single zip),
note the dataset slug it gives you.

## 3. Open the notebook on Kaggle

- New Notebook -> upload `notebooks/01_kaggle_text_pilot.ipynb`, or paste
  its cells into a fresh Kaggle notebook.
- Settings -> **Accelerator: GPU T4 x2**. Settings -> **Internet: On**.
- Add the dataset from step 2 under "Add Input". Section 2's load cell
  **auto-detects** its mount path (searches `/kaggle/input` for
  `src/urop/__init__.py`) and asserts exactly one match -- no path to edit
  by hand, but it also means exactly one project dataset should be
  attached at a time. It also **refuses to reuse an existing `REPO_ROOT`**
  from an earlier cell run in the same kernel session (restart the kernel,
  or delete `/kaggle/working/UROP_Prof_Hong_Atharva` first, for a clean
  copy) -- this is deliberate: it stops a stale, already-copied source
  tree from silently surviving into a run meant to test a fresh bundle.

## 4. Run it

Section numbers below match the notebook's own markdown headers exactly
(27 cells total; grew from 23 after the 2026-09-29 run's fixes).

- Section 0 (the gate): edit every `I_CONFIRMED_...`/`I_...` flag to `True`
  only once it is actually true, then run the cell. It raises and stops
  otherwise. Includes a new flag confirming you understand this is a
  ONE-TRAJECTORY diagnostic, not a fresh smoke test.
- Section 1 (preflight): confirm GPU/internet/disk before anything else.
- Section 2 (load the project): set `PROJECT_BUNDLE_INPUT` to your dataset's
  real mount path (`/kaggle/input/<slug>`), run the cell -- it `copytree`s
  the bundle into `/kaggle/working/UROP_Prof_Hong_Atharva` and adds `src/`
  to `sys.path`. Also creates `install_evidence/` under `/kaggle/working`
  (private, outside the repo copy). The next cell checks every path in
  `REQUIRED_PATHS` and raises, loudly, before any GPU work if anything is
  missing.
- Section 3 (serving-stack install -- now with real evidence): review
  `requirements/kaggle-candidate.txt` yourself before flipping
  `INSTALL_EXTRA_DEPS = True`. The lookup cell does a live PyPI check of
  which vLLM release matches Kaggle's pre-installed torch (via
  `urop.env_check.torch_versions_compatible`) -- set
  `CONFIRMED_VLLM_VERSION` explicitly from that output (the value verified
  2026-09-29/30 was `"0.18.1"` -- re-check it's still appropriate, don't
  assume). The install cell uses `requirements/kaggle-verified-pins.txt`
  as a pip constraints file when present (pins transitive deps to
  previously-verified versions), always passes `--only-binary=:all:`, and
  **refuses outright (prints why, installs nothing) if the plan would
  touch torch at all** -- both at plan time and, redundantly, as a hard
  assert right before the real install. It then actually runs `pip
  install` and saves the FULL stdout/stderr to
  `install_evidence/install_log.txt` -- **read it**; it is expected to
  show pip's own dependency-resolver conflict warnings against some of
  Kaggle's preinstalled packages (Google client libraries, Gradio,
  confirmed in the 2026-09-29 run), which are not silently hidden or
  auto-resolved by installing more things. Then: a server entry-point check
  (`vllm.entrypoints.openai.api_server --help`, no model launch) saved to
  `install_evidence/server_import_check.txt`, a model-revision resolution
  against Hugging Face's public API saved to
  `install_evidence/model_selection.json` (proposes a revision -- you still
  copy it into the config cell yourself), and the **CUDA-linker
  workaround** required to start the server at all on this Kaggle image
  (`ld: cannot find -lcuda` during FlashInfer's kernel JIT compile) -- all
  three previously ad hoc cells, now permanent and gracefully skip on a
  non-Linux/non-GPU environment instead of crashing.
- Section 4-5 (config, then ONE protected lifecycle): set
  `CANDIDATE_MODEL_REVISION` (copy from `model_selection.json` above) and
  `OUTPUT_MODE` (`"structured_json"` -- the opt-in schema-constrained
  mode, default and the one actually exercised 2026-09-30; or
  `"upstream_codes_block"` -- AutoTraceGT's own freeform format,
  unmodified). `MAX_RESULT_LEVEL_ATTEMPTS`/`SMOKE_BUDGET_MAX_CALLS` are
  both `1` -- this diagnostic is scoped to exactly one real request, no
  automatic retry; raise them deliberately, with justification, for a
  larger run. The data-loading cell filters the 3-trajectory sample down
  to exactly `instance_id="13"`, `run_id="trial_0"` and **asserts exactly
  one record is selected** -- change those IDs deliberately, not by
  deleting the assertion, if you want a different trajectory. Flip
  `RUN_SERVER = True` only when ready to spend GPU time; the request
  budget (`SMOKE_BUDGET`) is constructed fresh right before use in this
  cell, not in the config cell, so time spent editing settings doesn't
  silently consume it. Every real/blocked request is logged immediately
  (`runs/kaggle_smoke/<run_id>/evidence/evidence_NNNN_*.json`) --
  finish_reason, token usage, the exact request payload, timing, errors --
  independent of whether the trajectory it belongs to ever finishes. A
  context-limit block (`PromptTooLongError`) and a genuine schema failure
  (`StructuredOutputError`) are tracked SEPARATELY (`context_limit_blocks`
  vs `schema_failures`) -- neither discards an EARLIER chunk's real,
  completed result in the same trajectory. A result is only cached as done
  when it has >=1 code AND every cited quote verifies within its own step
  range -- an empty or quote-invalid result is preserved on disk but never
  cached as a success, and a CACHE HIT is itself revalidated against the
  current rules before being trusted (raises loudly, does not silently
  reuse, if an old entry now fails). The whole cell is one
  `try/except/finally`; ANY failure at ANY stage -- server launch,
  tokenizer load, agent construction, the coding loop -- still tears down
  the owned server process GROUP and saves a stop/error manifest via
  `urop.reporting.build_outcome_report`.
- Section 6 (report): re-reads and prints the manifest already saved above
  -- `outcome_label` (e.g. `"completed_with_zero_usable_outputs"`) can
  never be mistaken for success just because `stop_reason="completed"`.
- Section 7 (export results and evidence): zips `runs/kaggle_smoke/` and
  `install_evidence/` into `/kaggle/working/urop_smoke_debug.zip` --
  **private evidence; never commit it or attach it to either packaged
  bundle.**
- Section 8 (optional, separately gated `RUN_AXIAL_SMOKE`): **hard-raises
  if flipped `True`** -- it has no protected lifecycle of its own yet.
- Section 9 (shutdown): terminates the server process group, clears the
  CUDA cache, and reminds you that stopping the server does NOT stop the
  Kaggle session or its quota meter -- you must also use Kaggle's own
  **Stop Session** control.

## 5. Export results and evidence before the session ends

Section 7 of the notebook does this for you now (zips
`runs/kaggle_smoke/` + `install_evidence/` into
`/kaggle/working/urop_smoke_debug.zip`). Download it from the Kaggle
output pane. **This zip is private evidence** -- unzip it locally for your
own inspection (`scripts/summarize_run.py` can read the manifest inside
its `runs/` copy), but never commit it, never attach it to either
packaged bundle (`scripts/build_review_bundles.py` already excludes
`runs/`), and never publish its contents -- it may contain real model
output verbatim.

## 6. Shut down -- always

Run **Section 9** of the notebook (terminates the server process group,
clears the CUDA cache). Then use Kaggle's own **Stop Session** control
(top-right of the notebook editor). Confirm the GPU indicator drops to 0
before closing the tab -- **an idle allocated GPU still consumes your
weekly quota, and selecting "GPU T4 x2" bills for both GPUs for the whole
session even though this notebook only ever uses one.**
