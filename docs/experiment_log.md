# Experiment log

Append-only. Each entry: what ran, exact command, environment, result. Never
edit a past entry's result after the fact -- add a new entry that supersedes
it and say so.

## 2026-09-28 -- local repository setup (CPU only)

**Environment:** Windows 11, `uv`-managed Python 3.12.13 venv at `.venv/`
(system default Python was 3.14.5, likely too new for the ML stack Phase 3
will need -- `pyproject.toml` pins `>=3.11,<3.13`). Dependencies:
`requirements/cpu.txt` (compiled + installed via `uv pip compile` /
`uv pip install`, see file for exact pins).

**What ran:**

| Command | Result |
|---|---|
| `uv venv --python 3.12 .venv` | created |
| `uv pip install -r requirements/cpu.txt --python .venv/Scripts/python.exe` | 20 packages installed, clean |
| `uv pip install -e . --python .venv/Scripts/python.exe` | `urop-agent-diagnosis` installed editable |
| `.venv/Scripts/python.exe -m pytest tests/ -v` | **25/25 passed**, 0.59s |
| `.venv/Scripts/python.exe scripts/inspect_sources.py` | all 5 reference files present + hashed; upstream code confirmed extracted, `_ADD_RATE_MAX=0.1` confirmed live from source, no LICENSE file found |
| `.venv/Scripts/python.exe scripts/prepare_data.py` (synthetic fixture) | 5 accepted / 2 rejected / 1 duplicate of 8 lines; split 3 train / 1 test tasks |
| `.venv/Scripts/python.exe scripts/run_text_pilot.py --config configs/text_smoke.yaml` | mock backend, 3/3 records processed, `backend_label="mock"`; re-run confirmed resume cache skipped all 3 |
| `.venv/Scripts/python.exe scripts/run_text_pilot.py --config configs/text_pilot.yaml` | correctly refused (`backend="local"` not implemented in this script yet), exit code 2 |
| `.venv/Scripts/python.exe scripts/summarize_run.py --run-dir <smoke run>` with a fabricated `auroc` metric | correctly refused (`NotResearchDataError`, exit code 3) |
| `notebooks/00_cpu_data_check.ipynb`, executed via `nbclient` against a **disposable copy** (committed file stays output-free) | all 11 cells succeeded, including the full pytest run from inside the notebook |
| `notebooks/01_kaggle_text_pilot.ipynb` | validated as well-formed `nbformat`; every code cell compiles (syntax-check only). **Not executed** -- requires a real Kaggle GPU session. |
| AutoTraceGT paper text extraction via `uv run --with pypdf --python 3.12 python ...` (throwaway env, not added to project deps) | 33 pages extracted; author list, T3 finding, Table 3 range, and the saturation-rule count-vs-rate discrepancy all confirmed against the actual PDF text -- see `docs/source_audit.md` |

**Not run, by design, this session:** any real model inference, any GPU
allocation, any paid API call, any git push/commit, any Kaggle session.

**Stop reason:** task scope complete (local CPU setup + gated Kaggle
notebook authored). No budget exhaustion, no error.

## 2026-09-28 -- real tau-bench data preparation (CPU only)

**Disk-space incident, resolved mid-session:** the C: drive hit **0 bytes
free** (860,587,626,496 / 860,587,626,496 used) partway through cloning
tau-bench, which failed with "No space left on device". Diagnosed via
`Get-CimInstance Win32_LogicalDisk` (not just `df`, which kept reporting a
stale 0 even after small writes briefly succeeded off leftover NTFS slack).
Did not delete any of the user's personal files -- asked, user freed space
on C: themselves (confirmed via the same PowerShell check: 2.77 GB free,
later 7.6+ GB after the tau-bench historical-trajectories duplicate was
trimmed). All commands below ran after that fix.

**What ran:**

| Command | Result |
|---|---|
| `git clone https://github.com/sierra-research/tau-bench.git external/tau-bench` | succeeded; pinned `git rev-parse HEAD` -> `59a200c6d575d595120f1cb70fea53cef0632f6b` |
| `curl` of `historical_trajectories/gpt-4o-retail.json` (raw, not `git checkout` -- see `core.autocrlf` note in `external/PROVENANCE.md`) | 10,813,408 bytes, sha256 `df01707894836168ff0ec9616b0bf08f66c7e5afcf313e5fe4f7a2f5c2ec938b`, byte-count matches the GitHub API's blob size exactly |
| `git hash-object` on the (since-trimmed) checked-out copy | `27f588d9157513c3ad11de3db9e824021152d580`, matches the GitHub API's recorded blob sha exactly -- confirms the curl download is complete and uncorrupted |
| AST-based extraction of all 16 retail tool `get_info()` schemas from `external/tau-bench/tau_bench/envs/retail/tools/*.py` | 16/16 extracted; `tests/test_tau_bench_adapter.py::test_extract_retail_tool_schemas_from_real_vendored_source` passes against this live source |
| `.venv/Scripts/python.exe -m pytest tests/test_tau_bench_adapter.py -v` | **11/11 passed** (synthetic-fixture unit tests for the converter) |
| `.venv/Scripts/python.exe scripts/convert_tau_bench.py` | 460/460 converted, 0 rejected; 115 distinct tasks; reward 1.0/0.0 = 278/182; split 92/23 train/test tasks; wrote `tau_bench_train_3.jsonl` (smoke) and `tau_bench_train_30.jsonl` (pilot) |
| `.venv/Scripts/python.exe scripts/prepare_data.py --input data/tau_bench/tau_bench_retail_gpt4o_all.jsonl --domain tau_bench_retail` | 460 accepted / 0 rejected / 0 duplicate on the full real corpus |
| `.venv/Scripts/python.exe scripts/run_text_pilot.py --config configs/text_smoke_real_tau_bench.yaml` | mock backend, 3/3 real records processed, `backend_label="mock"` |
| `.venv/Scripts/python.exe -m pytest tests/ -q` (full suite) | **36/36 passed** (25 prior + 11 new) |
| `notebooks/01_kaggle_text_pilot.ipynb` re-validated after wiring in the real `OpenCodingAgent` call | still valid `nbformat`, all cells still compile; the CPU-runnable portion (real-data loading + tiktoken token-limit check) was extracted and run standalone against the real smoke file, matching the notebook's logic exactly: 3/3 trajectories fit the 7,168-token budget (4,196 / 4,373 / 3,677 tokens) |

**Not run, by design, this session:** any real model inference (no GPU
available), any GPU allocation, any paid API call, any git push/commit,
any Kaggle session, download of `sonnet-35-new-retail.json`.

**Stop reason:** task scope complete (tau-bench inspected and pinned, real
data downloaded/converted/split/sampled, CPU checks re-run on real data,
Kaggle notebook upgraded to real wiring). No budget exhaustion, no error.

## 2026-09-28 -- correctness fixes: sampling, tokenizer, pipeline scope (CPU only)

Follow-up disk-space incident: C: hit 0 bytes free again after the prior
session; user freed ~3 GB (confirmed 2.89 GB free via
`Get-CimInstance Win32_LogicalDisk` before starting). No local package
installs beyond what was already in `requirements/cpu.txt` this round.

**What changed and was re-verified:**

| Change | Verification |
|---|---|
| `src/urop/tau_bench_adapter.sample_train_records`: seeded `random.Random(seed).sample()` over train task IDs, replacing ascending-id truncation | 7 new unit tests (randomness, determinism, cross-seed difference, no outcome filtering, one-trial-per-task, manifest correctness, oversized-n handling) -- all pass |
| `scripts/convert_tau_bench.py` re-run on real data with the new sampler | smoke sample: task_ids `[86, 13, 102]` (seed 42) -- previously ascending `[4, 16, 99]` under the old (mislabeled) method; both `{smoke,pilot}_sample_manifest.json` written with exact selected IDs |
| `scripts/run_text_pilot.py --config configs/text_smoke_real_tau_bench.yaml` re-run against the new sample | 3/3 processed, `backend_label="mock"`, clean |
| `notebooks/01_kaggle_text_pilot.ipynb`: real Qwen-tokenizer context check (exact rendered prompt via the real `load_prompt`/`chunk_trajectory`/`format_chunk`, worst-case JSON-repair round reserved) added; tiktoken check relabeled informational-only; explicit open-coding-only scope statement; new separately-gated `RUN_FULL_PIPELINE` section (axial coding only, explicitly not saturation) | Notebook re-validated: valid `nbformat`, all 19 cells (up from 17) compile. The real-tokenizer code path itself is untested (needs `transformers` + network, not available locally) -- its non-GPU-dependent sibling logic (informational tiktoken check, real-data loading) was extracted and re-run standalone against the current real smoke sample: 3/3 fit |
| `requirements/kaggle-candidate.txt`: uncommented `transformers>=4.44` (now actively needed for the tokenizer check, not just future Phase 3) | not installed locally (per instruction to avoid large local installs); remains candidate/unverified on Kaggle |
| Full test suite | **43/43 passed** (`pytest tests/ -v`), 0.65s |

**Not run, by design, this session:** any model inference, any GPU
allocation, any paid API call, `transformers`/`torch` install (local),
any Kaggle session, git commit/push.

**Stop reason:** requested fixes complete and CPU-verified where CPU
verification is possible; GPU-side code remains syntax-checked only,
explicitly reported as such (see `docs/phase0_findings.md`, "Tested vs.
unverified").

## 2026-09-28 -- second review round: lifecycle, supervisor, cache validation, model revision, torch-compare (CPU only)

A second review of the (by-then-updated) bundles found 6 more targeted gaps
-- notebook Sections 4-5 sharing no cleanup path, no independent wall-clock
kill for the owned server process GROUP, resume-cache entries marked done
without validating the result, no pinned model revision in the cache key,
`RUN_AXIAL_SMOKE` reusing Section 5's server with no lifecycle of its own,
and a literal-string torch-version comparison that would never match
Kaggle's `+cuXXX`-suffixed reported version. All fixed and CPU/integration-
verified where verification is possible without a GPU.

**What changed:**

| Change | File(s) |
|---|---|
| `start_owned_process`/`shutdown_process_group` (POSIX process-group SIGTERM/SIGKILL; Windows `taskkill /F /T` -- a non-forced `taskkill /T` silently no-ops on a console process, caught by a test, see below) | `src/urop/process_control.py` |
| `DeadlineSupervisor`: independent daemon-thread wall-clock watchdog over one owned process group, decoupled from any HTTP timeout or the main thread's state | `src/urop/process_control.py` |
| `cache_key` gains a required `model_revision` field (separate from `model`); `ResumeCache` gains `record_attempt`/`is_exhausted` (bounded retries, never rerun indefinitely) and `result_path_for` (so a resumed run can reload a valid cached result, not just skip it) | `src/urop/budgets.py` |
| `classify_chunk_coding`/`classify_trajectory_coding`: tell a parse failure apart from a legitimate empty coding by checking the raw response for a clean empty-`CODES`-list marker, not just `codes == []` | `src/urop/autotrace_adapter.py` |
| `env_check.torch_versions_compatible`/`public_version`: PEP 440 public-release comparison (ignores a local `+cuXXX` build suffix) instead of literal string equality | `src/urop/env_check.py` (new) |
| `packaging>=23` declared explicitly (was only a transitive dep) | `requirements/cpu.in`/`.txt`, `requirements/kaggle-candidate.txt` |
| Notebook Sections 4-5 merged into one `try/except/finally`: server startup, `DeadlineSupervisor`, tokenizer load, agent construction, and the coding loop all share one cleanup path that ALWAYS saves a stop/error manifest, even on a setup-time failure before any coding call. Coding loop now classifies every result, routes `parse_failed` through `mark_invalid` with a bounded 2-attempt cap, and reloads a valid cache hit into a real `TrajectoryCoding` so it's counted in the run's report. `CANDIDATE_MODEL_REVISION` is now required before `RUN_SERVER=True` and flows into the server launch args, the tokenizer load, and the cache key; `GENERATION_SEED` is recorded and sent via `extra_body`. `RUN_AXIAL_SMOKE` now hard-raises if flipped `True` (no independent lifecycle yet). Section 3's install cell now shows a `pip install --dry-run --report -` plan before ever installing, and the torch/vLLM compatibility check uses `env_check.torch_versions_compatible`. | `notebooks/01_kaggle_text_pilot.ipynb` |

**What ran (this machine, no GPU):**

| Command | Result |
|---|---|
| `.venv/Scripts/python.exe -m pytest tests/ -v` | **70/70 passed**, 8.07s (was 51; +19 new: process-group/supervisor, cache-attempt/exhaustion, classify_*, env_check) |
| Same suite, throwaway venv on D:\ with `openai`/`python-dotenv`/`huggingface-hub` installed (`requirements/test-integration.txt`) | `tests/test_open_coding_integration.py`: **12/13 passed**. The 1 failure (`test_timeout_against_a_slow_fake_server_stops_promptly_and_is_recorded`) is a **pre-existing test this session did not modify**, failing on environment timing, not a regression: directly timed `OpenCodingAgent(...)` construction alone at **~0.97s** in this particular throwaway venv (cold Python 3.14, D:\, no bytecode cache), which alone exceeds the test's 0.3s budget window before any HTTP call happens -- `budget_exhausted_before_call` instead of the expected `_mid_call`. Reproduced twice, consistently. Not investigated further as out of scope for this round; flagged here rather than silently left unexplained. |
| A real bug this round's own new tests caught and fixed before being reported here: `shutdown_process_group`'s first Windows attempt used non-forced `taskkill /T`, which silently no-ops for a console subprocess (no GUI message loop to receive `WM_CLOSE`) -- `test_deadline_supervisor_independent_of_main_thread_blocking` failed until this was corrected to go straight to `taskkill /F /T` on Windows. | `src/urop/process_control.py` |
| Standalone execution (not just syntax-check) of the notebook's non-GPU-dependent cells -- config, real-data load, and the full protected-lifecycle cell with `RUN_SERVER=False` -- against the real repo state | Manifest correctly saved (`stop_reason="completed"`), 3/3 real records loaded, no exceptions |
| Same, forcing `RUN_SERVER=True` with `CANDIDATE_MODEL_REVISION` left unset (the guard's intended failure path) | Raised the expected `RuntimeError` **and** still saved a manifest with `stop_reason` recording the exact error -- proves item 1's "always save a manifest, even on setup failure" claim against real code, not just by inspection |
| `RUN_AXIAL_SMOKE` cell, text-patched to `True` and executed standalone | Raised the expected hard-disable `RuntimeError` |
| `nbformat.validate` + `compile()` on every code cell | notebook valid, 23 cells, all compile |
| Disk space (`Get-CimInstance Win32_LogicalDisk`) | C: dropped to **1.38 GB free** mid-session (was 11.4 GB at session start) from causes outside this session's own actions (two tiny ZIP rebuilds, no large installs on C:) -- switched the integration-test venv to D:\ (5.5 GB free) rather than risk it, consistent with the recurring incidents already logged above. Worth the user checking what's consuming C: outside this repo. |

**Not run, by design, this round:** any model inference, any GPU allocation,
any paid API call, git commit/push, a real vLLM server launch, a real
process-group kill against an actual multi-child vLLM tree (tested against
a synthetic parent+child instead, since vLLM itself needs a GPU to start).

**Stop reason:** all 6 requested corrections implemented and verified to
the extent possible without a GPU; both deliverable ZIPs rebuilt (see
`docs/kaggle_quickstart.md`).

## 2026-09-29 -- fixed the one flaky integration test, root cause confirmed (CPU only)

Diagnosed and fixed `test_timeout_against_a_slow_fake_server_stops_promptly_and_is_recorded`
(flagged, not yet fixed, in the entry above). Root cause confirmed exactly
as suspected: `Budget`'s clock starts at construction, but
`OpenCodingAgent(...)` construction (building the sync/async openai
clients, loading the prompt template) measured at ~0.97s in that
environment -- a 0.3s budget created BEFORE agent construction was silently
consumed by setup, so the first real HTTP attempt saw an
already-exhausted budget (`budget_exhausted_before_call`) instead of
genuinely timing out mid-request (`budget_exhausted_mid_call`).

**Fix:** construct the agent under a generous 60s budget so setup can't
exhaust it, then reset that SAME `Budget` object's `max_seconds`/
`started_at`/`calls_made`/`stop_reason` in place immediately before the
slow request. Confirmed (by tracing it through) that calling
`install_budget_guard` a second time with a brand-new `Budget` instead
would double-wrap `create` and raise a `TypeError` from a `timeout=`
keyword collision between the two guard layers -- mutating the existing
object in place is the correct fix, not a workaround. Also added an
explicit `len(server.requests) == 1` assertion, proving the request
actually reached the fake server before the client aborted it (a real
in-flight timeout, not e.g. a request that never went out).

**What ran** (throwaway venv on D:\ -- C: was at **0 bytes free** this
session, so `UV_CACHE_DIR` was also redirected to D:\ to avoid touching C:
at all):

| Command | Result |
|---|---|
| `pytest tests/test_open_coding_integration.py -v` | **13/13 passed** (was 12/13) |
| Same single test, repeated 3x in isolation | passed all 3 times, ~5s each -- not a lucky one-off |
| `pytest tests/test_process_control.py -v` (main `.venv`) | **10/10 passed**, unchanged -- `test_deadline_supervisor_independent_of_main_thread_blocking` (the process-GROUP-stops test) untouched |
| `pytest tests/ -q` (main `.venv`, full suite) | **70/70 passed**, 1 skipped (integration tests, no `openai` in main venv) |

**Not run, by design:** any model inference, any GPU allocation, any paid
API call, model-weight downloads, git commit/push.

**Stop reason:** the flagged flaky test is fixed and re-verified; both
deliverable ZIPs rebuilt.

## 2026-09-29 -- FIRST REAL KAGGLE GPU RUN: server started, 0 usable coded outputs

**Facts, confirmed directly from the attached `urop_smoke_debug.zip`
(private, git-ignored, never committed) and the executed notebook
(`urop-prof-hong.ipynb`, kept outside this repo) -- not the user's summary
alone:**

- Environment: Python 3.12.13, `torch==2.10.0+cu128`, `vllm==0.18.1`,
  `transformers==4.57.6`, `openai==2.24.0`, Tesla T4 (15.5 GB free),
  `Qwen/Qwen2.5-3B-Instruct` @ revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- **First launch failed during server startup** (manifest
  `kaggle_smoke_20260929T193026Z`: `stop_reason="error_during_setup_or_coding: ServerExitedDuringStartup('returncode=1')"`).
- **Second launch succeeded** after a CUDA-linker workaround (symlinking
  the actually-loaded `libcuda.so.1`, found via `/proc/self/maps` --
  `ldconfig` alone did not locate it -- into a writable dir and setting
  `LIBRARY_PATH`, verified with a real `c++ ... -lcuda` link test before
  trusting it). Manifest `kaggle_smoke_20260929T193910Z`:
  **5 real attempts, 0 records_coded, 2 records_invalid, 1
  records_skipped_prompt_too_long, `stop_reason="completed"`.**
- `data/tau_bench/kaggle_smoke/.../open_coding_13_trial_0_attempt1.json`:
  raw response contains `"quote": "<tool_call name=find_user_id_by_email>{"email":"mia.garcia2723@example.com"}"`
  -- an UNescaped nested-JSON quote inside a JSON string value, which
  breaks `json.loads` on the surrounding CODES array. Confirmed by reading
  the actual saved bytes.
- `open_coding_86_trial_0_attempt1.json`: response is wrapped in a
  ` ```json ` fence and ends mid-sentence ("...This code describes the
  step of") with no closing brackets/markers at all.
  **Suspected, NOT confirmed: truncation at `max_tokens=1024`** --
  `finish_reason` and token usage were not saved in that run, so this
  stays a suspicion, not a fact (fixed this session -- see below).
  Do not treat it as more than a suspicion.
- `resume_index.json`: the third record's (`instance_id=102`) entry shows
  `{"attempts": 1, "completed": false}` with NO `"reason"` and NO
  `"result_path"` -- **its result was never saved anywhere.** Root cause
  confirmed by reading the code path: `PromptTooLongError` raised during a
  chunk's JSON-repair round propagated out of `agent.code_record()`
  entirely, discarding the whole in-progress `TrajectoryCoding` --
  including any earlier chunk's real, already-received response -- and the
  notebook's `except PromptTooLongError` branch never called
  `resume_cache.mark_invalid()` either. Fixed this session (see below).
- Install evidence (`urop-install-pins.txt`, 67 packages) confirms
  `xgrammar`, `outlines_core`, `lm_format_enforcer`, and `llguidance` (vLLM's
  guided-decoding backends) were all installed -- structured/guided JSON
  output is available in this exact environment.
- The exact pip resolver conflict warnings against Kaggle's preinstalled
  Google client libraries and Gradio (reported directly by the user) are
  NOT preserved in any file in the attached evidence -- taken as a reported
  fact from direct observation of that session, not independently
  re-derived here. The install cell added this session now saves pip's
  full stdout/stderr specifically so this is captured as evidence on the
  next run, not just recalled from memory.

**Stop reason:** GPU session ended after this diagnostic; 0 usable
outputs. Not re-run this session (no GPU available here) -- see the fixes
below, applied in response, and `docs/methods_and_deviations.md` §16 for
why a structured-output mode was added rather than trying to patch the
freeform format's escaping.

## 2026-09-30 -- fixes in response to the 2026-09-29 run (CPU only, no GPU)

**What changed** (see `docs/methods_and_deviations.md` §16 for the
structured-output-mode rationale in full):

| Change | File(s) |
|---|---|
| `urop.evidence.EvidenceLog`/`CodingContext`: one JSON record per real/blocked request, written immediately at the response boundary -- survives a LATER request in the same trajectory failing or being blocked | `src/urop/evidence.py` (new) |
| `install_budget_guard` wired to log every request's finish_reason/usage/raw content/error, and to classify blocked-vs-attempted correctly (blocked requests still never count toward `budget.calls_made`, unchanged from before -- now also evidenced) | `src/urop/autotrace_adapter.py` |
| `code_trajectory_preserving_partial_results`: calls `code_chunk` directly instead of `code_record`, so a LATER chunk's block/failure no longer discards EARLIER chunks' real, completed results (`exc.partial_result`) -- the exact trajectory-102 bug | `src/urop/autotrace_adapter.py` |
| `urop.structured_coding` (new): opt-in JSON-schema-constrained output mode via vLLM's `response_format={"type":"json_schema",...}` (verified against current vLLM docs -- the older `extra_body={"guided_json":...}` form was deprecated in vLLM 0.12.0, must not be used against the installed 0.18.1). Directly targets the trajectory-13 failure: a properly JSON-encoded quote containing nested JSON parses cleanly, where the freeform mode's hand-written escaping did not. No repair round, no silent fallback -- `StructuredOutputError` on any schema/parse failure. Preserves upstream's own prompt substance verbatim (only the `## OUTPUT FORMAT` section is replaced) and its own freeform pipeline stays fully available, unmodified. | `src/urop/structured_coding.py` (new) |
| `classify_chunk_coding` fixed to check `raw` for BOTH modes' "genuinely empty, legitimately valid" case directly (structured mode: parse `raw` as JSON, check `"codes": []`) instead of only the freeform text-marker heuristic, which never matched structured-mode output | `src/urop/autotrace_adapter.py` |
| `urop.reporting.build_outcome_report`/`RunOutcomeReport`: explicit, separated counts (server_startup_success, real_attempts_made, context_limit_blocks, parse_or_schema_failures, length_limited_responses, quote_or_step_validation_failures, structurally_valid_outputs, outputs_passing_automated_evidence_checks) plus an `outcome_label` that can never read "completed" as "succeeded" -- reproducing the real 2026-09-29 run's exact shape now correctly reports `outcome_label="completed_with_zero_usable_outputs"`, not just `stop_reason="completed"` | `src/urop/reporting.py` |
| Notebook: CUDA-linker workaround, server entry-point check, and model-revision resolution are now permanent, well-positioned cells (were ad hoc); real install now saves full pip stdout/stderr + `--report` JSON + pins as evidence (previously dry-run-only); `OUTPUT_MODE` config flag; evidence-log/coding-context wired into the protected lifecycle; the request budget (`SMOKE_BUDGET`) is now constructed immediately before use rather than in the config cell, so time spent editing settings can't silently consume it; `RUN_AXIAL_SMOKE` hard-raises if flipped True; data-loading now selects exactly `instance_id="13"`/`run_id="trial_0"` with an assertion, for a targeted one-trajectory diagnostic rather than a fresh 3-trajectory smoke test; corrected wording that an idle allocated GPU or an unused second T4 doesn't cost quota -- both do. | `notebooks/01_kaggle_text_pilot.ipynb` |

**What ran (this machine, no GPU):**

| Command | Result |
|---|---|
| `.venv/Scripts/python.exe -m pytest tests/ -q` | **80/80 passed**, 1 skipped (integration tests need `openai`, not in main venv) -- was 70 before this round; +10 new (`tests/test_evidence_and_reporting.py`) |
| Same suite, throwaway venv on D:\ with `openai`/`python-dotenv`/`huggingface-hub` installed | `tests/test_open_coding_integration.py`: **15/15 passed** -- was 13; +2 new, both reproducing the exact real failure patterns (unescaped quotes, incomplete/truncated JSON) against the REAL vendored `agents.open_coding` module, confirming it rejects both without fabricating a repair |
| Standalone execution of the notebook's non-GPU cells: config, one-trajectory selection+assert, protected lifecycle with `RUN_SERVER=False` | Selected exactly 1 record (`tau_bench_retail:13:trial_0`); manifest saved with `outcome_label="not_run"` (a real bug caught and fixed here: it initially said `"server_startup_failed"` for a deliberate no-op) |
| Same, forcing `RUN_SERVER=True` with `CANDIDATE_MODEL_REVISION` unset | Raised the expected `RuntimeError` and still saved a manifest (`outcome_label="server_startup_failed"`, correctly distinct from `"not_run"` since the server WAS attempted this time) |
| CUDA-linker cell, executed standalone on this Windows machine | Correctly detected `sys.platform != "linux"` and skipped with a clear message -- a real gap caught here: it originally only checked for `nvidia-smi`, which this machine DOES have (for other purposes), and would otherwise have crashed on `ctypes.CDLL("libcuda.so.1")` with a confusing low-level error |
| `RUN_AXIAL_SMOKE` cell, text-patched to `True` | Raised the expected hard-disable `RuntimeError` |
| `nbformat.validate` (warnings promoted to errors) + `compile()` on every code cell | notebook valid, 27 cells (was 23), all compile, no missing-id warnings |

**Not run, by design, this session:** any model inference, any GPU
allocation, any paid API call, model-weight downloads, a real vLLM server
launch, git commit/push. The structured-output mode's actual behavior
against a real served model -- whether vLLM 0.18.1 actually honors
`response_format` as documented, whether guided decoding measurably slows
generation, whether the model's *content* quality changes under the
stricter constraint -- is UNVERIFIED until the next real Kaggle run.

**Stop reason:** all 6 requested items implemented and CPU-verified where
verification is possible without a GPU; both deliverable ZIPs rebuilt (see
`docs/kaggle_quickstart.md`). Next step is the one-trajectory Kaggle
diagnostic this notebook is now configured for.

## 2026-09-30 -- SECOND REAL KAGGLE RUN: structured mode works, empty output found, then real labels found

**Facts, confirmed directly from two attached debug ZIPs (private,
git-ignored, never committed) and a self-contained diagnostic notebook run
on Kaggle against the one-trajectory diagnostic from the entry above:**

- **Run `kaggle_smoke_20260930T034853Z` (first structured attempt):** 1
  record (`13`/`trial_0`), 1 real request, server started normally,
  response was valid JSON matching the schema, `codes: []` -- structurally
  valid, zero labels. Confirmed from the saved manifest and
  `open_coding_13_trial_0_attempt1.json`. **The report at the time
  incorrectly counted this as a usable output** -- `outputs_passing_
  automated_evidence_checks` only checked `invalid_codes == []`, which an
  empty list trivially satisfies. Root cause of the "LATEST CHANGES" fixes
  below.
- **Run `kaggle_smoke_20260930T041739Z` (after a prompt clarification --
  see below):** same 1 record, 1 real request, normal stop, 1875 input /
  323 output tokens (confirmed via the evidence log), valid JSON, **3 real
  behavioral codes returned.** Confirmed by reading
  `open_coding_13_trial_0_attempt1.json` directly: quotes like `"[Step 3]
  (assistant) <tool_call name=find_user_id_by_email>{"email":"mia.garcia2723@example.com"}"`.
  **All 3 failed strict cited-step verbatim quote validation** -- confirmed
  by reading the raw text: the model's quotes use a SPACE between the
  `[Step N] (role)` header and the content, where `format_chunk` (the real
  renderer) always uses a NEWLINE there. A separate, informal whitespace-
  normalized check (not part of the strict validator) confirmed the
  underlying content genuinely matches within the cited step ranges. **The
  strict (space-sensitive) result is kept as the official one on purpose --
  do not retroactively relax the validator to make this look like a pass.**
- Manual inspection of the 3 labels found the coding itself was weaker
  than intended: one label ("skip order ID") read the user's inability to
  provide an order ID as the assistant "insisting it needed the ID and not
  proceeding," which is a literal description of the exchange, not a
  named behavioral pattern -- and the labels generally leaned toward
  describing actions rather than naming conceptual patterns. **This is not
  independent human annotation and is not benchmark accuracy** -- it is
  this project's own qualitative read of automated output, stated as such.
- No fresh acting-agent trajectories, hidden-state extraction, probe
  training, or full AutoTraceGT replication happened in either run. This
  remains an adapted open-coding pilot on existing tau-bench retail
  records, not a benchmark run.

**Stop reason:** two bounded, one-request diagnostics completed as
configured; not re-run further this session (no GPU available here). See
the next entry for the CPU-side fixes made in response.

## 2026-09-30 -- merged externally-prepared fixes for the empty-output/cited-step findings (CPU only, no GPU)

A parallel session (working from the same evidence above) prepared source
changes and a self-contained diagnostic notebook with an embedded source-
patch cell (needed because the already-uploaded Kaggle dataset predated
these fixes). Diffed every incoming file against this repo's current state
before merging (`_local_review/`, git-ignored, never committed) -- only 3
source files actually differed from what was already here; everything else
matched exactly.

**What changed and why (see `docs/methods_and_deviations.md` §16 update):**

| Change | File(s) |
|---|---|
| `validate_coding_quotes` now restricts the verbatim check to the SPECIFIC step range a code cites (`code["steps"]`), reconstructed via the real `format_chunk`, instead of the whole chunk -- a quote that IS verbatim somewhere in the chunk but from the WRONG step now correctly fails. Also validates the `steps` field's own format (rejects malformed/out-of-range citations with a clear reason) | `src/urop/autotrace_adapter.py` |
| `build_outcome_report`: a genuinely empty (0-code) result no longer counts toward `outputs_passing_automated_evidence_checks`/`usable_outputs` just because an empty list trivially has zero invalid codes -- the real 2026-09-30 bug above. New `empty_code_outputs` field tracks it separately. New optional `failed_record_ids` param lets a caller fold in failures that never produced a `coded_results` entry at all (e.g. a schema failure) | `src/urop/reporting.py` |
| `structured_coding.py`: system prompt gets an appended "DIAGNOSTIC CLARIFICATION" section (ordinary/successful behavior can be labeled, empty output stays an explicit option with a required `skipped` explanation, no quota-filling); `status` is now withheld from the structured user prompt (was leaking the true resolved/failed outcome into the coding request -- a real leakage fix); a response with `finish_reason == "length"` now raises `StructuredOutputError` immediately with an attributed reason instead of attempting to parse truncated JSON | `src/urop/structured_coding.py` |
| Notebook: bundle-load cell now auto-detects the attached dataset path and refuses to reuse a stale `REPO_ROOT` from an earlier cell run; install cell now uses `requirements/kaggle-verified-pins.txt` (new, the real 67-package pin list from the 2026-09-29 install) as a pip constraints file plus `--only-binary=:all:` and a hard refusal if the plan would touch torch at all; config cell: `MAX_RESULT_LEVEL_ATTEMPTS`/`SMOKE_BUDGET_MAX_CALLS` reduced to 1 (bounded one-request diagnostic scope); protected-lifecycle cell: `schema_failures` tracked separately from `context_limit_blocks` (a `StructuredOutputError` is a genuine schema failure, not a context-limit block -- both now separately preserve any earlier chunk's completed work via `exc.partial_result`, not just the freeform-mode path); `cuda_linker_library_path` recorded in the manifest; the structured-mode prompt hash now folds in the ACTUAL rendered structured system prompt (not just upstream's freeform templates), so a structured-prompt text change correctly invalidates stale cache entries; a cache hit is now REVALIDATED against current (possibly stricter) rules before being trusted, raising loudly if a previously-cached result fails revalidation instead of silently reusing it; a result is only `mark_done`-cached when it has >=1 code AND all cited quotes verify -- empty or quote-invalid results are preserved but marked invalid, never cached as done | `notebooks/01_kaggle_text_pilot.ipynb` |
| `requirements/kaggle-verified-pins.txt` (new) -- the real, evidence-backed 67-package pin list from the 2026-09-29 install (torch, vllm, and their real transitive deps); safe to publish (version strings only, no secrets) | `requirements/kaggle-verified-pins.txt` (new) |

**What ran (this machine, no GPU):**

| Command | Result |
|---|---|
| `.venv/Scripts/python.exe -m pytest tests/ -q` | **81/81 passed**, 1 skipped -- +1 new test (`test_empty_but_structurally_valid_output_does_not_count_as_usable`, reproducing the exact real bug) |
| Same, throwaway venv with `openai` installed | `tests/test_open_coding_integration.py`: **19/19 passed** -- +4 new: two cited-step-validation tests (wrong-step quote rejected, out-of-range/malformed step citation rejected with a clear reason) against the REAL `format_chunk`/`chunk_trajectory`, and a `finish_reason=="length"` structured-mode test (required teaching `tests/fake_local_server.FakeLocalChatServer` an optional `finish_reason` param) |
| `nbformat.validate` (warnings promoted to errors) + `compile()` on every code cell | notebook valid, 27 cells, all compile |
| Standalone execution of the merged notebook's non-GPU cells (config, one-trajectory selection, protected lifecycle with `RUN_SERVER=False`; then forcing `RUN_SERVER=True` with no revision; then the CUDA-linker cell; then the axial hard-disable) | All 4 scenarios behaved as expected -- `outcome_label` correctly `"not_run"` vs `"server_startup_failed"` in the two RUN_SERVER states, CUDA-linker cell skipped cleanly on this Windows machine, axial cell still hard-raises |

**Not run, by design, this session:** any model inference, any GPU
allocation, any paid API call, model-weight downloads, git commit/push.
The merged notebook's ACTUAL behavior on Kaggle (does the revalidation-on-
cache-hit path, the separated schema-failure tracking, and the corrected
empty-output handling all behave as designed against a real server) is
UNVERIFIED until run there.

**Stop reason:** external fixes merged, reconciled with local CPU-side
work already in this repo, and re-verified; not committed or pushed per
instruction.
