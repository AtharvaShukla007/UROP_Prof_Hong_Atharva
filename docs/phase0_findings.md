# Phase 0 findings and go/no-go

Written after the local repository setup (this sprint's CPU-only task), not
after any GPU/Kaggle work -- none has happened yet. See `docs/source_audit.md`
for the detailed evidence behind each line here.

## Resolved in the 2026-09-28 second follow-up session (review fixes)

A review of the Kaggle notebook found real integration bugs, not just
documentation gaps. All fixed and CPU-verified where CPU verification is
possible:

- **No more GitHub dependency.** The notebook previously `git clone`d the
  public repo -- since local work is routinely uncommitted/unpushed, this
  would have silently run stale/missing code on Kaggle. Removed entirely;
  the notebook now only reads from a private Kaggle Dataset bundle, copied
  into writable `/kaggle/working`, with a hard path-validation cell before
  any GPU work can start.
- **Real cancellation, not an abandoned thread.** `src/urop/budgets.bounded_llm_call`
  replaces the old thread-based `call_with_watchdog` -- see
  `docs/methods_and_deviations.md` §4. Two real correctness bugs were found
  and fixed in this new code specifically *by running the full test suite
  repeatedly*, not just once: a timing race that occasionally misclassified
  a genuine timeout, and a second bug (introduced while fixing the first)
  that misclassified an unrelated `ValueError` as budget exhaustion. Both
  fixes verified by 5 consecutive full-suite runs with zero failures.
- **Retry semantics were actually broken**, not just imprecisely described:
  the old `JSON_RETRIES=1` setting meant upstream's repair loop condition
  (`attempt < MAX_JSON_RETRIES - 1`) was never true, so **zero** repairs
  ever ran, contrary to the "1 bounded repair" this repo had claimed.
  Fixed to `JSON_RETRIES=2`; verified against the real `OpenCodingAgent`
  that a malformed-then-valid response sequence takes exactly 2 real
  requests. See `docs/methods_and_deviations.md` §13.
- **Token checks now cover every real request**, not just an ahead-of-time
  estimate on the first chunk -- including JSON-repair rounds and later
  chunks whose prompt depends on a model-generated memo. See
  `docs/methods_and_deviations.md` §14.
- **10 new integration tests** against the REAL `external/autotracegt_upstream`
  `OpenCodingAgent`, talking over a real socket to a fake local HTTP server
  (`tests/fake_local_server.py`) -- not mocks of our own code. Covers the
  real request path, the corrected repair-count semantics, per-request
  token checks (including a repair round), resume-cache reuse, quote/span
  validation wired to a real coding result, and a real client-side timeout
  that provably stops promptly. Required installing `openai`,
  `python-dotenv`, `huggingface-hub` -- done in a throwaway venv on D:\\
  (C:\\ had 0 bytes free again at the time; see `docs/experiment_log.md`),
  not the main project `.venv`.
- **vLLM/torch compatibility researched, not guessed:** every vLLM release
  checked (0.10.0 through the current 0.30.0, via live PyPI metadata) pins
  an *exact* torch version -- there is no version range anywhere. This
  means installing vLLM will force a torch reinstall unless Kaggle's
  preinstalled torch happens to exactly match one release's pin. The
  notebook's Section 3 now does a live lookup against a shortlist of recent
  vLLM versions and reports which (if any) matches, rather than assuming.
  GPU compatibility (attention backend on T4/Turing, whether the server
  even starts) remains explicitly, loudly unverified.
- **RUN_FULL_PIPELINE renamed to RUN_AXIAL_SMOKE**, with corrected scope
  language: it runs `AxialCodingAgent.code_batch()` once, on one small
  batch -- stages 3-5 (post-axial eval, code manager/saturation loop,
  selective coding) are not wired up anywhere in this notebook.
- **Synthetic-data fallback removed** from the notebook -- it now raises if
  the real converted smoke file is missing, instead of silently degrading
  to `tests/fixtures/tiny_trajectories.jsonl`.

## Tested vs. unverified -- the exact boundary

**Tested against our own logic/fixtures, CPU-only, passing (51/51, main
`.venv`, `pytest tests/ -v`):** raw-record validation/rejection,
task-grouped splitting, `bounded_llm_call` cancellation (including the
elapsed-time-based budget-attribution logic, verified stable across 5
consecutive full-suite runs), resume-cache correctness, reporting's
mock-run refusal, the tau-bench raw -> our-schema converter (against both a
synthetic fixture and, live, the real vendored tau-bench source), the
seeded-random sampling method, and `src/urop/process_control` (terminate/
wait/kill escalation and startup-exit detection against real, if trivial,
OS subprocesses -- not mocked `Popen` objects).

**Tested against the REAL upstream code, CPU-only, passing (61/61 total,
51 above + 10 in `tests/test_open_coding_integration.py`, a separate D:\\
venv with `openai`/`python-dotenv`/`huggingface-hub` installed -- see
`docs/experiment_log.md`):** the real `external/autotracegt_upstream`
`OpenCodingAgent`, talking over a real socket to a fake local HTTP server.
Covers the real request path, corrected repair-count semantics (exactly 2
attempts for 1 initial + 1 repair), per-request token checks including a
repair round, resume-cache reuse, quote/span validation on a real coding
result, and a real client-side timeout that provably stops promptly
(elapsed < 3s against a 5s-delay fake server with a 0.3s budget).

**Not tested, cannot be from this machine, remains unverified:** anything
inside `notebooks/01_kaggle_text_pilot.ipynb`'s `if RUN_SERVER:` /
`if RUN_AXIAL_SMOKE:` blocks when actually pointed at a real vLLM server --
vLLM installing and starting successfully on Kaggle's T4 at all, the real
Qwen tokenizer's actual `apply_chat_template` output shape, and whether the
served model's tool-calling behaves as expected. The *logic* those blocks
call (`install_budget_guard`, `bounded_llm_call`, `ResumeCache`,
`validate_coding_quotes`, `process_control`) is the same code exercised by
the 61 passing tests above -- what's untested is specifically the real
network/GPU/model layer underneath it, which cannot exist on this machine.
Every notebook code cell is confirmed syntactically valid (`compile()`) and
the CPU-only portions (path validation, real-data loading, settings/prompt
hashing) were extracted and run standalone against the real local repo
state. A cell compiling and its non-GPU logic running correctly is not
evidence the GPU-side call succeeds -- that stays "candidate" until
actually run on Kaggle.

## Verified this session

- All five expected `reference_materials/` files are present, hashed (see
  `docs/source_audit.md`), and were actually read (not assumed).
- The AutoTraceGT paper (arXiv 2608.30391v1) was read in full (33 pages,
  text-extracted). Author list, T3 finding, and Table 3 baseline range all
  check out against the handoff's summary.
- The author code ZIP was extracted and its source read directly:
  `pipeline.py`'s saturation default (`_ADD_RATE_MAX = 0.1`), the
  `DATASETS` registry, `llm/endpoints.py`'s `register_endpoint` contract,
  and `utils/trajectory_processing.py`'s outcome-annotation behavior are all
  confirmed from the actual files, not from memory or the handoff's summary.
- No `LICENSE` file exists anywhere in the author code ZIP -- confirmed by
  directory listing. Code stays local-only under git-ignored `external/`.
- A found, real discrepancy between the paper and its own code on what the
  saturation threshold measures (count vs. rate) and what value to use
  (0.1 vs 0.2) -- see `docs/source_audit.md` and
  `docs/methods_and_deviations.md` §2. This is sharper than "the two don't
  match" -- the paper's own pseudocode and its own prose don't agree with
  each other either.
- A CPU-only foundation (`src/urop/`, `scripts/`, `tests/`) is built,
  installed, and passing: 25/25 pytest checks green against a Python 3.12
  venv created with `uv`, using pinned versions in `requirements/cpu.txt`.
  See `docs/experiment_log.md` for the exact run.
- `notebooks/00_cpu_data_check.ipynb` was executed end-to-end (via a
  disposable copy, not the committed file, so the committed notebook stays
  output-free) and every cell succeeded, including a live run of the full
  pytest suite from inside the notebook.
- `notebooks/01_kaggle_text_pilot.ipynb` is valid `nbformat` and every code
  cell compiles (syntax-checked only) -- it has **not** been executed and
  cannot be, from this machine. It is gated behind an explicit confirmation
  cell that must be hand-edited to `True` before anything downloads or runs.

## Resolved in the 2026-09-28 follow-up session (real-data preparation)

- **tau-bench repo inspected and pinned.** Cloned
  `sierra-research/tau-bench` at commit
  `59a200c6d575d595120f1cb70fea53cef0632f6b`. Retail task counts, the 16
  tools (7 state-changing), the exact policy text, and the strictly-binary
  reward semantics are all confirmed from source -- see
  `docs/source_audit.md`. Deliberately pinned this "not updated" repo rather
  than τ³-bench, because our real trajectories were generated against this
  commit's task/policy text (`docs/methods_and_deviations.md` §10).
- **Real historical trajectory data obtained.** Downloaded
  `historical_trajectories/gpt-4o-retail.json` (460 real records: 115 test-
  split tasks × up to 4 trials, 278 reward=1.0 / 182 reward=0.0), MIT
  licensed, hash-verified two independent ways -- see
  `data/tau_bench_raw/PROVENANCE.md`. This is real data, not synthetic;
  **no history for the train/dev retail splits or for the airline domain
  was obtained** -- only what tau-bench's own README already ships for
  retail/test.
- **Converter built, tested, and run on the real data.**
  `src/urop/tau_bench_adapter.py` (11 passing CPU tests against a synthetic
  fixture shaped like the real schema, plus a live check against the real
  vendored tau-bench source) maps tau-bench's raw schema to ours -- see
  `docs/methods_and_deviations.md` §9. Running
  `scripts/convert_tau_bench.py` on the real download: **0 of 460 records
  rejected**, converted, task-split (92 train tasks / 23 test tasks, seed
  42), then sampled (`src/urop/tau_bench_adapter.sample_train_records`,
  `random.Random(seed=42).sample()` over the full set of train-side task
  IDs, one trial per task, no outcome filtering -- see
  `docs/methods_and_deviations.md` §11 for why this is the only sampling
  method that actually supports an "unbiased-selection" claim; an earlier
  revision used ascending task-id order and wrongly called that unbiased)
  into samples of 3 (smoke) and 30 (pilot) real trajectories on the train
  side. Exact selected (task_id, trial) pairs recorded in
  `data/tau_bench/{smoke,pilot}_sample_manifest.json`. Full counts in
  `data/tau_bench/conversion_report.json`.
- **Real tool schemas reconstructed from source**, not fabricated: all 16
  retail tools' `get_info()` return values, parsed via `ast.literal_eval`
  directly from `external/tau-bench/tau_bench/envs/retail/tools/*.py`.
- **Approximate, informational-only token-length stats computed on the real
  data** (tiktoken `cl100k_base` -- an approximation for a GPT-family
  tokenizer, not the Qwen tokenizer the candidate agent model actually
  uses, and not the exact rendered open-coding prompt; see
  `docs/compute_budget.md`): across all 460 converted trajectories, min
  1,409 / median 3,733 / mean 3,890 / max 11,147 tokens; **1 of 460 (0.2%)
  exceeds the candidate 8,192-token context limit** and must be excluded or
  chunked, never silently truncated. The (now randomly-sampled)
  3-trajectory smoke sample is comfortably within budget (1,553-5,159
  tokens). The **authoritative** check -- real Qwen tokenizer, exact
  rendered prompt including a JSON-repair retry round -- runs in
  `notebooks/01_kaggle_text_pilot.ipynb` Section 5, not here.
- **CPU pipeline re-verified end-to-end on real data**, not just synthetic:
  `scripts/prepare_data.py` and `scripts/run_text_pilot.py` (mock backend)
  both run cleanly against the real converted corpus/smoke sample -- see
  `docs/experiment_log.md`'s 2026-09-28 entry.
- **`notebooks/01_kaggle_text_pilot.ipynb` upgraded from a stub to real
  wiring**, still gated and still not executed: it now actually constructs
  and calls `external/autotracegt_upstream`'s real `OpenCodingAgent`
  (`concurrency=1`, 1 bounded JSON-repair retry, per-record results saved
  to disk immediately) against the real converted smoke sample, with a
  token-limit check before scheduling any call. Every code cell still
  compiles and the notebook is still valid `nbformat`; it still cannot be
  executed from this machine.

## Partially resolved 2026-09-29 (first real Kaggle GPU run)

Items 1-2 below are no longer fully unverified -- a real run happened.
See `docs/experiment_log.md` for the complete, evidence-based account and
`docs/methods_and_deviations.md` §15 for the fixes made in response.
**What is now confirmed:** Kaggle's internet/GPU/torch-cuda stack works
(`torch==2.10.0+cu128`, Tesla T4, 15.5 GB free); vLLM 0.18.1 installs and
its server entry-point imports; Qwen2.5-3B-Instruct
(`aa8e72537993ba99e69dfaafa59ed015b17504d1`) loads and serves real
requests over the OpenAI-compatible API, given a CUDA-linker workaround
that is now a permanent notebook cell. **What is still NOT confirmed:**
`--enable-auto-tool-choice --tool-call-parser hermes` producing valid tool
calls (not exercised by open coding, which doesn't call tools itself);
whether the coding pipeline produces ANY usable output at all -- the one
real run so far produced zero (2 parse failures, 1 context-limit block);
whether the new structured-JSON output mode (§15) actually fixes that
against a real server, which is exactly what the next one-trajectory
diagnostic run (`docs/kaggle_quickstart.md`) checks.

## Not verified -- outstanding, in priority order

1. ~~Kaggle's actual state~~ -- **resolved 2026-09-29**, see above.
2. **Whether the coding pipeline (either output mode) produces usable
   output on a T4** -- the one real run so far produced zero usable
   outputs under the freeform mode; the structured-JSON mode has never
   been run against a real server. This is the single most important open
   question and exactly what the next Kaggle run is for.
3. **The axial/codebook-manager prompt's token budget** -- only checked at
   the individual-trajectory level so far (see above). The axial-coding
   prompt ingests a whole batch's open-coding *output*, which doesn't exist
   until real inference runs once; re-run the same style of check against
   real open-coding output before ever scheduling a real axial-coding call.
4. **The four "unverified" literature-survey papers** (arXiv 2608.27750,
   2605.07990, 2605.14038, 2604.19775) -- not checked this session; still
   carry "unverified" status from the original brief. Do not cite them.
5. **Hidden Error Awareness numbers** (in the prior deck) against their
   primary paper (arXiv 2605.09502) -- not re-verified this session. Do not
   put deck numbers in the new presentation before this is done.
6. **The live `ZhuoranLu/Qual-Agent-Behavior-Analysis` GitHub repo** has
   still not been fetched to diff against the supplied ZIP, even though
   GitHub access is now confirmed working (used to clone tau-bench).
7. **`sonnet-35-new-retail.json`** (the other real historical-trajectory
   file tau-bench ships, 25.3 MB) was deliberately not downloaded -- one
   real source was enough for this sprint's scope.

## Go / no-go

**Go, for this sprint's actual scope (local CPU setup + real-data
preparation):** done, see "Verified this session" / "Resolved in the
2026-09-28 follow-up session" above and `docs/compute_budget.md` /
`docs/experiment_log.md` for what ran and how long it took.

**Partial go, for a one-trajectory diagnostic run:** the server startup
path is now confirmed working (2026-09-29, see above), so a bounded,
single-trajectory diagnostic of the coding pipeline itself (outstanding
item 2 above) is reasonable next-step scope -- exactly what
`notebooks/01_kaggle_text_pilot.ipynb` is currently configured for
(`docs/kaggle_quickstart.md`). **If that run also fails or returns invalid
evidence, stop and preserve the artifacts rather than launching another
one** -- do not escalate model size, token limits, or retries to paper
over a failure without evidence justifying it.

**No-go, still, for Phase 1 (fresh multi-trajectory/multi-episode
generation with a served model):** blocked on outstanding item 2 above --
zero usable coding output has been produced yet, under either output mode.
tau-bench itself is fully inspected and real historical trajectories/
samples already exist locally -- what's left is specifically whether the
coding pipeline itself works at all against a real served model.

**No-go, still, for any run beyond the one-trajectory diagnostic:** the
gate cell in `notebooks/01_kaggle_text_pilot.ipynb` is deliberately
unsatisfied by default and now includes an explicit "this is a one-
trajectory diagnostic, not a smoke test" acknowledgment flag. Do not
expand scope until outstanding item 2 above is resolved with real,
recorded evidence and the user has given an explicit go-ahead per the
compute plan (`docs/compute_budget.md`).
