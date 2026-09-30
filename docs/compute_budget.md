# Compute budget

Limits, not promised runtimes. No GPU inference has been timed -- everything
GPU-related below is a proposed conservative default for a later,
user-triggered run, not a measured estimate or an authorization to start.

## CPU stages (measured, this session)

| Stage | What ran | Result |
|---|---|---|
| Source/data checks | `scripts/inspect_sources.py` | instant, no GPU |
| `pytest tests/` | 25 tests, Python 3.12 venv (`uv`) | **0.59s**, 25/25 passed |
| `scripts/prepare_data.py` (synthetic fixture) | load + validate + split | instant |
| `scripts/run_text_pilot.py --config configs/text_smoke.yaml` | mock-backend coding dry run, 3 records | well under 1s (mock has no real latency) |
| `notebooks/00_cpu_data_check.ipynb` | full notebook, executed via `nbclient` | all 11 cells succeeded, a few seconds total |

These numbers say the CPU foundation is fast and correct on tiny synthetic
input. They say **nothing** about GPU coding-call latency or Kaggle's actual
hardware -- do not extrapolate timing from them. Volume and token-length are
no longer synthetic-only, though -- see the real numbers below.

## Real-data volume and token lengths (measured, 2026-09-28)

From `scripts/convert_tau_bench.py` run against the real download
(`data/tau_bench_raw/gpt-4o-retail.json` -- see `docs/source_audit.md`):

| Quantity | Value |
|---|---|
| Raw records | 460 (0 rejected) |
| Distinct tasks | 115 (test split) |
| reward = 1.0 / reward = 0.0 | 278 / 182 |
| Train / test tasks (seed 42, 80/20) | 92 / 23 |
| Train / test trajectories | 368 / 92 |
| Smoke sample (seeded random draw over train task IDs, seed 42) | 3 trajectories |
| Pilot sample (seeded random draw over train task IDs, seed 42) | 30 trajectories (of 92 train-side available) |

Approximate token counts (tiktoken `cl100k_base` -- an approximation for
whatever tokenizer the eventual agent model actually uses, **not exact**;
re-check with the real tokenizer once a model is loaded):

| Sample | n | min | median | mean | max | over candidate 8,192-token limit |
|---|---|---|---|---|---|---|
| All 460 converted trajectories | 460 | 1,409 | 3,733 | 3,890.1 | 11,147 | 1 (0.2%) |
| 3-trajectory smoke sample (seeded random, seed 42) | 3 | 1,553 | 4,319 | 3,677.0 | 5,159 | 0 |

**These tiktoken `cl100k_base` numbers are informational only** -- they
approximate a GPT-family tokenizer, not the Qwen tokenizer the candidate
agent model actually uses, and they tokenize raw message content, not the
exact rendered open-coding prompt (system + user template + tools listing +
trajectory, as `agents.open_coding.format_chunk`/`load_prompt` build it) or
a JSON-repair retry round. The **authoritative** context-limit check runs in
`notebooks/01_kaggle_text_pilot.ipynb` against the real Qwen tokenizer and
the exact rendered prompt once a model is loaded -- see that notebook's
Section 5. A trajectory found to be over-budget there must be excluded or
chunked, never silently truncated. The axial/codebook prompt (which ingests
a whole batch's open-coding *output*, not raw trajectories) has not been
token-checked at all yet -- that output doesn't exist until real inference
runs once; see `docs/phase0_findings.md`.

## GPU/API stages (proposed limits, NOT measured -- from the handoff's compute plan)

| Stage | Initial scope | Stop budget |
|---|---|---|
| Source/data checks | CPU only | No GPU allocation |
| GPU compatibility/load check | One selected model, minimal request | 15 minutes |
| Text smoke test | 3 training trajectories, one tiny coding batch | 30 minutes |
| Text pilot | At most 30 training trajectories | 2 hours |
| Optional expansion | Up to 90 total training trajectories, batch 30 | Only after a measured projection from the pilot, and a separate user go-ahead |
| Optional fresh-agent smoke | 3-5 episodes | 1 hour, separate go-ahead |
| Optional extraction smoke | 5 eligible prefixes | 30 minutes, separate go-ahead |

Treat every smoke-scale run as an engineering check, not a comparable
codebook or saturation evidence. Freeze the real pilot settings only after
smoke tests are clean. Do not combine incompatible configurations
(different model, different chunk size, different saturation threshold)
into a single run without recording the change.

## Model/GPU strategy (candidate, unverified -- see `docs/phase0_findings.md`)

- One small instruction model (candidate: `Qwen2.5-3B-Instruct`) on **one**
  T4, FP16, batch/concurrency 1. Leave the second T4 unused unless a
  measured benefit justifies it.
- No FlashAttention-2 assumed by default; verify the actual supported
  attention backend on Kaggle's stack before selecting one.
- Candidate context limit 8192 tokens, max new tokens 1024 per coding call --
  tokenize actual inputs before scheduling; never silently truncate.
- No training/fine-tuning, no model sweep, no multiple seeds, no parallel
  codebooks, no repeated full-corpus reruns, no quantization changes, by
  default.

## API cost

Default paid API budget is **zero**. `scripts/run_text_pilot.py` hard-refuses
to run against `openai`/`deepseek`/`gemini` backends (exit code 2, no
network call attempted) -- see `docs/methods_and_deviations.md` §5 for why
the text-coding backbone is planned to be the same local model, not a paid
one. Any future paid-API use needs an explicit cost estimate and a separate
user go-ahead before code changes, not just a config edit.

## Budget mechanics (implemented, CPU-tested)

- `src/urop/budgets.Budget` + `call_with_watchdog`: cancels a call once the
  *total* remaining time budget hits zero, including mid-call -- not just
  between calls. Tested against a deliberately slow mock call in
  `tests/test_budget_and_resume.py`.
- `src/urop/budgets.ResumeCache`: skips work only when a cache key is marked
  completed **and** its result file still exists on disk; a changed model,
  prompt, settings, or data hash produces a different cache key and is never
  silently reused.
- `src/urop/reporting.summarize_run`: refuses to attach a metric to a run
  labeled `backend_label="mock"`/`"synthetic"`.
