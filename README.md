# UROP_Prof_Hong_Atharva
NUS UROP: studying agent failure behaviours through trajectory analysis and hidden-state probing.

Diagnosing agent decision-point failures from hidden states, building on
[AutoTraceGT](https://arxiv.org/abs/2608.30391)'s text-level grounded-theory
coding of agent trajectories. Advisor: Prof. Hong Junyuan (NUS ECE).
Status and scope: see `docs/phase0_findings.md` -- as of now this is a
CPU-verified local foundation plus a gated, unexecuted Kaggle GPU notebook;
no GPU inference or real research results exist yet.

## Setup (CPU, local)

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements/cpu.txt --python .venv/Scripts/python.exe
uv pip install -e . --python .venv/Scripts/python.exe
.venv/Scripts/python.exe -m pytest tests/ -v
```

Then, optionally:

```bash
.venv/Scripts/python.exe scripts/inspect_sources.py     # source/reference audit
.venv/Scripts/python.exe scripts/prepare_data.py         # load/validate/split (defaults to the synthetic test fixture)
.venv/Scripts/python.exe scripts/run_text_pilot.py --config configs/text_smoke.yaml
```

Real tau-bench retail data (not shipped -- git-ignored) can be reproduced
locally with `docs/kaggle_quickstart.md`'s step 1 commands, then converted:

```bash
.venv/Scripts/python.exe scripts/convert_tau_bench.py   # -> data/tau_bench/tau_bench_train_{3,30}.jsonl
.venv/Scripts/python.exe scripts/run_text_pilot.py --config configs/text_smoke_real_tau_bench.yaml
```

Open `notebooks/00_cpu_data_check.ipynb` with the `urop-cpu` kernel
(`.venv/Scripts/python.exe -m ipykernel install --user --name urop-cpu`) to
run the same checks interactively.

## Layout

- `src/urop/` -- data validation/cleaning, deterministic splits, budgets and
  a resume cache, provenance manifests, a reporting layer that refuses to
  fabricate research metrics, and the bridge to vendored AutoTraceGT code.
- `scripts/` -- thin CLIs around `src/urop/`.
- `configs/` -- one YAML per run stage (`text_smoke.yaml` runs today against
  a mock backend; the rest are candidate designs, see `docs/phase0_findings.md`).
- `notebooks/` -- `00_cpu_data_check.ipynb` (executed, CPU-only) and
  `01_kaggle_text_pilot.ipynb` (gated, **not yet run** -- see
  `docs/kaggle_quickstart.md`).
- `docs/` -- research plan, Phase 0 findings/go-no-go, documented deviations
  from AutoTraceGT, compute budget, source audit, experiment log,
  presentation outline, Kaggle quickstart.
- `tests/` -- CPU-only pytest suite against hand-written synthetic fixtures.
- `external/`, `reference_materials/`, `data/`, `runs/` -- git-ignored, local
  only (vendored author code, user-supplied source docs, real corpus data,
  raw run outputs). See `CLAUDE.md` and `external/PROVENANCE.md`.

## Rules this repo enforces in code, not just in docs

- No paid API call, ever, by default (`scripts/run_text_pilot.py` hard-
  refuses paid backends).
- A local no-auth credential can never be registered against a non-local
  endpoint (`src/urop/autotrace_adapter.register_local_endpoint`).
- A run cannot be summarized with a research metric unless its
  `backend_label` marks it as real inference (`src/urop/reporting.py`).
- Raw input files are never overwritten by any adapter (`src/urop/data.py`).

See `CLAUDE.md` for the full list and where everything lives.
