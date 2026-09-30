# Research plan (summary)

Full context lives in `reference_materials/urop_context_and_instructions_for_executing_ai.md`
and `reference_materials/CLAUDE_CODE_UROP_HANDOFF.md` (both git-ignored, local
only). This file is the short pointer version for anyone reading the repo.

## People

Atharva Shukla (NUS Year 2 Computer Engineering), advised by Prof. Hong
Junyuan (NUS ECE), with Sanchit collaborating on the shared UROP theme.
Showcase: Thursday, 8 October 2026 (Asia/Singapore).

## Question

Prior work presented to the advisor covered hidden-state linear probes for
single-model reasoning correctness. This sprint moves to **agents**: can
hidden states at an agent's first state-changing action distinguish cases
where the required task state was not resolved before acting, beyond text
and simple position baselines?

- **Assigned method to replicate/build on:** AutoTraceGT (arXiv 2608.30391v1)
  -- an LLM-only, text-level grounded-theory pipeline that discovers
  behavioral categories from agent trajectories. It has no hidden-state
  component; that's our addition.
- **Motivating finding (theirs, on Tau-Bench):** successful trajectories
  resolve the relevant task state (identity, records, constraints, payment
  status) into a coherent plan before an irreversible action; failed
  trajectories often postpone that alignment ("T3").
- **H1 (primary, falsifiable):** a small linear classifier (standardize ->
  L2 logistic regression, C=0.1) can distinguish T3-flagged trajectories at
  a defined decision point better than text/position controls, on held-out
  tasks.
- **H2 (secondary, deferred this sprint):** a probe for tool-call dependency
  edges is less confident on T3-flagged trajectories. Not authorized to
  start until H1's infrastructure exists and there's time left.
- **H0 (equally valid outcome):** no separation beyond controls. That is a
  real, reportable result, not a failure of the sprint.

This is **diagnosis, not steering or fine-tuning**. A predictive signal is
not proof of causation, awareness, or novelty. See
`docs/methods_and_deviations.md` for every deliberate deviation from the
paper/its code, and `docs/source_audit.md` for what's been verified.

## Phases (see `docs/compute_budget.md` for the time/GPU budget per phase)

0. **Recon and feasibility** -- this repo's current state. See
   `docs/phase0_findings.md` for the go/no-go.
1. **Agent + trajectories** -- generate fresh trajectories with an
   open-weight agent model through the unmodified tau-bench harness. Not
   started; blocked on tau-bench inspection (Phase 0 item 1).
2. **Labels** -- AutoTraceGT-style text coding (reusing
   `external/autotracegt_upstream`) for a T3-like category, plus an
   independent rule-based proxy label encoding the actual tau-bench retail
   policy text. Not started; blocked on Phase 1 and the same policy-text
   read as the proxy label.
3. **Hidden states** -- teacher-forced extraction via HuggingFace
   `transformers`, exact chat-template/tool-schema reconstruction, at a
   decision position fixed in writing before any extraction (see
   `docs/methods_and_deviations.md` §7). Not started.
4. **Probes + controls** -- the H1 probe plus mandatory controls
   (position-only, text-only/TF-IDF, label-permutation, control-position,
   within-failure subgroup). Not started; no metric-reporting path exists
   yet, and `src/urop/reporting.py` will refuse to report one from anything
   but a real backend.
5. **Analysis + demo materials** -- see `docs/presentation_outline.md`.

## Longer-term (context only, not this sprint)

Target venues: COLM 2027 (primary), ICML 2027 (internal pacing checkpoint).
Timeline: Aug-Sep 2026 literature/question, Sep-Oct infrastructure and
baselines, Oct-Nov methodology and pilot, Dec UROP report, Jan-Apr expanded
work and writing. A negative H1 result changes direction, not the plan.
