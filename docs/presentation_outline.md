# Presentation outline (~9 slides)

Outline only -- **no plots, numbers, or results exist yet.** Fill each
section in only from what actually ran, and mark anything unrun as
"design, not yet executed."

1. **Prior work, briefly.** Hidden-state linear probes for single-model
   reasoning correctness (Hidden Error Awareness recipe as the template) --
   one line: what was shown before, not re-litigated here.
2. **Move to agents.** Why a reasoning trace's single endpoint doesn't carry
   over to a multi-step agent trajectory; the decision-point definition
   problem (see `docs/methods_and_deviations.md` §7).
3. **AutoTraceGT.** What it is (text-only, grounded-theory taxonomy
   induction), the T3 finding on Tau-Bench, and that it has no hidden-state
   component -- that's the gap this sprint targets.
4. **Our question.** H1/H2/H0 as stated in `docs/research_plan.md`, framed
   as falsifiable, with the explicit caveat that a null result is a valid
   outcome.
5. **Implementation.** What was actually built this sprint: repo structure,
   CPU-verified data/split/budget/resume machinery, the gated Kaggle
   notebook -- and what's still a design/config, not run. Be precise about
   the boundary.
6. **Text-side results (if any ran).** Codebook categories, saturation
   curve (ADD count *and* rate per round -- see `docs/methods_and_deviations.md`
   §2), whether a T3-like category emerged, judge/rule agreement. If Phase 2
   didn't run before the showcase, this slide says so plainly and shows the
   design instead.
7. **Probe results, or the not-yet-run design.** AUROC table with bootstrap
   CIs and every control, if Phase 4 ran; otherwise the frozen probe design
   (layer candidates, decision position, controls list) with an explicit
   "not yet run" label. Never present one as the other.
8. **Limitations.** Small n, single (candidate, unverified) model, LLM-judge
   dependence with no human validation available, decision-point definition
   is a design choice, correlation is not causation, diagnostic only, our
   codebook's coding backbone is weaker than the paper's (see
   `docs/methods_and_deviations.md` §5).
9. **Next steps.** Concrete, scoped to what's actually blocking progress --
   pull from `docs/phase0_findings.md`'s outstanding-items list, not a
   generic "future work" slide.

Plus a **3-minute spoken summary** Atharva can say without notes, written
only once slides 1-9 are filled in with real content.
