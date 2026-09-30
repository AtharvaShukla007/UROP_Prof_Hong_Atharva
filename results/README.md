# results/

Small, curated summaries only -- the output of `scripts/summarize_run.py`
and similar, not raw run directories (those go under git-ignored `runs/`).

Nothing has been placed here yet: no probe stage has run, and
`src/urop/reporting.py` refuses to produce a metrics summary from anything
but a real (non-mock) backend. See `docs/phase0_findings.md` for what's
still outstanding before that changes.

A results bundle, once real, should be: config/manifest, timings, counts,
raw coding outputs or a pointer to them, codebook revisions, errors, and the
stop reason -- no weights, no huge arrays (those belong under git-ignored
`runs/`).
