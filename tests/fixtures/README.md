# tests/fixtures/

Tiny, hand-written, **synthetic** trajectories for exercising `src/urop/`
offline. None of this is real tau-bench (or any other) data -- do not use it
in any research summary (`reporting.summarize_run` refuses to attach metrics
to a `backend_label` of `mock`/`synthetic` for the same reason).

- `tiny_trajectories.jsonl` -- 8 raw records: some valid (with `tool_calls`
  preserved, one task with two trials), some deliberately invalid (missing
  `resolved`, wrong first-message roles, an exact duplicate) so
  `src/urop/data.py`'s validation path has something to reject.
