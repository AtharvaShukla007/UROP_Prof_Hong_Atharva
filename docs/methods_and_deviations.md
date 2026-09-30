# Methods and deviations from AutoTraceGT

Every place this project's code or plan deliberately differs from the
AutoTraceGT paper (arXiv 2608.30391v1) or its shipped code
(`external/autotracegt_upstream/`), and why. See `docs/source_audit.md` for
what was verified to arrive at each claim below.

## 1. Missing/invalid outcome is rejected, not defaulted to failure

Upstream `utils/trajectory_processing._annotate_tau`:
`"resolved" if record.get("resolved") else "failed"`. A record with no
`resolved` key, or one that is `None`/a string/an int, silently becomes a
`"failed"` label.

`src/urop/data.validate_raw_record` rejects any record where `"resolved"`
is absent or not a `bool`, before cleaning ever runs.

**Why:** a missing label is not evidence of failure. Folding "we don't know"
into "it failed" would quietly inflate the failure class and make our own
codebook/probe results look cleaner than the underlying data supports.

**How to apply:** any adapter for a new dataset must define its own
explicit outcome extraction and reject what it can't determine -- never
reuse `_annotate_tau`'s fallback-to-failure pattern.

## 2. Saturation rule: fraction semantics kept, threshold made explicit per run

See `docs/source_audit.md` for the full discrepancy: the paper's Algorithm 1
literally defines the per-round quantity as a raw count, its own prose calls
it a rate, and the shipped code implements the rate (fraction) version with
a default threshold of 0.1 -- while the paper's *reported* experiments used
0.2.

We keep the shipped code's `add_rate_streak` (fraction) semantics, since
it's the only one that is actually implemented and testable. Every
`configs/*.yaml` run must record both the raw ADD-action counts and the
fraction per round (not just whichever crosses the threshold), and state
which threshold value was used. **Do not describe any of our runs as an
exact replication of the paper's saturation criterion** -- the paper and its
own code disagree with each other, so "matching the paper" is not a
well-defined target.

## 3. A local dummy API key can never authenticate a real remote endpoint

`src/urop/autotrace_adapter.register_local_endpoint` refuses to register its
no-auth dummy key against any `base_url` whose host isn't `localhost`,
`127.0.0.1`, or `0.0.0.0`. A genuine paid/remote provider must go through
`external/autotracegt_upstream/autotracegt/llm/endpoints.register_endpoint`
directly, with a real `api_key_env`.

**Why:** upstream's endpoint registry only requires that
`os.environ[api_key_env]` be set -- it does not check whether the key is
real or whether the `base_url` is local. Without our own guard, a typo'd or
copy-pasted `base_url` could route a "local, free" call at a paid provider
using a dummy credential, either failing confusingly or (worse) succeeding
against a provider that doesn't actually check the key.

## 4. Real per-request cancellation, not an abandoned thread

**Revised after a review caught a real flaw.** An earlier version of this
module wrapped calls in a `ThreadPoolExecutor` and abandoned the worker
thread on timeout (`call_with_watchdog`). That is not real cancellation:
the caller returns early, but the network call the thread was blocked on
keeps running to completion in the background -- against a server we've
already decided to stop paying for.

`src/urop/budgets.bounded_llm_call` (the replacement) calls
`fn(timeout=<bounded>, **kwargs)` synchronously, in the caller's own
thread -- no thread pool, nothing left running once it returns or raises.
This relies on `fn` honouring its own `timeout` kwarg to abort the
underlying request at that deadline, which is a real, standard capability
of an OpenAI-SDK-style client (`.create(timeout=...)`, enforced by its
`httpx` transport) -- not something faked here. `MockCoder` and
`tests/fake_local_server.FakeLocalChatServer` implement the same contract
cooperatively, so a CPU test against them exercises real cancellation
semantics, not just "the caller gave up waiting."

`src/urop/autotrace_adapter.install_budget_guard` routes every actual
outbound request an `OpenCodingAgent` makes -- the first attempt, every
JSON-repair retry, every chunk -- through `bounded_llm_call` by
monkeypatching `agent.client.chat.completions.create` in place (no
upstream source edited). This also replaces upstream's own
`llm/endpoints.chat_with_retry` (up to 15 retries, exponential backoff,
worst case minutes per call -- fine for an attended paid job, not for a
budget-capped Kaggle smoke test) for the sync path we actually use, and
caps the openai SDK's own internal retry count at 0
(`agent.client = agent.client.with_options(max_retries=0)`) so a transient
network error can't silently multiply attempts underneath our own counting.

**A subtlety that broke on first attempt, worth recording:** deciding
whether a failed call "counts as budget exhaustion" cannot be done by
re-checking `Budget.exhausted()` after the call returns -- that races
against exactly when the underlying HTTP client's own timeout fires, and
was observed (in this project's own test suite, only when run as part of
the full suite, not in isolation) to occasionally misclassify a genuine
budget-caused timeout as a plain failure. It also cannot be decided purely
from which timeout bound was active for the call (an early attempted fix)
-- an immediate `ValueError` from a call given a budget-bound 60s timeout
is not a budget problem. The fix: measure how long *this specific call*
actually ran before failing, and only attribute the failure to the budget
if that duration is close to the timeout that was requested. See the
`bounded_llm_call` docstring.

Separately, `src/urop/process_control.shutdown_process`
(terminate -> wait -> escalate to kill) and `wait_for_server_ready`
(polls whether the owned server subprocess has died, instead of polling a
health check that will never succeed for the rest of a timeout) handle the
one thing that genuinely *can* be force-stopped: an owned OS process, not a
Python thread. The Kaggle notebook wraps its real-inference cell in
`try/finally` so the server subprocess is terminated on any exception, not
only a clean run.

## 5. Text-first pilot uses a local open-weight model as the *coding* backbone too

The paper's own runs use GPT-5-mini (etc.) as the coding backbone
(OpenCode/AxialCode/Manage), at $0.74-$1.91 per full Tau-Bench codebook. Our
default paid API budget is **zero** (handoff §2). The plan is therefore to
serve one local open-weight model on the Kaggle T4 and register it as the
`"local"` endpoint for *both* the agent-under-study and the coding backbone,
rather than paying for GPT-5-mini-class coding calls.

**Consequence, stated up front:** our codebook quality depends on a weaker,
unverified coding model than the paper used. Any codebook we produce is not
comparable in quality to the paper's GPT-5-mini/GPT-5 codebooks, and that
must be said explicitly in `docs/presentation_outline.md`, not left implicit.

## 6. GPT-OSS-120B excluded; agent/coding model choice is a documented candidate, not a tested one

The paper's largest open-weight backbone (GPT-OSS-120B) is not a practical
local baseline on a single T4 (16 GB nominal VRAM) -- excluded outright, not
downloaded. `Qwen2.5-3B-Instruct` / `Qwen2.5-7B-Instruct` are candidates
named in `configs/*.yaml` and the original brief; neither has been verified
against Kaggle's actual T4 software stack, chat template, or tool-schema
support in this session. Treat every mention of a specific model name in
this repo as "candidate" until `docs/phase0_findings.md` says otherwise.

## 7. Decision-point definition (for the eventual probe) is fixed in docs before any data collection

Per the handoff (§7), the primary decision position for H1 is: *the last
input token immediately before generation of the assistant turn that emits
the first state-changing tool call*, extracted from that turn's prefix only
(no tool name/arguments/results from that turn itself). An alternative
post-emission/pre-execution position (which *does* include the action
name/arguments) answers a different question and must never be mixed with
the primary position under a shared "before acting" label if both are ever
tried. Neither position has been extracted from real data yet -- this
section exists so the definition is settled in writing before Phase 3
starts, not decided ad hoc while writing extraction code.

## 8. Author code stays vendored and git-ignored, not redistributed

`external/autotracegt_upstream/` (from the supplied ZIP) ships without a
`LICENSE` file. It is kept out of the public GitHub repo
(`external/` is git-ignored) and only referenced by absolute-relative import
paths from `src/urop/autotrace_adapter.py`, never copied into tracked files.
See `external/PROVENANCE.md`.

## 9. tau-bench's raw historical-trajectory schema does not match AutoTraceGT's

Verified by downloading and parsing the real file (see
`docs/source_audit.md`), not assumed from either project's docs. tau-bench's
`historical_trajectories/*.json` records look like:

```json
{"task_id": 0, "reward": 1.0, "trial": 0, "info": {...}, "traj": [...]}
```

AutoTraceGT/`src/urop/data.py` expects:

```json
{"instance_id": "0", "run_id": "trial_0", "messages": [...], "tools": [...], "resolved": true}
```

`src/urop/tau_bench_adapter.py` converts one into the other:
`task_id -> instance_id` (stringified), `trial -> run_id` (as `trial_<n>`),
`traj -> messages` (already OpenAI chat format with `traj[0]=system`,
`traj[1]=user`, so no reshaping needed there), `reward -> resolved`
(`True` iff `reward == 1.0` exactly; any other value, including a `bool`
`reward`, is rejected outright -- **not** coerced, same "reject rather than
mark as failure" rule as §1 above applies to a field that is present but
wrong-shaped, not just missing).

**tau-bench's raw records carry no `tools` field at all.** The 16 retail
tool JSON-schemas (`{"type": "function", "function": {...}}`, OpenAI
function-calling shape) are reconstructed by
`tau_bench_adapter.extract_retail_tool_schemas`, which parses the literal
`return` statement of each `get_info()` static method in
`external/tau-bench/tau_bench/envs/retail/tools/*.py` via Python's `ast`
module (`ast.literal_eval` on the return expression) -- not by importing the
`tau_bench` package (whose top-level `__init__.py` pulls in
openai/anthropic/mistralai/google-generativeai/litellm just to reach a
handful of static dicts) and not by hand-transcribing the schemas (which
would risk silent transcription errors). If tau-bench's tool definitions
ever change, re-running the extractor picks up the change automatically;
nothing is cached as a static copy in this repo.

## 10. tau-bench's older, "not updated" repo was pinned on purpose

`external/tau-bench`'s own README says outdated tasks and points to
τ³-bench as the maintained successor (`docs/source_audit.md` has the exact
wording). We pinned the older repo anyway
(`59a200c6d575d595120f1cb70fea53cef0632f6b`) because the real historical
trajectories we use (`data/tau_bench_raw/gpt-4o-retail.json`) were generated
against *this* commit's retail task/policy text -- switching to τ³-bench's
fixed tasks would silently invalidate the trajectories' ground truth
(`task.actions`, `reward_info.gt_data_hash`) without any error. Revisit only
if a comparable set of historical trajectories becomes available for
τ³-bench, and re-derive everything (tool schemas, policy text, task counts)
from that repo rather than assuming they're unchanged.

## 11. Smoke/pilot sample selection is seeded random sampling, not "ascending order is unbiased"

An earlier revision of `scripts/convert_tau_bench.py` selected the smoke (3)
and pilot (30) trajectories by sorting train-side tasks in ascending
task-id order and truncating, and called that selection "unbiased" in code
comments and in `docs/compute_budget.md`/`docs/phase0_findings.md`.
**That claim was wrong.** Ascending numeric order is an arbitrary,
non-random ordering -- it happens not to correlate with outcome for this
corpus, but nothing about "sorted by id" *guarantees* that, and the term
"unbiased" implies a randomization guarantee that sorting doesn't provide.

`src/urop/tau_bench_adapter.sample_train_records` now does an actual
`random.Random(seed).sample()` draw over the full set of train-side task
keys (one trial -- the lowest-numbered -- per task, so multi-trial tasks
aren't over-weighted), with no filtering by outcome or any other category.
This *does* support an unbiased-with-respect-to-outcome claim, because the
draw has no mechanism by which outcome could influence selection. It still
does not guarantee representativeness on other axes (task length, category,
etc.) -- a random sample of task IDs is not automatically a stratified or
representative sample of everything about the corpus, and no such claim is
made here either.

Every sample now writes a manifest (`data/tau_bench/{smoke,pilot}_sample_manifest.json`)
recording the exact `(instance_id, run_id)` pairs selected, the seed, and
the pool size sampled from -- so the selection is auditable and exactly
reproducible (`--sample-seed`, independent of `--seed`, which controls only
the train/test split), not just described in prose.

## 12. Synthetic fixtures are wiring tests, never research inputs

`tests/fixtures/tiny_trajectories.jsonl` is hand-written and clearly
labeled as synthetic. `src/urop/reporting.summarize_run` raises
`NotResearchDataError` if asked to attach metrics to a run whose
`backend_label` is `"mock"`/`"synthetic"` (see `tests/test_budget_and_resume.py`).
No script in this repo currently computes a probe metric at all -- that is
Phase 4, not built this sprint -- but the refusal path is in place now so it
cannot be bypassed later by accident.

## 13. Retry semantics corrected: upstream's MAX_JSON_RETRIES counts total attempts

**A real bug, caught by review, that would have silently disabled JSON
repair entirely.** Upstream's `agents/open_coding.OpenCodingAgent.code_chunk`
loop is `for attempt in range(MAX_JSON_RETRIES): ...; if attempt <
MAX_JSON_RETRIES - 1: <append repair messages>`. `MAX_JSON_RETRIES` is
therefore the **total number of attempts** (initial + repairs), not the
number of repairs on top of a first try. The Kaggle notebook previously set
`JSON_RETRIES=1` intending "one repair retry" -- with `MAX_JSON_RETRIES=1`,
`attempt < 0` is never true, so the repair branch never runs: that
configuration silently permitted **zero** repairs, not one.

Fixed: the notebook now sets `JSON_RETRIES=2` (1 initial + 1 repair = 2
total attempts) via `os.environ["JSON_RETRIES"]`, set before the first
import of any upstream module in the process (module-level constant, read
once). `tests/test_open_coding_integration.py::test_malformed_response_triggers_exactly_one_repair_attempt`
verifies, against the real `OpenCodingAgent` and a real fake server, that a
malformed-then-valid response sequence takes exactly 2 real HTTP requests
under this setting -- not 1 (the old, broken config) and not more.

## 14. Token-limit enforcement moved from "ahead of time, once" to "every actual request"

An earlier design pre-checked only the first chunk's prompt, ahead of time,
using an approximate tokenizer, then scheduled the whole trajectory. That
missed two real cases: a JSON-repair round's message list (original prompt
+ the raw response we got back + a repair instruction) is larger than the
original prompt and wasn't re-checked, and a trajectory with more than one
open-coding chunk has later chunks whose prompt depends on a
model-generated `segment_memo` from the previous chunk -- unknowable ahead
of time.

`src/urop/autotrace_adapter.install_budget_guard` now checks tokens as
part of the same choke point that handles cancellation (§4): every time
`agent.client.chat.completions.create` is about to be called -- the first
attempt, a repair retry, or any chunk -- it tokenizes the EXACT `messages`
about to be sent via `tokenizer.apply_chat_template` (the real,
server-compatible chat template, not a reimplementation) and raises
`PromptTooLongError` rather than send an oversized request. Never silently
truncates. `tests/test_open_coding_integration.py::test_token_check_runs_on_every_actual_request_including_repair`
verifies this against a real repair round whose grown message list would
overflow a budget the first attempt alone fits comfortably within.

The standalone tiktoken-based check that runs ahead of time (before any
model is loaded) is kept, but is explicitly informational-only now -- it
approximates a GPT-family tokenizer, not Qwen's, and only ever looks at raw
message content, never the exact rendered/templated prompt. See
`docs/compute_budget.md`.

## 15. Opt-in structured (JSON-schema-constrained) output mode -- a deliberate deviation from the paper's own generation format

**Trigger: a real, confirmed failure, not a hypothetical one.** The first
real Kaggle GPU run (2026-09-29, see `docs/experiment_log.md`) found
Qwen2.5-3B-Instruct's freeform `===CODES===`-block output embedding a
tool-call's own JSON verbatim inside a `"quote"` string value WITHOUT
escaping its inner double quotes --
`"quote": "<tool_call name=X>{"key":"value"}"` -- which breaks
`json.loads` on the surrounding CODES array outright. This is not "almost
valid JSON" that a repair retry can fix (upstream's own JSON-repair loop
tries and fails on exactly this pattern -- see
`tests/test_open_coding_integration.py::test_unescaped_quotes_inside_tool_call_quotation_are_rejected`,
run against the real vendored `agents.open_coding` module).

`src/urop/structured_coding.py` asks the SERVER to constrain generation to
a JSON Schema instead, via vLLM's OpenAI-compatible
`response_format={"type": "json_schema", "json_schema": {...}}` --
verified against vLLM's current documentation
(https://docs.vllm.ai/en/latest/features/structured_outputs/, fetched
2026-09-30) before writing this, specifically because the OLDER
`extra_body={"guided_json": ...}` form was deprecated in vLLM 0.12.0 and
the installed server is 0.18.1. The real install evidence
(`urop-install-pins.txt`) confirms vLLM's guided-decoding backends
(`xgrammar`, `outlines_core`, `lm_format_enforcer`, `llguidance`) were
actually installed in that environment -- evidence the feature is
present, not evidence it has been exercised; nothing in this module has
run against a real server yet.

**This is a methodological ADAPTATION for the open-weight pilot, not a
replication of AutoTraceGT's own generation format.** The paper's own runs
(GPT-5-mini/GPT-5) used the freeform format throughout; switching a
materially weaker open-weight model to schema-constrained decoding is a
deliberate deviation made because that model needs a stronger generation
constraint to produce parseable output at all. State this explicitly in
`docs/presentation_outline.md` and anywhere results from this mode are
reported -- never describe a structured-mode result as directly comparable
to the paper's own codebooks.

**What is preserved, what is not:** the substantive coding instructions
(the core distinction, chunking guidance, per-chunk field definitions,
forbidden-vocabulary constraints) are copied verbatim from upstream's own
`prompts/open_coding_system.md` -- only its `## OUTPUT FORMAT` section is
replaced (`build_structured_system_prompt`). Upstream's own freeform
pipeline (`agents.open_coding.OpenCodingAgent.code_record`, or this
project's own `code_trajectory_preserving_partial_results` wrapper) stays
fully available and completely unmodified; this module is an addition
alongside it, selected via the notebook's `OUTPUT_MODE` flag, never a
silent rewrite of upstream.

**No silent fallback, ever.** If the response is not valid JSON, or does
not match the schema, `parse_structured_coding_response` raises
`StructuredOutputError` immediately -- no retry with a different prompt,
no repair, no placeholder/empty result invented and counted as success.
See `tests/test_evidence_and_reporting.py::test_incomplete_structured_json_is_rejected_without_fabricating`
and `::test_structured_output_error_never_gets_cached_as_done`.

**Classification fix that applies to BOTH modes.**
`autotrace_adapter.classify_chunk_coding` previously only recognized the
freeform mode's `===CODES===\s*\[\s*\]` marker as a legitimate "empty, not
failed" result -- a structured-mode response with `"codes": []` (valid,
schema-conformant, genuinely nothing to code) would have been
misclassified as `"parse_failed"` under the old logic, exactly the
"do not classify solely from whether the codes list is empty" failure
this project already commits to avoiding (§1, §12). Fixed to check `raw`
as JSON directly first (authoritative for structured mode, since
`ChunkCoding.raw` there IS the schema-validated response), falling back to
the freeform text-marker heuristic only when that doesn't apply.

**Update, 2026-09-30 -- now verified against a real server, twice.** The
one-trajectory diagnostic this section originally called "not verified"
has since run on Kaggle. First run: `response_format` WAS honored --
valid, schema-conformant JSON came back -- but with `codes: []`. Second
run, after appending a "DIAGNOSTIC CLARIFICATION" section to the system
prompt (below) and withholding `status` from the user prompt: 3 real
behavioral codes came back, still valid JSON. Guided decoding measurably
works against the installed vLLM 0.18.1/Qwen2.5-3B-Instruct combination.
What remains unverified: whether it holds at more than one chunk/
trajectory, whether generation speed is materially affected, and whether
the coding *quality* (see below) is acceptable at any scale.

**Two further fixes made after reading that first empty-output run's
evidence, both applied 2026-09-30:**

1. **`status` was leaking the true task outcome into the structured user
   prompt.** `code_chunk_structured` now passes
   `status="withheld for this coding diagnostic"` instead of the record's
   real `status` -- the same leakage-prevention principle already applied
   elsewhere in this project (`autotrace_adapter.LEAKAGE_KEYS`,
   `assemble_holdout_prompt`) had not been carried into this newer module.
2. **A response cut off by `max_tokens` must say so, not fail silently as
   a generic parse error.** `code_chunk_structured` now checks
   `response.choices[0].finish_reason == "length"` immediately after the
   response comes back and raises `StructuredOutputError` with that
   specific attribution, before ever attempting
   `parse_structured_coding_response` on what would likely be truncated
   JSON.

**The empty-first-run reporting bug, and its fix, belong to
`reporting.py` (see `RunOutcomeReport.empty_code_outputs`) -- recorded
here because the structured-output mode is what surfaced it**: a
genuinely empty (`codes: []`) result is schema-valid and trivially has
zero invalid quotes, so the original `outputs_passing_automated_evidence_
checks` check ("structurally valid AND `invalid_codes == []`") counted it
as usable. It is not usable -- it contains no evidence example at all.
Fixed to also require `total_codes > 0`. An empty result is also no longer
cached as `resume_cache.mark_done` -- see the notebook's protected-
lifecycle cell -- so a resumed run does not silently treat "the model said
nothing" as a completed, reusable result.

**Not verified:** whether the schema itself needs adjustment at a larger
scale, whether coding quality holds beyond this one hand-inspected
trajectory, and whether the prompt clarification above generalizes to
other trajectories or just happened to work for this one. All of this
stays candidate until a larger, still-bounded Kaggle run.

## 16. Cited-step quote validation, and why an empty/failed result is never cached as done

**Two related fixes made after reading the second (non-empty) structured-
mode Kaggle run's evidence (2026-09-30).**

**Cited-step validation.** `autotrace_adapter.validate_coding_quotes`
previously checked whether a code's `quote` was verbatim ANYWHERE in the
whole chunk's rendered text. That missed a real failure mode: a quote that
IS verbatim in the chunk but comes from a DIFFERENT step than the one the
code's `steps` field cites would pass. Fixed to restrict the verbatim
check to the reconstructed text of ONLY the cited step range (via the
real `format_chunk`, not a reimplementation), and to validate the `steps`
field's own format first (rejecting malformed or out-of-chunk-range
citations with an explicit reason, e.g. `"step reference outside this
chunk"`) -- see `tests/test_open_coding_integration.py::test_quote_verbatim_in_chunk_but_cited_under_the_wrong_step_is_rejected`
and the two format-validation tests next to it.

**A real, observed consequence of the stricter check, recorded rather than
worked around:** all 3 codes from the second real Kaggle run failed this
check -- not because the content was fabricated, but because the model's
quotes use a single space between a step's `[Step N] (role)` header and
its content, where `format_chunk` (the actual renderer, used both to build
the prompt and to validate the quote) always inserts a newline there. A
separate, informal whitespace-normalized comparison (not part of the
committed validator) confirmed the underlying content genuinely matches
within the correct step range. **The strict, space-sensitive result stays
the official one.** Do not relax `verify_quote`/`validate_coding_quotes`
to a whitespace-normalized comparison to make this or any other past
"invalid" result look valid after the fact -- if whitespace tolerance is
ever wanted, it must be a separately named, explicitly-labeled check run
alongside the strict one, never a silent replacement of it.

**Empty and quote-invalid results are never cached as `mark_done`.** Before
this fix, any structurally-valid `TrajectoryCoding` was cached as done
regardless of what quote validation found -- quote validation only fed
reporting, not caching. Now: a result with zero codes, or with at least
one code whose quote/step fails validation, is routed through
`resume_cache.mark_invalid` (bounded retry via `MAX_RESULT_LEVEL_ATTEMPTS`,
per §on resume caching above) instead of `mark_done`. A cache HIT is also
now revalidated against the current rules before being trusted -- an
entry marked done under an earlier, looser validator (e.g. before this fix
existed) is caught and the run stops loudly rather than silently reusing
it. See the notebook's protected-lifecycle cell for the exact revalidation
logic.
