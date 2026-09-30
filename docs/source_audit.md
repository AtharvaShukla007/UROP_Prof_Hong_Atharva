# Source audit

What was actually checked in this session against the real files, with what
was found. Anything not listed here as checked has not been verified --
don't treat silence as confirmation. Re-run `scripts/inspect_sources.py` any
time to regenerate the hashes/presence part of this table.

## Reference files (see `reference_materials/`, git-ignored)

| File | Present | SHA-256 |
|---|---|---|
| `urop_context_and_instructions_for_executing_ai.md` | yes | `82a3987a71f2c83111869368d5c269d7e41fe17d9e1865c114fa91e4530ad777` |
| `2608.30391v1.pdf` (AutoTraceGT paper) | yes | `b372259555e8a3ca9587277e7b7364a6db07ae1bdd9a91cb98126bd9c6e297bc` |
| `Qual-Agent-Behavior-Analysis-main.zip` (author code) | yes | `e6843e9d7978830b71215f1e4d4bc0b941c50526fe4a99fd26477ad189cfbdb3` |
| `hidden_error_awareness_deep_dive (1).pdf` (prior deck) | yes | `efe4fed040fcc30fc38075a5519d99b2df3f80042939f922d5d834ef697bed43` |
| `CLAUDE_CODE_UROP_HANDOFF.md` | yes | `9884118c6c05e8c5c8a53361a5c88ad29af799d43d4926258ba9af9e30d06444` |

## AutoTraceGT paper (arXiv 2608.30391v1) -- read in full this session

Extracted with `pypdf` via a throwaway `uv run --with pypdf` environment
(not added to the project's own dependencies). 33 pages, text extraction
mostly clean (a few embedded fonts triggered harmless `fontTools` warnings).

**Authors and affiliations -- confirmed, matches the handoff:**
Zhuoran Lu, Zhuoyan Li (Purdue), Yangyang Yu (Stevens Institute of
Technology), Yibo Meng, Chengxi Zang (Cornell), Nan Jiang (UT El Paso),
Jie Gao, Ziang Xiao (Johns Hopkins).

**T3 finding (Table 2, GLM support) -- confirmed exact match to the handoff:**
"Tau-Bench × Finds payment error late; escalates. +1.64" (failure-associated,
β = +1.64). T3 itself: "✓ Executes after consent and reconciliation. −0.99"
(success-associated). Prose (p.9-10): successful Tau-Bench trajectories
"resolve the relevant task state before taking irreversible actions,"
aligning "the user's identity, retrieved records, constraints, and payment
status into a coherent action plan before execution"; failed trajectories
"often postpone this."

**Table 3 (failure-prediction baselines, context only) -- roughly confirmed:**
Tau-Bench few-shot ROC AUC ranges 0.516-0.550 across the four backbone models
tested (GPT-4.1-mini, GPT-5, GPT-5-mini, GPT-OSS-120B) -- matches the
original brief's "roughly 0.52-0.55." Best codebook-feature-engineering AUC
on Tau-Bench: 0.699 (GPT-5-mini, "Complementary" variant) -- matches "up to
about 0.70." This is *their* text-only failure-prediction result on a
different task (predicting final outcome from a trajectory prefix), not
comparable to our own planned hidden-state probe.

**Saturation rule -- a real, verified discrepancy vs. the shipped code.**
This confirms and sharpens the handoff's claim; it is a subtler mismatch
than "count vs. fraction":

- Algorithm 1 (p.5) and its formal definition (p.5, §3, "Codebook
  management") both define the per-round quantity as a raw **count**:
  `a_t = #add actions logged this round` /
  `a_t = |{ℓ ∈ L_t \ L_{t-1} : ℓ.action = add}|` -- literally the size of a
  set, not divided by anything.
- The same paper's reproducibility section (Appendix A.6, p.20) calls the
  threshold an **"add-rate saturation threshold of ε = 0.2"**, and later
  describes "the mean add rate chang[ing] from 0.125 to 0.129" -- a value
  that only makes sense as a *fraction* of actions in a round, not a raw
  count. So the paper's own prose and its own formal pseudocode disagree
  about whether `a_t` is a count or a rate.
- The shipped code (`external/autotracegt_upstream/autotracegt/pipeline.py`,
  confirmed by reading it directly, not from memory: `add_rate =
  actions.count("ADD") / total`) implements the **rate** interpretation --
  consistent with the paper's prose, not its pseudocode.
- The shipped code's *default threshold* is **0.1**
  (`_ADD_RATE_MAX = 0.1`, confirmed live via `scripts/inspect_sources.py`),
  while the paper's actually-reported main-text experiments used **ε = 0.2**.

**Conclusion for our own runs:** use the repo's `add_rate_streak` criterion
(a fraction), and treat the threshold (default 0.1 in the repo, 0.2 in the
paper's reported runs) as a tunable we set and report explicitly -- see
`configs/*.yaml` and `docs/methods_and_deviations.md`. Never claim exact
replication of the paper's saturation behavior; the two artifacts don't even
agree with each other on what the criterion means.

**Other Table-3-adjacent numbers, license, and GPT-OSS-120B:** the paper
states all six benchmark trajectory sets are used under their original
licenses and that GPT-OSS-120B is used "under its released model license"
(p.12, Limitations/License) -- consistent with the handoff's instruction not
to download that 120B model as a local baseline; it was only ever a backbone
in the authors' own cloud-side experiments.

## Author code ZIP (`external/autotracegt_upstream/`, extracted, git-ignored)

Read directly, not from memory or the handoff's summary:

- `autotracegt/pipeline.py` -- confirms `DATASETS` registry, `_ADD_RATE_MAX
  = 0.1` default, `--saturation-criterion {add_rate_streak, gt_theoretical}`,
  `--add-rate-threshold` CLI flag.
- `autotracegt/llm/endpoints.py` -- confirms `register_endpoint(name, *,
  api_key_env, base_url=None, model_header=None, extra_headers=None)`, and
  that `build_async_client`/`build_sync_client` raise `RuntimeError` if
  `api_key_env` is unset in the environment -- so a local server still needs
  *some* env var set (see `src/urop/autotrace_adapter.register_local_endpoint`,
  which sets a clearly-named dummy var and refuses to do so for a non-
  localhost `base_url`).
- `autotracegt/utils/trajectory_processing.py` -- confirms `_annotate_tau`:
  `"resolved" if record.get("resolved") else "failed"` -- a **missing**
  `resolved` key silently becomes `"failed"` upstream. `src/urop/data.py`
  deliberately rejects a missing/non-bool `resolved` instead (see
  `docs/methods_and_deviations.md`).
- `autotracegt/data/README.md` -- confirms the raw record schema
  (`instance_id`, `run_id`, `messages`, `tools`, outcome field) and that
  `clean_record()` assumes `messages[0]` = system, `messages[1]` = user,
  `messages[2:]` = trajectory -- matches what `src/urop/data.py` validates
  against (with hard rejection instead of silent slicing).
- No `LICENSE` file anywhere in the ZIP (confirmed by directory listing) --
  code stays under git-ignored `external/`, not redistributed. See
  `external/PROVENANCE.md`.
- `requirements.txt` uses version ranges (`openai>=1.60,<3`, etc.), not
  exact pins -- copied verbatim into `requirements/kaggle-candidate.txt`,
  explicitly marked unverified there.

## tau-bench (`sierra-research/tau-bench`) -- cloned and read this session

Cloned via `git clone` (not just API browsing), pinned to commit
`59a200c6d575d595120f1cb70fea53cef0632f6b` (`main` at clone time -- also
has an `update-readme-tau3-bench` branch not used here). Kept under
git-ignored `external/tau-bench/` (code only -- see `external/PROVENANCE.md`
for what was trimmed and why).

- **License:** MIT, confirmed by reading `external/tau-bench/LICENSE`
  directly (Copyright (c) 2024 Sierra).
- **README's own warning, verbatim:** "The tasks in this repo are not
  updated. This repository contains outdated versions of the airline and
  retail tasks." Points to `tau2-bench` (now "τ³-bench") as the maintained
  successor. **Deliberately pinned the older repo anyway** -- the real
  historical trajectories we downloaded (below) were generated against
  *this* commit's task/policy text; moving to τ³-bench would silently
  mismatch the trajectories' ground truth. See
  `docs/methods_and_deviations.md`.
- **Retail task counts** (counted directly from
  `tau_bench/envs/retail/tasks_{test,train,dev}.py`, not guessed): **test =
  115, train = 500, dev = 20** (635 total). The original brief's "retail has
  on the order of ~100+ tasks" undersold it -- that's true only of the test
  split.
- **Retail tools** (`tau_bench/envs/retail/tools/__init__.py`, `ALL_TOOLS`):
  **16 total**, of which **7 are state-changing** (write) tools:
  `cancel_pending_order`, `exchange_delivered_order_items`,
  `modify_pending_order_address`, `modify_pending_order_items`,
  `modify_pending_order_payment`, `modify_user_address`,
  `return_delivered_order_items`. `terminate_tools = ["transfer_to_human_agents"]`
  (`tau_bench/envs/retail/env.py`). The other 8 tools (`calculate`,
  `find_user_id_by_email`, `find_user_id_by_name_zip`, `get_order_details`,
  `get_product_details`, `get_user_details`, `list_all_product_types`,
  `think`) are read-only/reasoning.
- **Retail policy text** (`tau_bench/envs/retail/wiki.md`, read in full):
  authenticate by email or name+zip *before* anything else, even if the
  user already gave a user id; one user per conversation; **"Before taking
  consequential actions that update the database (cancel, modify, return,
  exchange), you have to list the action detail and obtain explicit user
  confirmation (yes) to proceed."** This is the exact wording any rule-based
  T3 proxy label must encode -- not a paraphrase.
- **Reward semantics** (`tau_bench/envs/base.py`, `Env.calculate_reward`):
  strictly binary. Reward starts at `1.0`; replays the task's ground-truth
  actions against a fresh copy of the database and compares a hash of the
  resulting state (`r_actions`) -- any mismatch sets reward to `0.0`; if the
  task also specifies required `outputs` (substrings the agent's final
  response must contain), any missing one also sets reward to `0.0`. There
  is no partial credit.
- **Historical trajectories are real and included in the repo**
  (`historical_trajectories/`, referenced in the README's "Historical
  trajectories" section) -- not something we had to generate ourselves for
  a smoke sample. Four files: `{gpt-4o,sonnet-35-new}-{airline,retail}.json`.
  We downloaded only `gpt-4o-retail.json` (10,813,408 bytes) -- see
  `data/tau_bench_raw/PROVENANCE.md` for hashes and the exact `curl` command.

**Raw schema of the downloaded historical trajectories, verified by direct
inspection (not assumed to match AutoTraceGT's expected shape):**

```json
{
  "task_id": 0,            // int -- matches tau_bench.envs.retail.tasks_test.TASKS_TEST index
  "reward": 1.0,           // float, exactly 0.0 or 1.0 -- see reward semantics above
  "trial": 0,               // int, 0-3 observed -- matches the README's Pass^1..Pass^4
  "info": { "task": {...}, "source": "user", "user_cost": 0.0024475, "reward_info": {...} },
  "traj": [                 // OpenAI chat format -- traj[0]=system, traj[1]=user
    {"role": "system", "content": "<full retail policy text>"},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": null, "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}
  ]
}
```

This is **not** AutoTraceGT's expected raw schema (`instance_id`, `run_id`,
`messages`, `tools`, `resolved`) -- no top-level `tools` field at all (tool
schemas must be reconstructed separately from the tool source files), and
the outcome/id fields are named and shaped differently. Confirmed by
downloading and parsing the real file, not inferred from the README.
`src/urop/tau_bench_adapter.py` does this conversion; see
`docs/methods_and_deviations.md`.

**Cross-check that `task_id: 0` really is `TASKS_TEST[0]`:** the record's
`info.task.instruction` field is byte-identical to
`TASKS_TEST[0].instruction` in `tau_bench/envs/retail/tasks_test.py` at the
same commit (both begin "You are Yusuf Rossi in 19122...exchange the
mechanical keyboard..."). Confirms the historical trajectories were
generated against the **test** split, not train/dev.

**Windows download-integrity note:** this machine has `core.autocrlf=true`,
so the git-checked-out copy of `gpt-4o-retail.json` (had we kept it) differs
byte-for-byte from the published blob (CRLF vs LF). We instead downloaded it
directly via `curl` against `raw.githubusercontent.com` and verified the
byte count (10,813,408, exact match to the GitHub API's reported blob size)
and, separately, that `git hash-object` on the (since-deleted) checked-out
copy recovered the exact published blob sha
(`27f588d9157513c3ad11de3db9e824021152d580`). Full detail in
`external/PROVENANCE.md` and `data/tau_bench_raw/PROVENANCE.md`.

## Not checked, still (do not assume)

- The live `github.com/ZhuoranLu/Qual-Agent-Behavior-Analysis` repo has
  still not been fetched to diff against the supplied ZIP -- GitHub access
  is confirmed working from this environment (used extensively to clone/read
  `sierra-research/tau-bench` in a later session), so this is now just an
  outstanding task, not a blocked one.
- `sonnet-35-new-retail.json` (25.3 MB, the other real historical-trajectory
  file tau-bench ships) was deliberately **not** downloaded -- one real
  source (`gpt-4o-retail.json`) was enough for this sprint's smoke/pilot
  scope. Download it if a second model's trajectories are ever needed.
- The four "unverified" literature-survey papers (arXiv 2608.27750,
  2605.07990, 2605.14038, 2604.19775) -- not checked this session.
- `hidden_error_awareness_deep_dive (1).pdf` was not re-verified against its
  primary paper (arXiv 2605.09502) this session -- treat every number in it
  as unverified until that check is done; do not put its numbers in slides
  before then.
- Kaggle's actual internet setting, GPU quota accounting, and installed
  torch/CUDA build -- cannot be checked from this machine at all.
