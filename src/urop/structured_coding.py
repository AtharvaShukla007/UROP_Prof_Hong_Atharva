"""Opt-in structured (JSON-schema-constrained) open-coding output mode.

**Why this exists.** A real Kaggle smoke run (2026-09-29, see
docs/experiment_log.md) found that Qwen2.5-3B-Instruct's freeform
`===CODES===`-block output regularly embeds a tool-call's own JSON verbatim
inside a `"quote"` string VALUE without escaping its inner double quotes
(e.g. `"quote": "<tool_call name=X>{"key":"value"}"`), which breaks
`json.loads` on the surrounding CODES array outright -- not a formatting
quirk upstream's own repair-retry can fix, since the text is not "almost
JSON", it is JSON containing unescaped JSON. This mode sidesteps that class
of failure by asking the SERVER to constrain generation to a JSON Schema
(vLLM's guided-decoding backend, see below) instead of asking the model to
freehand quotes inside markdown-ish delimiters.

**This is a methodological ADAPTATION for the open-weight pilot, not a
replication of the original AutoTraceGT generation format.** The paper's
own runs (GPT-5-mini/GPT-5) used the freeform `===CODES===` format;
switching to schema-constrained decoding is a deliberate deviation, made
because a materially weaker open-weight model needs a materially stronger
generation constraint to produce usable output at all. State this
explicitly wherever results from this mode are reported --
`docs/methods_and_deviations.md` records it too.

**API verified against current vLLM docs before writing this** (per this
project's own rule to check version-specific APIs, not guess):
`response_format={"type": "json_schema", "json_schema": {"name": ...,
"schema": {...}}}` is vLLM's documented, CURRENT structured-output
interface (https://docs.vllm.ai/en/latest/features/structured_outputs/,
fetched 2026-09-30). The older `extra_body={"guided_json": ...}` form was
DEPRECATED in vLLM 0.12.0 and must not be used against the installed
0.18.1 server. Backed by the guided-decoding backends actually present in
the real Kaggle install evidence (`xgrammar`, `outlines_core`,
`lm_format_enforcer`, `llguidance` all appear in
`urop-install-pins.txt`) -- their presence is evidence the feature is
installed, not evidence it was exercised; nothing in this module has been
run against a real server. GPU behaviour stays UNVERIFIED until it is.

**No silent fallback.** If the server does not honour `response_format`
(ignores it and free-hands text, or rejects the request), this module never
retries with a different prompt style or repairs the result -- either the
response parses and validates against the schema, or
`StructuredOutputError` is raised. Upstream's own freeform pipeline stays
fully available and unmodified (`agents.open_coding.OpenCodingAgent`,
called directly, or via `autotrace_adapter.code_trajectory_preserving_partial_results`)
-- this module is an addition, never a silent rewrite of it.
"""
from __future__ import annotations

import json
from typing import Any

from urop.autotrace_adapter import PromptTooLongError, ensure_upstream_on_path

OUTPUT_MODE = "structured_json"

CODING_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "codes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "steps": {"type": "string"},
                    "quote": {"type": "string"},
                    "code": {"type": "string"},
                    "memo": {"type": "string"},
                },
                "required": ["steps", "quote", "code", "memo"],
                "additionalProperties": False,
            },
        },
        "skipped": {"type": "string"},
        "segment_memo": {"type": "string"},
    },
    "required": ["codes", "skipped", "segment_memo"],
    "additionalProperties": False,
}

_STRUCTURED_FORMAT_INSTRUCTIONS = """## OUTPUT FORMAT

Return a single JSON object matching the required schema exactly -- no
markdown code fences, no ```json wrapper, no text before or after the
object, and no ===CODES===/===SKIPPED===/===SEGMENT_MEMO===/===END===
section markers (those are for a different output mode; this response is
consumed as pure JSON, not scanned for those markers).

The object has exactly these fields:

  "codes": a JSON array (empty if nothing in this chunk warrants a code).
    Each element has exactly these string fields:
      "steps"  -- which step numbers belong to this code, e.g. "4-7"
      "quote"  -- the most telling fragment, copied verbatim from the log
      "code"   -- a 2-5 word conceptual label, verb + noun format
      "memo"   -- 2-4 sentences of analytic thinking
  "skipped": a string -- step numbers skipped and a brief reason, or "none"
  "segment_memo": a string -- 3-4 sentences on the arc of this segment
"""


def _load_upstream_system_prompt_body() -> str:
    """The upstream `open_coding_system.md` text UP TO its own '## OUTPUT
    FORMAT' section -- i.e. every substantive coding instruction (the core
    distinction, chunking guidance, per-chunk field definitions, forbidden-
    vocabulary constraints), preserved verbatim. Only the format section
    itself is swapped, never the actual coding guidance.
    """
    root = ensure_upstream_on_path()
    text = (root / "prompts" / "open_coding_system.md").read_text(encoding="utf-8")
    marker = "## OUTPUT FORMAT"
    idx = text.find(marker)
    if idx == -1:
        raise RuntimeError(
            f"upstream open_coding_system.md no longer contains {marker!r} -- "
            "this module's assumption about the template's structure is stale, fix it before using structured mode."
        )
    return text[:idx]


def build_structured_system_prompt() -> str:
    return _load_upstream_system_prompt_body() + _STRUCTURED_FORMAT_INSTRUCTIONS + '\n\n## DIAGNOSTIC CLARIFICATION (pilot v2)\nIdentify observable behavioural patterns, including ordinary or successful\nbehaviour. A code does not have to identify a failure. Ground each label\nin the numbered log, using the original step numbers and an exact quote.\nKeep the response concise: a short quote and a brief memo for each code.\nDo not replace the codes with a segment summary.\nIf nothing supports a code, you may return an empty codes list, but explain\nin skipped which steps you excluded and why. Do not invent a code to meet\na quota and do not assume every task contains a failure.\nThe final benchmark outcome is withheld for this diagnostic. Describe the\nobserved process without inferring correctness from an outcome label.\n'


def build_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": "open_coding_result", "schema": CODING_RESULT_SCHEMA},
    }


class StructuredOutputError(RuntimeError):
    """Raised whenever the response cannot be trusted as a valid structured
    coding result -- invalid JSON, or JSON that does not match
    `CODING_RESULT_SCHEMA`. Never repaired, never retried with a different
    prompt, never silently converted into an empty/placeholder result: this
    is the "fail explicitly, no silent fallback" contract for this mode.
    """


def parse_structured_coding_response(content: str) -> tuple[list[dict], str, str]:
    """Parse+validate a structured-mode response into (codes, skipped,
    segment_memo) -- the same return shape as upstream's own
    `agents.open_coding.parse_response`, so both modes plug into the same
    `ChunkCoding` construction. Raises `StructuredOutputError` (never
    returns a placeholder) on invalid JSON or a schema mismatch.
    """
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, ValueError) as exc:
        raise StructuredOutputError(f"response is not valid JSON: {exc}") from exc

    import jsonschema

    try:
        jsonschema.validate(parsed, CODING_RESULT_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise StructuredOutputError(f"response does not match the required schema: {exc.message}") from exc

    return parsed["codes"], parsed["skipped"], parsed["segment_memo"]


def code_chunk_structured(
    agent: Any,
    chunk: list[dict],
    *,
    step_offset: int,
    task: str,
    tools: str,
    status: str,
    preceding_memo: str = "",
) -> Any:
    """Structured-mode equivalent of `OpenCodingAgent.code_chunk`. Sends
    EXACTLY ONE request (no JSON-repair round -- there is nothing to repair
    freeform-parsing-wise once the server is actually constraining output
    to the schema; see the module docstring for the "no silent fallback"
    contract). Goes through `agent.client.chat.completions.create`, so any
    guard installed via `autotrace_adapter.install_budget_guard` (budget,
    token check, evidence logging) still applies transparently -- this
    function builds the request, it does not bypass the client.

    Raises `PromptTooLongError` (from the installed token-guard) or
    `StructuredOutputError` (from this module's own validation) rather than
    ever returning a fabricated/placeholder `ChunkCoding`.
    """
    ensure_upstream_on_path()
    from agents.open_coding import ChunkCoding, format_chunk
    from utils.utils import load_prompt

    chunk_index = (step_offset - 1) // agent.chunk_size
    trajectory_text = format_chunk(chunk, step_offset=step_offset)
    user_prompt = load_prompt(
        "open_coding_user", task=task, tools=tools, status="withheld for this coding diagnostic",
        preceding_memo=preceding_memo or "", trajectory=trajectory_text,
    )
    messages = [
        {"role": "system", "content": build_structured_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]

    kwargs = agent._build_kwargs(messages)  # reuse upstream's own model/temperature/max_tokens/extra_body assembly
    kwargs["response_format"] = build_response_format()
    response = agent.client.chat.completions.create(**kwargs)
    raw = response.choices[0].message.content or ""
    if response.choices[0].finish_reason == "length":
        raise StructuredOutputError("Generation reached the output limit; raw response preserved in evidence.")

    codes, skipped, segment_memo = parse_structured_coding_response(raw)  # raises StructuredOutputError, never fabricates

    return ChunkCoding(
        chunk_index=chunk_index, step_offset=step_offset, num_steps=len(chunk),
        codes=codes, skipped=skipped, segment_memo=segment_memo, raw=raw,
    )


def code_record_structured(agent: Any, record: dict) -> Any:
    """Structured-mode equivalent of
    `autotrace_adapter.code_trajectory_preserving_partial_results`: chunks a
    record and calls `code_chunk_structured` for each, preserving whatever
    chunks completed before a `PromptTooLongError`/`StructuredOutputError`
    on a LATER chunk (attached as `exc.partial_result`, same contract as
    the freeform-mode wrapper) instead of discarding them.
    """
    ensure_upstream_on_path()
    from agents.open_coding import TrajectoryCoding
    from utils.trajectory_processing import chunk_trajectory

    task = record["context"]["task_description"]
    tools = json.dumps([t["function"]["name"] for t in record["context"]["tools"]], indent=2)
    status = record["status"]
    trajectory = record["trajectory"]
    chunks = chunk_trajectory(trajectory, agent.chunk_size)

    result = TrajectoryCoding(instance_id=record["instance_id"], run_id=record["run_id"], status=status)
    preceding_memo = ""
    step_offset = 1
    for chunk in chunks:
        try:
            cc = code_chunk_structured(
                agent, chunk, step_offset=step_offset, task=task, tools=tools, status=status,
                preceding_memo=preceding_memo,
            )
        except (PromptTooLongError, StructuredOutputError) as exc:
            exc.partial_result = result
            raise
        result.chunk_codings.append(cc)
        preceding_memo = cc.segment_memo
        step_offset += len(chunk)
    return result
