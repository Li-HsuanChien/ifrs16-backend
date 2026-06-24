import json
import os
import pathlib
from typing import List, Dict, Any, Optional

import requests

try:  # shared date vocabulary (also used by the deterministic post-processor)
    from datekeywords import build_prompt_hint, is_date_task
except ImportError:
    from utils.datekeywords import build_prompt_hint, is_date_task


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LLMClient:
    def __init__(
        self,
        azure_endpoint: str,       # e.g. "https://my-resource.openai.azure.com"
        deployment_name: str,      # e.g. "gpt-4o"
        api_key: str,
        api_version: str = "2024-10-21",
    ):
        # Documented URL pattern:
        # POST https://{endpoint}/openai/deployments/{deployment-id}/chat/completions?api-version=...
        # Ref: https://learn.microsoft.com/en-us/azure/foundry/openai/reference
        self.url = (
            f"{azure_endpoint.rstrip('/')}"
            f"/openai/deployments/{deployment_name}"
            f"/chat/completions"
            f"?api-version={api_version}"
        )
        # Documented required request header: api-key (string)
        self.headers = {
            "api-key": api_key,
            "Content-Type": "application/json",
        }

    def generate(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.1,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Low-level call. `messages` is a fully assembled list of
        {"role": "user"|"assistant", "content": "..."} dicts,
        allowing callers to inject few-shot turns before the real input.

        Azure OpenAI REST differences vs Anthropic:
          - No top-level `system` field; system prompt goes as the first
            message with role "system" inside the messages array.
          - No `model` field in the body; deployment name is in the URL path.
          - `response_format` accepts:
              {"type": "json_object"}                          — valid JSON output
              {"type": "json_schema", "json_schema": {...}}    — structured outputs
            Ref: https://learn.microsoft.com/en-us/azure/foundry/openai/reference
        """
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        payload: Dict[str, Any] = {
            "messages": full_messages,
            "temperature": temperature,
        }

        # Tool plumbing — present so a replayed `tool` message (the deterministic
        # date-tool result) is a well-formed turn.  tool_choice="none" forces the
        # model to answer using the supplied result rather than call again.
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        if json_schema:
            # Use structured outputs (json_schema) when a schema is supplied.
            # Falls back to json_object mode if the schema is not compatible.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction_output",
                    "schema": json_schema,
                    "strict": True,
                },
            }

        response = requests.post(
            self.url,
            headers=self.headers,
            data=json.dumps(payload),
            timeout=120,
        )

        if response.status_code == 200:
            return response.json()

        raise Exception(
            f"API request failed {response.status_code}: {response.text}"
        )


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------

def build_system_prompt(task_entry: Dict[str, Any]) -> str:
    """
    Combines system_prompts + rules into a single system string.
    Rules are formatted as a numbered reference list so the model can
    cite them when reasoning.
    """
    parts = []

    if task_entry.get("system_prompts"):
        parts.append(task_entry["system_prompts"].strip())

    rules = task_entry.get("rules")
    if rules:
        if isinstance(rules, list):
            rules_block = "\n".join(f"{i+1}. {r}" for i, r in enumerate(rules))
        else:
            rules_block = rules.strip()

        if rules_block:
            parts.append(f"\n## Reference Rules\n{rules_block}")

    # Inject the shared date-keyword reference for date-bearing tasks so the
    # model captures the same signal phrases the post-processor relies on.
    if is_date_task(task_entry):
        parts.append(build_prompt_hint())

    return "\n\n".join(parts)


def build_few_shot_messages(task_entry: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Converts the `examples` list into alternating user/assistant message pairs.

    Each example is expected to carry a `source` field holding the *input* the
    model would receive for that example:
      - extraction step: `source` is the raw contract text (a string)
      - parse step:      `source` is the structured content object emitted by
                         the extraction step

    The pair becomes:
      user:      "<source>"                  ← the demonstrated input
      assistant: <example minus `source`>    ← the expected structured output

    If an example has no `source`, a placeholder is used for the input so the
    turn still demonstrates the output format.
    """
    messages = []
    for i, example in enumerate(task_entry.get("examples", []), start=1):
        demo = dict(example)
        source = demo.pop("source", None)

        if source is None:
            user_content = (
                f"[Example {i}] Extract the structured information from the "
                f"following input:\n\n<input_placeholder_{i}>"
            )
        else:
            source_text = (
                source if isinstance(source, str)
                else json.dumps(source, ensure_ascii=False, indent=2)
            )
            user_content = (
                f"[Example {i}] Extract the structured information from the "
                f"following input:\n\n{source_text}"
            )

        messages.append({"role": "user", "content": user_content})
        messages.append({
            "role": "assistant",
            "content": json.dumps(demo, ensure_ascii=False, indent=2),
        })
    return messages


# Declaration for the deterministic date tool whose result is replayed into the
# parse step.  The model never actually invokes it — we run it ourselves and
# inject the result — but the declaration makes the replayed turn well-formed.
GROUND_TRUTH_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "compute_ground_truth_dates",
        "description": (
            "Deterministically converts ROC/民國 dates to ISO 8601 and derives "
            "第N年至第M年 period dates from the rent-commencement anchor (起租日 / "
            "租金給付始期). Returns authoritative dates that OVERRIDE any model "
            "arithmetic."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

_GROUND_TRUTH_CALL_ID = "call_ground_truth_dates"


def build_tool_turn(tool_result: Any) -> List[Dict[str, Any]]:
    """
    Build the synthetic agent tool-call turn that carries the ground-truth date
    result into the model's context:

        assistant : (calls compute_ground_truth_dates)
        tool      : <authoritative ISO dates>

    Placed right after the live user input, this mirrors an agent that read the
    input, called the date tool, and is now about to answer using its result.
    """
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": _GROUND_TRUTH_CALL_ID,
                    "type": "function",
                    "function": {
                        "name": "compute_ground_truth_dates",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": _GROUND_TRUTH_CALL_ID,
            "content": json.dumps(tool_result, ensure_ascii=False),
        },
    ]


# ---------------------------------------------------------------------------
# Callable tools (real function calling) — distinct from the pre-injected,
# deterministic date tool above.  These the model actually invokes during the
# parse step; we execute them and feed the result back (see _run_tool_loop).
# ---------------------------------------------------------------------------

def _load_callable_tools():
    """
    Lazily import the calc tool registry so llmapi <-> calctools stay decoupled
    (calctools imports LLMClient from here; importing it lazily avoids a cycle).

    Returns (registry, declarations_by_name):
      registry            : name -> python callable
      declarations_by_name: name -> OpenAI tool declaration dict
    """
    try:
        from calctools import TOOL_REGISTRY, TOOL_DECLARATIONS
    except ImportError:
        from utils.calctools import TOOL_REGISTRY, TOOL_DECLARATIONS
    decls = {d["function"]["name"]: d for d in TOOL_DECLARATIONS}
    return TOOL_REGISTRY, decls


def _execute_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Run one model-requested callable tool and build its `tool` reply message."""
    try:
        from calctools import execute_tool_call
    except ImportError:
        from utils.calctools import execute_tool_call
    return execute_tool_call(tool_call)


def build_user_prompt(input_text: str, example_count: int = 0) -> Dict[str, str]:
    """
    Wraps the real input in a user message that mirrors the few-shot format.

    `input_text` is the raw contract text for the extraction step, or the
    serialized content object (from the extraction step) for the parse step.
    """
    prefix = f"[Input {example_count + 1}] " if example_count else ""
    return {
        "role": "user",
        "content": (
            f"{prefix}Extract the structured information from the "
            f"following input:\n\n{input_text}"
        ),
    }


# ---------------------------------------------------------------------------
# High-level extraction function
# ---------------------------------------------------------------------------

def extraction(
    client: LLMClient,
    task_entry: Dict[str, Any],
    input_text: str,
    tool_result: Any = None,
) -> Dict[str, Any]:
    """
    Runs a single LLM call for one task against `input_text`.

    Used for both pipeline steps:
      - extraction step: `input_text` is the raw contract text
      - parse step:      `input_text` is the serialized content object produced
                         by the extraction step

    When `tool_result` is supplied (parse step, date-bearing topics), a synthetic
    tool-call turn carrying the deterministic ground-truth dates is replayed after
    the live input — so the model answers using authoritative dates instead of
    doing ROC arithmetic itself.

    Message layout sent to the API:
    ┌──────────────────────────────────────────┐
    │ system   : system_prompts + rules        │  ← ground truth / instructions
    ├──────────────────────────────────────────┤
    │ user     : [Example 1] <source input>   │  ┐
    │ assistant: { ...example output... }     │  │ few-shot demonstration
    │ user     : [Example N] <source input>   │  │ (repeated per example)
    │ assistant: { ...example output... }     │  ┘
    ├──────────────────────────────────────────┤
    │ user     : [Input N+1] <real input>     │  ← live input
    │ assistant: (calls compute_ground_truth) │  ┐ tool turn (date tasks only)
    │ tool     : <authoritative ISO dates>    │  ┘
    └──────────────────────────────────────────┘
    """
    system_prompt = build_system_prompt(task_entry)
    few_shot = build_few_shot_messages(task_entry)
    live_input = build_user_prompt(
        input_text,
        example_count=len(task_entry.get("examples", [])),
    )

    messages = few_shot + [live_input]

    # Pre-injected deterministic date tool (model never calls it; result replayed).
    tools: List[Dict[str, Any]] = []
    if tool_result is not None:
        messages += build_tool_turn(tool_result)
        tools.append(GROUND_TRUTH_TOOL)

    # Callable tools the model may actually invoke (e.g. add_numbers).  Declared
    # per-task via "callTools": ["add_numbers", ...] in parsecalls.json.
    callable_names = task_entry.get("callTools") or []
    registry: Dict[str, Any] = {}
    if callable_names:
        registry, decls = _load_callable_tools()
        for name in callable_names:
            if name in decls:
                tools.append(decls[name])

    try:
        if callable_names:
            # Real function calling: loop call→execute→feed-back, then lock the
            # final answer to the output schema.
            return _run_tool_loop(
                client, system_prompt, messages, task_entry, tools, registry
            )

        # Single pass (current behaviour for date / no-tool tasks).
        tool_choice = "none" if tools else None
        return client.generate(
            system_prompt=system_prompt,
            messages=messages,
            json_schema=task_entry.get("outputSchema"),
            temperature=0.1,
            tools=tools or None,
            tool_choice=tool_choice,
        )
    except Exception as e:
        print(f"[{task_entry.get('task', 'unknown')}] Extraction error: {e}")
        return None


def _run_tool_loop(
    client: LLMClient,
    system_prompt: str,
    messages: List[Dict[str, Any]],
    task_entry: Dict[str, Any],
    tools: List[Dict[str, Any]],
    registry: Dict[str, Any],
    max_tool_turns: int = 5,
) -> Dict[str, Any]:
    """
    Drive a real function-calling conversation for one parse step.

    Why a loop (vs the single-pass date pattern): the model must *choose* which
    figures to add — distinguishing fee line items from dates / article numbers,
    a semantic judgement that is its strength — while the tool does the
    arithmetic the model gets wrong (e.g. 2,000+11,700+468+500 = 14,668, not the
    36,000 it once hallucinated).

    Flow:
      1. Loop with tools enabled (tool_choice="auto") and NO schema lock, so the
         model is free to emit tool_calls.  Each call is executed and its result
         fed back.  We stop once the model answers without calling a tool, or
         after max_tool_turns.
      2. One final schema-locked turn (tool_choice="none") turns the gathered
         results into the structured JSON the pipeline expects.

    Keeping the schema off during the loop avoids the strict-structured-output /
    tool-call conflict some Azure API versions exhibit.
    """
    work = list(messages)

    for _turn in range(max_tool_turns):
        raw = client.generate(
            system_prompt=system_prompt,
            messages=work,
            json_schema=None,             # don't lock to schema while tools may fire
            temperature=0.1,
            tools=tools,
            tool_choice="auto",
        )
        message = raw["choices"][0]["message"]
        work.append(message)

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            break  # model answered without a tool — go lock the schema

        for tool_call in tool_calls:
            name = tool_call.get("function", {}).get("name")
            if name in registry:
                work.append(_execute_tool_call(tool_call))
            else:
                # e.g. the model tries to re-call the pre-injected date tool;
                # remind it the result is already in context.
                work.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "content": json.dumps(
                        {"note": f"{name} result already provided above"},
                        ensure_ascii=False,
                    ),
                })

    # Final schema-locked synthesis turn — emit the structured payload using the
    # tool results now in context.
    return client.generate(
        system_prompt=system_prompt,
        messages=work,
        json_schema=task_entry.get("outputSchema"),
        temperature=0.1,
        tools=tools,
        tool_choice="none",
    )

