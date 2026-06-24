"""
calctools.py — a simple calculation tool (addition) for client-facing requests,
plus a *real* function-calling loop that runs it through the existing LLMClient.

How this differs from the date "tool" in llmapi.py / datetools.py
-----------------------------------------------------------------
The date pipeline uses a **deterministic-injection** pattern: regex computes the
answer (datetools.py) and the result is *replayed* into a single LLM pass as a
synthetic tool turn with tool_choice="none".  The model never decides to call
anything — it is handed the answer.  That is the right pattern when you must
OWN the result (ROC date arithmetic the model gets wrong).

This module shows the other pattern — **real function calling** — without
touching that structure:

    1. We declare the tool and send tool_choice="auto".
    2. The model decides whether to call add_numbers, and with what arguments.
    3. We execute the Python function and append the result as a tool message.
    4. We loop, re-calling the model, until it stops requesting tools and
       produces a final natural-language answer.

The only requirement is an agent LOOP (multiple passes) instead of the date
pipeline's single pass.  LLMClient.generate already passes tools/tool_choice
through, so no client changes are needed.

Both patterns are exposed so you can pick per task:
  * add_numbers(...)            — the deterministic tool itself (call directly).
  * run_calc_agent(client, ...) — let the model drive it via function calling.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

try:  # support both `python utils/calctools.py` and `import utils.calctools`
    from llmapi import LLMClient
except ImportError:
    from utils.llmapi import LLMClient


# ---------------------------------------------------------------------------
# The tool implementation (deterministic — never let the model do arithmetic)
# ---------------------------------------------------------------------------

def add_numbers(numbers: List[float]) -> Dict[str, Any]:
    """
    Add a list of numbers and return a structured result.

    Returns a small report rather than a bare float so the calling model gets
    an unambiguous, auditable result to relay to the client:

        {"operation": "add", "operands": [2, 3, 4], "result": 9}

    Raises ValueError on bad input so the agent loop can surface a clean error
    back to the model as the tool result.
    """
    if not isinstance(numbers, list) or not numbers:
        raise ValueError("`numbers` must be a non-empty list of numbers")

    coerced: List[float] = []
    for n in numbers:
        if isinstance(n, bool) or not isinstance(n, (int, float)):
            raise ValueError(f"not a number: {n!r}")
        coerced.append(n)

    total = sum(coerced)
    # Present whole-number sums as ints (9.0 -> 9) for cleaner client output.
    if isinstance(total, float) and total.is_integer():
        total = int(total)

    return {"operation": "add", "operands": coerced, "result": total}


# ---------------------------------------------------------------------------
# Tool declaration (OpenAI / Azure "tools" schema) + dispatch registry
# ---------------------------------------------------------------------------

ADD_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "add_numbers",
        "description": (
            "Add two or more numbers together and return their exact sum. "
            "Use this for ANY addition the client asks for — do not compute "
            "sums yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "description": "The list of numbers to add together.",
                }
            },
            "required": ["numbers"],
            "additionalProperties": False,
        },
    },
}

# name -> python callable.  Add more tools here to grow the agent's abilities
# (e.g. "subtract_numbers", "multiply_numbers") without touching the loop.
TOOL_REGISTRY: Dict[str, Callable[..., Any]] = {
    "add_numbers": add_numbers,
}

TOOL_DECLARATIONS: List[Dict[str, Any]] = [ADD_TOOL]

CALC_SYSTEM_PROMPT = (
    "You are a calculation assistant for client requests. When the client asks "
    "for a sum or total, call the add_numbers tool with the operands rather than "
    "doing the arithmetic yourself. After the tool returns, reply to the client "
    "in one clear sentence stating the result."
)


# ---------------------------------------------------------------------------
# Real function-calling loop
# ---------------------------------------------------------------------------

def execute_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute one model-requested tool call and build its `tool` reply message.

    Errors are caught and returned as the tool result (an {"error": ...} object)
    so the model can recover or apologise, rather than crashing the loop.

    Public entry point reused by the parse-step tool loop in llmapi.py — any
    name in TOOL_REGISTRY is callable; unknown names come back as an error
    result the model can react to.
    """
    fn = tool_call.get("function", {})
    name = fn.get("name")
    raw_args = fn.get("arguments") or "{}"

    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        impl = TOOL_REGISTRY.get(name)
        if impl is None:
            result: Any = {"error": f"unknown_tool:{name}"}
        else:
            result = impl(**args)
    except Exception as exc:  # bad JSON, bad args, or tool raised
        result = {"error": str(exc)}

    return {
        "role": "tool",
        "tool_call_id": tool_call.get("id"),
        "content": json.dumps(result, ensure_ascii=False),
    }


def run_calc_agent(
    client: LLMClient,
    user_request: str,
    max_turns: int = 5,
) -> str:
    """
    Drive a real tool-calling conversation: the model decides when to call
    add_numbers, we execute it, feed the result back, and loop until the model
    answers in natural language.

    `max_turns` bounds the call/execute loop so a misbehaving model can't spin
    forever.  Returns the model's final natural-language answer to the client.
    """
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": user_request},
    ]

    for _turn in range(max_turns):
        raw = client.generate(
            system_prompt=CALC_SYSTEM_PROMPT,
            messages=messages,
            tools=TOOL_DECLARATIONS,
            tool_choice="auto",   # the model decides — this is real function calling
            temperature=0.0,
        )

        choice = raw["choices"][0]
        message = choice["message"]
        messages.append(message)  # echo the assistant turn back into context

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            # No tool requested -> the model has produced its final answer.
            return message.get("content") or ""

        # Execute every requested tool call and append each result.
        for tool_call in tool_calls:
            messages.append(execute_tool_call(tool_call))

    return "Sorry — I couldn't complete the calculation within the allowed steps."


# ---------------------------------------------------------------------------
# Smoke test:  python -m utils.calctools   (deterministic tool only, no API)
# ---------------------------------------------------------------------------

def _selftest() -> None:
    checks = []

    def check(label, got, want):
        ok = got == want
        checks.append(ok)
        print(f"[{'ok ' if ok else 'FAIL'}] {label}\n        got={got!r}\n        want={want!r}")

    check("2 + 3 + 4", add_numbers([2, 3, 4])["result"], 9)
    check("1.5 + 2.5 -> int 4", add_numbers([1.5, 2.5])["result"], 4)
    check("negatives", add_numbers([-5, 10])["result"], 5)
    check("operands echoed", add_numbers([1, 2])["operands"], [1, 2])

    for bad, why in (([], "empty"), (["x"], "non-number"), ([True, 1], "bool")):
        try:
            add_numbers(bad)
            checks.append(False)
            print(f"[FAIL] {why}: expected ValueError")
        except ValueError:
            checks.append(True)
            print(f"[ok ] {why}: raised ValueError")

    print(f"\n{sum(checks)}/{len(checks)} checks passed")
    if not all(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    _selftest()
