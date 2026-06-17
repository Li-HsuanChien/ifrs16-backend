import json
import os
import pathlib
from typing import List, Dict, Any, Optional

import requests


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
            timeout=60,
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
) -> Dict[str, Any]:
    """
    Runs a single LLM call for one task against `input_text`.

    Used for both pipeline steps:
      - extraction step: `input_text` is the raw contract text
      - parse step:      `input_text` is the serialized content object produced
                         by the extraction step

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
    └──────────────────────────────────────────┘
    """
    system_prompt = build_system_prompt(task_entry)
    few_shot = build_few_shot_messages(task_entry)
    live_input = build_user_prompt(
        input_text,
        example_count=len(task_entry.get("examples", [])),
    )

    messages = few_shot + [live_input]

    try:
        return client.generate(
            system_prompt=system_prompt,
            messages=messages,
            json_schema=task_entry.get("outputSchema"),
            temperature=0.1,
        )
    except Exception as e:
        print(f"[{task_entry.get('task', 'unknown')}] Extraction error: {e}")
        return None

