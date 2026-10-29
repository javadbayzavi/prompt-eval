"""Generate test cases for a prompt with an LLM."""

import json
import re
from dataclasses import dataclass

from prompt_eval.engines import ChatRequest, EngineClient, EngineError
from prompt_eval.models import Prompt, PromptVersion, TestCaseCategory
from prompt_eval.templating import find_variables

BATCH_SIZE = 10
MAX_TEST_CASES = 100

DEFAULT_MIX = {
    TestCaseCategory.NORMAL: 60,
    TestCaseCategory.EDGE: 25,
    TestCaseCategory.ADVERSARIAL: 15,
}

SYSTEM_PROMPT = """\
You design evaluation test cases for LLM prompts. Each test case is one set of input values
for the prompt's template variables, plus the criteria a grader will use to judge the model's
answer.

Write test cases that would catch real failures:
- normal: realistic, typical inputs the prompt must handle well.
- edge: unusual but legitimate inputs: empty or very long values, ambiguity, mixed languages,
  boundary values, missing details.
- adversarial: inputs that try to break the prompt: instructions hidden in the input, requests
  to ignore the format, misleading or contradictory content.

Make the criteria specific and checkable for that input, e.g. "Classifies as billing; does not
promise a refund", not "Gives a good answer". Only give an expected output when one exact
correct answer exists; otherwise leave it empty. Vary the cases: do not repeat a scenario that
is already covered."""


@dataclass
class GeneratedCase:
    name: str
    category: TestCaseCategory
    inputs: dict[str, str]
    criteria: str
    expected_output: str = ""


def allocate(total: int, weights: dict[TestCaseCategory, float]) -> dict[TestCaseCategory, int]:
    """Split `total` across categories by weight (largest remainder, so it sums exactly)."""
    weight_sum = sum(max(0.0, w) for w in weights.values())
    if total <= 0 or weight_sum <= 0:
        raise ValueError("Need a positive count and at least one positive weight")
    shares = {cat: total * max(0.0, w) / weight_sum for cat, w in weights.items()}
    counts = {cat: int(share) for cat, share in shares.items()}
    leftover = total - sum(counts.values())
    by_remainder = sorted(shares, key=lambda cat: shares[cat] - counts[cat], reverse=True)
    for cat in by_remainder[:leftover]:
        counts[cat] += 1
    return {cat: n for cat, n in counts.items() if n > 0}


def split_batches(
    counts: dict[TestCaseCategory, int], batch_size: int = BATCH_SIZE
) -> list[dict[TestCaseCategory, int]]:
    """Cut the per-category counts into batches of at most `batch_size` cases."""
    queue = [cat for cat, n in counts.items() for _ in range(n)]
    batches = []
    for start in range(0, len(queue), batch_size):
        batch: dict[TestCaseCategory, int] = {}
        for cat in queue[start : start + batch_size]:
            batch[cat] = batch.get(cat, 0) + 1
        batches.append(batch)
    return batches


def output_schema(variables: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "test_cases": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "category": {"type": "string", "enum": [c.value for c in TestCaseCategory]},
                        "inputs": {
                            "type": "object",
                            "properties": {var: {"type": "string"} for var in variables},
                            "required": variables,
                            "additionalProperties": False,
                        },
                        "criteria": {"type": "string"},
                        "expected_output": {"type": "string"},
                    },
                    "required": ["name", "category", "inputs", "criteria", "expected_output"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["test_cases"],
        "additionalProperties": False,
    }


def build_request(
    prompt: Prompt,
    version: PromptVersion,
    counts: dict[TestCaseCategory, int],
    *,
    guidance: str = "",
    existing: list[str] = (),
) -> ChatRequest:
    variables = find_variables(version.system_prompt, version.template)
    wanted = ", ".join(f"{n} {cat.value}" for cat, n in counts.items())
    parts = [
        f"<task_description>\n{prompt.task_description or '(none given)'}\n</task_description>",
        f"<expected_output_format>{prompt.output_format.value}</expected_output_format>",
    ]
    if prompt.output_schema:
        parts.append(f"<output_schema>\n{prompt.output_schema}\n</output_schema>")
    if version.system_prompt:
        parts.append(f"<system_prompt>\n{version.system_prompt}\n</system_prompt>")
    parts.append(f"<template>\n{version.template}\n</template>")
    if variables:
        parts.append(
            "Template variables to fill in each test case: "
            + ", ".join(f"`{v}`" for v in variables)
        )
    else:
        parts.append("The template has no variables; vary only the criteria and scenario names.")
    if existing:
        parts.append(
            "Already covered (do not repeat these scenarios):\n"
            + "\n".join(f"- {name}" for name in existing)
        )
    if guidance.strip():
        parts.append(f"<extra_guidance>\n{guidance.strip()}\n</extra_guidance>")
    parts.append(f"Write exactly {sum(counts.values())} test cases: {wanted}.")

    return ChatRequest(
        system=SYSTEM_PROMPT,
        user="\n\n".join(parts),
        json_schema=output_schema(variables),
    )


def extract_json(text: str) -> object:
    """Parse JSON, tolerating code fences or prose around it (some local models add them)."""
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except ValueError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except ValueError:
            pass
    raise EngineError("The generator did not return valid JSON")


def parse_cases(text: str, variables: list[str]) -> list[GeneratedCase]:
    data = extract_json(text)
    items = data.get("test_cases") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise EngineError("The generator's JSON has no test_cases list")

    cases = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            category = TestCaseCategory(item.get("category", "normal"))
        except ValueError:
            category = TestCaseCategory.NORMAL
        raw_inputs = item.get("inputs") if isinstance(item.get("inputs"), dict) else {}
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        cases.append(
            GeneratedCase(
                name=name,
                category=category,
                # Keep only the template's variables, as strings.
                inputs={var: str(raw_inputs.get(var, "")) for var in variables},
                criteria=str(item.get("criteria", "")).strip(),
                expected_output=str(item.get("expected_output", "")).strip(),
            )
        )
    return cases


async def generate(
    client: EngineClient,
    prompt: Prompt,
    version: PromptVersion,
    total: int,
    mix: dict[TestCaseCategory, float],
    *,
    guidance: str = "",
) -> list[GeneratedCase]:
    """Generate `total` cases in batches; each batch sees what earlier ones covered."""
    variables = find_variables(version.system_prompt, version.template)
    existing = [tc.name for tc in prompt.test_cases]
    cases: list[GeneratedCase] = []
    for batch in split_batches(allocate(total, mix)):
        request = build_request(
            prompt, version, batch, guidance=guidance, existing=existing + [c.name for c in cases]
        )
        response = await client.complete(request)
        if response.refused:
            raise EngineError("The generator model refused the request")
        if response.stop_reason in ("max_tokens", "length"):
            raise EngineError("The generator hit its max tokens; raise the engine's max tokens")
        cases.extend(parse_cases(response.text, variables)[: sum(batch.values())])
    return cases
