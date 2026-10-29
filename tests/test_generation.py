import html
import json
import re

import pytest
from sqlmodel import Session, select

from prompt_eval import generation
from prompt_eval.engines.base import ChatResponse, EngineError
from prompt_eval.engines.client import EngineClient
from prompt_eval.engines.ratelimit import RateLimiter
from prompt_eval.models import (
    OutputFormat,
    Prompt,
    PromptVersion,
    TestCase,
    TestCaseCategory,
)

N, E, A = TestCaseCategory.NORMAL, TestCaseCategory.EDGE, TestCaseCategory.ADVERSARIAL


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_allocate_sums_exactly():
    assert generation.allocate(10, {N: 60, E: 25, A: 15}) == {N: 6, E: 3, A: 1}
    assert sum(generation.allocate(7, {N: 1, E: 1, A: 1}).values()) == 7
    assert generation.allocate(5, {N: 100, E: 0, A: 0}) == {N: 5}


def test_allocate_rejects_zero_weights():
    with pytest.raises(ValueError):
        generation.allocate(5, {N: 0, E: 0})


def test_split_batches():
    batches = generation.split_batches({N: 12, E: 5, A: 3}, batch_size=10)
    assert [sum(b.values()) for b in batches] == [10, 10]
    assert batches[0] == {N: 10}
    assert batches[1] == {N: 2, E: 5, A: 3}


def test_parse_cases_handles_fences_and_cleans_inputs():
    text = (
        "Here you go:\n```json\n"
        + json.dumps(
            {
                "test_cases": [
                    {
                        "name": "Refund",
                        "category": "edge",
                        "inputs": {"ticket": "money back", "extra": "dropped"},
                        "criteria": "billing",
                        "expected_output": "",
                    },
                    {"name": "", "category": "normal", "inputs": {}},  # no name: skipped
                    {"name": "Odd category", "category": "weird", "inputs": {"ticket": 5}},
                ]
            }
        )
        + "\n```"
    )
    cases = generation.parse_cases(text, ["ticket", "lang"])
    assert [c.name for c in cases] == ["Refund", "Odd category"]
    assert cases[0].inputs == {"ticket": "money back", "lang": ""}
    assert cases[0].category == E
    assert cases[1].category == N
    assert cases[1].inputs["ticket"] == "5"


def test_parse_cases_rejects_non_json():
    with pytest.raises(EngineError):
        generation.parse_cases("sorry, no", ["x"])


def make_prompt() -> tuple[Prompt, PromptVersion]:
    prompt = Prompt(
        project_id=1,
        name="Classify",
        task_description="Classify a ticket",
        output_format=OutputFormat.JSON,
        output_schema='{"type": "object"}',
    )
    version = PromptVersion(version=1, system_prompt="Lang: {{lang}}", template="T: {{ticket}}")
    prompt.versions = [version]
    return prompt, version


def test_build_request_includes_context_and_schema():
    prompt, version = make_prompt()
    request = generation.build_request(
        prompt, version, {N: 2, A: 1}, guidance="Use Persian", existing=["Old case"]
    )
    assert "Write exactly 3 test cases: 2 normal, 1 adversarial." in request.user
    assert "Classify a ticket" in request.user
    assert "Use Persian" in request.user
    assert "- Old case" in request.user
    inputs_schema = request.json_schema["properties"]["test_cases"]["items"]["properties"]["inputs"]
    assert inputs_schema["required"] == ["lang", "ticket"]


class ScriptedGenerator:
    """Answers each request with as many cases as it asks for."""

    def __init__(self):
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        wanted = re.search(r"Write exactly (\d+)", request.user).group(1)
        offset = sum(1 for _ in self.requests) * 100
        cases = [
            {
                "name": f"case {offset + i}",
                "category": "normal",
                "inputs": {"lang": "en", "ticket": f"t{i}"},
                "criteria": "c",
                "expected_output": "",
            }
            for i in range(int(wanted))
        ]
        return ChatResponse(text=json.dumps({"test_cases": cases}), model="fake")

    async def aclose(self):
        pass


@pytest.mark.anyio
async def test_generate_batches_and_passes_earlier_names():
    prompt, version = make_prompt()
    provider = ScriptedGenerator()
    client = EngineClient(provider, RateLimiter())
    cases = await generation.generate(client, prompt, version, 15, {N: 1})
    assert len(cases) == 15
    assert len(provider.requests) == 2
    # The second batch is told what the first one covered.
    assert "- case 100" in provider.requests[1].user


@pytest.mark.anyio
async def test_generate_reports_truncation():
    prompt, version = make_prompt()

    class Truncated(ScriptedGenerator):
        async def complete(self, request):
            return ChatResponse(text='{"test_cases": [', model="fake", stop_reason="max_tokens")

    client = EngineClient(Truncated(), RateLimiter())
    with pytest.raises(EngineError, match="max tokens"):
        await generation.generate(client, prompt, version, 3, {N: 1})


ENGINE_FORM = {"name": "gen", "provider": "anthropic", "model": "claude-opus-5-5"}


def setup_prompt(client) -> int:
    project = client.post("/projects", data={"name": "P"}, follow_redirects=False)
    project_id = int(project.headers["location"].rsplit("/", 1)[1])
    prompt = client.post(
        f"/projects/{project_id}/prompts",
        data={"name": "Classify", "template": "T: {{ticket}}", "system_prompt": "{{lang}}"},
        follow_redirects=False,
    )
    return int(prompt.headers["location"].rsplit("/", 1)[1])


def test_generate_preview_then_save(client):
    client.app.state.engines.provider_factory = lambda engine, settings: ScriptedGenerator()
    client.post("/engines", data=ENGINE_FORM)
    prompt_id = setup_prompt(client)

    page = client.get(f"/prompts/{prompt_id}/test-cases/generate")
    assert "Generate preview" in page.text

    preview = client.post(
        f"/prompts/{prompt_id}/generate",
        data={"engine_id": 1, "count": 3, "mix_normal": 100, "mix_edge": 0, "mix_adversarial": 0},
    )
    assert preview.status_code == 200
    encoded = re.findall(r'name="case" value="([^"]+)"', preview.text)
    assert len(encoded) == 3

    chosen = [html.unescape(value) for value in encoded[:2]]  # untick the third
    response = client.post(
        f"/prompts/{prompt_id}/test-cases/bulk", data={"case": chosen}, follow_redirects=False
    )
    assert response.status_code == 303

    with Session(client.app.state.engine) as session:
        saved = session.exec(select(TestCase)).all()
    assert len(saved) == 2
    assert all(tc.source == "generated" for tc in saved)
    assert saved[0].inputs == {"lang": "en", "ticket": "t0"}


def test_generate_validation_errors_render_inline(client):
    client.post("/engines", data=ENGINE_FORM)
    prompt_id = setup_prompt(client)
    too_many = client.post(f"/prompts/{prompt_id}/generate", data={"engine_id": 1, "count": 500})
    assert "Count must be between" in too_many.text
    no_engine = client.post(f"/prompts/{prompt_id}/generate", data={"engine_id": 99})
    assert "Pick an engine" in no_engine.text
    zero_mix = client.post(
        f"/prompts/{prompt_id}/generate",
        data={"engine_id": 1, "mix_normal": 0, "mix_edge": 0, "mix_adversarial": 0},
    )
    assert "share above 0%" in zero_mix.text


def test_bulk_save_rejects_bad_payload(client):
    prompt_id = setup_prompt(client)
    response = client.post(f"/prompts/{prompt_id}/test-cases/bulk", data={"case": ["{nope"]})
    assert response.status_code == 400
