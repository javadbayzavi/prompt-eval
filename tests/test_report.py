import json

import pytest
from sqlmodel import Session, select

from prompt_eval import suggestions
from prompt_eval.engines import EngineError
from prompt_eval.engines.base import ChatResponse
from prompt_eval.models import PromptVersion, Run
from prompt_eval.report import build_report
from tests.test_judge import JudgeProvider, save_judge
from tests.test_runs import BrokenProvider, EchoProvider, add_engine, start, wait_for
from tests.test_scoring import make_prompt

SUGGESTION = {
    "diagnosis": "The prompt never asks for JSON.",
    "changes": [
        {
            "change": "Ask for a JSON object with a greeting key",
            "reason": "Outputs were plain text, so parsing failed",
            "addresses": "Ana",
        }
    ],
    "revised_system_prompt": "",
    "revised_template": "JSON please {{name}}",
}


class TargetProvider(EchoProvider):
    """Plain text for the original prompt, valid JSON once the prompt asks for it."""

    async def complete(self, request):
        text = '{"greeting": "hi"}' if "JSON please" in request.user else "hello"
        return ChatResponse(text=text, model="target", latency_ms=12)


class AdvisorProvider(EchoProvider):
    def __init__(self):
        self.requests = []
        self.answer = SUGGESTION

    async def complete(self, request):
        self.requests.append(request)
        return ChatResponse(text=json.dumps(self.answer), model="advisor")


class Providers:
    def __init__(self):
        self.advisor = AdvisorProvider()

    def __call__(self, engine, settings):
        return {
            "judge": JudgeProvider(),
            "advisor": self.advisor,
            "broken": BrokenProvider(),
        }.get(engine.name, TargetProvider())


def setup(client):
    providers = Providers()
    client.app.state.engines.provider_factory = providers
    prompt_id = make_prompt(client)
    target, broken = add_engine(client, "target"), add_engine(client, "broken")
    judge_id = add_engine(client, "judge")
    advisor = add_engine(client, "advisor")
    save_judge(client, prompt_id, judge_id)
    run_id = start(client, prompt_id, [target, broken])
    wait_for(client, run_id)
    return providers, prompt_id, run_id, target, advisor


def test_report_page(client):
    _, _, run_id, _, _ = setup(client)
    page = client.get(f"/runs/{run_id}/report")
    assert page.status_code == 200
    text = page.text
    assert "Scores by engine" in text
    assert "Static checks: pass rate" in text
    assert "Judge criteria: average score" in text
    # The plain-text output fails parsing, and the report says why.
    assert "Invalid JSON" in text
    assert "why Correctness" in text
    assert "model not loaded" in text  # the broken engine's error
    # Errors sort before scored results.
    assert text.index("model not loaded") < text.index("why Correctness")


def test_build_report_numbers(client):
    _, _, run_id, _, _ = setup(client)
    with Session(client.app.state.engine) as session:
        report = build_report(session.get(Run, run_id))
        target = next(e for e in report.engines if e.name == "target")
        broken = next(e for e in report.engines if e.name == "broken")
        # Static: only "finished" passes (10%); judge 65; final 0.3 × 10 + 0.7 × 65 = 48.5
        assert target.avg_static == 10.0
        assert target.avg_judge == 65.0
        assert target.avg_final == 48.5
        assert broken.failed == 1 and broken.avg_final is None
        parse = next(c for c in report.checks if c.name == "parse")
        assert parse.per_engine["target"].percent == 0.0
        assert report.prompt_issues == [("Ana", "target", "Say which label set to use.")]


def test_markdown_download(client):
    _, _, run_id, _, _ = setup(client)
    response = client.get(f"/runs/{run_id}/report.md")
    assert response.headers["content-type"].startswith("text/markdown")
    assert f'filename="run-{run_id}-report.md"' in response.headers["content-disposition"]
    body = response.text
    assert body.startswith("# Report: Greet v1")
    assert "| target | `m-target` | **48.5** |" in body
    assert "- ✗ Parses (40%): Invalid JSON" in body
    assert "&#" not in body and "&lt;" not in body  # not HTML-escaped


def test_suggestions_and_apply(client):
    providers, prompt_id, run_id, _, advisor = setup(client)
    fragment = client.post(f"/runs/{run_id}/suggestions", data={"engine_id": advisor})
    assert "The prompt never asks for JSON." in fragment.text
    assert "Save as the next version" in fragment.text

    request = providers.advisor.requests[0]
    assert "<current_template>\nSay hi to {{name}}\n</current_template>" in request.user
    assert "Say which label set to use." in request.user  # judge's prompt issue
    assert "Invalid JSON" in request.user  # failed check detail of the worst case

    assert "The prompt never asks for JSON." in client.get(f"/runs/{run_id}/report.md").text

    client.post(f"/runs/{run_id}/suggestions/apply")
    with Session(client.app.state.engine) as session:
        versions = session.exec(
            select(PromptVersion).where(PromptVersion.prompt_id == prompt_id)
        ).all()
        assert [v.version for v in versions] == [1, 2]
        assert versions[1].template == "JSON please {{name}}"
        assert versions[1].notes == f"Suggested from run #{run_id} (v1)"


def test_suggestion_warns_about_dropped_variables(client):
    providers, _, run_id, _, advisor = setup(client)
    providers.advisor.answer = {**SUGGESTION, "revised_template": "Say hi to {{person}}"}
    fragment = client.post(f"/runs/{run_id}/suggestions", data={"engine_id": advisor})
    assert "drops name" in fragment.text
    assert "adds person" in fragment.text


def test_parse_rejects_missing_template():
    with pytest.raises(EngineError):
        suggestions.parse(json.dumps({"diagnosis": "x"}), ["name"])


def test_compare_versions(client):
    _, prompt_id, run_id, target, advisor = setup(client)
    client.post(f"/runs/{run_id}/suggestions", data={"engine_id": advisor})
    client.post(f"/runs/{run_id}/suggestions/apply")
    with Session(client.app.state.engine) as session:
        v2 = session.exec(select(PromptVersion).where(PromptVersion.version == 2)).one()
    response = client.post(
        f"/prompts/{prompt_id}/runs",
        data={"version_id": v2.id, "engine_ids": [target]},
        follow_redirects=False,
    )
    second = int(response.headers["location"].rsplit("/", 1)[1])
    wait_for(client, second)

    report = client.get(f"/runs/{second}/report").text
    assert f"Run #{run_id} · v1" in report  # offered for comparison

    page = client.get(f"/runs/{second}/compare", params={"other": run_id})
    assert page.status_code == 200
    # Static goes from 10 to 100, judge stays 65: final 48.5 → 75.5.
    assert "+27.0" in page.text
    assert f"Run #{run_id} (v1) → Run #{second} (v2)" in page.text
