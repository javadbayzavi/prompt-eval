import asyncio
import time

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from prompt_eval.config import Settings
from prompt_eval.engines.base import ChatResponse, EngineError
from prompt_eval.main import create_app
from prompt_eval.models import ResultStatus, Run, RunResult, RunStatus


class EchoProvider:
    async def complete(self, request):
        await asyncio.sleep(0)
        return ChatResponse(
            text=f"echo: {request.user}",
            model="fake",
            input_tokens=4,
            output_tokens=2,
            latency_ms=7,
        )

    async def aclose(self):
        pass


class BrokenProvider(EchoProvider):
    async def complete(self, request):
        raise EngineError("model not loaded")


class SlowProvider(EchoProvider):
    async def complete(self, request):
        await asyncio.sleep(30)


class Providers:
    """Pick a fake provider by engine name; tests can swap behaviour mid-way."""

    def __init__(self):
        self.by_name = {"broken": BrokenProvider, "slow": SlowProvider}

    def __call__(self, engine, settings):
        return self.by_name.get(engine.name, EchoProvider)()


@pytest.fixture
def providers(client) -> Providers:
    providers = Providers()
    client.app.state.engines.provider_factory = providers
    return providers


def add_engine(client, name: str) -> int:
    client.post("/engines", data={"name": name, "provider": "anthropic", "model": f"m-{name}"})
    return len(client.get("/engines").text.split('hx-post="/engines/')) - 1


def setup_prompt(client) -> int:
    project = client.post("/projects", data={"name": "P"}, follow_redirects=False)
    project_id = int(project.headers["location"].rsplit("/", 1)[1])
    prompt = client.post(
        f"/projects/{project_id}/prompts",
        data={"name": "Greet", "system_prompt": "Lang {{lang}}", "template": "Hi {{name}}"},
        follow_redirects=False,
    )
    prompt_id = int(prompt.headers["location"].rsplit("/", 1)[1])
    for name, category in (("Ana", "normal"), ("Bo", "edge")):
        client.post(
            f"/prompts/{prompt_id}/test-cases",
            data={"name": name, "category": category, "input__name": name, "input__lang": "en"},
        )
    return prompt_id


def start(client, prompt_id: int, engine_ids: list[int], **extra) -> int:
    response = client.post(
        f"/prompts/{prompt_id}/runs",
        data={"version_id": 1, "engine_ids": engine_ids, **extra},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text
    return int(response.headers["location"].rsplit("/", 1)[1])


def wait_for(client, run_id: int, timeout: float = 5.0) -> Run:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with Session(client.app.state.engine) as session:
            run = session.get(Run, run_id)
            if not run.status.active and not client.app.state.runs.is_active(run_id):
                return run
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish")


def results(client, run_id: int) -> list[RunResult]:
    with Session(client.app.state.engine) as session:
        return session.exec(
            select(RunResult).where(RunResult.run_id == run_id).order_by(RunResult.id)
        ).all()


def test_run_on_two_engines(client, providers):
    prompt_id = setup_prompt(client)
    a, b = add_engine(client, "a"), add_engine(client, "b")
    run_id = start(client, prompt_id, [a, b])

    assert wait_for(client, run_id).status == RunStatus.COMPLETED
    rows = results(client, run_id)
    assert len(rows) == 4
    assert all(r.status == ResultStatus.DONE for r in rows)
    assert [(r.test_case_name, r.engine_name) for r in rows] == [
        ("Ana", "a"),
        ("Ana", "b"),
        ("Bo", "a"),
        ("Bo", "b"),
    ]
    assert rows[0].system_text == "Lang en"
    assert rows[0].output == "echo: Hi Ana"
    assert rows[0].model == "m-a"

    page = client.get(f"/runs/{run_id}")
    assert "echo: Hi Ana" in page.text
    assert "4 / 4" in page.text
    # Finished runs stop polling.
    assert 'hx-trigger="every 2s"' not in client.get(f"/runs/{run_id}/progress").text
    assert f"/runs/{run_id}" in client.get(f"/prompts/{prompt_id}/runs").text


def test_category_filter(client, providers):
    prompt_id = setup_prompt(client)
    a = add_engine(client, "a")
    run_id = start(client, prompt_id, [a], categories=["edge"])
    wait_for(client, run_id)
    assert [r.test_case_name for r in results(client, run_id)] == ["Bo"]


def test_failures_are_recorded_and_retryable(client, providers):
    prompt_id = setup_prompt(client)
    broken = add_engine(client, "broken")
    run_id = start(client, prompt_id, [broken])

    assert wait_for(client, run_id).status == RunStatus.COMPLETED
    rows = results(client, run_id)
    assert all(r.status == ResultStatus.ERROR for r in rows)
    assert rows[0].error == "model not loaded"
    assert "Retry 2 failed" in client.get(f"/runs/{run_id}").text

    providers.by_name.pop("broken")  # the model is loaded now
    client.app.state.engines._clients.clear()
    client.post(f"/runs/{run_id}/retry")
    wait_for(client, run_id)
    assert all(r.status == ResultStatus.DONE for r in results(client, run_id))


def test_cancel(client, providers):
    prompt_id = setup_prompt(client)
    slow = add_engine(client, "slow")
    run_id = start(client, prompt_id, [slow])
    assert 'hx-trigger="every 2s"' in client.get(f"/runs/{run_id}/progress").text

    client.post(f"/runs/{run_id}/cancel")
    assert wait_for(client, run_id).status == RunStatus.CANCELLED
    assert all(r.status == ResultStatus.CANCELLED for r in results(client, run_id))


def test_deleted_engine_fails_its_results(client, providers):
    prompt_id = setup_prompt(client)
    slow = add_engine(client, "slow")
    run_id = start(client, prompt_id, [slow])
    client.post(f"/runs/{run_id}/cancel")
    wait_for(client, run_id)

    client.post(f"/engines/{slow}/delete")
    client.post(f"/runs/{run_id}/retry")
    wait_for(client, run_id)
    rows = results(client, run_id)
    assert all(r.error == "The engine was deleted" for r in rows)
    assert rows[0].engine_name == "slow"  # snapshot survives


def test_validation(client, providers):
    prompt_id = setup_prompt(client)
    a = add_engine(client, "a")
    no_engine = client.post(f"/prompts/{prompt_id}/runs", data={"version_id": 1})
    assert no_engine.status_code == 400
    bad_version = client.post(
        f"/prompts/{prompt_id}/runs", data={"version_id": 99, "engine_ids": a}
    )
    assert bad_version.status_code == 400


def test_delete_run(client, providers):
    prompt_id = setup_prompt(client)
    run_id = start(client, prompt_id, [add_engine(client, "a")])
    wait_for(client, run_id)
    client.post(f"/runs/{run_id}/delete")
    with Session(client.app.state.engine) as session:
        assert session.exec(select(Run)).all() == []
        assert session.exec(select(RunResult)).all() == []


def test_startup_marks_interrupted_runs(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path / 'db.sqlite'}")
    with TestClient(create_app(settings)) as first:
        first.app.state.engines.provider_factory = Providers()
        prompt_id = setup_prompt(first)
        run_id = start(first, prompt_id, [add_engine(first, "slow")])
        # Simulate a crash: the server stops without a clean shutdown.
        first.app.state.runs.tasks.clear()

    with Session(first.app.state.engine) as session:
        run = session.get(Run, run_id)
        run.status = RunStatus.RUNNING  # what a crashed process leaves behind
        session.commit()

    second = create_app(settings)
    with TestClient(second), Session(second.state.engine) as session:
        run = session.get(Run, run_id)
        assert run.status == RunStatus.INTERRUPTED
        assert all(r.status == ResultStatus.CANCELLED for r in run.results)
