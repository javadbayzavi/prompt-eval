import json

from sqlalchemy import create_engine, inspect, text
from sqlmodel import Session, select

from prompt_eval.db import add_missing_columns, init_db
from prompt_eval.models import Prompt, RunResult
from tests.test_runs import EchoProvider, add_engine, start, wait_for

SCHEMA = json.dumps({"type": "object", "required": ["greeting"]})


def test_add_missing_columns_upgrades_old_database(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with engine.begin() as conn:
        # A prompt table as it was before static checks existed.
        conn.execute(
            text(
                "CREATE TABLE prompt (id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL, "
                "name VARCHAR NOT NULL, task_description VARCHAR NOT NULL, "
                "output_format VARCHAR NOT NULL, output_schema VARCHAR NOT NULL, "
                "created_at DATETIME NOT NULL)"
            )
        )
        conn.execute(
            text("INSERT INTO prompt VALUES (1, 1, 'old', '', 'TEXT', '', '2026-01-01 00:00:00')")
        )
    init_db(engine)
    columns = {c["name"] for c in inspect(engine).get_columns("prompt")}
    assert "static_checks" in columns
    with Session(engine) as session:
        assert session.get(Prompt, 1).static_checks is None
    assert add_missing_columns(engine) == []  # idempotent


def make_prompt(client, output_format="json", template="Say hi to {{name}}") -> int:
    project = client.post("/projects", data={"name": "P"}, follow_redirects=False)
    project_id = int(project.headers["location"].rsplit("/", 1)[1])
    prompt = client.post(
        f"/projects/{project_id}/prompts",
        data={
            "name": "Greet",
            "template": template,
            "output_format": output_format,
            "output_schema": SCHEMA,
        },
        follow_redirects=False,
    )
    prompt_id = int(prompt.headers["location"].rsplit("/", 1)[1])
    client.post(f"/prompts/{prompt_id}/test-cases", data={"name": "Ana", "input__name": "Ana"})
    return prompt_id


def saved_checks(client, prompt_id):
    with Session(client.app.state.engine) as session:
        return session.get(Prompt, prompt_id).static_checks


def test_prompt_page_shows_default_checks(client):
    prompt_id = make_prompt(client)
    page = client.get(f"/prompts/{prompt_id}").text
    assert "Matches schema" in page
    assert "These are the defaults" in page


def test_save_valid_checks(client):
    prompt_id = make_prompt(client)
    fragment = client.post(
        f"/prompts/{prompt_id}/scoring",
        data={
            "enabled__parse": "on",
            "weight__parse": "50",
            "enabled__forbidden_regex": "on",
            "weight__forbidden_regex": "50",
            "param__forbidden_regex": "sorry",
            "weight__schema": "99",  # not enabled: ignored
        },
    )
    assert "✓ Saved." in fragment.text
    assert saved_checks(client, prompt_id) == [
        {"name": "parse", "weight": 50.0, "param": ""},
        {"name": "forbidden_regex", "weight": 50.0, "param": "sorry"},
    ]


def test_invalid_checks_are_not_saved(client):
    prompt_id = make_prompt(client)
    fragment = client.post(
        f"/prompts/{prompt_id}/scoring",
        data={"enabled__parse": "on", "weight__parse": "70", "enabled__regex": "on"},
    )
    assert "Must match: weight must be a number" in fragment.text
    assert 'name="enabled__regex" checked' in fragment.text  # keeps what was entered
    assert saved_checks(client, prompt_id) is None


def test_reset_to_defaults(client):
    prompt_id = make_prompt(client)
    client.post(
        f"/prompts/{prompt_id}/scoring", data={"enabled__parse": "on", "weight__parse": "100"}
    )
    fragment = client.post(f"/prompts/{prompt_id}/scoring", data={"reset": "1"})
    assert "Reset to defaults" in fragment.text
    assert saved_checks(client, prompt_id) is None


class JsonProvider(EchoProvider):
    async def complete(self, request):
        response = await super().complete(request)
        response.text = '```json\n{"greeting": "hi"}\n```'
        return response


def test_runs_are_scored_and_can_be_rescored(client):
    client.app.state.engines.provider_factory = lambda engine, settings: JsonProvider()
    prompt_id = make_prompt(client)
    run_id = start(client, prompt_id, [add_engine(client, "a")])
    wait_for(client, run_id)

    with Session(client.app.state.engine) as session:
        result = session.exec(select(RunResult)).one()
    # Defaults with a schema: parse 40 ✓, schema 40 ✓, no_wrapping 10 ✗ (fenced), finished 10 ✓
    assert result.static_score == 90.0
    assert [c["name"] for c in result.static_checks] == [
        "parse",
        "schema",
        "no_wrapping",
        "finished",
    ]
    page = client.get(f"/runs/{run_id}").text
    assert "Wrapped in a markdown code fence" in page

    # Stop caring about fences, then rescore the existing run without calling the model.
    client.post(
        f"/prompts/{prompt_id}/scoring",
        data={"enabled__parse": "on", "weight__parse": "100"},
    )
    client.post(f"/runs/{run_id}/rescore")
    with Session(client.app.state.engine) as session:
        assert session.exec(select(RunResult)).one().static_score == 100.0
