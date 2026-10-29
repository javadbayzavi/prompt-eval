import json

import pytest
from sqlmodel import Session, select

from prompt_eval.models import Engine, EngineProvider, Project, TestCaseCategory


@pytest.fixture
def db(client):
    with Session(client.app.state.engine) as session:
        yield session


def project_file(**prompt_overrides) -> dict:
    prompt = {
        "name": "Tagger",
        "task_description": "Tag a note",
        "output_format": "json",
        "output_schema": {"type": "object", "required": ["tags"]},
        "static_checks": [
            {"name": "parse", "weight": 40},
            {"name": "schema", "weight": 40},
            {"name": "forbidden_regex", "weight": 20, "param": "diagnos"},
        ],
        "static_weight": 25,
        "judge_criteria": [
            {"name": "Closed set", "description": "Only listed tags", "weight": 60},
            {"name": "Reasons", "description": "Grounded in the note", "weight": 40},
        ],
        "judge_instructions": "Be strict.",
        "versions": [
            {"template": "v1 {{note}}", "notes": "first"},
            {"system_prompt": "You tag.", "template": "Tags: {{tags}}\n<note>{{note}}</note>"},
        ],
        "test_cases": [
            {"name": "meal", "inputs": {"tags": "- a", "note": "Ate."}, "criteria": "Returns a"},
            {
                "name": "injection",
                "category": "adversarial",
                "inputs": {"tags": "- a", "note": "x"},
            },
        ],
    }
    return {
        "format": "prompt-eval/1",
        "project": {"name": "Kita", "description": "Synthetic"},
        "prompts": [prompt | prompt_overrides],
    }


def post(client, data: object):
    return client.post(
        "/projects/import", data={"content": json.dumps(data)}, follow_redirects=False
    )


def test_import_page_is_linked_from_projects(client):
    assert 'href="/projects/import"' in client.get("/projects").text
    assert "Import project" in client.get("/projects/import").text


def test_imports_a_project_with_versions_scoring_and_test_cases(client, db):
    response = post(client, project_file())

    assert response.status_code == 303
    project = db.exec(select(Project)).one()
    assert response.headers["location"] == f"/projects/{project.id}"
    prompt = project.prompts[0]
    assert (prompt.name, prompt.output_format, prompt.static_weight) == ("Tagger", "json", 25)
    assert json.loads(prompt.output_schema) == {"type": "object", "required": ["tags"]}
    assert [c["name"] for c in prompt.static_checks] == ["parse", "schema", "forbidden_regex"]
    assert [c["name"] for c in prompt.judge_criteria] == ["Closed set", "Reasons"]
    assert [(v.version, v.system_prompt) for v in prompt.versions] == [(1, ""), (2, "You tag.")]
    assert [(t.name, t.category) for t in prompt.test_cases] == [
        ("meal", TestCaseCategory.NORMAL),
        ("injection", TestCaseCategory.ADVERSARIAL),
    ]
    assert prompt.test_cases[0].inputs == {"tags": "- a", "note": "Ate."}


def test_imports_from_an_uploaded_file(client, db):
    body = json.dumps(project_file()).encode()
    response = client.post(
        "/projects/import",
        files={"file": ("kita.json", body, "application/json")},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert db.exec(select(Project)).one().name == "Kita"


def test_leaves_defaults_when_scoring_is_not_given(client, db):
    data = project_file()
    for key in ("static_checks", "judge_criteria"):
        del data["prompts"][0][key]

    post(client, data)

    prompt = db.exec(select(Project)).one().prompts[0]
    assert prompt.static_checks is None and prompt.judge_criteria is None


def test_sets_an_existing_judge_engine_by_name(client, db):
    db.add(Engine(name="Opus judge", provider=EngineProvider.ANTHROPIC, model="claude-opus-5-5"))
    db.commit()

    post(client, project_file(judge_engine="Opus judge"))

    prompt = db.exec(select(Project)).one().prompts[0]
    assert prompt.judge_engine_id == db.exec(select(Engine)).one().id


@pytest.mark.parametrize(
    ("overrides", "problem"),
    [
        ({"static_checks": [{"name": "parse", "weight": 50}]}, "add up to 100"),
        (
            {"output_schema": "", "static_checks": [{"name": "schema", "weight": 100}]},
            "JSON Schema",
        ),
        ({"judge_criteria": [{"name": "A", "weight": 90}]}, "add up to 100"),
        ({"judge_engine": "Missing"}, "No engine named"),
        ({"test_cases": [{"name": "bare", "inputs": {"note": "x"}}]}, "no value for tags"),
    ],
    ids=["check weights", "schema check without schema", "criteria weights", "engine", "inputs"],
)
def test_refuses_a_file_the_ui_would_refuse_and_saves_nothing(client, db, overrides, problem):
    response = post(client, project_file(**overrides))

    assert response.status_code == 400
    assert problem in response.text and "Tagger:" in response.text
    assert db.exec(select(Project)).all() == []


@pytest.mark.parametrize(
    "data",
    ["not json", {"format": "other/9", "project": {"name": "K"}}, {"project": {"name": ""}}],
    ids=["not json", "unknown format", "empty name"],
)
def test_refuses_a_malformed_file(client, db, data):
    response = client.post(
        "/projects/import",
        data={"content": data if isinstance(data, str) else json.dumps(data)},
        follow_redirects=False,
    )

    assert response.status_code == 400
    assert db.exec(select(Project)).all() == []
