import re

import pytest
from sqlmodel import Session, select

from prompt_eval.models import Project, Prompt, PromptVersion, TestCase


def location_id(response) -> int:
    return int(re.search(r"/(\d+)", response.headers["location"]).group(1))


@pytest.fixture
def db(client):
    with Session(client.app.state.engine) as session:
        yield session


@pytest.fixture
def project_id(client) -> int:
    response = client.post(
        "/projects", data={"name": "Support bot", "description": "x"}, follow_redirects=False
    )
    assert response.status_code == 303
    return location_id(response)


@pytest.fixture
def prompt_id(client, project_id) -> int:
    response = client.post(
        f"/projects/{project_id}/prompts",
        data={
            "name": "Classify ticket",
            "task_description": "Classify a support ticket",
            "output_format": "json",
            "output_schema": '{"type": "object"}',
            "system_prompt": "You are a classifier.",
            "template": "Ticket: {{ticket}}\nLanguage: {{lang}}",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    return location_id(response)


def test_project_crud(client, db, project_id):
    page = client.get("/projects")
    assert "Support bot" in page.text

    client.post(f"/projects/{project_id}", data={"name": "Renamed", "description": ""})
    assert db.get(Project, project_id).name == "Renamed"

    client.post(f"/projects/{project_id}/delete")
    db.expire_all()
    assert db.get(Project, project_id) is None


def test_blank_name_rejected(client):
    assert client.post("/projects", data={"name": "   "}).status_code == 400


def test_missing_project_is_404(client):
    assert client.get("/projects/999").status_code == 404


def test_create_prompt_with_first_version(client, db, prompt_id):
    prompt = db.get(Prompt, prompt_id)
    assert prompt.output_format == "json"
    assert [v.version for v in prompt.versions] == [1]
    assert prompt.versions[0].system_prompt == "You are a classifier."

    project_page = client.get(f"/projects/{prompt.project_id}")
    assert project_page.status_code == 200
    assert "Classify ticket" in project_page.text

    page = client.get(f"/prompts/{prompt_id}")
    assert page.status_code == 200
    # One input field per template variable.
    assert 'name="input__ticket"' in page.text
    assert 'name="input__lang"' in page.text


def test_new_versions_increment(client, db, prompt_id):
    client.post(f"/prompts/{prompt_id}/versions", data={"template": "v2 {{ticket}}", "notes": "a"})
    client.post(f"/prompts/{prompt_id}/versions", data={"template": "v3 {{ticket}}"})
    versions = db.exec(select(PromptVersion).where(PromptVersion.prompt_id == prompt_id)).all()
    assert sorted(v.version for v in versions) == [1, 2, 3]

    page = client.get(f"/prompts/{prompt_id}")
    assert "Save as v4" in page.text


def test_update_prompt_settings(client, db, prompt_id):
    client.post(
        f"/prompts/{prompt_id}",
        data={"name": "New", "task_description": "t", "output_format": "yaml", "output_schema": ""},
    )
    assert db.get(Prompt, prompt_id).output_format == "yaml"


def test_invalid_output_format_rejected(client, prompt_id):
    response = client.post(f"/prompts/{prompt_id}", data={"name": "x", "output_format": "csv"})
    assert response.status_code == 422


def test_test_case_crud(client, db, prompt_id):
    response = client.post(
        f"/prompts/{prompt_id}/test-cases",
        data={
            "name": "Refund request",
            "category": "edge",
            "input__ticket": "I want my money back",
            "input__lang": "en",
            "criteria": "Must classify as billing",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    test_case = db.exec(select(TestCase)).one()
    assert test_case.inputs == {"ticket": "I want my money back", "lang": "en"}
    assert test_case.category == "edge"
    assert test_case.source == "manual"

    edit = client.get(f"/test-cases/{test_case.id}/edit")
    assert "I want my money back" in edit.text

    client.post(
        f"/test-cases/{test_case.id}",
        data={"name": "Refund", "category": "normal", "input__ticket": "refund pls"},
    )
    db.refresh(test_case)
    assert test_case.name == "Refund"
    assert test_case.inputs == {"ticket": "refund pls"}

    client.post(f"/test-cases/{test_case.id}/delete")
    db.expire_all()
    assert db.exec(select(TestCase)).all() == []


def test_deleting_project_cascades(client, db, project_id, prompt_id):
    client.post(f"/prompts/{prompt_id}/test-cases", data={"name": "a", "input__ticket": "x"})
    client.post(f"/projects/{project_id}/delete")
    db.expire_all()
    assert db.exec(select(Prompt)).all() == []
    assert db.exec(select(PromptVersion)).all() == []
    assert db.exec(select(TestCase)).all() == []
