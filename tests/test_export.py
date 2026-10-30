import json

import pytest
from sqlmodel import Session, select

from prompt_eval.models import Engine, EngineProvider, Project
from prompt_eval.routes.file_format import PROJECT_EXAMPLE, PROMPT_EXAMPLE


@pytest.fixture
def db(client):
    with Session(client.app.state.engine) as session:
        yield session


def import_new(client, data: dict) -> int:
    response = client.post(
        "/projects/import", data={"content": json.dumps(data)}, follow_redirects=False
    )
    assert response.status_code == 303, response.text
    return int(response.headers["location"].rsplit("/", 1)[1])


def with_judge(db) -> dict:
    db.add(Engine(name="Opus judge", provider=EngineProvider.ANTHROPIC, model="claude-opus-5-5"))
    db.commit()
    data = json.loads(json.dumps(PROJECT_EXAMPLE))
    data["prompts"][0]["judge_engine"] = "Opus judge"
    return data


def test_a_project_comes_back_the_same_after_export_and_import(client, db):
    # Given an imported project with a judge engine
    first = import_new(client, with_judge(db))

    # When it is exported and the export imported again
    exported = client.get(f"/projects/{first}/export")
    second = import_new(client, exported.json())

    # Then the second export equals the first, as a download named after the project
    assert "attachment" in exported.headers["content-disposition"]
    assert 'filename="support.json"' in exported.headers["content-disposition"]
    assert client.get(f"/projects/{second}/export").json() == exported.json()
    assert exported.json()["prompts"][0]["judge_engine"] == "Opus judge"


def test_exports_one_prompt_under_its_project_without_the_others(client, db):
    # Given a project with two prompts
    import_new(client, PROJECT_EXAMPLE)
    prompt = db.exec(select(Project)).one().prompts[0]

    # When only the first is exported
    response = client.get(f"/prompts/{prompt.id}/export")

    # Then the file holds that prompt alone, under the project's name
    data = response.json()
    assert 'filename="classify-ticket.json"' in response.headers["content-disposition"]
    assert data["project"]["name"] == "Support"
    assert [p["name"] for p in data["prompts"]] == ["Classify ticket"]
    assert data["prompts"][0]["output_schema"]["type"] == "object"


def test_imports_a_prompt_into_an_existing_project_keeping_the_rest(client, db):
    # Given a project with one prompt, and an exported prompt from elsewhere
    project_id = import_new(
        client,
        {
            "format": "prompt-eval/1",
            "project": {"name": "Mine"},
            "prompts": [PROJECT_EXAMPLE["prompts"][1]],
        },
    )

    # When the single-prompt file is imported into it
    response = client.post(
        f"/projects/{project_id}/import",
        data={"content": json.dumps(PROMPT_EXAMPLE)},
        follow_redirects=False,
    )

    # Then the project keeps its name and gains the prompt, which the browser opens
    project = db.get(Project, project_id)
    db.refresh(project)
    assert response.status_code == 303
    assert response.headers["location"] == f"/prompts/{project.prompts[-1].id}"
    assert project.name == "Mine"
    assert [p.name for p in project.prompts] == ["Reply draft", "Classify ticket"]
    assert [v.version for v in project.prompts[-1].versions] == [1, 2]


def test_refuses_a_prompt_whose_name_the_project_already_has(client, db):
    # Given a project that has the prompt already
    project_id = import_new(client, PROMPT_EXAMPLE)

    # When it is imported again
    response = client.post(
        f"/projects/{project_id}/import",
        data={"content": json.dumps(PROMPT_EXAMPLE)},
        follow_redirects=False,
    )

    # Then nothing is added and the page says why
    assert response.status_code == 400
    assert "already has a prompt with this name" in response.text
    assert len(db.get(Project, project_id).prompts) == 1


def test_refuses_a_file_with_two_prompts_of_the_same_name(client, db):
    data = json.loads(json.dumps(PROMPT_EXAMPLE))
    data["prompts"].append(data["prompts"][0])

    response = client.post(
        "/projects/import", data={"content": json.dumps(data)}, follow_redirects=False
    )

    assert response.status_code == 400
    assert "more than one prompt with this name" in response.text
    assert db.exec(select(Project)).all() == []


def test_the_file_format_guide_shows_both_examples_and_is_in_the_sidebar(client):
    page = client.get("/help/file-format")

    assert page.status_code == 200
    assert "A project file" in page.text and "A single-prompt file" in page.text
    assert "forbidden_regex" in page.text
    assert 'data-copy="project-example"' in page.text and 'id="project-example"' in page.text
    assert 'data-copy="prompt-example"' in page.text and 'id="prompt-example"' in page.text
    assert 'href="/help/file-format"' in client.get("/projects").text


def test_the_project_page_links_to_export_and_import(client):
    project_id = import_new(client, PROJECT_EXAMPLE)

    page = client.get(f"/projects/{project_id}").text

    assert f'href="/projects/{project_id}/export"' in page
    assert f'href="/projects/{project_id}/import"' in page
