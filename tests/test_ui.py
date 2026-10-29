import re

from tests.test_report import setup


def active_links(html: str) -> list[str]:
    """Text of every link marked active (sidebar items and tabs)."""
    found = re.findall(r'<a class="[^"]*\bactive\b[^"]*"[^>]*>(.*?)</a>', html, re.DOTALL)
    return [" ".join(re.sub(r"<[^>]+>", " ", text).split()) for text in found]


def test_every_page_renders(client):
    _, prompt_id, run_id, target, _ = setup(client)
    with client:
        pages = [
            "/projects",
            "/projects/new",
            "/projects/1",
            "/projects/1/settings",
            "/projects/1/prompts/new",
            f"/prompts/{prompt_id}",
            f"/prompts/{prompt_id}/test-cases",
            f"/prompts/{prompt_id}/test-cases/new",
            f"/prompts/{prompt_id}/test-cases/generate",
            f"/prompts/{prompt_id}/scoring",
            f"/prompts/{prompt_id}/runs",
            f"/prompts/{prompt_id}/runs/new",
            f"/prompts/{prompt_id}/settings",
            "/test-cases/1/edit",
            "/engines",
            "/engines/new",
            f"/engines/{target}/edit",
            f"/runs/{run_id}",
            f"/runs/{run_id}/report",
        ]
        for url in pages:
            response = client.get(url)
            assert response.status_code == 200, url
            assert 'class="sidebar"' in response.text, url


def test_sidebar_lists_projects_and_highlights_the_open_prompt(client):
    _, prompt_id, _, _, _ = setup(client)
    page = client.get(f"/prompts/{prompt_id}/scoring").text
    sidebar = page.split('<main class="main">')[0]
    assert "P" in sidebar and "Greet" in sidebar
    assert "+ New prompt" in sidebar
    assert active_links(page) == ["Greet", "Scoring"]


def test_sidebar_sections(client):
    setup(client)
    assert active_links(client.get("/engines").text) == ["⚙ Engines"]
    assert active_links(client.get("/projects").text) == ["▤ Projects"]
    # The prompt list is only expanded for the open project.
    assert "Greet" not in client.get("/projects").text.split('<main class="main">')[0]


def test_tab_counts(client):
    _, prompt_id, _, _, _ = setup(client)
    page = client.get(f"/prompts/{prompt_id}").text
    assert re.search(r'Test cases<span class="count">1</span>', page)
    assert re.search(r'Runs<span class="count">1</span>', page)
    assert re.search(r'Prompt<span class="count">v1</span>', page)


def test_empty_states_link_to_the_next_step(client):
    assert 'href="/projects/new"' in client.get("/projects").text
    assert 'href="/engines/new"' in client.get("/engines").text
    project = client.post("/projects", data={"name": "Empty"}, follow_redirects=False)
    project_id = int(project.headers["location"].rsplit("/", 1)[1])
    assert (
        f'href="/projects/{project_id}/prompts/new"' in client.get(f"/projects/{project_id}").text
    )


def test_prompt_settings_save_shows_confirmation(client):
    _, prompt_id, _, _, _ = setup(client)
    response = client.post(f"/prompts/{prompt_id}", data={"name": "Greet", "output_format": "json"})
    assert response.url.path == f"/prompts/{prompt_id}/settings"
    assert "✓ Saved." in response.text
