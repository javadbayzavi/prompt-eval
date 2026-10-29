def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_redirects_to_projects(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/projects"


def test_static_css_served(client):
    response = client.get("/static/style.css")
    assert response.status_code == 200
