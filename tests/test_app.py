def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Prompt Eval" in response.text


def test_static_css_served(client):
    response = client.get("/static/style.css")
    assert response.status_code == 200
