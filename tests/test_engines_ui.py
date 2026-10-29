from sqlmodel import Session, select

from prompt_eval.engines.base import ChatResponse, EngineError
from prompt_eval.models import Engine
from prompt_eval.routes.engines import suggest_limits

FORM = {
    "name": "opus",
    "provider": "anthropic",
    "model": "claude-opus-5-5",
    "max_tokens": "4096",
    "effort": "high",
    "max_concurrency": "3",
    "requests_per_minute": "50",
    "tokens_per_minute": "",
    "max_retries": "4",
    "timeout_seconds": "120",
}


class FakeProvider:
    def __init__(self, error: Exception | None = None):
        self.error = error

    async def complete(self, request):
        if self.error:
            raise self.error
        return ChatResponse(text="pong", model="fake-model", input_tokens=5, output_tokens=1)

    async def aclose(self):
        pass


def get_engine(client, name="opus") -> Engine:
    with Session(client.app.state.engine) as session:
        return session.exec(select(Engine).where(Engine.name == name)).one()


def test_create_and_list_engine(client):
    response = client.post("/engines", data=FORM, follow_redirects=False)
    assert response.status_code == 303
    engine = get_engine(client)
    assert engine.effort == "high"
    assert engine.max_concurrency == 3
    assert engine.requests_per_minute == 50
    assert engine.tokens_per_minute is None
    assert engine.max_retries == 4

    page = client.get("/engines")
    assert "claude-opus-5-5" in page.text
    assert "50 req/min" in page.text


def test_duplicate_name_rejected(client):
    client.post("/engines", data=FORM)
    assert client.post("/engines", data=FORM).status_code == 400


def test_invalid_numbers_rejected(client):
    assert client.post("/engines", data={**FORM, "max_concurrency": "abc"}).status_code == 400
    assert client.post("/engines", data={**FORM, "requests_per_minute": "0"}).status_code == 400
    assert client.post("/engines", data={**FORM, "effort": "turbo"}).status_code == 400


def test_edit_engine(client):
    client.post("/engines", data=FORM)
    engine = get_engine(client)
    assert client.get(f"/engines/{engine.id}/edit").status_code == 200

    client.post(
        f"/engines/{engine.id}",
        data={**FORM, "provider": "openai_compatible", "model": "qwen", "temperature": "0.3"},
    )
    engine = get_engine(client)
    assert engine.provider == "openai_compatible"
    assert engine.temperature == 0.3


def test_delete_engine(client):
    client.post("/engines", data=FORM)
    engine = get_engine(client)
    client.post(f"/engines/{engine.id}/delete")
    with Session(client.app.state.engine) as session:
        assert session.exec(select(Engine)).all() == []


def test_test_button_shows_success(client):
    client.app.state.engines.provider_factory = lambda engine, settings: FakeProvider()
    client.post("/engines", data=FORM)
    engine = get_engine(client)
    fragment = client.post(f"/engines/{engine.id}/test")
    assert fragment.status_code == 200
    assert "✓" in fragment.text
    assert "pong" in fragment.text


def test_test_button_shows_error(client):
    client.app.state.engines.provider_factory = lambda engine, settings: FakeProvider(
        EngineError("bad key")
    )
    client.post("/engines", data=FORM)
    engine = get_engine(client)
    fragment = client.post(f"/engines/{engine.id}/test")
    assert "bad key" in fragment.text


def test_missing_api_key_reported(client):
    # Real factory with no key configured in test settings.
    client.post("/engines", data=FORM)
    engine = get_engine(client)
    fragment = client.post(f"/engines/{engine.id}/test")
    assert "ANTHROPIC_API_KEY" in fragment.text


LIMITS = {"requests": 10000, "input_tokens": 10000000, "output_tokens": 2000000}


class LimitedProvider(FakeProvider):
    async def complete(self, request):
        response = await super().complete(request)
        response.rate_limits = dict(LIMITS)
        return response


def test_suggest_limits_uses_the_tightest_token_limit():
    assert suggest_limits(LIMITS) == {"requests_per_minute": 8000, "tokens_per_minute": 1600000}
    assert suggest_limits({"tokens": 30000}) == {"tokens_per_minute": 24000}
    assert suggest_limits({}) == {}


def test_test_button_shows_limits_and_applies_suggestion(client):
    client.app.state.engines.provider_factory = lambda engine, settings: LimitedProvider()
    client.post("/engines", data=FORM)
    engine = get_engine(client)

    fragment = client.post(f"/engines/{engine.id}/test").text
    assert "Your limits per minute" in fragment
    assert "2,000,000 output tokens" in fragment
    assert "Use 8,000 req/min · 1,600,000 tok/min" in fragment

    response = client.post(
        f"/engines/{engine.id}/limits",
        data={"requests_per_minute": "8000", "tokens_per_minute": "1600000"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    engine = get_engine(client)
    assert (engine.requests_per_minute, engine.tokens_per_minute) == (8000, 1600000)
    assert engine.max_concurrency == 3  # other settings untouched


def test_test_button_without_reported_limits(client):
    client.app.state.engines.provider_factory = lambda engine, settings: FakeProvider()
    client.post("/engines", data=FORM)
    fragment = client.post(f"/engines/{get_engine(client).id}/test").text
    assert "reported no rate limits" in fragment
