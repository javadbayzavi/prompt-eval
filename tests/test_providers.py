import json

import anthropic
import httpx
import httpx2
import pytest

from prompt_eval.engines.anthropic_provider import AnthropicProvider
from prompt_eval.engines.base import ChatRequest, EngineError, RetryableError
from prompt_eval.engines.openai_compatible import OpenAICompatibleProvider


@pytest.fixture
def anyio_backend():
    return "asyncio"


def openai_provider(handler, **kwargs) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        base_url="http://lmstudio.test/v1",
        model="qwen",
        max_tokens=100,
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


@pytest.mark.anyio
async def test_openai_compatible_success():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "qwen",
                "choices": [{"message": {"content": "pong"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 1},
            },
        )

    provider = openai_provider(handler, temperature=0.2)
    response = await provider.complete(ChatRequest(user="ping", system="be brief"))
    assert response.text == "pong"
    assert (response.input_tokens, response.output_tokens) == (12, 1)
    assert seen["url"] == "http://lmstudio.test/v1/chat/completions"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "be brief"}
    assert seen["body"]["temperature"] == 0.2
    assert seen["body"]["max_tokens"] == 100


@pytest.mark.anyio
@pytest.mark.parametrize("status", [429, 500, 503])
async def test_openai_compatible_retryable_statuses(status):
    def handler(request):
        return httpx.Response(status, headers={"retry-after": "2"}, text="slow down")

    with pytest.raises(RetryableError) as info:
        await openai_provider(handler).complete(ChatRequest(user="x"))
    assert info.value.retry_after == 2
    assert info.value.rate_limited == (status == 429)


@pytest.mark.anyio
async def test_openai_compatible_client_error_is_final():
    def handler(request):
        return httpx.Response(400, text="bad model")

    with pytest.raises(EngineError) as info:
        await openai_provider(handler).complete(ChatRequest(user="x"))
    assert not isinstance(info.value, RetryableError)


@pytest.mark.anyio
async def test_openai_compatible_connection_error_is_retryable():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(RetryableError):
        await openai_provider(handler).complete(ChatRequest(user="x"))


def test_anthropic_requires_key():
    with pytest.raises(EngineError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider(api_key=None, model="claude-opus-5-5", max_tokens=100)


def anthropic_provider(handler, **kwargs) -> AnthropicProvider:
    client = anthropic.AsyncAnthropic(
        api_key="test",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return AnthropicProvider(
        api_key="test", model="claude-opus-5-5", max_tokens=100, client=client, **kwargs
    )


@pytest.mark.anyio
async def test_anthropic_rate_limit_maps_to_retryable():
    def handler(request):
        return httpx2.Response(
            429,
            headers={"retry-after": "4"},
            json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}},
        )

    with pytest.raises(RetryableError) as info:
        await anthropic_provider(handler).complete(ChatRequest(user="x"))
    assert info.value.rate_limited
    assert info.value.retry_after == 4


@pytest.mark.anyio
async def test_anthropic_overloaded_is_retryable():
    def handler(request):
        return httpx2.Response(
            529, json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
        )

    with pytest.raises(RetryableError) as info:
        await anthropic_provider(handler).complete(ChatRequest(user="x"))
    assert not info.value.rate_limited


@pytest.mark.anyio
async def test_anthropic_bad_request_is_final():
    def handler(request):
        return httpx2.Response(
            400, json={"type": "error", "error": {"type": "invalid_request_error", "message": "no"}}
        )

    with pytest.raises(EngineError) as info:
        await anthropic_provider(handler).complete(ChatRequest(user="x"))
    assert not isinstance(info.value, RetryableError)


@pytest.mark.anyio
async def test_anthropic_success_sends_effort_and_reads_text():
    seen = {}
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-5-5",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 9, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "pong"},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 2},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    sse = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse.encode()
        )

    provider = anthropic_provider(handler, effort="low")
    response = await provider.complete(ChatRequest(user="ping", system="sys"))
    assert response.text == "pong"
    assert response.stop_reason == "end_turn"
    assert (response.input_tokens, response.output_tokens) == (9, 2)
    assert seen["body"]["output_config"] == {"effort": "low"}
    assert seen["body"]["system"] == "sys"
    assert "temperature" not in seen["body"]
