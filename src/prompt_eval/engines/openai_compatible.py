"""Any server that speaks the OpenAI chat completions API, e.g. LM Studio."""

import time

import httpx

from prompt_eval.engines.base import (
    ChatRequest,
    ChatResponse,
    EngineError,
    RetryableError,
    parse_retry_after,
)


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        max_tokens: int,
        temperature: float | None = None,
        api_key: str | None = None,
        timeout: float = 600.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def complete(self, request: ChatRequest) -> ChatResponse:
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.user})
        body: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": request.max_tokens or self.max_tokens,
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature

        started = time.perf_counter()
        try:
            response = await self.client.post("/chat/completions", json=body)
        except httpx.TransportError as e:  # connect errors and timeouts
            raise RetryableError(f"Connection error: {e!r}") from e

        if response.status_code == 429 or response.status_code >= 500:
            raise RetryableError(
                f"HTTP {response.status_code}: {response.text[:300]}",
                retry_after=parse_retry_after(response.headers.get("retry-after")),
                rate_limited=response.status_code == 429,
            )
        if response.status_code >= 400:
            raise EngineError(f"HTTP {response.status_code}: {response.text[:300]}")

        try:
            data = response.json()
            choice = data["choices"][0]
        except (ValueError, KeyError, IndexError) as e:
            raise EngineError(f"Unexpected response: {response.text[:300]}") from e

        usage = data.get("usage") or {}
        return ChatResponse(
            text=choice.get("message", {}).get("content") or "",
            model=data.get("model", self.model),
            stop_reason=choice.get("finish_reason"),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=int((time.perf_counter() - started) * 1000),
            raw=data,
        )

    async def aclose(self) -> None:
        await self.client.aclose()
