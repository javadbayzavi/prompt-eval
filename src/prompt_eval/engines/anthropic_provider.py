import time

import anthropic

from prompt_eval.engines.base import (
    ChatRequest,
    ChatResponse,
    EngineError,
    RetryableError,
    parse_retry_after,
)


class AnthropicProvider:
    """Claude via the official SDK. Retries are handled by `EngineClient`, not the SDK."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        max_tokens: int,
        effort: str | None = None,
        timeout: float = 600.0,
        client: anthropic.AsyncAnthropic | None = None,
    ):
        if client is None:
            if not api_key:
                raise EngineError("ANTHROPIC_API_KEY is not set (add it to .env)")
            client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=0, timeout=timeout)
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort

    async def complete(self, request: ChatRequest) -> ChatResponse:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": request.max_tokens or self.max_tokens,
            "messages": [{"role": "user", "content": request.user}],
        }
        if request.system:
            kwargs["system"] = request.system
        output_config: dict = {}
        if self.effort:
            output_config["effort"] = self.effort
        if request.json_schema:
            output_config["format"] = {"type": "json_schema", "schema": request.json_schema}
        if output_config:
            kwargs["output_config"] = output_config

        started = time.perf_counter()
        try:
            # Streaming avoids HTTP timeouts on long outputs; we only need the final message.
            async with self.client.messages.stream(**kwargs) as stream:
                message = await stream.get_final_message()
        except anthropic.RateLimitError as e:
            raise RetryableError(
                f"Rate limited: {e.message}",
                retry_after=parse_retry_after(e.response.headers.get("retry-after")),
                rate_limited=True,
            ) from e
        except (anthropic.OverloadedError, anthropic.InternalServerError) as e:
            raise RetryableError(
                f"Server error {e.status_code}: {e.message}",
                retry_after=parse_retry_after(e.response.headers.get("retry-after")),
            ) from e
        except anthropic.APIStatusError as e:
            raise EngineError(f"API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:  # includes timeouts
            raise RetryableError(f"Connection error: {e}") from e

        return ChatResponse(
            text="".join(block.text for block in message.content if block.type == "text"),
            model=message.model,
            stop_reason=message.stop_reason,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    async def aclose(self) -> None:
        await self.client.close()
