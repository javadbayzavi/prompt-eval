"""Provider-neutral request/response types and errors for engines."""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ChatRequest:
    user: str
    system: str = ""
    max_tokens: int | None = None  # None: use the engine's configured default
    # When set, the provider constrains the reply to JSON matching this schema
    # (all object properties required, additionalProperties false).
    json_schema: dict | None = None


@dataclass
class ChatResponse:
    text: str
    model: str
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    attempts: int = 1
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"


class EngineError(Exception):
    """A failure that retrying will not fix (bad request, auth, missing config)."""


class RetryableError(EngineError):
    """A transient failure: rate limit, overload, server error, network problem."""

    def __init__(
        self, message: str, *, retry_after: float | None = None, rate_limited: bool = False
    ):
        super().__init__(message)
        self.retry_after = retry_after
        self.rate_limited = rate_limited


class Provider(Protocol):
    async def complete(self, request: ChatRequest) -> ChatResponse: ...

    async def aclose(self) -> None: ...


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None  # HTTP-date form; fall back to backoff
