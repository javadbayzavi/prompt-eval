import asyncio
import random

from prompt_eval.engines.base import ChatRequest, ChatResponse, Provider, RetryableError
from prompt_eval.engines.ratelimit import RateLimiter, Sleep


def estimate_tokens(request: ChatRequest) -> int:
    # Rough input-size estimate (~4 chars/token); corrected after the call with real usage.
    return max(1, (len(request.system) + len(request.user)) // 4)


class EngineClient:
    """A provider wrapped with its rate limiter and retry policy.

    One instance per engine is shared by everything in the process (see `EngineRegistry`),
    so concurrent runs on the same engine share its limits.
    """

    def __init__(
        self,
        provider: Provider,
        limiter: RateLimiter,
        *,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        sleep: Sleep = asyncio.sleep,
    ):
        self.provider = provider
        self.limiter = limiter
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.sleep = sleep

    def backoff(self, attempt: int) -> float:
        delay = self.base_delay * (2**attempt)
        return min(self.max_delay, delay + random.uniform(0, delay / 2))

    async def complete(self, request: ChatRequest) -> ChatResponse:
        estimated = estimate_tokens(request)
        attempt = 0
        while True:
            async with self.limiter.semaphore:
                await self.limiter.before_call(estimated)
                try:
                    response = await self.provider.complete(request)
                except RetryableError as e:
                    # The failed call still counted against the window.
                    self.limiter.after_call(estimated, estimated)
                    if attempt >= self.max_retries:
                        raise
                    delay = e.retry_after if e.retry_after is not None else self.backoff(attempt)
                    if e.rate_limited:
                        self.limiter.pause(delay)
                else:
                    self.limiter.after_call(
                        estimated, response.input_tokens + response.output_tokens
                    )
                    response.attempts = attempt + 1
                    return response
            # Sleep outside the semaphore so other workers can use the slot.
            await self.sleep(delay)
            attempt += 1

    async def aclose(self) -> None:
        await self.provider.aclose()
