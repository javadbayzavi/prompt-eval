"""Per-engine throttling: concurrency cap, requests/min, tokens/min, and 429 cooldown.

All waits go through injectable `clock`/`sleep` so tests can run on a fake clock.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable

Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]


class TokenBucket:
    """Refills `per_minute` units per minute, up to a burst of `per_minute`.

    The balance may go negative after `debit()` when a call used more than estimated;
    later callers then wait for it to refill.
    """

    def __init__(
        self, per_minute: int, *, clock: Clock = time.monotonic, sleep: Sleep = asyncio.sleep
    ):
        if per_minute <= 0:
            raise ValueError("per_minute must be positive")
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0
        self.tokens = self.capacity
        self.clock = clock
        self.sleep = sleep
        self.updated = clock()
        self.lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now

    async def acquire(self, amount: float = 1.0) -> None:
        # A request larger than the whole bucket can never fit; let it through at full bucket.
        amount = min(amount, self.capacity)
        async with self.lock:  # FIFO: one waiter at a time
            while True:
                self._refill()
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                await self.sleep((amount - self.tokens) / self.rate)

    def debit(self, amount: float) -> None:
        """Charge (or refund, if negative) after the real usage is known."""
        self._refill()
        self.tokens = min(self.capacity, self.tokens - amount)


class RateLimiter:
    def __init__(
        self,
        *,
        max_concurrency: int = 1,
        requests_per_minute: int | None = None,
        tokens_per_minute: int | None = None,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ):
        self.semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self.requests = (
            TokenBucket(requests_per_minute, clock=clock, sleep=sleep)
            if requests_per_minute
            else None
        )
        self.tokens = (
            TokenBucket(tokens_per_minute, clock=clock, sleep=sleep) if tokens_per_minute else None
        )
        self.clock = clock
        self.sleep = sleep
        self.paused_until = 0.0

    def pause(self, seconds: float) -> None:
        """Hold every caller of this engine, e.g. after a 429 with retry-after."""
        self.paused_until = max(self.paused_until, self.clock() + seconds)

    async def wait_if_paused(self) -> None:
        while (remaining := self.paused_until - self.clock()) > 0:
            await self.sleep(remaining)

    async def before_call(self, estimated_tokens: int) -> None:
        await self.wait_if_paused()
        if self.requests:
            await self.requests.acquire(1)
        if self.tokens:
            await self.tokens.acquire(estimated_tokens)

    def after_call(self, estimated_tokens: int, actual_tokens: int) -> None:
        if self.tokens:
            self.tokens.debit(actual_tokens - estimated_tokens)
