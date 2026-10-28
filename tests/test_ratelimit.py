import asyncio

import pytest

from prompt_eval.engines.base import ChatRequest, ChatResponse, EngineError, RetryableError
from prompt_eval.engines.client import EngineClient
from prompt_eval.engines.ratelimit import RateLimiter, TokenBucket


class FakeClock:
    """Time only moves when someone sleeps."""

    def __init__(self):
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_bucket_allows_burst_then_waits():
    clock = FakeClock()
    bucket = TokenBucket(60, clock=clock, sleep=clock.sleep)  # 1 per second
    for _ in range(60):
        await bucket.acquire()
    assert clock.now == 0
    await bucket.acquire()
    assert clock.now == pytest.approx(1.0)


@pytest.mark.anyio
async def test_bucket_debit_makes_later_callers_wait():
    clock = FakeClock()
    bucket = TokenBucket(600, clock=clock, sleep=clock.sleep)  # 10 per second
    await bucket.acquire(100)
    bucket.debit(600)  # real usage was 600 more than estimated
    await bucket.acquire(10)
    # Balance was 500 - 600 = -100; needs 110 at 10/s.
    assert clock.now == pytest.approx(11.0)


@pytest.mark.anyio
async def test_oversized_request_waits_for_full_bucket_only():
    clock = FakeClock()
    bucket = TokenBucket(60, clock=clock, sleep=clock.sleep)
    await bucket.acquire(1000)  # clamped to capacity, so it does not wait forever
    assert clock.now == 0


@pytest.mark.anyio
async def test_pause_holds_callers():
    clock = FakeClock()
    limiter = RateLimiter(clock=clock, sleep=clock.sleep)
    limiter.pause(5)
    await limiter.before_call(10)
    assert clock.now == pytest.approx(5.0)


class ScriptedProvider:
    """Raises the scripted errors in order, then succeeds."""

    def __init__(self, errors=(), delay: float = 0.0):
        self.errors = list(errors)
        self.delay = delay
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            if self.errors:
                raise self.errors.pop(0)
            return ChatResponse(text="ok", model="fake", input_tokens=3, output_tokens=2)
        finally:
            self.in_flight -= 1

    async def aclose(self) -> None:
        pass


async def no_sleep(_seconds: float) -> None:
    pass


@pytest.mark.anyio
async def test_retries_then_succeeds():
    provider = ScriptedProvider([RetryableError("busy"), RetryableError("busy")])
    client = EngineClient(provider, RateLimiter(), max_retries=5, sleep=no_sleep)
    response = await client.complete(ChatRequest(user="hi"))
    assert response.text == "ok"
    assert response.attempts == 3
    assert provider.calls == 3


@pytest.mark.anyio
async def test_gives_up_after_max_retries():
    provider = ScriptedProvider([RetryableError("busy")] * 3)
    client = EngineClient(provider, RateLimiter(), max_retries=2, sleep=no_sleep)
    with pytest.raises(RetryableError):
        await client.complete(ChatRequest(user="hi"))
    assert provider.calls == 3


@pytest.mark.anyio
async def test_non_retryable_error_is_not_retried():
    provider = ScriptedProvider([EngineError("bad key")])
    client = EngineClient(provider, RateLimiter(), sleep=no_sleep)
    with pytest.raises(EngineError):
        await client.complete(ChatRequest(user="hi"))
    assert provider.calls == 1


@pytest.mark.anyio
async def test_rate_limit_uses_retry_after_and_pauses_engine():
    slept = []

    async def record_sleep(seconds: float) -> None:
        slept.append(seconds)

    clock = FakeClock()
    limiter = RateLimiter(clock=clock, sleep=clock.sleep)
    provider = ScriptedProvider([RetryableError("429", retry_after=7, rate_limited=True)])
    client = EngineClient(provider, limiter, sleep=record_sleep)
    await client.complete(ChatRequest(user="hi"))
    assert slept == [7]
    # The retry waited on the engine-wide pause too.
    assert limiter.paused_until == 7
    assert clock.slept == [7]


@pytest.mark.anyio
async def test_concurrency_cap_is_respected():
    provider = ScriptedProvider(delay=0.01)
    client = EngineClient(provider, RateLimiter(max_concurrency=2))
    await asyncio.gather(*(client.complete(ChatRequest(user="hi")) for _ in range(8)))
    assert provider.calls == 8
    assert provider.max_in_flight == 2
