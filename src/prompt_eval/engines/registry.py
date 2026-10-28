from collections.abc import Callable

from prompt_eval.config import Settings
from prompt_eval.engines.anthropic_provider import AnthropicProvider
from prompt_eval.engines.base import Provider
from prompt_eval.engines.client import EngineClient
from prompt_eval.engines.openai_compatible import OpenAICompatibleProvider
from prompt_eval.engines.ratelimit import RateLimiter
from prompt_eval.models import Engine, EngineProvider

ProviderFactory = Callable[[Engine, Settings], Provider]


def build_provider(engine: Engine, settings: Settings) -> Provider:
    if engine.provider == EngineProvider.ANTHROPIC:
        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=engine.model,
            max_tokens=engine.max_tokens,
            effort=engine.effort or None,
            timeout=engine.timeout_seconds,
        )
    return OpenAICompatibleProvider(
        base_url=engine.base_url or settings.lmstudio_base_url,
        model=engine.model,
        max_tokens=engine.max_tokens,
        temperature=engine.temperature,
        timeout=engine.timeout_seconds,
    )


class EngineRegistry:
    """Keeps one `EngineClient` per engine so all work in the process shares its limits.

    A client is rebuilt when the engine's settings change.
    """

    def __init__(self, settings: Settings, provider_factory: ProviderFactory = build_provider):
        self.settings = settings
        self.provider_factory = provider_factory
        self._clients: dict[int, tuple[object, EngineClient]] = {}

    async def get(self, engine: Engine) -> EngineClient:
        version = engine.updated_at
        cached = self._clients.get(engine.id)
        if cached and cached[0] == version:
            return cached[1]
        if cached:
            await cached[1].aclose()
        client = EngineClient(
            self.provider_factory(engine, self.settings),
            RateLimiter(
                max_concurrency=engine.max_concurrency,
                requests_per_minute=engine.requests_per_minute,
                tokens_per_minute=engine.tokens_per_minute,
            ),
            max_retries=engine.max_retries,
        )
        self._clients[engine.id] = (version, client)
        return client

    async def forget(self, engine_id: int) -> None:
        if cached := self._clients.pop(engine_id, None):
            await cached[1].aclose()

    async def aclose(self) -> None:
        for engine_id in list(self._clients):
            await self.forget(engine_id)
