from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings, read from environment variables or a local `.env` file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Prompt Eval"
    database_url: str = "sqlite:///./data/prompt_eval.db"

    # Engines. Keys are optional so the app starts without them.
    anthropic_api_key: str | None = None
    lmstudio_base_url: str = "http://localhost:1234/v1"


@lru_cache
def get_settings() -> Settings:
    return Settings()
