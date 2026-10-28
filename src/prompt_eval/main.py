from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from prompt_eval.config import Settings, get_settings
from prompt_eval.db import init_db, make_engine
from prompt_eval.routes import home
from prompt_eval.web import BASE_DIR


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    engine = make_engine(settings.database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        init_db(engine)
        yield
        engine.dispose()

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = engine

    app.mount("/static", StaticFiles(directory=Path(BASE_DIR, "static")), name="static")
    app.include_router(home.router)
    return app
