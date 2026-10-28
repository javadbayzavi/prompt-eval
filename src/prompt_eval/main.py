from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from prompt_eval.config import Settings, get_settings
from prompt_eval.db import init_db, make_engine
from prompt_eval.engines import EngineRegistry
from prompt_eval.routes import engines, generation, home, projects, prompts, runs, test_cases
from prompt_eval.runner import RunManager
from prompt_eval.web import BASE_DIR


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    db_engine = make_engine(settings.database_url)
    registry = EngineRegistry(settings)
    run_manager = RunManager(db_engine, registry)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        init_db(db_engine)
        run_manager.recover_interrupted()
        yield
        await run_manager.shutdown()
        await registry.aclose()
        db_engine.dispose()

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = db_engine
    app.state.engines = registry
    app.state.runs = run_manager

    app.mount("/static", StaticFiles(directory=Path(BASE_DIR, "static")), name="static")
    for module in (home, projects, prompts, test_cases, engines, generation, runs):
        app.include_router(module.router)
    return app
