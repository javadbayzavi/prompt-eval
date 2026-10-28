"""Shared web helpers: template rendering and request-scoped dependencies."""

from collections.abc import Iterator
from pathlib import Path

from fastapi import Request
from fastapi.templating import Jinja2Templates
from sqlmodel import Session

from prompt_eval.db import session_scope

BASE_DIR = Path(__file__).parent

templates = Jinja2Templates(directory=BASE_DIR / "templates")


def get_session(request: Request) -> Iterator[Session]:
    yield from session_scope(request.app.state.engine)
