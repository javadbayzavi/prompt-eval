"""Shared web helpers: template rendering and request-scoped dependencies."""

from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, SQLModel

from prompt_eval.db import session_scope

BASE_DIR = Path(__file__).parent

templates = Jinja2Templates(directory=BASE_DIR / "templates")


def get_session(request: Request) -> Iterator[Session]:
    yield from session_scope(request.app.state.engine)


SessionDep = Annotated[Session, Depends(get_session)]


def get_or_404[ModelT: SQLModel](session: Session, model: type[ModelT], id: int) -> ModelT:
    obj = session.get(model, id)
    if obj is None:
        raise HTTPException(status_code=404, detail=f"{model.__name__} {id} not found")
    return obj


def redirect(url: str) -> RedirectResponse:
    # 303 so the browser follows a POST with a GET.
    return RedirectResponse(url, status_code=303)


def require_text(value: str, field: str) -> str:
    value = value.strip()
    if not value:
        raise HTTPException(status_code=400, detail=f"{field} is required")
    return value
