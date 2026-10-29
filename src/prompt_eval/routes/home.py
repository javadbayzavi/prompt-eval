from fastapi import APIRouter
from fastapi.responses import RedirectResponse

router = APIRouter()


@router.get("/")
def index() -> RedirectResponse:
    return RedirectResponse("/projects")


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
