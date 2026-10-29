from typing import Annotated, Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.datastructures import FormData

from prompt_eval.models import Prompt, TestCase, TestCaseCategory
from prompt_eval.templating import find_variables, merge_variables
from prompt_eval.web import SessionDep, get_or_404, redirect, require_text, templates

router = APIRouter()

# Each template variable is posted as its own field, e.g. `input__topic`.
INPUT_PREFIX = "input__"


def inputs_from_form(form: FormData) -> dict[str, Any]:
    return {
        key.removeprefix(INPUT_PREFIX): str(value)
        for key, value in form.multi_items()
        if key.startswith(INPUT_PREFIX)
    }


@router.post("/prompts/{prompt_id}/test-cases")
async def create_test_case(
    request: Request,
    session: SessionDep,
    prompt_id: int,
    name: Annotated[str, Form()],
    category: Annotated[TestCaseCategory, Form()] = TestCaseCategory.NORMAL,
    expected_output: Annotated[str, Form()] = "",
    criteria: Annotated[str, Form()] = "",
) -> RedirectResponse:
    prompt = get_or_404(session, Prompt, prompt_id)
    test_case = TestCase(
        prompt=prompt,
        name=require_text(name, "Name"),
        inputs=inputs_from_form(await request.form()),
        category=category,
        expected_output=expected_output.strip(),
        criteria=criteria.strip(),
    )
    session.add(test_case)
    session.commit()
    return redirect(f"/prompts/{prompt.id}/test-cases")


@router.get("/test-cases/{test_case_id}/edit", response_class=HTMLResponse)
def edit_test_case(request: Request, session: SessionDep, test_case_id: int) -> HTMLResponse:
    test_case = get_or_404(session, TestCase, test_case_id)
    latest = test_case.prompt.latest_version
    current = find_variables(latest.system_prompt, latest.template) if latest else []
    return templates.TemplateResponse(
        request,
        "test_cases/edit.html",
        {
            "prompt": test_case.prompt,
            "tab": "test-cases",
            "test_case": test_case,
            "variables": merge_variables([current, test_case.inputs]),
            "categories": list(TestCaseCategory),
        },
    )


@router.post("/test-cases/{test_case_id}")
async def update_test_case(
    request: Request,
    session: SessionDep,
    test_case_id: int,
    name: Annotated[str, Form()],
    category: Annotated[TestCaseCategory, Form()] = TestCaseCategory.NORMAL,
    expected_output: Annotated[str, Form()] = "",
    criteria: Annotated[str, Form()] = "",
) -> RedirectResponse:
    test_case = get_or_404(session, TestCase, test_case_id)
    test_case.name = require_text(name, "Name")
    test_case.inputs = inputs_from_form(await request.form())
    test_case.category = category
    test_case.expected_output = expected_output.strip()
    test_case.criteria = criteria.strip()
    session.commit()
    return redirect(f"/prompts/{test_case.prompt_id}/test-cases")


@router.post("/test-cases/{test_case_id}/delete")
def delete_test_case(session: SessionDep, test_case_id: int) -> RedirectResponse:
    test_case = get_or_404(session, TestCase, test_case_id)
    prompt_id = test_case.prompt_id
    session.delete(test_case)
    session.commit()
    return redirect(f"/prompts/{prompt_id}/test-cases")
