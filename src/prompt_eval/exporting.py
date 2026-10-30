"""Export a project, or one prompt, as the same `prompt-eval/1` file `importing` reads: prompts,
their versions, scoring settings and test cases, but no runs. Importing an exported file gives
back the same prompts.
"""

import json
import re
from typing import Any

from sqlmodel import Session

from prompt_eval.importing import FORMAT
from prompt_eval.models import Engine, Project, Prompt

_NOT_SLUG = re.compile(r"[^a-z0-9]+")


def project_file(session: Session, project: Project) -> dict[str, Any]:
    return _file(project, [prompt_dict(session, prompt) for prompt in project.prompts])


def prompt_file(session: Session, prompt: Prompt) -> dict[str, Any]:
    """One prompt, under its project's name, so it can be imported on its own anywhere."""
    return _file(prompt.project, [prompt_dict(session, prompt)])


def prompt_dict(session: Session, prompt: Prompt) -> dict[str, Any]:
    judge_engine = session.get(Engine, prompt.judge_engine_id) if prompt.judge_engine_id else None
    return {
        "name": prompt.name,
        "task_description": prompt.task_description,
        "output_format": prompt.output_format.value,
        "output_schema": _schema(prompt.output_schema),
        "static_checks": prompt.static_checks,
        "static_weight": prompt.static_weight,
        "judge_engine": judge_engine.name if judge_engine else None,
        "judge_criteria": prompt.judge_criteria,
        "judge_instructions": prompt.judge_instructions,
        "versions": [
            {"system_prompt": v.system_prompt, "template": v.template, "notes": v.notes}
            for v in prompt.versions
        ],
        "test_cases": [
            {
                "name": case.name,
                "category": case.category.value,
                "inputs": {name: str(value) for name, value in case.inputs.items()},
                "criteria": case.criteria,
                "expected_output": case.expected_output,
            }
            for case in prompt.test_cases
        ],
    }


def file_name(name: str) -> str:
    return f"{_NOT_SLUG.sub('-', name.lower()).strip('-') or 'export'}.json"


def _file(project: Project, prompts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "project": {"name": project.name, "description": project.description},
        "prompts": prompts,
    }


def _schema(text: str) -> object:
    # A JSON Schema goes out as an object, anything else as the text it is.
    try:
        value = json.loads(text)
    except ValueError:
        return text
    return value if isinstance(value, dict) else text
