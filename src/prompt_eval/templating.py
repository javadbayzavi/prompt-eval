"""Minimal `{{name}}` placeholder rendering for prompt templates."""

import re
from collections.abc import Iterable, Mapping
from typing import Any

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def find_variables(*texts: str) -> list[str]:
    """Placeholder names in order of first appearance, without duplicates."""
    seen: dict[str, None] = {}
    for text in texts:
        for match in PLACEHOLDER.finditer(text or ""):
            seen.setdefault(match.group(1))
    return list(seen)


def merge_variables(groups: Iterable[Iterable[str]]) -> list[str]:
    seen: dict[str, None] = {}
    for group in groups:
        for name in group:
            seen.setdefault(name)
    return list(seen)


def render(template: str, values: Mapping[str, Any]) -> str:
    """Fill placeholders; unknown names are left as-is so missing inputs stay visible."""

    def substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            return match.group(0)
        value = values[name]
        return value if isinstance(value, str) else str(value)

    return PLACEHOLDER.sub(substitute, template)
