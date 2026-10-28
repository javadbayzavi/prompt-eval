"""Static (rule-based) checks on a model's output, each with its own weight.

A prompt's static score is the weighted sum of its enabled checks: weights are percentages of
the static score and must add up to 100.
"""

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import yaml
from defusedxml import ElementTree
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from prompt_eval.models import OutputFormat

STRUCTURED = (OutputFormat.JSON, OutputFormat.YAML, OutputFormat.XML)
TRUNCATED_STOP_REASONS = ("max_tokens", "length")
FENCE = re.compile(r"^\s*```[\w-]*\s*\n(.*?)\n?```\s*$", re.DOTALL)


@dataclass
class Output:
    text: str
    stop_reason: str | None
    output_format: OutputFormat
    output_schema: str = ""


@dataclass
class Outcome:
    score: float  # 0.0 to 1.0
    detail: str


@dataclass
class CheckSpec:
    label: str
    help: str
    formats: tuple[OutputFormat, ...]
    run: Callable[[Output, str], Outcome]
    param_label: str = ""  # non-empty when the check takes a parameter


@dataclass
class CheckConfig:
    name: str
    weight: float  # percent of the static score
    param: str = ""


@dataclass
class CheckResult:
    name: str
    label: str
    weight: float
    score: float
    detail: str


@dataclass
class StaticResult:
    score: float  # 0 to 100
    checks: list[CheckResult] = field(default_factory=list)

    def to_json(self) -> list[dict]:
        return [asdict(c) for c in self.checks]


def strip_fences(text: str) -> str:
    match = FENCE.match(text)
    return match.group(1) if match else text.strip()


def parse_structured(output: Output) -> object:
    """Parse after removing code fences; raises ValueError with a readable message."""
    body = strip_fences(output.text)
    if output.output_format == OutputFormat.JSON:
        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e.msg} (line {e.lineno}, column {e.colno})") from e
    if output.output_format == OutputFormat.YAML:
        try:
            data = yaml.safe_load(body)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {str(e).splitlines()[0]}") from e
        # Any prose is a valid YAML string, so a bare scalar means "not structured".
        if not isinstance(data, dict | list):
            raise ValueError("Parsed as plain text, not a YAML mapping or list")
        return data
    if output.output_format == OutputFormat.XML:
        try:
            return ElementTree.fromstring(body)
        except Exception as e:  # ParseError and defusedxml's security errors
            raise ValueError(f"Invalid XML: {e}") from e
    raise ValueError(f"{output.output_format.value} has no parser")


def check_not_empty(output: Output, _param: str) -> Outcome:
    if output.text.strip():
        return Outcome(1.0, "Output is not empty")
    return Outcome(0.0, "Output is empty")


def check_finished(output: Output, _param: str) -> Outcome:
    if output.stop_reason in TRUNCATED_STOP_REASONS:
        return Outcome(0.0, "Output was cut off at the max token limit")
    return Outcome(1.0, "Output finished normally")


def check_parse(output: Output, _param: str) -> Outcome:
    try:
        parse_structured(output)
    except ValueError as e:
        return Outcome(0.0, str(e))
    return Outcome(1.0, f"Parses as {output.output_format.value.upper()}")


def check_no_wrapping(output: Output, _param: str) -> Outcome:
    text = output.text.strip()
    if FENCE.match(text):
        return Outcome(0.0, "Wrapped in a markdown code fence")
    if output.output_format == OutputFormat.JSON:
        try:
            json.loads(text)
        except ValueError:
            return Outcome(0.0, "Has text before or after the JSON")
    elif output.output_format == OutputFormat.XML and not (
        text.startswith("<") and text.endswith(">")
    ):
        return Outcome(0.0, "Has text before or after the XML")
    return Outcome(1.0, "Output is only the payload")


def check_schema(output: Output, _param: str) -> Outcome:
    if not output.output_schema.strip():
        return Outcome(0.0, "The prompt has no output schema")
    try:
        schema = json.loads(output.output_schema)
        validator = Draft202012Validator(schema)
        validator.check_schema(schema)
    except (ValueError, SchemaError) as e:
        return Outcome(0.0, f"The prompt's output schema is not a valid JSON Schema: {e}")
    try:
        data = parse_structured(output)
    except ValueError as e:
        return Outcome(0.0, f"Can't validate: {e}")
    errors = sorted(validator.iter_errors(data), key=lambda err: list(err.path))
    if not errors:
        return Outcome(1.0, "Matches the output schema")
    shown = "; ".join(
        f"{'/'.join(map(str, err.path)) or '(root)'}: {err.message}" for err in errors[:3]
    )
    more = f" (+{len(errors) - 3} more)" if len(errors) > 3 else ""
    return Outcome(0.0, f"{len(errors)} schema error(s): {shown}{more}")


def check_max_length(output: Output, param: str) -> Outcome:
    limit = int(param)
    length = len(output.text)
    if length <= limit:
        return Outcome(1.0, f"{length} characters (limit {limit})")
    return Outcome(0.0, f"{length} characters, over the limit of {limit}")


def check_regex(output: Output, param: str) -> Outcome:
    if re.search(param, output.text, re.MULTILINE):
        return Outcome(1.0, f"Matches /{param}/")
    return Outcome(0.0, f"Does not match /{param}/")


def check_forbidden_regex(output: Output, param: str) -> Outcome:
    if match := re.search(param, output.text, re.MULTILINE | re.IGNORECASE):
        return Outcome(0.0, f"Contains forbidden text: {match.group(0)!r}")
    return Outcome(1.0, f"Does not contain /{param}/")


ALL_FORMATS = tuple(OutputFormat)

CHECKS: dict[str, CheckSpec] = {
    "not_empty": CheckSpec("Not empty", "The output has some text.", ALL_FORMATS, check_not_empty),
    "finished": CheckSpec(
        "Not cut off",
        "The model stopped on its own, not at the max token limit.",
        ALL_FORMATS,
        check_finished,
    ),
    "parse": CheckSpec(
        "Parses",
        "The output parses in the expected format (code fences allowed).",
        STRUCTURED,
        check_parse,
    ),
    "no_wrapping": CheckSpec(
        "No extra text",
        "Only the payload: no code fences or prose around it.",
        STRUCTURED,
        check_no_wrapping,
    ),
    "schema": CheckSpec(
        "Matches schema",
        "The parsed output validates against the prompt's JSON Schema.",
        (OutputFormat.JSON, OutputFormat.YAML),
        check_schema,
    ),
    "max_length": CheckSpec(
        "Max length",
        "At most this many characters.",
        ALL_FORMATS,
        check_max_length,
        param_label="Max characters",
    ),
    "regex": CheckSpec(
        "Must match",
        "The output matches this regular expression.",
        ALL_FORMATS,
        check_regex,
        param_label="Regex",
    ),
    "forbidden_regex": CheckSpec(
        "Must not match",
        "The output does not contain this (case-insensitive regex), e.g. 'as an AI'.",
        ALL_FORMATS,
        check_forbidden_regex,
        param_label="Regex",
    ),
}


def checks_for(output_format: OutputFormat) -> dict[str, CheckSpec]:
    return {name: spec for name, spec in CHECKS.items() if output_format in spec.formats}


def default_config(output_format: OutputFormat, has_schema: bool) -> list[CheckConfig]:
    if output_format in (OutputFormat.JSON, OutputFormat.YAML):
        if has_schema:
            return [
                CheckConfig("parse", 40),
                CheckConfig("schema", 40),
                CheckConfig("no_wrapping", 10),
                CheckConfig("finished", 10),
            ]
        return [
            CheckConfig("parse", 70),
            CheckConfig("no_wrapping", 15),
            CheckConfig("finished", 15),
        ]
    if output_format == OutputFormat.XML:
        return [
            CheckConfig("parse", 70),
            CheckConfig("no_wrapping", 15),
            CheckConfig("finished", 15),
        ]
    return [CheckConfig("not_empty", 50), CheckConfig("finished", 50)]


def validate_config(configs: list[CheckConfig], output_format: OutputFormat) -> list[str]:
    """Problems that would make the config unusable; empty when it's fine."""
    problems = []
    available = checks_for(output_format)
    for config in configs:
        spec = available.get(config.name)
        if spec is None:
            problems.append(f"{config.name!r} does not apply to {output_format.value} output")
            continue
        if config.weight <= 0:
            problems.append(f"{spec.label}: weight must be above 0")
        if spec.param_label and not config.param.strip():
            problems.append(f"{spec.label}: {spec.param_label.lower()} is required")
        param = config.param.strip()
        if config.name == "max_length" and param and not param.isdigit():
            problems.append("Max length: must be a whole number")
        if config.name in ("regex", "forbidden_regex") and config.param:
            try:
                re.compile(config.param)
            except re.error as e:
                problems.append(f"{spec.label}: invalid regex ({e})")
    if configs and abs(sum(c.weight for c in configs) - 100) > 0.01:
        total = sum(c.weight for c in configs)
        problems.append(f"Weights must add up to 100% (now {total:g}%)")
    if not configs:
        problems.append("Enable at least one check")
    return problems


def evaluate(output: Output, configs: list[CheckConfig]) -> StaticResult:
    checks = []
    for config in configs:
        spec = CHECKS[config.name]
        try:
            outcome = spec.run(output, config.param)
        except Exception as e:  # a bad parameter should not break the run
            outcome = Outcome(0.0, f"Check failed to run: {e}")
        checks.append(
            CheckResult(config.name, spec.label, config.weight, outcome.score, outcome.detail)
        )
    total_weight = sum(c.weight for c in checks) or 1
    score = sum(c.weight * c.score for c in checks) / total_weight * 100
    return StaticResult(score=round(score, 1), checks=checks)


def configs_from_json(data: list[dict] | None) -> list[CheckConfig]:
    return [CheckConfig(**item) for item in data or []]


def configs_to_json(configs: list[CheckConfig]) -> list[dict]:
    return [asdict(c) for c in configs]


def has_json_schema(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except ValueError:
        return False


def prompt_config(prompt) -> list[CheckConfig]:
    """The prompt's saved checks, or the defaults for its output format."""
    if prompt.static_checks is not None:
        return configs_from_json(prompt.static_checks)
    return default_config(prompt.output_format, has_json_schema(prompt.output_schema))


def score_result(result, run) -> None:
    """Fill a finished result's static score from its run's scoring snapshot."""
    if run.static_checks is None:
        return  # a run from before scoring existed; "Rescore" adds a snapshot
    outcome = evaluate(
        Output(
            text=result.output,
            stop_reason=result.stop_reason,
            output_format=run.output_format,
            output_schema=run.output_schema,
        ),
        configs_from_json(run.static_checks),
    )
    result.static_score = outcome.score
    result.static_checks = outcome.to_json()
