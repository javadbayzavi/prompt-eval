import json

import pytest

from prompt_eval import static_checks as sc
from prompt_eval.models import OutputFormat
from prompt_eval.static_checks import CheckConfig, Output

SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {"label": {"type": "string", "enum": ["billing", "tech"]}},
        "required": ["label"],
    }
)


def out(text, fmt=OutputFormat.JSON, stop="end_turn", schema=SCHEMA) -> Output:
    return Output(text=text, stop_reason=stop, output_format=fmt, output_schema=schema)


def run(name: str, output: Output, param: str = "") -> sc.Outcome:
    return sc.CHECKS[name].run(output, param)


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"label": "billing"}', 1.0),
        ('```json\n{"label": "billing"}\n```', 1.0),  # fences are fine for parsing
        ('{"label": "billing",}', 0.0),
        ("Sure! Here it is.", 0.0),
    ],
)
def test_parse_json(text, expected):
    assert run("parse", out(text)).score == expected


def test_parse_error_is_readable():
    detail = run("parse", out('{"a": }')).detail
    assert detail.startswith("Invalid JSON") and "line 1" in detail


def test_yaml_must_be_structured():
    assert run("parse", out("label: billing\nscore: 3", OutputFormat.YAML)).score == 1.0
    plain = run("parse", out("just some words", OutputFormat.YAML))
    assert plain.score == 0.0 and "plain text" in plain.detail
    assert run("parse", out("a: [1, 2", OutputFormat.YAML)).score == 0.0


def test_xml_parse_and_entity_attack_rejected():
    assert run("parse", out("<r><a>1</a></r>", OutputFormat.XML)).score == 1.0
    assert run("parse", out("<r><a>1</r>", OutputFormat.XML)).score == 0.0
    bomb = '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><x>&a;</x>'
    assert run("parse", out(bomb, OutputFormat.XML)).score == 0.0


def test_no_wrapping():
    assert run("no_wrapping", out('{"label": "tech"}')).score == 1.0
    assert run("no_wrapping", out('```json\n{"label": "tech"}\n```')).score == 0.0
    assert run("no_wrapping", out('Result: {"label": "tech"}')).score == 0.0
    assert run("no_wrapping", out("Here: <a/>", OutputFormat.XML)).score == 0.0


def test_schema():
    assert run("schema", out('{"label": "billing"}')).score == 1.0
    bad = run("schema", out('{"label": "sales", "x": 1}'))
    assert bad.score == 0.0 and "label" in bad.detail
    assert run("schema", out("label: tech", OutputFormat.YAML)).score == 1.0
    assert "not a valid JSON Schema" in run("schema", out("{}", schema="not json")).detail
    assert "no output schema" in run("schema", out("{}", schema="")).detail


def test_finished_and_not_empty():
    assert run("finished", out("x", stop="max_tokens")).score == 0.0
    assert run("finished", out("x", stop="length")).score == 0.0
    assert run("finished", out("x")).score == 1.0
    assert run("not_empty", out("   ")).score == 0.0


def test_param_checks():
    assert run("max_length", out("hello"), "5").score == 1.0
    assert run("max_length", out("hello!"), "5").score == 0.0
    assert run("regex", out("Total: 42"), r"\d+").score == 1.0
    assert run("regex", out("Total: none"), r"\d+").score == 0.0
    forbidden = run("forbidden_regex", out("As an AI model, I"), "as an ai")
    assert forbidden.score == 0.0 and "As an AI" in forbidden.detail


def test_weighted_score():
    configs = [CheckConfig("parse", 40), CheckConfig("schema", 40), CheckConfig("finished", 20)]
    result = sc.evaluate(out('{"label": "sales"}'), configs)  # parses, wrong enum, finished
    assert result.score == 60.0
    assert [c.score for c in result.checks] == [1.0, 0.0, 1.0]
    assert result.to_json()[1]["detail"].startswith("1 schema error")


def test_bad_param_does_not_crash():
    result = sc.evaluate(out("x"), [CheckConfig("max_length", 100, "abc")])
    assert result.score == 0.0
    assert "failed to run" in result.checks[0].detail


def test_validate_config():
    fmt = OutputFormat.JSON
    assert sc.validate_config([CheckConfig("parse", 60), CheckConfig("finished", 40)], fmt) == []
    problems = sc.validate_config([CheckConfig("parse", 60), CheckConfig("finished", 30)], fmt)
    assert problems == ["Weights must add up to 100% (now 90%)"]
    assert "does not apply" in sc.validate_config([CheckConfig("parse", 100)], OutputFormat.TEXT)[0]
    assert "required" in sc.validate_config([CheckConfig("regex", 100)], fmt)[0]
    assert "invalid regex" in sc.validate_config([CheckConfig("regex", 100, "(")], fmt)[0]
    assert sc.validate_config([], fmt) == ["Enable at least one check"]


@pytest.mark.parametrize("fmt", list(OutputFormat))
@pytest.mark.parametrize("has_schema", [True, False])
def test_defaults_are_valid(fmt, has_schema):
    assert sc.validate_config(sc.default_config(fmt, has_schema), fmt) == []
