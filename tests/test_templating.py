from prompt_eval.templating import find_variables, merge_variables, render


def test_find_variables_keeps_first_appearance_order():
    assert find_variables("{{b}} {{ a }}", "{{b}} {{c}}") == ["b", "a", "c"]


def test_find_variables_ignores_invalid_names():
    assert find_variables("{{1x}} {{ }} {{ok}}") == ["ok"]


def test_render_fills_known_and_keeps_unknown():
    assert render("Hi {{ name }}, {{missing}}", {"name": "Ana"}) == "Hi Ana, {{missing}}"


def test_render_stringifies_values():
    assert render("{{n}}", {"n": 3}) == "3"


def test_merge_variables():
    assert merge_variables([["a", "b"], {"b": 1, "c": 2}]) == ["a", "b", "c"]
