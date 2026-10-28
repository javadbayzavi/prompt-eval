{%- macro n(value) %}{{ value if value is not none else '—' }}{% endmacro -%}
# Report: {{ run.prompt.name }} v{{ run.version.version }}

Run #{{ run.id }} · {{ run.created_at.strftime('%Y-%m-%d %H:%M') }} UTC · {{ run.results | length }} results · status {{ run.status.value }}

Final score = {{ run.static_weight | round(1) }}% static{% if run.judge_engine_id %} + {{ (100 - run.static_weight) | round(1) }}% judge{% endif %}

## Scores by engine

| Engine | Model | Final | Static | Judge |{% for cat in report.categories %} {{ cat.value }} |{% endfor %} Failed |
|---|---|---|---|---|{% for cat in report.categories %}---|{% endfor %}---|
{% for e in report.engines -%}
| {{ e.name }} | `{{ e.model }}` | **{{ n(e.avg_final) }}** | {{ n(e.avg_static) }} | {{ n(e.avg_judge) }} |{% for cat in report.categories %} {{ n(e.category_avg(cat)) }} |{% endfor %} {{ e.failed }}/{{ e.total }} |
{% endfor %}
{%- if report.checks %}
## Static checks: pass rate

| Check | Weight |{% for e in report.engines %} {{ e.name }} |{% endfor %}
|---|---|{% for e in report.engines %}---|{% endfor %}
{% for check in report.checks -%}
| {{ check.label }} | {{ check.weight | round(1) }}% |{% for e in report.engines %}{% set rate = check.per_engine.get(e.name) %} {{ (rate.percent ~ '%') if rate else '—' }} |{% endfor %}
{% endfor %}
{%- endif %}
{%- if report.criteria %}
## Judge criteria: average (0–10)

| Criterion | Weight |{% for e in report.engines %} {{ e.name }} |{% endfor %}
|---|---|{% for e in report.engines %}---|{% endfor %}
{% for crit in report.criteria -%}
| {{ crit.name }} | {{ crit.weight | round(1) }}% |{% for e in report.engines %} {{ n(crit.avg_for(e.name)) }} |{% endfor %}
{% endfor %}
{%- endif %}
{%- if run.suggestions %}{% set s = run.suggestions %}
## What to change in the prompt

_Suggested by {{ run.suggestions_engine }}._

{{ s.diagnosis }}

{% for c in s.changes -%}
{{ loop.index }}. **{{ c.change }}** {{ c.reason }}{% if c.addresses %} _(Addresses: {{ c.addresses }})_{% endif %}
{% endfor %}
{%- for w in s.warnings %}
> ⚠ {{ w }}
{% endfor %}
### Suggested prompt
{% if s.revised_system_prompt %}
System:

```
{{ s.revised_system_prompt }}
```
{% endif %}
Template:

```
{{ s.revised_template }}
```
{% endif %}
## Results, worst first
{% for rr in report.results %}{% set r = rr.result %}
### {{ n(r.final_score) }} · {{ r.test_case_name }} ({{ r.category.value }}) · {{ r.engine_name }}

Static {{ n(r.static_score) }} · Judge {{ n(r.judge_score) }}{% if r.status.value != 'done' %} · **{{ r.error or r.status.value }}**{% endif %}{% if r.judge_error %} · judge failed: {{ r.judge_error }}{% endif %}

{% for c in r.static_checks or [] -%}
- {{ '✓' if c.score >= 1 else '✗' }} {{ c.label }} ({{ c.weight | round(1) }}%): {{ c.detail }}
{% endfor -%}
{% for c in r.judge_scores or [] -%}
- {{ c.name }} {{ c.score | round(1) }}/10 ({{ c.weight | round(1) }}%): {{ c.reasoning }}
{% endfor %}
{%- if r.judge_summary %}
Judge: {{ r.judge_summary }}
{% endif %}
{%- if r.judge_prompt_issues %}
Prompt issue: {{ r.judge_prompt_issues }}
{% endif %}
<details><summary>Input and output</summary>

```
{{ r.user_text }}
```

```
{{ r.output or '(none)' }}
```

</details>
{% endfor %}
