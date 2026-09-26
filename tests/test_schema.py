import pytest
from pydantic import ValidationError

from schema import Category, Priority, Route, TriageDecision


def _valid_kwargs():
    return {
        "category": "billing",
        "priority": "P2",
        "route": "billing-team",
        "rationale": "Duplicate charge reported by customer.",
    }


def test_accepts_valid_decision():
    decision = TriageDecision(**_valid_kwargs())
    assert decision.category is Category.BILLING
    assert decision.priority is Priority.P2
    assert decision.route is Route.BILLING_TEAM
    assert decision.rationale == "Duplicate charge reported by customer."


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("category", "foobar"),
        ("priority", "P0"),
        ("route", "growth-team"),
    ],
)
def test_rejects_unknown_enum_value(field, bad_value):
    kwargs = _valid_kwargs() | {field: bad_value}
    with pytest.raises(ValidationError) as exc:
        TriageDecision(**kwargs)
    assert field in str(exc.value)


@pytest.mark.parametrize("missing", ["category", "priority", "route", "rationale"])
def test_rejects_missing_field(missing):
    kwargs = _valid_kwargs()
    del kwargs[missing]
    with pytest.raises(ValidationError) as exc:
        TriageDecision(**kwargs)
    assert missing in str(exc.value)


@pytest.mark.parametrize("empty", ["", "   "])
def test_rejects_empty_or_whitespace_rationale(empty):
    kwargs = _valid_kwargs() | {"rationale": empty}
    with pytest.raises(ValidationError) as exc:
        TriageDecision(**kwargs)
    assert "rationale" in str(exc.value)


def test_rejects_extra_field():
    kwargs = _valid_kwargs() | {"confidence": 0.9}
    with pytest.raises(ValidationError) as exc:
        TriageDecision(**kwargs)
    assert "confidence" in str(exc.value)


def test_rationale_is_stripped():
    kwargs = _valid_kwargs() | {"rationale": "  Trimmed rationale.  "}
    decision = TriageDecision(**kwargs)
    assert decision.rationale == "Trimmed rationale."


def test_enum_members_lock_the_spec():
    assert {c.value for c in Category} == {"billing", "bug", "access", "performance", "how-to"}
    assert {p.value for p in Priority} == {"P1", "P2", "P3", "P4"}
    assert {r.value for r in Route} == {
        "billing-team",
        "bug-team",
        "access-team",
        "performance-team",
        "how-to-team",
    }
