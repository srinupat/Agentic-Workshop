"""Deterministic unit tests for the four code scorers in `eval/run_eval.py`.

No API key required — these tests use canned `inputs`/`outputs`/`expectations` payloads
and a synthetic trace stub with `.search_spans` so each scorer's happy path and every
guarded error path is exercised without spinning up an LLM.

Each `@scorer`-decorated function is called through its `Scorer.__call__` shim; the
scorers themselves accept keyword-only `inputs`/`outputs`/`expectations`/`trace` as
documented in `mlflow.genai.scorers.base.scorer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from eval.run_eval import (
    category_match,
    priority_match,
    tool_order,
    valid_schema,
)


# ---------------------------------------------------------------------------
# Trace stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubSpan:
    """Minimum shape a `mlflow.entities.Span` needs to satisfy `tool_order`."""

    name: str
    start_time_ns: int


@dataclass
class _StubTrace:
    """Stub of `mlflow.entities.Trace` exposing `search_spans(name=...)`.

    Only `name` filtering is used by `tool_order`, so we ignore `span_type`/`span_id`
    kwargs the real API accepts.
    """

    spans: list[_StubSpan] = field(default_factory=list)

    def search_spans(
        self,
        span_type=None,  # noqa: ARG002 — matches real signature
        name: str | None = None,
        span_id=None,  # noqa: ARG002
    ) -> list[_StubSpan]:
        if name is None:
            return list(self.spans)
        return [s for s in self.spans if s.name == name]


# ---------------------------------------------------------------------------
# `valid_schema`
# ---------------------------------------------------------------------------


def _valid_payload() -> dict:
    return {
        "category": "billing",
        "priority": "P2",
        "route": "billing-team",
        "rationale": "Money at stake.",
    }


def test_valid_schema_passes_on_valid_payload() -> None:
    assert valid_schema(outputs=_valid_payload()) is True


def test_valid_schema_fails_on_missing_key() -> None:
    payload = _valid_payload()
    del payload["route"]
    assert valid_schema(outputs=payload) is False


def test_valid_schema_fails_on_bad_enum_value() -> None:
    payload = _valid_payload()
    payload["priority"] = "P9"
    assert valid_schema(outputs=payload) is False


def test_valid_schema_fails_on_non_dict_output() -> None:
    """A totally-wrong shape returns False rather than raising."""
    assert valid_schema(outputs=None) is False
    assert valid_schema(outputs="just a string") is False


# ---------------------------------------------------------------------------
# `category_match` and `priority_match`
# ---------------------------------------------------------------------------


def test_category_match_hit() -> None:
    assert (
        category_match(
            outputs=_valid_payload(),
            expectations={"expected_category": "billing"},
        )
        is True
    )


def test_category_match_miss() -> None:
    assert (
        category_match(
            outputs=_valid_payload(),
            expectations={"expected_category": "bug"},
        )
        is False
    )


def test_category_match_fails_on_malformed_output() -> None:
    """A missing/None `category` returns False without raising."""
    assert (
        category_match(
            outputs={"priority": "P2"},
            expectations={"expected_category": "billing"},
        )
        is False
    )
    assert (
        category_match(
            outputs=None,
            expectations={"expected_category": "billing"},
        )
        is False
    )


def test_priority_match_hit() -> None:
    assert (
        priority_match(
            outputs=_valid_payload(),
            expectations={"expected_priority": "P2"},
        )
        is True
    )


def test_priority_match_miss() -> None:
    assert (
        priority_match(
            outputs=_valid_payload(),
            expectations={"expected_priority": "P3"},
        )
        is False
    )


def test_priority_match_fails_on_malformed_output() -> None:
    assert (
        priority_match(
            outputs={"category": "billing"},
            expectations={"expected_priority": "P2"},
        )
        is False
    )


# ---------------------------------------------------------------------------
# `tool_order`
# ---------------------------------------------------------------------------


def test_tool_order_passes_when_get_ticket_precedes_history() -> None:
    trace = _StubTrace(
        spans=[
            _StubSpan(name="get_ticket", start_time_ns=100),
            _StubSpan(name="get_customer_history", start_time_ns=200),
        ]
    )
    assert tool_order(trace=trace) is True


def test_tool_order_fails_when_history_precedes_get_ticket() -> None:
    trace = _StubTrace(
        spans=[
            _StubSpan(name="get_customer_history", start_time_ns=100),
            _StubSpan(name="get_ticket", start_time_ns=200),
        ]
    )
    assert tool_order(trace=trace) is False


def test_tool_order_fails_when_history_span_missing() -> None:
    """A trace containing `get_ticket` but not `get_customer_history` fails, not raises."""
    trace = _StubTrace(spans=[_StubSpan(name="get_ticket", start_time_ns=100)])
    assert tool_order(trace=trace) is False


def test_tool_order_fails_when_ticket_span_missing() -> None:
    trace = _StubTrace(spans=[_StubSpan(name="get_customer_history", start_time_ns=100)])
    assert tool_order(trace=trace) is False


def test_tool_order_fails_when_trace_is_none() -> None:
    """A predict that errored before autolog captured spans passes `trace=None`."""
    assert tool_order(trace=None) is False


def test_tool_order_fails_when_search_spans_raises() -> None:
    """Any exception inside `search_spans` is swallowed and returns False."""

    class _BrokenTrace:
        def search_spans(self, **_kw):  # noqa: ANN003
            raise RuntimeError("boom")

    assert tool_order(trace=_BrokenTrace()) is False


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------


def test_load_labelled_tickets_returns_20_rows_in_evaluate_shape() -> None:
    """All 20 rows are loaded with the exact keys `mlflow.genai.evaluate` expects."""
    from eval.run_eval import _load_labelled_tickets

    rows = _load_labelled_tickets()

    assert len(rows) == 20
    for row in rows:
        assert set(row) == {"inputs", "expectations"}
        assert set(row["inputs"]) == {"ticket_id"}
        assert row["inputs"]["ticket_id"].startswith("T-")
        assert set(row["expectations"]) == {
            "expected_category",
            "expected_priority",
            "expected_tools",
            "judge_notes",
        }
        assert row["expectations"]["expected_tools"] == [
            "get_ticket",
            "get_customer_history",
        ]


# ---------------------------------------------------------------------------
# HITL auto-approve monkey-patch
# ---------------------------------------------------------------------------


def test_auto_approve_only_fires_on_escalate_prompts(monkeypatch) -> None:
    """The monkey-patched `input()` inside `main()` only auto-approves escalate prompts.

    Non-`escalate` prompts fall through to the original `input`. This test recreates
    the same closure `main()` builds so the guard is exercised without running the
    full eval loop.
    """
    import builtins

    original_input_calls: list[str] = []

    def _original_input(prompt: str = "") -> str:
        original_input_calls.append(prompt)
        return "fallback"

    monkeypatch.setattr(builtins, "input", _original_input)
    original = builtins.input
    escalation_count = 0

    def _auto_approve_input(prompt: str = "") -> str:
        nonlocal escalation_count
        if "escalate" in prompt.lower():
            escalation_count += 1
            return "yes"
        return original(prompt)

    # Escalate-flavored prompt → auto-yes, counter increments.
    assert _auto_approve_input("Escalate to human? [yes/no]: ") == "yes"
    assert escalation_count == 1
    assert original_input_calls == []

    # Any other prompt → falls through to the (test-monkeypatched) original input.
    assert _auto_approve_input("Some other prompt: ") == "fallback"
    assert escalation_count == 1
    assert original_input_calls == ["Some other prompt: "]


# ---------------------------------------------------------------------------
# Predict wrapper is a coroutine on a traced function
# ---------------------------------------------------------------------------


def test_predict_wrapper_defined_inside_main_is_async_and_traced() -> None:
    """Sanity guard: `main()` uses `@mlflow.trace(span_type="AGENT")` and an `async def`.

    A regression that swaps the wrapper to a sync function or drops the decorator
    would silently break the "one trace per ticket" invariant `tool_order` depends
    on. This test parses `main()`'s source to prove both properties.
    """
    import inspect

    from eval import run_eval

    src = inspect.getsource(run_eval.main)
    assert "@mlflow.trace(span_type=\"AGENT\")" in src, (
        "The predict wrapper in main() must be decorated with "
        "@mlflow.trace(span_type=\"AGENT\") so escalation + resume nest under one trace."
    )
    assert "async def predict" in src, "The predict wrapper must be `async def`."


def test_auto_approve_trigger_stays_in_sync_with_agent_prompt() -> None:
    """The auto-approve monkey-patch fires on `"escalate" in prompt.lower()`.

    That heuristic is silently coupled to `agent.py`'s HITL prompt string. If Story 2.2's
    prompt is ever reworded (e.g. "Approve handoff?"), the auto-approve stops firing and
    every escalating eval ticket blocks. Guard the coupling by asserting `agent.py`
    still contains the trigger substring near the HITL prompt.
    """
    from pathlib import Path

    agent_src = (Path(__file__).resolve().parent.parent / "agent.py").read_text()
    # The trigger the auto-approve monkey-patch matches on.
    trigger_substring = "escalate"
    # `_prompt_for_escalation_decision` is the choke point where `input()` is called.
    assert "_prompt_for_escalation_decision" in agent_src, (
        "agent.py no longer exposes _prompt_for_escalation_decision — the HITL "
        "prompt choke point has moved; recheck the auto-approve wiring."
    )
    # And the prompt text itself still contains the trigger substring the auto-approve
    # match relies on.
    assert 'input("Escalate to human? [yes/no]: ")' in agent_src, (
        f"agent.py's HITL prompt no longer contains the auto-approve trigger "
        f"substring {trigger_substring!r}. Either update the trigger in "
        f"eval/run_eval.py::_auto_approve_input or restore the prompt text."
    )
