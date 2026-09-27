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
    JudgeVerdict,
    category_match,
    priority_match,
    rationale_judge,
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


# ---------------------------------------------------------------------------
# `rationale_judge` — deterministic unit tests with a fake judge model
# ---------------------------------------------------------------------------


class _FakeJudgeModel:
    """Minimal `.invoke(messages) -> JudgeVerdict` stub for the rationale judge tests.

    Monkey-patched in for `_judge_model()` so the scorer's control flow can be exercised
    without any Groq API call.
    """

    def __init__(self, verdict: str = "pass", reason: str = "matches the rubric") -> None:
        self._verdict = verdict
        self._reason = reason
        self.calls: list = []

    def invoke(self, messages):  # noqa: ANN001
        self.calls.append(messages)
        return JudgeVerdict(verdict=self._verdict, reason=self._reason)  # type: ignore[arg-type]


class _RaisingJudgeModel:
    def invoke(self, messages):  # noqa: ANN001, ARG002
        raise RuntimeError("simulated groq failure")


def test_rationale_judge_pass_returns_feedback_true(monkeypatch) -> None:
    """A `"pass"` verdict maps to `Feedback(value=True, rationale=<reason>)`."""
    from eval import run_eval

    fake = _FakeJudgeModel(verdict="pass", reason="money at stake — P2 is correct")
    monkeypatch.setattr(run_eval, "_judge_model", lambda: fake)

    fb = rationale_judge(
        outputs=_valid_payload(),
        expectations={"judge_notes": "Double charge → money problem → P2."},
    )
    assert fb.value is True
    assert fb.rationale == "money at stake — P2 is correct"
    assert len(fake.calls) == 1  # judge was actually consulted


def test_rationale_judge_fail_returns_feedback_false(monkeypatch) -> None:
    """A `"fail"` verdict maps to `Feedback(value=False, rationale=<reason>)`."""
    from eval import run_eval

    fake = _FakeJudgeModel(verdict="fail", reason="rationale contradicts judge_notes")
    monkeypatch.setattr(run_eval, "_judge_model", lambda: fake)

    fb = rationale_judge(
        outputs=_valid_payload(),
        expectations={"judge_notes": "Should have been P3."},
    )
    assert fb.value is False
    assert fb.rationale == "rationale contradicts judge_notes"


def test_rationale_judge_handles_missing_rationale_without_calling_model(monkeypatch) -> None:
    """Missing/empty agent rationale returns a canned Feedback without invoking the judge."""
    from eval import run_eval

    fake = _FakeJudgeModel()
    monkeypatch.setattr(run_eval, "_judge_model", lambda: fake)

    payload = _valid_payload()
    del payload["rationale"]

    fb = rationale_judge(
        outputs=payload,
        expectations={"judge_notes": "anything"},
    )
    assert fb.value is False
    assert fb.rationale == "no rationale to judge"
    assert fake.calls == []  # judge model was NOT consulted


def test_rationale_judge_handles_empty_rationale_string(monkeypatch) -> None:
    """An empty-string `rationale` is also "no rationale to judge"."""
    from eval import run_eval

    fake = _FakeJudgeModel()
    monkeypatch.setattr(run_eval, "_judge_model", lambda: fake)

    payload = _valid_payload()
    payload["rationale"] = ""

    fb = rationale_judge(outputs=payload, expectations={"judge_notes": "x"})
    assert fb.value is False
    assert fb.rationale == "no rationale to judge"
    assert fake.calls == []


def test_rationale_judge_wraps_model_error_as_feedback_false(monkeypatch) -> None:
    """A model call error returns Feedback(value=False, rationale='judge error: …')."""
    from eval import run_eval

    monkeypatch.setattr(run_eval, "_judge_model", lambda: _RaisingJudgeModel())

    fb = rationale_judge(
        outputs=_valid_payload(),
        expectations={"judge_notes": "anything"},
    )
    assert fb.value is False
    assert fb.rationale.startswith("judge error: ")
    assert "simulated groq failure" in fb.rationale


def test_rationale_judge_non_dict_outputs_return_no_rationale(monkeypatch) -> None:
    """Totally-wrong `outputs` shape falls through to the missing-rationale branch."""
    from eval import run_eval

    fake = _FakeJudgeModel()
    monkeypatch.setattr(run_eval, "_judge_model", lambda: fake)

    fb = rationale_judge(outputs=None, expectations={"judge_notes": "x"})
    assert fb.value is False
    assert fb.rationale == "no rationale to judge"


# ---------------------------------------------------------------------------
# `_agent_total_tokens` — synthetic traces
# ---------------------------------------------------------------------------


@dataclass
class _StubSpanForToken:
    """Minimal Span shape for `_agent_total_tokens` — needs `parent_id` and `span_type`."""

    parent_id: str | None
    span_type: str
    name: str = "root"


@dataclass
class _StubTraceInfo:
    token_usage: dict[str, int] | None


@dataclass
class _StubTraceData:
    spans: list


@dataclass
class _StubTraceForToken:
    """Stub of `mlflow.entities.Trace` with `.info.token_usage` and `.data.spans`."""

    info: _StubTraceInfo
    data: _StubTraceData


def _make_trace(root_span_type: str, total_tokens: int | None) -> _StubTraceForToken:
    root = _StubSpanForToken(parent_id=None, span_type=root_span_type, name="predict")
    child = _StubSpanForToken(parent_id="root", span_type="TOOL", name="get_ticket")
    usage = None if total_tokens is None else {"total_tokens": total_tokens}
    return _StubTraceForToken(
        info=_StubTraceInfo(token_usage=usage),
        data=_StubTraceData(spans=[root, child]),
    )


def test_agent_total_tokens_sums_only_agent_rooted_traces(monkeypatch) -> None:
    """AGENT-rooted traces contribute; non-AGENT (e.g. judge CHAT_MODEL) traces do not."""
    from eval import run_eval

    traces = [
        _make_trace("AGENT", 100),
        _make_trace("AGENT", 250),
        _make_trace("CHAT_MODEL", 500),  # judge trace — must be excluded
    ]
    monkeypatch.setattr(
        run_eval.mlflow,
        "search_traces",
        lambda run_id, return_type: traces,
    )

    assert run_eval._agent_total_tokens("abc") == 350


def test_agent_total_tokens_tolerates_missing_usage(monkeypatch) -> None:
    """Traces whose `token_usage` is None (errored before autolog) contribute 0."""
    from eval import run_eval

    traces = [
        _make_trace("AGENT", 100),
        _make_trace("AGENT", None),  # no usage recorded
        _make_trace("AGENT", 50),
    ]
    monkeypatch.setattr(
        run_eval.mlflow,
        "search_traces",
        lambda run_id, return_type: traces,
    )

    assert run_eval._agent_total_tokens("abc") == 150


def test_agent_total_tokens_returns_zero_when_no_agent_rooted_traces(monkeypatch) -> None:
    from eval import run_eval

    traces = [
        _make_trace("CHAT_MODEL", 500),
        _make_trace("CHAT_MODEL", 750),
    ]
    monkeypatch.setattr(
        run_eval.mlflow,
        "search_traces",
        lambda run_id, return_type: traces,
    )

    assert run_eval._agent_total_tokens("abc") == 0


def test_agent_total_tokens_falls_back_to_input_plus_output_when_total_missing(monkeypatch) -> None:
    """Some providers autolog `input_tokens` + `output_tokens` without `total_tokens`.

    Falling back to the sum avoids silently reporting 0 when the underlying provider
    just uses different field names.
    """
    from eval import run_eval

    trace = _StubTraceForToken(
        info=_StubTraceInfo(token_usage={"input_tokens": 10, "output_tokens": 5}),  # no total
        data=_StubTraceData(spans=[_StubSpanForToken(parent_id=None, span_type="AGENT")]),
    )
    monkeypatch.setattr(
        run_eval.mlflow,
        "search_traces",
        lambda run_id, return_type: [trace],
    )

    assert run_eval._agent_total_tokens("abc") == 15


def test_agent_total_tokens_returns_zero_when_all_keys_missing(monkeypatch) -> None:
    """An empty `token_usage` dict contributes 0 (no components to sum)."""
    from eval import run_eval

    trace = _StubTraceForToken(
        info=_StubTraceInfo(token_usage={}),
        data=_StubTraceData(spans=[_StubSpanForToken(parent_id=None, span_type="AGENT")]),
    )
    monkeypatch.setattr(
        run_eval.mlflow,
        "search_traces",
        lambda run_id, return_type: [trace],
    )

    assert run_eval._agent_total_tokens("abc") == 0


# ---------------------------------------------------------------------------
# `_write_report`
# ---------------------------------------------------------------------------


def test_write_report_produces_pretty_json_with_trailing_newline(tmp_path) -> None:
    from eval.run_eval import _write_report

    path = tmp_path / "latest_report.json"
    report = {
        "run_id": "abc123",
        "scorer_means": {
            "valid_schema": 1.0,
            "category_match": 0.9,
            "priority_match": 0.85,
            "tool_order": 1.0,
            "rationale_judge": 0.75,
        },
        "agent_total_tokens": 12345,
        "escalation_count": 3,
    }

    _write_report(path, report)

    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n"), "Report file must end with a trailing newline."
    # Round-trips as JSON.
    import json as _json

    loaded = _json.loads(text)
    assert loaded == report
    # Pretty-printed (indent=2) → contains newlines and 2-space indent.
    assert "\n  " in text


def test_write_report_overwrites_existing_file(tmp_path) -> None:
    from eval.run_eval import _write_report

    path = tmp_path / "latest_report.json"
    path.write_text('{"stale": true}\n', encoding="utf-8")

    _write_report(path, {"fresh": True})

    import json as _json

    assert _json.loads(path.read_text(encoding="utf-8")) == {"fresh": True}


# ---------------------------------------------------------------------------
# Source-level guard: rationale_judge must NEVER read GEMINI_API_KEY
# ---------------------------------------------------------------------------


def test_run_eval_source_does_not_reference_gemini_api_key() -> None:
    """Story 3.2 boundary: the judge model must never touch `GEMINI_API_KEY`.

    Guard the boundary structurally — a grep on the module's source is the cheapest
    way to catch a future refactor that copy-pastes `os.environ["GEMINI_API_KEY"]`
    into the judge construction path.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "eval" / "run_eval.py").read_text(
        encoding="utf-8"
    )
    assert "GEMINI_API_KEY" not in src, (
        "eval/run_eval.py mentions GEMINI_API_KEY — the rationale_judge scorer must "
        "only ever read GROQ_API_KEY / JUDGE_MODEL. Remove the reference."
    )


# ---------------------------------------------------------------------------
# `_scorers()` registration guard
# ---------------------------------------------------------------------------


def test_scorers_registration_includes_all_five() -> None:
    """`main()` wires up exactly the five scorers Story 3.2 requires."""
    from eval.run_eval import _scorers

    names = {getattr(s, "name", getattr(s, "__name__", None)) for s in _scorers()}
    # `@scorer` wraps the function but preserves the underlying name via `.name` or
    # via the wrapped `__name__`; accept either.
    assert names >= {
        "valid_schema",
        "category_match",
        "priority_match",
        "tool_order",
        "rationale_judge",
    }, f"Missing scorers in registration: {names}"


# ---------------------------------------------------------------------------
# `main()` fail-fast on missing GROQ_API_KEY
# ---------------------------------------------------------------------------


def test_main_exits_when_groq_api_key_missing(monkeypatch) -> None:
    """`main()` must SystemExit before touching MLflow / evaluate when GROQ_API_KEY is unset.

    The judge always runs, no matter what `PROVIDER` is, so surfacing the config gap
    upfront prevents a half-scored run + a stale report on disk.
    """
    import pytest as _pytest

    from eval import run_eval

    # Neutralize load_dotenv so a developer's .env file doesn't accidentally provide
    # the key and mask the failure.
    monkeypatch.setattr(run_eval, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    # If evaluate is reached, the test failed — trip a loud error so the failure mode
    # is unambiguous.
    def _should_not_run(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("mlflow.genai.evaluate was called despite missing GROQ_API_KEY.")

    monkeypatch.setattr(run_eval.mlflow.genai, "evaluate", _should_not_run)

    with _pytest.raises(SystemExit) as excinfo:
        run_eval.main()

    assert "GROQ_API_KEY" in str(excinfo.value)
    # A `SystemExit("msg")` defaults to code 1, but the assertion above alone would
    # pass on `SystemExit(0)` — pin the non-zero exit so the failure actually surfaces
    # to the shell.
    assert excinfo.value.code not in (0, None), (
        f"Expected a non-zero SystemExit code; got {excinfo.value.code!r}."
    )
