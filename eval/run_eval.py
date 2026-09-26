"""Run the triage agent against the 20 labelled tickets and score every run.

Public entry point: `main()`. Invoked via `uv run python eval/run_eval.py`.

Flow:
1. Mirror `run_agent.py`'s MLflow setup: tracking URI `sqlite:///mlflow.db`, experiment
   `triage-agent`, LangChain autologging enabled.
2. Load `eval/labelled_tickets.csv` into the `{"inputs": {...}, "expectations": {...}}`
   shape that `mlflow.genai.evaluate` expects.
3. Monkey-patch `builtins.input` (scope-local, restored in `finally`) so any HITL
   escalation prompt fired from `agent.triage()` during the eval auto-approves with
   `"yes"`. A closure counter tracks how many escalations were auto-approved so we can
   print the total at the end. `run_agent.py`'s interactive prompt is unchanged — this
   patch only affects the eval process.
4. Wrap `agent.triage()` in an `async def predict(ticket_id)` that is decorated with
   `@mlflow.trace(span_type="AGENT")`, so the initial `agent.ainvoke` AND the
   HITL-resume `ainvoke` land under one trace per ticket. The `tool_order` scorer
   relies on both `get_ticket` and `get_customer_history` spans being in the same
   trace.
5. Call `mlflow.genai.evaluate(data=..., predict_fn=predict, scorers=[...])` with the
   four code scorers defined in this module.
6. Print the four per-scorer means (`{scorer}/mean` in `results.metrics`), the
   auto-approved escalation count, and the MLflow run_id.

Story 3.2 will add the `rationale_judge` LLM scorer and write `eval/latest_report.json`.
This file deliberately stops at the four code scorers + counter + printout.
"""

from __future__ import annotations

import builtins
import csv
import sys
from pathlib import Path
from typing import Any

import mlflow
from dotenv import load_dotenv
from mlflow.genai.scorers import scorer

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LABELLED_TICKETS_CSV = _REPO_ROOT / "eval" / "labelled_tickets.csv"

# `agent.triage` is a coroutine, so `mlflow.genai.evaluate` runs it as an async predict
# function directly (evaluate detects `iscoroutinefunction` and awaits internally).
# CLI invocation (`uv run python eval/run_eval.py`) puts `eval/` on `sys.path` instead of
# the repo root — `_ensure_repo_on_path()` (called from `main()`) fixes that. Import-time
# path mutation would leak into pytest sessions that import `eval.run_eval` for unit
# tests, so we defer it to `main()`.
try:
    from agent import triage
    from schema import TriageDecision
except ImportError:  # pragma: no cover — hit only under CLI invocation, exercised by main()
    triage = None  # type: ignore[assignment]
    TriageDecision = None  # type: ignore[assignment]


def _ensure_repo_on_path() -> None:
    """Prepend the repo root to `sys.path` if it isn't there. Idempotent.

    Called from `main()` only — never at import time — so importing `eval.run_eval` from
    a pytest session (which already has the repo root on `sys.path` via pyproject) does
    not silently mutate global state.
    """
    global triage, TriageDecision
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    if triage is None or TriageDecision is None:
        from agent import triage as _triage  # noqa: PLC0415
        from schema import TriageDecision as _TD  # noqa: PLC0415

        triage = _triage
        TriageDecision = _TD


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_labelled_tickets() -> list[dict[str, Any]]:
    """Read all 20 rows of `eval/labelled_tickets.csv` into the evaluate() shape.

    Each row becomes `{"inputs": {"ticket_id": ...}, "expectations": {...}}`.
    `expected_tools` is stored as a list of tool names split on comma (the CSV cell
    is `"get_ticket,get_customer_history"`), so a future scorer can compare span
    names against the expected sequence without re-parsing.
    """
    rows: list[dict[str, Any]] = []
    with _LABELLED_TICKETS_CSV.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            expected_tools = [t.strip() for t in row["expected_tools"].split(",") if t.strip()]
            rows.append(
                {
                    "inputs": {"ticket_id": row["ticket_id"]},
                    "expectations": {
                        "expected_category": row["expected_category"],
                        "expected_priority": row["expected_priority"],
                        "expected_tools": expected_tools,
                        "judge_notes": row["judge_notes"],
                    },
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


@scorer
def valid_schema(outputs: Any) -> bool:
    """Does the agent output round-trip through the Epic 1 `TriageDecision` schema?

    Returns True when `TriageDecision(**outputs)` validates. Any failure — missing
    key, wrong enum value, non-dict payload, `None`, etc. — returns False rather
    than raising, so a single malformed row never aborts the whole eval.
    """
    try:
        TriageDecision(**outputs)
        return True
    except Exception:
        return False


@scorer
def category_match(outputs: Any, expectations: dict[str, Any]) -> bool:
    """Does `outputs["category"]` equal `expectations["expected_category"]`?

    Independent of `valid_schema` — a missing/None `category` returns False without
    raising, so this scorer still reports per-row on malformed outputs.
    """
    try:
        return outputs.get("category") == expectations["expected_category"]
    except Exception:
        return False


@scorer
def priority_match(outputs: Any, expectations: dict[str, Any]) -> bool:
    """Does `outputs["priority"]` equal `expectations["expected_priority"]`?

    Same defensive shape as `category_match` — errors return False, not raise.
    """
    try:
        return outputs.get("priority") == expectations["expected_priority"]
    except Exception:
        return False


@scorer
def tool_order(trace: Any) -> bool:
    """Did the agent call `get_ticket` before `get_customer_history` in this trace?

    Reads the trace's tool spans by name and compares their `start_time_ns`. A
    missing trace or a missing span returns False (guarded, not raised), so the
    scorer reports a fail rather than aborting the eval on edge cases.
    """
    if trace is None:
        return False
    try:
        ticket_spans = trace.search_spans(name="get_ticket")
        history_spans = trace.search_spans(name="get_customer_history")
        if not ticket_spans or not history_spans:
            return False
        # `search_spans` list order isn't part of the API contract; if the agent's
        # retry-once path fires there could be more than one span with the same name.
        # Compare the earliest of each so retries can't flip a correct trace to False.
        earliest_ticket = min(s.start_time_ns for s in ticket_spans)
        earliest_history = min(s.start_time_ns for s in history_spans)
        return earliest_ticket < earliest_history
    except Exception:
        return False


def _scorers() -> list:
    """The four code scorers this story ships. `rationale_judge` is Story 3.2."""
    return [valid_schema, category_match, priority_match, tool_order]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _ensure_repo_on_path()
    load_dotenv()
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("triage-agent")
    mlflow.langchain.autolog()

    # Auto-approve HITL escalation prompts fired by `agent.triage()` during the eval.
    # Scoped to this process only: `run_agent.py` still prompts the operator on stdin.
    # A closure counter tracks the number of auto-approvals so we can report it after
    # the run.
    original_input = builtins.input
    escalation_count = 0

    def _auto_approve_input(prompt: str = "") -> str:
        nonlocal escalation_count
        if "escalate" in prompt.lower():
            escalation_count += 1
            return "yes"
        # Defensive: any non-escalate `input()` call falls through to the original so
        # we never auto-approve arbitrary prompts (e.g. a future debug prompt).
        return original_input(prompt)

    # `@mlflow.trace(span_type="AGENT")` on the wrapper guarantees the initial
    # `agent.ainvoke` AND the HITL-resume `ainvoke` nest under a single trace per
    # ticket. Without this wrapper each `ainvoke` autologs as a separate trace and
    # `tool_order` can't see both `get_ticket` and `get_customer_history` in the same
    # trace on escalating tickets.
    @mlflow.trace(span_type="AGENT")
    async def predict(ticket_id: str) -> dict:
        return await triage(ticket_id)

    data = _load_labelled_tickets()

    builtins.input = _auto_approve_input
    try:
        results = mlflow.genai.evaluate(
            data=data,
            predict_fn=predict,
            scorers=_scorers(),
        )
    finally:
        builtins.input = original_input

    # `results.metrics` keys are `{scorer_name}/mean` (default aggregation).
    metrics = results.metrics or {}
    print()
    print("=" * 60)
    print("Eval complete")
    print("=" * 60)
    print(f"MLflow run_id:              {results.run_id}")
    print(f"Auto-approved escalations:  {escalation_count}")
    print("Per-scorer means:")
    for name in ("valid_schema", "category_match", "priority_match", "tool_order"):
        key = f"{name}/mean"
        value = metrics.get(key)
        if value is None:
            print(f"  {name:<20} (missing)")
        else:
            print(f"  {name:<20} {float(value):.3f}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
