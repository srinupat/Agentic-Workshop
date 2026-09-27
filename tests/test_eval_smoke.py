"""Live-LLM smoke test for `eval/run_eval.py`.

Runs the full 20-ticket eval end-to-end on Groq (the workshop-realistic choice — Gemini
free-tier is 20 requests/day and this eval fires 20-40 model calls). Skipped when
`GROQ_API_KEY` is unset, so a keyless CI run stays green.

The smoke test asserts:

1. `results.tables["eval_results"]` — the per-row DataFrame MLflow exposes with each
   ticket's inputs, outputs, expectations and scorer values — has 20 distinct tickets.
2. All five scorer means (`valid_schema`, `category_match`, `priority_match`,
   `tool_order`, `rationale_judge`) are present in `results.metrics`.
3. The auto-approved escalation counter matches the number of P1+Enterprise tickets
   the seed data marks as needing escalation (`T-1044`, `T-1048`, `T-1057` at
   minimum — three).
4. `eval/latest_report.json` exists on disk after the run and its numbers are
   byte-identical to `results.metrics`, the auto-approved escalation count printed to
   stdout, and the agent-total-tokens number printed to stdout.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parent.parent

_EXPECTED_P1_ENTERPRISE_ESCALATIONS = 3  # T-1044, T-1048, T-1057


def _ensure_db_loaded() -> None:
    if (_REPO_ROOT / "app.db").exists():
        return
    subprocess.run(
        [sys.executable, str(_REPO_ROOT / "load_seed.py")],
        cwd=_REPO_ROOT,
        check=True,
        timeout=60,
    )


@pytest.fixture(scope="module", autouse=True)
def _seed_db() -> None:
    _ensure_db_loaded()


@pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="Live-LLM smoke: skipping without GROQ_API_KEY.",
)
def test_eval_run_end_to_end_on_groq(monkeypatch, capsys) -> None:
    """`eval.run_eval.main()` completes on the full 20-ticket dataset (Groq provider)."""
    monkeypatch.setenv("PROVIDER", "groq")

    # Import lazily so the module-level MLflow setup doesn't run at collection time.
    import mlflow
    from eval import run_eval

    # Capture the `results` object mlflow.genai.evaluate returns without duplicating
    # the eval driver — patch the fluent import site inside `run_eval` and delegate to
    # the real evaluate, storing the result for assertions.
    real_evaluate = mlflow.genai.evaluate
    captured: dict = {}

    def _capturing_evaluate(**kwargs):
        results = real_evaluate(**kwargs)
        captured["results"] = results
        return results

    monkeypatch.setattr(run_eval.mlflow.genai, "evaluate", _capturing_evaluate)

    run_eval.main()

    results = captured["results"]
    assert results is not None, "mlflow.genai.evaluate returned no result."
    assert results.run_id, "Expected a non-empty run_id."

    # Per-row table: 20 rows, one per labelled ticket. MLflow exposes it as the
    # `eval_results` DataFrame in `results.tables`.
    tables = results.tables or {}
    per_row = tables.get("eval_results")
    assert per_row is not None, (
        "Expected an `eval_results` DataFrame in results.tables; MLflow layout changed."
    )
    # Assert on distinct ticket IDs — `mlflow.genai.evaluate` may retry a failing row
    # and emit more than one entry per ticket, so `len(per_row) == 20` is too strict.
    ticket_col = "inputs.ticket_id"
    if ticket_col in per_row.columns:
        assert per_row[ticket_col].nunique() == 20, (
            f"Expected 20 distinct ticket IDs, got "
            f"{per_row[ticket_col].nunique()} across {len(per_row)} rows."
        )
    else:
        # Fallback if MLflow renames the column: at least one row per ticket.
        assert len(per_row) >= 20, f"Expected >= 20 rows, got {len(per_row)}."

    # All FIVE scorer means show up in the aggregate metrics (four code scorers +
    # rationale_judge from Story 3.2).
    metrics = results.metrics or {}
    scorer_names = (
        "valid_schema",
        "category_match",
        "priority_match",
        "tool_order",
        "rationale_judge",
    )
    for scorer_name in scorer_names:
        assert f"{scorer_name}/mean" in metrics, (
            f"Missing `{scorer_name}/mean` in results.metrics: {sorted(metrics)}"
        )

    # Auto-approved escalation count + agent total tokens: both are printed in the
    # stdout summary. Use regexes so trailing whitespace, extra padding, or a future
    # "(of N tickets)" suffix don't break the extraction the way a bare `split(":")`
    # would.
    import re

    stdout = capsys.readouterr().out
    esc_match = re.search(r"Auto-approved escalations:\s*(\d+)", stdout)
    assert esc_match, "Auto-approved escalations line missing from stdout."
    escalations = int(esc_match.group(1))

    tok_match = re.search(r"Agent total tokens:\s*(\d+)", stdout)
    assert tok_match, "Agent total tokens line missing from stdout."
    stdout_tokens = int(tok_match.group(1))

    assert escalations >= _EXPECTED_P1_ENTERPRISE_ESCALATIONS, (
        f"Expected at least {_EXPECTED_P1_ENTERPRISE_ESCALATIONS} P1+Enterprise "
        f"escalations (T-1044, T-1048, T-1057); got {escalations}."
    )

    # `eval/latest_report.json` must exist and match the same numbers stdout printed
    # and `results.metrics` reported. Single source of truth: no divergence allowed.
    import json

    report_path = _REPO_ROOT / "eval" / "latest_report.json"
    assert report_path.exists(), "eval/latest_report.json was not written."

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(report) == {"run_id", "scorer_means", "agent_total_tokens", "escalation_count"}, (
        f"Report top-level keys mismatch: {sorted(report)}"
    )
    assert report["run_id"] == results.run_id
    assert report["escalation_count"] == escalations
    assert report["agent_total_tokens"] == stdout_tokens
    assert set(report["scorer_means"]) == set(scorer_names), (
        f"scorer_means keys mismatch: {sorted(report['scorer_means'])}"
    )
    # Byte-identical numbers between the report and the aggregate metrics.
    for name in scorer_names:
        assert report["scorer_means"][name] == float(metrics[f"{name}/mean"]), (
            f"Report `{name}` mean diverges from results.metrics."
        )
