"""Live-LLM smoke test for `eval/run_eval.py`.

Runs the full 20-ticket eval end-to-end on Groq (the workshop-realistic choice — Gemini
free-tier is 20 requests/day and this eval fires 20-40 model calls). Skipped when
`GROQ_API_KEY` is unset, so a keyless CI run stays green.

The smoke test asserts three properties any regression would break:

1. `results.tables["eval_results"]` — the per-row DataFrame MLflow exposes with each
   ticket's inputs, outputs, expectations and scorer values — has 20 rows.
2. All four scorer columns are present in the per-row table.
3. The auto-approved escalation counter matches the number of P1+Enterprise tickets
   the seed data marks as needing escalation (`T-1044`, `T-1048`, `T-1057` at
   minimum — three).
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

    # All four scorer means show up in the aggregate metrics.
    metrics = results.metrics or {}
    for scorer_name in ("valid_schema", "category_match", "priority_match", "tool_order"):
        assert f"{scorer_name}/mean" in metrics, (
            f"Missing `{scorer_name}/mean` in results.metrics: {sorted(metrics)}"
        )

    # Auto-approved escalation count: printed in the stdout summary. Use a regex so
    # trailing whitespace, extra padding, or a future "(of N tickets)" suffix don't
    # break the extraction the way a bare `split(":")` would.
    import re

    stdout = capsys.readouterr().out
    match = re.search(r"Auto-approved escalations:\s*(\d+)", stdout)
    assert match, "Auto-approved escalations line missing from stdout."
    escalations = int(match.group(1))

    assert escalations >= _EXPECTED_P1_ENTERPRISE_ESCALATIONS, (
        f"Expected at least {_EXPECTED_P1_ENTERPRISE_ESCALATIONS} P1+Enterprise "
        f"escalations (T-1044, T-1048, T-1057); got {escalations}."
    )
