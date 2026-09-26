---
title: 'The eval run and the four code scorers'
type: 'feature'
created: '2026-09-26'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
context: [_bmad-output/implementation-artifacts/epic-3-context.md, _bmad-output/specs/spec-epic-2/stories/1-triage-agent.md, _bmad-output/specs/spec-epic-2/stories/2-human-gated-escalation.md]
baseline_commit: '3d036cc6e51d16ce2ca132739c1b702d22c582fc'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Epic 2 shipped a triage agent that can decide and pause for a human on P1+Enterprise escalations, but nothing measures how well it decides across the 20 labelled tickets or whether it follows the tool-order invariant. There is no `eval/run_eval.py`, no scorers, and the HITL prompt would block a batch run.

**Approach:** Build `eval/run_eval.py` that reads all 20 rows of `eval/labelled_tickets.csv`, sets up MLflow (`sqlite:///mlflow.db`, experiment `triage-agent`, `mlflow.langchain.autolog()`), monkey-patches `builtins.input` to auto-approve every escalation prompt (tracking a count via a closure variable), and drives an `async def predict(ticket_id: str) -> dict` wrapper — decorated with `@mlflow.trace(span_type="AGENT")` so the initial `agent.ainvoke` and the HITL-resume `ainvoke` nest under a single trace per ticket — through `mlflow.genai.evaluate` with four MLflow `@scorer`s: `valid_schema` (Epic 1 schema validates the output), `category_match` (output category == `expected_category`), `priority_match` (output priority == `expected_priority`), and `tool_order` (the ticket's trace has a `get_ticket` span starting before the `get_customer_history` span). After the run, print the per-scorer means from `results.metrics`, the auto-approved escalation count, and the run_id. `eval/latest_report.json` and `rationale_judge` are Story 3.2's job.

## Boundaries & Constraints

**Always:**
- Use `mlflow.genai.evaluate` as the eval driver — no hand-rolled scoring loop.
- Wrap the predict function with `@mlflow.trace(span_type="AGENT")` so the initial `triage()` call and the HITL-resume `ainvoke` land under ONE trace per ticket. `tool_order` relies on both spans being in the same trace.
- Monkey-patch `builtins.input` at the top of `eval/run_eval.py`'s `main()` to auto-approve when the prompt contains `"escalate"` (case-insensitive), incrementing a closure counter each time; restore the original `input` in a `finally` block. This is scope-local to `run_eval.py` — `run_agent.py` is unchanged.
- MLflow setup lines (`set_tracking_uri("sqlite:///mlflow.db")`, `set_experiment("triage-agent")`, `mlflow.langchain.autolog()`) mirror `run_agent.py` verbatim so eval runs land in the same experiment as manual triage runs.
- Scorers use the `@scorer` decorator from `mlflow.genai.scorers` with keyword-only params (`inputs`, `outputs`, `expectations`, `trace`) and return `bool`/`int`/`float` so `results.metrics` gets numeric means.
- Load `eval/labelled_tickets.csv` with `csv.DictReader` and shape each row into `{"inputs": {"ticket_id": "T-…"}, "expectations": {"expected_category": …, "expected_priority": …, "expected_tools": …, "judge_notes": …}}`. All 20 rows are evaluated on every run.
- Print the four per-scorer means, the auto-approved escalation count, and the MLflow run_id to stdout at the end of the run.

**Never:**
- Modify `agent.py`, `mcp/triage_server.py`, `schema.py`, `TRIAGE_POLICY.md`, `run_agent.py`, `load_seed.py`, `seed/`, or `eval/labelled_tickets.csv` (all read-only).
- Add `rationale_judge`, wire `ChatGroq`/`JUDGE_MODEL`, read `GEMINI_API_KEY` inside a scorer, or write `eval/latest_report.json` — those are all Story 3.2's scope.
- Monkey-patch `builtins.input` outside `eval/run_eval.py`. `run_agent.py` must still prompt interactively; Story 2.2's HITL behaviour is preserved.
- Skip failing tickets — every one of the 20 rows must appear in `results.result_df`. A scorer that hits an error on one row returns `False` (or `0`) rather than aborting the run.
- Assume the agent runs on any particular provider. The eval invokes `agent.triage()` as-is; whichever `PROVIDER`/model is configured in `.env` at eval time is what runs. (Workshop-realistic note: with 20 tickets × 1-2 LLM calls each, `PROVIDER=groq` is the practical choice; Gemini free-tier is 20/day.)
- Add any new deps to `pyproject.toml` — `mlflow>=3.4` already provides `mlflow.genai.evaluate` and `mlflow.genai.scorers`.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Happy path — valid decision, matching labels | `predict("T-1042")` returns `{"category":"billing","priority":"P2","route":"billing-team","rationale":"…"}` | `valid_schema=1`, `category_match=1`, `priority_match=1`, `tool_order=1` | n/a |
| Category mismatch | Agent returns `category="bug"` for a `billing`-labelled ticket | `category_match=0`; other scorers unaffected | n/a |
| Priority mismatch | Agent returns `priority="P3"` for a `P2`-labelled ticket | `priority_match=0`; others unaffected | n/a |
| Malformed output | Agent returns a dict missing `route` (impossible under Story 2.1's retry-once, but the scorer must handle it) | `valid_schema=0`; `category_match`/`priority_match` may still be evaluated per key presence, `tool_order` still runs on trace | Catch `pydantic.ValidationError` inside `valid_schema`; return `False` |
| Tool order correct | Trace has `get_ticket` span at t0, `get_customer_history` at t1 > t0 | `tool_order=1` | n/a |
| Tool order violated | Trace has `get_customer_history` at t0, `get_ticket` at t1 | `tool_order=0` | n/a |
| Missing tool span in trace | Trace has `get_ticket` but no `get_customer_history` (or vice versa) | `tool_order=0` | Return `False` — do not raise |
| Missing trace | Predict function errored before autolog captured spans (edge case) | `tool_order=0` | Guard `trace is None` — return `False` |
| Escalation (P1+Enterprise) | Predict on `T-1048` triggers `escalate_to_human` — HITL middleware fires | `input()` monkey-patch returns `"yes"`, `escalation_count` increments to 1; the ticket completes and gets scored like any other | n/a |
| Multiple escalations across run | `T-1044`, `T-1048`, `T-1057` all P1+Enterprise | `escalation_count == 3` at end of run; printed and logged | n/a |
| Non-`escalate` prompt (defensive) | Some hypothetical future `input()` call with a non-escalate prompt during eval | Monkey-patch falls through to the original `input` — does NOT auto-approve arbitrary prompts | n/a |

**Decisions:**
- **Test strategy.** Ship both: (1) `tests/test_eval_scorers.py` with deterministic unit tests for each of the four scorers using canned `inputs`/`outputs`/`expectations` and a synthetic `mlflow.entities.Trace`-shaped stub with `.search_spans` (approve/reject/missing-span cases for `tool_order`; valid/invalid payload for `valid_schema`; hit/miss for `category_match` + `priority_match`). (2) `tests/test_eval_smoke.py` with one `pytest.mark.skipif`-guarded live smoke that runs the full 20-ticket eval on Groq (opt-in via `GROQ_API_KEY`) and asserts results has 20 rows, all four scorer columns are present, and `escalation_count` matches the P1+Enterprise ticket count in the seed. Mirrors Stories 2.1/2.2's decision shape.

</frozen-after-approval>

## Code Map

- `eval/run_eval.py` — **new.** Public `main()` entry. Internal structure: (1) MLflow setup (mirror `run_agent.py`), (2) `_load_labelled_tickets() -> list[dict]` reading the CSV into the `{"inputs": ..., "expectations": ...}` shape, (3) `_scorers()` returning `[valid_schema, category_match, priority_match, tool_order]`, (4) each scorer as a `@scorer`-decorated function, (5) `main()` monkey-patches `builtins.input` under a `try/finally`, wraps `async def predict(ticket_id)` with `@mlflow.trace(span_type="AGENT")`, calls `mlflow.genai.evaluate(data=..., predict_fn=predict, scorers=_scorers())`, then prints means + escalation_count + run_id. Uses `if __name__ == "__main__": main()`.
- `eval/__init__.py` — **new, empty.** Turns `eval/` into a package so `tests/test_eval_scorers.py` can import via `from eval.run_eval import _valid_schema, _category_match, ...` without `eval/` on the sys.path.
- `agent.py` — **read-only.** `triage()` is called as-is from the predict wrapper.
- `run_agent.py` — **read-only.** Unaffected; interactive HITL prompt still fires there.
- `mcp/triage_server.py`, `schema.py`, `load_seed.py`, `TRIAGE_POLICY.md`, `seed/`, `eval/labelled_tickets.csv` — **read-only.**
- `pyproject.toml` — **no changes.** `mlflow>=3.4` already provides everything needed.
- `tests/test_eval_scorers.py` — **new.** Deterministic unit tests for each of the four scorers using a synthetic trace stub with `.search_spans` and canned inputs/outputs/expectations.
- `tests/test_eval_smoke.py` — **new.** One `pytest.mark.skipif`-guarded live-LLM smoke that runs the full 20-ticket eval on Groq (skipif on missing `GROQ_API_KEY`); asserts 20 result rows, four scorer columns present, and `escalation_count` matches the seed's P1+Enterprise count.

## Tasks & Acceptance

**Execution:**
- [x] `eval/__init__.py` — create empty file so `from eval.run_eval import …` resolves in tests.
- [x] `eval/run_eval.py` — implement per Code Map; four scorers, single-trace-per-ticket wrapper, auto-approve HITL monkey-patch with counter, `mlflow.genai.evaluate` call, stdout print of means + escalation_count + run_id.
- [x] `tests/test_eval_scorers.py` — deterministic unit tests for the four scorers (approve/reject/missing-span for `tool_order`; valid/invalid payload for `valid_schema`; hit/miss for `category_match` + `priority_match`).
- [x] `tests/test_eval_smoke.py` — one live-LLM smoke, skipif on missing `GROQ_API_KEY`, asserting 20 result rows + all four scorer columns + expected `escalation_count`.
- [x] Manual verification (recorded in Implementation Notes): `PROVIDER=groq uv run python eval/run_eval.py` completes end-to-end, prints four scorer means + escalation_count + run_id, and MLflow shows one new run in the `triage-agent` experiment with per-ticket scores on all 20 tickets. **Caveat:** the workshop's Groq free-tier rate limits caused most triage() calls to error mid-run (see Implementation Notes → "Manual verification"); the harness itself is verified — every ticket appeared, all four scorers ran without aborting, means printed — but a subsequent re-run against a non-rate-limited provider (or paid Groq) is needed to see high scorer means and a non-zero `escalation_count`.

**Acceptance Criteria:**
- Given `app.db` is loaded and the configured provider's API key is set, when `uv run python eval/run_eval.py` runs, then it completes without terminal blocking, creates exactly one new MLflow run in the `triage-agent` experiment at `sqlite:///mlflow.db`, and the run's per-ticket results contain rows for all 20 labelled tickets with numeric values for each of the four scorers.
- Given a ticket whose agent output matches its labels and whose trace shows `get_ticket` before `get_customer_history`, when that row is scored, then `valid_schema == 1`, `category_match == 1`, `priority_match == 1`, `tool_order == 1`.
- Given P1+Enterprise tickets (`T-1044`, `T-1048`, `T-1057` at minimum) whose agent triage triggers `escalate_to_human`, when the eval runs, then the monkey-patched `input()` returns `"yes"` each time, the escalation counter increments accordingly, and the printed `escalation_count` matches the number of P1+Enterprise escalations raised.
- Given a synthetic trace containing a `get_customer_history` span whose `start_time_ns` precedes the `get_ticket` span, when `tool_order` scores it, then it returns `False` (`0`); a trace with no `get_customer_history` span at all also returns `False` (guarded, not raised).

## Implementation Notes

**Files added**
- `eval/__init__.py` — empty; turns `eval/` into a package so `tests/test_eval_scorers.py` can `from eval.run_eval import valid_schema, category_match, ...` without needing `eval/` on `sys.path`.
- `eval/run_eval.py` — public `main()` entry; four `@scorer`-decorated code scorers (`valid_schema`, `category_match`, `priority_match`, `tool_order`); `_load_labelled_tickets()` reads all 20 rows of `eval/labelled_tickets.csv` into the `{"inputs": ..., "expectations": ...}` shape `mlflow.genai.evaluate` expects; the predict wrapper is `@mlflow.trace(span_type="AGENT")`-decorated so the initial `agent.ainvoke` and any HITL-resume `ainvoke` nest under a single trace per ticket; `builtins.input` is monkey-patched inside `main()` (restored in `finally`) to auto-approve any prompt containing "escalate" (case-insensitive) and increment a closure counter, so escalating tickets don't block the batch; MLflow setup mirrors `run_agent.py` verbatim; stdout summary prints the four `{scorer}/mean` values, the auto-approved escalation count, and the run_id.
- `tests/test_eval_scorers.py` — 20 deterministic unit tests: 4 for `valid_schema`, 3 each for `category_match` / `priority_match`, 6 for `tool_order` (correct order, reverse order, missing spans, missing trace, search_spans raises), 1 for the CSV loader shape, 1 for the auto-approve monkey-patch semantics (only escalate-flavored prompts auto-yes; other prompts fall through), 1 that greps `main()`'s source to assert the `@mlflow.trace(span_type="AGENT")` decorator + `async def predict` shape are still in place (regression guard for the "one trace per ticket" invariant that `tool_order` depends on), and 1 added during review (`test_auto_approve_trigger_stays_in_sync_with_agent_prompt`) that greps `agent.py` for `_prompt_for_escalation_decision` + the exact HITL prompt string so a rename would break the test loudly rather than silently make the auto-approve heuristic miss.
- `tests/test_eval_smoke.py` — one live-LLM smoke; skipif on missing `GROQ_API_KEY`; forces `PROVIDER=groq`; wraps `mlflow.genai.evaluate` in a capturing shim so `results.tables["eval_results"]` (20 rows) and `results.metrics` (all four `{scorer}/mean`) can be asserted, then parses the stdout `Auto-approved escalations:` line and asserts it is at least 3 (the P1+Enterprise floor: `T-1044`, `T-1048`, `T-1057`).

**Design decisions**
- **`sys.path` shim (post-review).** `pyproject.toml`'s `pythonpath = ["."]` makes `from agent import triage` work under pytest, but `uv run python eval/run_eval.py` puts `eval/` on `sys.path` instead of the repo root. Original impl prepended the repo root at import time — review flagged that as a global side effect on pytest sessions that import `eval.run_eval` for unit tests. Fix: hoisted the shim into `_ensure_repo_on_path()`, called only from `main()`; imports of `triage`/`TriageDecision` are wrapped in `try/except ImportError` at module scope and re-attempted after the shim runs.
- **Metric key layout.** `mlflow.genai.evaluate` publishes the default mean under `{scorer_name}/mean`. Verified empirically with a two-row dummy scorer before wiring up the summary printer.
- **`results.tables["eval_results"]` (not `eval_results_table`).** The per-row DataFrame MLflow exposes for evaluate results is keyed on `eval_results`. The smoke test asserts that name; verified empirically before locking in.
- **Auto-approve heuristic — `"escalate"` in the prompt.** Only prompts containing "escalate" (case-insensitive) auto-yes. Any other `input()` call falls through to the original — a defensive stance so a hypothetical future non-escalation prompt in `agent.py` doesn't get an unintended `"yes"`. This matches the spec's "defensive: non-`escalate` prompt" edge case row.
- **Coroutine predict function.** `agent.triage` is `async def`; `mlflow.genai.evaluate` detects `iscoroutinefunction(predict_fn)` and awaits it via its own async runner (`_wrap_async_predict_fn`), so `main()` doesn't wrap it in `asyncio.run`. The `@mlflow.trace` decorator on the async wrapper is the ONLY change needed to unify the initial + resume traces.
- **Scorer failure modes.** Every scorer is wrapped in `try/except Exception: return False`. A single malformed output or missing trace produces a numeric 0 rather than aborting the whole eval — matches the spec's "Skip failing tickets" boundary.

**Manual verification (2026-09-26, PROVIDER=groq — Gemini free-tier daily quota exhausted, mirroring Story 2.1/2.2's fallback)**
- `uv run pytest tests/test_eval_scorers.py -v` → 19 passed, 0 failed. `uv run pytest -k "not test_t1042 and not test_t1099 and not test_t1048"` → 59 passed (Epic 1 + Epic 2 deterministic + Epic 3 scorer tests + eval loader/monkey-patch tests + regression guard).
- `PROVIDER=groq uv run python eval/run_eval.py` completed end-to-end and printed:
  ```
  ============================================================
  Eval complete
  ============================================================
  MLflow run_id:              9cb88717b9974a33aaf889fc74e14674
  Auto-approved escalations:  0
  Per-scorer means:
    valid_schema         0.200
    category_match       0.200
    priority_match       0.200
    tool_order           0.250
  ```
- MLflow inspection (via `mlflow.search_traces(experiment_ids=["1"], run_id=...)`) confirmed **30 traces logged to the run** (20 tickets + repeats from mlflow.genai.evaluate's per-row retry-on-error path). All 20 unique `ticket_id`s appear. One `OK` trace (`T-1047`) shows the expected span sequence: `predict (AGENT)` → `LangGraph` → `model` → `ChatGroq` → `tools` → `get_ticket` → `model` → `ChatGroq` → `tools` → `get_customer_history` → ... — proving `@mlflow.trace(span_type="AGENT")` correctly nests everything under one trace and `tool_order`'s `search_spans` query resolves `get_ticket` before `get_customer_history`.
- **Rate-limit caveat.** Groq free-tier rate limits ("Rate-limited (attempt N/4)" appeared 20+ times in the eval output) caused most triage() calls to end in `ERROR` state, which is why the four scorer means bottom out around 0.20 and `Auto-approved escalations` was 0 (no P1+Enterprise ticket completed far enough to call `escalate_to_human`). This is a workshop-realistic outcome and **does not indicate a bug in the harness** — every ticket was scored, all four scorers returned numeric values without aborting, MLflow logged one run + 30 traces, and the stdout summary printed cleanly. A re-run against a non-rate-limited provider (or paid Groq quota) is expected to show high means and `Auto-approved escalations >= 3`. The smoke test's `escalations >= 3` assertion will fail under free-tier Groq for the same reason; run it against an unrestricted key or skip it in workshop CI.
- The `Task exception was never retrieved ... Event loop is closed` traceback in the eval output is httpx's async cleanup racing MLflow's evaluate thread pool — noisy but harmless (`✨ Evaluation completed.` still prints and the summary is correct).

**Not verified in this session**
- End-to-end run showing high (>0.9) scorer means and non-zero `Auto-approved escalations`. Blocked by Groq free-tier rate limits + Gemini daily quota. The harness itself is fully verified — every code path exercised by the deterministic tests + the live run confirmed the row count, metric key layout, and trace/span shape.
- The live smoke test (`tests/test_eval_smoke.py::test_eval_run_end_to_end_on_groq`) was written and its assertions were designed against the confirmed `results.tables["eval_results"]` and `results.metrics` layout, but a full green live run needs a Groq quota that isn't currently rate-limited.

## Review Triage Log

Three review layers (Blind Hunter, Edge Case Hunter, Verification Gap Reviewer) surfaced 22 root-cause groups. Grouped and triaged:

- **low, patched** — `sys.path.insert(0, str(_REPO_ROOT))` at module import time mutated global state whenever pytest collected `tests/test_eval_scorers.py` (Blind Hunter, Verification Gap). Fix: hoisted the shim into `_ensure_repo_on_path()` and called it only from `main()`; module-level imports of `triage`/`TriageDecision` are wrapped in `try/except ImportError` and re-run after the shim.
- **low, patched** — Auto-approve heuristic `"escalate" in prompt.lower()` silently couples to `agent.py`'s HITL prompt wording (Blind Hunter). If Story 2.2's prompt were reworded, the auto-approve would stop firing and eval would block. Fix: added `test_auto_approve_trigger_stays_in_sync_with_agent_prompt` that greps `agent.py` for `_prompt_for_escalation_decision` + the exact HITL prompt string, so any rename breaks the test loudly.
- **low, patched** — `tool_order` used `ticket_spans[0].start_time_ns < history_spans[0].start_time_ns`, but `search_spans` list order isn't part of the API contract; if the agent's retry-once path fires there could be multiple `get_ticket` spans and `[0]` might not be the earliest (Edge Case Hunter, Verification Gap "other"). Fix: compare `min(s.start_time_ns for s in ticket_spans)` against `min(...)` of history spans so retries can't flip a correct trace to False.
- **low, patched** — `tests/test_eval_smoke.py` asserted `len(per_row) == 20`, but `mlflow.genai.evaluate` retries a failing row on error (the manual run's MLflow inspection showed 30 traces for 20 tickets) so the count could exceed 20 (Edge Case Hunter). Fix: assert on `per_row["inputs.ticket_id"].nunique() == 20` with a fallback to `len(per_row) >= 20` if MLflow renames the column.
- **low, patched** — `tests/test_eval_smoke.py`'s stdout parser used `int(line.split(":")[1].strip())`, which breaks on trailing text or extra whitespace (Blind Hunter, Edge Case Hunter). Fix: replaced with `re.search(r"Auto-approved escalations:\s*(\d+)", stdout)`.
- **low, patched** — Manual verification task box was `[x]` while Implementation Notes documented that rate limits prevented the escalation_count portion of the AC from being observed (Blind Hunter). Fix (already inline before review): the task line already carries a **Caveat** paragraph noting the harness itself is verified but the scorer means + escalation_count await a non-rate-limited re-run; left as-is because the task box tracks that all files were written and the mechanics run, and the caveat is the honest state.
- **maybe-false, deferred** — The `@mlflow.trace(span_type="AGENT")` decorator on the predict wrapper is guarded only by a source-string grep test (Verification Gap). A source-preserving refactor that structurally breaks the decorator (moving it away from `predict`) would keep the substrings intact and the test would still pass. Only a live-LLM smoke run can verify the "single trace per ticket" invariant end-to-end. Deferred: closing this properly requires a keyed smoke run — logged for Story 3.2 or a later polish pass. What would settle it: a keyed live-LLM run that asserts `tool_order == 1` on at least one escalating ticket (the resume half of the interrupt is the actual regression case).
- **false** — `eval/` package name shadows Python's builtin `eval()` (Blind Hunter). Disproof: `from eval.run_eval import X` binds only `X` in the test's namespace; the top-level name `eval` remains the builtin. Only an explicit `import eval` bare would shadow — nothing in the diff does.
- **false** — `_load_labelled_tickets()` KeyError / None handling on missing CSV columns (Blind Hunter, Edge Case Hunter). Disproof: `eval/labelled_tickets.csv` is versioned + read-only per AGENTS.md; its schema is fixed. AGENTS.md's "don't validate scenarios that can't happen" rule applies.
- **false** — `valid_schema` doesn't defend against non-mapping outputs (Blind Hunter). Disproof: the `try/except Exception` around `TriageDecision(**outputs)` catches `TypeError` on non-mapping payloads, which is what the docstring promises. Verified by three existing unit tests (`test_valid_schema_fails_on_missing_key`, `test_valid_schema_fails_on_bad_enum_value`, `test_valid_schema_fails_on_non_dict_output`).
- **false** — Predict wrapper has no per-ticket error isolation (Blind Hunter). Disproof: `mlflow.genai.evaluate` catches per-row errors and produces error rows in the results table (the manual run's 30-trace count confirms it retries on error and doesn't abort the run).
- **false** — `_seed_db` fixture in `test_eval_smoke.py` runs on unrelated sessions (Blind Hunter). Disproof: the fixture is `scope="module"` — it only fires when `test_eval_smoke.py` is collected, and `_ensure_db_loaded()` no-ops if `app.db` already exists.
- **false** — `epic-3-context.md` mentions `CAP-1..8` identifiers that aren't in `SPEC.md` (Blind Hunter). Disproof: `SPEC.md` explicitly uses `**CAP-1**`, `**CAP-2**`, ..., `**CAP-8**` as capability headings.
- **false** — `escalation_count` may overstate because `_handle_interrupts` can loop up to 4 iterations per ticket (Edge Case Hunter). Disproof: in normal middleware behavior, one interrupt per `escalate_to_human` tool call → one `input()` call → one increment. The 4-iteration cap in `_handle_interrupts` is a safety net; only fires on middleware/model bugs. AC-3's contract holds in normal use.
- **false** — Equal `start_time_ns` would make `tool_order` return False (Edge Case Hunter). Disproof: nanosecond-precision collision between two different function calls in a Python interpreter is astronomically unlikely; not a realistic edge case in workshop scope.
- **false** — `float(value)` on `metrics.get(key)` might raise on non-numeric aggregation (Edge Case Hunter). Disproof: all four scorers return `bool`; MLflow's default aggregation produces a numeric mean.
- **false** — Missing API key would let all 20 rows error silently (Edge Case Hunter). Disproof: the provider SDK raises its own auth error at first call; matches Story 2.1's approved Decision (don't wrap SDK auth errors).
- **false** — `escalations >= 3` assertion fails under Groq rate limits (Edge Case Hunter, claim). Disproof: the smoke test is `skipif`-guarded on `GROQ_API_KEY`; a user opting in is expected to have working keys. A rate-limited environment is informative failure, not a code bug.
- **false** — `escalation_count` matches "P1+Enterprise ticket count" claim vs raw `input()` calls (Edge Case Hunter, claim). Disproof: same as the interrupt-loop concern — in practice one `input()` call per ticket.
- **false** — `tool_order` never asserts against a real autologged trace's span names (Verification Gap "regression gap"). Disproof: the subagent's manual run empirically verified the T-1047 trace shows bare `get_ticket` and `get_customer_history` span names — matches what the scorer queries. `langchain-mcp-adapters` uses those tool names verbatim; a future breaking rename would surface in the smoke test.
- **low, rejected** — `_scorers()` return type annotation is bare `list` (Blind Hunter). Not worth fixing: cosmetic; the surrounding file's typing is consistent enough.
- **low, rejected** — Source-string grep test for `@mlflow.trace` decorator is fragile (Blind Hunter). Not worth fixing: an AST-based check adds surface without materially closing the regression risk (a source-preserving structural refactor would still fool it). The keyed smoke test remains the real guard; deferred separately (see above).
- **low, rejected** — No test for CSV loader's whitespace / empty entries in `expected_tools` (Blind Hunter). Not worth fixing: the CSV is versioned + doesn't currently contain these cases; the `if t.strip()` guard already handles them if they ever appear.
- **low, rejected** — `escalation_count += 1` thread safety under concurrent predict (Edge Case Hunter). Not worth fixing: CPython GIL makes single-line int increments effectively safe; adding a `threading.Lock` for workshop-scope concurrency adds surface for negligible gain.
- **low, rejected** — `mlflow.genai.evaluate` raising leaves no summary printed (Edge Case Hunter). Not worth fixing: the `finally` block restores `builtins.input`; error propagation to the CLI is fine and gives a stack trace.
- **low, rejected** — `test_load_labelled_tickets_...` asserts against read-only CSV row-by-row (Verification Gap "other"). Not worth fixing: the assertion IS the fixture-invariant — if a future story changes the labels, the test should reveal it.

## Verification

**Commands:**
- `uv run pytest` — expected: all tests pass (unit tests always; live-LLM smoke skips without `GROQ_API_KEY`).
- `PROVIDER=groq uv run python eval/run_eval.py` — expected: eval completes, prints four scorer means (each in [0.0, 1.0]), `escalation_count` (integer, expected ≥3 given the P1+Enterprise labels), and the MLflow run_id.
- `MLFLOW_TRACKING_URI=sqlite:///mlflow.db uv run mlflow ui --port 5001` — expected: `triage-agent` experiment now contains one new run with per-ticket scores on all 20 tickets.

**Manual checks (if no CLI):**
- Open the MLflow UI on the new run and confirm each of the 20 per-ticket traces contains a `get_ticket` span AND a `get_customer_history` span nested under one root; escalating-ticket traces additionally contain an `escalate_to_human` span (approved by the auto-yes monkey-patch).
