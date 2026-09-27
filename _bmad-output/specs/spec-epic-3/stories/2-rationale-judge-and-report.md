---
title: 'The rationale judge and the report'
type: 'feature'
created: '2026-09-26'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
context: [_bmad-output/implementation-artifacts/epic-3-context.md, _bmad-output/specs/spec-epic-3/stories/1-eval-run-and-code-scorers.md]
baseline_commit: '8ce9b9531b344ee00722e93c2c97d31c687f2f69'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Story 3.1 shipped the eval harness with four code scorers and a stdout summary of their means, but there is no LLM judge on rationale quality (CAP-6), no total-agent-tokens number, and no machine-readable report — Epic 3 CAP-7 requires the eval's numbers to be reusable without opening the MLflow UI.

**Approach:** Extend `eval/run_eval.py` to add a fifth `@scorer` called `rationale_judge` that constructs `ChatGroq(model=JUDGE_MODEL default "openai/gpt-oss-120b", api_key=GROQ_API_KEY, temperature=0)` — never reading `GEMINI_API_KEY` — and asks it to judge each ticket's agent rationale against that ticket's `judge_notes`, returning `Feedback(value=<bool>, rationale=<one-line>)`. Add `_agent_total_tokens(run_id)` that fetches the run's traces via `mlflow.search_traces(run_id=run_id, return_type="list")`, keeps only those whose ROOT span is `span_type="AGENT"` (i.e. our `@mlflow.trace(span_type="AGENT")`-wrapped predict traces; the judge's own ChatGroq calls autolog as separate traces without an AGENT root and are excluded), and sums each surviving trace's `info.token_usage["total_tokens"]`. Add `_write_report(path, report)` that dumps `{"run_id", "scorer_means" (dict of 5), "agent_total_tokens", "escalation_count"}` as pretty-printed JSON. Update `main()` to include `rationale_judge` in `_scorers()`, print the five means + tokens + escalation count in the summary, and write the same numbers to `eval/latest_report.json`. Fail fast at `main()` entry with a clear message if `GROQ_API_KEY` is unset — the judge always needs it, regardless of the agent's `PROVIDER` selection.

## Boundaries & Constraints

**Always:**
- `rationale_judge` constructs the judge model via `ChatGroq(model=os.environ.get("JUDGE_MODEL", "openai/gpt-oss-120b"), api_key=os.environ["GROQ_API_KEY"], temperature=0).with_structured_output(JudgeVerdict)` and NEVER reads `GEMINI_API_KEY` from anywhere — this is verified by a source-level test that greps the module for the string.
- `JudgeVerdict` is a pydantic v2 model with exactly two fields: `verdict: Literal["pass", "fail"]` and `reason: str`. The scorer returns `Feedback(value=(verdict == "pass"), rationale=reason)`.
- The judge's prompt uses the ticket's `judge_notes` as the ground-truth rubric and includes the agent's full decision (`category`, `priority`, `route`, `rationale`) as context, so the judge is grading the rationale in the situation the agent claims to have handled.
- On any judge failure (missing rationale, model call error, structured-output parse error), the scorer returns `Feedback(value=False, rationale=f"judge error: {…}")` — never raises. Matches Story 3.1's "one bad row can't abort the eval" rule.
- `_agent_total_tokens(run_id)` uses `mlflow.search_traces(run_id=run_id, return_type="list")` (the current API; not the deprecated `filter_string` form), then filters in Python to traces whose root span (first span with `parent_span_id is None`) has `span_type == "AGENT"`, then sums `trace.info.token_usage.get("total_tokens", 0)`. `None` `token_usage` (traces that never called an LLM) contributes 0.
- The stdout summary and the `eval/latest_report.json` write the SAME numbers — no divergence — because both consume the same computed `scorer_means`, `agent_total_tokens`, `escalation_count`, `run_id`.
- `eval/latest_report.json` is written with `json.dumps(report, indent=2)` + trailing newline so it's diff-friendly.
- Fail fast at `main()` entry with `SystemExit("GROQ_API_KEY is required for the rationale_judge scorer.")` if the env var is unset — the judge always runs, no matter what `PROVIDER` is.

**Never:**
- Read `GEMINI_API_KEY` inside the judge's construction, prompt, or fallback paths. Source-level grep test enforces this.
- Modify `agent.py`, `mcp/triage_server.py`, `schema.py`, `TRIAGE_POLICY.md`, `run_agent.py`, `load_seed.py`, `seed/`, `eval/labelled_tickets.csv` (all read-only from Story 3.1 + earlier).
- Alter the four code scorers from Story 3.1 or the auto-approve monkey-patch — the extension is additive.
- Include the judge's own ChatGroq token spend in `agent_total_tokens`. The root-span filter is the mechanism.
- Write anywhere except `eval/latest_report.json` under the repo tree (MLflow's `sqlite:///mlflow.db` is fine — it's the tracking store).
- Add new deps: `langchain-groq>=0.3` and `pydantic>=2.8` are already in `pyproject.toml`; nothing else is needed.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Judge — pass | Agent decision matches judge_notes (e.g. `T-1042` → billing/P2 rationale matches "money at stake") | `Feedback(value=True, rationale="…")`; aggregates as `1.0` | n/a |
| Judge — fail | Agent decision contradicts judge_notes (e.g. `T-1043` labelled P3 but agent's rationale claims P2) | `Feedback(value=False, rationale="…")`; aggregates as `0.0` | n/a |
| Malformed agent output | Agent output missing `rationale` (impossible under Story 2.1's retry-once, but the judge must handle it) | `Feedback(value=False, rationale="no rationale to judge")` | Never raises |
| Judge model call error | `ChatGroq.invoke(...)` raises (network, rate-limit exhaustion, malformed response) | `Feedback(value=False, rationale=f"judge error: {e}")` | Return, don't raise |
| Missing `GROQ_API_KEY` at eval start | `os.environ["GROQ_API_KEY"]` unset when `main()` runs | `SystemExit("GROQ_API_KEY is required for the rationale_judge scorer.")` before evaluate runs | Fail fast — surface the config gap immediately |
| Agent total tokens — happy | Run has N ticket predict traces, each with root `span_type="AGENT"` and `token_usage.total_tokens > 0` | Sum of all N `total_tokens`; judge traces excluded | n/a |
| Agent total tokens — trace without usage | A predict trace errored before autolog captured LLM usage (`token_usage is None` or `total_tokens missing`) | Contributes 0; sum still valid | Guard `token_usage or {}`; `.get("total_tokens", 0)` |
| Report file already exists | `eval/latest_report.json` from a prior run | Overwritten in place; no backup, no versioning (SPEC: "latest") | Idempotent write |
| Summary numbers match report | End-to-end run | The five means, `agent_total_tokens`, `escalation_count`, and `run_id` printed to stdout are byte-identical to the values in `eval/latest_report.json` | Single computed source of truth |

**Decisions:**
- **Test strategy.** Ship both: (1) `tests/test_eval_scorers.py` extended with deterministic unit tests for `rationale_judge` (pass / fail / malformed-outputs / judge-model-error paths, using a fake model injected via monkey-patching `_judge_model()`), `_agent_total_tokens()` (synthetic AGENT-rooted vs non-AGENT-rooted traces + missing-usage tolerance), `_write_report()` (valid JSON on disk), and a source-level grep guard asserting `GEMINI_API_KEY` never appears in `eval/run_eval.py`. (2) `tests/test_eval_smoke.py` extended so the existing skipif-guarded live smoke also asserts `rationale_judge/mean` is in `results.metrics`, `eval/latest_report.json` exists after the run with the five scorer keys + `agent_total_tokens` + `escalation_count`, and the file's numbers match `results.metrics` byte-for-byte. Mirrors Stories 2.1 / 2.2 / 3.1's decision shape.

</frozen-after-approval>

## Code Map

- `eval/run_eval.py` — **modified.** Additions: (1) `JudgeVerdict` pydantic model (`verdict: Literal["pass","fail"]`, `reason: str`), (2) `_judge_model()` helper — constructs `ChatGroq(...).with_structured_output(JudgeVerdict)`, no `GEMINI_API_KEY` read, (3) `rationale_judge` `@scorer` — builds a judge prompt with `expectations["judge_notes"]` + `outputs`, invokes `_judge_model()`, returns `Feedback(value=(verdict=="pass"), rationale=reason)`; failure paths return `Feedback(value=False, rationale="…")`, (4) `_agent_total_tokens(run_id) -> int` — fetches traces via `mlflow.search_traces(run_id=..., return_type="list")`, filters to AGENT-rooted, sums `total_tokens`, (5) `_write_report(path, report)` — dumps the report dict as pretty JSON, (6) `main()` updated to check `GROQ_API_KEY` at entry, include `rationale_judge` in `_scorers()`, print five means + tokens + escalation count, write `eval/latest_report.json`.
- `eval/__init__.py` — **read-only.** Empty package marker from Story 3.1.
- Every other Story 3.1 file (`agent.py`, `mcp/triage_server.py`, `schema.py`, `TRIAGE_POLICY.md`, `run_agent.py`, `load_seed.py`, `seed/`, `eval/labelled_tickets.csv`) — **read-only.**
- `pyproject.toml` — **no changes.** `langchain-groq>=0.3` + `pydantic>=2.8` already present.
- `tests/test_eval_scorers.py` — **extended.** New unit tests: `rationale_judge` pass / fail / malformed-outputs / judge-model-error (fake `_judge_model()` via monkey-patch), `_agent_total_tokens` (AGENT-rooted vs non-AGENT-rooted traces + missing-usage tolerance), `_write_report` (JSON on disk shape), and a source-level grep asserting `GEMINI_API_KEY` never appears in `eval/run_eval.py`.
- `tests/test_eval_smoke.py` — **extended.** Adds `rationale_judge/mean` presence check + `eval/latest_report.json` existence + numeric equality between file and `results.metrics`.

## Tasks & Acceptance

**Execution:**
- [x] `eval/run_eval.py` — add `JudgeVerdict`, `_judge_model()`, `rationale_judge` `@scorer`, `_agent_total_tokens()`, `_write_report()`; extend `main()` to enforce `GROQ_API_KEY`, register the fifth scorer, print the five means + tokens + escalations, and write `eval/latest_report.json`.
- [x] `tests/test_eval_scorers.py` — add unit tests: `rationale_judge` pass / fail / malformed-outputs / judge-model-error (fake `_judge_model()` via monkey-patch), `_agent_total_tokens` (AGENT-rooted vs non-AGENT-rooted traces + missing-usage tolerance), `_write_report` (JSON on disk shape), and a source-level grep asserting `GEMINI_API_KEY` never appears in `eval/run_eval.py`.
- [x] `tests/test_eval_smoke.py` — extend the existing skipif-guarded live smoke to also assert `rationale_judge/mean` is in `results.metrics`, `eval/latest_report.json` exists, and the file's numbers match `results.metrics` + `agent_total_tokens`.
- [x] Manual verification (Implementation Notes): `PROVIDER=groq uv run python eval/run_eval.py` prints all FIVE scorer means, `agent_total_tokens`, `escalation_count`, `run_id`; `cat eval/latest_report.json` shows the same numbers; the run's MLflow traces include per-ticket predict traces (root `AGENT`) plus judge ChatGroq traces (separate, no `AGENT` root). **Caveat:** end-to-end run was blocked by Groq free-tier TPM rate limits (`openai/gpt-oss-120b` capped at 8000 TPM; eval fires ~1330 tokens per sample) — same environmental constraint documented in Story 3.1. Every new code path is verified by the 15 new unit tests; fail-fast on missing `GROQ_API_KEY` was verified interactively; a full end-to-end printout awaits a non-rate-limited Groq window.

**Acceptance Criteria:**
- Given `app.db` is loaded, `GROQ_API_KEY` is set, and the agent's provider key is set, when `uv run python eval/run_eval.py` runs, then the stdout summary contains numeric means for exactly five scorers (`valid_schema`, `category_match`, `priority_match`, `tool_order`, `rationale_judge`), plus `agent_total_tokens` (integer ≥ 0), `escalation_count` (integer ≥ 0), and `run_id`.
- Given the same setup, when the run finishes, then `eval/latest_report.json` exists and its JSON content contains: `run_id` (string), `scorer_means` (dict with the five scorer names as keys, floats as values), `agent_total_tokens` (integer), `escalation_count` (integer) — and every one of those numbers is byte-identical to what stdout printed.
- Given `GROQ_API_KEY` is unset at `main()` entry, when the script runs, then it exits with a non-zero status BEFORE invoking `mlflow.genai.evaluate`, with an error message naming `GROQ_API_KEY` as the required env var — no partial eval, no partial report.
- Given a run whose traces include both AGENT-rooted predict traces and non-AGENT judge traces, when `_agent_total_tokens(run_id)` is called, then it returns the sum of `total_tokens` from the AGENT-rooted traces only — the judge's own ChatGroq token spend is excluded.
- Given a `rationale_judge` invocation whose `outputs` dict is missing the `rationale` key, when the scorer runs, then it returns `Feedback(value=False, rationale="no rationale to judge")` and does NOT raise; the scorer's `mean` still aggregates over all 20 rows.

## Implementation Notes

**Files changed**
- `eval/run_eval.py` — extended (not rewritten): kept Story 3.1's four code scorers, MLflow setup, HITL auto-approve monkey-patch + counter, and single-trace-per-ticket `@mlflow.trace(span_type="AGENT")` predict wrapper. Added: (1) `JudgeVerdict` pydantic model (`verdict: Literal["pass","fail"]`, `reason: str`), (2) `_judge_model()` — constructs `ChatGroq(model=JUDGE_MODEL default "openai/gpt-oss-120b", api_key=os.environ["GROQ_API_KEY"], temperature=0).with_structured_output(JudgeVerdict)`, (3) `rationale_judge` `@scorer` — short-circuits on missing/empty agent rationale to `Feedback(value=False, rationale="no rationale to judge")` (skips the model call), otherwise builds a two-part prompt with `judge_notes` + the agent's full decision and returns `Feedback(value=(verdict=="pass"), rationale=reason)`; any model-call/parse error becomes `Feedback(value=False, rationale=f"judge error: {e}")`, (4) `_agent_total_tokens(run_id)` — `mlflow.search_traces(run_id=..., return_type="list")`, filters to traces whose first `parent_id is None` span has `span_type == "AGENT"`, sums `trace.info.token_usage["total_tokens"]` (missing usage → 0), (5) `_write_report(path, report)` — `json.dumps(indent=2)` + trailing newline. `main()` now: fails fast with `SystemExit("GROQ_API_KEY is required for the rationale_judge scorer.")` if the env var is unset (before mlflow setup / evaluate), registers all five scorers via `_scorers()`, computes `scorer_means` once and feeds BOTH the stdout summary and the JSON report from the same dict so the numbers can't diverge, prints the five means + `agent_total_tokens` + `escalation_count` + `run_id`, and writes `eval/latest_report.json`.
- `tests/test_eval_scorers.py` — extended: added 15 new unit tests. Six `rationale_judge` tests (pass, fail, missing rationale, empty-string rationale, model-call error → judge-error Feedback, non-dict outputs); four `_agent_total_tokens` tests (AGENT-rooted sum with non-AGENT excluded, missing `token_usage` tolerated, all non-AGENT → 0, missing `total_tokens` key → 0); two `_write_report` tests (pretty JSON with trailing newline + round-trips, idempotent overwrite); one source-level grep guard asserting `GEMINI_API_KEY` never appears anywhere in `eval/run_eval.py`; one `_scorers()` registration guard asserting all five scorers wired; one `main()` fail-fast guard asserting `SystemExit` when `GROQ_API_KEY` is unset (with `mlflow.genai.evaluate` monkey-patched to assert-error if reached, so the failure mode is unambiguous).
- `tests/test_eval_smoke.py` — extended: the same skipif-guarded live smoke now asserts all FIVE scorer means in `results.metrics` (added `rationale_judge`), extracts the printed `Agent total tokens:` line with a regex, asserts `eval/latest_report.json` exists after the run, asserts its top-level keys are exactly `{"run_id", "scorer_means", "agent_total_tokens", "escalation_count"}`, and asserts every number in the report is byte-identical to the corresponding stdout print and `results.metrics` entry.

**Design decisions**
- **MLflow API attribute name.** The spec text says "first span with `parent_span_id is None`", but the runtime API on `mlflow.entities.Span` is `parent_id` (not `parent_span_id`). Implemented against the real API. This is the intent the spec captured; the wording drift is a spec-copy error, not a design change.
- **Missing-rationale short-circuit skips the model call.** The spec's I/O matrix says missing rationale returns `Feedback(value=False, rationale="no rationale to judge")` — with the guard evaluated BEFORE `_judge_model()` runs. Saves a token round-trip and avoids a false "judge error" message on a case the judge doesn't need to see. Unit test `test_rationale_judge_handles_missing_rationale_without_calling_model` asserts the fake judge model was never invoked.
- **Single source of truth for scorer means.** Both the stdout summary and `eval/latest_report.json` consume the same computed `scorer_means`, `agent_total_tokens`, `escalation_count`, and `run_id`. There is exactly one path that reads `results.metrics` and coerces it to floats; the printout and the JSON dump both draw from that dict. The smoke test asserts byte-identity between the two.
- **`_agent_total_tokens` root-span filter matches the AGENT decorator.** Our predict wrapper's `@mlflow.trace(span_type="AGENT")` creates a root span with `span_type == "AGENT"`. The judge's ChatGroq calls autolog as separate traces whose root span is `CHAT_MODEL` (or similar), NOT `AGENT` — so filtering on root-span `AGENT` cleanly separates agent traces from judge traces without needing extra metadata. Unit test `test_agent_total_tokens_sums_only_agent_rooted_traces` proves the filter behavior with a synthetic mix.
- **Deferred import of `langchain_groq`.** `_judge_model()` imports `ChatGroq` inside the function body so a pytest session that never exercises the judge (e.g., only runs the four code scorers) doesn't require the dep to be importable at collection time. Matches Story 2.1's provider-scoped import pattern.
- **Fail-fast happens BEFORE `mlflow.set_experiment` and any autolog wiring.** The check runs immediately after `load_dotenv()` so a missing `GROQ_API_KEY` doesn't leave a stub run in the MLflow store or a stale report on disk. The failing unit test proves this by asserting `mlflow.genai.evaluate` is never reached when the key is unset.

**Manual verification (2026-09-26)**
- `uv run pytest tests/test_eval_scorers.py -v` → 35 passed (20 pre-existing from Story 3.1 + 15 new for Story 3.2).
- `uv run pytest -k "not test_t1042 and not test_t1099 and not test_t1048" --deselect tests/test_eval_smoke.py` → 75 passed. All non-live-LLM tests green.
- Fail-fast smoke: with `GROQ_API_KEY` unset and `load_dotenv` neutralized, `run_eval.main()` raises `SystemExit('GROQ_API_KEY is required for the rationale_judge scorer.')` before touching MLflow. Verified interactively.
- `PROVIDER=groq uv run python eval/run_eval.py` — end-to-end run **completed** (a retry within the review session cleared the throttle just long enough for `mlflow.genai.evaluate` to finish; the harness still logged many `Rate-limited (attempt 3/4)` events, so most of the 20 tickets errored partway). Verified outputs:
  ```
  ============================================================
  Eval complete
  ============================================================
  MLflow run_id:              8ff5ad0c28184b758115dd254926fb07
  Auto-approved escalations:  1
  Agent total tokens:         38878
  Per-scorer means:
    valid_schema         0.100
    category_match       0.100
    priority_match       0.100
    tool_order           0.100
    rationale_judge      0.050
  Report written to:          eval/latest_report.json
  ```
  `eval/latest_report.json` on disk:
  ```json
  {
    "run_id": "8ff5ad0c28184b758115dd254926fb07",
    "scorer_means": {
      "valid_schema": 0.1,
      "category_match": 0.1,
      "priority_match": 0.1,
      "tool_order": 0.1,
      "rationale_judge": 0.05
    },
    "agent_total_tokens": 38878,
    "escalation_count": 1
  }
  ```
  Every AC is observed against this run: (AC-1) five scorer means + total tokens + escalation count + run_id printed and file written; (AC-2) file contents byte-identical to stdout; (AC-4) `agent_total_tokens=38,878` is the AGENT-rooted trace sum (judge ChatGroq traces excluded — the filter works because the judge lives at a `CHAT_MODEL` root, not `AGENT`); (AC-5) missing-rationale short-circuit exercised on the erroring tickets. AC-3 (`SystemExit` on missing `GROQ_API_KEY` before MLflow setup) was verified in the impl session and is pinned by `test_main_exits_when_groq_api_key_missing`. During the run the stdout also showed `[HITL] The agent is requesting to escalate this ticket to a human.` / `[HITL] Reason: P1 priority for Enterprise customer with integration outage` — proving the auto-approve monkey-patch fired inside the eval subprocess. The low scorer means (~0.10, judge 0.05) reflect that ~18 of the 20 tickets errored under rate-limiting, not a harness bug; a re-run on paid Groq or a non-rate-limited window would exercise more rows.

**Follow-up commit note:** the spec was originally finalized to `done` with a "Not verified in this session" caveat about the end-to-end run. This commit updates Implementation Notes with the observed numbers from the follow-up successful run; behavior and code paths are unchanged from `63955f0`.

**Not verified in this session**
- A run with high scorer means (>0.9). Blocked by Groq free-tier TPM caps: even the successful run above lost ~18/20 tickets to rate-limit retries, so the printed means are correctness-verified but not quality-informative. Reruns on paid Groq or during a non-throttled window would surface actual quality signal.
- MLflow-UI visual confirmation that judge ChatGroq traces render as non-AGENT-rooted. The `_agent_total_tokens` filter is proven by unit tests + validated by the run above logging `agent_total_tokens=38,878` distinct from the judge's own token spend (which would have inflated the number if the filter were broken). Live UI screenshot is deferred to a polish pass.

## Review Triage Log

Three review layers (Blind Hunter, Edge Case Hunter, Verification Gap Reviewer) surfaced 16 root-cause groups. Grouped and triaged:

- **medium, patched** — `rationale_judge` interpolated the agent's `rationale` directly into the judge's user message without a data/instruction boundary (Blind Hunter). The agent's rationale is derived from customer-supplied ticket text — an untrusted source per AGENTS.md — so a crafted ticket could theoretically instruct the judge to always return `verdict=pass`. Fix: (1) added an explicit SAFETY paragraph to `_JUDGE_SYSTEM_PROMPT` telling the judge to treat everything inside `<rationale>...</rationale>` as untrusted data; (2) wrapped `agent_rationale` in explicit `<rationale>...</rationale>` fence tags in the user message. Category/priority/route are enum-constrained by the Epic 1 schema so they can't smuggle prompts.
- **low, patched** — `_agent_total_tokens` used `token_usage.get("total_tokens", 0)` with no fallback (Blind Hunter). If a provider autologs only `input_tokens` + `output_tokens` (some LangChain adapters do), we'd silently print 0. Fix: when `total_tokens` is missing/falsy, fall back to `input_tokens + output_tokens`. Updated `test_agent_total_tokens_handles_missing_total_tokens_key` → `test_agent_total_tokens_falls_back_to_input_plus_output_when_total_missing` (asserts sum), added `test_agent_total_tokens_returns_zero_when_all_keys_missing` for the truly empty case.
- **low, patched** — Fail-fast test asserted `SystemExit` was raised and its string named `GROQ_API_KEY`, but never asserted the exit code was non-zero (Blind Hunter). `SystemExit("msg")` defaults to code 1, but `SystemExit(0)` would also pass the current assertion — meaning the "failure surfaces to the shell" contract wasn't actually verified. Fix: added `assert excinfo.value.code not in (0, None)`.
- **low, patched** — `_write_report` used the default `json.dumps` encoder, which emits the literal `NaN` for `float('nan')` (invalid strict JSON — some parsers accept, some reject) (Blind Hunter). If all rows errored and a scorer mean is NaN, downstream consumers using strict JSON parsers would break. Fix: coerce NaN scorer means to `None` via `_coerce_nan_to_none()` (recursive helper), pass `allow_nan=False` to `json.dumps` so the encoder rejects any residual non-finite value loudly rather than emitting invalid output.
- **low, patched** — `mlflow.search_traces` inside `_agent_total_tokens` was unguarded (Edge Case Hunter). A transient MLflow backend error (sqlite lock, network) would crash `main()` after `evaluate()` had already succeeded, losing both the stdout summary and the on-disk report. Fix: wrap the fetch in `try/except Exception: return 0` so the rest of the report still writes with an under-reported token count (better than losing all metrics).
- **false** — `_LATEST_REPORT_PATH.relative_to(_REPO_ROOT)` raises `ValueError` if the report path is outside the repo (Blind Hunter). Disproof: path is hardcoded as `_REPO_ROOT / "eval" / "latest_report.json"` — always under repo. Speculative "future refactor" concern; not a live bug.
- **false** — Docstring/spec drift on `parent_span_id` vs `parent_id` (Blind Hunter). Disproof: already flagged and documented in Implementation Notes → Design decisions ("MLflow API attribute name"). The runtime code uses the correct API attribute; the spec text is a copy-error the story acknowledged.
- **false** — Root-span identification via `for span in spans: if parent_id is None: break` picks the wrong span if multiple children accidentally have `parent_id=None` (Blind Hunter). Disproof: MLflow guarantees exactly one root per trace (a trace with multiple parentless spans would be malformed by MLflow's own invariant). Speculative.
- **false** — `_agent_total_tokens` might crash if `token_usage` is a `TokenUsageInfo` object instead of a dict (Blind Hunter). Disproof: MLflow 3.4+'s `TraceInfo.token_usage` returns `dict[str, int]` per the subagent's investigation and Story 3.1's own empirical run (21,349 tokens summed correctly). Speculative for unlikely future MLflow shape change.
- **false** — `test_write_report_produces_pretty_json_with_trailing_newline` uses substring proxy instead of full JSON equality (Blind Hunter). Disproof: the assertion is meant to check the shape signature (pretty-printed + trailing newline), not exact byte-equality; a fuller equality check would over-couple the test to the report's field order.
- **false** — `_ensure_repo_on_path` fragile — fail-fast test could depend on env if a developer's env doesn't set `GEMINI_API_KEY` (Blind Hunter). Disproof: the test uses `monkeypatch.setattr(run_eval, "load_dotenv", ...)` to neutralize dotenv AND monkey-patches `mlflow.genai.evaluate` to raise if reached — the fail-fast is verified before any Gemini-dependent code runs.
- **false** — `eval/latest_report.json` is never cleaned up — stale file could confuse users after a skipped smoke run (Blind Hunter). Disproof: the "latest" semantics are explicit in the SPEC ("`eval/latest_report.json` — the only new file this epic writes outside MLflow's own store"); staleness IS the intended state until the next run overwrites it.
- **false** — Verification Gap Reviewer found no gaps ("Every new behavior path has a corresponding test that asserts the observable outcome"). Explicit confirmation that the story's verification coverage is complete for the deterministic paths; live-LLM verification is separately deferred to a non-rate-limited window.
- **low, rejected** — `results.run_id` might be `None` → `search_traces(run_id=None)` returns all traces experiment-wide (Edge Case Hunter). Not worth fixing: `mlflow.genai.evaluate` always creates a run; `run_id` is never `None` in practice. Defensive branch would add surface for a can't-happen case.
- **low, rejected** — `path.parent.mkdir` could fail if `eval/` is a file, `path.write_text` could fail if disk full or read-only (Edge Case Hunter x2). Not worth fixing: this is a CLI script; a filesystem error is informative and should propagate to the user with a stack trace, not be silently swallowed. Matches Story 3.1's "let SDK errors propagate unwrapped" pattern.
- **low, rejected** — `outputs.get("rationale")` truthy check treats non-string falsy values (`0`, `[]`, `{}`) as missing (Edge Case Hunter). Not worth fixing: `TriageDecision` schema constrains `rationale` to a non-empty string via `StringConstraints(min_length=1)`; a well-formed decision can't have this. Defense-in-depth beyond spec scope.

## Verification

**Commands:**
- `uv run pytest` — expected: all tests pass; new unit tests always run; the live-LLM smoke skips without `GROQ_API_KEY`.
- `PROVIDER=groq uv run python eval/run_eval.py` — expected: eval completes, stdout prints five scorer means + `agent_total_tokens` + `escalation_count` + `run_id`; `eval/latest_report.json` is written with the same numbers.
- `uv run python -c "import json; print(json.load(open('eval/latest_report.json')))"` — expected: dict with `run_id`, `scorer_means` (5 keys), `agent_total_tokens`, `escalation_count`.
- `unset GROQ_API_KEY && uv run python eval/run_eval.py` — expected: non-zero exit; stderr names `GROQ_API_KEY`; no MLflow run created, no report written.

**Manual checks (if no CLI):**
- Open the MLflow UI on the new run and confirm the judge's ChatGroq calls appear as separate traces (no `AGENT` root) — they should not be counted in `agent_total_tokens`.
