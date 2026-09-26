# Epic 3 Context: measure the agent

<!-- Compiled from planning artifacts. Edit freely. Regenerate with compile-epic-context if planning docs change. -->

## Goal

Close the loop on the triage agent: run it over 20 hand-labelled tickets and produce a measured baseline — five per-ticket scorers plus token spend and escalation count — so attendees leave with numbers, not just a working agent. Epic 1 shipped the schema + data foundation; Epic 2 shipped the agent + HITL escalation; Epic 3 is the eval harness that turns those into a repeatable measurement.

No traditional planning artifacts exist (no PRD, architecture, UX doc). The full contract lives in `_bmad-output/specs/spec-epic-3/SPEC.md` plus its companions `eval/labelled_tickets.csv`, `mcp/triage_server.py`, `INTENT.md`, and `spec-epic-2/SPEC.md`. This file distills the parts a story-level developer needs.

## Stories

- Story 1: The eval run and the four code scorers (CAP-1..5, CAP-8 — run harness, `valid_schema`, `category_match`, `priority_match`, `tool_order`, auto-approve escalation)
- Story 2: The rationale judge and the report (CAP-6, CAP-7 — `rationale_judge` on ChatGroq/JUDGE_MODEL/GROQ_API_KEY, print/write per-scorer means + total tokens + escalation count to `eval/latest_report.json`)

## Requirements & Constraints

- The eval harness is built with `mlflow.genai.evaluate` — a hand-rolled scoring loop is not acceptable.
- One MLflow run in the `triage-agent` experiment on `sqlite:///mlflow.db` per invocation of `eval/run_eval.py`.
- Every one of the 20 tickets in `eval/labelled_tickets.csv` is scored by every enabled scorer; individual failures do not abort the run.
- The agent's behaviour is unchanged — `run_eval.py` calls `agent.triage()` as-is; no edits to prompts, policy handling, or `agent.py` internals.
- `rationale_judge` (Story 2) always uses `ChatGroq` (`JUDGE_MODEL` default `openai/gpt-oss-120b`, `GROQ_API_KEY`) and never reads `GEMINI_API_KEY` — so judge calls never compete with the agent's Gemini quota.
- The four code scorers (`valid_schema`, `category_match`, `priority_match`, `tool_order`) run locally — no network calls. Only the agent's own model call and `rationale_judge`'s Groq call cross the network.
- `tool_order` reads each prediction's MLflow trace to verify `get_ticket`'s span starts before `get_customer_history`'s span.
- **Each ticket's prediction must live in a single trace.** An approved escalation resumes the agent in a second `ainvoke` call; without wrapping the predict function in one trace (`@mlflow.trace` on the wrapper), that second call autologs as a separate trace and `tool_order` can't see both spans.
- **Auto-approve applies only inside the eval scope.** `uv run python run_agent.py <ticket>` still pauses for a human's yes/no, unchanged from Story 2.2. The eval must intercept the HITL prompt without modifying `agent.py`.
- `eval/labelled_tickets.csv` and `TRIAGE_POLICY.md` are read-only.
- `eval/latest_report.json` is the only new file this epic writes outside MLflow's own store.
- No `GEMINI_API_KEY` in `rationale_judge` — verify by structure, not just intent.

## Technical Decisions

- Python 3.12+, packages managed with uv. Deps already in `pyproject.toml`: `mlflow>=3.4`, `langchain-groq>=0.3`. No new deps expected.
- The predict function wraps `asyncio.run(triage(ticket_id))` in a `@mlflow.trace`-decorated wrapper so both the initial `ainvoke` and any HITL-resume `ainvoke` nest under a single parent span. That parent span's trace_id is what scorers use to read span order.
- Auto-approve HITL inside the eval: monkey-patch `builtins.input` in `eval/run_eval.py` to return `"yes"`. This is a scope-local override — a normal `uv run python run_agent.py` invocation remains interactive. Rationale: `agent.py` stays untouched (per the "unchanged behaviour" constraint), and the `input()` boundary is the single choke point the agent uses to gate escalation.
- Scorer results align with the four 0/1 code scorers by convention: `pass=1`, `fail=0` for the judge scorer (Story 2). Mean over all 20 tickets = pass rate.
- The MLflow setup (`set_tracking_uri("sqlite:///mlflow.db")`, `set_experiment("triage-agent")`, `mlflow.langchain.autolog()`) is repeated at the top of `eval/run_eval.py` — same three-line preamble as `run_agent.py`.

## Cross-Story Dependencies

- Story 1 must land before Story 2: Story 2 adds `rationale_judge` alongside the four code scorers Story 1 wires up, and extends the reporting path Story 1 stubs (Story 1's reporting can be limited to the four code scorers + tokens + escalation count).
- Both stories depend on Epic 2's agent (`agent.triage`, `run_agent.py`'s MLflow setup, the `HumanInTheLoopMiddleware` wiring in `agent.py`) and Epic 1's `schema.TriageDecision`.
- The `triage-agent` MLflow experiment is shared with Epic 2 (`run_agent.py` traces land there too); Story 3.1's eval adds one run per invocation to the same experiment.
