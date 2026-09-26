---
title: 'Human-gated escalation'
type: 'feature'
created: '2026-09-26'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
context: [_bmad-output/implementation-artifacts/epic-2-context.md, TRIAGE_POLICY.md, _bmad-output/specs/spec-epic-2/stories/1-triage-agent.md]
baseline_commit: 'e1a716e3959442bb48a0da803a0963d2b6605d3c'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Story 2.1 ships a triage agent that returns a decision autonomously — including on P1 tickets from Enterprise customers, where `TRIAGE_POLICY.md` explicitly requires a human's approval before anything escalates. There is no `escalate_to_human` tool yet, no pause, and no mechanism to distinguish an approved escalation from a rejected one.

**Approach:** Add a local `escalate_to_human` tool to `agent.py` (a `@tool`-decorated function that only executes when approved) and wire LangChain's `HumanInTheLoopMiddleware` into `create_agent` with `interrupt_on={"escalate_to_human": {"allowed_decisions": ["approve", "reject"]}}` — MCP tools (`get_ticket`, `get_customer_history`) remain uninterrupted. Add a `MemorySaver` checkpointer (HITL requires one) and a per-invocation `thread_id`. Extend `triage()` with an interrupt-handling loop: when the first `ainvoke` returns `"__interrupt__"` in its result, print the pending escalation reason to stderr, read a `yes`/`no` from stdin, then resume with `Command(resume={"decisions": [{"type": "approve" | "reject"}]})`. Update the system prompt so the model calls `escalate_to_human(reason=...)` exactly when the policy's escalation rule fires (P1 AND Enterprise plan). `run_agent.py` is unchanged.

## Boundaries & Constraints

**Always:**
- Use LangChain's `HumanInTheLoopMiddleware` (from `langchain.agents.middleware`) — not a hand-rolled interrupt/resume scheme.
- Gate ONLY `escalate_to_human`; `get_ticket` and `get_customer_history` remain uninterrupted.
- `escalate_to_human` is a local `@tool` in `agent.py` (or a helper module imported from `agent.py`); it MUST NOT live in `mcp/triage_server.py` (that file is read-only).
- Prompt for approval on stdin with a plain `input("Escalate to human? [yes/no]: ")`; a case-insensitive `"yes"` (only) approves, everything else rejects. The prompt happens INSIDE `triage()` — `run_agent.py` is unchanged.
- Use `langgraph.checkpoint.memory.MemorySaver` as the `checkpointer` and a fresh `thread_id` per `triage()` call (derived from `ticket_id` + a UUID so retries do not collide).
- Preserve the retry-once behavior from Story 2.1: `_invoke_with_retry` still owns structured-output validation; HITL sits alongside it, on the tool-call path.
- Preserve everything read-only from Story 2.1: `mcp/triage_server.py`, `schema.py`, `TRIAGE_POLICY.md`, `run_agent.py`, `seed/`.
- The final printed decision continues to be `TriageDecision.model_dump(mode="json")` — 4 fields, no `escalated` flag. Escalation presence/absence is visible in the MLflow trace via the `escalate_to_human` span (or its absence).

**Never:**
- Modify `run_agent.py` (Story 2.1 explicitly left this file unchanged; keeping it that way makes the HITL flow a `triage()`-internal concern, symmetric with `_invoke_with_retry`).
- Auto-approve escalation. A `"yes"` (case-insensitive, exact) is the only affirmative; anything else — including empty, non-y responses, EOF, KeyboardInterrupt — is treated as reject.
- Escalate on non-P1 or non-Enterprise cases. The model is instructed via the policy; a false-positive `escalate_to_human` call still pauses (correct behavior), but the model should not call it outside the P1+Enterprise rule.
- Add an `escalated` field to `TriageDecision`. The schema is fixed by Story 1.1 (Epic 1) and referenced by Epic 3.
- Persist checkpointer state across process runs — `MemorySaver` is per-process; a new `triage()` call starts a fresh state.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| P1 + Enterprise + approve | `triage("T-1048")`, C-05 = Hooli/Enterprise/4 open; stdin `"yes"` | `escalate_to_human` tool is invoked (visible as an MLflow span); final decision is `bug`/`P1`/`bug-team` (route by category); JSON printed | n/a |
| P1 + Enterprise + reject | `triage("T-1048")`, stdin `"no"` | `escalate_to_human` call is cancelled by middleware; agent completes the loop without escalation; final decision is `bug`/`P1`/`bug-team` (route by category); JSON printed | n/a |
| Non-P1 (workshop happy path) | `triage("T-1042")`, C-77 = Northwind/Enterprise/2 open; money-at-stake ticket | Agent completes without ever calling `escalate_to_human`; no interrupt; JSON printout matches Story 2.1's CAP-1 output | n/a |
| Prompt injection (Story 2.1 CAP-6) | `triage("T-1099")` | Still `bug`/`P4`; no interrupt (P4, not P1); Story 2.1 behavior preserved | n/a |
| Empty / non-yes input | `triage("T-1048")`, stdin `""` or `"y"` or `"maybe"` | Treated as reject; escalation cancelled | n/a |
| EOF on stdin | `triage("T-1048")`, stdin closed (EOF) | Treated as reject; escalation cancelled | Catch `EOFError` inside the prompt call; log a warning; treat as reject |
| Retry-once during HITL run | Model returns malformed structured output on first attempt of an escalated run | `_invoke_with_retry`'s retry path runs; if the retry also produces an escalation, the HITL prompt fires again | If retry also fails schema, raise `RuntimeError` as in Story 2.1 |

**Decisions:**
- **Test strategy.** Ship both: (1) one live-LLM smoke test on T-1048 with monkey-patched `input()` returning `"no"` (skipif on missing key), asserting the run completes without exception and returns a schema-shaped dict; (2) deterministic unit tests using a `_FakeAgent` stub that returns an `"__interrupt__"` result on first `ainvoke` and a valid `structured_response` on the resumed call — assert `input()` was called, the resume `Command` was sent with the correct `decisions[0].type` (`approve` for `"yes"`, `reject` otherwise), and the final decision returned. Mirrors Story 2.1's decision so `tests/test_agent.py` grows consistently.

</frozen-after-approval>

## Code Map

- `agent.py` — **modified.** Add `escalate_to_human` as a `@tool`-decorated function (imported from `langchain_core.tools`). Extend `_build_system_prompt()` with a paragraph telling the model: "When the policy's escalation rule fires (final priority P1 AND Enterprise plan), call `escalate_to_human(reason=<one sentence>)` before finalizing the decision." Extend `triage()` to: (1) pass `middleware=[HumanInTheLoopMiddleware(interrupt_on={"escalate_to_human": {"allowed_decisions": ["approve", "reject"]}})]` and `checkpointer=MemorySaver()` to `create_agent`; (2) build a `config = {"configurable": {"thread_id": f"{ticket_id}-{uuid.uuid4().hex[:8]}"}}` and pass it to every `ainvoke`; (3) add `_handle_interrupts(agent, result, config)` — a helper that loops while `"__interrupt__"` is in the result, prompts stdin, resumes with `Command(resume={"decisions": [{"type": ...}]})`, and returns the final result. `_invoke_with_retry` is updated to invoke through `_handle_interrupts` so retries survive HITL.
- `run_agent.py` — **read-only.** No changes.
- `mcp/triage_server.py`, `schema.py`, `TRIAGE_POLICY.md`, `seed/`, `load_seed.py` — **read-only.**
- `tests/test_agent.py` — **extended.** One live-LLM smoke test on T-1048 (skipif on missing key) with monkey-patched `input()` returning `"no"`, plus deterministic unit tests for the interrupt-handling loop using a `_FakeAgent` stub (approve + reject paths). Existing Story 2.1 tests continue to pass.
- `pyproject.toml` — **no changes.** `langchain>=1.0` already provides `HumanInTheLoopMiddleware` and `MemorySaver`.

## Tasks & Acceptance

**Execution:**
- [x] `agent.py` — add `escalate_to_human` `@tool` function; add `_handle_interrupts()` helper; update `_build_system_prompt()`; wire `HumanInTheLoopMiddleware` + `MemorySaver` + `thread_id` into `create_agent`; thread the middleware through `_invoke_with_retry` so retries handle re-interrupts.
- [x] `tests/test_agent.py` — add one live-LLM smoke test on T-1048 with monkey-patched `input()` returning `"no"` (skipif on missing key), plus deterministic unit tests using a `_FakeAgent` for approve/reject paths of the interrupt-handling loop.
- [x] Manual verification (recorded in Implementation Notes): `uv run python run_agent.py T-1048` prompts for approval on stdin; answering `"yes"` completes the run with `escalate_to_human` visible in the MLflow trace; answering `"no"` completes the run without the tool span. `uv run python run_agent.py T-1042` (non-Enterprise-bump, non-P1) completes without a prompt and matches Story 2.1's CAP-1 output.

**Acceptance Criteria:**
- Given `app.db` is loaded and a live-LLM key is set, when `uv run python run_agent.py T-1048` runs and the operator answers `"yes"` at the terminal prompt, then the MLflow trace for that run contains an `escalate_to_human` tool span (approved), and the printed JSON contains valid `category`, `priority`, `route`, `rationale` fields.
- Given the same setup, when `uv run python run_agent.py T-1048` runs and the operator answers `"no"`, then no `escalate_to_human` tool span appears in the run's MLflow trace (the middleware cancels the call), and the printed JSON still contains a valid `TriageDecision` shape.
- Given `PROVIDER=groq` and `GROQ_API_KEY` set, when `uv run python run_agent.py T-1042` runs, then it completes with a valid `TriageDecision` shape and NO interrupt is raised (Story 2.1's CAP-1 output is preserved).
- Given a stubbed agent whose first `ainvoke` returns an `"__interrupt__"`, when the interrupt-handling loop is exercised via a monkey-patched `input()` returning `"yes"`, then a `Command(resume=...)` with `decisions[0].type == "approve"` is sent on the next `ainvoke`; when `input()` returns `"no"` (or anything non-`yes`), the resume payload's `decisions[0].type == "reject"`.

## Implementation Notes

**Files changed**
- `agent.py` — extended with (a) the local `escalate_to_human` `@tool` function, (b) an updated `_SAFETY_PREAMBLE` teaching the model when to call it (P1 AND Enterprise, before returning the final decision, never in other cases, don't change the decision based on the human's answer), (c) `_prompt_for_escalation_decision()` and `_handle_interrupts()` helpers that print the pending reason to stderr and read a single `yes`/`no` from stdin (EOF / KeyboardInterrupt → reject), (d) `_invoke_with_retry(agent, user_message, config)` now takes the LangGraph `config` and threads it through both the initial invocation and the retry so re-interrupts still get the prompt, and (e) `triage()` builds a `HumanInTheLoopMiddleware(interrupt_on={"escalate_to_human": {"allowed_decisions": ["approve", "reject"]}})`, a fresh `MemorySaver`, and a per-invocation `thread_id = f"{ticket_id}-{uuid.uuid4().hex[:8]}"`.
- `tests/test_agent.py` — added: one live-LLM smoke test `test_t1048_p1_enterprise_reject_escalation` (skipif on missing key, monkey-patches `input()` to return `"no"`, records the calls, and asserts `bug`/`P1`/`bug-team` — closes the "HITL wiring silently regressed" gap Verification Gap Reviewer flagged); and deterministic unit tests using a `_FakeAgent` stub — approve path, reject path, five parametrized `test_hitl_only_exact_yes_approves` cases, EOF-treated-as-reject, KeyboardInterrupt-treated-as-reject (added during review), `_handle_interrupts` bounded-loop guard (added during review), `_invoke_with_retry` forwards `config` to every `ainvoke` (added during review), and `escalate_to_human` is a `BaseTool` named correctly. Existing retry-once tests updated to pass the new `config` positional arg to `_invoke_with_retry`.

**HITL design decisions**
- Only `escalate_to_human` is on the middleware's `interrupt_on` map; the MCP lookup tools (`get_ticket`, `get_customer_history`) run without pause, per the spec.
- The prompt lives inside `triage()` (via `_handle_interrupts`), so `run_agent.py` is unchanged — matches the "make HITL a `triage()`-internal concern, symmetric with `_invoke_with_retry`" reasoning in the spec.
- Prompt writes to `stderr` so the JSON payload printed on `stdout` by `run_agent.py` stays clean for downstream tooling.
- Only case-insensitive, trimmed `"yes"` approves. EOF and KeyboardInterrupt are caught and treated as reject; a warning is logged to stderr.
- `MemorySaver` is per-process (a fresh `triage()` call re-instantiates it); the `thread_id` combines `ticket_id` with a UUID slice so retries of the same ticket in the same process don't collide.
- The interrupt payload from `HumanInTheLoopMiddleware` uses `action_requests` (plural) in LangChain 1.x, not `action_request` (singular). The prompt formatter handles both keys defensively so the reason line stays friendly if the shape changes again upstream.

**Manual verification (2026-09-26, PROVIDER=groq because the Gemini free-tier daily quota was exhausted; same setup Story 2.1 fell back to)**
- `echo "no" | uv run python run_agent.py T-1048` → prompt appeared on stderr with the extracted reason (`[HITL] Reason: P1 bug on Enterprise customer with multiple open tickets`), the escalation was cancelled, and stdout printed a valid `TriageDecision` JSON (`bug`/`P1`/`bug-team`).
- `echo "yes" | uv run python run_agent.py T-1048` → prompt appeared, escalation was approved, stdout printed a valid `TriageDecision` JSON.
- `uv run python run_agent.py T-1042` → no prompt at all; output matched Story 2.1's CAP-1 shape (`billing`/`P2`/`billing-team`) with rationale "Enterprise rule not triggered (open tickets <3)".
- MLflow trace inspection (via `mlflow.get_trace(...)`) on the four runs above confirmed the acceptance-criterion contract:
  - Approved T-1048 run: spans include `get_ticket` → `get_customer_history` → `escalate_to_human`.
  - Rejected T-1048 run: spans include `get_ticket` → `get_customer_history` and NO `escalate_to_human` span (middleware cancelled the call).
  - T-1042 run: `get_ticket` → `get_customer_history`, no escalate span.

**Test results (post-review)**
- `PROVIDER=groq uv run pytest tests/test_agent.py` → 3 live-LLM smoke tests pass (T-1042, T-1099, T-1048 with tightened `bug`/`P1`/`bug-team` assertions + `input()`-was-called assertion).
- `uv run pytest -k "not test_t1042 and not test_t1099 and not test_t1048"` → 40 deterministic tests pass (23 pre-existing from Epic 1 + 17 in `tests/test_agent.py`: 5 Story-2.1 retry/response-format tests + 12 Story-2.2 tests, including the 3 review-added regression guards).
- Gemini free-tier daily quota is exhausted this session; identical to Story 2.1's fallback. Re-run under Gemini once quota resets (~24 h) to fully close the loop.

**Known non-blocking artifacts**
- MLflow's LangChain autolog logs benign warnings (`AttributeError: 'MlflowLangchainTracer' object has no attribute 'on_interrupt'` / `on_resume`) when the interrupt/resume lifecycle fires. The trace still records correctly (spans and IO show up as expected in the search above); this is upstream MLflow-not-yet-aware-of-LangGraph-interrupts. Not a regression from Story 2.1.
- The `run_agent.py` MLflow lines (autolog, tracking URI, experiment) remain untouched, per the Story 2.1 read-only constraint that Story 2.2 also honors.

## Review Triage Log

Three review layers (Blind Hunter, Edge Case Hunter, Verification Gap Reviewer) surfaced 15 root-cause groups. Grouped and triaged:

- **medium, patched** — `_handle_interrupts` had no iteration cap; a middleware that kept re-emitting `__interrupt__` on every resume would prompt the operator forever (Blind Hunter, Edge Case Hunter). Fix: introduced `_MAX_HITL_INTERRUPTS = 4` and rewrote the `while` loop as a bounded `for`, raising `RuntimeError` if the cap is exceeded. Added `test_hitl_interrupt_loop_is_bounded` using an "always-interrupts" agent stub.
- **medium, patched** — T-1048 smoke test asserted only the shape of the returned decision, never that HITL fired (Verification Gap Reviewer). If `middleware=[hitl]` or the checkpointer were dropped from `create_agent`, the run would still complete and the test would still pass. Fix: capture `input()` calls into a list, `assert input_calls`, and tighten the decision assertions to CAP-5 values (`bug`/`P1`/`bug-team`).
- **medium, patched** — Retry-once contract fine, but the `config` (with `thread_id`) forwarded to `agent.ainvoke` had no test — a regression dropping the `config=` kwarg would silently break HITL because the checkpointer couldn't correlate the resume with the paused state. Fix: added `test_config_is_forwarded_to_agent_ainvoke` using a `_ConfigCapturingAgent` stub.
- **low, patched** — Spec Boundaries and `_prompt_for_escalation_decision` docstring said only an "exact" `"yes"` approves, but the code did `.strip().lower()` so `" yes "` also approves (Blind Hunter, Edge Case Hunter). Interpretation: "exact" means the specific word `yes` (not `y`, not `yeah`), not that whitespace-trimming is forbidden; `.strip()` is the reasonable CLI UX. Fix: tightened the `_prompt_for_escalation_decision` docstring and the "Manual verification" Implementation Note wording to say "case-insensitive, trimmed", matching what the tests already assert.
- **low, patched** — `KeyboardInterrupt` was promised in the spec Boundaries "Never" list but only `EOFError` was tested (Blind Hunter). Fix: added `test_hitl_keyboard_interrupt_treated_as_reject`, symmetric with the EOF test.
- **low, patched** — `escalate_to_human` docstring said "the returned string reports the decision" — misleading because the body only executes on approve; the reject path never runs the body (Verification Gap Reviewer "other"). Fix: rewrote the docstring to describe the middleware/body split explicitly.
- **low, patched** — Implementation Notes count mismatched ("seven deterministic unit tests" vs "9 new Story-2.2 tests" in different paragraphs) (Blind Hunter). Fix: reworded to name the tests by category rather than count.
- **maybe-false, deferred** — Single interrupt payload with multiple `action_requests` would receive only one `decision` in the resume `Command`, misaligning if the middleware ever batches (Edge Case Hunter). Verification: `HumanInTheLoopMiddleware(interrupt_on={"escalate_to_human": ...})` typically emits one interrupt per tool call, and `escalate_to_human` is called at most once per triage. Would be medium if batch escalation ever ships. Logged to `deferred-work.md`; what would settle it: exercise a scenario where the model attempts two `escalate_to_human` tool calls in one turn.
- **maybe-false, deferred** — Retry-once reuses the same `thread_id` (via same `config`), so the checkpointer's state from the failed first attempt may replay when `_run(retry_messages)` re-invokes (Edge Case Hunter). Verification: the deterministic retry tests use `_FakeAgent` (no checkpointer), so this only matters on live runs. Would be medium if it corrupts retry semantics. Logged to `deferred-work.md`; what would settle it: a live-LLM retry-path trace showing whether the resumed state includes stale messages.
- **false** — Non-`StructuredOutputValidationError` from `agent.ainvoke` propagates raw (network/tool crashes) (Edge Case Hunter). Disproof: pre-existing behavior from Story 2.1 and consistent with Story 2.1's Boundaries ("Missing API key → provider SDK raises its own error; let it propagate"). Not a Story 2.2 regression.
- **false** — `KeyboardInterrupt` raised outside `input()` (e.g. during `print`) isn't caught (Edge Case Hunter). Disproof: `input()` is the only blocking call in the prompt path; a SIGINT during a fast `print` is a negligible probability edge and would just abort the CLI (acceptable).
- **false** — UUID 8-hex-char slice collision risk on `thread_id` (Edge Case Hunter). Disproof: 32-bit space, single-process, single-triage-at-a-time in workshop scope — collision probability is astronomically low, and the workshop doesn't run concurrent triages.
- **false** — I/O matrix's approve row hard-codes `bug`/`P1`/`bug-team` but the T-1048 smoke test didn't originally assert it (Blind Hunter). Disproof: superseded by the patched smoke test, which now asserts those exact values.
- **low, rejected** — No unit test for the `_prompt_for_escalation_decision` payload-parsing helper (Blind Hunter). Not worth fixing: the parsed reason feeds a stderr message shown to the human; approve/reject logic depends only on `input()`. A regression here degrades the operator-facing prompt cosmetics, not the CAP-5 contract.
- **low, rejected** — `_FakeAgent.ainvoke`'s `payload` param lost its `dict` type annotation when the signature changed (Blind Hunter). Not worth fixing: the annotation is redundant in test code that accepts multiple payload shapes (`dict`, `Command`).
- **low, rejected** — Bare `except Exception` + `pragma: no cover` in `_prompt_for_escalation_decision` hides regressions (Blind Hunter). Not worth fixing: the fallback exists specifically because the interrupt payload's shape is upstream-owned and defensive; narrowing the exception adds surface for no functional gain.
- **low, rejected** — System prompt asks the model to call `escalate_to_human` BEFORE returning the structured decision; no test enforces the ordering (Blind Hunter). Not worth fixing: model-behavior enforcement via prompt is best-effort by nature; the T-1048 smoke test now asserts `input()` fired, which proves the ordering held on the run that was captured.
- **low, rejected** — Blocking `input()` inside an async coroutine blocks the event loop (Edge Case Hunter). Not worth fixing: workshop scope invokes `triage()` once per CLI run — nothing else is scheduled on the event loop to be blocked.

## Verification

**Commands:**
- `uv run pytest` — expected: all tests pass; new tests run according to Open Question resolution.
- `uv run python load_seed.py && uv run python run_agent.py T-1048` — expected: prompt appears; answering `yes` prints JSON and shows the escalate_to_human span in MLflow; answering `no` prints JSON without the span.
- `uv run python run_agent.py T-1042` — expected: no prompt; JSON matches Story 2.1's CAP-1 output.

**Manual checks (if no CLI):**
- Open the MLflow UI at http://127.0.0.1:5000 (experiment `triage-agent`) and confirm the escalated run's trace includes an `escalate_to_human` span with args and result; confirm the rejected run's trace does not.
