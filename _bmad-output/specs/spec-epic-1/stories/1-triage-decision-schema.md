---
title: 'Triage decision schema'
type: 'feature'
created: '2026-09-26'
status: 'done'
route: 'oneshot'
review_loop_iteration: 0
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Epic 2's triage agent will return decisions as structured JSON, but no validated schema exists yet — nothing rejects a decision with an unknown category, a missing field, or an empty rationale, and the agent has nothing to import when it wires up structured output.

**Approach:** Ship a small Python module (`schema.py` at repo root) that exposes a pydantic v2 `TriageDecision` model with three string-enum fields — `category` ∈ {billing, bug, access, performance, how-to}, `priority` ∈ {P1, P2, P3, P4}, `route` ∈ {billing-team, bug-team, access-team, performance-team, how-to-team} — plus a `rationale` string that must be non-empty. Any invalid value raises pydantic's `ValidationError`, which names the offending field. Cover the shape with a pytest suite Epic 2 can rely on.

</frozen-after-approval>

## Implementation Notes

- **Model.** pydantic v2 `BaseModel` at `schema.py`; `Category`, `Priority`, `Route` are `StrEnum` subclasses so agent-side JSON parses straight into enum members. `extra="forbid"` so unknown fields are rejected (matches SPEC "exactly four fields").
- **Rationale non-empty.** `Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]` — whitespace-only strings normalise to empty and fail `min_length=1`, so `"   "` is rejected the same as `""`.
- **Test discovery.** Added `pythonpath = ["."]` under `[tool.pytest.ini_options]` in `pyproject.toml` so pytest can import `schema` from repo root. `uv run python` on scripts at repo root (e.g. `run_agent.py`) already picks it up.
- **Files changed:** `schema.py` (new), `tests/test_schema.py` (new), `pyproject.toml` (one-line addition).
- **Verification:** `uv run pytest tests/test_schema.py` — 13 tests pass (12 base + 1 enum-exhaustiveness lock added during review).

## Review Triage Log

Blind Hunter (one layer; N=3 floor, 15 findings returned):

- **medium, patched** — enum sets not locked; a silent removal (e.g. `Category.HOW_TO`) would still pass all tests. Fix: added `test_enum_members_lock_the_spec` asserting the exact string sets for all three enums.
- **defer** — `.claude/settings.local.json` untracked with broad `Bash(git *)` allowlist. Evidence: file predates this story; a future `git add .` would carry it. Logged to `deferred-work.md`.
- **defer** — `epic-1-context.md` is a compiled artifact with no repo policy on commit-vs-gitignore. Logged to `deferred-work.md`.
- **false** — story `status: 'in-progress'` claimed inconsistent with the "review complete" state. Disproof: step-oneshot workflow keeps status at `in-progress` through implementation + review, and Finalize Spec sets it to `done` (as done here).
- **false** — no test enforces `category ↔ route` pairing. Disproof: SPEC CAP-1 defines the two enums as independent value sets; pairing is a policy concern (`TRIAGE_POLICY.md`) enforced by the agent, not the schema.
- **false** — rationale does not enforce "one sentence". Disproof: CAP-1 success criterion only rejects "empty rationale". "One-sentence" is intent-level guidance, not a hard validation; `NonEmptyRationale` matches the success criterion.
- **false** — story doesn't verify `mcp/triage_server.py` is unchanged. Disproof: this story only ships `schema.py`; `git diff HEAD -- mcp/` shows the server file is untouched, and AGENTS.md pins it read-only.
- **false** — tests use bare-string literals instead of enum members. Disproof: string input is what an LLM produces at runtime; testing coercion from strings is the correct path for a structured-output schema.
- **false** — no `tests/__init__.py`. Disproof: 13/13 tests collect and pass without it under the current single-file layout.
- **false** — story spec has no `## Acceptance` section. Disproof: for `route: 'oneshot'`, step-02 explicitly deletes Boundaries/Constraints, I/O Matrix, Code Map, and Tasks & Acceptance — Intent + Implementation Notes is the intended shape.
- **false** — `context: []` frontmatter empty despite references. Disproof: `context` per template is for project-wide standards not distilled into the spec; SPEC.md and epic-1-context.md are loaded via the workflow, not this field.
- **low, rejected** — `schema.py` module name is generic. Not worth fixing: no actual collision, rename cost churns the spec's Intent ("`schema.py` at repo root"), no evidence of stdlib or dep conflict in the workshop scope.
- **low, rejected** — no JSON-schema round-trip snapshot test. Not worth fixing: speculative future-proofing, larger than a simple addition, and `extra="forbid"` + enum tests already cover the key surface.
- **low, rejected** — `pythonpath = ["."]` broadens test-time import surface. Not worth fixing: intentional and necessary for the current layout; a `src/` migration is out of story scope.
- **low, rejected** — `test_rejects_extra_field` uses `"confidence"`, a plausible future field. Not worth fixing: if Epic 2 adds a `confidence` field, the test's failure is the correct signal to update it; no live bug.
