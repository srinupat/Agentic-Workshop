# Epic 1 Context: triage decision schema and seed loader

<!-- Compiled from planning artifacts. Edit freely. Regenerate with compile-epic-context if planning docs change. -->

## Goal

Give Epic 2's triage agent the two things it needs to exist end-to-end: a validated decision object it can return as structured output, and a local SQLite database (`app.db`) that `mcp/triage_server.py` can read through its existing `get_ticket` / `get_customer_history` tools. Nothing in this epic implements the agent itself — this is the data and schema foundation everything downstream builds on.

No traditional planning artifacts exist for this project (no PRD, architecture, or UX docs). The full contract lives in `_bmad-output/specs/spec-epic-1/SPEC.md`; this file distills the parts a story-level developer needs.

## Stories

- Story 1: Triage decision schema
- Story 2: Seed loader

## Requirements & Constraints

- The decision object has exactly four fields — `category`, `priority`, `route`, `rationale` — with these allowed values:
  - `category` ∈ {billing, bug, access, performance, how-to}
  - `priority` ∈ {P1, P2, P3, P4}
  - `route` ∈ {billing-team, bug-team, access-team, performance-team, how-to-team}
  - `rationale` is a non-empty one-sentence string.
- Any decision with an unknown enum value, missing field, or empty rationale is rejected with an error that names the offending field.
- The seed loader is a single command (`uv run python load_seed.py`) that produces `app.db` at the repo root, and running it twice yields the same database (byte-identical row counts and contents).
- Loaded tables are exactly `tickets(ticket_id, customer_id, created_at, text)` and `customers(customer_id, name, plan, open_tickets)` — column names must match the CSV headers and what `mcp/triage_server.py` already queries.
- `seed/` is read-only.
- No network calls and no API keys anywhere in this epic.

## Technical Decisions

- Python 3.12+, packages managed with uv; never pip.
- The decision schema is exposed as a Python module Epic 2 can `import` and validate against. Validator library (pydantic, dataclass + validator, jsonschema, hand-rolled) is a build-time choice; any option that meets the constraints is acceptable.
- Idempotency is achieved by dropping and recreating the `tickets` and `customers` tables on each loader run — the simplest way to guarantee identical state.
- `app.db` lives at the repo root (that's where `mcp/triage_server.py` looks for it).
- No transformation, renaming, or enrichment of CSV rows — copy them into SQLite as-is.

## Cross-Story Dependencies

- Story 1 (schema) and Story 2 (loader) are independent and can be built in either order; the spec's chosen order is schema → loader.
- Both stories are consumed by Epic 2: the agent imports the schema for its structured output, and its MCP tools read `app.db` via the unchanged `mcp/triage_server.py`.
