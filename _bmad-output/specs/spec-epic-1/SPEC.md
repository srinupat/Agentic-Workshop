---
id: SPEC-epic-1
companions: []
sources: [../../../INTENT.md]
---

> **Canonical contract.** This SPEC and the files in `companions:` are the complete, preservation-validated contract for what to build, test, and validate. Source documents listed in frontmatter are for traceability — consult them only if you need narrative rationale or prose color this contract intentionally omits.

# Epic 1: triage decision schema and seed loader

## Why

Epic 2's triage agent needs two things that do not exist yet: a validated decision object it can return, and a local SQLite database (`app.db`) that `mcp/triage_server.py` can read via its existing `get_ticket` / `get_customer_history` tools. Epic 1 delivers both — a strict decision schema and a repeatable loader — so that Epic 2 has a shape to fill and data to look up, and Epic 3 has a schema to assert against in its eval.

## Capabilities

- **CAP-1**
  - **intent:** A triage decision exists as a validated structured object with a category, a priority, a route, and a one-sentence rationale; any value outside the allowed set is rejected with a clear error.
  - **success:** `{category: "billing", priority: "P2", route: "billing-team", rationale: "..."}` validates. Any decision with an unknown value (e.g. `category: "foobar"`), a missing field, or an empty rationale is rejected with an error naming the offending field. Allowed values: `category ∈ {billing, bug, access, performance, how-to}`; `priority ∈ {P1, P2, P3, P4}`; `route ∈ {billing-team, bug-team, access-team, performance-team, how-to-team}`.

- **CAP-2**
  - **intent:** A single command loads the seed CSVs into a local SQLite database at the repo root, and running it a second time leaves the database in the same state.
  - **success:** `uv run python load_seed.py` produces `app.db` containing tables `tickets(ticket_id, customer_id, created_at, text)` and `customers(customer_id, name, plan, open_tickets)`, with every row from `seed/tickets.csv` and `seed/customers.csv`. Running the command twice yields identical row counts and contents. After a load, `mcp/triage_server.py`'s `get_ticket("T-1042")` returns the loaded ticket without changes to the server file.

## Constraints

- Python 3.12 or newer; packages installed and run through uv, never pip.
- `seed/` is read-only — the loader reads from it and never writes to it.
- No network calls and no API keys in this epic.
- `app.db` table and column names are exactly what `mcp/triage_server.py` already queries: `tickets(ticket_id, customer_id, created_at, text)` and `customers(customer_id, name, plan, open_tickets)`. The loader matches the server; the server is read-only.
- The decision schema is exposed as a Python module Epic 2's agent code can import to validate its structured output.

## Non-goals

- The triage agent, the MCP tools (already exist in `mcp/triage_server.py`), the eval harness, and any user interface — those belong to Epics 2 and 3.
- Any transformation, enrichment, or renaming of CSV columns; the loader copies rows as-is.

## Success signal

Running `uv run python load_seed.py` twice back-to-back produces the same `app.db`; `sqlite3 app.db "SELECT COUNT(*) FROM tickets"` matches the CSV row count, and `mcp/triage_server.py`'s `get_ticket("T-1042")` returns the seed row without server changes. In parallel, a hand-built decision that matches the schema validates cleanly, and a decision with `category="foobar"` fails validation with an error that names the offending field.

## Assumptions

- Idempotency is implemented by dropping and recreating the `tickets` and `customers` tables on each load — the simplest way to guarantee byte-identical state. INTENT.md is silent on approach; upserts would also satisfy CAP-2 but add complexity for no gain here.
- Which validator library to use (pydantic, dataclasses + a validator, jsonschema, hand-rolled) is a build-time choice, not a spec-level one — any of them meets CAP-1.
