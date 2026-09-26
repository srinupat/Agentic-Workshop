---
title: 'Seed loader'
type: 'feature'
created: '2026-09-26'
status: 'done'
route: 'oneshot'
review_loop_iteration: 0
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Epic 2's agent reads tickets and customer history via `mcp/triage_server.py`, which queries an `app.db` SQLite file at the repo root with specific table and column names — but no loader exists to populate `app.db` from the seed CSVs, and the load must be repeatable so re-running it leaves the database in the same state.

**Approach:** Ship `load_seed.py` at the repo root. It reads `seed/tickets.csv` and `seed/customers.csv` with the `csv` and `sqlite3` stdlibs, drops-and-recreates the `tickets(ticket_id, customer_id, created_at, text)` and `customers(customer_id, name, plan, open_tickets)` tables in `app.db`, and inserts every row as-is. Two consecutive runs yield identical row counts and contents. `seed/` and `mcp/triage_server.py` are not touched.

</frozen-after-approval>

## Implementation Notes

- **Loader.** `load_seed.py` at repo root; uses `sqlite3` + `csv` stdlibs — no new deps. `DB_PATH` and `SEED_DIR` derived from `Path(__file__).resolve().parent`, matching `mcp/triage_server.py`'s convention (`DB_PATH = Path(__file__).resolve().parent.parent / "app.db"`). Connection wrapped in both `contextlib.closing(...)` and the `sqlite3` context manager so the file handle is released (not just GC'd) — matters on Windows for pytest `tmp_path` cleanup.
- **Idempotency.** Drop-and-recreate (per `epic-1-context.md` Technical Decisions): each run issues `DROP TABLE IF EXISTS …; CREATE TABLE …; INSERT …`. Simplest guarantee of byte-identical state, and cheap at this data size (24 tickets, 20 customers).
- **Schema.** Exactly the columns `mcp/triage_server.py` queries. `open_tickets` stored as INTEGER (cast at insert time); all other columns TEXT. No PRIMARY KEY — the SPEC only requires column names, and adding one would fail-loudly on any future duplicate seed row without adding required behavior.
- **CSV read.** `csv.DictReader(open(..., newline=""))` — `newline=""` follows stdlib guidance so quoted commas/newlines round-trip correctly if seed data ever contains them.
- **`mcp/triage_server.py` import collision.** The local `mcp/` dir shadows the installed `mcp` PyPI package name, so `from mcp import triage_server` resolves to `.venv/`. The two integration tests below sidestep this with `importlib.util.spec_from_file_location(...)` to load `mcp/triage_server.py` as a module named `local_triage_server`. The public tools are decorated with `@server.tool()`; `getattr(tool, "fn", tool)` unwraps FastMCP's wrapper across mcp SDK versions.
- **Tests.** `tests/test_load_seed.py` — 10 cases: table set, row counts vs CSV, idempotency (two runs → identical `SELECT *`), column-shape match against the `triage_server` SELECT lists, `PRAGMA table_info` lock on the full DDL column set, `open_tickets` typed as int, verbatim value copy, presence of seed ticket `T-1042`, and `triage_server.get_ticket("T-1042")` + `get_customer_history("C-05")` returning the loaded rows (SPEC CAP-2 success signal).
- **Test isolation.** Tests monkeypatch `load_seed.DB_PATH` (and the importlib-loaded `triage_server.DB_PATH`) to a `tmp_path` so the repo's `app.db` is untouched during pytest.
- **Files changed:** `load_seed.py` (new), `tests/test_load_seed.py` (new).
- **Verification.** `uv run pytest` → 23/23 pass (13 schema + 10 loader). `rm -f app.db && uv run python load_seed.py && uv run python load_seed.py && sqlite3 app.db "SELECT COUNT(*) FROM tickets; SELECT COUNT(*) FROM customers"` → `24` / `20`, matching `wc -l seed/*.csv` (25 and 21 minus header rows).

## Review Triage Log

Blind Hunter (one layer; N=3 floor, 12 findings returned):

- **low, patched** — `sqlite3.connect(...)` context manager commits/rolls back but does not close; connection lingered until GC. Fix: wrapped in `contextlib.closing(...)` so the file handle releases immediately (Windows tmp_path cleanup safe).
- **medium, patched** — SPEC CAP-2 success signal ("`get_ticket('T-1042')` returns the loaded ticket") was not verified end-to-end; only the shared SELECT list was tested. Fix: added `test_triage_server_get_ticket_returns_seed_row` and `test_triage_server_get_customer_history_returns_seed_row`, loading `mcp/triage_server.py` via `importlib.util.spec_from_file_location` to bypass the `mcp` package-name collision.
- **low, patched** — no test guarded against accidental extra columns in the DDL. Fix: added `test_column_set_matches_ddl_exactly` using `PRAGMA table_info` to assert the exact column set for both tables.
- **low, patched** — Implementation Notes overstated `newline=""` as protecting existing multi-line ticket text (no seed row currently contains a newline). Fix: reworded to defensive-practice framing.
- **false** — `test_row_values_copied_verbatim` iterates DB rows so it misses silently-dropped rows. Disproof: `test_loads_every_seed_row` asserts count parity against the CSVs, and `test_row_values_copied_verbatim` asserts per-row identity; the two together cover completeness + fidelity.
- **false** — no ticket→customer referential integrity test. Disproof: SPEC + `epic-1-context.md` mandate "copy rows as-is" with "no transformation, renaming, or enrichment"; orphan detection is out of scope and would be feature creep.
- **false** — story status `in-progress` while sibling story 1 is `done`. Disproof: `in-progress` is the workflow's mid-implementation state; Finalize Spec (this step) moves it to `done`.
- **low, rejected** — `load_seed.py` gives a bare `FileNotFoundError` if a seed CSV is missing. Not worth fixing: `seed/` is versioned into the repo and pinned read-only; defensive error text would violate AGENTS.md's "don't validate scenarios that can't happen".
- **low, rejected** — final `print(f"Loaded {DB_PATH}")` reveals nothing about what was loaded. Not worth fixing: cosmetic; SPEC only requires the CLI to produce `app.db`, and the verification block already runs `SELECT COUNT(*)` as the smoke test.
- **low, rejected** — `_rows(db_path, table)` uses an f-string to interpolate the table name into SQL. Not worth fixing: test-only helper, called with two trusted literals, no live bug.
- **low, rejected** — `test_idempotent`'s `ORDER BY 1` relies on column-1 uniqueness. Not worth fixing: seed data is versioned; a duplicate `ticket_id` couldn't appear silently.
- **low, rejected** — `load_seed.py` at repo root couples to story 1's `pythonpath = ["."]` change. Not worth fixing: intentional and already documented in story 1's Implementation Notes.
