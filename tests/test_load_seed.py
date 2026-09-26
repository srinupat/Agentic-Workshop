import csv
import importlib.util
import sqlite3
from pathlib import Path

import pytest

import load_seed

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_triage_server():
    path = REPO_ROOT / "mcp" / "triage_server.py"
    spec = importlib.util.spec_from_file_location("local_triage_server", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tables(db_path: Path) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _rows(db_path: Path, table: str) -> list[tuple]:
    with sqlite3.connect(db_path) as conn:
        return list(conn.execute(f"SELECT * FROM {table} ORDER BY 1"))


def _csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def db_path(tmp_path, monkeypatch) -> Path:
    dbp = tmp_path / "app.db"
    monkeypatch.setattr(load_seed, "DB_PATH", dbp)
    return dbp


def test_creates_expected_tables(db_path):
    load_seed.load()
    assert _tables(db_path) == {"tickets", "customers"}


def test_loads_every_seed_row(db_path):
    load_seed.load()
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == len(
            _csv_rows(load_seed.SEED_DIR / "tickets.csv")
        )
        assert conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == len(
            _csv_rows(load_seed.SEED_DIR / "customers.csv")
        )


def test_idempotent(db_path):
    load_seed.load()
    tickets_first = _rows(db_path, "tickets")
    customers_first = _rows(db_path, "customers")
    load_seed.load()
    assert _rows(db_path, "tickets") == tickets_first
    assert _rows(db_path, "customers") == customers_first


def test_column_shape_matches_mcp_server_queries(db_path):
    load_seed.load()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        ticket = dict(
            conn.execute(
                "SELECT ticket_id, customer_id, created_at, text FROM tickets LIMIT 1"
            ).fetchone()
        )
        customer = dict(
            conn.execute(
                "SELECT customer_id, name, plan, open_tickets FROM customers LIMIT 1"
            ).fetchone()
        )
    assert set(ticket) == {"ticket_id", "customer_id", "created_at", "text"}
    assert set(customer) == {"customer_id", "name", "plan", "open_tickets"}


def test_open_tickets_is_integer(db_path):
    load_seed.load()
    with sqlite3.connect(db_path) as conn:
        for (val,) in conn.execute("SELECT open_tickets FROM customers"):
            assert isinstance(val, int)


def test_row_values_copied_verbatim(db_path):
    load_seed.load()
    csv_tickets = {r["ticket_id"]: r for r in _csv_rows(load_seed.SEED_DIR / "tickets.csv")}
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute("SELECT ticket_id, customer_id, created_at, text FROM tickets"):
            source = csv_tickets[row["ticket_id"]]
            assert row["customer_id"] == source["customer_id"]
            assert row["created_at"] == source["created_at"]
            assert row["text"] == source["text"]


def test_seed_ticket_T_1042_present(db_path):
    load_seed.load()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT ticket_id, customer_id, created_at, text FROM tickets WHERE ticket_id = ?",
            ("T-1042",),
        ).fetchone()
    assert row is not None
    assert row["ticket_id"] == "T-1042"


def test_column_set_matches_ddl_exactly(db_path):
    load_seed.load()
    with sqlite3.connect(db_path) as conn:
        ticket_cols = {row[1] for row in conn.execute("PRAGMA table_info(tickets)")}
        customer_cols = {row[1] for row in conn.execute("PRAGMA table_info(customers)")}
    assert ticket_cols == {"ticket_id", "customer_id", "created_at", "text"}
    assert customer_cols == {"customer_id", "name", "plan", "open_tickets"}


def test_triage_server_get_ticket_returns_seed_row(db_path, monkeypatch):
    load_seed.load()
    triage_server = _load_triage_server()
    monkeypatch.setattr(triage_server, "DB_PATH", db_path)
    get_ticket = getattr(triage_server.get_ticket, "fn", triage_server.get_ticket)
    row = get_ticket("T-1042")
    assert row["ticket_id"] == "T-1042"
    assert row["customer_id"] == "C-77"


def test_triage_server_get_customer_history_returns_seed_row(db_path, monkeypatch):
    load_seed.load()
    triage_server = _load_triage_server()
    monkeypatch.setattr(triage_server, "DB_PATH", db_path)
    get_customer_history = getattr(
        triage_server.get_customer_history, "fn", triage_server.get_customer_history
    )
    customer = get_customer_history("C-05")
    assert customer["customer_id"] == "C-05"
    assert "ticket_ids" in customer
