"""Load seed CSVs into app.db so mcp/triage_server.py can query them."""

import csv
import sqlite3
from contextlib import closing
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DB_PATH = REPO_ROOT / "app.db"
SEED_DIR = REPO_ROOT / "seed"

TICKETS_DDL = (
    "CREATE TABLE tickets ("
    "ticket_id TEXT NOT NULL, "
    "customer_id TEXT NOT NULL, "
    "created_at TEXT NOT NULL, "
    "text TEXT NOT NULL"
    ")"
)

CUSTOMERS_DDL = (
    "CREATE TABLE customers ("
    "customer_id TEXT NOT NULL, "
    "name TEXT NOT NULL, "
    "plan TEXT NOT NULL, "
    "open_tickets INTEGER NOT NULL"
    ")"
)


def load() -> None:
    with closing(sqlite3.connect(DB_PATH)) as conn, conn:
        conn.execute("DROP TABLE IF EXISTS tickets")
        conn.execute("DROP TABLE IF EXISTS customers")
        conn.execute(TICKETS_DDL)
        conn.execute(CUSTOMERS_DDL)

        with (SEED_DIR / "tickets.csv").open(newline="") as f:
            rows = [
                (r["ticket_id"], r["customer_id"], r["created_at"], r["text"])
                for r in csv.DictReader(f)
            ]
        conn.executemany(
            "INSERT INTO tickets (ticket_id, customer_id, created_at, text) VALUES (?, ?, ?, ?)",
            rows,
        )

        with (SEED_DIR / "customers.csv").open(newline="") as f:
            rows = [
                (r["customer_id"], r["name"], r["plan"], int(r["open_tickets"]))
                for r in csv.DictReader(f)
            ]
        conn.executemany(
            "INSERT INTO customers (customer_id, name, plan, open_tickets) VALUES (?, ?, ?, ?)",
            rows,
        )


if __name__ == "__main__":
    load()
    print(f"Loaded {DB_PATH}")
