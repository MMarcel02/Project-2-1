import sqlite3
import time
from dataclasses import asdict
from pathlib import Path

from .run_data_model import RunData

CURR_DIR = Path(__file__).parent
DB_PATH = CURR_DIR / "runs.db"
SCHEMA_DIR = CURR_DIR / "supabase" / "migrations"


def get_latest_schema() -> str:
    schemas = sorted(SCHEMA_DIR.glob("*.sql"))
    latest_schema = schemas[-1]
    return latest_schema.read_text()


def get_db(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection):
    conn.executescript(get_latest_schema())


def insert_run(conn: sqlite3.Connection, table_name: str, run_data: RunData):
    data = asdict(run_data)
    cols = ", ".join(data.keys())
    placeholders = ", ".join(["?"] * len(data))

    conn.execute(
        f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})",
        list(data.values()),
    )
    conn.commit()


def get_pending_runs(conn: sqlite3.Connection, table_name: str):
    rows = conn.execute(f"SELECT * FROM {table_name} WHERE synced_at IS NULL")
    return [dict(row) for row in rows]


def mark_synced(conn: sqlite3.Connection, table_name: str, sync_time: int, run_id: str):
    conn.execute(
        f"UPDATE {table_name} SET synced_at = ? WHERE id = ?", (sync_time, run_id)
    )
    conn.commit()
