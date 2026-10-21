import sqlite3
import time
from dataclasses import asdict
from pathlib import Path

from run_data_model import RunData

CURR_DIR = Path(__file__).parent
DB_PATH = CURR_DIR / "runs.db"
SCHEMA_PATH = CURR_DIR / "supabase" / "migrations" / "runs.sql"


def get_db(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection):
    # only dropping rn as testing will remove later
    conn.execute("DROP TABLE IF EXISTS runs")
    conn.execute(SCHEMA_PATH.read_text())


def insert_run(conn: sqlite3.Connection, run_data: RunData):
    data = asdict(run_data)
    cols = ", ".join(data.keys())
    placeholders = ", ".join(["?"] * len(data))

    conn.execute(
        f"INSERT INTO runs ({cols}) VALUES ({placeholders})", list(data.values())
    )


def get_pending_runs(conn: sqlite3.Connection):
    rows = conn.execute("SELECT * FROM runs WHERE synced_at IS NULL")
    return [dict(row) for row in rows]


def mark_synced(conn: sqlite3.Connection, sync_time: float, run_id: str):
    conn.execute("UPDATE runs SET synced_at = ? WHERE id = ?", sync_time, run_id)
