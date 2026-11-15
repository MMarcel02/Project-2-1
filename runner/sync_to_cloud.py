"""Uploads every run in the local database that isn't on Supabase yet.

Run from the project root: python runner/sync_to_cloud.py
Needs SUPABASE_URL, SUPABASE_KEY, NAME and MACHINE in .env.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # for the storage package

from dotenv import load_dotenv  # noqa: E402

from storage import local_store, storage_scripts  # noqa: E402

TABLES = ["our_run_data", "external_run_data"]  # the tables storage_scripts.sync_with_cloud uploads


def count_pending(conn):
    return sum(len(local_store.get_pending_runs(conn, table)) for table in TABLES)


def main():
    load_dotenv(os.path.join(ROOT, ".env"))
    name, machine, sqlite, supabase = storage_scripts.init()
    try:
        before = count_pending(sqlite)
        print(f"{before} run(s) waiting for upload in {local_store.DB_PATH}")
        if before > 0:
            storage_scripts.sync_with_cloud(sqlite, supabase)
            after = count_pending(sqlite)
            print(f"Uploaded {before - after} run(s), {after} still waiting (run this again to retry)")
    finally:
        sqlite.close()


if __name__ == "__main__":
    main()
