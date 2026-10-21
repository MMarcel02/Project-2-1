import os
import sqlite3
import sys

from storage import cloud_store, local_store
from dotenv import load_dotenv
from supabase import Client

load_dotenv()


def get_name() -> str:
    name = os.environ.get("NAME")
    if not name:
        raise ValueError("NAME must be set inside of your .env")
    return name


def get_machine_name() -> str:
    name = os.environ.get("MACHINE")
    if not name:
        raise ValueError("MACHINE must be set inside of your .env")
    return name


def init():
    try:
        name = get_name()
        machine = get_machine_name()
        sqlite = local_store.get_db()
        local_store.init_db(sqlite)
        supabase = cloud_store.get_db()
    except Exception as e:
        print(e)
        sys.exit(1)

    return (name, machine, sqlite, supabase)


def sync_with_cloud(sqlite: sqlite3.Connection, supabase: Client):
    tables = ["our_run_data", "external_run_data"]
    for table_name in tables:
        pending_runs = local_store.get_pending_runs(sqlite, table_name)
        print(pending_runs)
        for run_data in pending_runs:
            sync_time = cloud_store.insert_run(supabase, table_name, run_data)
            # Setting a sync time locally marks it as successfuly uploaded
            if sync_time:
                local_store.mark_synced(sqlite, table_name, sync_time, run_data["id"])
