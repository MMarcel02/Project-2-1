import os
import sqlite3
import sys

import cloud_store
import local_store
import run_data_model
from dotenv import load_dotenv
from supabase import Client

load_dotenv()


def get_name() -> str:
    name = os.environ.get("NAME")
    if not name:
        raise ValueError("NAME must be set inside of your .env")
    return name


def init():
    try:
        name = get_name()
        sqlite = local_store.get_db()
        local_store.init_db(sqlite)
        supabase = cloud_store.get_db()
    except Exception as e:
        print(e)
        sys.exit(1)

    return (name, sqlite, supabase)


def sync_with_cloud(sqlite: sqlite3.Connection, supabase: Client):
    pending_runs = local_store.get_pending_runs(sqlite)
    for run_data in pending_runs:
        sync_time = cloud_store.insert_run(supabase, run_data)
        # Setting a sync time locally marks it as successfuly uploaded
        if sync_time:
            local_store.mark_synced(sqlite, sync_time, run_data["id"])


# Example interaction
# init once
name, sqlite, supabase = init()
# data gathering loop that fills run_data
run_data = run_data_model.RunData("1", name, run_data_model.ExperimentType("test_only"))
# immediately save run data locally
local_store.insert_run(sqlite, run_data)
# on each uploading try can try to also sync all unuploaded stuff
sync_with_cloud(sqlite, supabase)
