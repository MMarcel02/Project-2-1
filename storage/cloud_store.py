import os
import time

from supabase import Client, create_client


def get_db() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        raise ValueError(
            "SUPABASE_KEY and SUPABASE_URL must be set inside of your .env"
        )
    return create_client(url, key)


def insert_run(client: Client, table_name: str, run_data: dict) -> int | None:
    synced_time = int(time.time())
    run_data["synced_at"] = synced_time
    try:
        client.table(table_name).upsert(run_data).execute()
    except Exception as e:
        print(f"Supabase upload failed: {e}")
    else:
        return synced_time
    return None
