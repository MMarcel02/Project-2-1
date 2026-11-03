from storage import cloud_store, local_store, storage_scripts

sqlite = local_store.get_db()
supabase = cloud_store.get_db()

storage_scripts.sync_with_cloud(sqlite, supabase)
sqlite.close()