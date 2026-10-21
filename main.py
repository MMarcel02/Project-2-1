import uuid

from storage import cloud_store, local_store, run_data_model, storage_scripts


def main():
    # Example interaction
    # init once
    name, sqlite, supabase = storage_scripts.init()
    # data gathering loop that fills run_data
    run_data = run_data_model.RunData(
        str(uuid.uuid4()), name, run_data_model.ExperimentType("test_only")
    )
    # immediately save run data locally
    local_store.insert_run(sqlite, run_data)
    # on each uploading try can try to also sync all unuploaded stuff
    storage_scripts.sync_with_cloud(sqlite, supabase)


if __name__ == "__main__":
    main()
