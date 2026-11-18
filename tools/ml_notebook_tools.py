from pathlib import Path
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


def get_data(URL: str, KEY: str, external: bool = False) -> pd.DataFrame:
    supabase = create_client(URL, KEY)
    table = "external_run_data" if external else "our_run_data"

    response = supabase.table(table).select("*").execute()
    return pd.DataFrame(response.data)


FEATURES = [
    "cpu_physical_cores_total",
    "cpu_logical_cores_total",
    "cpu_p_cores_allocated",
    "ram_total_mb",
    "ram_allocated_limit_mb",
    "learning_rate",
    "batch_size",
    "buffer_size",
    "hidden_units",
    "num_layers",
    "gamma",
    "ppo_beta",
    "ppo_epsilon",
    "ppo_lambd",
    "ppo_num_epoch",
]

NUMERIC_TARGETS = [
    "step_throughput",
    "time_to_convergence_s",
    "final_mean_reward",
    "final_std_reward",
]


def load_runs(external=False, env_file=".env") -> pd.DataFrame:
    load_dotenv(Path(env_file))
    return get_data(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_KEY"],
        external=external,
    )


def clean_runs(df):
    cleaned = df.copy()
    if "id" in cleaned:
        cleaned = cleaned.drop_duplicates(subset="id")
    for column in FEATURES + NUMERIC_TARGETS:
        if column in cleaned:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned.replace([np.inf, -np.inf], np.nan)


def throughput_xy(df):
    usable = df[df["step_throughput"].notna() & (df["step_throughput"] > 0)]
    features = [name for name in FEATURES if name in usable and usable[name].notna().any()]
    return usable[features], usable["step_throughput"]


def make_forest():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    )
