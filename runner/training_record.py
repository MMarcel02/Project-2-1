"""Turns a finished training result into a database row (RunData)"""

# there might be warning or errors in the editor (lines 8,9,19,44) but those are harmless
# if not use 'from run_and_log import TrainingResult'
import uuid
from dataclasses import asdict

from storage import local_store
from storage.run_data_model import RunData

import mlagents_output
from machine_specs import HardwareSpecs

# Placeholders until convergence is implemented and measured
CONVERGENCE_THRESHOLD = 95.0
CONSECUTIVE_HITS = 3


def get_reasons_for_failed_save(result: "TrainingResult") -> list[str]:
    """Returns any reasons the training run couldn't be added to the database"""
    reasons = []
    if result.exit_code != 0:
        reasons.append(f"ML-Agents exited with code {result.exit_code}")
    if result.interrupted:
        reasons.append("interrupted before finishing")
    if not result.summaries:
        reasons.append("no reward summaries")
    if not result.cpu_samples:
        reasons.append("no CPU/RAM samples")
    if result.hyperparameters is None:
        reasons.append("configuration.yaml missing or unreadable")
    else:
        missing = mlagents_output.get_missing_hyperparameters(result.hyperparameters)
        if missing:
            reasons.append("configuration.yaml lacks " + ", ".join(missing))
    return reasons


def average(samples: list[float]) -> float:
    return sum(samples) / len(samples)


def build_run_data(
    result: "TrainingResult",
    specs: HardwareSpecs,
    *,
    name: str,
    machine: str,
    experiment_type: str,
    seed: int,
) -> RunData:
    """The database row for a finished run. Expects a complete run that can be saved"""
    assert (
        result.hyperparameters is not None and result.summaries and result.cpu_samples
    )
    last = result.summaries[-1]
    return RunData(
        # identification
        id=str(uuid.uuid4()),
        name=name,
        machine_id=machine,
        experiment_type=experiment_type,
        seed=seed,
        # hardware and hyperparameters: their field names are the column names
        **asdict(specs),
        **asdict(result.hyperparameters),
        # runtime measurements
        ml_agents_avg_cpu_util=average(result.cpu_samples),
        ml_agents_peak_cpu_util=max(result.cpu_samples),
        background_avg_cpu_util=average(result.background_cpu_samples),
        p_core_avg_util=None,  # not measured yet
        e_core_avg_util=None,  # not measured yet
        avg_effective_clock_ghz=None,  # not measured yet
        avg_ram_usage_mb=average(result.ram_samples),
        peak_ram_usage_mb=max(result.ram_samples),
        # results
        total_steps_executed=result.final_steps
        if result.final_steps is not None
        else last.step,
        final_mean_reward=last.mean_reward,
        final_std_reward=last.std_reward,
        convergence_threshold=CONVERGENCE_THRESHOLD,  # placeholder
        consecutive_hits_above_threshold=CONSECUTIVE_HITS,  # placeholder
        converged=True,  # placeholder
        step_throughput=result.step_throughput,
        steps_to_convergence=None,  # not measured yet
        time_to_convergence_s=None,  # placeholder
    )


def save_to_database(row: RunData) -> None:
    """Insert row into the local database (creates it if missing)"""
    connection = local_store.get_db()
    try:
        local_store.init_db(connection)
        local_store.insert_run(connection, "our_run_data", row)
    finally:
        connection.close()
