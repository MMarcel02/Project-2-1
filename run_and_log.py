
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import json
import os
import subprocess
import sys
import time
import yaml
import psutil
import platform
import winreg

from dotenv import load_dotenv
from storage import local_store

# Configuration

ROOT = Path(__file__).resolve().parent
RESULTS_DIRECTORY = ROOT / "results"

load_dotenv(ROOT / ".env")

CONFIG_PATH = (
    ROOT
    / "ml-agents"
    / "config"
    / "ppo"
    / "3DBallHard_short_test.yaml"
    # TODO: Switch to 3DBallHard.yaml for the full experiments
)

OPERATOR = os.environ["NAME"]
MACHINE = os.environ["MACHINE"]



# Timing Measurements


def read_training_timing(run_id):
    timers_path = RESULTS_DIRECTORY / run_id / "run_logs" / "timers.json"

    with open(timers_path, "r", encoding="utf-8") as file:
        timers = json.load(file)

    controller = timers["children"]["TrainerController.start_learning"]
    children = controller["children"]

    return {
        "mlagents_total_seconds": timers["total"],
        "controller_seconds": controller["total"],
        "environment_reset_seconds": children[
            "TrainerController._reset_env"
        ]["total"],
        "training_advance_seconds": children[
            "TrainerController.advance"
        ]["total"],
    }

def read_training_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    behavior = config["behaviors"]["3DBallHard"]

    hyperparameters = behavior["hyperparameters"]
    network = behavior["network_settings"]
    reward = behavior["reward_signals"]["extrinsic"]

    return {
        "algo": behavior["trainer_type"],
        "learning_rate": hyperparameters["learning_rate"],
        "learning_rate_schedule": hyperparameters["learning_rate_schedule"],
        "batch_size": hyperparameters["batch_size"],
        "buffer_size": hyperparameters["buffer_size"],
        "normalize": network["normalize"],
        "hidden_units": network["hidden_units"],
        "num_layers": network["num_layers"],
        "vis_encode_type": network["vis_encode_type"],
        "gamma": reward["gamma"],
        "reward_strength": reward["strength"],
        "keep_checkpoints": behavior["keep_checkpoints"],
        "max_steps": behavior["max_steps"],
        "time_horizon": behavior["time_horizon"],
        "summary_freq": behavior.get("summary_freq", 5000),
        "ppo_beta": hyperparameters["beta"],
        "ppo_epsilon": hyperparameters["epsilon"],
        "ppo_lambd": hyperparameters["lambd"],
        "ppo_num_epoch": hyperparameters["num_epoch"],
    }


def read_hardware_info():
    # Collects the basic hardware specifications
    return {
        "os_name": platform.system(),
        "cpu_model": winreg.QueryValueEx(
            winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            ),
            "ProcessorNameString"
        )[0].strip(),
        "cpu_physical_cores_total": psutil.cpu_count(logical=False),
        "cpu_logical_cores_total": psutil.cpu_count(logical=True),
        "ram_total_mb": round(
            psutil.virtual_memory().total / (1024 * 1024)
        ),
    }



def read_process_resources(process):
    # Measure the CPU and RAM usage for a process and for its children
    try:
        processes = [process] + process.children(recursive=True)

        cpu_times = {}
        ram_mb = 0.0

        for proc in processes:
            try:
                times = proc.cpu_times()

                # Identify each process by PID and creation time.
                identity = (proc.pid, proc.create_time())

                cpu_times[identity] = times.user + times.system
                ram_mb += proc.memory_info().rss / (1024 * 1024)

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return {
            "cpu_times": cpu_times,
            "ram_mb": ram_mb,
        }

    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def run_training():
    run_id = (
        f"3DBallHard_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid4().hex[:8]}"
    )

    command = [
        sys.executable,
        "-m",
        "mlagents.trainers.learn",
        str(CONFIG_PATH),
        f"--run-id={run_id}",
        "--timeout-wait=180",
    ]

    print(f"Beginning training run: {run_id}", flush=True)
    print(f"Operator: {OPERATOR}", flush=True)
    print(f"Machine: {MACHINE}", flush=True)
    print("When prompted, press play in Unity", flush=True)

    start_time = time.perf_counter()

    cpu_samples = []
    ram_samples = []

    with subprocess.Popen(command, cwd=ROOT) as training:
        process = psutil.Process(training.pid)

        # We initialize the CPU counters before sampling
        previous_resources = read_process_resources(process)
        previous_time = time.perf_counter()

        while training.poll() is None:
            time.sleep(1)

            current_resources = read_process_resources(process)
            current_time = time.perf_counter()

            if current_resources is not None:
                ram_samples.append(current_resources["ram_mb"])

                
                if previous_resources is not None:
                    elapsed = current_time - previous_time
                    cpu_used = 0.0

                    previous_cpu_times = previous_resources["cpu_times"]
                    current_cpu_times = current_resources["cpu_times"]

                    # We compare only the processes present in both samples
                    for identity, current_cpu_time in current_cpu_times.items():
                        if identity in previous_cpu_times:
                            cpu_difference = (
                                current_cpu_time - previous_cpu_times[identity]
                            )

                            if cpu_difference >= 0:
                                cpu_used += cpu_difference

                    if elapsed > 0:
                        cpu_percent = (cpu_used / elapsed) * 100
                        cpu_samples.append(cpu_percent)


                previous_resources = current_resources
                previous_time = current_time

        if training.returncode != 0:
            raise subprocess.CalledProcessError(training.returncode, command)

    process_duration_seconds = time.perf_counter() - start_time

    average_cpu_utilisation = (
        sum(cpu_samples) / len(cpu_samples)
        if cpu_samples else None
    )

    peak_cpu_utilisation = max(cpu_samples) if cpu_samples else None

    average_ram_usage_in_mb = (
        sum(ram_samples) / len(ram_samples)
        if ram_samples else None
    )

    peak_ram_usage_in_mb = max(ram_samples) if ram_samples else None
    
    completed_at = datetime.now(timezone.utc).isoformat()

    timing = read_training_timing(run_id)

    print(f"\nTraining completed: {run_id}")
    print(f"Process duration: {process_duration_seconds:.2f} seconds")
    print(f"ML-Agents duration: {timing['mlagents_total_seconds']:.2f} seconds")
    print(f"Training advancement: {timing['training_advance_seconds']:.2f} seconds")
    print(f"Completed at: {completed_at}")

    return {
        "run_id": run_id,
        "operator": OPERATOR,
        "machine": MACHINE,
        "config_file": CONFIG_PATH.name,
        "process_duration_seconds": process_duration_seconds,
        "completed_at": completed_at,
        "ml_agents_avg_cpu_util": average_cpu_utilisation,
        "ml_agents_peak_cpu_util": peak_cpu_utilisation,
        "avg_ram_usage_mb": average_ram_usage_in_mb,
        "peak_ram_usage_mb": peak_ram_usage_in_mb,
        **read_hardware_info(),
        **read_training_config(),
        **timing,
    }




def main():
    # Initialize a local SQLite database.
    sqlite = local_store.get_db()

    try:
        local_store.init_db(sqlite)
        print("A local SQLite database was initialized")

        # Begin training and then collect measurements
        measurements = run_training()

        print("\nAll measurements were collected successfully")
        print(json.dumps(measurements, indent=4))

        # TODO:
        # Convert measurements + hardware specifications +
        # hyperparameters + convergence results into RunData.
        #
        # Then save using:
        # local_store.insert_run(sqlite, "our_run_data", run_data)
        #
        # After local saving, attempt Supabase synchronization
        # using storage_scripts.sync_with_cloud().

    finally:
        sqlite.close()


if __name__ == "__main__":
    main()
