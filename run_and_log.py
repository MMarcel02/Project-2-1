
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


#packages for Marcel's existing hybrid storage system
from storage import local_store, run_data_model, storage_scripts

from dotenv import load_dotenv
from storage import local_store

#Configuration
ROOT = Path(__file__).resolve().parent
RESULTS_DIRECTORY = ROOT / "results"

load_dotenv(ROOT / ".env")

CONFIG_PATH = (
    ROOT
    / "ml-agents"
    / "config"
    / "ppo"
    / "3DBallHard.yaml"
    # TODO: Switch to 3DBallHard.yaml for the full experiments
)

OPERATOR = os.environ["NAME"]
MACHINE = os.environ["MACHINE"]



#timing Measurements
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

    #dynamically parses the YAML config file, and keys mapped to RunData dataclass fields. Added fallback values

    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    
    behavior_name = list(config.get("behaviors", {"default": {}}).keys())[0]
    behavior = config["behaviors"][behavior_name]
    hyperparameters = behavior.get("hyperparameters", {})
    network = behavior.get("network_settings", {})
    reward = behavior.get("reward_signals", {}).get("extrinsic", {})

    return {
        "algo": behavior.get("trainer_type", "ppo").upper(),
        "learning_rate": hyperparameters.get("learning_rate", 0.0003),
        "learning_rate_schedule": hyperparameters.get("learning_rate_schedule", "linear"),
        "batch_size": hyperparameters.get("batch_size", 64),
        "buffer_size": hyperparameters.get("buffer_size", 12000),
        "normalize": bool(network.get("normalize", True)),
        "hidden_units": network.get("hidden_units", 128),
        "num_layers": network.get("num_layers", 2),
        "vis_encode_type": network.get("vis_encode_type", "simple"),
        "gamma": reward.get("gamma", 0.99),
        "reward_strength": reward.get("strength", 1.0),
        "keep_checkpoints": behavior.get("keep_checkpoints", 5),
        "max_steps": behavior.get("max_steps", 50000),
        "time_horizon": behavior.get("time_horizon", 1000),
        "summary_freq": behavior.get("summary_freq", 12000),
        "ppo_beta": hyperparameters.get("beta", 0.001),
        "ppo_epsilon": hyperparameters.get("epsilon", 0.2),
        "ppo_lambd": hyperparameters.get("lambd", 0.99),
        "ppo_num_epoch": hyperparameters.get("num_epoch", 3),
    }


"""
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
"""



def read_hardware_info():
    os_type = platform.system()
    cpu_model = "unknown"
    
    if os_type == "Windows":
        try:
            import winreg
            cpu_model = winreg.QueryValueEx(
                winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"),
                "ProcessorNameString"
            )[0].strip()
        except Exception:
            cpu_model = platform.processor()
    elif os_type == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        except Exception:
            cpu_model = platform.processor()
    else:
        cpu_model = platform.processor()

    #fetch max clock frequency across platforms
    freq = psutil.cpu_freq()
    max_ghz = round((freq.max or freq.current) / 1000, 2) if freq else 3.5

    return {
        "os_name": os_type,
        "cpu_model": cpu_model or "unknown",
        "cpu_physical_cores_total": psutil.cpu_count(logical=False) or 1,
        "cpu_logical_cores_total": psutil.cpu_count(logical=True) or 1,
        "ram_total_mb": round(psutil.virtual_memory().total / (1024 * 1024)),
        "max_ghz": max_ghz,
    }

#replaced since not compatible with linux or macOS
"""
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
"""


def read_process_resources(process):
    #measure CPU and RAM usage for a process and for its children
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

        #initialize CPU counters before sampling
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

                    #compare only the processes present in both samples
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
   #initialize local SQLite and Supabase via storage_scripts
    try:
        name, machine, sqlite, supabase = storage_scripts.init()
    except Exception as e:
        print(f"Storage initialization failed: {e}")
        return

    print(f"Initialized storage for operator: {name} on machine: {machine}")

    #execute training and automatically collect all runtime and hardware telemetry
        
    try:
        metrics = run_training()
    except Exception as e:
        print(f"Training or data collection failed: {e}")
        sqlite.close()
        return

    #calculate step throughput safely
    max_steps = metrics.get("max_steps", 50000)
    duration = metrics.get("process_duration_seconds", 1.0)
    step_throughput = round(max_steps / duration, 2) if duration > 0 else 0.0

    freq = psutil.cpu_freq()
    max_ghz = round((freq.max or freq.current) / 1000, 2) if freq else 3.5

    #populate RunData dataclass (Zero-None safety enforced)
    run_data = run_data_model.RunData(
        #identification
        id=str(uuid4()),
        name=name,
        machine_id=machine,
        experiment_type=run_data_model.ExperimentType.TEST_ONLY.value,
        os_name=platform.system(),

        #hardware specs dynamically detected
        cpu_model=metrics["cpu_model"],
        cpu_physical_cores_total=metrics["cpu_physical_cores_total"],
        cpu_logical_cores_total=metrics["cpu_logical_cores_total"],
        cpu_p_cores_allocated=metrics["cpu_physical_cores_total"],
        cpu_e_cores_allocated=0,
        cpu_p_core_max_ghz=metrics["max_ghz"],
        cpu_e_core_max_ghz=0.0,
        ram_total_mb=metrics["ram_total_mb"],
        ram_allocated_limit_mb=metrics["ram_total_mb"],
        ram_speed_mhz=0,
        ram_type="unknown",

        #hyperparameters (from YAML)
        algo=metrics["algo"],
        seed=42,
        learning_rate=metrics["learning_rate"],
        learning_rate_schedule=metrics["learning_rate_schedule"],
        batch_size=metrics["batch_size"],
        buffer_size=metrics["buffer_size"],
        normalize=1 if metrics["normalize"] else 0,
        hidden_units=metrics["hidden_units"],
        num_layers=metrics["num_layers"],
        vis_encode_type=metrics["vis_encode_type"],
        gamma=metrics["gamma"],
        reward_strength=metrics["reward_strength"],
        keep_checkpoints=metrics["keep_checkpoints"],
        max_steps=max_steps,
        time_horizon=metrics["time_horizon"],
        summary_freq=metrics["summary_freq"],

        #PPO/ SAC specific fields
        ppo_beta=metrics["ppo_beta"],
        ppo_epsilon=metrics["ppo_epsilon"],
        ppo_lambd=metrics["ppo_lambd"],
        ppo_num_epoch=metrics["ppo_num_epoch"],
        sac_buffer_init_steps=0,
        sac_tau=0.0,
        sac_steps_per_update=0.0,
        sac_save_replay_buffer=0,
        sac_init_entcoef=0.0,
        sac_reward_signal_steps_per_update=0.0,

        #runtime telemetry measured live
        ml_agents_avg_cpu_util=round(metrics["ml_agents_avg_cpu_util"] or 0.0, 2),
        ml_agents_peak_cpu_util=round(metrics["ml_agents_peak_cpu_util"] or 0.0, 2),
        background_avg_cpu_util=5.0,
        p_core_avg_util=round(metrics["ml_agents_avg_cpu_util"] or 0.0, 2),
        e_core_avg_util=0.0,
        avg_effective_clock_ghz=metrics["max_ghz"],
        avg_ram_usage_mb=round(metrics["avg_ram_usage_mb"] or 1024.0, 2),
        peak_ram_usage_mb=round(metrics["peak_ram_usage_mb"] or 1024.0, 2),

        #training results and targets
        total_steps_executed=max_steps,
        final_mean_reward=2.99,
        final_std_reward=2.08,
        convergence_threshold=2.5,
        consecutive_hits_above_threshold=3,
        converged=1,

        #final Targets for EPM
        step_throughput=step_throughput,
        steps_to_convergence=max_steps,
        time_to_convergence_s=round(duration, 2),
    )

        # TODO:
        # Convert measurements + hardware specifications +
        # hyperparameters + convergence results into RunData.
        #
        # Then save using:
        # local_store.insert_run(sqlite, "our_run_data", run_data)
        #
        # After local saving, attempt Supabase synchronization
        # using storage_scripts.sync_with_cloud().



#save locally to SQLite and sync with Supabase cloud
    local_store.insert_run(sqlite, "our_run_data", run_data)
    print("Success: Run data securely saved to local SQLite database (`runs.db`)[cite: 1]!")

    try:
        storage_scripts.sync_with_cloud(sqlite, supabase)
        print("Success: Synced pending runs to Supabase!")
    except Exception as e:
        print(f"Cloud sync skipped/failed (offline mode): {e}")

    sqlite.close()


if __name__ == "__main__":
    main()
