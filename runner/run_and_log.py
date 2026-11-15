"""TODO DOCSTRING"""

import argparse
import os
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from signal import SIGINT, SIGTERM, signal

import psutil
from dotenv import load_dotenv

import machine_specs
import mlagents_output

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from storage.run_data_model import ExperimentType

BUILDS = {
    "win32": Path("windows") / "3DBall.exe",
    "linux": Path("linux") / "3DBall.x86_64",
    "darwin": Path("macos") / "3DBall.app",
}

DEFAULT_UNITY_BUILD_PATH = ROOT / "builds" / BUILDS.get(sys.platform, BUILDS["linux"])
DEFAULT_TRAINING_CONFIG_PATH = ROOT / "runner" / "configs" / "training" / "3DBall.yaml"
RESULTS_DIR_PATH = ROOT / "results"

EXIT_MESSAGE = "Stopping, please wait..."

MEASURE_INTERVAL_S = 1  # measure once this many seconds
CPU_COUNT = psutil.cpu_count() or 1  # logical cores (threads); 'or 1' when unknown
TRAINING_STARTED = "Connected new brain"


@dataclass
class TrainingResult:
    """A structure which contains training measurements:
    - duration - the elapsed time since starting the program
    - startup - elapsed time before actual training happens
    - summaries - parsed summary lines from the live output of ML-Agents
    - final steps - actual total step count read from ML-Agents' output file
    """

    run_id: str
    exit_code: int
    duration_s: float
    startup_s: float | None
    cpu_samples: list[float]
    background_cpu_samples: list[float]  # % background utilization of the whole machine
    ram_samples: list[float]
    summaries: list[mlagents_output.RewardSummary]
    final_steps: int | None  # exact step count; None if unavailable
    hyperparameters: (
        mlagents_output.Hyperparameters | None
    )  # read from file; None if unavailable
    interrupted: bool  # stopped via Keyboard Interrupt or `terminate` before finishing

    @property
    def step_throughput(self) -> float | None:
        """Steps per second for the training"""
        if not self.summaries:
            return None
        last = self.summaries[-1]
        if last.elapsed_s <= 0:
            return None
        return last.step / last.elapsed_s


def show_project_relative(path: Path) -> Path:
    """show path as <project_root>/..., with the OS's separator"""
    return Path("<project_root>") / path.relative_to(ROOT)


def parse_args() -> argparse.Namespace:
    """Create CLI-like arguments (with default values, type safety and documentation),
    then build and return them"""
    generated_run_id = "train_" + time.strftime("%Y%m%d_%H%M%S")
    generated_seed = random.randint(0, 2**31 - 1)

    parser = argparse.ArgumentParser(description="Run one ML-Agents training")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG_PATH,
        help=f"path to training config (YAML) (default: {show_project_relative(DEFAULT_TRAINING_CONFIG_PATH)})",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=DEFAULT_UNITY_BUILD_PATH,
        help=f"environment (Unity build) to use for training (default: {show_project_relative(DEFAULT_UNITY_BUILD_PATH)})",
    )
    parser.add_argument(
        "--run-id",
        default=generated_run_id,
        help="name of the run (default: train_<YYYYMMDD>_<HHMMSS>, e.g. train_20261224_124200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=generated_seed,
        help="random seed (default: a random number)",
    )
    parser.add_argument(
        "--graphics",
        action="store_true",
        help="display the game window while training (default: no window)",
    )
    parser.add_argument(
        "--experiment-type",
        choices=[experiment.value for experiment in ExperimentType],
        default=ExperimentType.TEST_ONLY.value,
        help="which research question the run belongs to (default: test_only)",
    )
    return parser.parse_args()


def read_identity() -> tuple[str, str]:
    """Retrieved infomation about who performed the training and their machine"""
    load_dotenv(ROOT / ".env")
    name = os.environ.get("NAME")
    machine = os.environ.get("MACHINE")
    if not name or not machine:
        sys.exit("NAME and MACHINE must be set in .env")
    return name, machine


def check_paths(args: argparse.Namespace) -> None:
    """Stops executing further if the unity build or the training config don't exist"""
    for path in [args.config, args.env]:
        if not path.exists():
            sys.exit(f"Not found: {path.resolve()}")


def build_command(args: argparse.Namespace) -> list[str]:
    """Builds a complete ML-Agents command from the passed arguments ready to execute"""
    command = [
        sys.executable,
        "-m",
        "mlagents.trainers.learn",
        str(args.config),
        f"--env={args.env}",
        f"--run-id={args.run_id}",
        f"--seed={args.seed}",
    ]
    if not args.graphics:
        command.append("--no-graphics")
    return command


def start_mlagents(command: list[str]) -> psutil.Popen:
    """Start ML-Agents with output forwarded to the script"""
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    return psutil.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def install_stop_handlers(process: psutil.Popen) -> threading.Event:
    """Provides safe stop/terminate logic, and records if it has occured"""

    stop_requested = threading.Event()

    def on_keyboard_interrupt(signum, frame):
        stop_requested.set()
        print(EXIT_MESSAGE)

    def on_terminate(signum, frame):
        stop_requested.set()
        print(EXIT_MESSAGE)
        process.send_signal(SIGINT)

    signal(SIGINT, on_keyboard_interrupt)
    if sys.platform != "win32":
        signal(SIGTERM, on_terminate)
    return stop_requested


def forward_output(
    process: psutil.Popen,
    training_started: threading.Event,
    summaries: list[mlagents_output.RewardSummary],
) -> None:
    """Shows ML-Agents' output and notifies when actual training starts.
    Collect the reward summary lines into `summaries`
    """
    for line in process.stdout:
        print(line, end="", flush=True)
        if not training_started.is_set() and TRAINING_STARTED in line:
            training_started.set()
        summary = mlagents_output.parse_summary_line(line)
        if summary:
            summaries.append(summary)


def start_output_reader(
    process: psutil.Popen, summaries: list[mlagents_output.RewardSummary]
) -> tuple[threading.Thread, threading.Event]:
    """Starts the background thread that reads ML-Agents' output (see forward_output)"""

    training_started = threading.Event()
    reader = threading.Thread(
        target=forward_output, args=(process, training_started, summaries), daemon=True
    )
    reader.start()
    return reader, training_started


def wait_for_training_start(process: psutil.Popen, started: threading.Event) -> bool:
    """Wait until actual training starts (so not to record the setup phase for ML-Agents)
    - Returns True when setup is finished successfully
    - Returns False when there is an early exit
    """
    while process.poll() is None:
        if started.wait(timeout=MEASURE_INTERVAL_S):
            return True
    return False


class ResourceSampler:
    """CPU and RAM samples of a process and all its child processes"""

    def __init__(self, process: psutil.Popen):
        self.process = process
        self.tracked: dict[int, psutil.Process] = {process.pid: process}
        self.cpu_samples: list[float] = []  # % utilization of the whole machine
        self.ram_samples: list[float] = []  # MB usage
        self.background_cpu_samples: list[float] = []  # % background utilization

    def start(self) -> None:
        """Starts the CPU counters now; the first sample() measures from this moment"""
        self._track_new_children()
        for proc in self.tracked.values():
            try:
                proc.cpu_percent()
            except psutil.Error:
                pass
        psutil.cpu_percent()

    def sample(self) -> None:
        """Records one CPU and one RAM sample (measures all tracked processes together)"""
        self._track_new_children()
        cpu = 0.0
        ram = 0
        for process in self.tracked.values():
            try:
                cpu += process.cpu_percent()
                ram += process.memory_info().rss
            except psutil.Error:  # this process has ended
                pass

        training_cpu_utilization = cpu / CPU_COUNT
        system_cpu_utilization = psutil.cpu_percent()
        self.cpu_samples.append(training_cpu_utilization)
        self.background_cpu_samples.append(
            max(0.0, system_cpu_utilization - training_cpu_utilization)
        )
        self.ram_samples.append(ram / 1024**2)

    def _track_new_children(self) -> None:
        try:
            for child in self.process.children(recursive=True):
                pid = child.pid
                if pid not in self.tracked:
                    child.cpu_percent()  # the first call only starts its CPU counter
                    self.tracked[pid] = child
        except psutil.Error:  # the process ended
            pass


def execute_training(command: list[str], run_id: str) -> TrainingResult:
    """Runs single ML-agents training and measures it

    Returns:
        - Exit code
        - Duration (total and startup time in seconds)
        - CPU and RAM samples
        - Startup time (startup_s) is `None` when actual training never started
    """

    start = time.perf_counter()
    process = start_mlagents(command)
    stop_requested = install_stop_handlers(process)

    summaries: list[mlagents_output.RewardSummary] = []
    reader, training_started = start_output_reader(process, summaries)

    sampler = ResourceSampler(process)
    startup_s = None

    if wait_for_training_start(process, training_started):
        startup_s = time.perf_counter() - start
        sampler.start()
        while process.poll() is None:
            time.sleep(MEASURE_INTERVAL_S)
            # process could end while main thread is sleeping
            if process.poll() is None:
                sampler.sample()

    duration_s = time.perf_counter() - start
    reader.join(timeout=5)  # time buffer for ML-Agents to print its final output
    final_steps = mlagents_output.read_final_output_steps(RESULTS_DIR_PATH / run_id)
    hyperparameters = mlagents_output.read_hyperparameters(RESULTS_DIR_PATH / run_id)

    return TrainingResult(
        run_id=run_id,
        exit_code=process.returncode,
        duration_s=duration_s,
        startup_s=startup_s,
        cpu_samples=sampler.cpu_samples,
        background_cpu_samples=sampler.background_cpu_samples,
        ram_samples=sampler.ram_samples,
        summaries=summaries,
        final_steps=final_steps,
        hyperparameters=hyperparameters,
        interrupted=stop_requested.is_set(),
    )


def format_stats(samples: list[float], unit: str) -> str:
    """Produces a nicely formatted mean and max values from a list"""
    if not samples:
        return "no samples"
    average = sum(samples) / len(samples)
    return f"mean {average:.1f}{unit}, peak {max(samples):.1f}{unit}"


def format_table(rows: list[tuple[str, object]]) -> str:
    """Lines of 'label:  value' with the values lined up, e.g. [("run id", "train_1"), ("steps", 36009)]"""
    width = max(len(label) for label, _ in rows) + 1  # +1 for the colon
    return "\n".join(f"{label + ':':<{width}}  {value}" for label, value in rows)


def print_summary(result: TrainingResult, specs: machine_specs.HardwareSpecs) -> None:
    """Display the summary (measurements, exit code and hardware specs)
    of the training (table-like)"""
    startup = (
        f"{result.startup_s:.1f}s"
        if result.startup_s is not None
        else "training never started"
    )
    rows = [
        ("run id", result.run_id),
        ("exit code", result.exit_code),
        ("startup", startup),
        ("duration", f"{result.duration_s:.1f}s"),
        ("CPU training", format_stats(result.cpu_samples, "%")),
        ("CPU background", format_stats(result.background_cpu_samples, "%")),
        ("RAM", format_stats(result.ram_samples, " MB")),
    ]

    if result.summaries:
        last = result.summaries[-1]
        best = max(summary.mean_reward for summary in result.summaries)
        steps = result.final_steps if result.final_steps is not None else last.step
        rows += [
            ("steps", f"{steps} ({len(result.summaries)} reward summaries)"),
            ("throughput", f"{result.step_throughput:.1f} steps/s"),
            (
                "reward",
                f"final {last.mean_reward:.3f} +/- {last.std_reward:.3f}, best {best:.3f}",
            ),
        ]
    else:
        rows.append(("reward", "no reward summaries"))

    if result.hyperparameters is not None:
        hp = result.hyperparameters
        rows.append(
            (
                "settings",
                f"{hp.algo}, lr {hp.learning_rate}, batch {hp.batch_size}, max {hp.max_steps} steps, ...",
            )
        )
    rows.append(("interrupted", result.interrupted))
    print("--- Training ---")
    print(format_table(rows))
    print("\n--- Hardware ---")
    print(machine_specs.get_summary(specs))


def main():
    args: argparse.Namespace = parse_args()
    check_paths(args)
    name, machine = read_identity()

    try:
        hardware = machine_specs.read_specs()
    except (ValueError, TypeError) as error:
        sys.exit(f"Problem in machine_overrides.yaml: {error}")

    command: list[str] = build_command(args)
    print("Running:\n" + "\n".join(command))
    result = execute_training(command, args.run_id)

    print(f"\nUser: {name} on {machine}, experiment type: {args.experiment_type}")
    print_summary(result, hardware)
    sys.exit(result.exit_code)


if __name__ == "__main__":
    main()
