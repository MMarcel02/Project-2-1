"""TODO DOCSTRING"""

import argparse
import os
import random
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from signal import SIGINT, SIGTERM, signal

import psutil
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from storage.run_data_model import ExperimentType

import machine_specs
import mlagents_output
import training_record

BUILDS = {
    "win32": Path("windows") / "3DBall.exe",
    "linux": Path("linux") / "3DBall.x86_64",
    "darwin": Path("macos") / "3DBall.app",
}

DEFAULT_UNITY_BUILD_PATH = ROOT / "builds" / BUILDS.get(sys.platform, BUILDS["linux"])
DEFAULT_TRAINING_CONFIG_PATH = ROOT / "runner" / "configs" / "training" / "3DBall.yaml"
DEFAULT_DATABASE_PATH = ROOT / "storage" / "runs.db"
RESULTS_DIR_PATH = ROOT / "results"

EXIT_MESSAGE = "Stopping, please wait..."

MAX_SEED = 2**31 - 1
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
    try:
        return Path("<project_root>") / path.relative_to(ROOT)
    except ValueError:
        return path


def parse_args() -> argparse.Namespace:
    """Create CLI-like arguments (with default values, type safety and documentation),
    then build and return them"""
    generated_run_id = "train_" + time.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(
        description="Run ML-Agents trainings, one per config"
    )
    parser.add_argument(
        "--config",
        type=Path,
        nargs="+",  # one or more values
        default=[DEFAULT_TRAINING_CONFIG_PATH],
        help=(
            "training config(s): (1+) YAML files and/or folder(s); a folder means all its *.yaml files, in name order"
            f" (default: {show_project_relative(DEFAULT_TRAINING_CONFIG_PATH)})"
        ),
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
        default=None,
        help="the first seed for same/increment seed modes, or the seed used for randomizing training seeds (default: random number between 0 and 2147483647)",
    )
    parser.add_argument(
        "--seed-mode",
        choices=["same", "increment", "random"],
        default="same",
        help=(
            "same: every run uses --seed;"
            " increment: run 1 uses --seed, run 2 --seed + 1, ...;"
            " random: a new random seed generated per run (default: same);"
            " if --seed is not specified, random --seed will be used"
        ),
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
    """Stops if the unity build doesn't exist; configs are now checked by `collect_configs()`"""
    args.env = args.env.resolve()
    if not args.env.exists():
        sys.exit(f"Not found: {args.env.resolve()}")


def collect_configs(paths: list[Path]) -> list[Path]:
    """Retrieves all config files for training: files as they are,
    for folders searches for *.yaml files"""
    configs: list[Path] = []
    for path in paths:
        if path.is_dir():
            found = sorted(path.glob("*.yaml"))
            if not found:
                sys.exit(f"No .yaml files in {path.resolve()}")
            configs.extend(found)  # append all items from list to another list
        elif path.is_file():
            configs.append(path)
        else:
            sys.exit(f"Not found: {path.resolve()}")

    by_name: dict[str, list[Path]] = {}
    for config in configs:
        # <Path>.stem keeps the file name without extension
        by_name.setdefault(config.stem, []).append(config)

    duplicates = {name: paths for name, paths in by_name.items() if len(paths) > 1}
    if duplicates:
        name_path_lines = [
            f"  {name}: {', '.join(str(path) for path in paths)}"
            for name, paths in sorted(duplicates.items())
        ]
        sys.exit(
            "Config file names must be unique (they are part of the run_id)"
            "\nduplicates:\n" + "\n".join(name_path_lines)
        )
    return [config.resolve() for config in configs]


def read_behavior_name(config: Path) -> str:
    """Retrieves the behavior name of a training config; stops if unable to read file, or improper config"""
    try:
        with open(config, encoding="utf-8") as file:
            content = yaml.safe_load(file)
    except OSError as error:
        sys.exit(f"Cannot read {config}: {error}")
    except yaml.YAMLError as error:
        sys.exit(f"{config} is not valid YAML: {error}")

    behaviors = content.get("behaviors") if isinstance(content, dict) else None
    if not isinstance(behaviors, dict) or len(behaviors) != 1:
        sys.exit(
            f"{config} must define exactly one behavior; found: {len(behaviors or {})}"
        )

    name = next(iter(behaviors))
    settings = behaviors[name] or {}
    if not isinstance(settings, dict):
        sys.exit(
            f"{show_project_relative(config)}: the settings of {name} must be 'name: value' lines"
        )
    for key in ("summary_freq", "max_steps", "time_horizon"):
        value = settings.get(key)
        if value is not None and (not isinstance(value, int) or value <= 0):
            sys.exit(
                f"{show_project_relative(config)}:{key} must be whole number above 0; found {value!r}"
            )

    return name  # the name of the single behavior


def check_configs(configs: list[Path]) -> str:
    """Checks if all configs are for the same behavior; Returns the behavior's name"""
    expected = read_behavior_name(
        configs[0]
    )  # set the first config as expected behavior
    for config in configs[1:]:
        name = read_behavior_name(config)
        if name != expected:
            sys.exit(
                "All configs must be for the same behavior:\n"
                f" expected behavior is {expected} (from {configs[0]})\n"
                f" {config} is for {name}\n"
            )
    return expected


def check_build_matches(env: Path, behavior: str) -> None:
    """IMPORTANT: assumes builds are named after the behavior inside them (builds/.../3DBall.exe -> 3DBall)

    Stops if the unity build name doesn't match the config's behavior
    """
    build_name = env.stem
    if build_name != behavior:
        sys.exit(
            f"The provided configs are for behavior {behavior}, but the build is {env.name} (behavior {build_name})"
        )


def seeds_for_runs(first: int | None, mode: str, count: int) -> list[int]:
    """Produces the seeds for each run"""
    if first is not None and not 0 <= first <= MAX_SEED:
        sys.exit(
            f"--seed must be between 0 and {MAX_SEED} (inclusive for both); got {first}"
        )
    if mode == "random":
        generator = random.Random(first)
        return [generator.randint(0, MAX_SEED) for _ in range(count)]

    if first is None:
        first = random.randint(0, MAX_SEED - count + 1)
    if mode == "same":
        return [first] * count
    seeds = list(range(first, first + count))
    if seeds[-1] > MAX_SEED:
        sys.exit(
            f"--seed {first} + {count - 1} runs goes beyond {MAX_SEED}; use a smaller --seed"
        )
    return seeds


def build_command(
    args: argparse.Namespace, config: Path, run_id: str, seed: int
) -> list[str]:
    """Builds a complete ML-Agents command from the passed arguments ready to execute"""
    command = [
        sys.executable,
        "-m",
        "mlagents.trainers.learn",
        str(config),
        f"--env={args.env}",
        f"--run-id={run_id}",
        f"--seed={seed}",
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


def install_stop_handlers(
    process: psutil.Popen, stop_requested: threading.Event
) -> None:
    """Provides safe stop/terminate logic, and records if it has occured

    For mutliple training runs stops the whole batch
    """

    def on_keyboard_interrupt(signum, frame):
        stop_requested.set()
        print(EXIT_MESSAGE)

    def on_terminate(signum, frame):
        stop_requested.set()
        print(EXIT_MESSAGE)
        try:
            process.send_signal(SIGINT)
        except psutil.Error:
            pass

    signal(SIGINT, on_keyboard_interrupt)
    if sys.platform != "win32":
        signal(SIGTERM, on_terminate)


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


def execute_training(
    command: list[str], run_id: str, stop_requested: threading.Event
) -> TrainingResult:
    """Runs single ML-agents training and measures it

    Returns:
        - Exit code
        - Duration (total and startup time in seconds)
        - CPU and RAM samples
        - Startup time (startup_s) is `None` when actual training never started
    """

    start = time.perf_counter()
    process = start_mlagents(command)
    install_stop_handlers(process, stop_requested)

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


def format_columns(rows: Sequence[tuple[str, object]]) -> str:
    """Lines of 'label:  value' with the values lined up, e.g. [("run id", "train_1"), ("steps", 36009)]"""
    width = max(len(label) for label, _ in rows) + 1  # +1 for the colon
    return "\n".join(f"{label + ':':<{width}}  {value}" for label, value in rows)


def format_table(rows: Sequence[Sequence[object]], separator: str = " | ") -> str:
    """Creates a dynamically generated table and returns it"""
    texts = [[str(cell) for cell in row] for row in rows]
    widths = [max(len(cell) for cell in column) for column in zip(*texts)]
    return "\n".join(
        separator.join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip()
        for row in texts
    )


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
    print(format_columns(rows))
    print("\n--- Hardware ---")
    print(machine_specs.get_summary(specs))


def print_batch_summary(
    configs: list[Path], outcomes: list[tuple[Path, int, TrainingResult, bool]]
) -> None:
    """Displays the summary of all executed trainings"""

    rows: list[tuple[object, ...]] = [
        ("config", "seed", "status", "throughput", "final reward")
    ]
    for config, seed, result, saved in outcomes:
        if saved:
            status = "saved"
        elif result.interrupted:
            status = "stopped, not saved"
        else:
            status = f"not saved (exit code {result.exit_code})"
        throughput = (
            f"{result.step_throughput:.0f} steps/s"
            if result.step_throughput is not None
            else "-"
        )
        final_reward = (
            f"{result.summaries[-1].mean_reward:.3f}" if result.summaries else "-"
        )
        rows.append((config.stem, seed, status, throughput, final_reward))

    for config in configs[len(outcomes) :]:
        rows.append((config.stem, "-", "skipped", "-", "-"))

    print(f"\n----- Batch: {len(outcomes)} of {len(configs)} run(s) -----")
    print(format_table(rows))


def record_run(
    result: TrainingResult,
    specs: machine_specs.HardwareSpecs,
    *,  # pass next arguments by name (positional passing is forbidden)
    name: str,
    machine: str,
    experiment_type: str,
    seed: int,
) -> bool:
    """Saves recorded metrics after a finished run to local database;
    if it fails prints why; returns True if successful, False otherwise
    """
    reasons = training_record.get_reasons_for_failed_save(result)
    if reasons:
        print(f"\nNot saved to the database: {'; '.join(reasons)}")
        return False
    row = training_record.build_run_data(
        result,
        specs,
        name=name,
        machine=machine,
        experiment_type=experiment_type,
        seed=seed,
    )
    try:
        training_record.save_to_database(row)
    except sqlite3.Error as error:
        print(f"\nCould not save to the database: {error}")
        return False
    print(
        f"\nSaved to the database {show_project_relative(DEFAULT_DATABASE_PATH)} as {row.id}"
    )
    return True


def main():
    args: argparse.Namespace = parse_args()
    check_paths(args)
    configs = collect_configs(args.config)
    behavior = check_configs(configs)
    check_build_matches(args.env, behavior)
    print(f"Configs ({len(configs)}), behavior {behavior}:")
    for config in configs:
        print(f"  {config}")
    name, machine = read_identity()

    seeds = seeds_for_runs(args.seed, args.seed_mode, len(configs))

    try:
        hardware = machine_specs.read_specs()
    except (ValueError, TypeError) as error:
        sys.exit(f"Problem in machine_overrides.yaml: {error}")

    stop_requested = threading.Event()
    results: list[tuple[Path, int, TrainingResult, bool]] = []
    for number, (config, seed) in enumerate(zip(configs, seeds), start=1):
        if stop_requested.is_set():
            remaining = len(configs) - number + 1
            print(
                f"\nStopped by request: {number - 1} of {len(configs)} runs completed, {remaining} config(s) skipped"
            )
            break
        run_id = f"{args.run_id}_{config.stem}"
        print(
            f"\n----- Run {number}/{len(configs)}: {show_project_relative(config)} (run id {run_id}, seed {seed}) -----"
        )

        command = build_command(args, config, run_id, seed)
        print("Running:\n" + "\n".join(command))
        result = execute_training(command, run_id, stop_requested)

        print(f"\nUser: {name} on {machine}, experiment type: {args.experiment_type}")
        print_summary(result, hardware)
        saved_record = record_run(
            result,
            hardware,
            name=name,
            machine=machine,
            experiment_type=args.experiment_type,
            seed=seed,
        )
        results.append((config, seed, result, saved_record))

    print_batch_summary(configs, results)
    all_saved = len(results) == len(configs) and all(saved for *_, saved in results)

    sys.exit(0 if all_saved else 1)


if __name__ == "__main__":
    main()
