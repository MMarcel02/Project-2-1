"""Reading ML-Agents' output: terminal lines and final results files"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import get_args, get_type_hints

from yaml import YAMLError
from yaml import safe_load as yaml_safe_load

NUMBER_REGEX = r"-?\d+(?:\.\d+)?"  # (negative) integers and/or floats
SUMMARY_LINE_REGEX = re.compile(
    r"\] (?P<behavior>\S+)\. Step: (?P<step>\d+)\. "
    rf"Time Elapsed: (?P<elapsed>{NUMBER_REGEX}) s\. "
    r"(?:.*?\. )?"  # optional parts in between, e.g. "Rank: 0. "
    rf"Mean Reward: (?P<mean>{NUMBER_REGEX})\. "
    r"(?:.*?\. )?"  # e.g. "Mean Group Reward: 1.000. "
    rf"Std of Reward: (?P<std>{NUMBER_REGEX})\."
)  # regex to extract  values of summary lines


@dataclass(frozen=True)
class Hyperparameters:
    """Configuration settings used by ML-Agents for particular training run

    The field names are the database's column names
    """

    algo: str  # "PPO" or "SAC"
    learning_rate: float
    learning_rate_schedule: str
    batch_size: int
    buffer_size: int
    normalize: bool
    hidden_units: int
    num_layers: int
    vis_encode_type: str
    gamma: float
    reward_strength: float
    keep_checkpoints: int
    max_steps: int
    time_horizon: int
    summary_freq: int
    # PPO only (None for SAC)
    ppo_beta: float | None = None
    ppo_epsilon: float | None = None
    ppo_lambd: float | None = None
    ppo_num_epoch: int | None = None
    # SAC only (None for PPO)
    sac_buffer_init_steps: int | None = None
    sac_tau: float | None = None
    sac_steps_per_update: int | None = None
    sac_save_replay_buffer: bool | None = None
    sac_init_entcoef: float | None = None
    sac_reward_signal_steps_per_update: int | None = None


@dataclass(frozen=True)
class RewardSummary:
    """One ML-Agents' summary line"""

    behavior: str
    step: int
    elapsed_s: float
    mean_reward: float
    std_reward: float


def read_hyperparameters(run_dir: Path) -> Hyperparameters | None:
    """Retrieves the configuration settings ML-Agents actually used for training
    (with the included defaults)"""
    try:
        with open(run_dir / "configuration.yaml", encoding="utf-8") as file:
            config = yaml_safe_load(file)
        behavior = next(iter(config["behaviors"].values()))  # the first behavior
    except (
        OSError,
        YAMLError,
        KeyError,
        TypeError,
        AttributeError,
        StopIteration,
    ):
        return None

    hyper = behavior.get("hyperparameters") or {}
    network = behavior.get("network_settings") or {}
    extrinsic = (behavior.get("reward_signals") or {}).get("extrinsic") or {}
    algo = str(behavior.get("trainer_type", "")).upper()

    values = {
        "algo": algo,
        "learning_rate": hyper.get("learning_rate"),
        "learning_rate_schedule": hyper.get("learning_rate_schedule"),
        "batch_size": hyper.get("batch_size"),
        "buffer_size": hyper.get("buffer_size"),
        "normalize": network.get("normalize"),
        "hidden_units": network.get("hidden_units"),
        "num_layers": network.get("num_layers"),
        "vis_encode_type": network.get("vis_encode_type"),
        "gamma": extrinsic.get("gamma"),
        "reward_strength": extrinsic.get("strength"),
        "keep_checkpoints": behavior.get("keep_checkpoints"),
        "max_steps": behavior.get("max_steps"),
        "time_horizon": behavior.get("time_horizon"),
        "summary_freq": behavior.get("summary_freq"),
    }
    if algo == "PPO":
        values["ppo_beta"] = hyper.get("beta")
        values["ppo_epsilon"] = hyper.get("epsilon")
        values["ppo_lambd"] = hyper.get("lambd")
        values["ppo_num_epoch"] = hyper.get("num_epoch")
    elif algo == "SAC":
        values["sac_buffer_init_steps"] = hyper.get("buffer_init_steps")
        values["sac_tau"] = hyper.get("tau")
        values["sac_steps_per_update"] = hyper.get("steps_per_update")
        values["sac_save_replay_buffer"] = hyper.get("save_replay_buffer")
        values["sac_init_entcoef"] = hyper.get("init_entcoef")
        values["sac_reward_signal_steps_per_update"] = hyper.get(
            "reward_signal_steps_per_update"
        )
    return Hyperparameters(**values)


def get_missing_hyperparameters(hyperparameters: Hyperparameters) -> list[str]:
    """Returns any missing required parameters"""
    missing = []
    for name, expected in get_type_hints(Hyperparameters).items():
        optional = type(None) in get_args(expected)
        if not optional and getattr(hyperparameters, name) is None:
            missing.append(name)
    return missing


def read_final_output_steps(run_dir: Path) -> int | None:
    """Retrieves the exact number of steps for training (`integer` or `None` if unavailable)"""
    status_path = run_dir / "run_logs" / "training_status.json"
    try:
        with open(status_path, encoding="utf-8") as file:
            status = json.load(file)
    except (OSError, ValueError):
        return None
    for behavior, info in status.items():
        if behavior == "metadata":
            continue
        try:
            return int(info["final_checkpoint"]["steps"])
        except (KeyError, TypeError, ValueError):
            return None
    return None


def parse_summary_line(line: str) -> RewardSummary | None:
    """Read the summary line ML-Agents produces and return the values
    or None for anything else"""
    match = SUMMARY_LINE_REGEX.search(line)
    if not match:
        return None
    return RewardSummary(
        behavior=match["behavior"],
        step=int(match["step"]),
        elapsed_s=float(match["elapsed"]),
        mean_reward=float(match["mean"]),
        std_reward=float(match["std"]),
    )


if __name__ == "__main__":
    examples = [
        # parsed line
        "[INFO] 3DBall. Step: 24000. Time Elapsed: 40.512 s. Mean Reward: 12.345. Std of Reward: 3.210. Training.",
        # irrelevant lines
        "[INFO] 3DBall. Step: 12000. Time Elapsed: 20.113 s. No episode was completed since last summary. Training.",
        "[INFO] Connected new brain: 3DBall?team=0",
    ]

    for line in examples:
        print(parse_summary_line(line))
