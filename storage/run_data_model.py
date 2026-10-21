from dataclasses import dataclass
from enum import Enum


# Adding this as a col in our data so we can easily separate for each research question
class ExperimentType(str, Enum):
    TEST_ONLY = "test_only"
    ALTER_HARDWARE = "alter_hardware"
    ALTER_PARAMETERS = "alter_parameters"
    ALTER_BOTH = "alter_both"


# Avoids assigning dict for every object and read only
@dataclass(slots=True, frozen=True)
class RunData:
    # RUN Identification
    id: str  # UUID for local/Supabase sync
    name: str  # Name of person doing the run
    machine_id: (
        str  # Identifier of machine (in case someone does runs on 2 diff machines)
    )
    experiment_type: str  # Identifies which phase/rq this run was for
    os_name: str  # Windows / Ubuntu / macOS

    # Pre-run cpu hw specs (dont care about gpu since will run with no graphics)
    cpu_model: str  # e.g., "Intel i7-13700H" or "Apple M2"
    cpu_physical_cores_total: int  # Total physical cores on the machine
    cpu_logical_cores_total: int  # Total threads on the machine
    cpu_p_cores_allocated: int  # P-cores pinned via psutil cpu_affinity
    cpu_e_cores_allocated: int  # E-cores pinned via psutil cpu_affinity
    cpu_p_core_max_ghz: float  # Max rated clock speed of P-cores
    cpu_e_core_max_ghz: float  # Max rated clock speed of E-cores (0.0 if none)
    ram_total_mb: int  # Total system RAM in MB
    ram_allocated_limit_mb: (
        int | None
    )  # Artificial RAM limit set via psutil (if throttled)
    ram_speed_mhz: int | None  # RAM speed (bandwidth proxy, e.g., 3200, 6400)
    ram_type: str | None  # e.g., "DDR4", "DDR5", "LPDDR5"

    # Hyperparams for rq2
    algo: str  # "PPO" or "SAC"
    seed: int  # Random seed
    learning_rate: float
    learning_rate_schedule: str  # e.g., "linear", "constant"
    batch_size: int
    buffer_size: int
    normalize: bool  # Maps to INTEGER (0 or 1) in SQLite
    hidden_units: int
    num_layers: int
    vis_encode_type: str  # e.g., "simple"
    gamma: float  # Discount factor
    reward_strength: float  # Extrinsic reward strength
    keep_checkpoints: int
    max_steps: int  # Planned maximum step threshold
    time_horizon: int
    summary_freq: int

    # PPO specific (None if algo == "SAC")
    ppo_beta: float | None
    ppo_epsilon: float | None
    ppo_lambd: float | None
    ppo_num_epoch: int | None
    # SAC specific (None if algo == "PPO")
    sac_buffer_init_steps: int | None
    sac_tau: float | None
    sac_steps_per_update: int | None
    sac_save_replay_buffer: bool | None
    sac_init_entcoef: float | None
    sac_reward_signal_steps_per_update: int | None

    # Runtime data, can also use to predict throughput at current moment (rq3)
    ml_agents_avg_cpu_util: float  # Avg CPU % used specifically by Unity + Python
    ml_agents_peak_cpu_util: float  # Peak CPU % used by Unity + Python
    background_avg_cpu_util: (
        float  # Avg CPU % used by everything else (gaming/OS noise)
    )
    p_core_avg_util: float | None  # Avg % utilization across allocated P-cores
    e_core_avg_util: float | None  # Avg % utilization across allocated E-cores
    avg_effective_clock_ghz: (
        float  # Actual average clock speed during run (catches thermal throttling)
    )
    avg_ram_usage_mb: float  # Avg RAM used by the training processes
    peak_ram_usage_mb: float  # Peak RAM used by the training processes

    # Actual targets want to capture
    total_steps_executed: int  # Actual steps completed before termination
    final_mean_reward: float  # Final evaluation score
    final_std_reward: float  # Final standard deviation of reward
    convergence_threshold: float  # Score threshold required to consider task solved
    consecutive_hits_above_threshold: (
        int  # Only consider converged if hits above threshold this many times
    )
    converged: bool  # True (1) if solved, False (0) if hit max_steps

    # Final targets (a bit redundant as can be calculated on fly but easier to just store)
    step_throughput: float  # RQ1 Target: Total steps / train_duration_s (steps/sec)
    steps_to_convergence: (
        int | None
    )  # RQ2 Target: Total steps until we mark it as converged
    time_to_convergence_s: float | None  # RQ3 Target: Total wall-clock time in seconds
