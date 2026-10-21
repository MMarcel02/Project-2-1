CREATE TABLE IF NOT EXISTS our_run_data(
    -- RUN Identification
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    synced_at BIGINT,
    machine_id TEXT NOT NULL,
    experiment_type TEXT NOT NULL,
    os_name TEXT NOT NULL,

    -- Pre-run cpu hw specs
    cpu_model TEXT NOT NULL,
    cpu_physical_cores_total INTEGER NOT NULL,
    cpu_logical_cores_total INTEGER NOT NULL,
    cpu_p_cores_allocated INTEGER NOT NULL,
    cpu_e_cores_allocated INTEGER NOT NULL,
    cpu_p_core_max_ghz REAL NOT NULL,
    cpu_e_core_max_ghz REAL NOT NULL,
    ram_total_mb INTEGER NOT NULL,
    ram_allocated_limit_mb INTEGER NOT NULL,
    ram_speed_mhz INTEGER NOT NULL,
    ram_type TEXT NOT NULL,

    -- Hyperparams for rq2
    algo TEXT NOT NULL,
    seed INTEGER NOT NULL,
    learning_rate REAL NOT NULL,
    learning_rate_schedule TEXT NOT NULL,
    batch_size INTEGER NOT NULL,
    buffer_size INTEGER NOT NULL,
    normalize INTEGER NOT NULL,
    hidden_units INTEGER NOT NULL,
    num_layers INTEGER NOT NULL,
    vis_encode_type TEXT NOT NULL,
    gamma REAL NOT NULL,
    reward_strength REAL NOT NULL,
    keep_checkpoints INTEGER NOT NULL,
    max_steps INTEGER NOT NULL,
    time_horizon INTEGER NOT NULL,
    summary_freq INTEGER NOT NULL,

    -- PPO specific (NULL if algo == "SAC")
    ppo_beta REAL,
    ppo_epsilon REAL,
    ppo_lambd REAL,
    ppo_num_epoch INTEGER,

    -- SAC specific (NULL if algo == "PPO")
    sac_buffer_init_steps INTEGER,
    sac_tau REAL,
    sac_steps_per_update INTEGER,
    sac_save_replay_buffer INTEGER, -- Stored as 0 or 1 for boolean
    sac_init_entcoef REAL,
    sac_reward_signal_steps_per_update INTEGER,

    -- Runtime data
    ml_agents_avg_cpu_util REAL NOT NULL,
    ml_agents_peak_cpu_util REAL NOT NULL,
    background_avg_cpu_util REAL NOT NULL,
    p_core_avg_util REAL NOT NULL, -- 0 if no p cores
    e_core_avg_util REAL NOT NULL, -- 0 if no e cores
    avg_effective_clock_ghz REAL NOT NULL,
    avg_ram_usage_mb REAL NOT NULL,
    peak_ram_usage_mb REAL NOT NULL,

    -- Actual targets want to capture
    total_steps_executed INTEGER NOT NULL,
    final_mean_reward REAL NOT NULL,
    final_std_reward REAL NOT NULL,
    convergence_threshold REAL NOT NULL,
    consecutive_hits_above_threshold INTEGER NOT NULL,
    converged INTEGER NOT NULL, -- Stored as 0 or 1 for boolean

    -- Final targets
    step_throughput REAL NOT NULL,
    steps_to_convergence INTEGER NOT NULL,
    time_to_convergence_s REAL NOT NULL
);


CREATE TABLE IF NOT EXISTS external_run_data(
    -- RUN Identification
    id TEXT PRIMARY KEY,
    synced_at BIGINT,
    group_name TEXT NOT NULL,
    environment_type TEXT NOT NULL,
    os_name TEXT,

    -- Pre-run cpu hw specs
    cpu_model TEXT ,
    cpu_physical_cores_total INTEGER ,
    cpu_logical_cores_total INTEGER ,
    cpu_p_cores_allocated INTEGER ,
    cpu_e_cores_allocated INTEGER ,
    cpu_p_core_max_ghz REAL ,
    cpu_e_core_max_ghz REAL ,
    ram_total_mb INTEGER ,
    ram_allocated_limit_mb INTEGER,
    ram_speed_mhz INTEGER,
    ram_type TEXT,

    -- Hyperparams for rq2
    algo TEXT ,
    seed INTEGER ,
    learning_rate REAL ,
    learning_rate_schedule TEXT ,
    batch_size INTEGER ,
    buffer_size INTEGER ,
    normalize INTEGER ,
    hidden_units INTEGER ,
    num_layers INTEGER ,
    vis_encode_type TEXT ,
    gamma REAL ,
    reward_strength REAL ,
    keep_checkpoints INTEGER ,
    max_steps INTEGER ,
    time_horizon INTEGER ,
    summary_freq INTEGER ,

    -- PPO specific (NULL if algo == "SAC")
    ppo_beta REAL,
    ppo_epsilon REAL,
    ppo_lambd REAL,
    ppo_num_epoch INTEGER,

    -- SAC specific (NULL if algo == "PPO")
    sac_buffer_init_steps INTEGER,
    sac_tau REAL,
    sac_steps_per_update INTEGER,
    sac_save_replay_buffer INTEGER, -- Stored as 0 or 1 for boolean
    sac_init_entcoef REAL,
    sac_reward_signal_steps_per_update INTEGER,

    -- Actual targets want to capture
    total_steps_executed INTEGER NOT NULL,
    final_mean_reward REAL NOT NULL,
    final_std_reward REAL NOT NULL,
    convergence_threshold REAL NOT NULL,
    consecutive_hits_above_threshold INTEGER NOT NULL,
    converged INTEGER NOT NULL, -- Stored as 0 or 1 for boolean

    -- Final targets
    step_throughput REAL NOT NULL,
    steps_to_convergence INTEGER NOT NULL,
    time_to_convergence_s REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_experiment_type ON our_run_data(experiment_type);
