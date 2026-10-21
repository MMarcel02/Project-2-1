import uuid

from storage import cloud_store, local_store, run_data_model, storage_scripts


def create_sample_run_data(name, machine) -> run_data_model.RunData:
    run_data = run_data_model.RunData(
        # --- RUN Identification ---
        id=str(uuid.uuid4()),
        name=name,
        machine_id=machine,
        experiment_type=run_data_model.ExperimentType.TEST_ONLY.value,
        os_name="Windows",
        # --- Pre-run cpu hw specs ---
        cpu_model="Intel i7-13700H",
        cpu_physical_cores_total=14,
        cpu_logical_cores_total=20,
        cpu_p_cores_allocated=6,
        cpu_e_cores_allocated=8,
        cpu_p_core_max_ghz=4.8,
        cpu_e_core_max_ghz=3.7,
        ram_total_mb=16384,
        ram_allocated_limit_mb=1000,
        ram_speed_mhz=3200,
        ram_type="DDR4",
        # --- Hyperparams for rq2 ---
        algo="PPO",
        seed=42,
        learning_rate=0.0003,
        learning_rate_schedule="linear",
        batch_size=64,
        buffer_size=12000,
        normalize=True,
        hidden_units=128,
        num_layers=2,
        vis_encode_type="simple",
        gamma=0.99,
        reward_strength=1.0,
        keep_checkpoints=5,
        max_steps=500000,
        time_horizon=64,
        summary_freq=5000,
        # --- PPO specific (Populated) ---
        ppo_beta=0.01,
        ppo_epsilon=0.2,
        ppo_lambd=0.95,
        ppo_num_epoch=3,
        # --- SAC specific (None for PPO runs) ---
        sac_buffer_init_steps=None,
        sac_tau=None,
        sac_steps_per_update=None,
        sac_save_replay_buffer=None,
        sac_init_entcoef=None,
        sac_reward_signal_steps_per_update=None,
        # --- Runtime telemetry data ---
        ml_agents_avg_cpu_util=35.5,
        ml_agents_peak_cpu_util=78.2,
        background_avg_cpu_util=12.0,
        p_core_avg_util=42.1,
        e_core_avg_util=18.5,
        avg_effective_clock_ghz=4.1,
        avg_ram_usage_mb=1024.5,
        peak_ram_usage_mb=1450.0,
        # --- Actual targets captured ---
        total_steps_executed=150000,
        final_mean_reward=102.5,
        final_std_reward=3.2,
        convergence_threshold=100.0,
        consecutive_hits_above_threshold=3,
        converged=True,
        # --- Final targets ---
        step_throughput=1250.4,  # total_steps_executed / train_duration_s
        steps_to_convergence=150000,  # Matches steps since it converged
        time_to_convergence_s=119.96,  # Total wall-clock time
    )
    return run_data


def main():
    # Example interaction
    # init once
    name, machine, sqlite, supabase = storage_scripts.init()
    # data gathering loop that fills run_data
    run_data = create_sample_run_data(name, machine)
    # immediately save run data locally
    local_store.insert_run(sqlite, "our_run_data", run_data)
    # on each uploading try can try to also sync all unuploaded stuff
    storage_scripts.sync_with_cloud(sqlite, supabase)


if __name__ == "__main__":
    main()
