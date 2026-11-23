from pathlib import Path
import json
import math

from configuration.generate_config import (
    ROOT,
    TRAINING_PARAMETERS_ALL_RANGES,
    TRAINING_PARAMETERS_CURRENT_RANGES,
    HARDWARE_ALL_RANGES,
    HARDWARE_CURRENT_RANGES,
)

# TODO (integration): Verify that RunData values match our configuration values.
# - cpu_cores: Determine whether to use P-cores, E-cores, or their sum.
# - memory_limit_gb: Convert ram_allocated_limit_mb from MB to GB.
# - Verify that all training parameter field names match RunData.
# Incorrect mappings will produce incorrect sampling frequencies. -> Marcel
# Delete this after completion as this was an ai comment - too lazy to modify it tho the code is human

PPO_SUPABASE_MAP = {
    "beta": "ppo_beta",
    "epsilon": "ppo_epsilon",
    "lambd": "ppo_lambd",
    "num_epoch": "ppo_num_epoch",
    "cpu_cores": "cpu_p_cores_allocated",
    "memory_limit_gb": "ram_allocated_limit_mb",
}

def adapt_ranges(completed_runs):
    hardware_runs = [
        run for run in completed_runs
        if run.experiment_type in ("alter_hardware", "alter_both")
    ]

    training_parameter_runs = [
        run for run in completed_runs
        if run.experiment_type in ("alter_parameters", "alter_both")
    ]

    with open(HARDWARE_ALL_RANGES, "r", encoding="utf-8") as file:
        all_hardware_definitions = json.load(file)

    with open(HARDWARE_CURRENT_RANGES, "r", encoding="utf-8") as file:
        current_hardware_ranges = json.load(file)

    unlock_parameters_for_randomization(
        all_definitions=all_hardware_definitions, 
        current_ranges=current_hardware_ranges, 
        completed_count=len(hardware_runs)
    )

    with open(TRAINING_PARAMETERS_ALL_RANGES, "r", encoding="utf-8") as file:
        all_training_definitions = json.load(file)

    with open(TRAINING_PARAMETERS_CURRENT_RANGES, "r", encoding="utf-8") as file:
        current_training_ranges = json.load(file)

    unlock_parameters_for_randomization(
        all_definitions=all_training_definitions, 
        current_ranges=current_training_ranges, 
        completed_count=len(training_parameter_runs)
    )

    if hardware_runs:
        adapt_choice_ranges(hardware_runs, current_hardware_ranges)
        adapt_continuous_ranges(hardware_runs, current_hardware_ranges)

    if training_parameter_runs:
        adapt_choice_ranges(training_parameter_runs, current_training_ranges)
        adapt_continuous_ranges(training_parameter_runs, current_training_ranges)

def adapt_choice_ranges(runs, current_ranges):
    for name, settings in current_ranges.items():
        if not settings.get("active", False):
            continue

        if settings["sampling"] != "choice":
            continue

        counts = count_choice_frequencies(runs, name, settings)
        settings["weights"] = calculate_new_weights(counts)


def count_choice_frequencies(runs, parameter_name, settings):
    values = settings["values"]
    field_name = PPO_SUPABASE_MAP.get(parameter_name, parameter_name) #Ensure we get the right RunData name

    counts = {}

    for value in values:
        counts[value] = 0

    for run in runs:
        value = getattr(run, field_name) #example value = run.learning_rate, as RunData is a data class and not a dictionary

        if value in counts:
            counts[value]+=1

    return counts

def adapt_continuous_ranges(runs, current_ranges):
    for name, settings in current_ranges.items():
        if not settings.get("active", False):
                    continue
        
        if settings["sampling"] in ("uniform", "log_uniform"):
            minimum = settings["min"]
            maximum = settings["max"]
            bins = settings["bins"]

            counts = count_continuous_frequencies(runs, name, minimum, maximum, bins, settings)
            settings["weights"] = calculate_new_weights(counts)

        else:
            continue

def count_continuous_frequencies(runs, parameter_name, minimum, maximum, bins, settings):
    field_name = PPO_SUPABASE_MAP.get(parameter_name, parameter_name)

    counts = {}
    for index in range(bins):
        counts[bin] = 0
    
    for run in runs:
        value = getattr(run, field_name)
    
        if settings["sampling"] == "uniform":
            index = get_uniform_bin_index(value, minimum, maximum, bins)
        else:
            index = get_log_uniform_bin_index(value, minimum, maximum, bins)

        if index is not None:
            counts[bin] +=1

    return counts

        

def calculate_new_weights(counts):
    weights = []

    for count in counts.values():
        weights.append(1 / (count + 1))

    return weights

def get_log_uniform_bin_index(value, minimum, maximum, bins):
    if value < minimum or value > maximum:
        return None

    if value == maximum:
        return bins-1

    log_minimum = math.log(minimum)
    log_maximum = math.log(maximum)
    log_value = math.log(value)

    bin_width = (log_maximum - log_minimum) / bins
    index = int((log_value - log_minimum) / bin_width)

    return index
    

def get_uniform_bin_index(value, minimum, maximum, bins):
    if value < minimum or value > maximum:
        return None

    if value == maximum:
        return bins-1

    bin_width = (maximum - minimum) / bins
    index = int((value - minimum)/ bin_width)

    return index

def unlock_parameters_for_randomization(all_definitions, current_ranges, completed_count):
    for name, definition in all_definitions.items():
        if not definition.get("randomizable", True) or not definition.get("meaningful", True):
            continue


        unlock_after_value = definition.get("unlock_after", 0)

        if completed_count >= unlock_after_value:
            current_ranges[name]["active"] = True

