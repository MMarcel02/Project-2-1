from pathlib import Path
import yaml
import random
import json
import math

ROOT = Path(__file__).resolve().parent.parent # Parent of this parent folder is root

BASE_CONFIG_TEMPLATE = ROOT / "ml-agents" / "config" / "ppo" / "3DBallHard.yaml"
OUTPUT_DIRECTORY = ROOT / "configuration" / "generated_configs"

PLAN_PATH = OUTPUT_DIRECTORY.parent / "experiment_plan.json"

MAX_EXPERIMENT_PLAN_SIZE = 25

TRAINING_PARAMETERS_ALL_RANGES = ROOT / "configuration" / "training_parameters" / "all_ranges.json"
TRAINING_PARAMETERS_CURRENT_RANGES = ROOT / "configuration" / "training_parameters" / "current_ranges.json"

HARDWARE_ALL_RANGES = ROOT / "configuration" / "hardware" / "all_ranges.json"
HARDWARE_CURRENT_RANGES = ROOT / "configuration" / "hardware" / "current_ranges.json"

ADAPTABILITY_MODE = True #Set to true if you want an experiment plan that adapts to what data is most lacking in SUPABASE
MEANINGFUL_MODE = True #Set to true if you only want parameters that are most likely to affect to training performances to be randomized

AUTO_PLAN = False #Set to true if you want automatic planning of new experiments

RESEARCH_QUESTION_FOCUS = "RQ3" 
# RQ1 randomizes only hardware features
# RQ2 randomizes only reinforcement learning hyperparameters
# RQ3 randomizes both

DEFAULT_HARDWARE = {
    "cpu_cores": 4,
    "memory_limit_gb": None
} # TODO: Add the additional hardware features


def generate_config(**training_parameters):
    with open(BASE_CONFIG_TEMPLATE, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    with open(TRAINING_PARAMETERS_ALL_RANGES, "r", encoding="utf-8") as file:
        parameter_definitions = json.load(file)

    ball_config = config["behaviors"]["3DBallHard"]

    for parameter, value in training_parameters.items():
        if parameter not in parameter_definitions:
            raise ValueError(f"Unknown training parameter: {parameter}")

        definition = parameter_definitions[parameter]
        section = definition["section"]
        yaml_key = definition.get("yaml_key", parameter)

        target = ball_config

        if section != "behavior":
            for section_name in section.split("."):
                target = target[section_name]

        target[yaml_key] = value

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIRECTORY / f"3DBallHard_current.yaml"

    with open(output_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    return output_path

def sample_parameters(rng, all_ranges_path, current_ranges_path):
    with open(all_ranges_path, "r", encoding="utf-8") as file:
        all_ranges = json.load(file)

    if ADAPTABILITY_MODE:
        with open(current_ranges_path, "r", encoding="utf-8") as file:
            ranges = json.load(file)
    else:
        ranges = all_ranges

    selected = {}

    for name, settings in ranges.items():
        feature = all_ranges[name]

        if not feature.get("randomizable", True):
            continue

        if not feature.get("meaningful", True) and MEANINGFUL_MODE:
            continue

        if ADAPTABILITY_MODE and not settings.get("active", False):
            continue

        sampling = settings["sampling"]

        if sampling == "choice":
            selected[name] = rng.choices(
                settings["values"],
                weights=settings.get("weights"),
                k=1
            )[0]
        elif sampling in ("uniform", "log_uniform"):
            minimum = settings["min"]
            maximum = settings["max"]

            if sampling == "log_uniform":
                minimum = math.log(minimum)
                maximum = math.log(maximum)

            if "bins" in settings:
                bins = settings["bins"]

                bin_index = rng.choices(
                    range(bins),
                    weights=settings.get("weights"),
                    k=1
                )[0]

                width = (maximum - minimum) / bins

                bin_min = minimum + bin_index * width
                bin_max = minimum + (bin_index + 1) * width

                minimum = bin_min
                maximum = bin_max
                
            value = rng.uniform(minimum, maximum)

            if sampling == "log_uniform":
                value = math.exp(value)

            selected[name] = float(f"{value:.6g}") #Ensures values are rounded to six significant digits

    return selected

def generate_experiment_plan(count=25, seed=42):

    if not 1 <= count <= MAX_EXPERIMENT_PLAN_SIZE:
        raise ValueError(f"Count must be between 1 and {MAX_EXPERIMENT_PLAN_SIZE}")
    
    rng = random.Random(seed)

    if RESEARCH_QUESTION_FOCUS not in ("RQ1", "RQ2", "RQ3"):
        raise ValueError(f"Invalid focus: {RESEARCH_QUESTION_FOCUS}")

    if PLAN_PATH.exists():
        with open(PLAN_PATH, "r", encoding="utf-8") as file:
            experiments = json.load(file)
    else:
        experiments = []

    next_id = max(
        (experiment["experiment_id"] for experiment in experiments),
        default=0
    ) + 1


    for index in range(next_id, next_id + count):
        hardware = DEFAULT_HARDWARE.copy()
        training_parameters = get_default_training_parameters()

        if RESEARCH_QUESTION_FOCUS in ("RQ1", "RQ3"):
            hardware.update(sample_parameters(
                rng,
                HARDWARE_ALL_RANGES,
                HARDWARE_CURRENT_RANGES
            ))

        if RESEARCH_QUESTION_FOCUS in ("RQ2", "RQ3"):
            training_parameters.update(sample_parameters(
                rng,
                TRAINING_PARAMETERS_ALL_RANGES,
                TRAINING_PARAMETERS_CURRENT_RANGES
            ))

        # TODO: INSERT VALIDATION FOR HYPERPARAMETERS HERE -> Marcel

        experiments.append({
            "experiment_id": index,
            "status": "Awaiting execution",
            "research_focus": RESEARCH_QUESTION_FOCUS,
            "adaptability_mode": ADAPTABILITY_MODE,
            "meaningful_mode": MEANINGFUL_MODE,
            "seed": seed,
            "hardware": hardware,
            "training_parameters": training_parameters
        })


    with open(PLAN_PATH, "w", encoding="utf-8") as file:
        json.dump(experiments, file, indent=4)

    return PLAN_PATH

def get_default_training_parameters():
    with open(BASE_CONFIG_TEMPLATE, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    behavior = config["behaviors"]["3DBallHard"]
    extrinsic = behavior["reward_signals"]["extrinsic"]

    defaults = {
        **behavior["hyperparameters"],
        **behavior["network_settings"],
        "gamma": extrinsic["gamma"],
        "reward_strength": extrinsic["strength"],
        "time_horizon": behavior["time_horizon"],
        "trainer_type": behavior["trainer_type"],
        "max_steps": behavior["max_steps"],
        "keep_checkpoints": behavior["keep_checkpoints"],
        "summary_freq": behavior["summary_freq"]
    }

    return defaults

def get_next_experiment():

    with open(PLAN_PATH, "r", encoding="utf-8") as file:
        experiments = json.load(file)

        pending_experiments = [
            experiment
            for experiment in experiments
            if experiment["status"] == "Awaiting execution"
        ]

        if not pending_experiments:
            print(f"No more experiments left to perform from latest plan")
            return None

        return min(pending_experiments, key=lambda experiment: experiment["experiment_id"])

def update_experiment_status(experiment_id, new_status):

    with open(PLAN_PATH, "r", encoding="utf-8") as file:
        experiments = json.load(file)

    valid_statuses = {"Awaiting execution", "Executing", "Done", "Failure"}

    if new_status not in valid_statuses:
        raise ValueError(f"Status {new_status} is invalid")

    for experiment in experiments:
        if experiment["experiment_id"] == experiment_id:
            experiment["status"] = new_status
            break
    else: 
        raise ValueError(f"Experiment {experiment_id} could not be found")

    with open(PLAN_PATH, "w", encoding="utf-8") as file:
        json.dump(experiments, file, indent=4)


if __name__ == "__main__":

    experiment = get_next_experiment()

    if experiment is None:
        print("No experiments awaiting execution.")
    else:
        config_path = generate_config(**experiment["training_parameters"])

        print(f"Selected experiment: {experiment['experiment_id']}")
        print(f"Generated configuration: {config_path}")