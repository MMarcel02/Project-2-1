from pathlib import Path
import yaml
import random
import json

ROOT = Path(__file__).resolve().parent.parent # Parent of this parent folder is root

BASE_CONFIG_TEMPLATE = ROOT / "ml-agents" / "config" / "ppo" / "3DBallHard.yaml"
OUTPUT_DIRECTORY = ROOT / "configuration" / "generated_configs"


def generate_config(**hyperparameters):
    with open(BASE_CONFIG_TEMPLATE, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    parameter_list = config["behaviors"]["3DBallHard"]["hyperparameters"]

    for parameter, value in hyperparameters.items():
        if parameter not in parameter_list:
            raise ValueError("Hyperparameter {parameter} does not exist or can not be identified")

        parameter_list[parameter] = value

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIRECTORY / f"3DBallHard_current.yaml"

    with open(output_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    return output_path

def generate_experiment_plan(count=10, seed=42):
    rng = random.Random(seed)

    possible_combinations = [
        (learning_rate, batch_size, beta)
        for learning_rate in [0.0001, 0.0003, 0.001]
        for batch_size in [64, 120, 256]
        for beta in [0.001, 0.005, 0.01]
    ]

    if count > len(possible_combinations):
        raise ValueError("There are less unique combinations than {count}")

    selected = rng.sample(possible_combinations, count)

    experiments = [
        {
            "experiment_id": index,
            "status": "Awaiting execution",
            "hyperparameters": {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "beta": beta
            }
        }
        for index, (learning_rate, batch_size, beta) in enumerate(selected, start=1)
    ]

    plan_path = OUTPUT_DIRECTORY.parent / "experiment_plan.json"

    with open(plan_path, "w", encoding="utf-8") as file:
        json.dump(experiments, file, indent=4)

    return plan_path

def get_next_experiment():
    plan_path = OUTPUT_DIRECTORY.parent / "experiment_plan.json"

    with open(plan_path, "r", encoding="utf-8") as file:
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
    plan_path = OUTPUT_DIRECTORY.parent / "experiment_plan.json"

    with open(plan_path, "r", encoding="utf-8") as file:
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

    with open(plan_path, "w", encoding="utf-8") as file:
        json.dump(experiments, file, indent=4)


if __name__ == "__main__":

    experiment = get_next_experiment()

    if experiment is None:
        print("No experiments awaiting execution.")
    else:
        config_path = generate_config(**experiment["hyperparameters"])

        print(f"Selected experiment: {experiment['experiment_id']}")
        print(f"Generated configuration: {config_path}")