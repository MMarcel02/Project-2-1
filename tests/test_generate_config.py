
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from configuration import generate_config as generator


class TestGenerateConfig(unittest.TestCase):

    def test_default_training_parameters(self):
        defaults = generator.get_default_training_parameters()

        self.assertIn("learning_rate", defaults)
        self.assertIn("batch_size", defaults)
        self.assertIn("hidden_units", defaults)
        self.assertIn("gamma", defaults)
        self.assertIn("time_horizon", defaults)
        self.assertIn("reward_strength", defaults)

        self.assertEqual(defaults["trainer_type"], "ppo")

    def test_adaptive_mode_skips_inactive_parameters(self):
        with (
            patch.object(generator, "ADAPTABILITY_MODE", True),
            patch.object(generator, "MEANINGFUL_MODE", True)
        ):
            selected = generator.sample_parameters(
                random.Random(42),
                generator.TRAINING_PARAMETERS_ALL_RANGES,
                generator.TRAINING_PARAMETERS_CURRENT_RANGES
            )

        self.assertIn("learning_rate", selected)
        self.assertIn("batch_size", selected)

        self.assertNotIn("epsilon", selected)
        self.assertNotIn("num_layers", selected)
        self.assertNotIn("normalize", selected)

    def test_nonadaptive_mode_ignores_locks(self):
        with (
            patch.object(generator, "ADAPTABILITY_MODE", False),
            patch.object(generator, "MEANINGFUL_MODE", True)
        ):
            selected = generator.sample_parameters(
                random.Random(42),
                generator.TRAINING_PARAMETERS_ALL_RANGES,
                generator.TRAINING_PARAMETERS_CURRENT_RANGES
            )

        self.assertIn("epsilon", selected)
        self.assertIn("num_layers", selected)
        self.assertIn("normalize", selected)

        self.assertNotIn("vis_encode_type", selected)
        self.assertNotIn("reward_strength", selected)

    def test_meaningful_mode_disabled(self):
        with (
            patch.object(generator, "ADAPTABILITY_MODE", False),
            patch.object(generator, "MEANINGFUL_MODE", False)
        ):
            selected = generator.sample_parameters(
                random.Random(42),
                generator.TRAINING_PARAMETERS_ALL_RANGES,
                generator.TRAINING_PARAMETERS_CURRENT_RANGES
            )

        self.assertIn("vis_encode_type", selected)
        self.assertIn("reward_strength", selected)

        self.assertNotIn("trainer_type", selected)
        self.assertNotIn("max_steps", selected)

    def test_rq1_uses_fixed_training_parameters(self):
        self._test_research_focus("RQ1")

    def test_rq2_uses_fixed_hardware(self):
        self._test_research_focus("RQ2")

    def test_rq3_samples_both(self):
        self._test_research_focus("RQ3")

    def _test_research_focus(self, research_focus):
        with tempfile.TemporaryDirectory() as directory:
            temporary_plan = Path(directory) / "experiment_plan.json"

            with (
                patch.object(
                    generator, "RESEARCH_QUESTION_FOCUS", research_focus
                ),
                patch.object(generator, "ADAPTABILITY_MODE", True),
                patch.object(generator, "MEANINGFUL_MODE", True),
                patch.object(generator, "PLAN_PATH", temporary_plan)
            ):
                generator.generate_experiment_plan(count=5, seed=42)

                experiments = json.loads(
                    temporary_plan.read_text(encoding="utf-8")
                )

            self.assertEqual(len(experiments), 5)

            defaults = generator.get_default_training_parameters()

            for experiment in experiments:
                training_parameters = experiment["training_parameters"]
                hardware = experiment["hardware"]

                self.assertEqual(
                    experiment["research_focus"],
                    research_focus
                )
                self.assertEqual(
                    experiment["status"],
                    "Awaiting execution"
                )

                if research_focus == "RQ1":
                    self.assertEqual(training_parameters, defaults)

                if research_focus == "RQ2":
                    self.assertEqual(hardware, generator.DEFAULT_HARDWARE)

                if research_focus in ("RQ2", "RQ3"):
                    self.assertEqual(
                        training_parameters["epsilon"],
                        defaults["epsilon"]
                    )

                self.assertIn("cpu_cores", hardware)
                self.assertIn("memory_limit_gb", hardware)

    def test_plan_appends_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary_plan = Path(directory) / "experiment_plan.json"

            with (
                patch.object(generator, "RESEARCH_QUESTION_FOCUS", "RQ2"),
                patch.object(generator, "ADAPTABILITY_MODE", True),
                patch.object(generator, "MEANINGFUL_MODE", True),
                patch.object(generator, "PLAN_PATH", temporary_plan)
            ):
                generator.generate_experiment_plan(count=3, seed=42)

                experiments = json.loads(
                    temporary_plan.read_text(encoding="utf-8")
                )

                experiments[0]["status"] = "Done"

                temporary_plan.write_text(
                    json.dumps(experiments, indent=4),
                    encoding="utf-8"
                )

                generator.generate_experiment_plan(count=2, seed=43)

                updated = json.loads(
                    temporary_plan.read_text(encoding="utf-8")
                )

            self.assertEqual(len(updated), 5)
            self.assertEqual(updated[0]["status"], "Done")

            ids = [experiment["experiment_id"] for experiment in updated]

            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(ids, [1, 2, 3, 4, 5])

    def test_generated_values_within_ranges(self):
        with (
            patch.object(generator, "ADAPTABILITY_MODE", False),
            patch.object(generator, "MEANINGFUL_MODE", True)
        ):
            selected = generator.sample_parameters(
                random.Random(42),
                generator.TRAINING_PARAMETERS_ALL_RANGES,
                generator.TRAINING_PARAMETERS_CURRENT_RANGES
            )

        self.assertGreaterEqual(selected["learning_rate"], 0.0001)
        self.assertLessEqual(selected["learning_rate"], 0.001)

        self.assertIn(
            selected["batch_size"],
            [64, 128, 256, 512]
        )

        self.assertIn(
            selected["buffer_size"],
            [4096, 8192, 16384, 32768]
        )

        self.assertGreaterEqual(selected["epsilon"], 0.1)
        self.assertLessEqual(selected["epsilon"], 0.3)

    def test_plan_size_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary_plan = Path(directory) / "experiment_plan.json"

            with patch.object(generator, "PLAN_PATH", temporary_plan):
                with self.assertRaises(ValueError):
                    generator.generate_experiment_plan(count=26)

    def test_generate_config_writes_correct_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(generator, "OUTPUT_DIRECTORY", Path(directory)):
                output_path = generator.generate_config(
                    learning_rate=0.0005,
                    hidden_units=256,
                    gamma=0.95,
                    reward_strength=2.0,
                    time_horizon=128
                )

                with open(output_path, "r", encoding="utf-8") as file:
                    config = yaml.safe_load(file)

        ball_config = config["behaviors"]["3DBallHard"]

        self.assertEqual(
            ball_config["hyperparameters"]["learning_rate"], 0.0005
        )
        self.assertEqual(
            ball_config["network_settings"]["hidden_units"], 256
        )
        self.assertEqual(
            ball_config["reward_signals"]["extrinsic"]["gamma"], 0.95
        )
        self.assertEqual(
            ball_config["reward_signals"]["extrinsic"]["strength"], 2.0
        )
        self.assertEqual(ball_config["time_horizon"], 128)

    def test_generate_config_preserves_defaults(self):
        defaults = generator.get_default_training_parameters()

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(generator, "OUTPUT_DIRECTORY", Path(directory)):
                output_path = generator.generate_config(**defaults)

                with open(output_path, "r", encoding="utf-8") as file:
                    generated = yaml.safe_load(file)

        with open(generator.BASE_CONFIG_TEMPLATE, "r", encoding="utf-8") as file:
            original = yaml.safe_load(file)

        self.assertEqual(generated, original)

    def test_generate_config_rejects_unknown_parameter(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(generator, "OUTPUT_DIRECTORY", Path(directory)):
                with self.assertRaises(ValueError):
                    generator.generate_config(
                        completely_unknown_parameter=123
                    )

    
    def test_inactive_parameters_keep_defaults(self):
        defaults = generator.get_default_training_parameters()

        with (
            patch.object(generator, "ADAPTABILITY_MODE", True),
            patch.object(generator, "MEANINGFUL_MODE", True)
        ):
            sampled = generator.sample_parameters(
                random.Random(42),
                generator.TRAINING_PARAMETERS_ALL_RANGES,
                generator.TRAINING_PARAMETERS_CURRENT_RANGES
            )

        inactive_parameters = [
            "epsilon",
            "lambd",
            "num_layers",
            "time_horizon",
            "normalize",
            "learning_rate_schedule",
            "vis_encode_type",
            "reward_strength",
            "trainer_type",
            "max_steps",
            "keep_checkpoints",
            "summary_freq"
        ]

        for parameter in inactive_parameters:
            self.assertNotIn(parameter, sampled)

        training_parameters = defaults.copy()
        training_parameters.update(sampled)

        for parameter in inactive_parameters:
            self.assertEqual(
                training_parameters[parameter],
                defaults[parameter]
            )



if __name__ == "__main__":
    unittest.main()
