#!/usr/bin/env python3
"""Tests for reproducible episode-level randomized synthetic schedules."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import numpy as np

from prorl import ROOT_DIR
from prorl.common.config import ExportMode
from prorl.emulator.models.synthetic_model import SyntheticModel
from prorl.run.multi_run_config import get_multi_run_config

from experiments.proactivity_ablation.analyze_randomized_heldout import event_specs
from experiments.proactivity_ablation.run_randomized_heldout_evaluations import apply_fixed_scenario


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "randomized_schedule_scenarios.json"
RANDOMIZED = HERE / "generated_randomized"
LEGACY = HERE / "generated" / "forecast_no_time.yaml"


def load_run(filename: Path):
    config = get_multi_run_config(ROOT_DIR, str(filename.relative_to(ROOT_DIR)))
    return config.generate_runs_config()[0]


def model(run_config, seed: int = 123) -> SyntheticModel:
    return SyntheticModel(
        base_station_names=["bs_1", "bs_2", "bs_3", "bs_4"],
        resource_names=["res_1"],
        em_config=run_config.emulator,
        random_state=np.random.RandomState(seed),
        random_seed=seed,
    )


class RandomizedScheduleTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_manifest_has_disjoint_fixed_banks(self):
        training = self.manifest["training"]
        held_out = self.manifest["held_out"]
        train_ids = {scenario["id"] for scenario in training}
        heldout_ids = {scenario["id"] for scenario in held_out}
        self.assertEqual(len(training), 12)
        self.assertEqual(len(held_out), 4)
        self.assertTrue(train_ids.isdisjoint(heldout_ids))
        train_payloads = {json.dumps(item["couples"], sort_keys=True) for item in training}
        heldout_payloads = {json.dumps(item["couples"], sort_keys=True) for item in held_out}
        self.assertTrue(train_payloads.isdisjoint(heldout_payloads))
        overrides = [couple for scenario in training for couple in scenario["couples"]]
        self.assertEqual({item["swap_stress"] for item in overrides}, {False, True})
        self.assertEqual({item["stress_load"] for item in overrides}, {0.70, 0.75, 0.80})
        self.assertEqual({item["keep_stress"] + 1 for item in overrides}, {1, 2, 3})

    def test_paired_conditions_only_differ_by_forecast_state(self):
        no_time = load_run(RANDOMIZED / "randomized_no_time.yaml")
        forecast = load_run(RANDOMIZED / "randomized_forecast_no_time.yaml")
        no_time_raw = no_time.export(mode=ExportMode.DICT)
        forecast_raw = forecast.export(mode=ExportMode.DICT)
        no_time_state = no_time_raw["environment"]["state"]
        forecast_state = forecast_raw["environment"]["state"]
        for key in ("base_features", "features"):
            forecast_state[key].remove("node-demand-forecast")
        forecast_state["additional_properties"]["forecast_horizon"] = 0
        forecast_state["additional_properties"]["forecast_type"] = "oracle"
        no_time_state["additional_properties"]["forecast_horizon"] = 0
        no_time_state["additional_properties"]["forecast_type"] = "oracle"
        no_time_raw["multi_run"]["multi_run_code"] = forecast_raw["multi_run"]["multi_run_code"]
        self.assertEqual(no_time_raw, forecast_raw)

    def test_same_seed_reproduces_schedule_sequence_across_conditions(self):
        no_time = model(load_run(RANDOMIZED / "randomized_no_time.yaml"), seed=777)
        forecast = model(load_run(RANDOMIZED / "randomized_forecast_no_time.yaml"), seed=777)
        for _ in range(20):
            no_time.reset(show_log=False)
            forecast.reset(show_log=False)
            # Forecast generation must not consume the live random stream.
            forecast.forecast(forecast.couples_status[0].last_step_updated, 6)
        self.assertEqual(no_time.schedule_scenario_history, forecast.schedule_scenario_history)
        self.assertGreater(len(set(no_time.schedule_scenario_history)), 1)

    def test_selected_scenario_overrides_all_couples(self):
        run = load_run(RANDOMIZED / "randomized_forecast_no_time.yaml")
        selected = copy.deepcopy(self.manifest["held_out"][0])
        run.emulator.model.synthetic_model.schedule_randomization = {
            "enabled": True,
            "scenarios": [selected],
        }
        generator = model(run)
        generator.reset(show_log=False)
        self.assertEqual(generator.active_schedule_scenario_id, selected["id"])
        for couple, expected in zip(generator.couples_config, selected["couples"]):
            self.assertEqual(couple.stress_every, expected["stress_every"])
            self.assertEqual(couple.stress_load, expected["stress_load"])
            self.assertEqual(couple.keep_stress, expected["keep_stress"])
            self.assertEqual(couple.swap_stress, expected["swap_stress"])

    def test_invalid_randomized_scenario_is_rejected(self):
        run = load_run(RANDOMIZED / "randomized_no_time.yaml")
        invalid = copy.deepcopy(self.manifest["training"][0])
        invalid["couples"][0]["stress_every"]["hour"] = [24]
        run.emulator.model.synthetic_model.schedule_randomization = {
            "enabled": True,
            "scenarios": [invalid],
        }
        with self.assertRaisesRegex(ValueError, "invalid hour"):
            model(run)

    def test_heldout_evaluation_disables_randomization(self):
        run = load_run(RANDOMIZED / "randomized_forecast_no_time.yaml")
        raw = run.export(mode=ExportMode.DICT)
        selected = self.manifest["held_out"][1]
        apply_fixed_scenario(raw, selected)
        synthetic = raw["emulator"]["model"]["synthetic_model"]
        self.assertFalse(synthetic["schedule_randomization"]["enabled"])
        self.assertEqual(synthetic["schedule_randomization"]["scenarios"], [])
        for couple, expected in zip(synthetic["couples_config"], selected["couples"]):
            for key, value in expected.items():
                self.assertEqual(couple[key], value)

    def test_heldout_event_targets_and_durations_follow_manifest(self):
        selected = self.manifest["held_out"][0]
        events = event_specs(selected)
        self.assertEqual(events[0], {"start": 37, "duration": 3, "target": 1})
        self.assertEqual(events[1], {"start": 127, "duration": 2, "target": 2})

    def test_legacy_config_keeps_randomization_disabled(self):
        generator = model(load_run(LEGACY))
        before = [copy.deepcopy(couple.stress_every) for couple in generator.couples_config]
        generator.reset(show_log=False)
        after = [couple.stress_every for couple in generator.couples_config]
        self.assertFalse(generator.randomize_schedule)
        self.assertIsNone(generator.active_schedule_scenario_id)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main(verbosity=2)
