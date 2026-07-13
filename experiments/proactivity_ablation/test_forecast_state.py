#!/usr/bin/env python3
"""Compatibility and integration tests for Forecast-No-Time PRORL."""

from __future__ import annotations

import copy
import os
import unittest
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("ENV", "test")

from prorl import ROOT_DIR  # noqa: E402
from prorl.core.timestep import TimeStep  # noqa: E402
from prorl.environment.state_builder import StateFeatureName, StateType  # noqa: E402
from prorl.run.config import SingleRunConfig  # noqa: E402
from prorl.run.runner import TestRunner  # noqa: E402


HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated"


def load_config(name: str) -> SingleRunConfig:
    with (GENERATED / f"{name}.yaml").open(encoding="utf-8") as stream:
        raw = copy.deepcopy(yaml.safe_load(stream)["base_run_config"])
    raw["run"]["validation_run"]["enabled"] = False
    raw["redis"]["enabled"] = False
    raw["saver"]["enabled"] = False
    raw["logger"]["level"] = 40
    return SingleRunConfig(root_dir=ROOT_DIR, **raw)


def build_env(name: str):
    runner = TestRunner(run_code=f"forecast-test-{name}", config=load_config(name))
    runner._init()
    state = runner.env.reset()
    return runner.env, state


def status_snapshot(model):
    return [
        (status.current_load, status.load_multiplier, status.is_stressing,
         status.stressing_steps, status.swapped, status.last_step_updated)
        for status in model.couples_status
    ]


class ForecastStateTest(unittest.TestCase):

    def test_legacy_config_remains_compatible(self):
        env, states = build_env("no_time")
        self.assertNotIn(StateFeatureName.NodeDemandForecast, env._state_features)
        self.assertIsNone(env.current_demand_forecast)
        for state_type, state in states.items():
            self.assertEqual(state.size, env.state_spaces[state_type])

    def test_forecast_state_shape_layout_and_no_time(self):
        legacy_env, legacy_states = build_env("no_time")
        env, states = build_env("forecast_no_time")
        self.assertNotIn(StateFeatureName.TimeEncoded, env._state_features)
        self.assertEqual(len(env.current_demand_forecast), 6)

        expected_extra = env.n_nodes * 6 * len(env.resources_info)
        for state_type in states:
            self.assertEqual(states[state_type].size,
                             legacy_states[state_type].size + expected_extra)

        resource = env.resources_info[0]
        step_major = np.array([
            load.get_resource_values(resource.name, as_array=True)
            for load in env.current_demand_forecast
        ])
        divider = resource.allocated / resource.units_allocated
        expected = np.ceil(step_major.T.reshape(-1) / divider) / resource.total_units
        actual = states[StateType.Add].get_feature_value(
            f"{StateFeatureName.NodeDemandForecast.value}_{resource.name}")
        np.testing.assert_allclose(np.asarray(actual), expected)

    def test_oracle_sees_upcoming_peak_without_mutating_generator(self):
        env, _ = build_env("forecast_no_time")
        model = env.load_generator
        clock = TimeStep(step_size=model.model_step_size, initial_date=env.current_demand.step)
        # Thursday 08:00 (week_day=3) is one hour before couple 0's configured peak.
        for _ in range(80):
            step = clock.next()
            model.generate(step)
        self.assertEqual((step.week_day, step.hour), (3, 8))

        random_before = model.random.get_state()
        statuses_before = status_snapshot(model)
        forecasts = model.forecast(step, 6)
        random_after = model.random.get_state()
        self.assertEqual(statuses_before, status_snapshot(model))
        self.assertEqual(random_before[0], random_after[0])
        np.testing.assert_array_equal(random_before[1], random_after[1])
        self.assertEqual(random_before[2:], random_after[2:])

        first = forecasts[0].get_resource_values(env.resources_info[0].name, as_array=True)
        self.assertEqual((forecasts[0].step.week_day, forecasts[0].step.hour), (3, 9))
        # swap_stress toggles the high node on the first stress step.
        np.testing.assert_allclose(first[:2], [16.0, 64.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
