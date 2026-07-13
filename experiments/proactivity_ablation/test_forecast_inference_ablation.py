#!/usr/bin/env python3
"""Tests for counterfactual forecast transformations."""

from __future__ import annotations

import unittest

from prorl.core.step import Step
from prorl.core.step_data import StepData, StepDataEntry

from experiments.proactivity_ablation.forecast_perturbation import perturb_forecast


def load(step_index: int, values: list[float]) -> StepData:
    step = Step(hour=step_index, total_steps=step_index * 3600)
    data = StepData(generator_model="test-oracle")
    for index, value in enumerate(values):
        data.add_entry(StepDataEntry("res_1", value, f"bs_{index}", step))
    return data


class ForecastInferenceAblationTest(unittest.TestCase):

    def setUp(self):
        self.loads = [load(index + 1, [24, 24, 24, 24]) for index in range(5)]
        self.loads.append(load(6, [16, 64, 24, 24]))

    def values(self, loads):
        return [item.get_resource_values("res_1", as_array=True) for item in loads]

    def test_oracle_is_cloned_without_value_changes(self):
        result = perturb_forecast(self.loads, "oracle")
        self.assertEqual(self.values(result), self.values(self.loads))
        self.assertIsNot(result[0], self.loads[0])

    def test_masked_uses_in_distribution_calm_baseline(self):
        result = perturb_forecast(self.loads, "masked")
        self.assertEqual(self.values(result), [[24.0] * 4] * 6)

    def test_node_permutation_swaps_within_pairs(self):
        result = perturb_forecast(self.loads, "node-permuted")
        self.assertEqual(self.values([result[-1]])[0], [64.0, 16.0, 24.0, 24.0])

    def test_time_reversal_preserves_values_and_reverses_horizon(self):
        result = perturb_forecast(self.loads, "time-reversed")
        self.assertEqual(self.values(result)[0], [16, 64, 24, 24])
        self.assertEqual(self.values(result)[-1], [24, 24, 24, 24])

    def test_transformations_do_not_mutate_oracle(self):
        before = self.values(self.loads)
        for mode in ("masked", "node-permuted", "time-reversed"):
            perturb_forecast(self.loads, mode)
        self.assertEqual(self.values(self.loads), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
