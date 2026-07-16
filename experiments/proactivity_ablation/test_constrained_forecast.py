#!/usr/bin/env python3
"""Compatibility and primal-dual tests for Constrained Forecast-PRORL."""

from __future__ import annotations

import copy
import os
import unittest

os.environ.setdefault("ENV", "test")

from prorl.common.data_structure import RunMode  # noqa: E402
from prorl.environment.reward import (  # noqa: E402
    ConstrainedGapSurplusCostReward,
    RewardFunctionType,
)
from prorl.environment.state_builder import StateFeatureName  # noqa: E402
from prorl.run.runner import TestRunner  # noqa: E402

from experiments.proactivity_ablation.test_forecast_state import load_config  # noqa: E402


def build_runner(name: str) -> TestRunner:
    config = load_config(name)
    runner = TestRunner(run_code=f"constrained-test-{name}", config=config)
    runner._init()
    return runner


class ConstrainedForecastTest(unittest.TestCase):

    def test_new_config_is_isolated_and_legacy_reward_is_unchanged(self):
        legacy = load_config("forecast_no_time")
        constrained = load_config("constrained_forecast_no_time")

        self.assertEqual(legacy.environment.reward.type, RewardFunctionType.GapSurplusCost)
        self.assertEqual(
            constrained.environment.reward.type,
            RewardFunctionType.ConstrainedGapSurplusCost,
        )
        self.assertNotIn(StateFeatureName.TimeEncoded, constrained.environment.state.features)
        self.assertIn(StateFeatureName.NodeDemandForecast, constrained.environment.state.features)
        self.assertEqual(constrained.environment.state.additional_properties["forecast_horizon"], 6)
        self.assertEqual(constrained.environment.reward.parameters["sla_target_rate"], 0.1)
        self.assertEqual(constrained.environment.reward.parameters["objective_weights"], [0.5, 0.5])
        self.assertEqual(constrained.environment.reward.parameters["training_normalization_range"], [0, 1])

    def test_dual_update_window_and_validation_freeze(self):
        runner = build_runner("constrained_forecast_no_time")
        reward = runner.env.reward_class
        self.assertIsInstance(reward, ConstrainedGapSurplusCostReward)
        reward.dual_update_interval = 2
        reward.dual_learning_rate = 0.5
        reward.sla_target_rate = 0.25
        reward.lagrangian_multiplier = 1.0

        reward._observe_constraint(1.0)
        self.assertEqual(reward.lagrangian_multiplier, 1.0)
        reward._observe_constraint(0.0)
        self.assertAlmostEqual(reward.last_observed_sla_rate, 0.5)
        self.assertAlmostEqual(reward.lagrangian_multiplier, 1.125)
        self.assertEqual(reward.dual_updates, 1)

        reward.run_mode = RunMode.Validation
        frozen = copy.deepcopy(reward.get_constraint_state())
        reward._observe_constraint(1.0)
        self.assertEqual(reward.get_constraint_state(), frozen)

    def test_step_info_and_checkpoint_round_trip(self):
        runner = build_runner("constrained_forecast_no_time")
        state = runner.env.reset()
        action, state, _ = runner.agent.choose(
            state,
            nodes=runner.env.nodes,
            resource=runner.resource_name,
            node_groups=runner.env.node_groups,
            demand=runner.env.current_demand,
            pool_node=runner.env.pool_node,
        )
        _, reward, step_info = runner.env.step(action, resource=runner.resource_name)
        self.assertIsInstance(reward, float)
        for key in (
            "sla_cost",
            "sla_target_rate",
            "base_reward",
            "constraint_penalty",
            "lagrangian_multiplier",
            "next_lagrangian_multiplier",
            "observed_sla_rate",
            "dual_updates",
        ):
            self.assertIn(key, step_info["reward_info"])

        runner._sync_constraint_state_from_env(runner.env, runner.agent)
        saved = runner.agent.get_agent_state()
        self.assertEqual(saved["constraint_state"], runner.env.reward_class.get_constraint_state())

        runner.env.reward_class.lagrangian_multiplier = 0.0
        runner.agent.load_agent_state(saved)
        runner._sync_constraint_state_to_env(runner.env, runner.agent)
        self.assertEqual(
            runner.env.reward_class.get_constraint_state(),
            saved["constraint_state"],
        )

    def test_lagrangian_reward_has_expected_sign(self):
        compute = ConstrainedGapSurplusCostReward.lagrangian_reward
        self.assertAlmostEqual(compute(0.5, 2.0, 1.0, 0.1), -1.3)
        self.assertAlmostEqual(compute(0.5, 2.0, 0.0, 0.1), 0.7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
