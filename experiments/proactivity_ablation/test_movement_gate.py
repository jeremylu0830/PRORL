#!/usr/bin/env python3
"""Unit tests for frozen-policy movement gating."""

from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

from prorl.environment.action_space import Action, ActionSpaceWrapper

from experiments.proactivity_ablation.movement_gate import apply_movement_gate


class Demand:
    def __init__(self, values):
        self.values = list(values)

    def get_resource_values(self, resource, as_array=False):
        assert resource == "res_1"
        return list(self.values)


class Node:
    def __init__(self, capacity):
        self.capacity = capacity
        self.initial_resources = {
            "res_1": SimpleNamespace(classes={"resource": {"capacity": 10}}),
        }

    def get_current_allocated(self, resource):
        assert resource == "res_1"
        return self.capacity


def environment(capacities=(30, 30), current=(20, 20), forecast=((20, 20), (40, 10))):
    wrapper = ActionSpaceWrapper(
        n_nodes=2,
        n_quantities=1,
        n_resource_classes=1,
        nodes_mapping={0: 0, 1: 1},
        quantities_mapping={0: 1},
        resource_classes_mapping={0: "resource"},
    )
    return SimpleNamespace(
        nodes=[Node(value) for value in capacities],
        current_demand=Demand(current),
        current_demand_forecast=[Demand(values) for values in forecast],
        action_space_wrapper=wrapper,
    )


def split_actions(env, add=0, remove=1):
    return [
        Action(add, -1, 0, 0),
        Action(-1, remove, 0, 0),
    ]


class MovementGateTest(unittest.TestCase):

    def test_baseline_is_exact_noop(self):
        env = environment()
        actions = split_actions(env)
        before = [tuple(action) for action in actions]
        decision = apply_movement_gate(actions, env, "res_1", "baseline")
        self.assertEqual([tuple(action) for action in actions], before)
        self.assertTrue(decision.proposed_add)
        self.assertTrue(decision.proposed_remove)
        self.assertFalse(decision.canceled_add)
        self.assertFalse(decision.canceled_remove)

    def test_satisfied_gate_uses_maximum_forecast_demand(self):
        env = environment(capacities=(30, 30))
        actions = split_actions(env, add=0)
        decision = apply_movement_gate(actions, env, "res_1", "satisfied")
        self.assertFalse(decision.canceled_add)  # future node-0 demand is 40

        env.nodes[0].capacity = 40
        actions = split_actions(env, add=0)
        decision = apply_movement_gate(actions, env, "res_1", "satisfied")
        self.assertTrue(decision.canceled_add)
        self.assertTrue(env.action_space_wrapper.is_wait_action(actions[0]))
        self.assertFalse(env.action_space_wrapper.is_wait_action(actions[1]))

    def test_safe_remove_gate_accounts_for_moved_capacity(self):
        env = environment(capacities=(50, 30), current=(20, 20), forecast=((20, 20),))
        actions = split_actions(env, remove=1)
        decision = apply_movement_gate(actions, env, "res_1", "safe-remove")
        self.assertFalse(decision.canceled_remove)  # 30 - 10 >= 20

        env.nodes[1].capacity = 25
        actions = split_actions(env, remove=1)
        decision = apply_movement_gate(actions, env, "res_1", "safe-remove")
        self.assertTrue(decision.canceled_remove)  # 25 - 10 < 20
        self.assertFalse(env.action_space_wrapper.is_wait_action(actions[0]))
        self.assertTrue(env.action_space_wrapper.is_wait_action(actions[1]))

    def test_combined_gates_subactions_independently_without_mutating_demand(self):
        env = environment(capacities=(40, 25))
        current_before = copy.deepcopy(env.current_demand.values)
        forecast_before = [copy.deepcopy(item.values) for item in env.current_demand_forecast]
        actions = split_actions(env, add=0, remove=1)
        decision = apply_movement_gate(actions, env, "res_1", "combined")
        self.assertTrue(decision.canceled_add)
        self.assertTrue(decision.canceled_remove)
        self.assertEqual(env.current_demand.values, current_before)
        self.assertEqual([item.values for item in env.current_demand_forecast], forecast_before)

    def test_always_wait_cancels_both_subactions(self):
        env = environment()
        actions = split_actions(env)
        decision = apply_movement_gate(actions, env, "res_1", "always-wait")
        self.assertTrue(decision.canceled_add)
        self.assertTrue(decision.canceled_remove)
        self.assertFalse(decision.executed_add)
        self.assertFalse(decision.executed_remove)
        self.assertTrue(env.action_space_wrapper.is_wait_action(actions[0]))
        self.assertTrue(env.action_space_wrapper.is_wait_action(actions[1]))

    def test_current_only_differs_from_forecast_horizons(self):
        env = environment(capacities=(30, 30), current=(20, 20),
                          forecast=((20, 20), (40, 20), (50, 20)))

        current_actions = split_actions(env, add=0)
        current = apply_movement_gate(current_actions, env, "res_1", "combined-current")
        self.assertTrue(current.canceled_add)

        h1_actions = split_actions(env, add=0)
        h1 = apply_movement_gate(h1_actions, env, "res_1", "combined-h1")
        self.assertTrue(h1.canceled_add)

        h3_actions = split_actions(env, add=0)
        h3 = apply_movement_gate(h3_actions, env, "res_1", "combined-h3")
        self.assertFalse(h3.canceled_add)

        h6_actions = split_actions(env, add=0)
        h6 = apply_movement_gate(h6_actions, env, "res_1", "combined")
        self.assertFalse(h6.canceled_add)

    def test_horizon_gate_requires_enough_forecast_steps(self):
        env = environment(forecast=((20, 20),))
        with self.assertRaisesRegex(ValueError, "requires 3 forecast steps"):
            apply_movement_gate(split_actions(env), env, "res_1", "combined-h3")

    def test_requires_two_split_actions(self):
        env = environment()
        with self.assertRaisesRegex(ValueError, "expects two split actions"):
            apply_movement_gate(split_actions(env)[:1], env, "res_1", "combined")


if __name__ == "__main__":
    unittest.main(verbosity=2)
