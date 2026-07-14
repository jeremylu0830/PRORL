"""Inference-only movement gates for split pool actions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from prorl.environment.action_space import Action


class MovementGate(str, Enum):
    BASELINE = "baseline"
    ALWAYS_WAIT = "always-wait"
    COMBINED_CURRENT = "combined-current"
    COMBINED_H1 = "combined-h1"
    COMBINED_H3 = "combined-h3"
    SATISFIED = "satisfied"
    SAFE_REMOVE = "safe-remove"
    COMBINED = "combined"


@dataclass(frozen=True)
class GateDecision:
    proposed_add: bool
    proposed_remove: bool
    canceled_add: bool
    canceled_remove: bool

    @property
    def executed_add(self) -> bool:
        return self.proposed_add and not self.canceled_add

    @property
    def executed_remove(self) -> bool:
        return self.proposed_remove and not self.canceled_remove


def wait_action(action_space) -> Action:
    """Build a wait action compatible with the environment action wrapper."""
    return Action(
        add_node=action_space.add_node_space.wait_action_index,
        remove_node=action_space.remove_node_space.wait_action_index,
        resource_class=action_space.resource_classes_space.wait_action_index,
        quantity=action_space.quantity_space.wait_action_index,
    )


def required_capacities(current_demand, forecast: Sequence, resource: str) -> list[float]:
    """Maximum per-node demand over now and the available forecast horizon."""
    demand_vectors = [current_demand.get_resource_values(resource, as_array=True)]
    demand_vectors.extend(item.get_resource_values(resource, as_array=True) for item in (forecast or []))
    n_nodes = len(demand_vectors[0])
    if any(len(values) != n_nodes for values in demand_vectors):
        raise ValueError("Current and forecast demand have inconsistent node counts")
    return [max(float(values[node]) for values in demand_vectors) for node in range(n_nodes)]


def moved_capacity(action: Action, env, resource: str) -> float:
    """Capacity represented by one split add/remove sub-action."""
    wrapper = env.action_space_wrapper
    resource_class = wrapper.resource_classes_space.actions_mapping[action.resource_class]
    quantity = float(wrapper.quantity_space.actions_mapping[action.quantity])
    capacity_per_unit = float(env.nodes[0].initial_resources[resource].classes[resource_class]["capacity"])
    return quantity * capacity_per_unit


def gate_forecast(mode: MovementGate, forecast: Sequence) -> Sequence:
    """Select the forecast prefix visible to a control gate."""
    values = forecast or []
    horizons = {
        MovementGate.COMBINED_CURRENT: 0,
        MovementGate.COMBINED_H1: 1,
        MovementGate.COMBINED_H3: 3,
    }
    if mode not in horizons:
        return values
    horizon = horizons[mode]
    if len(values) < horizon:
        raise ValueError(f"Gate {mode.value} requires {horizon} forecast steps, got {len(values)}")
    return values[:horizon]


def required_demand_description(mode: MovementGate | str) -> str:
    gate = MovementGate(mode)
    if gate is MovementGate.BASELINE:
        return "none (policy action unchanged)"
    if gate is MovementGate.ALWAYS_WAIT:
        return "none (all proposed movement forced to wait)"
    if gate is MovementGate.COMBINED_CURRENT:
        return "per-node current demand only"
    if gate is MovementGate.COMBINED_H1:
        return "per-node max(current, oracle forecast t+1)"
    if gate is MovementGate.COMBINED_H3:
        return "per-node max(current, oracle forecast t+1..t+3)"
    return "per-node max(current, oracle forecast t+1..t+6)"


def apply_movement_gate(actions: list[Action], env, resource: str,
                        mode: MovementGate | str) -> GateDecision:
    """Replace gated sub-actions with wait in place, then report the decision."""
    gate = MovementGate(mode)
    if len(actions) != 2:
        raise ValueError(f"Movement gating expects two split actions, got {len(actions)}")

    wrapper = env.action_space_wrapper
    add_action, remove_action = actions
    proposed_add = not wrapper.is_wait_action(add_action)
    proposed_remove = not wrapper.is_wait_action(remove_action)
    canceled_add = False
    canceled_remove = False

    if gate is MovementGate.ALWAYS_WAIT:
        if proposed_add:
            actions[0] = wait_action(wrapper)
            canceled_add = True
        if proposed_remove:
            actions[1] = wait_action(wrapper)
            canceled_remove = True

    if gate not in (MovementGate.BASELINE, MovementGate.ALWAYS_WAIT) and (proposed_add or proposed_remove):
        forecast = gate_forecast(gate, env.current_demand_forecast)
        required = required_capacities(env.current_demand, forecast, resource)

        if gate in (
                MovementGate.SATISFIED, MovementGate.COMBINED_CURRENT,
                MovementGate.COMBINED_H1, MovementGate.COMBINED_H3,
                MovementGate.COMBINED) and proposed_add:
            target = add_action.add_node
            if target >= len(env.nodes):
                raise ValueError(f"Invalid add target {target}")
            current = float(env.nodes[target].get_current_allocated(resource))
            if current >= required[target]:
                actions[0] = wait_action(wrapper)
                canceled_add = True

        if gate in (
                MovementGate.SAFE_REMOVE, MovementGate.COMBINED_CURRENT,
                MovementGate.COMBINED_H1, MovementGate.COMBINED_H3,
                MovementGate.COMBINED) and proposed_remove:
            source = remove_action.remove_node
            if source >= len(env.nodes):
                raise ValueError(f"Invalid remove source {source}")
            current = float(env.nodes[source].get_current_allocated(resource))
            if current - moved_capacity(remove_action, env, resource) < required[source]:
                actions[1] = wait_action(wrapper)
                canceled_remove = True

    return GateDecision(proposed_add, proposed_remove, canceled_add, canceled_remove)
