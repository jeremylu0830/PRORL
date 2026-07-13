"""Pure, evaluation-only transformations for oracle demand forecasts."""

from __future__ import annotations

import copy
from enum import Enum
from statistics import median
from typing import Iterable

from prorl.core.step_data import StepData


class ForecastPerturbation(str, Enum):
    Oracle = "oracle"
    Masked = "masked"
    NodePermuted = "node-permuted"
    TimeReversed = "time-reversed"


def _clone(load: StepData) -> StepData:
    payload = copy.deepcopy(load.to_dict())
    payload["generator_model"] = load.generator_model
    return StepData.from_dict(payload)


def _resource_names(loads: list[StepData]) -> list[str]:
    return list(loads[0].to_dict()["data"])


def _set_values(load: StepData, resource: str, values: Iterable[float]) -> None:
    entries = load._data[resource]
    values = list(values)
    if len(entries) != len(values):
        raise ValueError(f"Expected {len(entries)} node values, got {len(values)}")
    for entry, value in zip(entries, values):
        base_station = next(iter(entry))
        entry[base_station] = float(value)


def perturb_forecast(loads: list[StepData], mode: ForecastPerturbation | str) -> list[StepData]:
    """Transform forecast values without modifying the oracle output.

    `masked` replaces the entire horizon with the resource-wise median. In the
    synthetic experiment this is the in-distribution calm demand (24 units).
    `node-permuted` swaps forecast values inside each configured node pair.
    `time-reversed` reverses t+1..t+h while preserving the input dimension.
    """
    try:
        selected = ForecastPerturbation(mode)
    except ValueError as error:
        raise ValueError(f"Unsupported forecast perturbation: {mode}") from error
    if not loads:
        raise ValueError("Cannot perturb an empty forecast")

    transformed = [_clone(load) for load in loads]
    resources = _resource_names(transformed)
    if selected == ForecastPerturbation.Oracle:
        return transformed
    if selected == ForecastPerturbation.TimeReversed:
        return list(reversed(transformed))

    if selected == ForecastPerturbation.Masked:
        for resource in resources:
            baseline = median(
                value for load in loads
                for value in load.get_resource_values(resource, as_array=True)
            )
            for load in transformed:
                _set_values(load, resource, [baseline] * len(load._data[resource]))
        return transformed

    for load in transformed:
        for resource in resources:
            values = load.get_resource_values(resource, as_array=True)
            if len(values) % 2 != 0:
                raise ValueError("node-permuted requires an even number of demand nodes")
            permuted = []
            for index in range(0, len(values), 2):
                permuted.extend((values[index + 1], values[index]))
            _set_values(load, resource, permuted)
    return transformed
