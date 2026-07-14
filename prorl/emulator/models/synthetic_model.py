import copy
from dataclasses import dataclass
from logging import Logger
from typing import List, Optional, Dict, Union, Tuple, Any

import numpy as np
from numpy.random import RandomState

from prorl.common.data_structure import RunMode
from prorl.core.step import Step
from prorl.core.step_data import StepDataEntry, StepData
from prorl.core.timestep import TimeStep
from prorl.emulator import EmulatorConfig
from prorl.emulator.config import CoupleConfig
from prorl.emulator.models.abstract import AbstractModel


@dataclass
class CoupleStatus:
    current_load: float
    config: CoupleConfig
    bs_1_name: str
    bs_1_index: int
    bs_2_name: str
    bs_2_index: int
    load_multiplier: float = 1.0
    is_stressing: bool = False
    stressing_steps: int = 0
    swapped: bool = False
    last_step_updated: Step = None

    def set_current_load(self, step: Step, step_size: int):
        diff = step.total_steps - self.last_step_updated.total_steps
        if diff != step_size:
            # we are in a step in the middle, we do not change anything
            return
        else:
            # we can change the load, so update last_step_update
            self.last_step_updated = step
        is_stress_step = False
        if self.is_stressing and self.stressing_steps <= self.config.keep_stress:
            is_stress_step = True
        elif isinstance(self.config.stress_every, int):
            stress_frequency = self.config.stress_every * step_size
            if step.total_steps > 0 and step.total_steps % stress_frequency == 0:
                is_stress_step = True
        else:
            dict_step = step.to_dict()
            match = True
            for unit_name, unit_value in self.config.stress_every.items():
                if isinstance(unit_value, list):
                    if dict_step[unit_name] not in unit_value:
                        match = False
                else:
                    if dict_step[unit_name] != unit_value:
                        match = False
            if match:
                is_stress_step = True

        if is_stress_step:
            self.current_load = self.config.stress_load
            self.stressing_steps += 1
            self.is_stressing = True
            if self.stressing_steps == 1:
                self.swapped = not self.swapped
        else:
            self.current_load = self.config.calm_load
            self.stressing_steps = 0
            self.is_stressing = False

    def get_high_node_demand_profile(self, absolute_demand: float) -> float:
        demand = absolute_demand * self.current_load
        return demand * self.load_multiplier

    def get_low_node_demand_profile(self, absolute_demand: float) -> float:
        low_node_load = 1 - self.current_load
        if not self.is_stressing and self.config.calm_load_equal:
            low_node_load = self.current_load
        demand = absolute_demand * low_node_load
        return demand * self.load_multiplier

    def is_swapped(self) -> bool:
        if self.config.swap_stress and self.swapped and self.is_stressing:
            return True
        else:
            return False

    def get_bs_1_data_entry(
            self,
            demand_profile: float,
            resource: str,
            step: Step,
            random_state: RandomState
    ) -> StepDataEntry:
        return StepDataEntry(
            resource=resource,
            value=max(0, random_state.normal(demand_profile, self.config.std)),
            base_station=self.bs_1_name,
            step=step,
        )

    def get_bs_2_data_entry(
            self,
            demand_profile: float,
            resource: str,
            step: Step,
            random_state: RandomState
    ) -> StepDataEntry:
        return StepDataEntry(
            resource=resource,
            value=max(0, random_state.normal(demand_profile, self.config.std)),
            base_station=self.bs_2_name,
            step=step,
        )


class SyntheticModel(AbstractModel):

    def __init__(
            self,
            base_station_names: List[str],
            resource_names: List[str],
            em_config: Optional[EmulatorConfig] = None,
            random_state: Optional[RandomState] = None,
            log: Optional[Logger] = None,
            disk_data_folder: Optional[str] = None,
            disable_log: bool = False,
            run_mode: RunMode = RunMode.Train,
            random_seed: int = 42,
            use_pool_node: bool = False,
            **kwargs
    ):
        super(SyntheticModel, self).__init__(
            base_station_names=base_station_names,
            resource_names=resource_names,
            model_name='SyntheticModel',
            emulator_configuration=em_config,
            random_state=random_state,
            log=log,
            disk_data_folder=disk_data_folder,
            disable_log=disable_log,
            run_mode=run_mode,
            random_seed=random_seed,
            use_pool_node=use_pool_node,
            **kwargs
        )
        model_config = self.emulator_config.model.synthetic_model
        self.episode_length: int = model_config.episode_length
        self.change_distribution_frequency: int = model_config.change_distribution_frequency
        self.base_couples_config: List[CoupleConfig] = copy.deepcopy(model_config.couples_config)
        self.couples_config: List[CoupleConfig] = copy.deepcopy(self.base_couples_config)
        self.schedule_randomization: Dict[str, Any] = copy.deepcopy(model_config.schedule_randomization)
        self.schedule_scenarios: List[Dict[str, Any]] = self.schedule_randomization.get('scenarios', [])
        self.randomize_schedule: bool = bool(self.schedule_randomization.get('enabled', False))
        self.active_schedule_scenario_id: Optional[str] = None
        self.schedule_scenario_history: List[str] = []
        self._validate_schedule_randomization()
        # every steps the means are multiplied by increase_multiplier
        self.distribution_multipliers: List[float] = model_config.distribution_multipliers
        self.model_step_size: int = model_config.model_step_size
        self.demand_absolute_value: float = model_config.demand_absolute_value

        self.couples_status: List[CoupleStatus] = []

        self.distribution_index = 0
        self.overall_steps = 0

        self._init_bs_status()

    def _validate_schedule_randomization(self):
        if not self.randomize_schedule:
            return
        if not self.schedule_scenarios:
            raise ValueError('schedule_randomization requires at least one scenario when enabled')
        expected_couples = len(self.base_couples_config)
        allowed = {
            'calm_load', 'calm_load_equal', 'stress_load', 'stress_every',
            'keep_stress', 'swap_stress', 'std',
        }
        scenario_ids = set()
        for index, scenario in enumerate(self.schedule_scenarios):
            scenario_id = str(scenario.get('id', f'scenario-{index}'))
            if scenario_id in scenario_ids:
                raise ValueError(f'duplicate randomized schedule scenario id: {scenario_id}')
            scenario_ids.add(scenario_id)
            couples = scenario.get('couples')
            if not isinstance(couples, list) or len(couples) != expected_couples:
                raise ValueError(
                    f'randomized schedule scenario {scenario_id} must define {expected_couples} couples')
            for override in couples:
                if not isinstance(override, dict):
                    raise ValueError(
                        f'randomized schedule scenario {scenario_id} couple overrides must be dictionaries')
                unknown = set(override) - allowed
                if unknown:
                    raise ValueError(
                        f'randomized schedule scenario {scenario_id} has unknown fields: {sorted(unknown)}')
                schedule = override.get('stress_every')
                if not isinstance(schedule, dict) or 'week_day' not in schedule or 'hour' not in schedule:
                    raise ValueError(
                        f'randomized schedule scenario {scenario_id} requires week_day and hour')
                days = schedule['week_day'] if isinstance(schedule['week_day'], list) \
                    else [schedule['week_day']]
                hours = schedule['hour'] if isinstance(schedule['hour'], list) else [schedule['hour']]
                if not days or any(not isinstance(day, int) or day < 0 or day > 6 for day in days):
                    raise ValueError(f'randomized schedule scenario {scenario_id} has invalid week_day')
                if not hours or any(not isinstance(hour, int) or hour < 0 or hour > 23 for hour in hours):
                    raise ValueError(f'randomized schedule scenario {scenario_id} has invalid hour')
                if 'keep_stress' in override and (
                        not isinstance(override['keep_stress'], int) or override['keep_stress'] < 0):
                    raise ValueError(f'randomized schedule scenario {scenario_id} has invalid keep_stress')
                if 'stress_load' in override and not 0 <= override['stress_load'] <= 1:
                    raise ValueError(f'randomized schedule scenario {scenario_id} has invalid stress_load')

    def _sample_schedule_scenario(self):
        self.couples_config = copy.deepcopy(self.base_couples_config)
        self.active_schedule_scenario_id = None
        if not self.randomize_schedule:
            return
        scenario_index = int(self.random.randint(0, len(self.schedule_scenarios)))
        scenario = self.schedule_scenarios[scenario_index]
        self.active_schedule_scenario_id = str(scenario.get('id', f'scenario-{scenario_index}'))
        self.schedule_scenario_history.append(self.active_schedule_scenario_id)
        for couple, overrides in zip(self.couples_config, scenario['couples']):
            for key, value in overrides.items():
                setattr(couple, key, copy.deepcopy(value))

    def _init_bs_status(self):
        self.couples_status: List[CoupleStatus] = []
        counter = 0
        if self.use_pool_node:
            base_station_names = [bs_name for bs_name in self.base_station_names if bs_name != self.pool_bs_name]
        else:
            base_station_names = self.base_station_names
        for i in range(0, len(base_station_names), 2):
            config = self.couples_config[counter % len(self.couples_config)]
            status = CoupleStatus(
                current_load=config.calm_load,
                config=config,
                bs_1_name=base_station_names[i],
                bs_1_index=i,
                bs_2_name=base_station_names[i+1],
                bs_2_index=i+1,
                load_multiplier=self.distribution_multipliers[self.distribution_index],
                last_step_updated=Step.from_str('')
            )
            self.couples_status.append(status)
            counter += 1

    def generate_couple_data(self, step: Step, couple_index: int, resource: str) -> Tuple[StepDataEntry, StepDataEntry]:
        couple_status: CoupleStatus = self.couples_status[couple_index]
        couple_status.set_current_load(step, step_size=self.model_step_size)
        if couple_status.is_swapped():
            bs_1_demand = couple_status.get_low_node_demand_profile(self.demand_absolute_value)
            bs_2_demand = couple_status.get_high_node_demand_profile(self.demand_absolute_value)
        else:
            bs_1_demand = couple_status.get_high_node_demand_profile(self.demand_absolute_value)
            bs_2_demand = couple_status.get_low_node_demand_profile(self.demand_absolute_value)
        bs_1_data = couple_status.get_bs_1_data_entry(bs_1_demand, resource, step, self.random)
        bs_2_data = couple_status.get_bs_2_data_entry(bs_2_demand, resource, step, self.random)
        return bs_1_data, bs_2_data

    def _counter_increase(self, step: Step) -> int:
        increase = 0
        if step.total_steps % self.model_step_size == 0:
            increase = 1
        return increase

    def _generate_single_base_station(self, step: Step, resource_name: str, base_station_name: str) -> StepDataEntry:
        pass

    def _generate_step_data(self, step: Step, list_resources: List[str], list_base_stations: List[str]) -> StepData:
        data = StepData(generator_model=self.model_name())
        for i, resource in enumerate(list_resources):
            if self.use_pool_node:
                data.add_entry(self.pool_step_data_entry(resource, step))
            for j, couple_state in enumerate(self.couples_status):
                bs_1_data, bs_2_data = self.generate_couple_data(step, j, resource)
                data.add_entry(bs_1_data)
                data.add_entry(bs_2_data)

        if self.overall_steps > 0 and self.overall_steps % self.change_distribution_frequency == 0:
            self.distribution_index += 1
            current_multiplier = self.distribution_multipliers[
                self.distribution_index % len(self.distribution_multipliers)]
            for couple in self.couples_status:
                couple.load_multiplier = current_multiplier

        self.overall_steps += self._counter_increase(step)
        return data

    def generate(
            self,
            step: Step,
            resources: List[str] = None,
            base_stations: List[str] = None,
            is_last_step: bool = False,
    ) -> StepData:
        list_base_stations = self.base_station_names
        list_resources = self.resource_names
        if base_stations is not None:
            list_base_stations = base_stations
        if resources is not None:
            list_resources = resources

        return self._generate_step_data(step, list_resources, list_base_stations)

    def forecast(self, current_step: Step, horizon: int) -> List[StepData]:
        """Schedule-oracle forecast of expected demand, excluding random observation noise.

        The copied couple statuses reproduce future stress/swap transitions while keeping
        the live generator state and its random stream untouched.
        """
        if horizon <= 0:
            raise ValueError('forecast horizon must be greater than zero')

        statuses = copy.deepcopy(self.couples_status)
        distribution_index = self.distribution_index
        overall_steps = self.overall_steps
        clock = TimeStep(
            step_per_second=1,
            step_size=self.model_step_size,
            stop_step=current_step.total_steps + horizon * self.model_step_size,
            initial_date=current_step,
            show_log=False,
        )
        forecasts: List[StepData] = []
        for _ in range(horizon):
            future_step = clock.next()
            data = StepData(generator_model=f'{self.model_name()}-ScheduleOracle')
            for resource in self.resource_names:
                if self.use_pool_node:
                    data.add_entry(self.pool_step_data_entry(resource, future_step))
                for status in statuses:
                    status.set_current_load(future_step, step_size=self.model_step_size)
                    if status.is_swapped():
                        bs_1_demand = status.get_low_node_demand_profile(self.demand_absolute_value)
                        bs_2_demand = status.get_high_node_demand_profile(self.demand_absolute_value)
                    else:
                        bs_1_demand = status.get_high_node_demand_profile(self.demand_absolute_value)
                        bs_2_demand = status.get_low_node_demand_profile(self.demand_absolute_value)
                    data.add_entry(StepDataEntry(
                        resource=resource, value=max(0, bs_1_demand),
                        base_station=status.bs_1_name, step=future_step))
                    data.add_entry(StepDataEntry(
                        resource=resource, value=max(0, bs_2_demand),
                        base_station=status.bs_2_name, step=future_step))
            forecasts.append(data)

            if overall_steps > 0 and overall_steps % self.change_distribution_frequency == 0:
                distribution_index += 1
                multiplier = self.distribution_multipliers[
                    distribution_index % len(self.distribution_multipliers)]
                for status in statuses:
                    status.load_multiplier = multiplier
            if future_step.total_steps % self.model_step_size == 0:
                overall_steps += 1
        return forecasts

    def reset(self, show_log=True, full_reset=True):
        super(SyntheticModel, self).reset(show_log)
        self.distribution_index = 0
        self.overall_steps = 0
        self._sample_schedule_scenario()
        self._init_bs_status()

    def get_stop_step(self, step_size) -> Union[None, int, Step]:
        return self.episode_length * self.model_step_size * len(self.distribution_multipliers)
