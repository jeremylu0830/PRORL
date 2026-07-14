"""Static baseline that never moves resources."""

from logging import Logger
from typing import Dict, List, Optional, Tuple

import numpy as np

from prorl import SingleRunConfig
from prorl.core.state import State
from prorl.core.step_data import StepData
from prorl.environment.action_space import ActionSpaceWrapper, ActionType
from prorl.environment.agent import AgentType
from prorl.environment.agent.baseline.random import BaselineAgent
from prorl.environment.node import Node
from prorl.environment.node_groups import NodeGroups
from prorl.environment.state_builder import StateType


class WaitAgent(BaselineAgent):
    """Select the wait action in every supported action-space layout."""

    def __init__(
            self,
            action_space_wrapper: ActionSpaceWrapper,
            random_state: np.random.RandomState,
            action_spaces: Dict[ActionType, int],
            state_spaces: Dict[StateType, int],
            log: Logger,
            config: Optional[SingleRunConfig] = None,
            **kwargs
    ):
        super().__init__(
            action_space_wrapper,
            random_state=random_state,
            name=AgentType.Wait,
            action_spaces=action_spaces,
            state_spaces=state_spaces,
            log=log,
            config=config,
            **kwargs,
        )

    def _choose_node_remove(self, state: State, *args, **kwargs) -> int:
        return self.action_space_wrapper.remove_node_space.wait_action_index

    def _choose_node_add(self, state: State, *args, **kwargs) -> int:
        return self.action_space_wrapper.add_node_space.wait_action_index

    def _choose_combined_sub_action(
            self,
            state: State,
            add_node: int,
            nodes: List[Node],
            *args,
            **kwargs
    ) -> Tuple[int, int, int]:
        wait = self.action_space_wrapper.combined_space.wait_action_index
        return self._handle_combined_action(wait)

    def _choose_quantity(self, state: State, *args, **kwargs) -> int:
        return self.action_space_wrapper.quantity_space.wait_action_index

    def _choose_add(self, state: State, *args, **kwargs) -> int:
        return self.action_space_wrapper.add_action_space.wait_action_index

    def _choose_remove(self, state: State, *args, **kwargs) -> int:
        return self.action_space_wrapper.remove_action_space.wait_action_index
