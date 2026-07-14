from prorl.common.enum_utils import ExtendedEnum


class AgentType(str, ExtendedEnum):
    Random = 'random'
    Wait = 'wait'
    DoubleDQN = 'prorl'
    DoubleDQNFullSpace = 'prorl-no-split'
    Heuristic = 'heuristic'
    Greedy = 'greedy'
    Oracle = 'oracle'
    # agents below are not part of this release, members kept for compatibility
    Reinforce = 'reinforce'
    TD_AC = 'td-actor-critic'
    MC_AC = 'mc-actor-critic'
    SamplingOptimal = 'sampling-optimal'

    @classmethod
    def _missing_(cls, value):
        # config files may still use the pre-release agent names
        legacy_names = {
            'double-dqn': cls.DoubleDQN,
            'double-dqn-full-space': cls.DoubleDQNFullSpace,
            'greedy-optimal': cls.Heuristic,
            'exhaustive-search': cls.Greedy,
        }
        return legacy_names.get(value)

    @staticmethod
    def is_value_based(agent_type: 'AgentType') -> bool:
        v_based_types = [AgentType.DoubleDQN, AgentType.DoubleDQNFullSpace]
        return agent_type in v_based_types

    @staticmethod
    def is_baseline(agent_type: 'AgentType') -> bool:
        baselines = [
            AgentType.Random,
            AgentType.Wait,
            AgentType.Heuristic,
            AgentType.Greedy,
            AgentType.Oracle,
            AgentType.SamplingOptimal,
        ]
        return agent_type in baselines

    @staticmethod
    def is_mc_method(agent_type: 'AgentType') -> bool:
        mc_methods = [AgentType.Reinforce, AgentType.MC_AC, AgentType.SamplingOptimal]
        return agent_type in mc_methods

    @staticmethod
    def is_policy_gradient(agent_type: 'AgentType') -> bool:
        pg_methods = [AgentType.Reinforce, AgentType.TD_AC, AgentType.MC_AC]
        return agent_type in pg_methods
