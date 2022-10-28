from typing import Dict, Any, Tuple, List, Union, Sequence

import gym

from malib.utils.typing import AgentID
from malib.utils.logging import Logger
from malib.rollout.envs.env import Environment


class MDPEnvironment(Environment):
    def __init__(self, **configs):
        super().__init__(**configs)

        try:
            from blackhc import mdp
            from blackhc.mdp import example as mdp_examples
        except ImportError as e:
            Logger.error(
                "please run `pip install -e .[dev]` before using MDPEnvironment"
            )
            raise e

        scenarios = {
            "one_round_dmdp": mdp_examples._one_round_dmdp,
            "two_round_dmdp": mdp_examples._two_round_dmdp,
            "one_round_nmdp": mdp_examples._one_round_nmdp,
            "two_round_nmdp": mdp_examples._two_round_nmdp,
            "multi_round_nmdp": mdp_examples._multi_round_nmdp,
        }

        env_id = configs["env_id"]
        if env_id not in scenarios:
            raise ValueError(
                f"env_id={env_id} not a legal environment id, you should init mdp environments from one of them: {scenarios.keys()}"
            )

        self.env = scenarios[env_id]().to_env()
        self._possible_agents = ["agent"]

    @property
    def possible_agents(self) -> List[AgentID]:
        return self._possible_agents

    @property
    def observation_spaces(self) -> Dict[AgentID, gym.Space]:
        return dict.fromkeys(self.possible_agents, self.env.observation_space)

    @property
    def action_spaces(self) -> Dict[AgentID, gym.Space]:
        return dict.fromkeys(self.possible_agents, self.env.action_space)

    def time_step(
        self, actions: Dict[AgentID, Any]
    ) -> Tuple[
        Dict[AgentID, Any],
        Dict[AgentID, float],
        Dict[AgentID, bool],
        Dict[AgentID, Any],
    ]:
        obs, rew, done, info = self.env._step(actions["agent"])

        obs = dict.fromkeys(self.possible_agents, obs)
        rew = dict.fromkeys(self.possible_agents, rew)
        done = dict.fromkeys(self.possible_agents, done)

        return None, obs, rew, done, info

    def reset(self, max_step: int = None) -> Union[None, Sequence[Dict[AgentID, Any]]]:
        super(MDPEnvironment, self).reset(max_step=max_step)

        observation = self.env._reset()
        return None, dict.fromkeys(self.possible_agents, observation)

    def close(self):
        return self.env.close()

    def render(self, *args, **kwargs):
        return self.env._render()

    def seed(self, seed: int = None):
        return self.env.seed(seed)
