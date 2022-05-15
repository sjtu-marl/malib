import importlib
import gym

from malib.rollout.envs.env import Environment
from malib.utils.typing import Dict, Any, AgentID, List, Union
from malib.utils.episode import Episode, Episode


class MPE(Environment):
    def __init__(self, **configs):
        super(MPE, self).__init__(**configs)

        env_id = self._configs["env_id"]
        scenario_configs = self._configs.get("scenario_configs", {})

        env_module = importlib.import_module(f"pettingzoo.mpe.{env_id}")
        ori_caller = env_module.parallel_env

        self.is_sequential = False
        self._env = ori_caller(**scenario_configs)
        self._trainable_agents = self._env.possible_agents
        self._action_spaces = {
            aid: self._env.action_space(aid) for aid in self._env.possible_agents
        }
        self._observation_spaces = {
            aid: self._env.observation_space(aid) for aid in self._env.possible_agents
        }
        self.max_step = 25  # default is 25

    @property
    def possible_agents(self) -> List[AgentID]:
        return self._env.possible_agents

    @property
    def action_spaces(self) -> Dict[AgentID, gym.Space]:
        return self._action_spaces

    @property
    def observation_spaces(self) -> Dict[AgentID, gym.Space]:
        return self._observation_spaces

    def time_step(self, actions: Dict[AgentID, Any]) -> Dict[str, Any]:
        # for agent, action in actions.items():
        #     assert self.action_spaces[agent].contains(action), f"Action is not in space: {action} with type={type(action)}"
        observations, rewards, dones, infos = self._env.step(actions)
        return observations, rewards, dones, infos

    def render(self, *args, **kwargs):
        self._env.render()

    def close(self):
        pass

    def reset(
        self, max_step: int = None, custom_reset_config: Dict[str, Any] = None
    ) -> Union[None, Dict[str, Dict[AgentID, Any]]]:
        super(MPE, self).reset(
            max_step=max_step, custom_reset_config=custom_reset_config
        )
        observations = self._env.reset()
        return observations
