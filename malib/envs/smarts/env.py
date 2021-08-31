import importlib
import gym

from malib.envs import Environment
from malib.utils.typing import Dict, Any, AgentID
from malib.backend.datapool.offline_dataset_server import Episode
from malib.envs.smarts import gen_config


class SMARTS(Environment):
    def __init__(self, **configs):
        super(SMARTS, self).__init__(**configs)

        env_id = self._configs["env_id"]
        scenario_configs = self._configs.get("scenario_configs", {})
        parsed_configs = gen_config(**scenario_configs)

        env_config = parsed_configs["env_config"]
        agent_config = parsed_configs["agent"]

        # build agent specs with agent interfaces
        agent_specs = {
            agent: interface for agent, interface in agent_config["interface"]
        }

        self.is_sequential = False
        self._env = gym.make(
            "smarts.env:hiway-v0",
            scenarios=[env_id],
            agent_specs=agent_specs,
        )
        self._env.possible_agents = None
        self._trainable_agents = self._env.agents
        self._max_step = 1000

    def step(self, actions: Dict[AgentID, Any]) -> Dict[str, Any]:
        observations, rewards, dones, infos = self._env.step(actions)
        super(SMARTS, self).step(actions, rewards=rewards, dones=dones, infos=infos)

        return {
            Episode.CUR_OBS: observations,
            Episode.REWARD: rewards,
            Episode.DONE: dones,
        }

    def render(self, *args, **kwargs):
        self._env.render()

    def reset(self, *args, **kwargs):
        observations = super(SMARTS, self).reset(*args, **kwargs)
        self._max_step = self._max_step or kwargs.get("max_step", None)
        return {Episode.CUR_OBS: observations}
