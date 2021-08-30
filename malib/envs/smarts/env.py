import importlib
import gym

from malib.envs import Environment
from malib.utils.typing import Dict, Any, AgentID
from malib.backend.datapool.offline_dataset_server import Episode
from malib.envs.smarts._env.smarts.core.agent_interface import AgentInterface, AgentType
from malib.envs.smarts._env.smarts.core.agent import AgentSpec, Agent


class SMARTS(Environment):
    def __init__(self, **configs):
        super(SMARTS, self).__init__(**configs)

        env_id = self._configs["env_id"]
        scenario_configs = self._configs.get("scenario_configs", {})
        agent_specs = scenario_configs["agent_specs"]

        # TODO(ming): read agent specs from scenarios folder
        agent_specs = {}
        self.is_sequential = False
        self._env = env = gym.make(
            "smarts.env:hiway-v0",
            scenarios=[env_id],
            agent_specs=agent_specs,
        )
        self._env.possible_agents = None
        self._trainable_agents = self._env.agents

    def step(self, actions: Dict[AgentID, Any]) -> Dict[str, Any]:
        observations, rewards, dones, infos = self._env.step(actions)
        return {
            Episode.NEXT_OBS: observations,
            Episode.REWARD: rewards,
            Episode.DONE: dones,
            Episode.INFO: infos,
        }

    def render(self, *args, **kwargs):
        self._env.render()

    def reset(self):
        observations = self._env.reset()
        return {Episode.CUR_OBS: observations}
