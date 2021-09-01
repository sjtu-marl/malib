import os.path as osp
import gym

from malib.envs import Environment
from malib.utils.typing import Dict, Any, AgentID
from malib.backend.datapool.offline_dataset_server import Episode
from malib.envs.smarts import gen_config


BASE_DIR = osp.dirname(osp.abspath(__file__))


class SMARTS(Environment):
    def __init__(self, **configs):
        super(SMARTS, self).__init__(**configs)

        env_id = self._configs["env_id"]
        scenario_configs: Dict[str, Any] = self._configs["scenario_configs"]

        scenario_paths = scenario_configs["path"]
        agent_type = scenario_configs["agent_type"]

        # generate abs paths
        scenario_paths = list(map(lambda x: osp.join(BASE_DIR, x), scenario_paths))
        max_step = scenario_configs["max_step"]

        parsed_configs = gen_config(
            scenarios=scenario_paths,
            agent_config_file=osp.join(BASE_DIR, "agenst", agent_type),
        )
        env_config = parsed_configs["env_config"]

        # build agent specs with agent interfaces
        self.is_sequential = False
        self.scenarios = scenario_paths
        self._env = gym.make(
            "smarts.env:hiway-v0",
            **env_config,
        )
        self._env.possible_agents = list(self._env.agent_specs.keys())
        self._trainable_agents = self._env.possible_agents
        self._max_step = max_step

    def step(self, actions: Dict[AgentID, Any]) -> Dict[str, Any]:
        observations, rewards, dones, infos = self._env.step(actions)
        # remove dones all
        dones.pop("__all__")
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
