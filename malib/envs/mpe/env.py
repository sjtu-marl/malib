import importlib

from malib.envs import Environment
from malib.utils.typing import Dict, Any, AgentID
from malib.backend.datapool.offline_dataset_server import Episode


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
