"""
The `VectorEnv` is an interface that integrates multiple environment instances to support parllel rollout 
with shared multi-agent policies. Currently, the `VectorEnv` support parallel rollout for environment which steps in simultaneous mode.
"""

from collections import defaultdict
import logging
import gym

from malib.utils.typing import Dict, AgentID, Any, List, Tuple
from malib.envs import Environment


logger = logging.getLogger(__name__)


class AgentItems:
    def __init__(self) -> None:
        self._data = {}

    def update(self, agent_items: Dict[AgentID, Any]):
        for k, v in agent_items.items():
            if k not in self._data:
                self._data[k] = []
            self._data[k].append(v)

    def cleaned_data(self):
        return self._data


class VectorEnv:
    def __init__(
        self,
        observation_spaces: Dict[AgentID, gym.Space],
        action_spaces: Dict[AgentID, gym.Space],
        creator: type,
        configs: Dict[str, Any],
        num_envs: int = 0,
    ):
        """Create a VectorEnv instance.

        :param Dict[AgentID,gym.Space] observation_spaces: A dict of agent observation space
        :param Dict[AgentID,gym.Space] action_spaces: A dict of agent action space
        :param type creator: The handler to create environment
        :param Dict[str,Any] configs: Environment configuration
        :param int num_envs: The number of nested environment

        """
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.possible_agents = list(observation_spaces.keys())

        self._num_envs = num_envs
        self._creator = creator
        self._configs = configs.copy()
        self._envs: List[Environment] = []

        if num_envs > 0:
            self._envs.append(creator(**configs))
            self._validate_env(self._envs[0])
            self._envs.extend([creator(**configs) for _ in range(num_envs - 1)])

        self._limits = len(self._envs)

    def _validate_env(self, env: Environment):
        assert not self._envs[
            0
        ].is_sequential, (
            "The nested environment must be an instance which steps in simultaneous."
        )

    @property
    def num_envs(self) -> int:
        """The total number of environments"""

        return self._num_envs

    @property
    def envs(self) -> List[Environment]:
        """Return a limited list of enviroments"""

        return self._envs[: self._limits]

    @property
    def extra_returns(self) -> List[str]:
        """Return extra columns required by this environment"""

        return self.envs[0].extra_returns

    @property
    def env_creator(self):
        return self._creator

    @property
    def env_configs(self):
        return self._configs

    @property
    def limits(self):
        return self._limits

    @classmethod
    def from_envs(cls, envs: List, config: Dict[str, Any]):
        """Generate vectorization environment from exisiting environments."""

        observation_spaces = envs[0].observation_spaces
        action_spaces = envs[0].action_spaces

        vec_env = cls(observation_spaces, action_spaces, config, 0)
        vec_env.add_envs(envs=envs)

        return vec_env

    def add_envs(self, envs: List = None, num: int = 0):
        """Add exisiting `envs` or `num` new environments to this vectorization environment.
        If `envs` is not empty or None, the `num` will be ignored.
        """

        if envs and len(envs) > 0:
            for env in envs:
                self._validate_env(env)
                self._envs.append(env)
                self._num_envs += 1
            logger.debug(f"added {len(envs)} exisiting environments.")
        elif num > 0:
            for _ in range(num):
                self._envs.append(self.env_creator(**self.env_configs))
                self._num_envs += 1
            logger.debug(f"created {num} new environments.")

    def reset(self, limits: int = None) -> Dict:
        self._limits = limits or self.num_envs
        transitions = defaultdict(lambda: AgentItems())
        for i, env in enumerate(self.envs[: self._limits]):
            ret = env.reset()
            for k, agent_items in ret.items():
                transitions[k].update(agent_items)
        data = {k: v.cleaned_data() for k, v in transitions.items()}
        return data

    def step(self, actions: Dict[AgentID, List]) -> Dict:
        transitions = defaultdict(lambda: AgentItems())
        for i, env in enumerate(self.envs):
            ret = env.step({_agent: array[i] for _agent, array in actions.items()})
            for k, agent_items in ret.items():
                transitions[k].update(agent_items)

        # merge transitions by keys
        data = {k: v.cleaned_data() for k, v in transitions.items()}
        return data

    def close(self):
        for env in self._envs:
            env.close()
