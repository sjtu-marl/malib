"""
Independent agent interface, for independent algorithms training. Policy/Trainer adding rule: one policy one trainer.
"""

from typing import Dict, Tuple, Any, Callable

import gym

from malib.utils.typing import (
    AgentID,
    PolicyID,
    MetricEntry,
)

from malib.agent.agent_interface import AgentInterface
from malib.algorithm.common.policy import Policy
from malib.algorithm import get_algorithm_space
from malib.utils import metrics

import pickle as pkl


class IndependentAgent(AgentInterface):
    def __init__(
        self,
        assign_id: str,
        env_desc: Dict[str, Any],
        algorithm_candidates: Dict[str, Any],
        training_agent_mapping: Callable,
        observation_spaces: Dict[AgentID, gym.spaces.Space],
        action_spaces: Dict[AgentID, gym.spaces.Space],
        exp_cfg: Dict[str, Any],
        population_size: int = -1,
        algorithm_mapping: Callable = None,
    ):
        """Create an independent agent instance work in asynchronous mode.

        :param str assign_id: Naming independent agent interface.
        :param Dict[str,Any] env_desc: Environment description.
        :param Dict[str,Any] algorithm_candidates: Mapping from readable name to algorithm configuration.
        :param Callable training_agent_mapping: Mapping from environment agents to training agent interfaces.
        :param Dict[AgentID,gym.spaces.Space] observation_spaces: Dict of raw agent observation spaces, it is a
            completed description of all possible agents' observation spaces.
        :param Dict[Agent,gym.spaces.Space] action_spaces: Dict of raw agent action spaces, it is a completed
            description of all possible agents' action spaces.
        :param Dict[str,Any] exp_cfg: Experiment description.
        :param int population_size: The maximum number of policies in the policy pool. Default to -1, which means no
            limitation.
        :param Callable algorithm_mapping: Mapping from agent to algorithm name in `algorithm_candidates`, for
            constructing your custom algorithm configuration getter. It is optional. Default to None, which means
            random selection.
        """

        AgentInterface.__init__(
            self,
            assign_id,
            env_desc,
            algorithm_candidates,
            training_agent_mapping,
            observation_spaces,
            action_spaces,
            exp_cfg,
            population_size,
            algorithm_mapping,
        )

    def optimize(
        self,
        policy_ids: Dict[AgentID, PolicyID],
        batch: Dict[AgentID, Any],
        training_config: Dict[str, Any],
    ) -> Dict[AgentID, Dict[str, MetricEntry]]:
        """Execute optimization for a group of policies with given batches.

        :param policy_ids: Dict[AgentID, PolicyID], Mapping from environment agent ids to policy ids. The agent ids in
            this dictionary should be registered in groups, and also policy ids should have been existed ones in the
            policy pool.
        :param Dict[Agent, Any] batch, Mapping from agent ids to batches.
        :param Dict[Agent,Any] training_config: Training configuration.
        :return: An agent-wise training statistics dict.
        """

        res = {}
        for env_aid, pid in policy_ids.items():
            trainer = self.get_trainer(pid)
            if env_aid not in batch:
                print("[WARNING] No registered agent id detected in batch keys, please check your buffer description")
                continue
            trainer.reset(self.policies[pid], training_config)
            res[env_aid] = metrics.to_metric_entry(
                trainer.optimize(batch[env_aid]), prefix=pid
            )
        return res

    def add_policy_for_agent(
        self, env_agent_id: AgentID, trainable: bool
    ) -> Tuple[PolicyID, Policy]:

        """Add new policy according to env_agent_id.

        :param AgentID env_agent_id: The agent_id with which observation, action space will be determined if is None.
        :param bool trainable: Whether the added policy is trainable or not.
        :return:
        """

        assert env_agent_id in self._group, (env_agent_id, self._group)
        algorithm_conf = self.get_algorithm_config(env_agent_id)
        algorithm = get_algorithm_space(algorithm_conf["name"])
        policy = algorithm.policy(
            registered_name=algorithm_conf["name"],
            observation_space=self._observation_spaces[env_agent_id],
            action_space=self._action_spaces[env_agent_id],
            model_config=algorithm_conf.get("model_config", {}),
            custom_config=algorithm_conf.get("custom_config", {}),
        )

        pid = self.default_policy_id_gen(algorithm_conf)
        self._policies[pid] = policy
        self._trainers[pid] = algorithm.trainer(env_agent_id)

        return pid, policy

    def save(self, model_dir: str) -> None:
        """Save policies and states.

        :param str model_dir: Model saving directory path.
        :return: None
        """

        raise NotImplementedError

    def load(self, model_dir) -> None:
        """Load states and policies from local storage.

        :param str model_dir: Local model directory path.
        :return: None
        """

        raise NotImplementedError

    def load_single_policy(self, env_agent_id, model_dir) -> None:
        """Load one policy for one env_agent.

        Temporarily used for single agent imitation learning.
        """

        assert env_agent_id in self._group, (env_agent_id, self._group)
        algorithm_conf = self.get_algorithm_config(env_agent_id)

        with open(model_dir, "rb") as f:
            policy = pkl.load(f)

        pid = self.default_policy_id_gen(algorithm_conf)
        self._policies[pid] = policy

        return pid, policy
