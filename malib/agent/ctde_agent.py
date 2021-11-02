"""
This file gives an implementation of a typical training framework in multi-agent learning: Centralized training and
decentralized execution training workflow. Users can use this agent interface to implement algorithms like MADDPG.
Besides, our implementation can work in a more general case: users can register multiple policies in heterogeneous.

Example:
    >>> algorithm_candidates = {
    ...     "PPO": {},
    ...     "DDPG": {},
    ...     "DQN": {}
    ... }
    >>> learner = CTDEAgent("agent_0", ..., algorithm_candidates)
"""

import copy
import gym

from malib.utils.typing import (
    BufferDescription,
    Dict,
    Any,
    Callable,
    Tuple,
    AgentID,
    PolicyID,
    MetricEntry,
)
from malib.agent.agent_interface import AgentInterface
from malib.algorithm.common.policy import Policy
from malib.algorithm import get_algorithm_space
from malib.utils import metrics


class CTDEAgent(AgentInterface):
    """An agent whose learning paradigm is centralized training decentralized execution.
    Notice that, for a single policy in the CTDEAgent, it doesn't need to be CTDE, namely
    In fact, CTDEAgent just specify which global information the inside CTDE-type policy
    can use, e.g. which policies it can pretend to know.
    """

    def __init__(
        self,
        assign_id: str,
        env_desc: Dict[str, Any],
        algorithm_candidates: Dict[str, Any],
        training_agent_mapping: Callable,
        observation_spaces: Dict[AgentID, gym.spaces.Space],
        action_spaces: Dict[AgentID, gym.spaces.Space],
        exp_cfg: Dict[str, Any],
        use_init_policy_pool: bool,
        population_size: int = -1,
        algorithm_mapping: Callable = None,
    ):
        """Create a centralized agent interface instance.

        :param str assign_id: Naming independent agent interface.
        :param Dict[str,Any] env_desc: Environment description.
        :param Dict[str,Any] algorithm_candidates: Mapping from readable name to algorithm configuration.
        :param Callable training_agent_mapping: Mapping from environment agents to training agent interfaces.
        :param Dict[AgentID,gym.spaces.Space] observation_spaces: Dict of raw agent observation spaces, it is a
            completed description of all possible agents' observation spaces.
        :param Dict[Agent,gym.spaces.Space] action_spaces: Dict of raw agent action spaces, it is a completed
            description of all possible agents' action spaces.
        :param Dict[str, Any] exp_cfg: Experiment description.
        :param int population_size: The maximum number of policies in the policy pool. Default to -1, which means no
            limitation.
        :param Callable algorithm_mapping: Mapping from agent to algorithm name in `algorithm_candidates`, for
            constructing your custom algorithm configuration getter. It is optional. Default to None, which means
            random selection.
        """

        AgentInterface.__init__(
            self,
            assign_id=assign_id,
            env_desc=env_desc,
            algorithm_candidates=algorithm_candidates,
            training_agent_mapping=training_agent_mapping,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            exp_cfg=exp_cfg,
            use_init_policy_pool=use_init_policy_pool,
            population_size=population_size,
            algorithm_mapping=algorithm_mapping,
        )

    def gen_buffer_description(
        self,
        agent_policy_mapping: Dict[AgentID, PolicyID],
        batch_size: int,
        sample_mode: str,
    ):
        """Generate a buffer description which description in a batch of agents"""
        agent_policy_mapping = {
            aid: pid for aid, (pid, _) in agent_policy_mapping.items()
        }
        return BufferDescription(
            env_id=self._env_desc["config"]["env_id"],
            agent_id=list(agent_policy_mapping.keys()),
            policy_id=list(agent_policy_mapping.values()),
            batch_size=batch_size,
            sample_mode=sample_mode,
        )

    def optimize(
        self,
        policy_ids: Dict[AgentID, PolicyID],
        batch: Dict[AgentID, Any],
        training_config: Dict[str, Any],
    ) -> Dict[AgentID, Dict[str, MetricEntry]]:
        """Execute optimization for a group of policies with given batches.

        :param Dict[AgentID,PolicyID] policy_ids: Mapping from environment agent ids to policy ids. The agent ids in
            this dictionary should be registered in groups, and also policy ids should have been existed ones in the
            policy pool.
        :param Dict[AgentID,Any] batch: Mapping from agent ids to batches.
        :param Dict[str,Any] training_config: Training configuration
        :return: An agent-wise training statistics dict.
        """

        res = {}
        # extract a group of policies
        t_policies = {}
        for env_agent_id, pid in policy_ids.items():
            t_policies[env_agent_id] = self.policies[pid]

        for env_agent_id, trainer in self._trainers.items():
            trainer.reset(t_policies[env_agent_id], training_config)
            agent_batch = trainer.preprocess(batch, other_policies=t_policies)
            res[env_agent_id] = metrics.to_metric_entry(
                trainer.optimize(agent_batch.copy()),
                prefix=policy_ids[env_agent_id],
            )

        return res

    def add_policy_for_agent(
        self, env_agent_id: AgentID, trainable: bool
    ) -> Tuple[PolicyID, Policy]:
        """Add policy and assign trainer to agent tagged with `env_agent_id`.

        :param AgentID env_agent_id: Environment agent id
        :param boll trainable: Specify the added policy is trainable or not
        :return: A tuple of policy id and policy
        """

        assert env_agent_id in self._group, (env_agent_id, self._group)
        algorithm_conf = self.get_algorithm_config(env_agent_id)
        pid = self.default_policy_id_gen(algorithm_conf)

        if pid in self.policies:
            return pid, self.policies[pid]
        else:
            algorithm_space = get_algorithm_space(algorithm_conf["name"])
            custom_config = algorithm_conf.get("custom_config", {})
            # group spaces into a new space
            custom_config.update(
                {
                    "global_state_space": gym.spaces.Dict(
                        {
                            "observations": gym.spaces.Dict(**self._observation_spaces),
                            "actions": gym.spaces.Dict(**self._action_spaces),
                        }
                    )
                }
            )

            policy = algorithm_space.policy(
                registered_name=algorithm_conf["name"],
                observation_space=self._observation_spaces[env_agent_id],
                action_space=self._action_spaces[env_agent_id],
                model_config=algorithm_conf.get("model_config", {}),
                custom_config=custom_config,
            )

            if env_agent_id not in self._trainers:
                self._trainers[env_agent_id] = algorithm_space.trainer(env_agent_id)

                # register main id and other agents
                self._trainers[env_agent_id].main_id = env_agent_id
                self._trainers[env_agent_id].agents = self.agent_group().copy()
            self.register_policy(pid, policy)
            return pid, policy

    def save(self, model_dir):
        raise NotImplementedError

    def load(self, model_dir):
        raise NotImplementedError
