import statistics
import gym
import time
import ray

from malib import settings

from malib.utils.typing import (
    Dict,
    Any,
    AgentID,
    Callable,
    ParameterDescription,
    TaskDescription,
    MetaParameterDescription,
    BufferDescription,
    Tuple,
    Union,
    PolicyID,
)
from malib.utils.logger import Logger
from malib.algorithm.common.policy import Policy
from malib.agent.indepdent_agent import IndependentAgent


def gen_policy_description(policy_template: Policy, env_desc, name="MAPPO"):
    agents = env_desc["possible_agents"]
    env_id = env_desc["config"]["env_id"]
    rets = {}

    for agent in agents:
        rets[agent] = (
            f"{name}_0",
            policy_template.description,
            ParameterDescription(
                time.time(),
                identify=agent,
                env_id=env_id,
                id=f"{name}_0",
                lock=False,
                description=policy_template.description,
                data=policy_template.state_dict(),
            ),
        )

    return rets


class SimpleLearner(IndependentAgent):
    def __init__(
        self,
        runtime_id: str,
        env_desc: Dict[str, Any],
        algorithm_candidates: Dict[str, Any],
        training_agent_mapping: Callable,
        observation_spaces: Dict[AgentID, gym.spaces.Space],
        action_spaces: Dict[AgentID, gym.spaces.Space],
        exp_cfg: Dict[str, Any],
        population_size: int = -1,
        use_init_policy_pool: bool = False,
        algorithm_mapping: Callable = None,
        local_buffer_config: Dict = None,
    ):
        IndependentAgent.__init__(
            self,
            runtime_id,
            env_desc,
            algorithm_candidates,
            training_agent_mapping,
            observation_spaces,
            action_spaces,
            exp_cfg,
            population_size,
            use_init_policy_pool,
            algorithm_mapping,
            local_buffer_config,
        )

    def start(self) -> None:
        self._parameter_server = ray.get_actor(settings.PARAMETER_SERVER_ACTOR)
        self._offline_dataset = ray.get_actor(settings.OFFLINE_DATASET_ACTOR)

    def optimize(
        self,
        policy_ids: Dict[AgentID, PolicyID],
        batch: Dict[AgentID, Any],
        training_config: Dict[str, Any],
    ) -> Dict[str, float]:
        """Execute optimization for a group of policies with given batches.

        :param policy_ids: Dict[AgentID, PolicyID], Mapping from environment agent ids to policy ids. The agent ids in
            this dictionary should be registered in groups, and also policy ids should have been existed ones in the
            policy pool.
        :param Dict[Agent, Any] batch, Mapping from agent ids to batches.
        :param Dict[Agent,Any] training_config: Training configuration.
        :return: An agent-wise training statistics dict.
        """

        res = {}
        pid = policy_ids[self.agent_group()[0]]
        trainer = self.get_trainer(pid)
        trainer.reset(self.policies[pid], training_config)
        training_info = trainer.optimize(batch[self.runtime_id])

        res.update(
            dict(
                map(
                    lambda kv: (f"{self.runtime_id}/{kv[0]}", kv[1]),
                    training_info.items(),
                )
            )
        )

        return res

    def add_policy(self, trainable: bool = True):
        trainable = True
        # policy_dict = {
        #     env_aid: self.add_policy_for_agent(env_aid, trainable)
        #     for env_aid in self._group
        # }

        Logger.info(
            "Learner={} adds policies for agent={}".format(
                self.runtime_id, self.agent_group()
            )
        )

        # random select an agent for policy generation
        pid, policy = self.add_policy_for_agent(
            self.agent_group()[0], trainable=trainable
        )

        for env_aid in self.agent_group():
            self._agent_to_pids[env_aid].append(pid)

        parameter_desc = self.parameter_desc_gen(
            self.runtime_id, pid, trainable, data=policy.state_dict()
        )
        ray.get(self._parameter_server.push.remote(parameter_desc=parameter_desc))
        parameter_desc.data = None
        self._parameter_desc[pid] = parameter_desc

        if self._meta_parameter_desc.get(self.runtime_id, None) is None:
            self._meta_parameter_desc[self.runtime_id] = MetaParameterDescription(
                meta_pid=self.runtime_id, parameter_desc_dict={}
            )

        self._meta_parameter_desc[self.runtime_id].parameter_desc_dict[
            pid
        ] = parameter_desc

    def train(self, training_config: Dict[str, Any] = None):
        """Execute only one epoch per call

        :param training_config: _description_, defaults to None
        :type training_config: Dict[str, Any], optional
        """

        batch_size = training_config.get("batch_size", 64)
        sample_mode = None

        env_aid = self.agent_group()[0]
        policy_id_mapping = {
            env_aid: self._agent_to_pids[env_aid][-1] for env_aid in self.agent_group()
        }
        buffer_desc = self.gen_buffer_description(
            policy_id_mapping,
            batch_size,
            sample_mode,
        )

        batch, size = self.request_data(buffer_desc)
        statistics = self.optimize(policy_id_mapping, batch, training_config)

        pid = policy_id_mapping[env_aid]
        status = self.push(self.runtime_id, pid)

        return statistics
