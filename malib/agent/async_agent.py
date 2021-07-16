"""
Implementation of async training workflow. Async agent support async learning of independent
reinforcement learning, or, which perform single agent training behavior. `AsyncAgent` has different behaviors compared
to `IndependentAgent`. Specifically, it push a list of stacked gradients instead of parameters to remote **ParameterServer**,
then it will try to pull the newest parameter from remote server. Note this operation will finish until all the other
async agents have pushed their stacked gradients, and the aggregation operation is done.
"""

from collections import defaultdict

import time
import gym
import ray

from malib.utils.typing import (
    Dict,
    Any,
    Callable,
    List,
    AgentID,
    PolicyID,
    Status,
    ParameterDescription,
    MetricEntry,
)
from malib.utils import metrics
from malib.agent.indepdent_agent import IndependentAgent


class AsyncAgent(IndependentAgent):
    def __init__(
        self,
        assign_id: str,
        env_desc: Dict[str, Any],
        algorithm_candidates: Dict[str, Any],
        training_agent_mapping: Callable,
        observation_spaces: Dict[AgentID, gym.spaces.Space],
        action_spaces: Dict[AgentID, gym.spaces.Space],
        exp_cfg: Dict[str, Any],
        population_size: int,
        algorithm_mapping: Callable = None,
    ):
        """Create an independent agent instance works in async mode.

        This agent interface inherits most parts of `IndependentAgent`, but performs different behavior in the push
        stage. In details, async agent interface push stacked gradients instead of parameters.

        :param str assign_id: Naming async agent interface.
        :param Dict[str,Any] env_desc: Environment description.
        :param Dict[str,Any] algorithm_candidates: Mapping from readable name to algorithm configuration.
        :param Callable training_agent_mapping: Callable, Mapping from environment agents to training agent interfaces.
        :param Dict[AgentID, gym.spaces.Space] observation_spaces: Dict of raw agent observation spaces, it is a
            completed description of all possible agents' observation spaces.
        :param Dict[AgentID, gym.spaces.Space] action_spaces: Dict of raw agent action spaces, it is a completed
            description of all possible agents' action spaces.
        :param Dict[str,Any] exp_cfg: Experiment description.
        :param int population_size: int, The maximum number of policies in the policy pool. Default to -1, which means no
            limitation.
        :param Callable algorithm_mapping: Mapping from agent to algorithm name in `algorithm_candidates`, for
            constructing your custom algorithm configuration getter. It is optional. Default to None, which means
            random selection.
        """

        IndependentAgent.__init__(
            self,
            assign_id=assign_id,
            env_desc=env_desc,
            algorithm_candidates=algorithm_candidates,
            training_agent_mapping=training_agent_mapping,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            exp_cfg=exp_cfg,
            population_size=population_size,
            algorithm_mapping=algorithm_mapping,
        )

        self._cumulative_grads: Dict[PolicyID, List[Dict]] = defaultdict(lambda: [])

    def parameter_desc_gen(
        self, env_aid: AgentID, policy_id: PolicyID, trainable: bool, data=None
    ):
        """Called when add policy, gen parameter description with locking.

        :param AgentID env_aid: Environment agent id.
        :param PolicyID policy_id: Policy id.
        :param bool trainable: Tag this policy is trainable or not.
        :param Any data: Parameter or gradients.
        :return: A parameter description entity.
        """

        policy = self.policies[policy_id]
        res = self._training_agent_mapping(env_aid)

        if isinstance(res, str):
            parallel_num = 1
        else:
            parallel_num = len(res)

        return ParameterDescription(
            env_id=self._env_desc["id"],
            identify=env_aid,
            id=policy_id,
            time_stamp=time.time(),
            description=policy.description,
            data=data,
            # non trainable policy will be locked, no updates for it.
            lock=not trainable,
            parallel_num=parallel_num,
        )

    def push(self, env_aid: AgentID, pid: PolicyID) -> Status:
        """Push stacked gradients to remote parameter server.

        :param AgentID env_aid: Environment agent id.
        :param PolicyID pid: Policy id.
        :return: Returned gradients pushing status.
        """

        parameter_desc = self._parameter_desc[pid]
        parameter_desc.type = "gradient"
        parameter_desc.data = self._cumulative_grads[pid] or []
        status = ray.get(self._parameter_server.push.remote(parameter_desc))
        parameter_desc.data = None
        # clean old gradients
        self._cumulative_grads[pid] = []
        return status

    def pull(self, env_aid: AgentID, pid: PolicyID) -> Status:
        """Pull latest parameters from remote parameter server.

        Note:
            This looped operation will terminate when returned status.gradient_status is not Waiting. Legal
            gradient status could be Status.NORMAL, Status.Waiting and Status.LOCKED. Specifically, Status.WAITING means
            that other training interfaces have not upload their stacked gradients yet; Status.NORMAL means the remote
            parameter version could be pulled down, all stacked gradients have been applied; Status.LOCKED means no
            gradients will be accepted, remote parameter has been locked.

        :param AgentID env_aid: Agent id.
        :param PolicyID pid: Policy id.
        :return: Returned parameters pulling status `TableStatus(locked, gradient_status)`.
        """

        parameter_desc = self._parameter_desc[pid]
        policy = self.policies[pid]
        parameter_desc.data = None

        # waiting until all gradients have been applied
        # FIXME(ming): pull may failed since the gradient status switched too fast
        while True:
            status, content = ray.get(
                self._parameter_server.pull.remote(parameter_desc)
            )
            if status.gradient_status in [Status.NORMAL, Status.LOCKED]:
                break

        self._parameter_desc[pid].version += 1
        # debug_tools.compare(policy.actor.state_dict(), content.data["actor"])
        policy.set_weights(content.data)
        return status

    def optimize(
        self,
        policy_ids: Dict[AgentID, PolicyID],
        batch: Dict[AgentID, Any],
        training_config: Dict[str, Any],
    ) -> Dict[AgentID, Dict[str, MetricEntry]]:
        """Execute optimization and gradient collecting for a group of policies with given batches.

        :param Dict[AgentID,PolicyID] policy_ids: Mapping from environment agent ids to policy ids. The agent ids in
            this dictionary should be registered in groups, and also policy ids should have been existed ones in the
            policy pool.
        :param Dict[Agent,Any] batch: Mapping from agent ids to batches.
        :param Dict[str,Any] training_config: Training configuration.
        :return: An agent-wise training statistics dict.
        """

        res = {}
        for env_aid, pid in policy_ids.items():
            trainer = self.get_trainer(pid)
            trainer.reset(self.policies[pid], training_config)
            res[env_aid] = trainer.optimize(batch[env_aid])
            assert (
                res[env_aid].get("gradients") is not None
            ), f"You must return gradients from optimizer {type(trainer)}"
            gradients = res[env_aid].pop("gradients")
            res[env_aid] = metrics.to_metric_entry(res[env_aid])
            self._cumulative_grads[pid].append(gradients)

        return res

    def save(self, model_dir):
        pass

    def load(self, model_dir):
        pass
