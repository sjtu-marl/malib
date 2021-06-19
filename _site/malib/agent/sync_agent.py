"""
Implementation of synchronous agent interface, work with `SyncRolloutWorker`.
"""
import gym.spaces
import ray

from malib.utils.typing import (
    Callable,
    BufferDescription,
    Status,
    Dict,
    AgentID,
    PolicyID,
    Any,
    Tuple,
)
from malib.agent.indepdent_agent import IndependentAgent


class SyncAgent(IndependentAgent):
    """SyncAgent inherits `IndependentAgent`, but work in synchronous mode. SyncAgent will do optimization after one
    rollout task has been terminated
    """

    def save(self, model_dir):
        pass

    def load(self, model_dir):
        pass

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
        """Create an independent agent instance work in synchronous mode.

        :param str assign_id: Naming agent interface.
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

        IndependentAgent.__init__(
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

    def request_data(
        self, buffer_desc: Dict[AgentID, BufferDescription]
    ) -> Tuple[Dict, str]:
        """Request training data from remote `OfflineDatasetServer`.

        Note:
            This method could only be called in multi-instance scenarios. Or, `OfflineDataset` and `CoordinatorServer`
            have been started.

        :param Dict[AgentID,BufferDescription] buffer_desc: A dictionary of agent buffer description
        :return: A tuple of agent batches and information.
        """

        status = Status.FAILED
        wait_list = {aid: None for aid in buffer_desc}
        while status == Status.FAILED:
            status = ray.get(
                self._offline_dataset.lock.remote(
                    lock_type="pull", desc={aid: buffer_desc[aid] for aid in wait_list}
                )
            )
            tmp = Status.SUCCESS
            ks = list(status.keys())
            for k in ks:
                v = status[k]
                if v == Status.SUCCESS:
                    wait_list.pop(k)
                elif v == Status.FAILED:
                    tmp = Status.FAILED
            status = tmp
        assert status == Status.SUCCESS, status

        batch, info = ray.get(self._offline_dataset.sample.remote(buffer_desc))
        if batch is None:
            batch = dict()
        return batch, info

    def push(self, env_aid: AgentID, pid: PolicyID) -> Status:
        """Push parameter to remote parameter server, then unlock pull.

        :param AgentID env_aid: Registered environment agent id.
        :param PolicyID pid: Registered policy id.
        :return: A status code
        """

        status = IndependentAgent.push(self, env_aid, pid)
        # then release table lock
        ray.get(
            self._offline_dataset.unlock.remote(
                lock_type="pull",
                desc={
                    env_aid: BufferDescription(
                        env_id=self._env_desc["id"], agent_id=env_aid, policy_id=pid
                    )
                },
            )
        )

        return status
