"""
Users can register their rollout func here, with the same parameters list like method `sequential`
and return a Dict-like metric results.

Examples:
    >>> def custom_rollout_function(
    ...     agent_interfaces: List[env.AgentInterface],
    ...     env_desc: Dict[str, Any],
    ...     metric_type: str,
    ...     max_iter: int,
    ...     behavior_policy_mapping: Dict[AgentID, PolicyID],
    ... ) -> Dict[str, Any]

In your custom rollout function, you can decide extra data
you wanna save by specifying extra columns when Episode initialization.
"""

import collections
from typing import Iterator
import ray
import numpy as np

from malib import settings
from malib.utils.general import iter_many_dicts_recursively
from malib.utils.typing import (
    AgentID,
    BufferDescription,
    Dict,
    Any,
    Tuple,
    List,
    BehaviorMode,
    EnvID,
    Callable,
    Union,
)
from malib.utils.logger import Log
from malib.utils.episode import Episode, NewEpisodeDict, EpisodeKey
from malib.rollout.postprocessor import get_postprocessor
from malib.envs.vector_env import VectorEnv, SubprocVecEnv
from malib.envs.agent_interface import AgentInterface


def _process_environment_returns(
    env_rets: Dict[EnvID, Dict[str, Dict[AgentID, Any]]],
    agent_interfaces: Dict[AgentID, AgentInterface],
    filtered_env_outputs: Dict[EnvID, Dict[str, Dict[AgentID, Any]]],
) -> Tuple[
    Dict[EnvID, Dict[str, Dict[AgentID, Any]]],
    Dict[EnvID, Dict[str, Dict[AgentID, Any]]],
    List[EnvID],
]:
    """Processes environment returns, including observation, rewards. Also the agent
    communication.
    """

    policy_inputs = {}
    drop_env_ids = []

    for env_id, rets in env_rets.items():
        # preset done if no done
        policy_input = {}
        drop = False

        if env_id not in filtered_env_outputs:
            filtered_env_outputs[env_id] = {}
        filtered_env_output = filtered_env_outputs[env_id]

        for k, ret in rets.items():
            if k in [EpisodeKey.CUR_OBS, EpisodeKey.NEXT_OBS]:
                output = {
                    aid: agent_interfaces[aid].transform_observation(
                        observation=obs, state=None
                    )["obs"]
                    for aid, obs in ret.items()
                }
                if k == EpisodeKey.NEXT_OBS:
                    if EpisodeKey.CUR_OBS not in filtered_env_output:
                        filtered_env_output[EpisodeKey.CUR_OBS] = output
                    policy_input[EpisodeKey.CUR_OBS] = output
            elif k == EpisodeKey.NEXT_STATE:
                if EpisodeKey.CUR_STATE not in filtered_env_output:
                    filtered_env_output[EpisodeKey.CUR_STATE] = ret
                policy_input[EpisodeKey.CUR_STATE] = ret
            else:
                if k == EpisodeKey.DONE:
                    done = ret["__all__"]
                    drop = done
                    drop_env_ids.append(env_id)
                    output = {k: v for k, v in ret.items() if k != "__all__"}
                else:
                    output = ret
            policy_input[k] = output
            filtered_env_output[k] = output

        if not drop:
            policy_inputs[env_id] = policy_input
            # we transfer DONE key as a signal for some masking behaviors
            if EpisodeKey.DONE not in policy_input:
                policy_input = {
                    EpisodeKey.DONE: dict.fromkeys(rets[EpisodeKey.CUR_OBS], False)
                }

    return policy_inputs, filtered_env_outputs, drop_env_ids


def _do_policy_eval(
    policy_inputs: Dict[EnvID, Dict[str, Dict[AgentID, Any]]],
    agent_interfaces: Dict[AgentID, AgentInterface],
    episodes: NewEpisodeDict,
) -> Dict[str, Dict[EnvID, Dict[AgentID, Any]]]:
    actions, action_dists, next_rnn_state = {}, {}, {}

    env_ids = list(policy_inputs.keys())
    # we need to link environment id to agent ids, especially in the case of
    # sequential rollout
    env_agent_ids = []

    # collect by agent wise
    agent_wise_inputs = collections.defaultdict(
        lambda: collections.defaultdict(lambda: [])
    )
    for env_id in env_ids:
        env_episode = episodes[env_id]
        # for agent_id, interface in agent_interfaces.items():
        env_agent_ids.append(list(policy_inputs[env_id][EpisodeKey.CUR_OBS].keys()))
        for agent_id in policy_inputs[env_id][EpisodeKey.CUR_OBS].keys():
            interface = agent_interfaces[agent_id]
            if len(env_episode[EpisodeKey.RNN_STATE][agent_id]) < 1:
                obs_shape = policy_inputs[env_id][EpisodeKey.CUR_OBS][agent_id].shape
                env_episode[EpisodeKey.RNN_STATE][agent_id].append(
                    interface.get_initial_state(
                        batch_size=None if len(obs_shape) == 1 else obs_shape[0]
                    )
                )

                # FIXME(ming): maybe wrong in some cases, I didn't load it yet.
                last_done = np.zeros(obs_shape[:-1])
            else:
                last_done = env_episode[EpisodeKey.DONE][agent_id][-1]
            last_rnn_state = env_episode[EpisodeKey.RNN_STATE][agent_id][-1]
            agent_wise_inputs[agent_id][EpisodeKey.RNN_STATE].append(last_rnn_state)
            # rnn mask dependences on done or not
            agent_wise_inputs[agent_id][EpisodeKey.DONE].append(last_done)

        for k, agent_v in policy_inputs[env_id].items():
            for agent_id, v in agent_v.items():
                agent_wise_inputs[agent_id][k].append(v)
    for agent_id, inputs in agent_wise_inputs.items():
        interface = agent_interfaces[agent_id]
        (
            actions[agent_id],
            action_dists[agent_id],
            next_rnn_state[agent_id],
        ) = interface.compute_action(**inputs)

    return {
        EpisodeKey.ACTION: actions,
        EpisodeKey.ACTION_DIST: action_dists,
        EpisodeKey.RNN_STATE: next_rnn_state,
    }, dict(zip(env_ids, env_agent_ids))


def _process_policy_outputs(
    active_env_to_agent_ids: Dict[EnvID, List[AgentID]],
    # policy_outputs: Dict[str, Dict[AgentID, DataTransferType]],
    policy_outputs: Dict[str, Dict[AgentID, Iterator]],
    env: VectorEnv,
) -> Dict[EnvID, Dict[AgentID, Any]]:
    """Proceses the policy returns. Here we convert the policy return to legal environment step inputs."""

    assert (
        EpisodeKey.ACTION in policy_outputs and EpisodeKey.ACTION_DIST in policy_outputs
    ), "`action` and `action_prob` are required in the policy outputs, please check the return of `_do_policy_eval`: {}".format(
        list(policy_outputs.keys())
    )

    detached_policy_outputs = {}
    for i, (env_id, agent_ids) in enumerate(active_env_to_agent_ids.items()):
        detached = collections.defaultdict(lambda: collections.defaultdict())
        for k, agent_v in policy_outputs.items():
            for aid in agent_ids:
                _v = agent_v[aid]
                if k == EpisodeKey.RNN_STATE:
                    detached[k][aid] = [next(__v) for __v in _v]
                else:
                    detached[k][aid] = next(_v)
        detached_policy_outputs[env_id] = detached
    env_actions: Dict[EnvID, Dict[AgentID, Any]] = env.action_adapter(
        detached_policy_outputs
    )
    return env_actions, detached_policy_outputs


# def _reduce_rollout_info(rollout_info) -> Dict[str, float]:
#     res = {}
#     if isinstance(rollout_info, list) or isinstance(rollout_info, tuple):
#         _item = np.array(rollout_info)
#         res["mean"] = np.mean(_item)
#         res["min"] = np.min(_item)
#         res["max"] = np.max(_item)
#     elif isinstance(rollout_info, dict):
#         for k, item in rollout_info.items():
#             res[k] = _reduce_rollout_info(item)
#     else:
#         res = rollout_info

#     return res


def env_runner(
    env: VectorEnv,
    agent_interfaces: Dict[AgentID, AgentInterface],
    buffer_desc: BufferDescription,
    runtime_config: Dict[str, Any],
    dataset_server: ray.ObjectRef = None,
    custom_environment_return_processor: Callable = None,
    custom_policy_output_processor: Callable = None,
    custom_do_policy_eval: Callable = None,
) -> Dict[str, Dict[str, Any]]:
    """Rollout in simultaneous mode, support environment vectorization.

    :param VectorEnv env: The environment instance.
    :param Dict[Agent,AgentInterface] agent_interfaces: The dict of agent interfaces for interacting with environment.
    :param ray.ObjectRef dataset_server: The offline dataset server handler, buffering data if it is not None.
    :return: A dict of rollout information.
    """

    # collect runtime configuration into runtime_config
    # keys in runtime config:
    #   1. num_envs: determines how many environments will this runner use
    #   2. max_step: the maximum length of one episode
    #   3. fragement_length: the total length you wanna run in this runner
    #   4. behavior_policies: a dict of policy ids, mapping from agent id to policy id
    behavior_policies = runtime_config["behavior_policies"]
    sample_dist = runtime_config.get("behavior_policy_dist", None)
    for agent_id, interface in agent_interfaces.items():
        interface.reset(policy_id=behavior_policies[agent_id], sample_dist=sample_dist)

    # if buffer_desc is not None:
    #     assert runtime_config["trainable_mapping"] is None, (runtime_config, dataset_server)

    rets = env.reset(
        limits=runtime_config["num_envs"],
        fragment_length=runtime_config["fragment_length"],
        max_step=runtime_config["max_step"],
        custom_reset_config=runtime_config["custom_reset_config"],
        trainable_mapping=runtime_config["trainable_mapping"],
    )
    if isinstance(env, VectorEnv):
        assert len(env.active_envs) > 0, (env._active_envs, rets, env)

    episodes = NewEpisodeDict(lambda env_id: Episode(behavior_policies, env_id=env_id))

    # process_environment_returns = (
    #     custom_environment_return_processor or _process_environment_returns
    # )
    # process_policy_outputs = custom_policy_output_processor or _process_policy_outputs
    # do_policy_eval = custom_do_policy_eval or _do_policy_eval

    # XXX(ming): currently, we mute all processors' cutomization to avoid unpredictable behaviors
    process_environment_returns = _process_environment_returns
    process_policy_outputs = _process_policy_outputs
    do_policy_eval = _do_policy_eval

    while not env.is_terminated():
        filtered_env_outputs = {}
        # ============ a frame =============
        (
            active_policy_inputs,
            filtered_env_outputs,
            drop_env_ids,
        ) = process_environment_returns(rets, agent_interfaces, filtered_env_outputs)

        active_policy_outputs, active_env_to_agent_ids = do_policy_eval(
            active_policy_inputs, agent_interfaces, episodes
        )

        env_inputs, detached_policy_outputs = process_policy_outputs(
            active_env_to_agent_ids, active_policy_outputs, env
        )

        # XXX(ming): maybe more general inputs.
        rets = env.step(env_inputs)

        # again, filter next_obs here
        (
            active_policy_inputs,
            filtered_env_outputs,
            drop_env_ids,
        ) = process_environment_returns(rets, agent_interfaces, filtered_env_outputs)
        # filter policy inputs here
        # =================================

        episodes.record(detached_policy_outputs, filtered_env_outputs)

    rollout_info = env.collect_info()
    if dataset_server:
        policies = {
            aid: interface.get_policy(behavior_policies[aid])
            for aid, interface in agent_interfaces.items()
        }
        batch_mode = runtime_config["batch_mode"]
        trainable_agents = list(runtime_config["trainable_mapping"].keys())

        episodes = list(episodes.to_numpy(batch_mode, filter=trainable_agents).values())
        for handler in get_postprocessor(runtime_config["postprocessor_types"]):
            episodes = handler(episodes, policies)

        buffer_desc.batch_size = (
            env.batched_step_cnt if batch_mode == "time_step" else len(episodes)
        )
        indices = None
        while indices is None:
            batch = ray.get(dataset_server.get_producer_index.remote(buffer_desc))
            indices = batch.data
        # buffer_desc.batch_size = len(indices)
        buffer_desc.data = episodes
        buffer_desc.indices = indices
        dataset_server.save.remote(buffer_desc)

    ph = list(rollout_info.values())

    holder = {}
    for history, ds, k, vs in iter_many_dicts_recursively(*ph, history=[]):
        arr = [np.sum(_vs) for _vs in vs]
        prefix = "/".join(history)
        # print(history, prefix, _arr, vs)
        holder[prefix] = arr

    return {"total_fragment_length": env.batched_step_cnt, "eval_info": holder}


class Stepping:
    def __init__(
        self,
        env_desc: Dict[str, Any],
        dataset_server=None,
        use_subproc_env: bool = False,
        batch_mode: str = "time_step",
        postprocessor_types: List[Union[str, Callable]] = ["default"],
    ):

        # init environment here
        self.env_desc = env_desc
        self.batch_mode = batch_mode
        self.postprocessor_types = postprocessor_types

        # if not env.is_sequential:
        if use_subproc_env:
            self.env = SubprocVecEnv(
                env_desc["observation_spaces"],
                env_desc["action_spaces"],
                env_desc["creator"],
                env_desc["config"],
                max_num_envs=2,  # FIXME(ziyu): currently just fixed it.
            )
        else:
            self.env = VectorEnv(
                observation_spaces=env_desc["observation_spaces"],
                action_spaces=env_desc["action_spaces"],
                creator=env_desc["creator"],
                configs=env_desc["config"],
            )

        self._dataset_server = dataset_server

    @classmethod
    def as_remote(
        cls,
        num_cpus: int = None,
        num_gpus: int = None,
        memory: int = None,
        object_store_memory: int = None,
        resources: dict = None,
    ) -> type:
        """Return a remote class for Actor initialization."""

        return ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            object_store_memory=object_store_memory,
            resources=resources,
        )(cls)

    @Log.data_feedback(enable=settings.DATA_FEEDBACK)
    def run(
        self,
        agent_interfaces: Dict[AgentID, AgentInterface],
        fragment_length: int,
        desc: Dict[str, Any],
        buffer_desc: BufferDescription = None,
    ) -> Tuple[str, Dict[str, List]]:
        """Environment stepping, rollout/simulate with environment vectorization if it is feasible.

        :param Dict[AgentID,AgentInterface] agent_interface: A dict of agent interfaces.
        :param Union[str,type] metric_type: Metric type or handler.
        :param int fragment_length: The maximum length of an episode.
        :param Dict[str,Any] desc: The description of task.
        :param str role: Indicator of stepping type. Values in `rollout` or `simulation`.
        :returns: A tuple of a dict of MetricEntry and the caculation of total frames.
        """

        task_type = desc["flag"]
        behavior_policies = {}
        if task_type == "rollout":
            for interface in agent_interfaces.values():
                interface.set_behavior_mode(BehaviorMode.EXPLORATION)
        else:
            for interface in agent_interfaces.values():
                interface.set_behavior_mode(BehaviorMode.EXPLOITATION)

        # desc: policy_distribution, behavior_policies, num_episodes
        policy_distribution = desc.get("policy_distribution")
        for agent, interface in agent_interfaces.items():
            if policy_distribution:
                interface.reset(sample_dist=policy_distribution[agent])
            behavior_policies[agent] = interface.behavior_policy

        # behavior policies is a mapping from agents to policy ids
        # update with external behavior_policies
        behavior_policies.update(desc["behavior_policies"] or {})
        # specify the number of running episodes
        num_episodes = desc["num_episodes"]
        max_step = desc.get("max_step", None)

        self.add_envs(num_episodes)

        rollout_results = env_runner(
            self.env,
            agent_interfaces,
            buffer_desc if task_type == "rollout" else None,
            runtime_config={
                "max_step": max_step,
                "fragment_length": fragment_length,
                "num_envs": num_episodes,
                "behavior_policies": behavior_policies,
                "custom_reset_config": None,
                "batch_mode": self.batch_mode,
                "trainable_mapping": desc["behavior_policies"]
                if task_type == "rollout"
                else None,
                "postprocessor_types": self.postprocessor_types,
            },
            dataset_server=self._dataset_server if task_type == "rollout" else None,
        )
        return task_type, rollout_results

    def add_envs(self, maximum: int) -> int:
        """Create environments, if env is an instance of VectorEnv, add these new environment instances into it,
        otherwise do nothing.

        :returns: The number of nested environments.
        """

        if not isinstance(self.env, VectorEnv):
            return 1

        existing_env_num = getattr(self.env, "num_envs", 1)

        if existing_env_num >= maximum:
            return self.env.num_envs

        self.env.add_envs(num=maximum - existing_env_num)

        return self.env.num_envs

    def close(self):
        if self.env is not None:
            self.env.close()
