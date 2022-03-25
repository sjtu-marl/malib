import time
import ray
import numpy as np

from ray.util import ActorPool

from malib import settings
from malib.rollout import rollout_func

from malib.utils.typing import BufferDescription, Dict, AgentID
from malib.utils.episode import EpisodeKey
from malib.rollout.inference_server import InferenceWorkerSet
from malib.rollout.inference_client import InferenceClient
from malib.envs.agent_interface import AgentInterface


@ray.remote(num_cpus=0)
class _RemoteEnv:
    def __init__(self, creater, env_config) -> None:
        self.env = creater(**env_config)

    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        return ret

    def step(self, action):
        ret = self.env.step(action)
        return ret


def act_func(obs, act_space):
    return np.asarray([act_space.sample() for _ in obs])


def run_simple_loop(env_desc, num):
    envs = [env_desc["creator"](**env_desc["config"]) for _ in range(num)]
    action_spaces = env_desc["action_spaces"]

    rets = [env.reset() for env in envs]
    for i, ret in enumerate(rets):
        rets[i][EpisodeKey.NEXT_OBS] = ret[EpisodeKey.CUR_OBS]

    done = False
    step_limit = 3000

    cnt = 0
    n_frames = 0

    start_time = time.time()
    while len(envs) > 0 and cnt < step_limit:
        new_envs = []
        new_rets = []
        for i, env in enumerate(envs):
            actions = {
                aid: act_func(obs, action_spaces[aid])
                for aid, obs in rets[i][EpisodeKey.NEXT_OBS].items()
            }
            ret = env.step(actions)
            done = ret[EpisodeKey.DONE]["__all__"]
            if not done:
                new_envs.append(env)
                new_rets.append(ret)
            else:
                n_frames += cnt
        envs = new_envs
        rets = new_rets
        cnt += 1
    end_time = time.time()

    if len(envs) > 0:
        n_frames += cnt * len(envs)

    return n_frames, n_frames / (end_time - start_time)


def run_simple_ray_pool(env_desc, num):
    creator = env_desc["creator"]
    config = env_desc["config"]
    action_spaces = env_desc["action_spaces"]
    # create remote env

    envs = [_RemoteEnv.remote(creater=creator, env_config=config) for _ in range(num)]
    rets = ray.get([env.reset.remote() for env in envs])
    for i, ret in enumerate(rets):
        rets[i][EpisodeKey.NEXT_OBS] = ret[EpisodeKey.CUR_OBS]

    done = False
    step_limit = 3000

    cnt = 0
    n_frames = 0

    start_time = time.time()
    while len(envs) > 0 and cnt < step_limit:
        new_envs = []
        new_rets = []
        actions = {
            aid: act_func(obs, action_spaces[aid])
            for aid, obs in rets[i][EpisodeKey.NEXT_OBS].items()
        }
        rets_vec = ray.get([env.step.remote(actions) for env in envs])
        for i, ret in enumerate(rets_vec):
            done = ret[EpisodeKey.DONE]["__all__"]
            if not done:
                new_envs.append(envs[i])
                new_rets.append(ret)
            else:
                n_frames += cnt
        envs = new_envs
        rets = new_rets
        cnt += 1
    end_time = time.time()

    if len(envs) > 0:
        n_frames += cnt * len(envs)

    return n_frames, n_frames / (end_time - start_time)


def run_async_vec_env(
    env_desc,
    num,
    runtime_configs,
    agent_interfaces: Dict[AgentID, InferenceWorkerSet] = None,
    actor_pool: ActorPool = None,
):
    # runtime_configs required:
    #   num_episodes,
    #   num_env_per_worker
    #   flags
    #   use_subproc_env
    #   batch_mode
    #   max_step
    #   postprocessor_types
    #   policy_combinations
    #   trainable_pairs
    obs_spaces = env_desc["observation_spaces"]
    act_spaces = env_desc["action_spaces"]
    parameter_server = ray.get_actor(name=settings.PARAMETER_SERVER_ACTOR)

    trainable_pairs = runtime_configs["trainable_pairs"]
    policy_distribution = None
    policy_combinations = runtime_configs["policy_combinations"]

    if agent_interfaces is None:
        agent_interfaces = {
            agent: InferenceWorkerSet.remote(
                agent_id=agent,
                observation_space=obs_spaces[agent],
                action_space=act_spaces[agent],
                parameter_server=parameter_server,
                force_weight_update=True,
            )
            for agent in env_desc["possible_agents"]
        }

    # stepping num: num_episodses, num_env_per_worker
    num_rollout_actors = (
        runtime_configs["num_episodes"] // runtime_configs["num_env_per_worker"]
    )
    num_eval_actors = 1

    if runtime_configs["flags"] == "rollout":
        dataserver = ray.get_actor(name=settings.OFFLINE_DATASET_ACTOR)
    else:
        dataserver = None

    if actor_pool is None:
        actors = [
            InferenceClient.remote(
                env_desc,
                dataserver,
                use_subproc_env=runtime_configs["use_subproc_env"],
                batch_mode=runtime_configs["batch_mode"],
                postprocessor_types=["defaults"],
            )
            for _ in range(num_rollout_actors + num_eval_actors)
        ]

        actor_pool = ActorPool(actors)

    # start eval
    tasks = [
        {
            "flag": runtime_configs["flags"],
            "behavior_policies": policy_combinations[0],
            "policy_distribution": policy_distribution,
            "parameter_desc_dict": runtime_configs["parameter_desc_dict"],
            "num_episodes": runtime_configs["num_env_per_worker"],
            "max_step": runtime_configs["max_step"],
            "postprocessor_types": runtime_configs["postprocessor_types"],
        }
        for _ in range(num_rollout_actors)
    ]

    if runtime_configs["flags"] == "rollout":
        tasks.extend(
            [
                {
                    "flag": "evaluation",
                    "behavior_policies": policy_combinations[0],
                    "policy_distribution": policy_distribution,
                    "parameter_desc_dict": runtime_configs["parameter_desc_dict"],
                    "num_episodes": 4,
                    "max_step": runtime_configs["max_step"],
                }
                for _ in range(num_eval_actors)
            ]
        )

    buffer_desc = BufferDescription(
        env_id=env_desc["config"][
            "env_id"
        ],  # TODO(ziyu): this should be move outside "config"
        agent_id=list(trainable_pairs.keys()),
        policy_id=[pid for pid, _ in trainable_pairs.values()],
        capacity=None,
        sample_start_size=None,
    )

    if dataserver:
        ray.get(dataserver.create_table.remote(buffer_desc, ignore=True))

    rets = actor_pool.map(
        lambda a, task: a.run.remote(
            agent_interfaces=agent_interfaces,
            desc=task,
            buffer_desc=buffer_desc,
        ),
        tasks,
    )

    num_frames = 0
    stats_list = []
    fps = 0.0
    num_workers = num_eval_actors + num_rollout_actors
    avg_connect, avg_env_reset, avg_policy_step, avg_env_step = 0.0, 0.0, 0.0, 0.0

    for ret in rets:
        # we retrieve only results from evaluation/simulation actors.
        performance = ret["performance"]

        avg_connect += performance["inference_server_connect"] / num_workers
        avg_env_reset += performance["environment_reset"] / num_workers
        avg_policy_step += performance["policy_step"] / num_workers
        avg_env_step += performance["environment_step"] / num_workers
        num_frames += ret["total_fragment_length"]
        fps += performance["FPS"]

        if ret["task_type"] in ["evaluation", "simulation"]:
            stats_list.append(ret["eval_info"])

    env_rollout_fps = fps
    eval_stats = stats_list

    # average evaluation results
    merged_eval_stats = {}
    for e in stats_list:
        for k, v in e.items():
            if k not in merged_eval_stats:
                merged_eval_stats[k] = []
            merged_eval_stats[k].append(v)
    for k in merged_eval_stats.keys():
        merged_eval_stats[k] = np.mean(merged_eval_stats[k])

    return num_frames, env_rollout_fps, eval_stats


def run_vec_env(env_desc, num, runtime_configs, agent_interfaces=None):
    obs_spaces = env_desc["observation_spaces"]
    act_spaces = env_desc["action_spaces"]
    parameter_server = ray.get_actor(name=settings.PARAMETER_SERVER_ACTOR)

    policy_description = runtime_configs["policy_description"]
    trainable_pairs = runtime_configs["trainable_pairs"]
    policy_distribution = None
    policy_combinations = runtime_configs["policy_combinations"]

    if agent_interfaces is None:
        agent_interfaces = {
            aid: AgentInterface(aid, obs_spaces[aid], act_spaces[aid], parameter_server)
            for aid in env_desc["possible_agents"]
        }

        for agent, interface in agent_interfaces.items():
            policy_id, policy_description, parameter_desc = policy_description[agent]
            interface.add_policy(
                env_aid=agent,
                policy_id=policy_id,
                policy_description=policy_description,
                parameter_desc=parameter_desc,
            )
    for interface in agent_interfaces.values():
        interface.update_weights(["MAPPO_0"], True)

    # stepping num: num_episodses, num_env_per_worker
    num_rollout_actors = (
        runtime_configs["num_episodes"] // runtime_configs["num_env_per_worker"]
    )
    num_eval_actors = 1

    Stepping = rollout_func.Stepping.as_remote()
    if runtime_configs["flags"] == "rollout":
        dataserver = ray.get_actor(name=settings.OFFLINE_DATASET_ACTOR)
    else:
        dataserver = None

    actors = [
        Stepping.remote(
            env_desc,
            dataserver,
            use_subproc_env=runtime_configs["use_subproc_env"],
            batch_mode=runtime_configs["batch_mode"],
            postprocessor_types=["value"],
        )
        for _ in range(num_rollout_actors + num_eval_actors)
    ]

    rollout_actor_pool = ActorPool(actors)

    # start eval
    tasks = [
        {
            "flag": runtime_configs["flags"],
            "num_episodes": runtime_configs["num_env_per_worker"],
            "behavior_policies": policy_combinations[0],
            "policy_distribution": policy_distribution,
            "fragment_length": runtime_configs["fragment_length"],
        }
        for _ in range(num_rollout_actors)
    ]

    if runtime_configs["flags"] == "rollout":
        tasks.extend(
            [
                {
                    "flag": "evaluation",
                    "num_episodes": 1,
                    "behavior_policies": policy_combinations[0],
                    "policy_distribution": policy_distribution,
                    "fragment_length": 1 * 3000,
                }
                for _ in range(num_eval_actors)
            ]
        )

    buffer_desc = BufferDescription(
        env_id=env_desc["config"][
            "env_id"
        ],  # TODO(ziyu): this should be move outside "config"
        agent_id=list(trainable_pairs.keys()),
        policy_id=[pid for pid, _ in trainable_pairs.values()],
        capacity=None,
        sample_start_size=None,
    )

    if dataserver:
        ray.get(dataserver.create_table.remote(buffer_desc, ignore=True))

    start = time.time()
    rets = rollout_actor_pool.map(
        lambda a, task: a.run.remote(
            agent_interfaces=agent_interfaces,
            fragment_length=task["fragment_length"],
            desc=task,
            buffer_desc=buffer_desc,
        ),
        tasks,
    )

    num_frames = 0
    stats_list = []

    for ret in rets:
        # we retrieve only results from evaluation/simulation actors.
        if ret[0] in ["evaluation", "simulation"]:
            stats_list.append(ret[1]["eval_info"])
            if runtime_configs["flags"] == "evaluation":
                num_frames += ret[1]["total_fragment_length"]
        # and total fragment length tracking from rollout actors
        elif ret[0] == "rollout":
            num_frames += ret[1]["total_fragment_length"]
        else:
            raise ValueError("Unknow task type: {}".format(ret[0]))
    end = time.time()

    env_rollout_fps = num_frames / (end - start)
    eval_stats = stats_list

    # average evaluation results
    merged_eval_stats = {}
    for e in stats_list:
        for k, v in e.items():
            if k not in merged_eval_stats:
                merged_eval_stats[k] = []
            merged_eval_stats[k].append(v)
    for k in merged_eval_stats.keys():
        merged_eval_stats[k] = np.mean(merged_eval_stats[k])

    return num_frames, env_rollout_fps, eval_stats
