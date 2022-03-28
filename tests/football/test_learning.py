"""
Run this file with `pytest ./tests/football/test_learning.py -s`.
"""

from collections import ChainMap
import os
import traceback
import yaml
import time
import pytest
import ray

from ray.util.queue import Queue
from ray.util.actor_pool import ActorPool
from torch.utils import tensorboard

from malib import settings
from malib.utils.typing import List, ParameterDescription, Sequence, Dict, AgentID
from malib.utils import logger
from malib.utils.logger import Logger
from malib.envs import gr_football
from malib.rollout.inference_client import InferenceClient
from malib.rollout.inference_server import InferenceWorkerSet
from malib.backend.datapool import parameter_server
from malib.backend.datapool import offline_dataset_server

from tests.football.rollout_case import (
    run_simple_loop,
    run_simple_ray_pool,
    run_vec_env,
    run_async_vec_env,
)
from tests.football.training_case import SimpleLearner


BLOCK_SIZE = 2000


def write_to_tensorboard(
    writer: tensorboard.SummaryWriter, info: Dict, global_step: int, prefix: str = ""
):
    """Write learning info to tensorboard.
    Args:
        writer (tensorboard.SummaryWriter): The summary writer instance.
        info (Dict): The information dict.
        global_step (int): The global step indicator.
        prefix (str): Prefix added to keys in the info dict.
    """
    if writer is None:
        return

    prefix = f"{prefix}/" if len(prefix) > 0 else ""
    for k, v in info.items():
        if isinstance(v, dict):
            # add k to prefix
            write_to_tensorboard(writer, v, global_step, f"{prefix}{k}")
        elif isinstance(v, Sequence):
            writer.add_scalar(f"{prefix}{k}", sum(v), global_step=global_step)
            # raise NotImplementedError(
            #     f"Sequence value cannot be logged currently: {k}, {v}."
            # )
        elif v is None:
            continue
        else:
            writer.add_scalar(f"{prefix}{k}", v, global_step=global_step)


@pytest.fixture(scope="session")
def env_desc():

    return gr_football.env_desc_gen(
        {
            "env_id": "PSGFootball",
            "use_built_in_GK": True,
            "scenario_configs": {
                "env_name": "5_vs_5",
                "number_of_left_players_agent_controls": 4,
                "number_of_right_players_agent_controls": 0,
                "representation": "raw",
                "stacked": False,
                "logdir": "/tmp/football/malib_psro",
                "write_goal_dumps": False,
                "write_full_episode_dumps": False,
                "render": False,
            },
        }
    )


@pytest.fixture(scope="session")
def servers():
    if not ray.is_initialized():
        ray.init()

    dataset_config = {"episode_capacity": 300, "fragment_length": BLOCK_SIZE}

    ParameterServer = parameter_server.ParameterServer.as_remote()
    pserver = ParameterServer.options(
        name=settings.PARAMETER_SERVER_ACTOR, max_concurrency=100
    ).remote()
    dataset_server = offline_dataset_server.OfflineDataset.options(
        name=settings.OFFLINE_DATASET_ACTOR, max_concurrency=100
    ).remote(dataset_config)

    return (pserver, dataset_server)


@ray.remote
def run_optimize(request_queue, response_queue, env_desc, yaml_config, exp_cfg):
    try:
        remote_learner_cls = SimpleLearner.as_remote(num_gpus=1, num_cpus=4)
        possible_agents = env_desc["possible_agents"]
        observation_spaces = env_desc["observation_spaces"]
        action_spaces = env_desc["action_spaces"]
        training_agent_mapping = lambda agent: agent[:6]
        algorithms = yaml_config["algorithms"]
        local_buffer_config = yaml_config["training"]["interface"].get(
            "local_buffer_config", None
        )

        learners = {
            aid: remote_learner_cls.remote(
                aid,
                env_desc,
                algorithms,
                training_agent_mapping,
                observation_spaces,
                action_spaces,
                exp_cfg=exp_cfg,
                use_init_policy_pool=False,
                local_buffer_config=local_buffer_config,
            )
            for aid in possible_agents
        }

        # sorted_env_agents = sorted(env_desc["possible_agents"])
        for aid, learner in learners.items():
            ray.get(learner.start.remote())
            ray.get(learner.register_env_agent.remote(aid))
            ray.get(learner.add_policy.remote())

        loop_cnt = 0
        ave_FPS = 0
        batch_size = yaml_config["training"]["config"]["batch_size"]
        total_frames = 0

        training_config = yaml_config["training"]["config"]

        while True:
            if not request_queue.empty():
                message = request_queue.get()
                if message["op"] == "terminate":
                    response_queue.put({"logs": {}, "status": 200})
                    break

            start = time.time()
            statistics = ray.get(
                [learner.train.remote(training_config) for learner in learners.values()]
            )
            end = time.time()

            statistics = dict(ChainMap(*statistics))
            washed = {}
            for k, v in statistics.items():
                washed[f"training/{k}"] = v

            FPS = (
                len(learners)
                * batch_size
                * BLOCK_SIZE
                * training_config["ppo_epoch"]
                / (end - start)
            )
            total_frames += len(learners) * batch_size * BLOCK_SIZE
            ave_FPS = (ave_FPS * loop_cnt + FPS) / (loop_cnt + 1)
            loop_cnt += 1

            Logger.info(
                "training at epoch {} with TFPS={} ave_TFPS={} total_frames={}".format(
                    loop_cnt, FPS, ave_FPS, total_frames
                )
            )

            response_queue.put(
                {
                    "logs": {
                        "training/N_Frames": len(learners) * batch_size * BLOCK_SIZE,
                        "training/FPS": FPS,
                        "training/AVE_FPS": ave_FPS,
                        "training/Total_Frames": total_frames,
                        **washed,
                    },
                    "status": 200,
                }
            )
    except Exception as e:
        traceback.print_exc()
        raise e


@ray.remote
def run_rollout(
    request_queue: Queue,
    response_queue: Queue,
    env_desc,
    yaml_config,
    exp_cfg,
    algo_name,
):
    try:
        total_frames, ave_FPS = 0, 0.0
        loop_cnt = 0
        num_envs = 16
        num_env_per_worker = 4

        pserver = ray.get_actor(name=settings.PARAMETER_SERVER_ACTOR)

        obs_spaces = env_desc["observation_spaces"]
        action_spaces = env_desc["action_spaces"]

        model_config = yaml_config["algorithms"][algo_name]["model_config"]
        custom_config = yaml_config["algorithms"][algo_name].get("custom_config", {})

        _description = {
            "registered_name": algo_name,
            "observation_space": obs_spaces["team_0"],
            "action_space": action_spaces["team_0"],
            "model_config": model_config,
            "custom_config": custom_config,
        }

        agent_interfaces = {
            agent: InferenceWorkerSet.options(num_cpus=10).remote(
                agent_id=agent,
                observation_space=obs_spaces[agent],
                action_space=action_spaces[agent],
                parameter_server=pserver,
                force_weight_update=False,
            )
            for agent in env_desc["possible_agents"]
        }

        num_rollout_actors = num_envs // num_env_per_worker
        num_eval_actors = 1

        p_descs: Dict[AgentID, ParameterDescription] = {
            agent: ParameterDescription(
                time_stamp=time.time(),
                identify=agent,
                env_id=env_desc["config"]["env_id"],
                id=f"{algo_name}_0",
                type=ParameterDescription.Type.PARAMETER,
                lock=False,
                description=_description,
                data=None,
                parallel_num=1,
                version=0,
            )
            for agent in env_desc["possible_agents"]
        }

        runtime_configs = {
            "num_episodes": num_envs,
            "num_env_per_worker": num_env_per_worker,
            "use_subproc_env": False,
            "batch_mode": "episode",
            "max_step": BLOCK_SIZE,
            "flags": "rollout",
            "postprocessor_types": ["defaults"],
            "parameter_desc_dict": p_descs,
            "trainable_pairs": {"team_0": (f"{algo_name}_0", None)},
            "policy_combinations": [{"team_0": f"{algo_name}_0"}],
        }

        actor_pool = ActorPool(
            [
                InferenceClient.remote(
                    env_desc,
                    ray.get_actor(settings.OFFLINE_DATASET_ACTOR),
                    use_subproc_env=runtime_configs["use_subproc_env"],
                    batch_mode=runtime_configs["batch_mode"],
                    postprocessor_types=runtime_configs["postprocessor_types"],
                )
                for _ in range(num_rollout_actors + num_eval_actors)
            ]
        )

        while True:
            if not request_queue.empty():
                message = request_queue.get()
                if message["op"] == "terminate":
                    response_queue.put({"logs": {}, "status": 200})
                    break

            # update weights
            # N_Frames, FPS, eval_stats = run_vec_env(
            #     env_desc, None, runtime_configs, agent_interfaces
            # )
            N_Frames, FPS, eval_stats = run_async_vec_env(
                env_desc, None, runtime_configs, agent_interfaces, actor_pool
            )

            total_frames += N_Frames
            ave_FPS = (ave_FPS * loop_cnt + FPS) / (loop_cnt + 1)
            loop_cnt += 1
            Logger.info(
                "rollout epoch at {} with RFPS={} ave_RFPS={} total_frames={}".format(
                    loop_cnt, FPS, ave_FPS, total_frames
                )
            )

            eval_stats = eval_stats[0]
            washed_eval = {}
            for k, v in eval_stats.items():
                washed_eval[f"rollout/{k}"] = v

            response_queue.put(
                {
                    "logs": {
                        "rollout/N_Frames": N_Frames,
                        "rollout/FPS": FPS,
                        "rollout/AVE_FPS": ave_FPS,
                        "rollout/Total_Frames": total_frames,
                        **washed_eval,
                    },
                    "status": 200,
                }
            )

            # save every 10
            if loop_cnt % 10 == 0:
                save_dir = os.path.join(
                    settings.LOG_DIR,
                    exp_cfg["expr_group"],
                    exp_cfg["expr_name"],
                    "models",
                )
                for aid, interface in agent_interfaces.items():
                    _save_dir = os.path.join(save_dir, aid)
                    ray.get(interface.save.remote(_save_dir))
    except Exception as e:
        traceback.print_exc()
        raise e


# XXX(ming): @yanxue, if you wanna run ppo, please replace the string mappo with ppo
@pytest.mark.parametrize("yaml_name", ["ppo"])
def test_learning(env_desc, servers, yaml_name: str):
    comm_optimization = [Queue(), Queue()]
    comm_rollout = [Queue(), Queue()]

    yaml_path = os.path.join(
        settings.BASE_DIR, f"examples/mappo_gfootball/{yaml_name}_5_vs_5.yaml"
    )
    # load yaml
    with open(yaml_path, "r") as f:
        configs = yaml.safe_load(f)

    algo_name = yaml_name.upper()
    if configs["algorithms"][algo_name].get("custom_config"):
        configs["algorithms"][algo_name]["custom_config"].update(
            {"global_state_space": env_desc["state_spaces"]}
        )

    start_ray_info = ray.nodes()[0]

    exp_cfg = logger.start(
        group="experiment",
        name=f"case_{time.time()}",
        host=start_ray_info["NodeManagerAddress"],
    )

    run_optimize.remote(*comm_optimization, env_desc, configs, exp_cfg)
    time.sleep(10)
    # start rollout and optimization process
    run_rollout.remote(*comm_rollout, env_desc, configs, exp_cfg, algo_name)

    responders: List[Queue] = [comm_rollout[1], comm_optimization[1]]
    senders: List[Queue] = [comm_rollout[0], comm_optimization[0]]

    # create local writer
    writer = tensorboard.SummaryWriter(
        log_dir=os.path.join(settings.BASE_DIR, "logs/test/case_{}".format(time.time()))
    )

    cnt = 0
    empty = True
    while True:

        for responder in responders:
            if responder.empty():
                continue
            empty = False
            message = responder.get_nowait()
            # parse message
            write_to_tensorboard(writer, message["logs"], cnt)

        if not empty:
            cnt += 1
        empty = True

        if cnt == 2000:
            for sender in senders:
                sender.put({"op": "terminate"})
            break
    ray.shutdown()
