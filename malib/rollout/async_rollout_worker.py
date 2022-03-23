import ray
import traceback
import time
import os
import operator
import copy
import numpy as np

from functools import reduce
from ray.util import ActorPool

from malib import settings
from malib.utils.typing import (
    Dict,
    List,
    AgentID,
    PolicyID,
    TaskDescription,
    Tuple,
    Sequence,
    BufferDescription,
    Any,
    Status,
    RolloutFeedback,
    TaskRequest,
    TaskType,
)
from malib.utils.general import iter_many_dicts_recursively
from malib.utils.logger import get_logger, Log, Logger
from malib.utils.stoppers import get_stopper
from malib.utils.remote.ray_actor import RayActor
from malib.rollout.inference_server import InferenceWorkerSet


def _parse_rollout_info(raw_statistics: List[Dict[str, Any]]) -> Dict[str, Any]:
    holder = {}
    for history, ds, k, vs in iter_many_dicts_recursively(*raw_statistics, history=[]):
        prefix = "/".join(history)
        vs = reduce(operator.add, vs)
        holder[f"{prefix}_mean"] = np.mean(vs)
        holder[f"{prefix}_max"] = np.max(vs)
        holder[f"{prefix}_min"] = np.min(vs)
    return holder


class AsyncRolloutWorker(RayActor):
    def __init__(
        self, worker_index: Any, env_desc: Dict[str, Any], save: bool = False, **kwargs
    ):
        self._worker_index = worker_index
        self._env_description = env_desc
        self._agents = env_desc["possible_agents"]
        self._kwargs = kwargs

        self._num_rollout_actors = kwargs.get("num_rollout_actors", 1)
        self._num_eval_actors = kwargs.get("num_eval_actors", 1)
        # XXX(ming): computing resources for rollout / evaluation
        self._resources = kwargs.get(
            "resources",
            {
                "num_cpus": None,
                "num_gpus": None,
                "memory": None,
                "object_store_memory": None,
                "resources": None,
            },
        )

        assert (
            self._num_rollout_actors > 0
        ), f"num_rollout_actors should be positive, but got `{self._num_rollout_actors}`"

        self.connect()
        self.init_pool(env_desc, **kwargs)

        self.logger = get_logger(
            log_level=settings.LOG_LEVEL,
            log_dir=settings.LOG_DIR,
            name="async_rollout_worker_{}".format(os.getpid()),
            remote=settings.USE_REMOTE_LOGGER,
        )

    def connect(self):
        retries = 100
        while True:
            try:
                if self._coordinator is None:
                    self._coordinator = ray.get_actor(settings.COORDINATOR_SERVER_ACTOR)

                if self._parameter_server is None:
                    self._parameter_server = ray.get_actor(
                        settings.PARAMETER_SERVER_ACTOR
                    )

                if self._offline_dataset is None:
                    self._offline_dataset = ray.get_actor(
                        settings.OFFLINE_DATASET_ACTOR
                    )
                self._status = Status.IDLE
                break
            except Exception as e:
                retries -= 1
                if retries == 0:
                    self.logger.error("reached maximum retries")
                    raise RuntimeError(traceback.format_exc())
                else:
                    self.logger.warning(
                        f"waiting for coordinator server initialization ... {self._worker_index}\n{traceback.format_exc()}"
                    )
                    time.sleep(1)

    def init_pool(self, env_desc, **kwargs):
        self.actors = []
        self.actors.extend(
            [InferenceClient.remote() for _ in range(self._num_eval_actors)]
        )
        self.rollout_actor_pool = ActorPool(self.actor[: self._num_rollout_actors])
        self.eval_actor_pool = ActorPool(self.actors[self._num_eval_actors])
        self.agent_interfaces = {}

        # init inference worker set
        act_spaces = self._env_description["action_spaces"]
        obs_spaces = self._env_description["observation_spaces"]

        for aid in self._agents:
            self.agent_interfaces[aid] = InferenceWorkerSet.remote(
                aid,
                action_space=act_spaces[aid],
                observation_space=obs_spaces[aid],
                parameter_server=self._parameter_server,
            )

    def _set_state(self, task_desc: TaskDescription):
        self.add_policies(task_desc)
        # XXX(ming): collect to task description
        if hasattr(task_desc.content, "policy_distribution"):
            self.ready_for_sample(
                policy_distribution=task_desc.content.policy_distribution
            )

    def sample(
        self,
        num_episodes: int,
        fragment_length: int,
        role: str,
        policy_combinations: List,
        policy_distribution: Dict[AgentID, Dict[PolicyID, float]] = None,
        buffer_desc: BufferDescription = None,
    ) -> Tuple[Sequence[Dict[str, List]], int]:
        if role == "simulation":
            tasks = [
                {
                    "num_episodes": num_episodes,
                    "behavior_policies": comb,
                    "flag": "simulation",
                }
                for comb in policy_combinations
            ]
            actor_pool = self.eval_actor_pool
        elif role == "rollout":
            seg_num = self._num_rollout_actors
            x = num_episodes // seg_num
            y = num_episodes - seg_num * x
            episode_segs = [x] * seg_num + ([y] if y else [])
            assert len(policy_combinations) == 1
            assert policy_distribution is not None
            tasks = [
                {
                    "flag": "rollout",
                    "num_episodes": episode,
                    "behavior_policies": policy_combinations[0],
                    "policy_distribution": policy_distribution,
                }
                for episode in episode_segs
            ]
            # add tasks for evaluation
            tasks.extend(
                [
                    {
                        "flag": "evaluation",
                        "num_episodes": 4,  # FIXME(ziyu): fix it and debug
                        "behavior_policies": policy_combinations[0],
                        "policy_distribution": policy_distribution,
                    }
                    for _ in range(self._num_eval_actors)
                ]
            )
            actor_pool = self.rollout_actor_pool
        else:
            raise TypeError(f"Unkown role: {role}")

        # self.check_actor_pool_available()
        rets = actor_pool.map(
            lambda a, task: a.run.remote(
                agent_interfaces=self.agent_interfaces,
                fragment_length=fragment_length,
                desc=task,
                buffer_desc=buffer_desc,
            ),
            tasks,
        )

        num_frames, stats_list = 0, []
        for ret in rets:
            # we retrieve only results from evaluation/simulation actors.
            if ret[0] in ["evaluation", "simulation"]:
                stats_list.append(ret[1]["eval_info"])
            # and total fragment length tracking from rollout actors
            if ret[0] == "rollout":
                num_frames += ret[1]["total_fragment_length"]

        return stats_list, num_frames

    def rollout(self, task_desc: TaskDescription):
        """Collect training data asynchronously and stop it until the evaluation results meet the stopping conditions"""

        stopper = get_stopper(task_desc.content.stopper)(
            config=task_desc.content.stopper_config, tasks=None
        )
        merged_statics = {}
        epoch = 0
        self._set_state(task_desc)
        start_time = time.time()
        total_num_frames = 0
        print_every = 100  # stopper.max_iteration // 3

        # create data table
        trainable_pairs = task_desc.content.agent_involve_info.trainable_pairs
        # XXX(ming): shall we authorize learner to determine the buffer description?
        buffer_desc = BufferDescription(
            env_id=self._env_description["config"][
                "env_id"
            ],  # TODO(ziyu): this should be move outside "config"
            agent_id=list(trainable_pairs.keys()),
            policy_id=[pid for pid, _ in trainable_pairs.values()],
            capacity=None,
            sample_start_size=None,
        )
        ray.get(self._offline_dataset.create_table.remote(buffer_desc))

        while not stopper(merged_statics, global_step=epoch):
            trainable_behavior_policies = {
                aid: pid
                for aid, (
                    pid,
                    _,
                ) in task_desc.content.agent_involve_info.trainable_pairs.items()
            }
            # get behavior policies of other fixed agent
            raw_statistics, num_frames = self.sample(
                num_episodes=task_desc.content.num_episodes,
                policy_combinations=[trainable_behavior_policies],
                fragment_length=task_desc.content.fragment_length,
                role="rollout",
                policy_distribution=task_desc.content.policy_distribution,
                buffer_desc=buffer_desc,
            )

            self.after_rollout(task_desc.content.agent_involve_info.trainable_pairs)
            total_num_frames += num_frames
            time_consump = time.time() - start_time

            holder = _parse_rollout_info(raw_statistics)

            # log to tensorboard
            if (epoch + 1) % print_every == 0:
                Logger.info("\tepoch: %s (evaluation) %s", epoch, holder)
            if self.logger.is_remote:
                for k, v in holder.items():
                    self.logger.send_scalar(
                        tag="Evaluation/{}".format(k),
                        content=v,
                        global_step=epoch,
                    )
                self.logger.send_scalar(
                    tag="Performance/rollout_FPS",
                    content=total_num_frames / time_consump,
                    global_step=epoch,
                )
            epoch += 1

        # self.save_model()

        rollout_feedback = RolloutFeedback(
            worker_idx=self._worker_index,
            agent_involve_info=task_desc.content.agent_involve_info,
            statistics=holder,
        )
        self.callback(task_desc, rollout_feedback, role="rollout", relieve=True)

    @Log.method_timer(enable=settings.PROFILING)
    def simulation(self, task_desc: TaskDescription):
        """Handling simulation task."""

        self._set_state(task_desc)
        combinations = task_desc.content.policy_combinations
        agent_involve_info = task_desc.content.agent_involve_info
        # print(f"simulation for {task_desc.content.num_episodes}")
        raw_statistics, num_frames = self.sample(
            num_episodes=task_desc.content.num_episodes,
            fragment_length=task_desc.content.max_episode_length,
            policy_combinations=[
                {k: p for k, (p, _) in comb.items()} for comb in combinations
            ],
            role="simulation",
        )
        for statistics, combination in zip(raw_statistics, combinations):
            holder = _parse_rollout_info([statistics])
            rollout_feedback = RolloutFeedback(
                worker_idx=self._worker_index,
                agent_involve_info=agent_involve_info,
                statistics=holder,
                policy_combination={k: p for k, (p, _) in combination.items()},
            )
            task_req = TaskRequest.from_task_desc(
                task_desc=task_desc,
                task_type=TaskType.UPDATE_PAYOFFTABLE,
                content=rollout_feedback,
            )
            self._coordinator.request.remote(task_req)
        self.set_status(Status.IDLE)

    def callback(
        self,
        task_desc: TaskDescription,
        content: Any,
        role: str,
        relieve: bool,
    ):
        if role == "simulation":
            task_req = TaskRequest.from_task_desc(
                task_desc=task_desc,
                task_type=TaskType.UPDATE_PAYOFFTABLE,
                content=content,
            )
            self._coordinator.request.remote(task_req)
        else:
            parameter_desc_dict = content.agent_involve_info.meta_parameter_desc_dict
            for agent, (
                pid,
                _,
            ) in content.agent_involve_info.trainable_pairs.items():
                parameter_desc = copy.copy(
                    parameter_desc_dict[agent].parameter_desc_dict[pid]
                )
                parameter_desc.type = "parameter"
                parameter_desc.lock = True
                parameter_desc.data = (
                    self._agent_interfaces[agent].policies[pid].state_dict()
                )
                _ = ray.get(self._parameter_server.push.remote(parameter_desc))
            self._coordinator.request.remote(
                TaskRequest.from_task_desc(
                    task_desc=task_desc,
                    task_type=TaskType.EVALUATE,
                    content=content,
                )
            )
        if relieve:
            # unlock worker
            self.set_status(Status.IDLE)

    def get_status(self):
        return self._status

    def set_status(self, status):
        if status == self._status:
            return Status.FAILED
        else:
            self._status = status
            return Status.SUCCESS
