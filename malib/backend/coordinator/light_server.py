import ray

from malib.utils.typing import (
    TaskDescription,
    TaskRequest,
    TaskType,
    TrainingDescription,
    RolloutDescription,
    TrainingFeedback,
)
from malib.utils.logging import Logger
from malib.manager.rollout_worker_manager import RolloutWorkerManager
from malib.manager.training_manager import TrainingManager


@ray.remote
class LightCoordinator:
    def start(self, yaml_config, env_desc, exp_cfg):

        self._configs = yaml_config
        training_config = yaml_config["training"]["config"]
        algorithms = yaml_config["algorithms"]
        rollout_config = yaml_config["rollout"]

        self._training_manager = TrainingManager(
            algorithms,
            env_desc,
            interface_config=yaml_config["training"]["interface"],
            training_agent_mapping=lambda agent: agent,
            training_config=training_config,
            exp_cfg=exp_cfg,
        )

        rollout_config["worker_num"] = self._training_manager.get_agent_interface_num()
        self._rollout_manager = RolloutWorkerManager(rollout_config, env_desc, exp_cfg)

        self._training_manager.init()

    def _request_optimize(self, task_request):
        task_request = self._training_manager.retrieve_information(task_request)
        task_desc = TaskDescription(
            task_type=TaskType.OPTIMIZE,
            content=TrainingDescription(
                agent_involve_info=task_request.content.agent_involve_info,
                stopper=self._configs["training"]["config"]["stopper"],
                stopper_config=self._configs["training"]["config"].get(
                    "stopper_config", None
                ),
                batch_size=self._configs["training"]["config"]["batch_size"],
                update_interval=self._configs["training"]["config"]["update_interval"],
            ),
            state_id=None,
        )
        self._training_manager.optimize(task_desc)

    def _request_rollout(self, task_request):
        task_request = self._training_manager.retrieve_information(task_request)
        assert isinstance(task_request.content, TrainingFeedback)

        populations = task_request.content.agent_involve_info.populations
        population_mapping = {}
        for k, v in populations.items():
            assert len(v) > 0, v
            population_mapping[k] = [p[0] for p in v]
        agent_involve_info = task_request.content.agent_involve_info

        policy_distribution = {}
        for aid, plist in population_mapping.items():
            policy_distribution[aid] = dict(zip(plist, [1 / len(plist)] * len(plist)))

        Logger.info("population {}".format(policy_distribution))

        rollout_config = self._configs["rollout"]
        task = TaskDescription(
            task_type=TaskType.ROLLOUT,
            content=RolloutDescription(
                agent_involve_info=agent_involve_info,
                policy_distribution=policy_distribution,
                fragment_length=rollout_config["fragment_length"],
                num_episodes=rollout_config["num_episodes"],
                stopper=rollout_config["stopper"],
                stopper_config=rollout_config["stopper_config"],
                terminate_mode=rollout_config["terminate"],
                mode=rollout_config["mode"],
                callback=rollout_config["callback"],
                episode_seg=rollout_config["episode_seg"],
            ),
            state_id=None,
        )

        self._rollout_manager.rollout(task_desc=task)

    def request(self, task_request: TaskRequest):
        if task_request.task_type == TaskType.ROLLOUT:
            self._request_rollout(task_request)
        elif task_request.task_type == TaskType.OPTIMIZE:
            self._request_optimize(task_request)
        else:
            raise ValueError(
                "Unregistered task request type: {}".format(task_request.task_type)
            )
