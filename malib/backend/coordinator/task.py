from malib.utils.tasks_register import task_handler_register
from malib.utils.typing import (
    TaskDescription,
    TaskRequest,
    TaskType,
    TrainingFeedback,
    TrainingDescription,
    RolloutDescription,
)
from malib.backend.coordinator.base_coordinator import BaseCoordinator

task_handler_register = BaseCoordinator.task_handler_register


@task_handler_register
def _request_optimize(coordinator: BaseCoordinator, task_request: TaskRequest):
    task_request = coordinator._training_manager.retrieve_information(task_request)
    task_desc = TaskDescription(
        task_type=TaskType.OPTIMIZE,
        content=TrainingDescription(
            agent_involve_info=task_request.content.agent_involve_info,
            stopper=coordinator._configs["training"]["config"]["stopper"],
            stopper_config=coordinator._configs["training"]["config"].get(
                "stopper_config", None
            ),
            batch_size=coordinator._configs["training"]["config"]["batch_size"],
            update_interval=coordinator._configs["training"]["config"][
                "update_interval"
            ],
        ),
        state_id=None,
    )
    coordinator._training_manager.optimize(task_desc)


@task_handler_register
def _request_rollout(coordinator: BaseCoordinator, task_request: TaskRequest):
    task_request = coordinator._training_manager.retrieve_information(task_request)
    assert isinstance(task_request.content, TrainingFeedback)

    populations = task_request.content.agent_involve_info.populations
    population_mapping = {}
    for k, v in populations.items():
        assert len(v) > 0, v
        population_mapping[k] = [p[0] for p in v]
    agent_involve_info = task_request.content.agent_involve_info

    if all([len(p_list) for p_list in population_mapping.values()]):
        policy_distribution = coordinator._payoff_manager.get_equilibrium(
            population_mapping
        )
        for env_aid, (pid, _) in agent_involve_info.trainable_pairs.items():
            policy_distribution[env_aid] = {pid: 1.0}
        # since in meta_policy this is a default_dict with value 0.0
    else:
        policy_distribution = {}

    rollout_config = coordinator._configs["rollout"]
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

    coordinator._rollout_worker_manager.rollout(task_desc=task)
