import argparse

from malib.envs import MPE
from malib.runner import run
from malib.utils.typing import PolicyID

# from malib.utils.tasks_register import task_handler_register, register_task_type
from malib.backend.coordinator.server import CoordinatorServer


@CoordinatorServer.task_handler_register
def pre_launching(self, init_config=None):
    assert hasattr(self, load_model)
    assert hasattr(self, save_model)
    assert hasattr(self, load_model)


@CoordinatorServer.task_handler_register
def load_model(self, checkpoint_path):
    print("Customize loading model")
    self._training_manager.load(checkpoint_path)


@CoordinatorServer.task_handler_register
def save_model(self, pid: PolicyID = None):
    print("Customize saving model")
    self._training_manager.dump(pid)


@CoordinatorServer.task_handler_register
def load_data(self, data_path):
    print("Customize loading data")
    self._offline_data_server.load(data_path)


parser = argparse.ArgumentParser(
    "Register customized tasks and handlers to the coordinator"
)
parser.add_argument(
    "--num_learner",
    type=int,
    default=3,
    help="The number of agent training interfaces. Default by 3.",
)
parser.add_argument(
    "--batch_size", type=int, default=64, help="Trianing batch size. Default by 64."
)
parser.add_argument(
    "--num_epoch", type=int, default=100, help="Training epoch. Default by 100."
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="DQN",
    help="The single-agent RL algortihm registered in MALib. Default by DQN",
)

if __name__ == "__main__":
    args = parser.parse_args()
    env_config = {
        "scenario_configs": {"max_cycles": 25},
        "env_id": "simple_v2",
    }
    env = MPE(**env_config)
    possible_agents = env.possible_agents
    observation_spaces = env.observation_spaces
    action_spaces = env.action_spaces

    run(
        group="example/",
        name="register_customized_tasks",
        init={
            "load_model": True,
            "model_path": "checkpoints/model.pth",
            "load_data": True,
            "data_path": "data/data.pkl",
        },
        env_description={
            "creator": MPE,
            "config": env_config,
            "possible_agents": possible_agents,
        },
        agent_mapping_func=lambda agent: [
            f"{agent}_async_{i}" for i in range(args.num_learner)
        ],
        training={
            "interface": {
                "type": "async",
                "observation_spaces": observation_spaces,
                "action_spaces": action_spaces,
                "population_size": -1,
            },
            "config": {
                "update_interval": 1,
                "saving_interval": 10,
                "batch_size": args.batch_size,
                "num_epoch": 100,
                "return_gradients": True,
            },
        },
        algorithms={
            "Async": {"name": args.algorithm},
        },
        rollout={
            "type": "async",
            "stopper": "simple_rollout",
            "metric_type": "simple",
            "fragment_length": env_config["scenario_configs"]["max_cycles"],
            "num_episodes": 100,  # episode for each evaluation/training epoch
            "terminate": "any",
        },
        global_evaluator={
            "name": "generic",
            "config": {"stop_metrics": {}},
        },
        dataset_config={"episode_capacity": 30000},
    )
