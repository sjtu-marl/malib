import logging
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_LEVEL = logging.DEBUG
STATISTIC_FEEDBACK = True
DATA_FEEDBACK = False
USE_REMOTE_LOGGER = True
USE_MONGO_LOGGER = False
PROFILING = False

PARAMETER_SERVER_ACTOR = "ParameterServer"
OFFLINE_DATASET_ACTOR = "OfflineDataset"
COORDINATOR_SERVER_ACTOR = "coordinator"

# default episode capacity when initializing
DEFAULT_EPISODE_INIT_CAPACITY = int(1e6)
# default episode maximum capacity
DEFAULT_EPISODE_CAPACITY = 30000  # int(1e15)
# related to each group of expr settings
DEFAULT_EPISODE_BLOCK_SIZE = int(75)
PICKLE_PROTOCOL_VER = 4

PARAM_DIR = os.path.join(BASE_DIR, "../checkpoints")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# __sphinx_doc_begin__
DEFAULT_CONFIG = {
    # configuration for training agent usage, specify which type you wanna use
    "training": {
        # configuration for agent interface
        "interface": {
            # agent type could be:
            # 1. independent,
            # 2. opponent_modeling,
            # 3. distributed,
            # 4. ctde (centralized training and decentralized execution),
            # 5. fully_centralized (centralized training and decentralized execution)
            # 6. communication (attention based)
            "type": "independent",
            # observation spaces should be a mapping from environment agent ids to observation spaces
            "observation_spaces": None,
            # action spaces should be a mapping from environment agent ids to action spaces
            "action_spaces": None,
            # population size specify the maximum of policy pool size, default by -1, means no limitation
            # on it.
            "population_size": -1,
            # algorithm mapping could be a mapping function from environment agents algorithm keys
            # if default, then training agent interface will add policy randomly.
            "algorithm_mapping": None,
        },
        # training config, determine batch size and ...
        "config": {
            # update interval
            "update_interval": 1,  # 111
            # model saving interval, every 10 epoch
            "saving_interval": 10,
            # training batch size
            "batch_size": 64,
            # stopper function to control the training workflow, default by None, means the learning
            # process not determined by the training results. instead, we specify the stopper in rollout config
            "stopper": "none",
            # optimizer you wanna use in optimization stage, default by SGD
            "optimizer": "SGD",
            # use offline training algorithms, whether or not sampling in the env
            "offline": False,
            "lr": 1e-4,
            "actor_lr": 1e-4,
            "critic_lr": 1e-4,
        },
    },
    # environment description for environment creation, creator, config and id are required
    "env_description": {
        "creator": None,
        "config": None,
        "id": None,  #
        "possible_agents": [],  #
    },
    # mapping environment agent ids to policy ids
    # TODO(ming): support multiple algorithms
    # {id: {"name": human readable algorithm name, "model_config": {}, "custom_config": {}}}
    "algorithms": {},  #
    # use environment reward by default, and can be replaced with adversarial reward, etc.
    "rewards": {
        "ENV": {
            "name": "ENV",
        }
    },
    # mapping environment agents to training agent interfaces
    "agent_mapping_func": lambda agent: agent,
    # configuration for rollout
    "rollout": {
        "type": "async",
        # provide stopping rules for rollout, see rollout/rollout_worker.py::rollout
        "metric_type": "simple",
        "fragment_length": 25000,  #
        # if vector_env is enabled, there will be (num_episodes + episode_seg - 1) // episode_seg environments
        # rollout in parallel.
        "num_episodes": 1000,  #
        "episode_seg": 100,
        "test_num_episodes": 10,
        "test_episode_seg": 10,
        # terminate condition for environment stepping, any means once an agent dead, then terminate all agents
        # all means terminate environment until all agents dead.
        "terminate": "any",
        # on_policy means collect only newest samples, off_policy will maintain an increasing data buffer
        "mode": "on_policy",
        # default stopper for rollout is simple_rollout, will terminate training by rollout evaluation
        "stopper": "simple_rollout",
        "stopper_config": {},
        # callback specify which manner you wanna use for rollout/simulation, default is sequential
        # feasible choices: sequential, simultaneous, or any registered name of rollout func
        "callback": None,
        "worker_num": -1,
    },
    # for evaluation, if not specified, MALib will use rollout configuration to do evaluation
    "evaluation": {},  #
    # global evaluator controls the learning workflow from the highest level,
    # in multi-agent mode: the global evaluator will determine whether the training/rollout evaluation
    # results achieves to global convergence conditions, i.e. policy converged in reward/loss.
    # in meta-game mode: the global evaluator will determine whether the training/rollout evaluation
    # results achieves to global convergence conditions, i.e. current best response has no positive exploitability
    # if converged, terminate training workflow, if not, generate simulation task, followed by payoff_table_update
    # task, followed by add_policy task.
    "global_evaluator": {"name": "psro", "config": {}},
    # configuration for dataset server
    "dataset_config": {},  #
    "parameter_server": {},
}
# __sphinx_doc_end__
