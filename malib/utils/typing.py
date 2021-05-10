import enum
import time
from collections import namedtuple
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Tuple, Sequence, Callable

import gym
import numpy as np

from malib.utils.notations import deprecated

""" Rename and definition of basic data types which are correspond to the inputs (args, kwargs) """
PolicyConfig = Tuple[str, Dict[str, Any]]
MetaPolicyConfig = Tuple[gym.spaces.Space, gym.spaces.Space, Sequence[PolicyConfig]]
EnvConfig = Dict[str, Any]
RolloutConfig = Dict[str, Any]
ParameterLibConfig = Dict[str, Any]
DatasetConfig = Dict[str, Any]
TrainingConfig = Dict[str, Any]
ModelConfig = Dict[str, Any]
AgentConfig = Dict[str, TrainingConfig]

AgentID = str

PolicyID = str
EnvID = int
EpisodeID = int
DataBlockID = str

DataTransferType = np.ndarray
EnvObservationType = Any

# next_observation, rewards, done, infos
StandardEnvReturns = Tuple[
    Dict[str, DataTransferType],
    Dict[str, float],
    Dict[str, bool],
    Dict[str, Any],
]

# TODO(ming): mute info temporally to avoid data transferring errors
StandardTransition = namedtuple(
    # "StandardTransition", "obs, new_obs, action, reward, done, info"
    "StandardTransition",
    "obs, new_obs, actions, rewards, dones",
)

ObservationSpaceType = gym.spaces.Space
ActionSpaceType = gym.spaces.Space


""" For task categorical and status tagging """


class TaskType(enum.Enum):
    ASYNC_LEARNING = "async_learning"
    ADD_WORKER = "add_worker"
    SAVE_MODEL = "save_model"
    LOAD_MODEL = "load_model"
    OPTIMIZE = "optimization"
    ROLLOUT = "rollout"
    UPDATE_PARAMETER = "update_PARAMETER"
    PULL_PARAMETER = "pull_parameter"
    PUSH_PARAMETER = "push_parameter"
    SAMPLE_BATCH = "sample_batch"
    PUSH_SAMPLES = "push_samples"
    NO = "no"
    TRAINING_EVALUATE = "evaluate_for_training"
    ROLLOUT_EVALUATE = "evaluate_for_rollouts"
    ADD_POLICY = "add_policy"
    UPDATE_POPULATION = "update_population"
    EVALUATE = "evaluate"
    EVALUATE_WRITE_BACK = "evaluate_write_back"
    INIT = "initialization"
    CHECK_ADD = "check_add"
    TERMINATE = "terminate"
    SIMULATION = "simulation"
    UPDATE_PAYOFFTABLE = "update_payofftable"


class Status(enum.Enum):
    TERMINATE = "terminate"
    NORMAL = "normal"
    LOCKED = "locked"
    WAITING = "waiting"
    SUCCESS = "success"
    IDLE = "idle"
    IN_PROGRESS = "in progress"
    EXCEED = "exceed"
    FAILED = "failed"


class Paradigm(enum.Enum):
    MARL = "marl"
    META_GAME = "meta_game"


class BehaviorMode(enum.IntEnum):
    """Behavior mode, indicates environment agent behavior"""

    EXPLORATION = 0
    """Trigger exploration mode"""

    EXPLOITATION = 1
    """Trigger exploitation mode"""


class MetricType:
    REWARD = "reward"
    """Reward"""

    LIVE_STEP = "live_step"
    """Agent live step"""

    REACH_MAX_STEP = "reach_max_step"
    """Whether reach max step or not"""


Parameter = Any

""" Description: """


@dataclass
class ParameterDescription:
    class Type:
        PARAMETER = "parameter"
        GRADIENT = "gradient"

    time_stamp: float
    identify: str  # meta policy id
    env_id: str
    id: PolicyID
    type: str = Type.PARAMETER
    lock: bool = False
    description: Any = None
    data: Parameter = None
    parallel_num: int = 1
    version: int = -1


@dataclass
class MetaParameterDescription:
    meta_pid: PolicyID
    parameter_desc_dict: Dict[PolicyID, ParameterDescription]
    timestamp: float = time.time()
    identify: str = "MetaParameterDescription"  # meta policy id

    def __post_init__(self):
        self.identify = f"{self.identify}_mpid_{self.meta_pid}_{self.timestamp}"


@dataclass
class BufferDescription:
    env_id: str
    agent_id: AgentID
    policy_id: PolicyID
    batch_size: int = 0
    sample_mode: str = ""


@dataclass
class AgentInvolveInfo:
    """`AgentInvolveInfo` describes the trainable pairs, populations, environment id and the
    meta parameter descriptions.
    """

    training_handler: str
    trainable_pairs: Dict[AgentID, Tuple[PolicyID, PolicyConfig]]
    """ describe the environment agent id and their binding policy configuration """

    populations: Dict[AgentID, Sequence[Tuple[PolicyID, PolicyConfig]]]
    """ describe the policy population of agents """

    env_id: str = None
    """ environment id """

    meta_parameter_desc_dict: Dict[AgentID, MetaParameterDescription] = None
    """ meta parameter description """


@dataclass
class TrainingDescription:
    agent_involve_info: AgentInvolveInfo
    stopper: str = "none"
    stopper_config: Dict[str, Any] = field(default_factory=dict)
    policy_distribution: Dict[AgentID, Dict[PolicyID, float]] = None
    update_interval: int = 1
    batch_size: int = 64
    mode: str = "step"
    time_stamp: float = time.time()


@dataclass
class RolloutDescription:
    agent_involve_info: AgentInvolveInfo
    fragment_length: int
    num_episodes: int
    episode_seg: int
    terminate_mode: str
    mode: str  # on_policy or off_policy or imitation learning ?
    # parameter_desc_seq: Sequence[MetaParameterDescription] = None
    callback: Union[str, Callable] = "sequential"
    stopper: str = "none"
    stopper_config: Dict[str, Any] = field(default_factory=dict)
    policy_distribution: Dict[AgentID, Dict[PolicyID, float]] = None
    time_stamp: float = time.time()


@dataclass
class SimulationDescription:
    agent_involve_info: AgentInvolveInfo
    policy_combinations: List[Dict[AgentID, Tuple[PolicyID, PolicyConfig]]]
    num_episodes: int
    callback: Union[str, Callable] = "sequential"
    max_episode_length: int = None
    time_stamp: float = time.time()


@dataclass
class TrainingFeedback:
    agent_involve_info: AgentInvolveInfo
    statistics: Dict[AgentID, Any]


@dataclass
class RolloutFeedback:
    """RolloutFeedback for rollout tasks"""

    worker_idx: str
    """id of rollout worker"""

    agent_involve_info: AgentInvolveInfo
    """agent involve info describes the ..."""

    statistics: Dict[AgentID, Dict[str, Any]]
    policy_combination: Dict[PolicyID, Tuple[PolicyID, PolicyConfig]] = None


@deprecated
@dataclass
class EvaluationFeedback:
    # env_id: str
    agent_involve_info: AgentInvolveInfo
    statistics: Dict[PolicyID, Dict[str, Any]]
    policy_combination: Dict[PolicyID, Tuple[PolicyID, PolicyConfig]]


@dataclass
class TaskDescription:
    """TaskDescription is a general description of
    Training, Rollout and Simulation tasks.
    """

    task_type: TaskType
    """task type used to identify which task description will be used"""

    content: Union[TrainingDescription, RolloutDescription, SimulationDescription]
    """content is a detailed task description entity"""

    state_id: Any

    timestamp: float = None
    source_task_id: str = None
    identify: str = None

    def __post_init__(self):
        timestamp = time.time()
        self.timestamp = timestamp

        if self.task_type == TaskType.OPTIMIZE:
            prefix = "TrainingDescription"
        elif self.task_type == TaskType.ROLLOUT:
            prefix = "RolloutDescription"
        elif self.task_type == TaskType.SIMULATION:
            prefix = "SimulationDescription"
        else:
            prefix = "UnknowDescription"

        self.identify = f"{prefix}_{timestamp}"


@dataclass
class TaskRequest:
    """TaskRequest is a description of"""

    task_type: TaskType
    """defines the requested task type"""

    content: Any
    """content is the feedback of current handler which request for next task"""

    timestamp: float = None  # time.time()

    identify: str = None

    def __post_init__(self):
        timestamp = time.time()
        self.timestamp = timestamp
        self.identify = f"TaskRequest_{timestamp}"


class BColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class EvaluateResult:
    CONVERGED = "converged"
    AVE_REWARD = "average_reward"
    REACHED_MAX_ITERATION = "reached_max_iteration"

    @staticmethod
    def default_result():
        return {
            EvaluateResult.CONVERGED: False,
            EvaluateResult.AVE_REWARD: -float("inf"),
            EvaluateResult.REACHED_MAX_ITERATION: False,
        }


class TrainingMetric:
    LOSS = "loss"


@dataclass
class BatchMetaInfo:
    episode_id: str
    created_time: float
    meta_policy_id: str = None
    policy_id: str = None
    env_id: Any = None
    policy_type: Any = None


class ExperimentManagerTableName:
    primary: str = ""
    secondary: str = ""
    tag: str = ""
    key: int = 0
    nid: int = 0


class EventReportStatus:
    START = "start"
    END = "end"


class MetricEntry:
    def __init__(self, value: Any, agg: str = "mean", tag: str = "", log: bool = True):
        self.value = value
        self.agg = agg
        self.tag = tag
        self.log = log
