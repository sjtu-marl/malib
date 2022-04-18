import enum
import time
from collections import namedtuple
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Tuple, Sequence, Callable, Optional

import gym
import numpy as np

from malib.utils.notations import deprecated

""" Rename and definition of basic data types which are correspond to the inputs (args, kwargs) """
PolicyConfig = Dict[str, Any]
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
EnvID = str
EpisodeID = str
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

    @classmethod
    def gen_template(cls, **kwargs):
        return cls(
            time_stamp=time.time(),
            identify=kwargs.get("identify", None),
            id=kwargs["id"],
            lock=kwargs.get("lock", True),
            env_id=kwargs["env_id"],
            type=kwargs.get("type", cls.Type.PARAMETER),
            data=kwargs.get("data", None),
            description=kwargs.get(
                "description",
                {
                    "registered_name": "random",
                    "observation_space": None,
                    "action_space": None,
                    "model_config": {},
                    "custom_config": {},
                },
            ),
        )


@dataclass
class MetaParameterDescription:
    meta_pid: PolicyID
    parameter_desc_dict: Dict[PolicyID, ParameterDescription]
    timestamp: float = time.time()
    identify: str = "MetaParameterDescription"  # meta policy id

    def __post_init__(self):
        self.identify = f"{self.identify}_mpid_{self.meta_pid}_{self.timestamp}"

    @classmethod
    def gen_template(cls, **kwargs):
        return cls(
            meta_pid=kwargs["meta_pid"],
            parameter_desc_dict={
                k: ParameterDescription.gen_template(id=k) for k in kwargs["pids"]
            },
        )


@dataclass
class BufferDescription:
    env_id: str
    agent_id: Union[AgentID, List[AgentID]]
    policy_id: Union[PolicyID, List[PolicyID]]
    batch_size: int = 0
    sample_mode: str = ""
    indices: List[int] = None
    data: Any = None
    data_shapes: Dict[str, Tuple] = None
    sample_start_size: int = 0
    capacity: int = 1000
    identify: str = None

    def __post_init__(self):
        if self.identify is None:
            self.identify = "_".join(sorted(self.agent_id))

    def __str__(self):
        return "<BufferDescription: agent_id={} policy_id={}".format(
            self.agent_id, self.policy_id
        )


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

    @classmethod
    def gen_template(
        cls,
        agent_ids: List[AgentID],
        observation_space: gym.Space,
        action_space: gym.Space,
    ):
        example_ptup = (
            "policy_0",
            {
                "registered_name": "test",
                "observation_space": observation_space,
                "action_space": action_space,
                "mode_config": None,
                "custom_config": None,
            },
        )
        return cls(
            training_handler="test",
            trainable_pairs=dict.fromkeys(agent_ids, example_ptup),
            populations=dict.fromkeys(agent_ids, [example_ptup]),
            env_id="test",
            meta_parameter_desc_dict=dict.fromkeys(
                agent_ids,
                MetaParameterDescription.gen_template(meta_pid=None, pids=["policy_0"]),
            ),
        )


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

    @classmethod
    def gen_template(cls, **template_attr_kwargs):
        raise NotImplementedError


@dataclass
class RolloutDescription:
    agent_involve_info: AgentInvolveInfo
    fragment_length: int
    num_episodes: int
    max_step: int
    callback: Union[str, Callable] = "sequential"
    stopper: str = "none"
    stopper_config: Dict[str, Any] = field(default_factory=dict)
    policy_distribution: Dict[AgentID, Dict[PolicyID, float]] = None
    time_stamp: float = time.time()

    @classmethod
    def gen_template(cls, **template_attr_kwargs):
        agent_involve_info_kwargs = template_attr_kwargs.pop("agent_involve_info")
        instance = cls(
            agent_involve_info=AgentInvolveInfo.gen_template(
                **agent_involve_info_kwargs
            ),
            policy_distribution=dict.fromkeys(
                agent_involve_info_kwargs["agent_ids"], {"policy_0": 1.0}
            ),
            **template_attr_kwargs,
        )
        template_attr_kwargs["agent_involve_info"] = agent_involve_info_kwargs
        return instance


@dataclass
class SimulationDescription:
    agent_involve_info: AgentInvolveInfo
    policy_combinations: List[Dict[AgentID, Tuple[PolicyID, PolicyConfig]]]
    num_episodes: int
    callback: Union[str, Callable] = "sequential"
    max_episode_length: int = None
    time_stamp: float = time.time()

    @classmethod
    def gen_template(cls, **kwargs):
        agent_involve_template_attrs = kwargs.pop("agent_involve_info")
        instance = cls(
            agent_involve_info=AgentInvolveInfo.gen_template(
                **agent_involve_template_attrs
            ),
            **kwargs,
        )
        kwargs["agent_involve_info"] = agent_involve_template_attrs
        return instance


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

    statistics: Dict[str, Any]
    policy_combination: Dict[PolicyID, PolicyID] = None

    def __post_init__(self):
        pass
        # for res in self.statistics.values():
        # for k, v in res.items():
        #     if isinstance(v, MetricEntry):
        #         res[k] = v.value


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

    @classmethod
    def gen_template(cls, **template_attr_kwargs):
        task_type = template_attr_kwargs["task_type"]
        if task_type == TaskType.OPTIMIZE:
            desc_cls = TrainingDescription
        elif task_type == TaskType.ROLLOUT:
            desc_cls = RolloutDescription
        elif task_type == TaskType.SIMULATION:
            desc_cls = SimulationDescription
        else:
            raise ValueError("Unknow task type: {}".format(task_type))
        content_template_attr_kwargs = template_attr_kwargs.pop("content")
        instance = cls(
            content=desc_cls.gen_template(**content_template_attr_kwargs),
            **template_attr_kwargs,
        )
        template_attr_kwargs["content"] = content_template_attr_kwargs
        return instance


@dataclass
class TaskRequest:
    """TaskRequest is a description of"""

    task_type: TaskType
    """defines the requested task type"""

    content: Any
    """content is the feedback of current handler which request for next task"""

    state_id: str

    timestamp: float = None  # time.time()

    identify: str = None

    computing_mode: str = "bulk_sync"  # bulk_sync, async

    def __post_init__(self):
        assert self.state_id, "State id cannot be None"
        timestamp = time.time()
        self.timestamp = timestamp
        self.identify = f"TaskRequest_{timestamp}"

    @staticmethod
    def from_task_desc(task_desc: TaskDescription, **kwargs) -> "TaskRequest":
        return TaskRequest(
            task_type=kwargs.get("task_type", task_desc.task_type),
            content=kwargs.get("content", task_desc.content),
            state_id=kwargs.get("state_id", task_desc.state_id),
            timestamp=kwargs.get("timestamp", None),
            identify=kwargs.get("identify", None),
        )


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


# TODO(jing): add docs for MetricEntry
class MetricEntry:
    def __init__(self, value: Any, agg: str = "mean", tag: str = "", log: bool = True):
        self.value = value
        self.agg = agg
        self.tag = tag
        self.log = log

    def cleaned_data(self):
        """Return values"""


@dataclass
class DataFrame:
    identifier: Any
    data: Any
    runtime_config: Dict[str, Any]
