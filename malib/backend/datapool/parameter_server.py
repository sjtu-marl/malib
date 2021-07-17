import os
import threading

from dataclasses import dataclass
from collections import namedtuple

import pickle5 as pickle
import ray

from malib import settings
from malib.utils.logger import get_logger
from malib.utils import errors
from malib.algorithm.common.misc import GradientOps
from malib.utils.typing import (
    Status,
    ParameterDescription,
    Parameter,
    Dict,
    Tuple,
    List,
    AgentID,
    PolicyID,
)

PARAMETER_TABLE_NAME_GEN = (
    lambda env_id, agent_id, pid, policy_type: f"{env_id}_{agent_id}_{pid}_{policy_type}"
)
DEFAULT_TABLE_SIZE = 2
TableStatus = namedtuple("TableStatus", "locked, gradient_status")

TableStatus.__doc__ = """\
Name tuple to define the table status with two sub-status: `locked`, `gradient_status`. `locked` has a higher priority than
`gradient_status`.
"""

TableStatus.locked.__doc__ = """\
bool - Indicates whether parameters in some table is locked or not. If locked, parameters cannot be updated anymore. Any
update request will have no any response.
"""

TableStatus.gradient_status.__doc__ = """\
Status - Indicates the gradient status, could be `Status.WAITING`, `Status.NORMAL` (default)

- Status.WAITING: waiting for gradients upload.
- Status.NORMAL: all gradients have been applied to parameters, remote actor can pull them down.
"""


class Forbidden(Exception):
    pass


@dataclass
class Table:
    """Parameter holder for one policy for an agent in an environment. Currently, `Table` holds only the latest parameter
    version.
    """

    env_id: str
    """Environment id."""

    agent_id: AgentID
    """Environment agent id."""

    policy_id: str
    """Policy id."""

    policy_type: str
    """Policy type name."""

    gradients: List = None
    """A list of stacked gradients, for update parameters."""

    parallel_num: int = 1
    """Indicates how many copies/parameter learners share this table."""

    locked: bool = False
    """Indicates the parameter is locked or not."""

    version: int = 0
    """Indicates the parameter version, will increase after executing parameter update."""

    gradient_status: Status = Status.NORMAL
    """Flag the gradient list, legal status could be NORMAL, WAITING and LOCKED """

    @property
    def status(self) -> TableStatus:
        """Return the table status.

        :return: A `TableStatus` entity.
        """
        with self._threading_lock:
            return TableStatus(self.locked, self.gradient_status)

    def __post_init__(self):
        self._data: Dict[str, Parameter] = dict()
        self.gradients = []
        self._threading_lock = threading.Lock()

    @staticmethod
    def parameter_hash_key(parameter_desc: ParameterDescription) -> str:
        """Generate a hash key.

        :param ParameterDescription parameter_desc: A parameter description entity.
        :return: string key.
        """

        key = str(parameter_desc.id)
        return key

    @property
    def name(self) -> str:
        """Return table name.

        :return: string Table name.
        """

        return PARAMETER_TABLE_NAME_GEN(
            env_id=self.env_id,
            agent_id=self.agent_id,
            pid=self.policy_id,
            policy_type=self.policy_type,
        )

    def _aggregate(self, parameter_desc: ParameterDescription):
        """Aggregate gradients for policy parameters.

        :param ParameterDescription parameter_desc: An parameter description entity
        """

        if len(self.gradients) == self.parallel_num:
            # concatenate
            # gradients = list(itertools.chain(*self.gradients))
            # parse and merge
            if not self.locked:
                gradients = GradientOps.mean(
                    [GradientOps.sum(grad) for grad in self.gradients]
                )
                # convert gradients to tensor then apply to parameter
                key = Table.parameter_hash_key(parameter_desc)
                self._data[key] = GradientOps.add(self._data[key], gradients)
                # debug_tools.check_nan(key, self._data[key])
                self.version += 1
                # then switch status to NORMAL which means can pull
                self.gradients = []
                self.gradient_status = Status.NORMAL
            else:
                self.gradients = []
                self.gradient_status = Status.LOCKED
        else:
            pass

    def get(self, parameter_desc: ParameterDescription) -> Parameter:
        """Always return the newest parameters.

        :param ParameterDescription parameter_desc: A parameter description entity.
        :return: A parameter or None (if parameter not ready)
        """

        with self._threading_lock:
            key = Table.parameter_hash_key(parameter_desc)
            if self.gradient_status == Status.NORMAL:
                return self._data[key]
            else:
                return None

    def insert(self, parameter_desc: ParameterDescription) -> None:
        """Insert/update parameters/gradients. If parameters have not been locked yet, insertion will be successful.

        :param ParameterDescription parameter_desc: A parameter description entity.
        :return: None
        """

        with self._threading_lock:
            k = Table.parameter_hash_key(parameter_desc)
            if parameter_desc.type == "parameter":
                if self.locked:
                    return
                else:
                    self.version += 1
                    data = parameter_desc.data

                    self._data[k] = data
                    self.locked = parameter_desc.lock
            elif parameter_desc.type == "gradient":
                if self.locked:
                    # clear gradients
                    self.gradients = []
                    return
                if len(self.gradients) < self.parallel_num:
                    self.gradient_status = Status.WAITING
                    self.gradients.append(parameter_desc.data)
                    # try to do aggregation
                    self._aggregate(parameter_desc)
                else:
                    raise IndexError("reached maximum")

    @staticmethod
    def _save_helper_func(obj, fp, candidate_name=""):
        if os.path.isdir(fp):
            tfp = os.path.join(fp, candidate_name + ".pkl")
        else:
            paths = os.path.split(fp)[0]
            try:
                os.makedirs(paths)
            except:
                pass
            tfp = fp + ".pkl"
        with open(tfp, "wb") as f:
            pickle.dump(obj, f, protocol=settings.PICKLE_PROTOCOL_VER)

    def dump(self, fp):
        with self._threading_lock:
            serial_dict = {
                "meta": {
                    "env_id": self.env_id,
                    "agent_id": self.agent_id,
                    "policy_id": self.policy_id,
                    "policy_type": self.policy_type,
                    "parallel_num": self.parallel_num
                },
                "data": self._data
            }
            self._save_helper_func(serial_dict, fp, self.name)

    @staticmethod
    def load(cls, fp):
        with open(fp, "rb") as f:
            serial_dict = pickle.load(f)
            table = Table(**serial_dict["meta"])
            table._data = serial_dict.get("_data")
            return table


@ray.remote
class ParameterServer:
    """Parameter lib can be initialized with existing database or create a new one"""

    def __init__(self, **kwargs):
        self._table: Dict[str, Table] = dict()
        self._table_status: Dict[str, Status] = dict()

        self.logger = get_logger(
            log_level=settings.LOG_LEVEL,
            log_dir=settings.LOG_DIR,
            name="parameter_server",
            remote=settings.USE_REMOTE_LOGGER,
            mongo=settings.USE_MONGO_LOGGER,
            **kwargs["exp_cfg"],
        )
        self._threading_lock = threading.Lock()

        init_job_config = kwargs.get("init_job", {})
        if init_job_config.get("load_when_start"):
            self.load(init_job_config.get("path"))

        quit_job_config = kwargs.get("quit_job", {})
        self.dump_when_closed = quit_job_config.get("dump_when_closed")
        self.dump_path = quit_job_config.get("path")

    def check_ready(self, parameter_desc):
        table_name = PARAMETER_TABLE_NAME_GEN(
            env_id=parameter_desc.env_id,
            agent_id=parameter_desc.identify,
            pid=parameter_desc.id,
            policy_type=parameter_desc.description["registered_name"],
        )
        assert (
            self._table.get(table_name, None) is not None
        ), f"No such a table named={table_name}, {list(self._table.keys())}"

        return self._table[table_name].gradient_status == Status.NORMAL

    def pull(
        self, parameter_desc: ParameterDescription, keep_return=False
    ) -> Tuple[TableStatus, ParameterDescription]:
        """Pull parameter from table

        :parameter parameter_desc: ParameterDescription, parameter description
        :parameter keep_return: bool, determine whether to force return (parameter)
        """
        try:
            table_name = PARAMETER_TABLE_NAME_GEN(
                env_id=parameter_desc.env_id,
                agent_id=parameter_desc.identify,
                pid=parameter_desc.id,
                policy_type=parameter_desc.description["registered_name"],
            )
        except Exception as e:
            parameter_desc.data = None
        table = self._table.get(table_name, None)
        assert (
            table is not None
        ), f"No such a table named={table_name}, {list(self._table.keys())}"
        if table.parallel_num != parameter_desc.parallel_num:
            # (hanjing): Fix the possible conflicts when recovering from dumped files
            self.logger.info("Inconsistence found in parallel num, reassigned")
            table.parallel_num = parameter_desc.parallel_num

        parameter = self._table[table_name].get(parameter_desc)
        status = self._table[table_name].status
        if self._table[table_name].version <= parameter_desc.version:
            parameter_desc.data = None
        else:
            parameter_desc.version = self._table[table_name].version
            parameter_desc.data = parameter
        return status, parameter_desc

    def push(self, parameter_desc: ParameterDescription) -> TableStatus:
        """Push parameters or gradients to table.

        :param parameter_desc: ParameterDescription, parameter description entity
        :return TableStatus entity
        """

        table_name = PARAMETER_TABLE_NAME_GEN(
            env_id=parameter_desc.env_id,
            agent_id=parameter_desc.identify,
            pid=parameter_desc.id,
            policy_type=parameter_desc.description["registered_name"],
        )

        table = self._table.get(table_name, None)
        if table is None:
            with self._threading_lock:
                self._table[table_name] = Table(
                    env_id=parameter_desc.env_id,
                    agent_id=parameter_desc.identify,
                    policy_id=parameter_desc.id,
                    policy_type=parameter_desc.description["registered_name"],
                    parallel_num=parameter_desc.parallel_num,
                )
        else:
            # (hanjing): Check for consistence
            assert (table.parallel_num == parameter_desc.parallel_num)
        self._table[table_name].insert(parameter_desc)
        status = self._table[table_name].status
        return status

    def dump(self, file_path=None):
        """ Export parameters to local storage """
        # (hanjing): The original implementation directly serialize the table,
        # however, which contains a threading lock that can not be serialized.
        # Now reorganize the table when dumping, each table is dumped as a separated
        # file. Note that consider possible changes in the  parallel num, we will trust
        # the parallel num recovered from the serialized files. Hence it will be checked and
        # modified during each pull request if there is a conflict with parameter_desc.parallel_num
        with self._threading_lock:
            file_path = file_path or settings.PARAM_DIR

            f = open("ps.log", "w")
            dumped_list = []
            for tn in self._table.keys():
                dumped_list.append(tn)
                f.write(f"Trying to dump table {tn}, locked {self._table[tn].locked}\n")
                self._table[tn].dump(file_path)
            f.close()
        return dumped_list

    def load(self, file_path=None, protocol=None):
        """ Load parameters from local storage """

        with self._threading_lock:
            protocol = protocol or settings.PICKLE_PROTOCOL_VER
            file_path = file_path or settings.PARAM_DIR

            for fn in os.listdir(file_path):
                if fn.endswith(".pkl"):
                    try:
                        table = Table.load(os.path.join(file_path, fn))
                        self._table[table.name] = table
                    except Exception as e:
                        self.logger.error(f"Loading {fn} failed ({e})")

    def shutdown(self):
        result = None
        if self.dump_when_closed:
            result = self.dump(self.dump_path)
        self.logger.info(f"Parameter server closed.")
        return result
