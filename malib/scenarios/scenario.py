from abc import ABC, abstractmethod
from types import LambdaType
from typing import Callable, Union, Dict, Any
from copy import deepcopy


DEFAULT_STOPPING_CONDITIONS = {}


class Scenario(ABC):
    @abstractmethod
    def __init__(
        self,
        name: str,
        log_dir: str,
        env_desc: Dict[str, Any],
        algorithms: Dict[str, Any],
        agent_mapping_func: LambdaType,
        training_config: Dict[str, Any],
        rollout_config: Dict[str, Any],
        stopping_conditions: Dict[str, Any],
        dataset_config: Dict[str, Any],
        parameter_server_config: Dict[str, Any],
    ):
        self.name = name
        self.log_dir = log_dir
        self.env_desc = env_desc
        self.algorithms = algorithms
        self.agent_mapping_func = agent_mapping_func
        self.training_config = training_config
        self.rollout_config = rollout_config
        self.stopping_conditions = stopping_conditions or DEFAULT_STOPPING_CONDITIONS
        self.dataset_config = dataset_config or {"table_capacity": 1000}
        self.parameter_server_config = parameter_server_config or {}
        self.validate_properties()

    def validate_properties(self):
        # validate name
        for c in self.name:
            if c.isspace() or c in ["\\", "/"] or not c.isprintable():
                raise ValueError(
                    "Scenario names must not contain whitespace, '\\', or '/'. "
                    "It needs to be usable as a directory name. "
                    f"Yours was '{self.name}'."
                )

    def copy(self):
        return deepcopy(self)

    def with_updates(self, **kwargs) -> "Scenario":
        new_copy = self.copy()
        for k, v in kwargs.items():
            if not hasattr(new_copy, k):
                raise KeyError(f"{k} is not an attribute of {new_copy.__class__}")
            setattr(new_copy, k, v)
        return new_copy


class RayScenario(Scenario, ABC):
    @abstractmethod
    def __init__(
        self,
        name: str,
        ray_cluster_cpus: Union[int, float],
        ray_cluster_gpus: Union[int, float],
        ray_object_store_memory_cap_gigabytes: Union[int, float],
        ray_should_log_result_filter: Callable[[Dict], bool],
    ):
        super().__init__(name=name)
        self.ray_cluster_cpus = ray_cluster_cpus
        self.ray_cluster_gpus = ray_cluster_gpus
        self.ray_object_store_memory_cap_gigabytes = (
            ray_object_store_memory_cap_gigabytes
        )

        self.ray_should_log_result_filter = ray_should_log_result_filter
