from abc import ABC, abstractmethod
from typing import Callable, Union
from copy import deepcopy


class Scenario(ABC):
    @abstractmethod
    def __init__(self, name: str):
        self.name = name
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
        ray_should_log_result_filter: Callable[[ResultDict], bool],
    ):
        super().__init__(name=name)
        self.ray_cluster_cpus = ray_cluster_cpus
        self.ray_cluster_gpus = ray_cluster_gpus
        self.ray_object_store_memory_cap_gigabytes = (
            ray_object_store_memory_cap_gigabytes
        )

        self.ray_should_log_result_filter = ray_should_log_result_filter
