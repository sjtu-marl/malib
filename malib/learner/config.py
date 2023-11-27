from typing import Dict, Any, Union, Type, Callable

from dataclasses import dataclass, field

from malib.learner.learner import Learner
from malib.backend.dataset_server.feature import BaseFeature


# TODO(ming): rename it as LearnerConfig
@dataclass
class LearnerConfig:
    learner_type: Type[Learner]
    feature_handler_meta_gen: Callable[["EnvDesc", str], Callable[[str], BaseFeature]]
    """what is it?"""

    custom_config: Dict[str, Any] = field(default_factory=dict())

    @classmethod
    def from_raw(
        cls, config: Union["LearnerConfig", Dict[str, Any]]
    ) -> "LearnerConfig":
        """Cat dict-style configuration to LearnerConfig instance

        Args:
            config (Dict[str, Any]): A dict

        Raises:
            RuntimeError: Unexpected config type

        Returns:
            LearnerConfig: A training config instance
        """

        if isinstance(config, Dict):
            return cls(**config)
        elif isinstance(config, cls):
            return config
        else:
            raise RuntimeError(f"Unexpected training config type: {type(config)}")
