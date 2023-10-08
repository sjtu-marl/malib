from typing import Dict, Any, Union

from dataclasses import dataclass, field


@dataclass
class TrainingConfig:

    trainer_config: Dict[str, Any]
    learner_type: str
    custom_config: Dict[str, Any] = field(default_factory=dict())

    @classmethod
    def from_raw(cls, config: Union["TrainingConfig", Dict[str, Any]]) -> "TrainingConfig":
        """Cat dict-style configuration to TrainingConfig instance

        Args:
            config (Dict[str, Any]): A dict

        Raises:
            RuntimeError: Unexpected config type

        Returns:
            TrainingConfig: A training config instance
        """

        if isinstance(config, Dict):
            return cls(**config)
        elif isinstance(config, cls):
            return config
        else:
            raise RuntimeError(f"Unexpected training config type: {type(config)}")
