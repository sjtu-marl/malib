from typing import Dict, Any, Union

from dataclasses import dataclass, field


@dataclass
class RolloutConfig:
    inference_server_type: str

    @classmethod
    def from_raw(
        cls, config: Union["RolloutConfig", Dict[str, Any]]
    ) -> "RolloutConfig":
        """Cat dict-style configuration to RolloutConfig instance

        Args:
            config (Dict[str, Any]): A dict

        Raises:
            RuntimeError: Unexpected config type

        Returns:
            RolloutConfig: A rollout config instance
        """

        if isinstance(config, cls):
            return config
        elif isinstance(config, Dict):
            return cls(**config)
        else:
            raise RuntimeError(f"Unexpected rollout config type: {type(config)}")
