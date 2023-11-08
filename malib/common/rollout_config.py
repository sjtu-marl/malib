from typing import Dict, Any, Union

from dataclasses import dataclass, field


@dataclass
class RolloutConfig:
    inference_server_type: str
    """Inference server type"""

    num_workers: int = 1
    """Defines how many workers will be used for executing one rollout task, default is 1"""

    n_envs_per_worker: int = 1
    """Indicates how many environments will be activated for a rollout task per rollout worker, default is 1"""

    timelimit: int = 256
    """Specifying how many time steps will be collected for each rollout, default is 256"""

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
