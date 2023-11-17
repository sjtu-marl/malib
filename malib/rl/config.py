from typing import Dict, Any, Type

from dataclasses import dataclass

from malib.rl.common.policy import Policy
from malib.rl.common.trainer import Trainer


@dataclass
class Algorithm:

    policy: Type[Policy]

    trainer: Type[Trainer]

    model_config: Dict[str, Any]

    trainer_config: Dict[str, Any]
