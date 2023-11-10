from typing import Dict, Any

from dataclasses import dataclass

from malib.rl.common.policy import Policy
from malib.rl.common.trainer import Trainer


@dataclass
class Algorithm:

    policy: Policy

    trainer: Trainer

    model_config: Dict[str, Any]
