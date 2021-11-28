from .policy import MAPPO
from .trainer import MAPPOTrainer
from .loss import MAPPOLoss

NAME = "MAPPO"
TRAINER = MAPPOTrainer
POLICY = MAPPO
LOSS = None

__all__ = ["NAME", "TRAINER", "POLICY"]
