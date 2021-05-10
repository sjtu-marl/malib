from .policy import BehaviorCloning
from .trainer import BCTrainer
from .loss import BCLoss


NAME = "BC"
LOSS = BCLoss
TRAINER = BCTrainer
POLICY = BehaviorCloning


__all__ = ["NAME", "LOSS", "TRAINER", "POLICY"]
