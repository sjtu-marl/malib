from .policy import BC
from .trainer import BCTrainer
from .loss import BCLoss

NAME = "BC"
LOSS = BCLoss
TRAINER = BCTrainer
POLICY = BC


__all__ = ["NAME", "LOSS", "TRAINER", "POLICY"]
