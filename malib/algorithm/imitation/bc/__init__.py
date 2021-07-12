from .policy import BC
from .bc_trainer import BCTrainer

NAME = "BC"
LOSS = None
TRAINER = BCTrainer
POLICY = BC


__all__ = ["NAME", "LOSS", "TRAINER", "POLICY"]
