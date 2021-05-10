from .policy import SAC
from .sac_trainer import SACTrainer

NAME = "SAC"
LOSS = None
TRAINER = SACTrainer
POLICY = SAC


__all__ = ["NAME", "LOSS", "TRAINER", "POLICY"]
