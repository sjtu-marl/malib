from .policy import SAC
from .advirl_trainer import AdvIRLTrainer

NAME = "AdvIRL"
LOSS = None
TRAINER = AdvIRLTrainer
POLICY = None


__all__ = ["NAME", "LOSS", "TRAINER", "POLICY"]
