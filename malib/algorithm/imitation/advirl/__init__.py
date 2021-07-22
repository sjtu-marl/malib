from .reward import AdvIRLReward
from .trainer import AdvIRLTrainer
from .loss import AdvIRLLoss

NAME = "ADVIRL"
LOSS = AdvIRLLoss
TRAINER = AdvIRLTrainer
REWARD = AdvIRLReward


__all__ = ["NAME", "LOSS", "TRAINER", "REWARD"]
