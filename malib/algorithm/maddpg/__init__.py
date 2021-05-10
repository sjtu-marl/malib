from malib.algorithm.ddpg.policy import DDPG
from .trainer import MADDPGTrainer
from .loss import MADDPGLoss


NAME = "MADDPG"
LOSS = None  # PPOLoss
TRAINER = MADDPGTrainer
POLICY = DDPG


__all__ = ["NAME", "LOSS", "TRAINER", "POLICY"]
