from .policy import PPO
from .ppo_trainer import PPOTrainer
from .loss import PPOLoss


NAME = "PPO"
LOSS = PPOLoss
TRAINER = PPOTrainer
POLICY = PPO


__all__ = ["NAME", "LOSS", "TRAINER", "POLICY"]
