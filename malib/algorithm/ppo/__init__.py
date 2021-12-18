from .policy import PPO
from .trainer import PPOTrainer
from .loss import PPOLoss


NAME = "PPO"
LOSS = PPOLoss
TRAINER = PPOTrainer
POLICY = PPO


__all__ = ["NAME", "LOSS", "TRAINER", "POLICY"]

CONFIG = {
    "training": {
        "saving_interval": 10,
        "batch_size": 1024,
        "optimizer": "Adam",
        "lr": 0.001,
        "ppo_epoch": 1,
        "entropy_coef": 0.02,
        "grad_norm_clipping": 0.5,
    },
    "policy": {"gamma": 0.95, "use_cuda": False},
}
