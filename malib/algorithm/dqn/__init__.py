from .policy import DQN
from .trainer import DQNTrainer
from .loss import DQNLoss


NAME = "DQN"
LOSS = DQNLoss
TRAINER = DQNTrainer
POLICY = DQN

# custom_config
CONFIG = {
    "training": {"tau": 0.01},
    "policy": {
        "gamma": 0.98,
        "eps_min": 1e-2,
        "eps_max": 1.0,
        "eps_decay": 2000,
        "dueling": False,
        "use_cuda": False,
    },
}

__all__ = ["NAME", "LOSS", "TRAINER", "POLICY"]
