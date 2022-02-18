from .policy import DQN
from .trainer import DQNTrainer
from .loss import DQNLoss
from malib.utils.preprocessor import Mode


NAME = "DQN"
LOSS = DQNLoss
TRAINER = DQNTrainer
POLICY = DQN

# custom_config
CONFIG = {
    "training": {
        "tau": 0.01,
        "batch_size": 64,
    },
    "policy": {
        "gamma": 0.98,
        "eps_min": 1e-2,
        "eps_max": 1.0,
        "eps_anneal_time": 5000,
        "dueling": False,
        "use_cuda": False,
        "preprocess_mode": Mode.FLATTEN,
    },
}

__all__ = ["NAME", "LOSS", "TRAINER", "POLICY"]
