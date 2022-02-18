from malib.algorithm.ddpg.policy import DDPG
from .trainer import MADDPGTrainer
from .loss import MADDPGLoss


NAME = "MADDPG"
LOSS = None  # PPOLoss
TRAINER = MADDPGTrainer
POLICY = DDPG


__all__ = ["NAME", "LOSS", "TRAINER", "POLICY"]


CONFIG = {
    "training": {
        "update_interval": 1,
        "saving_interval": 10,
        "batch_size": 1024,
        "optimizer": "Adam",
        "actor_lr": 0.01,
        "critic_lr": 0.01,
        "lr": 0.01,
        "tau": 0.01,
        "grad_norm_clipping": 0.5,
    },
    "policy": {
        "gamma": 0.95,
        "use_cuda": False,
    },
}
