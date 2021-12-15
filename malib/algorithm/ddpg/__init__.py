from .policy import DDPG
from .trainer import DDPGTrainer
from .loss import DDPGLoss


NAME = "DDPG"
LOSS = DDPGLoss
TRAINER = DDPGTrainer
POLICY = DDPG


CONFIG = {
    "training": {
        "update_interval": 1,
        "batch_size": 1024,
        "tau": 0.01,
        "optimizer": "Adam",
        "actor_lr": 1e-2,
        "critic_lr": 1e-2,
        "grad_norm_clipping": 0.5,
    },
    "policy": {},
}
