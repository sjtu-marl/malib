from .policy import SAC
from .trainer import SACTrainer
from .loss import SACLoss


NAME = "SAC"
LOSS = SACLoss
TRAINER = SACTrainer
POLICY = SAC


TRAINING_CONFIG = {
    "update_interval": 1,
    "batch_size": 1024,
    "tau": 0.01,
    "optimizer": "Adam",
    "actor_lr": 1e-2,
    "critic_lr": 1e-2,
    "grad_norm_clipping": 0.5,
}
