from .policy import DiscreteSAC
from .trainer import DiscreteSACTrainer
from .loss import DiscreteSACLoss


NAME = "DiscreteSAC"
LOSS = DiscreteSACLoss
TRAINER = DiscreteSACTrainer
POLICY = DiscreteSAC


CONFIG = {
    "training": {
        "update_interval": 1,
        "batch_size": 1024,
        "tau": 0.01,
        "optimizer": "Adam",
        "actor_lr": 1e-2,
        "critic_lr": 1e-2,
        "grad_norm_clipping": 0.5,
        "sac_alpha": 0.2,
    },
    "policy": {},
}
