from .policy import MAPPO
from .trainer import MAPPOTrainer
from .loss import MAPPOLoss

NAME = "MAPPO"
TRAINER = MAPPOTrainer
POLICY = MAPPO
LOSS = None

__all__ = ["NAME", "TRAINER", "POLICY"]

CONFIG = {
    'training': {
        'critic_lr': 3e-4,
        'actor_lr': 3e-4,
        'opti_eps': 1.e-5,
        'weight_decay': 0.0,
        'batch_size': 64
    },
    'policy': {
        'gamma': 0.99,
        'use_q_head': False,
        'ppo_epoch': 4,
        'num_mini_batch': 1,
        'return_mode': 'gae',
        'gae': {'gae_lambda': 0.95},
        'vtrace': {
            'clip_rho_threshold': 1.0,
            'clip_pg_threshold': 1.0
        },
        # this is not used, instead it is fixed to last hidden in actor/critic
        'use_rnn': False,
        'rnn_layer_num': 1,
        'rnn_data_chunk_length': 16,
        
        'use_feature_normalization': True,
        'use_popart': True,
        'popart_beta': 0.99999,

        'entropy_coef': 1.e-2
    }
}