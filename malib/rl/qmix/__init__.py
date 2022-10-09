# -*- coding: utf-8 -*-
from .trainer import QMIXTrainer
from .loss import QMIXLoss
from malib.rl.dqn.policy import DQN


NAME = "QMIX"
LOSS = QMIXLoss
TRAINER = QMIXTrainer
POLICY = DQN


CONFIG = {
    "training": {
        "update_interval": 1,
        "batch_size": 1024,
        "optimizer": "Adam",
        "lr": 0.0005,
        "tau": 0.01,  # soft update
    },
    "policy": {
        "eps_min": 0.05,
        "eps_max": 1.0,
        "eps_anneal_time": 50000,
        "gamma": 0.99,
        "use_cuda": False,  # enable cuda or not
    },
}
