# -*- coding: utf-8 -*-
from .trainer import QMIXTrainer
from .loss import QMIXLoss
from malib.algorithm.dqn.policy import DQN


NAME = "QMIX"
LOSS = QMIXLoss
TRAINER = QMIXTrainer
POLICY = DQN
