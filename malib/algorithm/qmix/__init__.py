# -*- coding: utf-8 -*-
from .trainer import QMIXTrainer
from malib.algorithm.dqn.policy import DQN


NAME = "QMIX"
TRAINER = QMIXTrainer
POLICY = DQN
