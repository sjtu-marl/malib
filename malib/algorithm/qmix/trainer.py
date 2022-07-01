from typing import Dict

import torch

from malib.algorithm.common.trainer import Trainer
from malib.algorithm.common.policy import Policy

from malib.algorithm.common import misc
from malib.algorithm.qmix.q_mixer import QMixer
from malib.utils.preprocessor import Preprocessor, get_preprocessor

from malib.utils.typing import AgentID


class QMIXTrainer(Trainer):
    pass
