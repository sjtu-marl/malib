from typing import Dict

import torch

from malib.rl.common.trainer import Trainer
from malib.rl.common.policy import Policy

from malib.rl.common import misc
from malib.rl.qmix.q_mixer import QMixer
from malib.utils.preprocessor import Preprocessor, get_preprocessor

from malib.utils.typing import AgentID


class QMIXTrainer(Trainer):
    pass
