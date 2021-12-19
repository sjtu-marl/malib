import torch

from malib.algorithm.common.trainer import Trainer
from malib.algorithm.common.policy import Policy

from malib.algorithm.common import misc
from malib.algorithm.qmix.q_mixer import QMixer
from malib.algorithm.qmix.loss import QMIXLoss
from malib.utils.preprocessor import Preprocessor, get_preprocessor

from malib.utils.typing import AgentID, Dict


class QMIXTrainer(Trainer):
    def __init__(self, tid):
        super(QMIXTrainer, self).__init__(tid)
        self._loss = QMIXLoss()
        self.global_state_preprocessor: Preprocessor = None

    def optimize(self, batch):
        self.loss.zero_grad()
        self.loss.agents = self.agents
        loss_stat = self.loss(batch)
        _ = self.loss.step()
        return loss_stat

    def reset(self, policy, training_config):
        global_state_space = policy.custom_config["global_state_space"]
        if self.loss.mixer is None:
            self.global_state_preprocessor = get_preprocessor(global_state_space)(
                global_state_space
            )
            self.loss.set_mixer(
                QMixer(self.global_state_preprocessor.size, len(self.agents))
            )
            self.loss.set_mixer_target(
                QMixer(self.global_state_preprocessor.size, len(self.agents))
            )

            # sync mixer
            with torch.no_grad():
                misc.hard_update(self.loss.mixer_target, self.loss.mixer)

        super(QMIXTrainer, self).reset(policy, training_config)

    def preprocess(self, batch, **kwargs):
        return batch
