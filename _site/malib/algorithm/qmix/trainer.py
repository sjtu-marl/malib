import numpy as np

from malib.algorithm.common.trainer import Trainer
from malib.algorithm.common.policy import Policy

from malib.algorithm.common import misc
from malib.algorithm.common.model import QMixer
from malib.algorithm.qmix.loss import QMIXLoss
from malib.backend.datapool.offline_dataset_server import Episode
from malib.utils.preprocessor import Preprocessor, get_preprocessor

from malib.utils.typing import AgentID, Dict


class QMIXTrainer(Trainer):
    def __init__(self, tid):
        super(QMIXTrainer, self).__init__(tid)
        self._loss = QMIXLoss()
        self.preprocessor: Preprocessor = None

    def optimize(self, batch):
        self.loss.zero_grad()
        loss_stat = self.loss(batch)
        _ = self.loss.step()
        return loss_stat

    def reset(self, policies: Dict[AgentID, Policy], training_config):
        global_state_space = list(policies.values())[0].custom_config[
            "global_state_space"
        ]
        if self.loss.mixer is None:
            self.preprocessor = get_preprocessor(global_state_space["obs"])(
                global_state_space["obs"]
            )
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
            misc.hard_update(self.loss.mixer_target, self.loss.mixer)

        super(QMIXTrainer, self).reset(policies, training_config)

    def preprocess(self, batch, **kwargs):
        # add keys to batch: CUR_STATE, NEXT_STATE, REWARDS, DONES
        ego_agent = self.agents[0]
        batch[Episode.CUR_STATE] = np.concatenate(
            [
                self.policy[aid].preprocessor.transform(batch[aid][Episode.CUR_OBS])
                for aid in self.agents
            ],
            axis=-1,
        )
        batch[Episode.NEXT_STATE] = np.concatenate(
            [
                self.policy[aid].preprocessor.transform(batch[aid][Episode.NEXT_OBS])
                for aid in self.agents
            ],
            axis=-1,
        )
        batch[Episode.REWARDS] = batch[ego_agent][Episode.REWARDS]
        batch[Episode.DONES] = batch[ego_agent][Episode.DONES]
        return batch
