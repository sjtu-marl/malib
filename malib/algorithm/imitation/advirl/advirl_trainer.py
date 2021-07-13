from malib.algorithm.common.trainer import Trainer
from malib.algorithm.imitation.imitation_trainer import ImitationTrainer
from malib.algorithm.imitation.advirl.reward import Discriminator


class AdvIRLTrainer(ImitationTrainer):
    def __init__(self, tid):
        super(AdvIRLTrainer, self).__init__(tid)

        return super().loss_stats

    def optimize_reward(self, batch):
        assert isinstance(self._policy, Discriminator), type(self._reward)
        super.optimize_reward(batch)
