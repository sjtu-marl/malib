from malib.algorithm.common.trainer import Trainer
from malib.algorithm.sac.policy import SAC


class SACTrainer(Trainer):
    def __init__(self, observation_space, action_space, policy=None):
        super(SACTrainer, self).__init__(None, observation_space, action_space)
        self._policy = policy

    def optimize(self, batch, other_agent_batches=None):
        assert isinstance(self._policy, SAC), type(self._policy)
        loss_stats = self._policy.update(batch)
        return loss_stats

    def save(self, **kwargs):
        raise NotImplementedError

    def load(self, **kwargs):
        raise NotImplementedError

    def preprocess(self, **kwargs):
        pass
