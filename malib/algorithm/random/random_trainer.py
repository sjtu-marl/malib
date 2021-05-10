from malib.algorithm.common.trainer import Trainer


class RandomTrainer(Trainer):
    def __init__(self, observation_space, action_space, policy=None):
        super().__init__(observation_space, action_space)
        self._policy = policy

    def optimize(self, batch, other_agent_batches=None):
        return {}

    def save(self, **kwargs):
        raise NotImplementedError

    def load(self, **kwargs):
        raise NotImplementedError
