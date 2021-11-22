from malib.trainer.trainer import Trainer
from malib.algorithm.ppo.policy import PPO


class PPOTrainer(Trainer):
    def __init__(self, observation_space, action_space, policy=None):
        super(PPOTrainer, self).__init__(observation_space, action_space)
        self._policy = policy

        self.cnt = 0
        self.update_interval = 5

    def optimize(self, batch, other_agent_batches=None):
        assert isinstance(self._policy, PPO), type(self._policy)
        self.cnt = (self.cnt + 1) % self.update_interval
        if self.cnt == 0:
            self._policy.update_target()
        loss_stats = self._policy.loss_func(batch)
        return loss_stats

    def save(self, **kwargs):
        raise NotImplementedError

    def load(self, **kwargs):
        raise NotImplementedError
