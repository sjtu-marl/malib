from malib.algorithm.common.trainer import Trainer
from malib.algorithm.common.policy import Policy
from malib.algorithm.common.loss_func import LossFunc


class ImitationTrainer(Trainer):
    def __init__(self, tid, policy_trainer):
        super(ImitationTrainer, self).__init__(tid)
        self._reward = None
        self._policy_loss = None
        self._reward_loss = None
        self._policy_trainer = None

    @property
    def reward(self) -> Policy:
        return self._reward

    @property
    def policy_loss(self) -> LossFunc:
        return self._policy_loss

    @property
    def reward_loss(self) -> LossFunc:
        return self._reward_loss

    @property
    def policy_trainer(self) -> Trainer:
        return self._policy_trainer

    def optimize_policy(self, batch):

        self.policy_loss.zero_grad()
        policy_loss_stats = self.policy_loss(batch)
        self.policy_loss.step()

        return policy_loss_stats

    def optimize_reward(self, batch):
        self.reward_loss.zero_grad()
        reward_loss_stats = self.reward_loss(batch)
        self.reward_loss.step()

        return reward_loss_stats

    def preprocess(self, **kwargs):
        pass
        # TODO: BC pretraining

    def reset(self, policy, reward, training_config):
        """ Reset policy, called before optimize, and read training configuration """
        self._policy_trainer.reset(policy, training_config)

        self._reward = reward
        if self._loss is not None:
            self._loss.reset(reward, training_config)
