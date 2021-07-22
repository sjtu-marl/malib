from typing import Dict, Any

from malib.algorithm.common.trainer import Trainer
from malib.backend.datapool.offline_dataset_server import Episode
from malib.algorithm.imitation.advirl.reward import AdvIRLReward
from malib.algorithm.imitation.advirl.loss import AdvIRLLoss


class AdvIRLTrainer(Trainer):
    def __init__(self, tid, policy_trainer: Trainer):
        """ Serve as a wrapper for RL policy trainer.

        We implement AdvIRLTrainer as a special trainer to wrap other RL trainers, so that one can
        easily utilize different RL algorithms without specifying mutiple AdvIRLTrainer classes.

        :param str tid: Specify trainer id.
        :param Trainer policy_trainer: The trainer to optimize forward RL algorithm during AdvIRL training.

        """

        super(AdvIRLTrainer, self).__init__(tid)
        self._loss = AdvIRLLoss()
        self._policy_trainer = policy_trainer

        self._state_only = False
        self._wrap_absorbing = False

    def preprocess(self, batch, **kwargs) -> Any:
        return batch

    def reset(self, policy, reward, training_config):
        """ Reset policy, called before optimize, and read training configuration """

        self._reward = reward
        self._training_config.update(training_config)
        if self._loss is not None:
            self._loss.reset(reward, training_config)
        self._policy_trainer.reset(policy, training_config)

    def replace_reward(self, batch):
        batch[Episode.REWARD] = self._reward.compute_rewards(
            batch[Episode.CUR_OBS],
            batch[Episode.ACTION],
        ).detach().cpu().numpy()
        return batch

    def optimize(self, batch) -> Dict[str, Any]:
        agent_batch, expert_batch = batch[0], batch[1]
        print(type(agent_batch), type(expert_batch))

        policy_loss_stats = self.optimize_policy(agent_batch)

        reward_loss_stats = self.optimize_reward(agent_batch, expert_batch)

        return {**policy_loss_stats, **reward_loss_stats}

    def optimize_policy(self, agent_batch) -> Dict[str, Any]:
        # replace environment reward with adversarial reward
        agent_batch = self.replace_reward(agent_batch)
        loss_stats = self._policy_trainer.optimize(agent_batch)
        return loss_stats

    def optimize_reward(self, agent_batch, expert_batch) -> Dict[str, Any]:
        assert isinstance(self._reward, AdvIRLReward), type(self._reward)

        if self._wrap_absorbing:
            raise NotImplementedError
            # expert_obs = torch.cat([expert_obs, expert_batch['absorbing'][:, 0:1]], dim=-1)
            # policy_obs = torch.cat([policy_obs, policy_batch['absorbing'][:, 0:1]], dim=-1)

        if self._state_only:
            raise NotImplementedError
            # expert_next_obs = expert_batch['next_observations']
            # policy_next_obs = policy_batch['next_observations']
            # if self.wrap_absorbing:
            #     expert_next_obs = torch.cat([expert_next_obs, expert_batch['absorbing'][:, 1:]], dim=-1)
            #     policy_next_obs = torch.cat([policy_next_obs, policy_batch['absorbing'][:, 1:]], dim=-1)
            # expert_disc_input = torch.cat([expert_obs, expert_next_obs], dim=1)
            # policy_disc_input = torch.cat([policy_obs, policy_next_obs], dim=1)

        self.loss.zero_grad()
        loss_stats = self.loss(agent_batch, expert_batch)
        self.loss.step()

        return loss_stats
