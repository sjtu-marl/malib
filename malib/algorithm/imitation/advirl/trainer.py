from typing import Dict, Any

from malib.algorithm.imitation.imitation_trainer import ImitationTrainer
from malib.algorithm.imitation.advirl.reward import Discriminator
from malib.algorithm.imitation.advirl.loss import DiscriminatorLoss


class AdvIRLTrainer(ImitationTrainer):
    def __init__(self, tid, policy_trainer):
        super(AdvIRLTrainer, self).__init__(tid)
        self.loss = DiscriminatorLoss()

        return super().loss_stats

    def optimize(self, batch) -> Dict[str, Any]:
        # optimize policy

        # optimize reward

        pass

    def optimize_reward(self, batch):
        assert isinstance(self._policy, Discriminator), type(self._reward)
        expert_batch, agent_batch = batch[0], batch[1]

        if self.wrap_absorbing:
            pass
            # expert_obs = torch.cat([expert_obs, expert_batch['absorbing'][:, 0:1]], dim=-1)
            # policy_obs = torch.cat([policy_obs, policy_batch['absorbing'][:, 0:1]], dim=-1)

        if self.state_only:
            pass
            # expert_next_obs = expert_batch['next_observations']
            # policy_next_obs = policy_batch['next_observations']
            # if self.wrap_absorbing:
            #     expert_next_obs = torch.cat([expert_next_obs, expert_batch['absorbing'][:, 1:]], dim=-1)
            #     policy_next_obs = torch.cat([policy_next_obs, policy_batch['absorbing'][:, 1:]], dim=-1)
            # expert_disc_input = torch.cat([expert_obs, expert_next_obs], dim=1)
            # policy_disc_input = torch.cat([policy_obs, policy_next_obs], dim=1)
        else:
            pass

        self.loss.zero_grad()
        loss_stats = self.loss(expert_batch, agent_batch)
        self.loss.step()

        return loss_stats
