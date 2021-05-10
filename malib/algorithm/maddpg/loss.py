import torch
import gym

from malib.algorithm.common import misc
from malib.algorithm.ddpg.loss import DDPGLoss
from malib.backend.datapool.offline_dataset_server import Episode
from malib.algorithm.common.model import get_model


class MADDPGLoss(DDPGLoss):
    def __init__(self):
        super(MADDPGLoss, self).__init__()
        self.cnt = 0

    def _set_centralized_critic(self):
        global_state_space = self.policy.custom_config["global_state_space"]

        self.policy.deregister_state("critic")
        self.policy.deregister_state("target_critic")

        model_cls = get_model(self.policy.model_config["critic"])
        self.policy.set_critic(model_cls(global_state_space, gym.spaces.Discrete(1)))
        self.policy.target_critic = model_cls(
            global_state_space, gym.spaces.Discrete(1)
        )

        self.policy.update_target()

    def reset(self, policy, config):
        """Replace critic with a centralized critic"""
        self._params.update(config)
        if policy is not self.policy:
            self._policy = policy
            self._set_centralized_critic()
            self.setup_optimizers()

    def step(self):
        self.policy.soft_update(tau=self._params["tau"])
        return None

    def __call__(self, agent_batch):
        FloatTensor = (
            torch.cuda.FloatTensor
            if self.policy.custom_config["use_cuda"]
            else torch.FloatTensor
        )
        cast_to_tensor = lambda x: FloatTensor(x.copy())
        cliprange = self._params["grad_norm_clipping"]

        # print(all_agent_batch[agent_id])
        rewards = cast_to_tensor(agent_batch[self.main_id][Episode.REWARDS]).view(-1, 1)
        dones = cast_to_tensor(agent_batch[self.main_id][Episode.DONES]).view(-1, 1)
        cur_obs = cast_to_tensor(agent_batch[self.main_id][Episode.CUR_OBS])

        gamma = self.policy.custom_config["gamma"]

        target_vf_in_list_obs = []
        target_vf_in_list_act = []
        vf_in_list_obs = []
        vf_in_list_act = []

        # set target state
        for aid in self.agents:
            batch = agent_batch[aid]
            target_vf_in_list_obs.append(cast_to_tensor(batch[Episode.NEXT_OBS]))
            target_vf_in_list_act.append(batch["next_act_by_target"])

            vf_in_list_obs.append(cast_to_tensor(batch[Episode.CUR_OBS]))
            vf_in_list_act.append(cast_to_tensor(batch[Episode.ACTION_DIST]))

        target_vf_state = torch.cat(
            [*target_vf_in_list_obs, *target_vf_in_list_act], dim=1
        )
        vf_state = torch.cat([*vf_in_list_obs, *vf_in_list_act], dim=1)

        # ============================== Critic optimization ================================
        target_value = rewards + gamma * (1.0 - dones) * self.policy.target_critic(
            target_vf_state
        )
        eval_value = self.policy.critic(vf_state)
        assert eval_value.shape == target_value.shape, (
            eval_value.shape,
            target_value.shape,
        )
        value_loss = torch.nn.MSELoss()(eval_value, target_value.detach())

        self.optimizers["critic"].zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), cliprange)
        self.optimizers["critic"].step()
        # ==================================================================================

        # ================================ Actor optimization ==============================
        main_idx = None
        for i, aid in enumerate(self.agents):
            # replace with tensor
            if aid == self.main_id:
                vf_in_list_act[i] = self.policy.compute_actions(cur_obs)
                main_idx = i
                break

        vf_state = torch.cat([*vf_in_list_obs, *vf_in_list_act], dim=1)
        policy_loss = -self.policy.critic(vf_state).mean()  # need add regularization?
        policy_loss += (vf_in_list_act[main_idx] ** 2).mean() * 1e-3

        self.optimizers["actor"].zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), cliprange)
        self.optimizers["actor"].step()
        # ==================================================================================

        loss_names = [
            "policy_loss",
            "value_loss",
            "target_value_est",
            "value_est",
        ]
        stats_list = [
            policy_loss.detach().numpy(),
            value_loss.detach().numpy(),
            target_value.mean().detach().numpy(),
            eval_value.mean().detach().numpy(),
        ]

        return dict(zip(loss_names, stats_list))
