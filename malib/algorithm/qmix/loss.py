import torch

from torch.nn import functional as F

from malib.algorithm.common import misc
from malib.utils.typing import Dict, Any
from malib.algorithm.common.loss_func import LossFunc
from malib.backend.datapool.offline_dataset_server import Episode
from malib.algorithm.dqn.policy import DQN


class QMIXLoss(LossFunc):
    def __init__(self):
        super(QMIXLoss, self).__init__()
        self._cast_to_tensor = None
        self._mixer = None
        self._mixer_target = None
        self._params = {"gamma": 0.99, "lr": 5e-4, "tau": 0.01}

    @property
    def mixer(self):
        return self._mixer

    @property
    def mixer_target(self):
        return self._mixer_target

    def set_mixer(self, mixer):
        self._mixer = mixer

    def set_mixer_target(self, mixer_target):
        self._mixer_target = mixer_target

    def update_target(self):
        for _, p in self.policy.items():
            assert isinstance(p, DQN), type(p)
            p.soft_update(self._params["tau"])
        with torch.no_grad():
            misc.soft_update(self.mixer_target, self.mixer, self._params["tau"])

    def reset(self, policy, configs):
        super(QMIXLoss, self).reset(policy, configs)
        self._params.update(list(self.policy.values())[0].custom_config)
        if self._cast_to_tensor is None:
            self._cast_to_tensor = (
                lambda x: torch.cuda.FloatTensor(x.copy())
                if self._params["use_cuda"]
                else torch.FloatTensor(x.copy())
            )

    def setup_optimizers(self, *args, **kwargs):
        assert self.mixer is not None, "Mixer has not been set yet!"
        if self.optimizers is None:
            self.optimizers = self.optim_cls(
                self.mixer.parameters(), lr=self._params["lr"]
            )
        else:
            self.optimizers.param_groups = []
            self.optimizers.add_param_group({"params": self.mixer.parameters()})
        for policy in self.policy.values():
            self.optimizers.add_param_group({"params": policy.critic.parameters()})

    def step(self) -> Any:
        _ = [item.backward() for item in self.loss]
        self.optimizers.step()
        self.update_target()
        for p in self.policy.values():
            p._step += 1

    def __call__(self, batch) -> Dict[str, Any]:
        self.loss = []
        state = self._cast_to_tensor(list(batch.values())[0][Episode.CUR_STATE])
        next_state = self._cast_to_tensor(list(batch.values())[0][Episode.NEXT_STATE])
        rewards = self._cast_to_tensor(list(batch.values())[0][Episode.REWARD]).view(
            -1, 1
        )
        dones = self._cast_to_tensor(list(batch.values())[0][Episode.DONE]).view(-1, 1)

        # ================= handle for each agent ====================================
        q_vals, next_max_q_vals = [], []
        for env_agent_id in self.agents:
            _batch = batch[env_agent_id]
            obs = self._cast_to_tensor(_batch[Episode.CUR_OBS])
            next_obs = self._cast_to_tensor(_batch[Episode.NEXT_OBS])
            act = torch.LongTensor(_batch[Episode.ACTION])
            next_action_mask = self._cast_to_tensor(_batch[Episode.NEXT_ACTION_MASK])
            policy: DQN = self.policy[env_agent_id]
            q = policy.critic(obs).gather(-1, act.unsqueeze(1)).squeeze()
            q_vals.append(q)
            next_q = policy.target_critic(next_obs)
            next_q[next_action_mask == 0] = -9999999
            next_max_q = next_q.max(1)[0]
            next_max_q_vals.append(next_max_q)

        q_vals = torch.stack(q_vals, dim=-1)
        next_max_q_vals = torch.stack(next_max_q_vals, dim=-1)
        q_tot = self.mixer(q_vals, state)

        next_max_q_tot = self.mixer_target(next_max_q_vals, next_state)
        targets = (
            rewards + self._params["gamma"] * (1.0 - dones) * next_max_q_tot.detach()
        )
        loss = F.smooth_l1_loss(q_tot, targets)
        self.loss.append(loss)

        return {
            "mixer_loss": loss.detach().numpy(),
            "value": q_tot.mean().detach().numpy(),
            "target_value": targets.mean().detach().numpy(),
        }
