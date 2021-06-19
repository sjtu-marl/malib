import torch

from torch.nn import functional as F

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
        for p in self.policy.items():
            assert isinstance(p, DQN)
            p.soft_update(self._params["tau"])

    def reset(self, policy, configs):
        super(QMIXLoss, self).reset(policy, configs)
        if self._cast_to_tensor is None:
            self._cast_to_tensor = (
                lambda x: torch.cuda.FloatTensor(x.copy())
                if self.policy.custom_config["use_cuda"]
                else torch.FloatTensor(x.copy())
            )

    def setup_optimizers(self, *args, **kwargs):
        optimizer = self.optim_cls(self.mixer.parameters(), lr=self._params["lr"])
        optimizer.add_param_group(
            {f"{aid}_params": p.parameters() for aid, p in self.policy.items()}
        )
        self.optimizers.append(optimizer)

    def step(self) -> Any:
        super(QMIXLoss, self).step()
        self.update_target()

    def __call__(self, batch) -> Dict[str, Any]:
        self.loss = []
        state = self._cast_to_tensor(batch[Episode.CUR_STATE])
        next_state = self._cast_to_tensor(batch[Episode.NEXT_STATE])
        rewards = self._cast_to_tensor(batch[Episode.REWARDS]).view(-1, 1)
        dones = self._cast_to_tensor(batch[Episode.DONES]).view(-1, 1)

        # ================= handle for each agent ====================================
        q_vals, next_max_q_vals = [], []
        for env_agent_id in self.agents:
            _batch = batch[env_agent_id]
            obs = self._cast_to_tensor(_batch[Episode.CUR_OBS])
            next_obs = self._cast_to_tensor(_batch[Episode.NEXT_OBS])
            act = self._cast_to_tensor(_batch[Episode.ACTIONS])
            next_action_mask = self._cast_to_tensor(_batch[Episode.NEXT_ACTION_MASK])
            policy: DQN = self.policy[env_agent_id]
            q = policy.critic(obs).gather(1, act.unsqueeze(1)).squeeze()

            q_vals.append(q)
            next_q = policy.target_critic(next_obs)
            next_q[next_action_mask == 0] = -9999999
            next_max_q = next_q.max(1)[0]
            next_max_q_vals.append(next_max_q)

        q_vals = torch.stack(q_vals, dim=1)
        next_max_q_vals = torch.stack(next_max_q_vals, dim=1)
        q_tot = self.mixer(q_vals, state).squeeze()

        next_max_q_tot = self.mixer_target(next_max_q_vals, next_state).squeeze()
        targets = rewards + self._params["gamma"] * (1.0 - dones) * next_max_q_tot
        loss = F.smooth_l1_loss(q_tot, targets.detach())
        self.loss.append(loss)

        return {"mixer_loss": loss.detach().numpy()}
