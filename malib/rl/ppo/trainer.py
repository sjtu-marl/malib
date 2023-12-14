# MIT License

# Copyright (c) 2021 MARL @ SJTU

# Author: Ming Zhou

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import List, Dict

import torch

from torch import nn

from malib.utils.tianshou_batch import Batch
from malib.rl.a2c import A2CTrainer


class PPOTrainer(A2CTrainer):
    def train(self, batch: Batch) -> Dict[str, List[float]]:
        repeats = self.training_config["repeats"]
        ratio_clip = self.training_config["ratio_clip"]
        dual_clip = self.training_config["dual_clip"]
        vf_ratio = self.training_config["vf_ratio"]
        ent_ratio = self.training_config["ent_ratio"]
        use_adv_norm = self.training_config["use_adv_norm"]
        adv_norm_eps = self.training_config["adv_norm_eps"]
        use_grad_norm = self.training_config["use_grad_norm"]
        use_value_clip = self.training_config["use_value_clip"]

        # XXX(ming): or we should keep a list of them
        losses, clip_losses, vf_losses, ent_losses = 0.0, 0.0, 0.0, 0.0

        for step in range(repeats):
            dist = self.policy.dist_fn.proba_distribution(batch.logits)

            if use_adv_norm:
                mean, std = batch.adv.mean(), batch.adv.std()
                batch.adv = (batch.adv - mean) / (std + adv_norm_eps)

            ratio = (dist.log_prob(batch.act) - batch.logp_old).exp().float()
            ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
            surr1 = ratio * batch.adv
            surr2 = ratio.clamp(1.0 - ratio_clip, 1.0 + ratio_clip) * batch.adv

            if dual_clip:
                clip1 = torch.min(surr1, surr2)
                clip2 = torch.max(clip1, dual_clip * batch.adv)
                clip_loss = -torch.where(batch.adv < 0, clip2, clip1).mean()
            else:
                clip_loss = -torch.min(surr1, surr2).mean()

            value = self.policy.critic(batch.obs).flatten()
            if use_value_clip:
                v_clip = batch.state_value + (value - batch.state_value).clamp(
                    -ratio_clip, ratio_clip
                )
                vf1 = (batch.returns - value).pow(2)
                vf2 = (batch.returns - v_clip).pow(2)
                vf_loss = torch.max(vf1, vf2).mean()
            else:
                vf_loss = (batch.returns - value).pow(2).mean()

            ent_loss = dist.entropy().mean()
            loss = clip_loss + vf_ratio * vf_loss - ent_ratio * ent_loss

            self.optimizer.zero_grad()
            loss.backward()
            if use_grad_norm:  # clip large gradient
                nn.utils.clip_grad_norm_(
                    self.parameters, max_norm=self.training_config["grad_norm"]
                )
            self.optimizer.step()

            clip_losses += clip_loss.item() / repeats
            vf_losses += vf_loss.item() / repeats
            ent_losses += ent_loss.item() / repeats
            losses += loss.item() / repeats

        return {
            "loss": losses,
            "loss/clip": clip_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
        }
