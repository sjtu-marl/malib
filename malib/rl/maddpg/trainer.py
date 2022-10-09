from malib.rl.common.trainer import Trainer
from malib.utils.episode import Episode
from .loss import MADDPGLoss


class MADDPGTrainer(Trainer):
    def __init__(self, tid):
        super(MADDPGTrainer, self).__init__(tid)
        self._loss = MADDPGLoss()

    def optimize(self, batch):
        assert batch is not None, "MADDPG must use other agent batch"
        # main_id and agents are registered outerside (in ctde agent)
        self.loss.main_id = self.main_id
        self.loss.agents = self.agents

        self.loss.zero_grad()
        loss_stat = self.loss(batch)
        _ = self.loss.step()
        return loss_stat

    def preprocess(self, batch, **kwargs):
        """Execution decision making with next observations from batched data, and fill these online
        execution results into batch.

        :param batch: Dict[AgentID, Any], a mapping from environment agents to batched data entities
        """

        other_policies = kwargs["other_policies"]
        for aid, policy in other_policies.items():
            if batch[aid].get("next_act_by_target") is None:
                batch[aid][
                    "next_act_by_target"
                ] = policy.compute_actions_by_target_actor(
                    batch[aid][Episode.NEXT_OBS].copy()
                ).detach()

        return batch

    def save(self):
        pass

    def load(self, **kwargs):
        pass
