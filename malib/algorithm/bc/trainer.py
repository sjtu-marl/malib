from malib.algorithm.bc.policy import BehaviorCloning
from malib.algorithm.common.trainer import Trainer
from malib.backend.datapool.offline_dataset_server import Episode


class BCTrainer(Trainer):
    def optimize(self, batch, other_agent_batches=None):
        assert isinstance(self.policy, BehaviorCloning), type(self.policy)

        observations, expert_actions = batch[Episode.CUR_OBS], batch[Episode.ACTIONS]
        eval_actions = self.policy.compute_actions(observations)
        # XXX(ming): loss func will be transferred as a parameter
        self.loss_func.zero_grad()
        loss_stats = self.loss_func(eval_actions, expert_actions)
        self.loss_func.step()

        return loss_stats

    def save(self, **kwargs):
        pass

    def load(self, **kwargs):
        pass
