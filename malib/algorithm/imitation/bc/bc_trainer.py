from malib.algorithm.common.trainer import Trainer
from malib.algorithm.imitation.bc.policy import BC
from malib.algorithm.imitation.bc.loss import BCLoss


class BCTrainer(Trainer):
    def __init__(self, tid):
        super(BCTrainer, self).__init__(tid)
        self._loss = BCLoss()
        self.cnt = 0

    def optimize(self, batch):
        assert isinstance(self._policy, BC), type(self._policy)

        self.loss.zero_grad()
        loss_stats = self.loss(batch)
        self.loss.step()
        
        return loss_stats

    def preprocess(self, **kwargs) -> Any:
        pass
