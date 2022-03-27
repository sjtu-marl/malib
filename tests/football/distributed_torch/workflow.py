import torch

from ray.util.sgd.torch import TrainingOperator
from ray.util.sgd.torch import TorchTrainer


class IMPALAOperator(TrainingOperator):
    def setup(self, config):
        return super().setup(config)

    def train_batch(self, batch, batch_info):
        return super().train_batch(batch, batch_info)


if __name__ == "__main__":
    num_workers = 1
    config = None
    trainer = TorchTrainer(
        training_operator_cls=IMPALAOperator,
        num_workers=num_workers,
        config=config,
        use_gpu=True,
    )
