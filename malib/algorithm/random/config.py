DEFAULT_CONFIG = {
    "training_config": {
        "optimizer": "Adam",
        "lr": 1e-4,
        "reward_norm": None,
        "n_repeat": 2,
        "minibatch": 2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "max_gae_batchsize": 256,
        "value_coef": 1.0,
        "entropy_coef": 1e-3,
        "grad_norm": 5.0,
        "use_cuda": False,
    },
    "model_config": {
        "preprocess_net": {"net_type": None, "config": {"hidden_sizes": [64]}},
        "hidden_sizes": [64],
    },
}
