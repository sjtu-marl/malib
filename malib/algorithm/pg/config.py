DEFAULT_CONFIG = {
    "training_config": {
        "optimizer": "Adam",
        "lr": 1e-4,
        "reward_norm": None,
        "n_repeat": 2,
        "minibatch": 2,
        "gamma": 0.99,
    },
    "model_config": {
        "preprocess_net": {"net_type": None, "config": {"hidden_sizes": [64]}},
        "hidden_sizes": [64],
    },
    "custom_config": {},
}
