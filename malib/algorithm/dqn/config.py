DEFAULT_CONFIG = {
    "training_config": {
        "optimizer": "Adam",
        "lr": 1e-4,
        "reward_norm": None,
        "n_repeat": 2,
        "minibatch": 2,
        "gamma": 0.99,
        "update_interval": 1,
        "tau": 0.05,
    },
    "model_config": {
        "net_type": "general_net",
        "config": {"hidden_sizes": [256, 256, 256, 64]},
    },
    "custom_config": {"schedule_timesteps": 10000, "final_p": 0.05, "initial_p": 1.0},
}
