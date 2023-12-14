class Config:

    TRAINING_CONIG = {
        "gae_lambda": 0.95,
        "optimizer": "Adam",
        "lr": 1e-4,
        "reward_norm": None,
        "n_repeat": 2,
        "minibatch": 2,
        "batch_size": 32,
        "gamma": 0.99,
        "repeats": 1,
        "ratio_clip": 0.2,
        "dual_clip": None,
        "vf_ratio": 0.1,
        "ent_ratio": 0.01,
        "use_adv_norm": False,
        "adv_norm_eps": 1e-8,
        "use_grad_norm": False,
        "use_value_clip": False,
    }

    CUSTOM_CONFIG = {}

    MODEL_CONFIG = {
        "preprocess_net": {"net_type": None, "config": {"hidden_sizes": [64]}},
        "hidden_sizes": [64],
    }
