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


class Config:

    TRAINING_CONFIG = {
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
