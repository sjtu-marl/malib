from operator import ge
from torch import nn
import torch
from torch.nn.modules import rnn
from malib.algorithm.common.model import get_model
from malib.algorithm.mappo.utils import RNNLayer, init_fc_weights
from malib.utils.preprocessor import get_preprocessor


class RNNNet(nn.Module):
    def __init__(
        self,
        model_config,
        observation_space,
        action_space,
        custom_config,
        initialization,
    ):
        super().__init__()
        self.base = get_model(model_config)(
            observation_space,
            None,
            custom_config.get("use_cuda", False),
            use_feature_normalization=custom_config["use_feature_normalization"],
        )
        fc_last_hidden = model_config["layers"][-1]["units"]

        act_dim = act_dim = get_preprocessor(action_space)(action_space).size
        self.out = nn.Linear(fc_last_hidden, act_dim)

        use_orthogonal = initialization["use_orthogonal"]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_weights(m):
            if type(m) == nn.Linear:
                init_fc_weights(m, init_method, initialization["gain"])

        self.base.apply(init_weights)
        self.out.apply(init_weights)
        self._use_rnn = custom_config["use_rnn"]
        if self._use_rnn:
            self.rnn = RNNLayer(
                fc_last_hidden,
                fc_last_hidden,
                custom_config["rnn_layer_num"],
                use_orthogonal,
            )

    def forward(self, obs, rnn_states, masks):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        rnn_states = torch.as_tensor(rnn_states, dtype=torch.float32)
        feat = self.base(obs)
        if self._use_rnn:
            masks = torch.as_tensor(masks, dtype=torch.float32)
            feat, rnn_states = self.rnn(feat, rnn_states, masks)

        act_out = self.out(feat)
        return act_out, rnn_states
