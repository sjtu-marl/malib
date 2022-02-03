import pytest
from malib.algorithm.common.model import Model, get_model, mlp, RNN, MLP
from gym import spaces
import numpy as np


@pytest.fixture
def layers_config():
    return [
        {"units": 32, "activation": "ReLU"},
        {"units": 32, "activation": "Tanh"},
        {"units": 32, "activation": "LeakyReLU"},
    ]


def test_mlp_func(layers_config):
    mlp(layers_config)


@pytest.mark.parametrize(
    "model_fn, model_config",
    [
        (
            MLP,
            {
                "layers": [
                    {"units": 32, "activation": "ReLU"},
                    {"units": 32, "activation": "Tanh"},
                    {"units": 32, "activation": "LeakyReLU"},
                ],
                "output": {"activation": "Identity"},
            },
        ),
        (RNN, {"rnn_hidden_dim": 32}),
    ],
    scope="class",
)
@pytest.mark.parametrize(
    "obs_space",
    [
        spaces.Box(0.0, 1.0, (8,)),
        spaces.Dict({"1": spaces.Box(0.0, 1.0, (8,)), "2": spaces.Box(0.0, 1.0, (6,))}),
    ],
    scope="class",
)
@pytest.mark.parametrize(
    "act_space",
    [
        spaces.Box(0.0, 1.0, (8,)),
        spaces.Dict({"1": spaces.Box(0.0, 1.0, (8,)), "2": spaces.Box(0.0, 1.0, (6,))}),
    ],
    scope="class",
)
class TestModel:
    @pytest.fixture(autouse=True)
    def setUp(self, model_fn, model_config, obs_space, act_space):
        self.model = model_fn(obs_space, act_space, model_config)
        assert isinstance(self.model, Model)

    def test_initial_state(self):
        init_state = self.model.get_initial_state()
        assert isinstance(init_state, list)

    def test_forward(self):
        batch_size = 32
        if isinstance(self.model, MLP):
            inputs = np.zeros((batch_size, self.model.input_dim))
            outputs = self.model(inputs)
            assert outputs.shape[0] == batch_size
        elif isinstance(self.model, RNN):
            inputs = (
                np.zeros((batch_size, self.model.input_dim)),
                self.model.get_initial_state(batch_size)[0],
            )
            outputs = self.model(*inputs)
            assert outputs[0].shape[0] == outputs[1].shape[0] == batch_size
        else:
            raise ValueError("Please add tests for {}".format(type(self.model)))


@pytest.mark.parametrize("model_type", ["mlp", "rnn", "cnn", "rcnn"])
def test_get_model(model_type, layers_config):

    model_config = {"network": model_type, "layers": layers_config}

    if model_type in ["rnn", "rcnn"]:
        model_config.update({"rnn_hidden_dim": 32})

    try:
        get_model(model_config)
    except NotImplementedError:
        print("model_type {} is not implement yet.".format(model_type))
