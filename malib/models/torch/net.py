# reference: https://github.com/thu-ml/tianshou/blob/master/tianshou/utils/net/common.py

import operator
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import gym
import functools
import copy

from torch import nn

from malib.utils.preprocessor import get_preprocessor


ModuleType = Type[nn.Module]
Device = Type[torch.device]


def miniblock(
    input_size: int,
    output_size: int = 0,
    norm_layer: Optional[ModuleType] = None,
    activation: Optional[ModuleType] = None,
    linear_layer: Type[nn.Linear] = nn.Linear,
) -> List[nn.Module]:
    """Construct a miniblock with given input/output-size, norm layer and activation.

    Args:
        input_size (int): The input size.
        output_size (int, optional): The output size. Defaults to 0.
        norm_layer (Optional[ModuleType], optional): A nn.Module as normal layer. Defaults to None.
        activation (Optional[ModuleType], optional): A nn.Module as active layer. Defaults to None.
        linear_layer (Type[nn.Linear], optional): A nn.Module as linear layer. Defaults to nn.Linear.

    Returns:
        List[nn.Module]: A list of layers.
    """

    layers: List[nn.Module] = [linear_layer(input_size, output_size)]
    if norm_layer is not None:
        layers += [norm_layer(output_size)]  # type: ignore
    if activation is not None:
        layers += [activation()]
    return layers


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        device: Optional[Union[str, int, torch.device]] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
    ) -> None:
        """Create a MLP.

        Args:
            input_dim (int): dimension of the input vector.
            output_dim (int, optional): dimension of the output vector. If set to 0, \
                there is no final linear layer. Defaults to 0.
            hidden_sizes (Sequence[int], optional): shape of MLP passed in as a list, not \
                including input_dim and output_dim. Defaults to ().
            norm_layer (Optional[Union[ModuleType, Sequence[ModuleType]]], optional): use which normalization before \
                activation, e.g., ``nn.LayerNorm`` and ``nn.BatchNorm1d``. Default to no normalization. \
                You can also pass a list of normalization modules with the same length of hidden_sizes, \
                to use different normalization module in different layers. Default to no normalization. Defaults to None.
            activation (Optional[Union[ModuleType, Sequence[ModuleType]]], optional): which activation \
                to use after each layer, can be both the same activation for all layers if passed in nn.Module, \
                or different activation for different Modules if passed in a list. Defaults to nn.ReLU.
            device (Optional[Union[str, int, torch.device]], optional): which device to create this model \
                on. Defaults to None.
            linear_layer (Type[nn.Linear], optional): use this module as linear layer. Defaults to nn.Linear.
        """

        super().__init__()
        self.device = device
        if norm_layer:
            if isinstance(norm_layer, list):
                assert len(norm_layer) == len(hidden_sizes)
                norm_layer_list = norm_layer
            else:
                norm_layer_list = [norm_layer for _ in range(len(hidden_sizes))]
        else:
            norm_layer_list = [None] * len(hidden_sizes)
        if activation:
            if isinstance(activation, list):
                assert len(activation) == len(hidden_sizes)
                activation_list = activation
            else:
                activation_list = [activation for _ in range(len(hidden_sizes))]
        else:
            activation_list = [None] * len(hidden_sizes)
        hidden_sizes = [input_dim] + list(hidden_sizes)
        model = []
        for in_dim, out_dim, norm, activ in zip(
            hidden_sizes[:-1], hidden_sizes[1:], norm_layer_list, activation_list
        ):
            model += miniblock(in_dim, out_dim, norm, activ, linear_layer)
        if output_dim > 0:
            model += [linear_layer(hidden_sizes[-1], output_dim)]
        self.output_dim = output_dim or hidden_sizes[-1]
        self.model = nn.Sequential(*model)

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if not isinstance(obs, torch.Tensor):
            obs = obs.copy()  # avoid not-writable warning here
        if self.device is not None:
            obs = torch.as_tensor(
                obs,
                device=self.device,  # type: ignore
                dtype=torch.float32,
            )
        else:
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        return self.model(obs.flatten(-1))  # type: ignore


class Net(nn.Module):
    def __init__(
        self,
        state_shape: Union[int, Sequence[int]],
        action_shape: Union[int, Sequence[int]] = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: Optional[ModuleType] = None,
        activation: Optional[ModuleType] = nn.ReLU,
        device: Union[str, int, torch.device] = "cpu",
        softmax: bool = False,
        concat: bool = False,
        num_atoms: int = 1,
        dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.softmax = softmax
        self.num_atoms = num_atoms
        input_dim = int(np.prod(state_shape))
        action_dim = int(np.prod(action_shape)) * num_atoms
        if concat:
            input_dim += action_dim
        self.use_dueling = dueling_param is not None
        output_dim = action_dim if not self.use_dueling and not concat else 0
        self.model = MLP(
            input_dim, output_dim, hidden_sizes, norm_layer, activation, device
        )
        self.output_dim = self.model.output_dim
        if self.use_dueling:  # dueling DQN
            q_kwargs, v_kwargs = dueling_param  # type: ignore
            q_output_dim, v_output_dim = 0, 0
            if not concat:
                q_output_dim, v_output_dim = action_dim, num_atoms
            q_kwargs: Dict[str, Any] = {
                **q_kwargs,
                "input_dim": self.output_dim,
                "output_dim": q_output_dim,
                "device": self.device,
            }
            v_kwargs: Dict[str, Any] = {
                **v_kwargs,
                "input_dim": self.output_dim,
                "output_dim": v_output_dim,
                "device": self.device,
            }
            self.Q, self.V = MLP(**q_kwargs), MLP(**v_kwargs)
            self.output_dim = self.Q.output_dim

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: obs -> flatten (inside MLP)-> logits."""
        logits = self.model(obs)
        bsz = logits.shape[0]
        if self.use_dueling:  # Dueling DQN
            q, v = self.Q(logits), self.V(logits)
            if self.num_atoms > 1:
                q = q.view(bsz, -1, self.num_atoms)
                v = v.view(bsz, -1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        elif self.num_atoms > 1:
            logits = logits.view(bsz, -1, self.num_atoms)
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state


class Recurrent(nn.Module):
    def __init__(
        self,
        layer_num: int,
        state_shape: Union[int, Sequence[int]],
        action_shape: Union[int, Sequence[int]],
        device: Union[str, int, torch.device] = "cpu",
        hidden_layer_size: int = 128,
    ) -> None:
        super().__init__()
        self.device = device
        self.nn = nn.LSTM(
            input_size=hidden_layer_size,
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            batch_first=True,
        )
        self.fc1 = nn.Linear(int(np.prod(state_shape)), hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, int(np.prod(action_shape)))

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Dict[str, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Mapping: obs -> flatten -> logits.
        In the evaluation mode, `obs` should be with shape ``[bsz, dim]``; in the
        training mode, `obs` should be with shape ``[bsz, len, dim]``. See the code
        and comment for more detail.
        """
        obs = torch.as_tensor(
            obs,
            device=self.device,  # type: ignore
            dtype=torch.float32,
        )
        # obs [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(-2)
        obs = self.fc1(obs)
        self.nn.flatten_parameters()
        if state is None:
            obs, (hidden, cell) = self.nn(obs)
        else:
            # we store the stack data in [bsz, len, ...] format
            # but pytorch rnn needs [len, bsz, ...]
            obs, (hidden, cell) = self.nn(
                obs,
                (
                    state["hidden"].transpose(0, 1).contiguous(),
                    state["cell"].transpose(0, 1).contiguous(),
                ),
            )
        obs = self.fc2(obs[:, -1])
        # please ensure the first dim is batch size: [bsz, len, ...]
        return obs, {
            "hidden": hidden.transpose(0, 1).detach(),
            "cell": cell.transpose(0, 1).detach(),
        }


class ActorCritic(nn.Module):
    """An actor-critic network for parsing parameters.
    Using ``actor_critic.parameters()`` instead of set.union or list+list to avoid
    issue #449.
    :param nn.Module actor: the actor network.
    :param nn.Module critic: the critic network.
    """

    def __init__(self, actor: nn.Module, critic: nn.Module) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic


class DataParallelNet(nn.Module):
    """DataParallel wrapper for training agent with multi-GPU.
    This class does only the conversion of input data type, from numpy array to torch's
    Tensor. If the input is a nested dictionary, the user should create a similar class
    to do the same thing.
    :param nn.Module net: the network to be distributed in different GPUs.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = nn.DataParallel(net)

    def forward(
        self, obs: Union[np.ndarray, torch.Tensor], *args: Any, **kwargs: Any
    ) -> Tuple[Any, Any]:
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        return self.net(obs=obs.cuda(), *args, **kwargs)


def _parse_model_config_from_dict(**kwargs) -> Dict[str, Any]:
    """Parse a given raw dict configuration to capable parameter dict.

    Raises:
        NotImplementedError: _description_

    Returns:
        Dict[str, Any]: _description_
    """

    res = {}
    for k, v in kwargs.items():
        if k in ["input_dim", "output_dim", "num_atoms", "hidden_layer_size"]:
            res[k] = int(v)
        elif k == "hidden_sizes":
            assert isinstance(v, Sequence) and isinstance(v[0], int)
            res[k] = copy.deepcopy(v)
        elif k in ["norm_layer", "activation", "linear_layer"]:
            # convert str to module obj
            if isinstance(v, str):
                res[k] = getattr(torch.nn, v)
            elif isinstance(v, Sequence):
                res[k] = [getattr(torch.nn, _v) for _v in v]
            else:
                raise TypeError(f"unexpected type for model configuration: {type(v)}")
        elif k in ["softmax", "concat"]:
            res[k] = bool(v)
        elif k in ["dueling_param", "actor", "critic", "net"]:
            res[k] = v if k != "dueling_param" else copy.deepcopy(v)
        elif k in ["state_shape", "action_shape", "layer_num"]:
            if isinstance(v, Sequence):
                res[k] = [int(_v) for _v in v]
            else:
                res[k] = int(v)
    return res


def _make_net_from_observation(
    observation_space: gym.Space, device: Device, parsed_model_config: Dict[str, Any]
) -> nn.Module:
    """Make a network from given observation and parsed_model_config. Note here for \
        any legal `observation_space`, they will be all set to flatten shape.

    Args:
        observation_space (gym.Space): Observation space.
        parsed_model_config (Dict[str, Any]): Parsed model configuration dict.

    Returns:
        nn.Module: An network instance.
    """

    # always flatten
    preprocessor = get_preprocessor(observation_space)(observation_space)
    parsed_model_config["device"] = device
    return Net(state_shape=preprocessor.shape, **parsed_model_config)


def make_net(
    observation_space: gym.Space,
    action_space: gym.Space,
    device: Device,
    net_type: str = None,
    **kwargs,
) -> nn.Module:
    """Create a network instance with specific network configuration.

    Args:
        observation_space (gym.Space): The observation space used to determine \
            which network type will be used, if net_type is not be specified
        action_space (gym.Space): The action space will be used to determine the network \
            output dim, if `output_dim` or `action_shape` is not given in kwargs
        device (Device): Indicates device allocated.
        net_type (str, optional): Indicates the network type, could be one from \
            {mlp, net, rnn, actor_critic, data_parallel}

    Raises:
        ValueError: Unexpected network type.

    Returns:
        nn.Module: A network instance.
    """

    # parse custom_config
    cls = None
    parsed_model_config = _parse_model_config_from_dict(device=device, **kwargs)

    if net_type is None:
        return _make_net_from_observation(
            observation_space, device, parsed_model_config
        )
    else:
        if net_type == "mlp":
            cls = MLP
            # compute input dim here
            parsed_model_config["input_dim"] = (
                functools.reduce(
                    operator.mul,
                    get_preprocessor(observation_space)(observation_space).shape,
                )
                if not isinstance(observation_space, gym.spaces.Discrete)
                else 1
            )
            if "output_dim" not in parsed_model_config:
                parsed_model_config["output_dim"] = (
                    action_space.n
                    if isinstance(action_space, gym.spaces.Discrete)
                    else functools.reduce(operator.mul, action_space.shape)
                )
        elif net_type == "general_net":
            cls = Net
            parsed_model_config["state_shape"] = get_preprocessor(observation_space)(
                observation_space
            ).shape
            if "action_shape" not in parsed_model_config:
                parsed_model_config["action_shape"] = (
                    (action_space.n,)
                    if isinstance(action_space, gym.spaces.Discrete)
                    else action_space.shape
                )
        elif net_type == "rnn":
            cls = Recurrent
            parsed_model_config["state_shape"] = (
                get_preprocessor(observation_space)(observation_space).shape
                if not isinstance(observation_space, gym.spaces.Discrete)
                else (1,)
            )
            if "action_shape" not in parsed_model_config:
                parsed_model_config["action_shape"] = (
                    (action_space.n,)
                    if isinstance(action_space, gym.spaces.Discrete)
                    else action_space.shape
                )
        elif net_type == "actor_critic":
            cls = ActorCritic
        elif net_type == "data_parallel":
            cls = DataParallelNet
        else:
            raise ValueError("Unexpected net type: {}".format(net_type))

        if net_type not in ["actor_critic", "data_parallel"]:
            parsed_model_config["device"] = device

        net = cls(**parsed_model_config).to(device)
        return net
