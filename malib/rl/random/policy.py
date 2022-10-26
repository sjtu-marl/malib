from typing import Dict, Any

from gym import spaces

from malib.rl.pg import PGPolicy


class RandomPolicy(PGPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        model_config: Dict[str, Any],
        custom_config: Dict[str, Any],
        **kwargs
    ):

        super().__init__(
            observation_space, action_space, model_config, custom_config, **kwargs
        )
