# MIT License

# Copyright (c) 2021 MARL @ SJTU

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

from typing import Dict, Callable, Union


class Registry:
    """Global registry of algorithms, models, preprocessors and environments

    Examples:
        >>> # register custom model
        >>> Registry.register_custom_model("MyCustomModel", model_class)
        >>> # register custom policy
        >>> Registry.register_custom_policy("MyCustomPolicy", policy_class)
        >>> # register custom environment
        >>> Registry.register_custom_env("MyCustomEnvironment", environment_class)
        >>> # register custom algorithm
        >>> Registry.register_custom_algorithm(
        ...     name="MyCustomAlgo",
        ...     policy="registered_policy_name_or_cls",
        ...     trainer="registered_trainer_name_or_cls",
        ...     loss="registered_loss_name_or_cls")
        >>>
    """

    @staticmethod
    def register_custom_algorithm(
        name: str,
        policy: Union[type, str],
        trainer: Union[type, str],
        loss: Union[type, str] = None,
    ) -> None:
        """Register a custom algorithm by name.

        :param name: str, Name to register the algorithm under.
        :param policy: Union[type, str], Python class or registered name of policy.
        :param trainer: Union[type, str], Python class or registered name of trainer.
        :param loss: Union[type, str], Python class or registered name of loss function.
        :return:
        """
        # _global_registry.register(ALGORITHM, name, policy, trainer, loss)
        pass

    @staticmethod
    def register_custom_model(name: str, model_class: type) -> None:
        """Register a custom model by name.

        :param name: str, Name to register the model under.
        :param model_class: type, Python class of the model.
        :return:
        """
        # _global_registry.register(MODEL, name, model_class)
        pass

    @staticmethod
    def register_custom_policy(name: str, policy_class: type) -> None:
        """Register a custom policy by name.

        :param name: str, Name to register the  policy under.
        :param policy_class: type, Python class of the policy.
        """
        pass

    @staticmethod
    def register_custom_env(name: str, env_class: type) -> None:
        """Register a custom environment by name.

        :param name: str, Name to register the environment under.
        :param env_class: type, Python class of the environment.
        """
        pass
