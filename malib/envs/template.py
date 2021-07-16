"""
The implementation of environment builder. The builder accepts configuration from users to create an instance inherits from `Environment`.
"""

from malib.utils.typing import Dict, Any, List, AgentID
from malib.envs import Environment


def build_environment_template(
    creator: type,
    configs: Dict[str, Any],
    is_sequential: bool,
    extra_returns: List = None,
    step_func: type = None,
    render_func: type = None,
    **kwargs
):
    """Build environment from a template.

    :param type creator: Environment creator
    :param Dict[str,Any] configs: Environment configuration
    :param bool is_sequential: Indicates whether this environment steps in sequential or not
    :param type step_func: Customized stepping function
    :param type render_func: Customized rendering function
    :param Dict kwargs: Default keys, leaves it blank
    """

    class environment(Environment):
        def __init__(self, *args, **kwargs):
            Environment.__init__(self, *args, **kwargs)
            self.is_sequential = is_sequential
            self._extra_returns = extra_returns or self._extra_returns
            self._env = creator(**configs)

        def reset(self, *args, **kwargs):
            return self._env.reset(*args, **kwargs)

        def step(self, actions: Dict[AgentID, Any]):
            if step_func:
                return step_func(self._env, actions)
            else:
                return self._env.step(actions)

        def render(self, *args, **kwargs):
            if render_func:
                return render_func(self._env, *args, **kwargs)
            else:
                return self._env.render(*args, **kwargs)

    return environment(**kwargs)
