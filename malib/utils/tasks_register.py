# MIT License

# Copyright (c) 2021 MARL @ SJTU

# Author: Hanjing Wang

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

from typing import Dict

import enum

from malib.utils.logging import Logger
from malib.utils.errors import RegisterFailure


def helper_register(cls):
    def decorator(func):
        # @wraps(func)
        # def wrapper(self, *args, **kwargs):
        #     return func(*args, **kwargs)

        if not hasattr(cls, func.__name__):
            setattr(cls, func.__name__, func)
            Logger.info("registered helper handler={}".format(func.__name__))
        return func

    return decorator


def task_handler_register(cls, link):
    def decorator(func):
        # @wraps(func)
        # def wrapper(self, *args, **kwargs):
        #     return func(*args, **kwargs)

        name = "_request_{}".format(link)
        if not hasattr(cls, name):
            setattr(cls, name, func)
            Logger.info("registered request handler={}".format(link))
        return func

    return decorator


RESERVED_TASK_NAMES = []


def register_task_type(tasks: Dict[str, str]):
    existing_tasks = {t.name: t.value for t in TaskType}
    for tasks_name, tasks_value in tasks.items():
        if (
            tasks_name.lower() in RESERVED_TASK_NAMES
            or tasks_name.upper() in RESERVED_TASK_NAMES
            or tasks_value.lower() in RESERVED_TASK_NAMES
            or tasks_value.upper() in RESERVED_TASK_NAMES
        ):
            raise RegisterFailure(
                f"Encountered potential conflicts "
                f"with reserved task names or value "
                f"in registering task {tasks_name}:{tasks_value}"
            )
    existing_tasks.update(tasks)
    TaskType = enum.EnumMeta("TaskType", existing_tasks)
