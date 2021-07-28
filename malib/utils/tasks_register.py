# -*- encoding: utf-8 -*-
# -----
# Created Date: 2021/7/14
# Author: Hanjing Wang
# -----
# Last Modified:
# Modified By:
# -----
# Copyright (c) 2020 MARL @ SJTU
# -----

import enum
from functools import wraps
from malib.backend.coordinator.server import CoordinatorServer
from malib.utils.typing import TaskType, Dict
from malib.utils.errors import RegisterFailure


def task_handler_register():
    print("Registering")

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(*args, **kwargs)

        setattr(CoordinatorServer, func.__name__, func)
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
