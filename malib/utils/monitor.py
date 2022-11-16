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

from typing import Dict, Union, Sequence

from torch.utils import tensorboard


def write_to_tensorboard(
    writer: tensorboard.SummaryWriter,
    info: Dict,
    global_step: Union[int, Dict],
    prefix: str,
):
    """Write learning info to tensorboard.

    Args:
        writer (tensorboard.SummaryWriter): The summary writer instance.
        info (Dict): The information dict.
        global_step (int): The global step indicator.
        prefix (str): Prefix added to keys in the info dict.
    """
    if writer is None:
        return

    dict_step = isinstance(global_step, Dict)

    prefix = f"{prefix}/" if len(prefix) > 0 else ""
    for k, v in info.items():
        if isinstance(v, dict):
            # add k to prefix
            write_to_tensorboard(
                writer,
                v,
                global_step if not dict_step else global_step[k],
                f"{prefix}{k}",
            )
        elif isinstance(v, Sequence):
            raise NotImplementedError(
                f"Sequence value cannot be logged currently: {v}."
            )
        elif v is None:
            continue
        else:
            writer.add_scalar(
                f"{prefix}{k}",
                v,
                global_step=global_step
                if not dict_step
                else global_step["global_step"],
            )
