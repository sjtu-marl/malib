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
