# https://github.com/project-codeflare/zero-copy-model-loading

import torch
import ray

from malib.utils.logging import Logger

try:
    import zerocopy
except ImportError:
    Logger.warning("No package named zerocopy, please install it first.")


# the following code piece can load a BertModel in 0.004s
tmp = torch.nn.Module()
ref = ray.put(zerocopy.extract_tensors(tmp))
model_graph, tensors = ray.get(ref)
zerocopy.replace_tensors(model_graph, tensors)


# zero-copy method: stateless task
@ray.remote
def run_model(model_and_tensors, model_input):
    model_graph, tensors = model_and_tensors
    zerocopy.replace_tensors(model_graph, tensors)
    with torch.inference_mode():
        return model_graph(**model_input)


model_result = ray.get(run_model.remote(ref, model_input))


async def get_model_result(model_ref, model_input):
    return await zerocopy.call_model.remote(model_ref, [], model_input)
