from malib.utils.typing import AgentInvolveInfo

ignored_key = [
    "action_space",
    "model_config",
    "custom_config",
    "observation_space",
    "registered_name",
]


def pretty_dict(d, indent=0):
    res = "\n"
    for k, v in d.items():
        if k in ignored_key:
            continue
        res += "\t" * indent + str(k) + "\n"
        res += pretty_print(v, indent + 1) + "\n"
    return res


def pretty_tuple(d, indent=0):
    pending_str = []
    types = set([type(e) for e in d])
    for e in d:
        pending_str.append(pretty_print(e, indent + 1))

    if len(types) == 1:
        res = "(" + pending_str + ")"
    else:
        res = "(" + "\n".join(pending_str) + ")\n"

    return "\t" * indent + res


def pretty_list(d, indent=0):
    pending_str = []
    types = set([type(e) for e in d])
    for e in d:
        pending_str.append(pretty_print(e, indent + 1))

    if len(types) == 1:
        res = "[" + "".join(pending_str) + "]"
    else:
        res = "[" + "\n".join(pending_str) + "]\n"

    return "\t" * indent + res


def pretty_print(data, indent=0):
    if isinstance(data, dict):
        res = pretty_dict(data, indent)
    elif isinstance(data, tuple):
        res = pretty_tuple(data, indent)
    elif isinstance(data, list):
        res = pretty_list(data, indent)
    elif isinstance(data, AgentInvolveInfo):
        res = "\t" * indent + "AgentInvolveInfo:\n"
        res += (
            "\t" * indent
            + "populations:\n"
            + pretty_print(data.populations, indent + 1)
        )
        res += (
            "\t" * indent
            + "trainable_pairs:\n"
            + pretty_print(data.trainable_pairs, indent + 1)
        )
    else:
        res = "\t" * indent + str(data)
    return res


if __name__ == "__main__":
    a = {
        "agent-0": {"policy-1": 0.323, "policy-2": 0.4, "policy-3": 0.3},
        "agent-1": {"policy-1": 0.3, "policy-2": 0.4, "policy-3": 0.3},
        "agent-2": {"policy-1": 0.3, "policy-2": 0.4, "policy-3": 0.3},
    }
    print(pretty_dict(a))
