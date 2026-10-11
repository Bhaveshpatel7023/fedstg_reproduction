from importlib import import_module


def get_method(method_name):
    name = method_name.lower().strip()

    if name != "fedstg":
        raise ValueError("Only FedSTG method is supported.")

    module = import_module("methods.fedstg")

    if not hasattr(module, "compute_loss") or not hasattr(module, "aggregate"):
        raise NotImplementedError("FedSTG is missing required functions.")

    return module
