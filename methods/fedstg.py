
import torch
import torch.nn.functional as F

from methods.common import weighted_average, kd_kl


def compute_loss(
    client,
    inputs,
    labels,
    logits,
    global_model=None,
    teacher=None,
):
    ce_loss = F.cross_entropy(logits, labels)

    if teacher is None:
        return ce_loss

    with torch.no_grad():
        teacher_logits = teacher(inputs)

    temperature = getattr(
        __import__("config"),
        "TEMPERATURE",
        1.0,
    )
    beta = getattr(__import__("config"), "BETA", 1.0)

    distillation_loss = kd_kl(
        logits,
        teacher_logits,
        temperature=temperature,
    )

    return ce_loss + beta * distillation_loss


def aggregate(client_results, client_sizes):
    states = [
        result["state_dict"]
        if isinstance(result, dict) and "state_dict" in result
        else result
        for result in client_results
    ]

    sizes = [
        result.get("num_samples", size)
        if isinstance(result, dict)
        else size
        for result, size in zip(client_results, client_sizes)
    ]

    return weighted_average(states, sizes)