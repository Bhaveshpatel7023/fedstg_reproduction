import torch
import torch.nn.functional as F


def weighted_average(state_dicts, client_sizes):
    total_samples = sum(client_sizes)
    if total_samples <= 0:
        raise ValueError("Total client samples must be positive.")

    avg_state = {}
    for key in state_dicts[0]:
        avg_state[key] = sum(
            state[key].float() * (size / total_samples)
            for state, size in zip(state_dicts, client_sizes)
        )
    return avg_state


def kd_kl(student_logits, teacher_logits, temperature=1.0):
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (temperature ** 2)
