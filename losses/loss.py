
import torch
import torch.nn.functional as F


def get_loss():
    return torch.nn.CrossEntropyLoss()


def fedstg_loss(student_logits, labels, teacher_logits=None, beta=1.0):
    ce_loss = F.cross_entropy(student_logits, labels)

    if teacher_logits is None:
        return ce_loss

    temperature = 1.0
    num_classes = student_logits.size(1)

    mask = F.one_hot(labels, num_classes).bool()

    student_logits = student_logits / temperature
    teacher_logits = teacher_logits.detach() / temperature

    student_logits = student_logits.masked_fill(mask, float("-inf"))
    teacher_logits = teacher_logits.masked_fill(mask, float("-inf"))

    student_log_probs = F.log_softmax(student_logits, dim=1)
    teacher_probs = F.softmax(teacher_logits, dim=1)

    kd_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean"
    )

    return ce_loss + beta * kd_loss