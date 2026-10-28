import torch
import torch.nn.functional as F

# --- 1. Client Side: FedNTD Loss Implementation (Eq. 3 Masking Logic) ---
def compute_loss(client, inputs, labels, logits, global_model=None, teacher=None):
    """
    Direct FedSTG Local Loss (Eq. 3): CrossEntropy + Beta * KL_NonTrue
    Stabilized with strict numerical thresholds to avoid 'nan' values.
    """
    ce_loss = F.cross_entropy(logits, labels)
    if teacher is None:
        return ce_loss

    with torch.no_grad():
        teacher_logits = teacher(inputs)

    temperature = 1.0
    beta = 1.0

    # Apply scaling factor directly across logit spaces safely
    s_prob = F.softmax(logits / temperature, dim=1)
    t_prob = F.softmax(teacher_logits / temperature, dim=1)
    
    # Construct binary indicator tracking arrays to wipe true labels
    mask = torch.ones_like(s_prob).scatter_(1, labels.unsqueeze(1), 0)
    
    # CRITICAL NUMERICAL STABILITY REPAIR: Clamp standard summation flags to avoid division by zero
    s_prob_masked = (s_prob * mask) / (s_prob * mask).sum(dim=1, keepdim=True).clamp(min=1e-7)
    t_prob_masked = (t_prob * mask) / (t_prob * mask).sum(dim=1, keepdim=True).clamp(min=1e-7)
    
    # Force minimal boundary threshold limits directly inside log scaling blocks to prevent log(0) -> nan
    s_log_prob_masked = torch.log(s_prob_masked.clamp(min=1e-7))
    
    distillation_loss = F.kl_div(s_log_prob_masked, t_prob_masked, reduction='batchmean')
    return ce_loss + beta * distillation_loss
