import torch
import torch.nn.functional as F

def weighted_average(state_dicts, client_sizes):
    """
    Computes the standard federated averaging (FedAvg baseline) 
    by scaling model weights based on client local sample proportions.
    """
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
    """
    Computes the standard standard Knowledge Distillation loss 
    using full-coordinate KL Divergence scaling.
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (temperature ** 2)


def evaluate(model, test_dataset, device="cpu"):
    """
    Evaluates the global generalization performance accuracy metric 
    against an isolated target validation dataset split.
    """
    model.eval()
    model.to(device)
    
    # Instantiate validation streaming container for target metrics tracking
    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )
    
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Extract logit probability distributions outputs matrices
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
    # Calculate performance ratio percentage baseline metric value
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    return accuracy
