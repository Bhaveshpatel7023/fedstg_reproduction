import torch
import torch.nn.functional as F
from methods.common import weighted_average


# --- 1. Server Side: Class Confidence Metric Evaluation (Eq. 1) ---
def evaluate_class_confidence(model, auxiliary_loader, num_classes=10):
    """
    Vectorized Class Inference Confidence Metric (Eq. 1)
    """
    model.eval()
    device = next(model.parameters()).device
    confidence_sum = torch.zeros(num_classes, device=device)
    class_counts = torch.zeros(num_classes, device=device)
    
    with torch.no_grad():
        for images, labels in auxiliary_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = F.softmax(model(images), dim=1)
            
            true_probs = outputs.gather(1, labels.unsqueeze(1)).squeeze(1)
            confidence = 2 * true_probs - 1.0
            
            confidence_sum.scatter_add_(0, labels, confidence)
            class_counts.scatter_add_(0, labels, torch.ones_like(confidence))
                
    return confidence_sum / class_counts.clamp(min=1)


# --- 2. Server Side: Dual-Teacher Aggregation & Fusion (Algorithm 1 & Eq. 2) ---
def aggregate(client_results, client_sizes, global_model, super_teacher, auxiliary_loader, num_classes=10, lr=0.01):
    """
    Direct Server-Side Aggregation (FedAvg) and Dual-Teacher Fusion (Eq. 2)
    """
    states = [r["state_dict"] for r in client_results]
    sizes = client_sizes
    
    new_global_weights = weighted_average(states, sizes)
    global_model.load_state_dict(new_global_weights)
    
    device = next(global_model.parameters()).device
    global_model.to(device)
    
    super_teacher = super_teacher.to(device)
    super_teacher.eval()
    
    v_s = evaluate_class_confidence(super_teacher, auxiliary_loader, num_classes)
    v_t = evaluate_class_confidence(global_model, auxiliary_loader, num_classes)
    sequence = (v_s > v_t).long()
    
    global_model.train()
    optimizer = torch.optim.SGD(global_model.parameters(), lr=lr)
    
    for images, labels in auxiliary_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        logits_global_train = global_model(images)
        
        with torch.no_grad():
            logits_super = super_teacher(images)
            global_model.eval()
            logits_global_target = global_model(images)
            global_model.train()
            
        p_1_t = F.log_softmax(logits_global_train, dim=1)
        p_super = F.softmax(logits_super, dim=1)
        p_global = F.softmax(logits_global_target, dim=1)
        
        lam = sequence[labels].unsqueeze(1).float()
        p_target = (1.0 - lam) * p_global + lam * p_super
        
        loss = F.kl_div(p_1_t, p_target, reduction='batchmean')
        loss.backward()
        optimizer.step()
        
    global_model.eval()
    
    decoupled_global = {k: v.cpu().detach().clone() for k, v in global_model.state_dict().items()}
    return decoupled_global, decoupled_global
