
# config.py
# Central configuration for FedSTG reproduction

# ==================== Experiment ====================
DATASET_NAME = "cifar10" # mnist, cifar10, cifar100, cinic10
METHOD_NAME = "fedstg"

# ==================== Federated Learning ====================
NUM_CLIENTS = 100
CLIENTS_PER_ROUND = 10
DIRICHLET_ALPHA = 0.1
NUM_ROUNDS = 106

# ==================== Local Training ====================
LOCAL_EPOCHS = 5
BATCH_SIZE = 29

# ==================== Optimizer ====================
LEARNING_RATE = 0.01
LR_DECAY = 0.99
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-5

# ==================== FedSTG ====================
TEMPERATURE = 1.0
BETA = 1.0

# ==================== Auxiliary Dataset ====================
AUXILIARY_SAMPLES_PER_CLASS = 64

# ==================== Dataset Information ====================
DATASET_CONFIG = {
    "mnist": {
        "num_classes": 10,
        "input_channels": 1,
        "target_accuracy": 0.90,
        "num_clients": 100,
        "clients_per_round": 10,
        "alpha": 0.1,
        "num_rounds": 22,
        "auxiliary_samples_per_class": 64,
    },
    "cifar10": {
        "num_classes": 10,
        "input_channels": 3,
        "target_accuracy": 0.60,
        "num_clients": 100,
        "clients_per_round": 10,
        "alpha": 0.1,
        "num_rounds": 106,
        "auxiliary_samples_per_class": 64,
    },
    "cifar100": {
        "num_classes": 100,
        "input_channels": 3,
        "target_accuracy": 0.25,
        "num_clients": 100,
        "clients_per_round": 10,
        "alpha": 0.1,
        "num_rounds": 53,
        "auxiliary_samples_per_class": 10,
    },
    "cinic10": {
        "num_classes": 10,
        "input_channels": 3,
        "target_accuracy": 0.35,
        "num_clients": 200,
        "clients_per_round": 10,
        "alpha": 0.1,
        "num_rounds": 29,
        "auxiliary_samples_per_class": 64,
    },
}


def get_config(dataset_name=None):
    """Return the active configuration for the selected dataset."""
    name = (dataset_name or DATASET_NAME).lower()

    if name not in DATASET_CONFIG:
        raise ValueError(
            f"Unsupported dataset: {name}. "
            f"Choose from: {list(DATASET_CONFIG.keys())}"
        )

    return DATASET_CONFIG[name].copy()