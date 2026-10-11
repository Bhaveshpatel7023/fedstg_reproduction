
import numpy as np
from torch.utils.data import Subset


def dirichlet_partition(dataset, num_clients, alpha=0.1, seed=42):
    rng = np.random.default_rng(seed)

    if isinstance(dataset, Subset):
        targets = np.array(dataset.dataset.targets)[
            np.asarray(dataset.indices)
        ]
    else:
        targets = np.array(dataset.targets)

    num_classes = len(np.unique(targets))
    clients_indices = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        class_indices = np.where(targets == class_id)[0]
        rng.shuffle(class_indices)

        proportions = rng.dirichlet(
            np.repeat(alpha, num_clients)
        )

        split_points = (
            np.cumsum(proportions)[:-1] * len(class_indices)
        ).astype(int)

        split_indices = np.split(class_indices, split_points)

        for client_id, indices in enumerate(split_indices):
            clients_indices[client_id].extend(indices.tolist())

    return clients_indices