from torch.utils.data import Subset
from collections import defaultdict
import random


def get_auxiliary_dataset(dataset, samples_per_class=64):
    class_indices = defaultdict(list)

    # Use labels directly when the dataset exposes them.
    targets = getattr(dataset, "targets", None)

    if targets is not None:
        for index, label in enumerate(targets):
            class_indices[int(label)].append(index)
    else:
        for index in range(len(dataset)):
            _, label = dataset[index]
            class_indices[int(label)].append(index)

    rng = random.Random(42)
    selected_indices = []

    for label in sorted(class_indices):
        indices = class_indices[label].copy()
        rng.shuffle(indices)
        selected_indices.extend(indices[:samples_per_class])

    return Subset(dataset, selected_indices)