import numpy as np


def calculate_forgetting(accuracy_history):
    """
    Calculate Forgetting Measure (F).

    accuracy_history:
        Shape = (communication_rounds, num_classes)
        Values should be between 0 and 1.
    """

    accuracy_history = np.array(accuracy_history)

    if accuracy_history.ndim != 2:
        raise ValueError("accuracy_history must be a 2D array.")

    if accuracy_history.shape[0] < 2:
        raise ValueError("At least 2 communication rounds are required.")

    max_previous_accuracy = np.max(
        accuracy_history[:-1],
        axis=0
    )

    final_accuracy = accuracy_history[-1]

    forgetting = np.mean(
        max_previous_accuracy - final_accuracy
    )

    return float(forgetting)