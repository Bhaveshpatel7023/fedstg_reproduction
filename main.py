from data.dataset import get_dataset
from data.partition import dirichlet_partition
from torch.utils.data import Subset
from models.cnn import CNN
from clients.client import Client
from server.server import Server
from server.evaluate import evaluate
import random
import torch
from data.auxiliary import get_auxiliary_dataset
from torch.utils.data import Subset
import csv
import os
import config
import time

torch.set_num_threads(4)
torch.set_num_interop_threads(1)


random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CFG = config.get_config()
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

NUM_CLIENTS = CFG["num_clients"]
NUM_ROUNDS = CFG["num_rounds"]
CLIENTS_PER_ROUND = CFG["clients_per_round"]
ALPHA = CFG["alpha"]

def main(method_name=None):
    if method_name is not None:
        config.METHOD_NAME = method_name

    experiment_start = time.perf_counter()
    target_accuracy = CFG["target_accuracy"]
    target_round = None

    train_dataset, original_test_dataset = get_dataset(config.DATASET_NAME)

    # 64 samples per class from the original test set
    auxiliary_dataset = get_auxiliary_dataset(
        original_test_dataset,
        samples_per_class=CFG["auxiliary_samples_per_class"]
    )

    # Remaining test samples are used only for evaluation
    auxiliary_indices = set(auxiliary_dataset.indices)
    test_indices = [
        i for i in range(len(original_test_dataset))
        if i not in auxiliary_indices
    ]
    test_dataset = Subset(original_test_dataset, test_indices)

    # Use the complete training set for client partitioning
    partitions = dirichlet_partition(
        train_dataset,
        NUM_CLIENTS,
        ALPHA
    )

    print("Auxiliary dataset size:", len(auxiliary_dataset))
    print("Client training dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))

    server = Server(
        CNN().to(DEVICE),
        auxiliary_dataset=auxiliary_dataset,
        total_rounds=NUM_ROUNDS
    )   
    

    clients = [
        Client(CNN(), Subset(train_dataset, partitions[i]))
        for i in range(NUM_CLIENTS)
    ]

    os.makedirs("results", exist_ok=True)
    result_file = f"results/{config.DATASET_NAME.lower()}_{config.METHOD_NAME.lower()}.csv"

    with open(result_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["round", "accuracy", "round_time_seconds"])

        
        for round_num in range(1, NUM_ROUNDS + 1):
            round_start = time.perf_counter()

            selected = random.sample(clients, CLIENTS_PER_ROUND)
            server.run_round(selected)

            accuracy = evaluate(server.model, test_dataset, device=DEVICE)
            round_time = time.perf_counter() - round_start

            if target_round is None and accuracy >= target_accuracy:
                target_round = round_num
                print(
                    f"Target accuracy {target_accuracy:.0%} "
                    f"reached at round {target_round}"
                )

            writer.writerow([round_num, accuracy, round_time])
            file.flush()

            print(
                f"Round {round_num}/{NUM_ROUNDS} "
                f"- Accuracy: {accuracy:.4f} "
                f"- Time: {round_time:.2f}s"
            )

    total_time = time.perf_counter() - experiment_start

    print("\n===== EXPERIMENT SUMMARY =====")
    print(f"Dataset: {config.DATASET_NAME}")
    print(f"Target accuracy: {target_accuracy:.2%}")
    print(f"Final accuracy: {accuracy:.4f}")
    print(f"Total experiment time: {total_time:.2f} seconds")
    print(f"Super Teacher construction time (last round): {server.super_teacher_generation_time:.6f} seconds")
    print(f"Results saved to: {result_file}")

if __name__ == "__main__":
    main()