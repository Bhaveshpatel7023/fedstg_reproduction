import csv
import os
import random
import time
import torch
from torch.utils.data import Subset
import config
from clients.client import Client
from data.auxiliary import get_auxiliary_dataset
from data.dataset import get_dataset
from data.partition import dirichlet_partition
from models.cnn import CNN
from methods.common import evaluate
from server.server import Server

# Optimize thread allocation performance for machine learning processing
torch.set_num_threads(8)
torch.set_num_interop_threads(2)

# Set global deterministic random seeds for rigorous reproducibility
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Fetch environment parameters from configuration module
CFG = config.get_config()

# Set device allocation dynamically based on the system hardware availability
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

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
    accuracy = 0.0  # Initialize variable to handle logging gracefully

    # Load datasets based on the unified training framework requirements
    train_dataset, original_test_dataset = get_dataset(config.DATASET_NAME)

    # Allocate samples per class to compose the server validation partition (D_u)
    auxiliary_dataset = get_auxiliary_dataset(
        original_test_dataset,
        samples_per_class=CFG["auxiliary_samples_per_class"]
    )

    # Segregate remaining instances into an evaluation validation dataset split
    auxiliary_indices = set(auxiliary_dataset.indices)
    test_indices = [
        i for i in range(len(original_test_dataset))
        if i not in auxiliary_indices
    ]
    test_dataset = Subset(original_test_dataset, test_indices)

    # Perform partition allocation using the non-IID Dirichlet distribution logic
    partitions = dirichlet_partition(
        train_dataset,
        NUM_CLIENTS,
        ALPHA
    )

    print(f"Server Auxiliary Dataset (D_u) Size: {len(auxiliary_dataset)}")
    print(f"Total Client Training Samples Pool: {len(train_dataset)}")
    print(f"Isolated Test Evaluation Split Size: {len(test_dataset)}")

    # Instantiate the centralized parameter server on the correct execution device
    server = Server(
        model=CNN().to(DEVICE),
        auxiliary_dataset=auxiliary_dataset,
        total_rounds=NUM_ROUNDS
    )   

    # Instantiate the client pool with local models pushed onto the targeted device configuration
    clients = [
        Client(CNN().to(DEVICE), Subset(train_dataset, partitions[i]))
        for i in range(NUM_CLIENTS)
    ]

    # Initialize results directories and tracking log files
    os.makedirs("results", exist_ok=True)
    result_file = f"results/{config.DATASET_NAME.lower()}_{config.METHOD_NAME.lower()}.csv"

    with open(result_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["round", "accuracy", "round_time_seconds"])

        # Execute training cycles across all federation execution rounds
        for round_num in range(1, NUM_ROUNDS + 1):
            round_start = time.perf_counter()

            # Uniform random sampling selection of clients for the communication round
            selected_clients = random.sample(clients, CLIENTS_PER_ROUND)
            
            # --- FIXED STATE SYNCHRONIZATION PASS ---
            # Access the server model state dictionary directly to bypass the missing wrapper attribute
            current_global_state = server.model.state_dict()
            
            # Synchronize each selected client model explicitly before training starts
            for client in selected_clients:
                client.model.load_state_dict(current_global_state)
            
            # Execute standard communication cycle round on the selected subsets
            server.run_round(selected_clients)

            # --- FIXED METRIC EVALUATION AND LOGGING LOGIC ---
            # Compute current global generalization capability scores against test split dataset
            accuracy = evaluate(server.model, test_dataset, device=DEVICE)
            round_time = time.perf_counter() - round_start

            # Track milestone convergence requirements against metric targets
            if target_round is None and accuracy >= target_accuracy:
                target_round = round_num
                print(
                    f"\nTarget accuracy checkpoint of {target_accuracy:.2%} "
                    f"successfully achieved at round {target_round}!"
                )

            # Record metrics to csv tracking file log
            writer.writerow([round_num, accuracy, round_time])
            file.flush()

            print(
                f"Round {round_num}/{NUM_ROUNDS} "
                f"- Generalization Accuracy: {accuracy:.4f} "
                f"- Duration: {round_time:.2f}s"
            )

    total_time = time.perf_counter() - experiment_start

    print("\n===== FEDERATED PIPELINE EXPERIMENT SUMMARY =====")
    print(f"Dataset Name: {config.DATASET_NAME}")
    print(f"Target Threshold: {target_accuracy:.2%}")
    print(f"Final Convergence Metric: {accuracy:.4f}")
    print(f"Total Experiment Time: {total_time:.2f} seconds")
    print(f"Super Teacher Generation Duration (Last Round): {server.super_teacher_generation_time:.6f} seconds")
    print(f"Tracking Statistics Output Location: {result_file}")

if __name__ == "__main__":
    main()
