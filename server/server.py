import copy
import time
import torch
from torch.utils.data import DataLoader
import config
import methods.fedstg as fedstg

class Server:
    def __init__(self, model, auxiliary_dataset=None, total_rounds=None):
        self.model = model
        self.auxiliary_dataset = auxiliary_dataset
        self.current_round = 0
        self.super_teacher_generation_time = 0.0
        
        # Lock initial Super Teacher state weights array parameters explicitly
        self.super_teacher_state = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

    def build_super_teacher(self):
        """
        Instantiates a deep copy model to serve as the structural Super Teacher.
        """
        teacher = copy.deepcopy(self.model)
        teacher.load_state_dict(self.super_teacher_state)
        teacher.to(next(self.model.parameters()).device)
        teacher.eval()
        return teacher

    def run_round(self, clients):
        """
        Executes a single communication round of Federated Learning (Algorithm 1).
        """
        self.current_round += 1
        device = next(self.model.parameters()).device

        # Explicitly clone the master server weights state dictionary mapping
        global_state = {
            key: value.detach().clone() for key, value in self.model.state_dict().items()
        }

        # Build and track the historical reference Super Teacher instance
        teacher_start = time.perf_counter()
        super_teacher_model = self.build_super_teacher()
        self.super_teacher_generation_time = time.perf_counter() - teacher_start

        client_results = []
        client_sizes = []

        # Run client local training processes
        for client in clients:
            client.device = device
            # Synchronize client model with current global parameters
            client.model.load_state_dict(global_state)
            
            # Execute targeted optimization loop pass directly
            client_output = client.train(global_model=self.model, teacher=super_teacher_model)
            
            # CRITICAL STATE DICTIONARY STRUCTURE TRACKING FORMAT ALIGNMENT FIX
            client_results.append({
                "state_dict": client_output["state_dict"],
                "num_samples": client_output["num_samples"]
            })
            client_sizes.append(client_output["num_samples"])

        # Create validation dataset stream interface container
        aux_loader = DataLoader(
            self.auxiliary_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        num_classes = config.get_config()["num_classes"]
        
        # Direct functional call to aggregate pipeline matrix variables
        new_global, new_super = fedstg.aggregate(
            client_results=client_results,
            client_sizes=client_sizes,
            global_model=self.model,
            super_teacher=super_teacher_model,
            auxiliary_loader=aux_loader,
            num_classes=num_classes,
            lr=0.01
        )

        # Force persistent weight updates back onto the central server tracking variables
        self.model.load_state_dict(new_global)
        self.super_teacher_state = new_super

