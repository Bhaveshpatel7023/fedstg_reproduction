
from collections import deque

from server.aggregation import fedAvg
import torch
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Subset
import config
import time


class Server:
    def __init__(
        self,
        model,
        auxiliary_dataset=None,
        history_size=5,
        total_rounds=None
    ):
        if total_rounds is None:
            total_rounds = config.get_config()["num_rounds"]
        self.model = model
        self.auxiliary_dataset = auxiliary_dataset
        self.history = deque(maxlen=history_size)
        self.total_rounds = total_rounds
        self.current_round = 0
        self.super_teacher_generation_time = 0.0
        self.super_teacher_generation_times = []

        self.super_teacher_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    def get_global_state(self):
        return self.model.state_dict()

    def set_global_state(self, state_dict):
        self.model.load_state_dict(state_dict)

    
    def evaluate_on_auxiliary(self, model):
        if self.auxiliary_dataset is None:
            return None

        loader = DataLoader(
            self.auxiliary_dataset,
            batch_size=128,
            shuffle=False
        )

        model.eval()
        num_classes = config.get_config()["num_classes"]
        correct_by_class = torch.zeros(num_classes)
        total_by_class = torch.zeros(num_classes)

        device = next(model.parameters()).device

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                predictions = model(x).argmax(dim=1)

                for label in range(10):
                    mask = y == label
                    total_by_class[label] += mask.sum().item()
                    correct_by_class[label] += (
                        (predictions == y) & mask
                    ).sum().item()

        return correct_by_class / total_by_class.clamp(min=1)


    def split_auxiliary_by_teacher(self, selection):
        if self.auxiliary_dataset is None:
            return None, None

        base_dataset = self.auxiliary_dataset.dataset
        indices = self.auxiliary_dataset.indices
        targets = getattr(base_dataset, "targets", None)

        if targets is None:
            labels = [
                int(self.auxiliary_dataset[i][1])
                for i in range(len(self.auxiliary_dataset))
            ]
        else:
            labels = [int(targets[i]) for i in indices]

        labels = torch.tensor(labels, dtype=torch.long)
        selected_labels = selection.cpu()[labels]

        global_indices = torch.where(selected_labels == 0)[0].tolist()
        teacher_indices = torch.where(selected_labels == 1)[0].tolist()

        global_data = Subset(self.auxiliary_dataset, global_indices)
        teacher_data = Subset(self.auxiliary_dataset, teacher_indices)

        return global_data, teacher_data
    
    
    def evaluate_class_confidence(self, model):
        if self.auxiliary_dataset is None:
            return None

        loader = DataLoader(
            self.auxiliary_dataset,
            batch_size=128,
            shuffle=False
        )

        device = next(model.parameters()).device
        num_classes = config.get_config()["num_classes"]
        confidence_sum = torch.zeros(num_classes, device=device)
        count = torch.zeros(num_classes, device=device)

        model.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)

                probs = torch.softmax(model(x), dim=1)

                # Confidence: p(y|x) - sum of probabilities for other classes
                true_probs = probs.gather(1, y.unsqueeze(1)).squeeze(1)
                confidence = 2 * true_probs - 1

                confidence_sum.scatter_add_(0, y, confidence)
                count.scatter_add_(0, y, torch.ones_like(confidence))

        return (confidence_sum / count.clamp(min=1)).cpu()

    
    def select_class_teacher(self, global_model, super_teacher):
        global_scores = self.evaluate_class_confidence(global_model)
        teacher_scores = self.evaluate_class_confidence(super_teacher)

        if global_scores is None or teacher_scores is None:
            return None

        # 0 = current global model; 1 = Super Teacher
        selection = (teacher_scores > global_scores).to(torch.int64)

        return selection


    
    
    def build_super_teacher(self):
        teacher = type(self.model)()
        teacher.load_state_dict(self.super_teacher_state)
        teacher.to(next(self.model.parameters()).device)
        teacher.eval()

        for param in teacher.parameters():
            param.requires_grad = False

        return teacher


    
    
    
    def train_teacher_on_auxiliary(
        self,
        student,
        dataset,
        global_model,
        super_teacher,
        selection,
        epochs=1,
        lr=0.01
    ):
        if dataset is None or len(dataset) == 0:
            return student

        device = next(student.parameters()).device
        selection = selection.to(device)

        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True
        )

        global_model.eval()
        super_teacher.eval()
        student.train()

        optimizer = torch.optim.SGD(
            student.parameters(),
            lr=lr,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )

        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                student_logits = student(x)

                with torch.no_grad():
                    global_logits = global_model(x)
                    teacher_logits = super_teacher(x)

                student_log_probs = torch.log_softmax(
                    student_logits, dim=1
                )
                global_probs = torch.softmax(global_logits, dim=1)
                teacher_probs = torch.softmax(teacher_logits, dim=1)

                # lambda = 0 when Super Teacher wins;
                # lambda = 1 when global model wins.
                
                use_global = ~selection[y].bool()

                global_kl = torch.nn.functional.kl_div(
                    student_log_probs,
                    global_probs,
                    reduction="none"
                ).sum(dim=1)

                teacher_kl = torch.nn.functional.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="none"
                ).sum(dim=1)

                # lambda = 1 for D2 (global wins),
                # lambda = 0 for D1 (Super Teacher wins)
                lam = use_global.float()

                loss = (
                    lam * global_kl
                    + (1.0 - lam) * teacher_kl
                ).mean()

                loss.backward()
                optimizer.step()

        student.eval()
        return student
    
    
    
    
    def run_round(self, clients):
        self.current_round += 1

        from methods import get_method
        import copy

        method = config.METHOD_NAME.lower()
        algorithm = get_method(method)

        global_state = {
            key: value.detach().clone()
            for key, value in self.model.state_dict().items()
        }

        # Prepare teacher only for FedSTG
        teacher = None
        if method == "fedstg":
            teacher_start = time.perf_counter()
            teacher = self.build_super_teacher()
            elapsed = time.perf_counter() - teacher_start

            self.super_teacher_generation_time = elapsed
            self.super_teacher_generation_times.append(elapsed)
        else:
            self.super_teacher_generation_time = 0.0

        # Every client starts from the same global model
        for client in clients:
            client.model.load_state_dict(global_state)

        # Train clients using the selected algorithm

        client_start = time.perf_counter()

        client_results = [
            client.train(
                epochs=config.LOCAL_EPOCHS,
                batch_size=config.BATCH_SIZE,
                lr=config.LEARNING_RATE,
                method=method,
                global_model=self.model,
                teacher=teacher,
            )
            for client in clients
        ]

        client_sizes = [len(client.dataset) for client in clients]

        # Use the selected method's aggregation
        new_state = algorithm.aggregate(client_results, client_sizes)
        self.set_global_state(new_state)

        # Save the starting global state for history
        self.history.append(global_state)

        # Update the historical teacher for FedSTG
        if method == "fedstg" and self.current_round < self.total_rounds:
            previous_teacher = self.build_super_teacher()

            if self.auxiliary_dataset is not None:
                selection = self.select_class_teacher(
                    self.model,
                    previous_teacher,
                )

                if selection is not None:
                    student_replica = copy.deepcopy(self.model)

                    teacher_train_start = time.perf_counter()

                    student_replica = self.train_teacher_on_auxiliary(
                        student=student_replica,
                        dataset=self.auxiliary_dataset,
                        global_model=self.model,
                        super_teacher=previous_teacher,
                        selection=selection,
                        epochs=config.LOCAL_EPOCHS,
                        lr=config.LEARNING_RATE,
                    )

                    teacher_training_time = time.perf_counter() - teacher_train_start

                    self.super_teacher_state = {
                        key: value.detach().clone()
                        for key, value in student_replica.state_dict().items()
                    }

        return new_state
    