import torch
import config


class Client:
    def __init__(self, model, dataset, device="cpu"):
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.previous_model_state = None

    def train(
        self,
        epochs=None,
        batch_size=None,
        lr=None,
        method=None,
        global_model=None,
        teacher=None,
    ):
        if epochs is None:
            epochs = config.LOCAL_EPOCHS
        if batch_size is None:
            batch_size = config.BATCH_SIZE
        if lr is None:
            lr = config.LEARNING_RATE

        if method is None:
            method = config.METHOD_NAME.lower()

        from methods import get_method
        algorithm = get_method(method)

        if global_model is not None:
            self.device = next(global_model.parameters()).device
        self.model = self.model.to(self.device)

        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY,
        )

        self.model.train()
        local_steps = 0

        for _ in range(epochs):
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()

                logits = self.model(x)

                loss = algorithm.compute_loss(
                    client=self,
                    inputs=x,
                    labels=y,
                    logits=logits,
                    global_model=global_model,
                    teacher=teacher,
                )

                loss.backward()
                optimizer.step()
                local_steps += 1

        self.previous_model_state = {
            key: value.detach().clone()
            for key, value in self.model.state_dict().items()
        }

        self.model = self.model.to("cpu")

        return {
            "state_dict": {
                key: value.detach().clone()
                for key, value in self.model.state_dict().items()
            },
            "num_samples": len(self.dataset),
            "local_steps": local_steps,
        }