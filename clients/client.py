import torch
import config
import methods.fedstg as fedstg
import losses.loss as lossfun

class Client:
    def __init__(self, model, dataset, device="cpu"):
        self.device = device
        # Force the model onto the correct hardware execution device immediately at instantiation
        self.model = model.to(self.device)
        self.dataset = dataset
        self.train_dataset = dataset

    def train(self, global_model=None, teacher=None):
        epochs = 5
        batch_size = 32
        lr = 0.01

        # Synchronize active hardware processing device directly from the incoming model state
        if global_model is not None:
            self.device = next(global_model.parameters()).device
            
        # Ensure the structural parameters are mapped cleanly to the active processing device
        self.model = self.model.to(self.device)
        
        if teacher is not None:
            teacher = teacher.to(self.device)

        # INITIALIZE OPTIMIZER: Placed cleanly after all device transfers are finalized
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-5
        )

        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

        self.model.train()
        local_steps = 0

        for _ in range(epochs):
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(x)

                # Custom cross-entropy and masked distillation calculations
                loss = lossfun.compute_loss(
                    client=self,
                    inputs=x,
                    labels=y,
                    logits=logits,
                    global_model=global_model,
                    teacher=teacher
                )

                loss.backward()
                optimizer.step()
                local_steps += 1

        # Force state dictionary clone on cpu to decouple weights cleanly for server transmission
        cpu_state_dict = {
            key: value.cpu().detach().clone() 
            for key, value in self.model.state_dict().items()
        }

        return {
            "state_dict": cpu_state_dict,
            "num_samples": len(self.dataset),
            "local_steps": local_steps,
        }
