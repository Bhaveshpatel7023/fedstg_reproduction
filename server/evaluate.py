import torch

from torch.utils.data import DataLoader


def evaluate(model, dataset, batch_size=128, device="cpu"):
    loader = DataLoader(dataset, batch_size=batch_size)

    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x,y = x.to(device), y.to(device)

            output = model(x)
            predictions = output.argmax(dim=1)

            correct += (predictions == y).sum().item()
            total += y.size(0)

    return correct / total
