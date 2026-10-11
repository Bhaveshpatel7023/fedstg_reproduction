
import torch.nn as nn
import config


class CNN(nn.Module):
    def __init__(self, num_classes=None, input_channels=None):
        super().__init__()

        # Load active dataset settings from config.py
        cfg = config.get_config()

        self.dataset_name = config.DATASET_NAME.lower()

        if num_classes is None:
            num_classes = cfg["num_classes"]

        if input_channels is None:
            input_channels = cfg["input_channels"]

        self.features = nn.Sequential(
            nn.Conv2d(
                input_channels, 32,
                kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                32, 64,
                kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate flattened feature size based on configured image size
        image_size = 28 if self.dataset_name == "mnist" else 32
        feature_size = image_size // 4
        flattened_size = 64 * feature_size * feature_size

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)