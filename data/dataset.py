
import torchvision
import torchvision.transforms as transforms


def get_dataset(name=None, root="./data"):
    import config

    if name is None:
        name = config.DATASET_NAME

    name = name.lower()

    if name == "mnist":

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = torchvision.datasets.MNIST(
            root=root,
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=root,
            train=False,
            download=True,
            transform=transform
        )

    elif name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616)
            )
        ])

        train_dataset = torchvision.datasets.CIFAR10(
            root="./data/CIFAR10",
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="./data/CIFAR10",
            train=False,
            download=True,
            transform=transform
        )

    elif name == "cifar100":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761)
            )
        ])

        train_dataset = torchvision.datasets.CIFAR100(
            root="./data/CIFAR100",
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root="./data/CIFAR100",
            train=False,
            download=True,
            transform=transform
        )

    elif name == "cinic10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4789, 0.4723, 0.4305),
                (0.2421, 0.2383, 0.2587)
            )
        ])
        
        train_dataset = torchvision.datasets.ImageFolder(
            root=f"{root}/CINIC10/train",
            transform=transform
        )
        test_dataset = torchvision.datasets.ImageFolder(
            root=f"{root}/CINIC10/test",
            transform=transform
        )

    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return train_dataset, test_dataset