import utils

import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split


def use_partial_dataloaders(dataloaders, percent=None, count=None, mode=None):
    train_dataset = (
        dataloaders["validate"].dataset if mode else dataloaders["train"].dataset
    )

    partial_train_dataset_count = 0
    if percent:
        partial_train_dataset_count = int(len(train_dataset) * percent / 100)
    elif count:
        partial_train_dataset_count = count

    partial_train_dataset, _ = random_split(
        train_dataset,
        [
            partial_train_dataset_count,
            len(train_dataset) - partial_train_dataset_count,
        ],
        torch.manual_seed(0),
    )  # split the dataset definitely, because we want to get a stable result

    dataloaders["train"] = DataLoader(
        partial_train_dataset, batch_size=dataloaders["train"].batch_size, shuffle=True
    )


def get_MNIST_dataloader(use_train_set_percent=100):
    data_folder = "../data"
    validate_ratio_in_testset = 0.5

    train_dataset = torchvision.datasets.MNIST(
        root=data_folder, train=True, transform=transforms.ToTensor(), download=True
    )
    train_dataset_count = int(len(train_dataset) * use_train_set_percent / 100)
    train_dataset, _ = random_split(
        train_dataset,
        [
            train_dataset_count,
            len(train_dataset) - train_dataset_count,
        ],
        torch.manual_seed(0),
    )  # split the dataset definitely, because we want to get a stable result

    total_test_dataset = torchvision.datasets.MNIST(
        root=data_folder, train=False, transform=transforms.ToTensor(), download=True
    )
    validate_dataset_count = int(len(total_test_dataset) * validate_ratio_in_testset)
    validate_dataset, test_dataset = random_split(
        total_test_dataset,
        [validate_dataset_count, len(total_test_dataset) - validate_dataset_count],
        torch.manual_seed(0),
    )  # split the dataset definitely, because we want to get a stable result"""

    dataloaders = {
        "name": "MNIST",
        "epoch": 128,
        "train": DataLoader(train_dataset, batch_size=128, shuffle=True),
        "validate": DataLoader(
            validate_dataset, batch_size=len(validate_dataset), shuffle=True
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=len(test_dataset),
            shuffle=True,
            pin_memory=True,
        ),
    }

    return dataloaders


def get_EMNIST_dataloader(use_train_set_percent=100):
    data_folder = "../data"
    validate_ratio_in_testset = 0.5

    train_dataset = torchvision.datasets.EMNIST(
        root=data_folder,
        train=True,
        transform=transforms.ToTensor(),
        download=True,
        split="letters",
    )
    train_dataset_count = int(len(train_dataset) * use_train_set_percent / 100)
    train_dataset, _ = random_split(
        train_dataset,
        [
            train_dataset_count,
            len(train_dataset) - train_dataset_count,
        ],
        torch.manual_seed(0),
    )  # split the dataset definitely, because we want to get a stable result

    total_test_dataset = torchvision.datasets.EMNIST(
        root=data_folder,
        train=False,
        transform=transforms.ToTensor(),
        download=True,
        split="letters",
    )
    validate_dataset_count = int(len(total_test_dataset) * validate_ratio_in_testset)
    validate_dataset, test_dataset = random_split(
        total_test_dataset,
        [validate_dataset_count, len(total_test_dataset) - validate_dataset_count],
        torch.manual_seed(0),
    )  # split the dataset definitely, because we want to get a stable result"""

    batch_size = 128
    dataloaders = {
        "name": "EMNIST",
        "epoch": 128,
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "validate": DataLoader(
            validate_dataset, batch_size=len(validate_dataset), shuffle=True
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=int(len(validate_dataset) / 2),  # limit of 8192 from SEAL
            shuffle=False,
        ),
    }

    return dataloaders


def get_GTSRB_dataloader(use_train_set_percent=100):
    data_folder = "../data"
    validate_ratio_in_testset = 0.5

    train_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = torchvision.datasets.GTSRB(
        root=data_folder, split="train", transform=train_transform, download=True
    )
    train_dataset_count = int(len(train_dataset) * use_train_set_percent / 100)
    train_dataset, _ = random_split(
        train_dataset,
        [
            train_dataset_count,
            len(train_dataset) - train_dataset_count,
        ],
        torch.manual_seed(0),
    )  # split the dataset definitely, because we want to get a stable result

    test_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    total_test_dataset = torchvision.datasets.GTSRB(
        root=data_folder, split="test", transform=test_transform, download=True
    )
    validate_dataset_count = int(len(total_test_dataset) * validate_ratio_in_testset)
    validate_dataset, test_dataset = random_split(
        total_test_dataset,
        [validate_dataset_count, len(total_test_dataset) - validate_dataset_count],
        torch.manual_seed(0),
    )  # split the dataset definitely, because we want to get a stable result"""

    dataloaders = {
        "name": "GTSRB",
        "epoch": 128,
        "train": DataLoader(train_dataset, batch_size=128, shuffle=True),
        "validate": DataLoader(
            validate_dataset, batch_size=len(validate_dataset), shuffle=True
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=len(test_dataset),
            shuffle=False,
        ),
    }

    return dataloaders


def get_CIFAR10_dataloader(use_train_set_percent=100):
    data_folder = "../data"
    validate_ratio_in_testset = 0.5

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_folder, train=True, transform=train_transform, download=True
    )
    train_dataset_count = int(len(train_dataset) * use_train_set_percent / 100)
    train_dataset, _ = random_split(
        train_dataset,
        [
            train_dataset_count,
            len(train_dataset) - train_dataset_count,
        ],
        torch.manual_seed(0),
    )  # split the dataset definitely, because we want to get a stable result

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    total_test_dataset = torchvision.datasets.CIFAR10(
        root=data_folder, train=False, transform=test_transform, download=False
    )
    validate_dataset_count = int(len(total_test_dataset) * validate_ratio_in_testset)
    validate_dataset, test_dataset = random_split(
        total_test_dataset,
        [validate_dataset_count, len(total_test_dataset) - validate_dataset_count],
        torch.manual_seed(0),
    )  # split the dataset definitely, because we want to get a stable result"""

    dataloaders = {
        "name": "CIFAR10",
        "epoch": 128,
        "train": DataLoader(train_dataset, batch_size=128, shuffle=True),
        "validate": DataLoader(
            validate_dataset, batch_size=len(validate_dataset), shuffle=True
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=len(test_dataset),
            shuffle=False,
        ),
    }

    return dataloaders


def get_TinyImageNet_dataloader(use_train_set_percent=100):
    source_path = f"../data/tiny-imagenet-200"

    train_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(
        os.path.join(source_path, "train"), train_transform
    )

    test_transform = transforms.Compose([transforms.ToTensor()])
    validate_dataset = datasets.ImageFolder(
        os.path.join(source_path, "validate"), test_transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(source_path, "test"), test_transform
    )

    dataloaders = {
        "name": "TinyImageNet",
        "epoch": 128,
        "train": DataLoader(train_dataset, batch_size=128, shuffle=True),
        "validate": DataLoader(validate_dataset, batch_size=128, shuffle=True),
        "test": DataLoader(test_dataset, batch_size=2500, shuffle=False),
    }

    return dataloaders
