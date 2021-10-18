import os
import argparse
import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset

from utils import seed_all
from noises import noisify
from datasets.cifar100 import sparse2coarse


class Subset(Dataset):
    def __init__(self, dataset, indices, dataset_name, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.dataset_name = dataset_name
        self.transform = transform

        if self.dataset_name == "deepmind-cifar100":
            self.coarses = np.array(self.dataset.coarses)[self.indices]
        else:
            self.coarses = np.array(self.dataset.targets)[self.indices]
            if self.dataset_name == "cifar100":
                self.coarses = sparse2coarse(self.coarses)

        self.classes = self.dataset.classes
        self.coarse_classes = self.dataset.coarse_classes
        self.targets = np.array(self.dataset.targets)[self.indices]

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform:
            x = self.transform(x)
        return x, y, idx

    def __len__(self):
        return len(self.indices)


class IndicesSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        indices: List of index
        Shuffle: Everyday I'm shuffling
    """

    def __init__(self, indices, shuffle):
        self.indices = indices
        if shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def filter_loader(train_loader, indices):
    sampler = IndicesSampler(indices, True)
    selected_dataloader = DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        num_workers=train_loader.num_workers,
        sampler=sampler,
    )
    return selected_dataloader


def flip_label(
    dataset_name,
    noise_ratio,
    noise_type,
    train_dataset,
    noise_label_path,
    noise_ind_path,
    seed,
):
    if os.path.exists(noise_label_path) and os.path.exists(noise_ind_path):
        train_dataset.targets = np.load(noise_label_path)
        noise_ind = np.load(noise_ind_path)
    else:
        labels = np.asarray(
            [[train_dataset.targets[i]] for i in range(len(train_dataset.targets))]
        )
        noisy_labels, P = noisify(
            dataset_name,
            len(train_dataset.classes),
            labels,
            noise_type,
            noise_ratio,
            seed,
        )
        noisy_labels = np.array([i[0] for i in noisy_labels])
        noise_ind = np.where(np.array(train_dataset.targets) != noisy_labels)[0]
        train_dataset.targets = list(noisy_labels)
        np.save(noise_label_path, noisy_labels)
        np.save(noise_ind_path, noise_ind)

    return train_dataset, noise_ind


def divide_train(
    train_ratio, train_dataset, noise_ind, train_subset_path, valid_subset_path, dataset_name, train_transform, test_transform
):
    if os.path.exists(train_subset_path) and os.path.join(valid_subset_path):
        train_subset = Subset(train_dataset, np.load(train_subset_path), dataset_name, train_transform)
        valid_subset = Subset(train_dataset, np.load(valid_subset_path), dataset_name, test_transform)
    else:
        train_size = int(len(train_dataset) * train_ratio)
        val_size = len(train_dataset) - train_size

        indices = torch.randperm(train_size + val_size, generator=torch.default_generator).tolist()
        train_subset = Subset(train_dataset, indices[:train_size], dataset_name, train_transform)
        valid_subset = Subset(train_dataset, indices[train_size:], dataset_name, test_transform)
        np.save(train_subset_path, train_subset.indices)
        np.save(valid_subset_path, valid_subset.indices)
    train_noise_ind = np.where(np.in1d(train_subset.indices, noise_ind))[0]
    return train_subset, valid_subset, train_noise_ind


def load_datasets(
    dataset_name,
    root,
    train_ratio,
    noise_type,
    noise_ratio,
    seed,
):
    noise_ind = None
    if dataset_name == "mnist":
        from datasets.mnist import get_mnist

        (
            train_dataset,
            valid_dataset,
            test_dataset,
            train_transform,
            test_transform,
        ) = get_mnist(root)
    elif dataset_name == "cifar10":
        from datasets.cifar10 import get_cifar10

        (
            train_dataset,
            valid_dataset,
            test_dataset,
            train_transform,
            test_transform,
        ) = get_cifar10(root)
    elif dataset_name == "cifar100":
        from datasets.cifar100 import get_cifar100

        (
            train_dataset,
            valid_dataset,
            test_dataset,
            train_transform,
            test_transform,
        ) = get_cifar100(root)
    elif dataset_name == "deepmind-cifar10":
        from datasets.deepmind import get_cifar10

        (
            train_dataset,
            valid_dataset,
            test_dataset,
            train_transform,
            test_transform,
        ) = get_cifar10(root, noise_level="low")
        noise_ind = np.where(
            np.array(train_dataset.clean_labels) != np.array(train_dataset.noisy_labels)
        )[0]
    elif dataset_name == "deepmind-cifar100":
        from datasets.deepmind import get_cifar100

        (
            train_dataset,
            valid_dataset,
            test_dataset,
            train_transform,
            test_transform,
        ) = get_cifar100(root, noise_level="low")
        noise_ind = np.where(
            np.array(train_dataset.clean_labels) != np.array(train_dataset.noisy_labels)
        )[0]
    elif dataset_name == "tiny-imagenet":
        from datasets.tinyimagenet import get_tinyimagenet

        (
            train_dataset,
            valid_dataset,
            test_dataset,
            train_transform,
            test_transform,
        ) = get_tinyimagenet(root)
    elif dataset_name == "clothing1m":
        from datasets.clothing1m import get_clothing1m

        (
            train_dataset,
            valid_dataset,
            test_dataset,
            train_transform,
            test_transform,
        ) = get_clothing1m(root)
        noise_ind = []
    else:
        raise NameError(f"Invalid dataset name: {dataset_name}")

    sub_dir = os.path.join("./data", f"changed_{dataset_name}_{seed}")
    os.makedirs(sub_dir, exist_ok=True)

    if noise_ind is None:
        if noise_type in ["symmetric", "asymmetric"]:
            noise_label_path = os.path.join(
                sub_dir, f"{noise_type}_{noise_ratio}_label.npy"
            )
            noise_ind_path = os.path.join(
                sub_dir, f"{noise_type}_{noise_ratio}_ind.npy"
            )
            train_dataset, noise_ind = flip_label(
                dataset_name,
                noise_ratio,
                noise_type,
                train_dataset,
                noise_label_path,
                noise_ind_path,
                seed,
            )
        else:
            raise NameError(f"Invalid noisy type: {noise_type}")

    if train_ratio < 1 and valid_dataset is None:
        train_subset_path = os.path.join(sub_dir, f"{train_ratio}_train_subset.npy")
        valid_subset_path = os.path.join(sub_dir, f"{train_ratio}_valid_subset.npy")
        train_subset, valid_subset, train_noise_ind = divide_train(
            train_ratio, train_dataset, noise_ind, train_subset_path, valid_subset_path, dataset_name, train_transform, test_transform
        )
    else:
        train_subset = Subset(train_dataset, list(range(len(train_dataset))), dataset_name, train_transform)
        if valid_dataset is None:
            valid_subset = None
        else:
            train_subset = Subset(valid_dataset, list(range(len(valid_dataset))), dataset_name, test_transform)
        train_noise_ind = noise_ind
    test_subset = Subset(test_dataset, list(range(len(test_dataset))), dataset_name, test_transform)

    return (
        train_subset,
        valid_subset,
        test_subset,
        train_noise_ind,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dataset_name", type=str, default="mnist")
    parser.add_argument("--dataset_path", type=str, default="data")
    parser.add_argument("--train_ratio", type=float, default=1.0)
    parser.add_argument("--noise_type", type=str, default="symmetric")
    parser.add_argument("--noise_ratio", type=float, default=0.2)

    args = parser.parse_args()

    seed_all(args.seed)

    train_dataset, valid_dataset, test_dataset, train_noise_ind = load_datasets(
        args.dataset_name,
        args.dataset_path,
        args.train_ratio,
        args.noise_type,
        args.noise_ratio,
        args.seed,
    )
