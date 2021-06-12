import os
import argparse
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import random_split, DataLoader, Dataset, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils import noisify, seed_all


class DatasetWithIndex(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.classes = self.dataset.classes

    def __getitem__(self, index):
        x, y = self.dataset[index]
        return x, y, index

    def __len__(self):
        return len(self.dataset)


class DatasetFromSubset(Dataset):
    """https://discuss.pytorch.org/t/torch-utils-data-dataset-random-split/32209/4"""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

        if isinstance(self.subset, Subset):
            self.classes = self.subset.dataset.classes
        else:
            self.classes = self.subset.classes

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


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


class TinyImageNet(Dataset):
    """from https://github.com/leemengtaiwan/tiny-imagenet
    Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.
    """

    def __init__(
        self, root, transform=None, mode="train",
    ):
        self.root = os.path.join(root, "tiny-imagenet")
        self.mode = mode
        self.transform = transform

        self.images = []
        self.targets = []

        # build class label - number mapping
        self.classes = []
        self.cls2idx = {}
        with open(os.path.join(self.root, "wnids.txt"), "r") as fp:
            for i, text in enumerate(fp.readlines()):
                num = text.strip("\n")
                self.classes.append(num)
                self.cls2idx[num] = i

        if self.mode == "train":
            self.image_paths = sorted(
                glob.iglob(
                    os.path.join(self.root, "train", "**", "*.JPEG"), recursive=True
                )
            )
            for img_path in self.image_paths:
                file_name = os.path.basename(img_path)
                label_text = file_name.split("_")[0]
                self.targets.append(self.cls2idx[label_text])

        elif self.mode == "test":
            self.image_paths = []
            with open(os.path.join(self.root, "val", "val_annotations.txt"), "r") as fp:
                for line in fp.readlines():
                    terms = line.split("\t")
                    file_name, label_text = terms[0], terms[1]
                    self.image_paths.append(
                        os.path.join(self.root, "val", "images", file_name)
                    )
                    self.targets.append(self.cls2idx[label_text])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        target = self.targets[index]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, target


class Clothing1M(Dataset):
    r"""https://github.com/LiJunnan1992/MLNT"""

    def __init__(self, root, transform, mode):
        self.root = os.path.join(root, "Clothing 1M")
        self.anno_dir = os.path.join(self.root, "annotations")
        self.transform = transform
        self.mode = mode

        self.imgs = []
        self.labels = {}

        if self.mode == "train":
            self.img_list_file = "noisy_train_key_list.txt"
            self.label_list_file = "noisy_label_kv.txt"
        elif self.mode == "test":
            self.img_list_file = "clean_test_key_list.txt"
            self.label_list_file = "clean_label_kv.txt"
        elif self.mode == "valid":
            self.img_list_file = "clean_val_key_list.txt"
            self.label_list_file = "clean_label_kv.txt"

        self.classes = [
            "T-Shirt",
            "Shirt",
            "Knitwear",
            "Chiffon",
            "Sweater",
            "Hoodie",
            "Windbreaker",
            "Jacket",
            "Downcoat",
            "Suit",
            "Shawl",
            "Dress",
            "Vest",
            "Underwear",
        ]

        with open(os.path.join(self.anno_dir, self.img_list_file), "r") as f:
            lines = f.read().splitlines()
            for l in lines:
                self.imgs.append(os.path.join(self.root, l))

        with open(os.path.join(self.anno_dir, self.label_list_file), "r") as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = os.path.join(self.root, entry[0])
                self.labels[img_path] = int(entry[1])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        target = self.labels[img_path]

        image = Image.open(img_path).convert("RGB")
        img = self.transform(image)
        return img, target

    def __len__(self):
        return len(self.imgs)


def selected_loader(train_loader, indices):
    sampler = IndicesSampler(indices, True)
    selected_dataloader = DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        num_workers=train_loader.num_workers,
        sampler=sampler,
    )
    return selected_dataloader


def delete_class(
    deleted_classes,
    train_dataset,
    test_dataset,
    noise_train_label_path,
    noise_test_ind_path,
    noise_ind_path,
):
    clean_classes = [
        i for i in range(len(train_dataset.classes)) if i not in deleted_classes
    ]
    train_dataset.classes = [train_dataset.classes[c] for c in clean_classes]
    test_dataset.classes = [test_dataset.classes[c] for c in clean_classes]

    if (
        os.path.exists(noise_train_label_path)
        and os.path.exists(noise_test_ind_path)
        and os.path.exists(noise_ind_path)
    ):
        train_dataset.targets = np.load(noise_train_label_path)
        test_ind = np.load(noise_test_ind_path)
        test_subdataset = Subset(test_dataset, test_ind)
        noise_ind = np.load(noise_ind_path)
    else:
        test_deleted_ind = None
        noise_ind = None
        for deleted_class in deleted_classes:
            train_ind = np.where(train_dataset.targets == deleted_class)[0]
            changed = np.random.choice(clean_classes, len(train_ind))
            train_dataset.targets[train_ind] = torch.LongTensor(changed)
            if noise_ind is None:
                noise_ind = train_ind
            else:
                noise_ind = np.concatenate([noise_ind, train_ind], axis=0)

            test_ind = np.where(test_dataset.targets == deleted_class)[0]
            if test_deleted_ind is None:
                test_deleted_ind = test_ind
            else:
                test_deleted_ind = np.concatenate([test_deleted_ind, test_ind], axis=0)
        test_ind = np.setdiff1d(range(len(test_dataset.targets)), test_deleted_ind)
        test_subdataset = Subset(test_dataset, test_ind)
        np.save(noise_train_label_path, train_dataset.targets)
        np.save(noise_test_ind_path, test_ind)
        np.save(noise_ind_path, noise_ind)

    return train_dataset, test_subdataset, noise_ind


def flip_label(
    dataset_name,
    noise_ratio,
    noise_type,
    train_dataset,
    noise_label_path,
    noise_ind_path,
):
    if os.path.exists(noise_label_path) and os.path.exists(noise_ind_path):
        train_dataset.targets = np.load(noise_label_path)
        noise_ind = np.load(noise_ind_path)
    else:
        labels = np.asarray(
            [[train_dataset.targets[i]] for i in range(len(train_dataset.targets))]
        )
        noisy_labels, P = noisify(
            dataset_name, len(train_dataset.classes), labels, noise_type, noise_ratio,
        )
        noisy_labels = np.array([i[0] for i in noisy_labels])
        noise_ind = np.where(np.array(train_dataset.targets) != noisy_labels)[0]
        train_dataset.targets = list(noisy_labels)
        np.save(noise_label_path, noisy_labels)
        np.save(noise_ind_path, noise_ind)

    return train_dataset, noise_ind


def divide_train(
    train_ratio, train_dataset, noise_ind, train_subset_path, valid_subset_path
):
    if os.path.exists(train_subset_path) and os.path.join(valid_subset_path):
        train_subset = Subset(train_dataset, np.load(train_subset_path))
        valid_subset = Subset(train_dataset, np.load(valid_subset_path))
    else:
        train_size = int(len(train_dataset) * train_ratio)
        val_size = len(train_dataset) - train_size
        train_subset, valid_subset = random_split(train_dataset, [train_size, val_size])
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
    noise_classes,
    seed,
    need_original=False,
):
    if dataset_name == "mnist":
        train_transform = transforms.Compose(
            [
                # transforms.Resize(28),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize([0.5], [0.5]),
            ]
        )
        test_transform = transforms.Compose(
            [
                # transforms.Resize(28),
                transforms.ToTensor(),
                # transforms.Normalize([0.5], [0.5]),
            ]
        )
    elif dataset_name.startswith("cifar"):
        train_transform = transforms.Compose(
            [
                # transforms.Resize(32),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        test_transform = transforms.Compose(
            [
                # transforms.Resize(32),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    elif dataset_name == "tiny-imagenet":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(56),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
                ),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(56),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
                ),
            ]
        )
    elif dataset_name == "clothing1m":
        train_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        raise NameError("Invalid dataset name")

    if dataset_name == "mnist":
        train_dataset = datasets.MNIST(root=root, download=True, train=True)
        test_dataset = datasets.MNIST(
            root=root, download=True, train=False, transform=test_transform
        )
    elif dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(root=root, download=True, train=True)
        test_dataset = datasets.CIFAR10(
            root=root, download=True, train=False, transform=test_transform
        )
    elif dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(root=root, download=True, train=True)
        test_dataset = datasets.CIFAR100(
            root=root, download=True, train=False, transform=test_transform
        )
    elif dataset_name == "tiny-imagenet":
        train_dataset = TinyImageNet(root=root, mode="train")
        test_dataset = TinyImageNet(root=root, mode="test", transform=test_transform)
    elif dataset_name == "clothing1m":
        train_dataset = Clothing1M(root=root, mode="train", transform=train_transform)
        test_dataset = Clothing1M(root=root, mode="test", transform=test_transform)
        valid_dataset = Clothing1M(root=root, mode="valid", transform=test_transform)

    else:
        raise NameError("Invalid dataset name")

    if dataset_name == "clothing1m":
        return (
            DatasetWithIndex(train_dataset),
            valid_dataset,
            test_dataset,
            [],
        )

    sub_dir = os.path.join("./data", f"changed_{dataset_name}_{seed}")
    os.makedirs(sub_dir, exist_ok=True)

    if len(noise_classes) > 0:
        noise_train_label_path = os.path.join(
            sub_dir, f"{noise_type}_{noise_classes}_train_label.npy",
        )
        noise_test_ind_path = os.path.join(
            sub_dir, f"{noise_type}_{noise_classes}_test_ind.npy",
        )
        noise_ind_path = os.path.join(sub_dir, f"{noise_type}_{noise_classes}_ind.npy",)
        train_dataset, test_subdataset, noise_ind = delete_class(
            noise_classes,
            train_dataset,
            test_dataset,
            noise_train_label_path,
            noise_test_ind_path,
            noise_ind_path,
        )

    elif noise_type == "symmetric" or noise_type == "asymmetric":
        noise_label_path = os.path.join(
            sub_dir, f"{noise_type}_{noise_ratio}_label.npy"
        )
        noise_ind_path = os.path.join(sub_dir, f"{noise_type}_{noise_ratio}_ind.npy")
        original_labels = np.array(train_dataset.targets).copy()
        train_dataset, noise_ind = flip_label(
            dataset_name,
            noise_ratio,
            noise_type,
            train_dataset,
            noise_label_path,
            noise_ind_path,
        )
        test_subdataset = test_dataset

    if train_ratio < 1:
        train_subset_path = os.path.join(sub_dir, f"{train_ratio}_train_subset.npy")
        valid_subset_path = os.path.join(sub_dir, f"{train_ratio}_valid_subset.npy")
        train_subset, valid_subset, train_noise_ind = divide_train(
            train_ratio, train_dataset, noise_ind, train_subset_path, valid_subset_path
        )
        valid_subdataset = DatasetFromSubset(valid_subset, test_transform)
    else:
        train_subset = train_dataset
        train_noise_ind = noise_ind
        valid_subdataset = None

    train_subdataset = DatasetWithIndex(
        DatasetFromSubset(train_subset, train_transform)
    )

    if need_original:
        return (
            train_subdataset,
            valid_subdataset,
            test_subdataset,
            train_noise_ind,
            original_labels,
        )
    else:
        return (
            train_subdataset,
            valid_subdataset,
            test_subdataset,
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
    parser.add_argument("--noise_classes", type=list, default=[])

    args = parser.parse_args()

    seed_all(args.seed)

    train_dataset, valid_dataset, test_dataset, train_noise_ind = load_datasets(
        args.dataset_name,
        args.dataset_path,
        args.train_ratio,
        args.noise_type,
        args.noise_ratio,
        args.noise_classes,
        args.seed,
    )
