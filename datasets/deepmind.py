import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import tensorflow as tf

from datasets.cifar100 import sparse2coarse


class DeepMind(Dataset):
    r"""https://github.com/deepmind/deepmind-research/blob/master/noisy_label"""

    def __init__(self, root, transform, task_name, noise_level, mode, rater_idx=0):
        self.root = os.path.join(root, "DeepMind", task_name, noise_level)
        self.transform = transform
        self.task_name = task_name
        self.noise_level = noise_level
        self.mode = mode
        self.rater_idx = rater_idx

        image_path = os.path.join(self.root, mode) + "*"
        raw_image_dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(image_path))

        # Create a dictionary describing the features.
        if task_name == "cifar10":
            num_raters = 10
            image_feature_description = {
                # the raw image
                "image/raw": tf.io.FixedLenFeature([], tf.string),
                # the clean label
                "image/class/label": tf.io.FixedLenFeature([1], tf.int64),
                # noisy labels from all the raters
                "noisy_labels": tf.io.FixedLenFeature([num_raters], tf.int64),
                # the IDs of rater models
                "rater_ids": tf.io.FixedLenFeature([num_raters], tf.string),
            }
            self.image_key = "image/raw"
            self.clean_label_key = "image/class/label"
        elif task_name == "cifar100":
            num_raters = 11
            image_feature_description = {
                # the raw image
                "image/encoded": tf.io.FixedLenFeature([], tf.string),
                # the fine-grained clean label, value in [0, 99]
                "image/class/fine_label": tf.io.FixedLenFeature([1], tf.int64),
                # the coarse clean label, value in [0, 19]
                "image/class/coarse_label": tf.io.FixedLenFeature([1], tf.int64),
                # noisy labels from all the raters
                "noisy_labels": tf.io.FixedLenFeature([num_raters], tf.int64),
                # the IDs of rater models
                "rater_ids": tf.io.FixedLenFeature([num_raters], tf.string),
            }
            self.image_key = "image/encoded"
            self.clean_label_key = "image/class/fine_label"

        def _parse_image_function(example_proto):
            # Parse the input tf.train.Example proto using the dictionary above.
            return tf.io.parse_single_example(example_proto, image_feature_description)

        self.parsed_image_dataset = list(raw_image_dataset.map(_parse_image_function))

        self.clean_labels = []
        self.targets = []
        for index in range(len(self.parsed_image_dataset)):
            features = self.parsed_image_dataset[index]
            clean_label = features[self.clean_label_key].numpy()[0]
            noisy_label = features["noisy_labels"].numpy()[self.rater_idx]
            self.clean_labels.append(clean_label)
            self.targets.append(noisy_label)

        if task_name == "cifar10":
            self.courses = self.targets
        elif task_name == "cifar100":
            self.courses = sparse2coarse(self.targets)

    def __getitem__(self, index):
        features = self.parsed_image_dataset[index]
        image = tf.reshape(
            tf.io.decode_raw(features[self.image_key], tf.uint8), (32, 32, 3)
        ).numpy()
        image = Image.fromarray(image)
        noisy_label = self.targets[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, noisy_label

    def __len__(self):
        return len(self.clean_labels)


def get_cifar10(root, noise_level):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = DeepMind(
        root=root,
        task_name="cifar10",
        noise_level=noise_level,
        mode="train",
        transform=train_transform,
    )
    valid_dataset = DeepMind(
        root=root,
        task_name="cifar10",
        noise_level=noise_level,
        mode="valid",
        transform=test_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=root, download=True, train=False, transform=test_transform
    )
    train_dataset.coarse_classes = list(range(10))
    valid_dataset.coarse_classes = list(range(10))
    test_dataset.coarse_classes = list(range(10))
    return train_dataset, valid_dataset, test_dataset, train_transform, test_transform


def get_cifar100(root, noise_level):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = DeepMind(
        root=root,
        task_name="cifar100",
        noise_level=noise_level,
        mode="train",
    )
    valid_dataset = DeepMind(
        root=root,
        task_name="cifar100",
        noise_level=noise_level,
        mode="valid",
    )
    test_dataset = datasets.CIFAR100(root=root, download=True, train=False)
    train_dataset.coarse_classes = list(range(20))
    valid_dataset.coarse_classes = list(range(20))
    test_dataset.coarse_classes = list(range(20))
    return train_dataset, valid_dataset, test_dataset, train_transform, test_transform
