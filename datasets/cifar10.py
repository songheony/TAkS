import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_cifar10(root):
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
    train_dataset = datasets.CIFAR10(root=root, download=True, train=True)
    test_dataset = datasets.CIFAR10(root=root, download=True, train=False)
    train_dataset.coarse_classes = list(range(10))
    test_dataset.coarse_classes = list(range(10))
    train_dataset.courses = train_dataset.targets
    test_dataset.courses = test_dataset.targets
    return train_dataset, None, test_dataset, train_transform, test_transform
