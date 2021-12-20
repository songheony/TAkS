import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_mnist(root):
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST(root=root, download=True, train=True, transform=train_transform)
    test_dataset = datasets.MNIST(root=root, download=True, train=False, transform=test_transform)
    train_dataset.coarse_classes = list(range(10))
    test_dataset.coarse_classes = list(range(10))
    train_dataset.courses = train_dataset.targets
    test_dataset.courses = test_dataset.targets
    return train_dataset, None, test_dataset, train_transform, test_transform
