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
    train_dataset = datasets.MNIST(root=root, download=True, train=True)
    test_dataset = datasets.MNIST(
        root=root, download=True, train=False, transform=test_transform
    )
    return train_dataset, None, test_dataset, train_transform, test_transform
