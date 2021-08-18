import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class TinyImageNet(Dataset):
    """from https://github.com/leemengtaiwan/tiny-imagenet
    Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.
    """

    def __init__(self, root, transform=None, mode="train"):
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


def get_tinyimagenet(root):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(56),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.CenterCrop(56),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]
    )
    train_dataset = TinyImageNet(root=root, mode="train")
    test_dataset = TinyImageNet(root=root, mode="test", transform=test_transform)
    return train_dataset, None, test_dataset, train_transform, test_transform
