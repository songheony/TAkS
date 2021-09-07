import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


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

        # According to Class1Simi
        self.classes.remove("Sweater")

        with open(os.path.join(self.anno_dir, self.img_list_file), "r") as f:
            lines = f.read().splitlines()
            for l in lines:
                self.imgs.append(os.path.join(self.root, l))

        with open(os.path.join(self.anno_dir, self.label_list_file), "r") as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                label = int(entry[1])

                # According to Class2Simi
                if label == 4:
                    label = 2
                elif label > 4:
                    label = label - 1

                img_path = os.path.join(self.root, entry[0])
                self.labels[img_path] = label

        self.targets = [self.labels[img_path] for img_path in self.imgs]
        self.courses = self.targets

    def __getitem__(self, index):
        img_path = self.imgs[index]
        target = self.targets[img_path]

        image = Image.open(img_path).convert("RGB")
        img = self.transform(image)
        return img, target

    def __len__(self):
        return len(self.imgs)


def get_clothing1m(root):
    train_transform = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = Clothing1M(root=root, mode="train")
    test_dataset = Clothing1M(root=root, mode="test")
    valid_dataset = Clothing1M(root=root, mode="valid")
    return train_dataset, valid_dataset, test_dataset, train_transform, test_transform
