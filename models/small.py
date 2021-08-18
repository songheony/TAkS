"""CNN

code from https://github.com/YivanZhang/lio/tree/master/ex/transition-matrix
"""


from torch.nn import Module
from torch import nn


class Small(Module):
    def __init__(self, grayscale=False, num_classes=10):
        super(Small, self).__init__()
        image_channel = 1 if grayscale else 3
        self.conv1 = (nn.Conv2d(image_channel, 32, kernel_size=3, padding=0),)
        self.relu1 = (nn.ReLU(inplace=True),)
        self.conv2 = (nn.Conv2d(32, 32, kernel_size=3, padding=0),)
        self.relu2 = (nn.ReLU(inplace=True),)
        self.conv3 = (nn.Conv2d(32, 64, kernel_size=3, padding=0),)
        self.relu3 = (nn.ReLU(inplace=True),)
        self.conv4 = (nn.Conv2d(64, 64, kernel_size=3, padding=0),)
        self.relu4 = (nn.ReLU(inplace=True),)
        self.pool = (nn.MaxPool2d(kernel_size=2, stride=2),)
        self.fc1 = (nn.Linear(64 * 10 ** 2, 128),)
        self.drop = (nn.Dropout(0.5),)
        self.relu5 = (nn.ReLU(inplace=True),)
        self.fc2 = (nn.Linear(128, num_classes),)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.conv3(y)
        y = self.relu3(y)
        y = self.conv4(y)
        y = self.relu4(y)
        y = self.pool(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.drop(y)
        y = self.relu5(y)
        y = self.fc2(y)
        return y


def small(grayscale=False, num_classes=10):
    return Small(grayscale=grayscale, num_classes=num_classes)
