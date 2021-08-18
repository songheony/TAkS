"""LeNets

code from https://github.com/scifancier/Class2Simi
"""


from torch import nn


class LeNet(nn.Module):
    def __init__(self, num_classes=10, img_sz = 28, grayscale=True):
        super(LeNet, self).__init__()
        image_channel = 1 if grayscale else 3
        feat_map_sz = img_sz//4
        self.n_feat = 50 * feat_map_sz * feat_map_sz

        self.conv = nn.Sequential(
            nn.Conv2d(image_channel, 20, 5, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 50, 5, padding=2),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.linear = nn.Sequential(
            nn.Linear(self.n_feat, 500),
            nn.BatchNorm1d(500),
        )
        self.last = nn.Linear(500, num_classes)  # Subject to be replaced dependent on task


    def features(self, x):
        x = self.conv(x)
        x = self.linear(x.view(-1, self.n_feat))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        out = self.logits(x)
        return out


def lenet(grayscale=False, num_classes=10):
    return LeNet(grayscale=grayscale, num_classes=num_classes)
