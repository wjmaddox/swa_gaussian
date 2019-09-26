import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# import bnn

__all__ = ["LeNet5"]


class LeNet5Base(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5Base, self).__init__()
        self.conv_part = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc_part = nn.Sequential(
            nn.Linear(800, 500), nn.ReLU(True), nn.Linear(500, num_classes.item())
        )

        # Initialize weights
        for m in self.conv_part.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv_part(x)
        x = x.view(x.size(0), -1)
        x = self.fc_part(x)
        return x


"""class LeNet5BNN(bnn.BayesianModule):

    def __init__(self, num_classes):
        super(LeNet5BNN, self).__init__()
        self.conv_part = nn.Sequential(
            bnn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
            bnn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc_part = nn.Sequential(
            bnn.Linear(800, 500),
            nn.ReLU(True),
            bnn.Linear(500, num_classes)
        )

        # Initialize weights

        for m in self.modules():
            if isinstance(m, bnn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight_mean.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.conv_part(x)
        x = x.view(x.size(0), -1)
        x = self.fc_part(x)
        return x"""


class LeNet5:
    base = LeNet5Base
    # bnn = LeNet5BNN
    args = list()
    kwargs = {}

    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        ]
    )

    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))]
    )
