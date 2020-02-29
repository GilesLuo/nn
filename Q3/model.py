import torch
import torch.nn as nn
import torch.nn.functional as F
from Q3.ResNet import resnet18


class MLP(torch.nn.Module):
    def __init__(self, n_features, two_output=False):
        super(MLP, self).__init__()
        self.n_output = 1 if not two_output else 2
        self.dense = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.Dropout(0.5),
            # nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.Dropout(0.5),
            # nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.Dropout(0.5),
            # nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.Dropout(0.5),
            # nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, self.n_output),
            nn.Dropout(0.5),
            nn.Sigmoid(),
        )

    def forward(self, x):
        o = self.dense(x)
        return o


class MLP2(torch.nn.Module):
    def __init__(self, n_features, two_output=False):
        super(MLP2, self).__init__()
        self.n_output = 1 if not two_output else 2
        self.dense = nn.Sequential(
            nn.Linear(n_features, 4),
            nn.Tanh(),
            nn.Linear(4, self.n_output),
            nn.Sigmoid(),
            # nn.Threshold(threshold=0.5, value=1),

        )

    def forward(self, x):
        o = self.dense(x)
        return o


class Perceptron(nn.Module):
    def __init__(self, n_features, two_output=False):
        super(Perceptron, self).__init__()
        self.n_output = 1 if not two_output else 2
        self.fc1 = nn.Linear(n_features, self.n_output)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        return x


class Cnn(nn.Module):
    def __init__(self, in_dim=1, n_class=2):
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 62 * 62, 40),
            nn.Linear(40, n_class)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = resnet18(input_channels=1)
        self.denses = nn.Sequential(
            nn.Linear(32768, 1000),
            nn.Dropout(0.5),
            nn.Linear(1000, 100),
            nn.Dropout(0.5),
            nn.Linear(100, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        feature = self.resnet(x)
        out = torch.flatten(feature, 1)
        out = self.denses(out)
        return out
