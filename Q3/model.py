import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, n_features, two_output=False):
        super(MLP, self).__init__()
        self.n_output = 1 if not two_output else 2
        self.dense = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.Dropout(0.5),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.Sigmoid(),
            nn.Linear(64, 32),
            nn.Dropout(0.5),
            nn.Sigmoid(),
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
            nn.Linear(n_features, 256),
            # nn.Dropout(0.5),
            nn.Sigmoid(),
            nn.Linear(256, 32),
            # nn.Dropout(0.5),
            nn.Sigmoid(),
            nn.Linear(32, self.n_output),
            nn.Sigmoid(),
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

