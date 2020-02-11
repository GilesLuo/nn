import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, n_features):
        super(MLP, self).__init__()
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
            nn.Linear(32, 1),
            nn.Dropout(0.5),
            nn.Sigmoid(),
        )

    def forward(self, x):
        o = self.dense(x)
        return o


class MLP2(torch.nn.Module):
    def __init__(self, n_features):
        super(MLP2, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(n_features, 256),
            # nn.Dropout(0.5),
            nn.Sigmoid(),
            nn.Linear(256, 32),
            # nn.Dropout(0.5),
            nn.Sigmoid(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        o = self.dense(x)
        return o
