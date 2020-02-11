import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.h1 = nn.Linear(128*128, 256)
        self.h2 = nn.Linear(256, 64)
        self.h3 = nn.Linear(64, 8)
        self.o = nn.Linear(8, 1)

    def forward(self, x):
        h1 = torch.sigmoid((self.h1(x)))
        h2 = torch.sigmoid(self.h2(h1))
        h3 = torch.sigmoid(self.h3(h2))
        o = torch.sigmoid(self.o(h3))
        return o
