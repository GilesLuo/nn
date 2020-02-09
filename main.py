import numpy as np
import matplotlib
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import random
import math
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SingleLayer(torch.nn.Module):
    def __init__(self, n_features=1, n_hidden=1, n_output=1):
        super().__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        h = torch.tanh(self.hidden(x))
        out = self.output(h)
        return out


def train(x, y, num_hidden, lr=0.6, max_epoch=1000000):
    loss_history = []
    net = SingleLayer(1, num_hidden, 1)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()
    for t in tqdm(range(max_epoch)):
        prediction = net(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.data.float())

        if t % 5 == 0:
            # plot and show learning process
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.scatter(x.data.numpy(), prediction.data.numpy())
            plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)

        if loss.data.float() < 0.01:
            return loss_history, t
    print('training timeout')
    return loss_history, None


if __name__ == '__main__':
    # generate training data
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    y = 1.2 * torch.sin(math.pi * x) - torch.cos(2.4 * math.pi * x)
    x, y = Variable(x), Variable(y)

    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.show()

    index = [i for i in range(len(x))]
    random.shuffle(index)
    x = x[index]
    y = y[index]

    loss, t = train(x, y, 10, lr=0.1)
    print(loss)
    print(t)
