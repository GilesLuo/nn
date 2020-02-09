import numpy as np
import matplotlib
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import random
import math
import matplotlib.pyplot as plt
import imageio

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


def train(x, y, num_hidden, lr=0.6, max_epoch=100000):
    frames = []
    loss_history = []
    net = SingleLayer(1, num_hidden, 1)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()
    figure = matplotlib.pyplot.figure()

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
            frames.append(np.array(figure.canvas.renderer.buffer_rgba()))
        if loss.data.float() < 0.001:
            plt.close(figure)
            return loss_history, t, frames
    print('training timeout')
    plt.close(figure)
    return loss_history, t, frames


if __name__ == '__main__':
    # generate training data
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    y = 1.2 * torch.sin(math.pi * x) - torch.cos(2.4 * math.pi * x)
    x, y = Variable(x), Variable(y)

    index = [i for i in range(len(x))]
    random.shuffle(index)
    x = x[index]
    y = y[index]

    loss, t, frames = train(x, y, 1, lr=0.05)
    imageio.mimwrite('lr=0.5_hidden=1_epoch=' + str(t) + '.gif', frames)
    loss, t, frames = train(x, y, 2, lr=0.05)
    imageio.mimwrite('lr=0.5_hidden=2_epoch=' + str(t) + '.gif', frames)
    loss, t, frames = train(x, y, 5, lr=0.05)
    imageio.mimwrite('lr=0.5_hidden=5_epoch=' + str(t) + '.gif', frames)
    loss, t, frames = train(x, y, 10, lr=0.05)
    imageio.mimwrite('lr=0.5_hidden=1_epoch=' + str(t) + '.gif', frames)
    loss, t, frames = train(x, y, 20, lr=0.05)
    imageio.mimwrite('lr=0.5_hidden=1_epoch=' + str(t) + '.gif', frames)
    loss, t, frames = train(x, y, 50, lr=0.05)
    imageio.mimwrite('lr=0.5_hidden=1_epoch=' + str(t) + '.gif', frames)
