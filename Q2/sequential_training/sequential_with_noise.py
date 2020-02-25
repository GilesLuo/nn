import numpy as np
import matplotlib
import xlwt
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import random
import math
import matplotlib.pyplot as plt
import imageio
import os

if not os.path.isdir('./sequential_w'):
    os.mkdir('./sequential_w')
if torch.cuda.is_available():
    device = torch.device("cuda:0")


class SingleLayer(torch.nn.Module):
    def __init__(self, n_features=1, n_hidden=1, n_output=1):
        super().__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        h = torch.tanh(self.hidden(x))
        out = self.output(h)
        return out


def train(x, y, num_hidden, lr=0.6, max_epoch=10000):
    frames = []
    loss_history = []
    net = SingleLayer(1, num_hidden, 1)
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net, device_ids=[0]).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()
    figure = matplotlib.pyplot.figure()
    last_loss = 0
    for t in tqdm(range(max_epoch)):
        prediction = net(x)
        loss = loss_func(prediction, y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if t % 40 == 0:
            # plot and show learning process

            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.scatter(x.data.numpy(), prediction.data.numpy())
            plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
            plt.text(0.5, -0.5, 'Epoch=%.4f' % t, fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)
            frames.append(np.array(figure.canvas.renderer.buffer_rgba()))
        if loss.data.float() < 0.001 or abs(last_loss - loss.item()) < 0.0000001:
            plt.close(figure)
            return loss_history, t, frames
        else:
            last_loss = loss.item()

    print('training timeout')
    plt.close(figure)
    return loss_history, t, frames


if __name__ == '__main__':
    # generate training data

    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    y = 1.2 * torch.sin(math.pi * x) - torch.cos(
        2.4 * math.pi * x) + 0.6*torch.rand(x.size())
    if torch.cuda.is_available():
        x, y = x.cuda(), y.cuda()
    x, y = Variable(x), Variable(y)

    hidden_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50]

    for i in range(len(hidden_list)):
        loss, t, frames = train(x, y, hidden_list[i], lr=0.05, max_epoch=10000)
        print(loss)
        imageio.mimwrite('./sequential_w/lr=0.05_hidden=' + str(hidden_list[i]) + '_epoch=' + str(t) + '.gif', frames,
                         duration=0.02)
        # loss = np.array(loss)
        plt.title('lr=0.05_hidden=' + str(hidden_list[i]) + '_epoch=' + str(t))
        plt.xlabel('num_epoch')
        plt.ylabel('loss value')
        plt.plot(range(len(loss)), loss)
        plt.savefig('./sequential_w/lr=0.05_hidden=' + str(hidden_list[i]) + '_epoch=' + str(t) + '.png')
        plt.close()
