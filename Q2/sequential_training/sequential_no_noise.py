import numpy as np
import matplotlib
from tqdm import tqdm
import torch
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import imageio
import os

file_name = './plot_no_noise'
if torch.cuda.is_available():
    device = torch.device("cuda:0")
if not os.path.isdir(file_name):
    os.mkdir(file_name)


class SingleLayer(torch.nn.Module):
    def __init__(self, n_features=1, n_hidden=1, n_output=1):
        super().__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        h = torch.tanh(self.hidden(x))
        out = self.output(h)
        return out


def train(x_full, y_full, x, y, num_hidden, lr=0.6, max_epoch=10000,
          show_training=False, write_plot=True, write_GIF=True):
    frames = []
    loss_history = []
    net = SingleLayer(1, num_hidden, 1)
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net, device_ids=[0]).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_func = torch.nn.L1Loss()
    figure = matplotlib.pyplot.figure()
    last_loss = 0
    for t in tqdm(range(max_epoch)):
        prediction = net(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if show_training:
            if t % 30 == 0:
                # plot and show learning process
                pre = net(x_full)
                plt.cla()
                plt.scatter(x_full.data.numpy(), y_full.data.numpy())
                plt.plot(x_full.data.numpy(), pre.data.numpy(), 'r')
                plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
                plt.text(0.5, -0.5, 'Epoch=%i' % t, fontdict={'size': 20, 'color': 'red'})
                plt.pause(0.1)
                if write_GIF:
                    frames.append(np.array(figure.canvas.renderer.buffer_rgba()))
        if loss.data.float() < 0.00001:
            if write_plot:
                plt.savefig(file_name + '/regression_hidden=' + str(hidden_list[i]) + '_epoch=' + str(t) + '.png')
                plt.close(figure)
                print('training timeout')

            return loss_history, t, frames
        else:
            last_loss = loss.item()
    if write_plot:
        plt.savefig(file_name + '/regression_hidden=' + str(hidden_list[i]) + '_epoch=' + str(t) + '.png')
        plt.close(figure)
        print('training timeout')

    return loss_history, t, frames


if __name__ == '__main__':
    # generate training data
    x = torch.unsqueeze(torch.linspace(-1, 1, 41), dim=1)
    y = 1.2 * torch.sin(math.pi * x) - torch.cos(2.4 * math.pi * x)
    x_full = torch.unsqueeze(torch.linspace(-3, 3, 1000), dim=1)
    y_full = 1.2 * torch.sin(math.pi * x_full) - torch.cos(2.4 * math.pi * x_full)
    if torch.cuda.is_available():
        x, y = x.cuda(), y.cuda()
    x, y = Variable(x), Variable(y)

    hidden_list = [50]
    # hiddern_list means the number of hidden neurons in the network
    loss_list = []
    for i in range(len(hidden_list)):
        loss, t, frames = train(x_full, y_full, x, y, hidden_list[i], lr=0.01, max_epoch=28000,
                                write_GIF=True, write_plot=True, show_training=True)
        loss_list.append(loss)
        imageio.mimwrite(file_name + '/hidden=' + str(hidden_list[i]) + '_epoch=' + str(t) + '.gif', frames,
                         duration=0.02)
        loss = np.array(loss)

        plt.title('hidden=' + str(hidden_list[i]) + '_epoch=' + str(t))
        plt.xlabel('num_epoch')
        plt.ylabel('loss value')
        plt.plot(range(len(loss)), loss)
        plt.savefig(file_name + '/loss_hidden=' + str(hidden_list[i]) + '_epoch=' + str(t) + '.png')
        plt.close()

    plt.title('comparison')
    plt.xlabel('num_epoch')
    plt.ylabel('loss value')
    for i in range(len(loss_list)):
        plt.plot(range(len(loss_list[i])), loss_list[i])

    # plt.savefig(file_name + 'comparison.png')
    plt.show()
