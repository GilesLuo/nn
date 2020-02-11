import numpy as np
import matplotlib
import xlwt
from tqdm import tqdm
import torch
import Q2.datagenerator
import math
import matplotlib.pyplot as plt
import imageio
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./')

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


class BatchSingleLayer(torch.nn.Module):
    def __init__(self, n_features=1, n_hidden=1, n_output=1):
        super().__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        h = torch.tanh(self.hidden(x))
        out = self.output(h)
        return out


def train(x_list, y_list, num_hidden, lr=0.6, max_epoch=10000, batch_size=1):
    frames = []
    loss_history = []

    train_dataset = Q2.datagenerator.ReadDataSource(x_list, y_list)
    train_loader = Q2.datagenerator.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    batch_iterator = iter(train_loader)

    net = BatchSingleLayer(1, num_hidden, 1)

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()
    figure = matplotlib.pyplot.figure()

    for epoch in tqdm(range(max_epoch)):
        x_plot = []
        p_plot = []
        for step, (x, y) in enumerate(train_loader):
            x, y = Variable(x), Variable(y)
            x = x.unsqueeze(1)

            prediction = net(x)
            prediction = prediction.squeeze()

            loss = loss_func(prediction, y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            x_plot.extend(x.data.tolist())
            p_plot.extend(prediction.data.tolist())

        if epoch % 40 == 0:
            # plot and show learning process
            plt.cla()
            plt.scatter(x_list, y_list)
            plt.scatter(x_plot, p_plot, c='r')

            plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
            plt.text(0.5, -0.5, 'Epoch=%.4f' % epoch, fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)
            frames.append(np.array(figure.canvas.renderer.buffer_rgba()))

        if loss.data.float() < 0.001:
            plt.close(figure)
            return loss_history, epoch, frames
        writer.add_scalar(str(num_hidden) + '_loss', loss.item(), epoch)
    print('training timeout')
    plt.close(figure)

    return loss_history, max_epoch, frames


if __name__ == '__main__':
    # generate training data
    x_list = np.linspace(-1, 1, 100)
    y_list = [1.2 * math.sin(math.pi * x_list[i]) - math.cos(2.4 * math.pi * x_list[i])
              for i in range(len(x_list))]

    hidden_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50]
    f = xlwt.Workbook()  # 创建工作簿
    sheet_1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
    for i in range(len(hidden_list)):
        loss, t, frames = train(x_list, y_list, hidden_list[i], lr=0.05, max_epoch=1000, batch_size=4)

        imageio.mimwrite('lr=0.05_hidden=' + str(hidden_list[i]) + '_epoch=' + str(t) + '.gif', frames, duration=0.02)

