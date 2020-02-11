import Q3.model
from Q3.datagenerator import ReadDataSource
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable


def train(num_epoch=1, batch_size=1, lr=0.001):
    tr_dir = './group_4/train/'
    tr_data = ReadDataSource(tr_dir)
    tr_loader = DataLoader(dataset=tr_data, batch_size=batch_size, shuffle=True, drop_last=True)

    val_dir = './group_4/val/'
    val_data = ReadDataSource(val_dir)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=True, drop_last=True)

    model = Q3.model.MLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()

    loss_history = []
    accuracy_history = []
    for epoch in range(num_epoch):
        print('epoch:' + str(epoch))
        avg_training_loss = 0
        wrong_clas = 0

        model.train()

        for step, (x, y) in enumerate(tr_loader):
            count = step
            x, y = Variable(x), Variable(y)
            y = y.unsqueeze(1)
            prediction = model(x)

            loss = loss_func(prediction, y.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_training_loss += loss.item()
        avg_training_loss = avg_training_loss / (count + 1)
        loss_history.append(avg_training_loss)
        print('training loss:' + str(avg_training_loss))

        with torch.no_grad():
            model.eval()
            for step, (x, y) in enumerate(val_loader):
                count = step
                x, y = Variable(x), Variable(y)

                prediction = model(x)
                prediction = torch.Tensor([1]) if float(prediction) > 0.5 else torch.Tensor([1])
                loss = loss_func(prediction, y.float())
                wrong_clas += abs(loss.item())
            accuracy = round(1 - wrong_clas / (count + 1), 6)
            accuracy_history.append(accuracy)
            print('accuracy: ' + str(accuracy*100) + '%')


if __name__ == '__main__':
    train(num_epoch=20, batch_size=2, lr=0.01)
