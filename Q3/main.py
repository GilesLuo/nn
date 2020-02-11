import Q3.model
from Q3.datagenerator import ReadDataSource
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable

if torch.cuda.is_available():
    print('available gpu detected')
else:
    print('available gpu not found, use cpu instead')


def train(num_epoch=1, batch_size=1, lr=0.001, img_size=256):
    tr_dir = './group_4/train/'
    tr_data = ReadDataSource(tr_dir, img_size)
    tr_loader = DataLoader(dataset=tr_data, batch_size=batch_size, shuffle=True, drop_last=True)

    val_dir = './group_4/val/'
    val_data = ReadDataSource(val_dir, img_size)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=True, drop_last=True)

    model = Q3.model.MLP2(img_size * img_size)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()

    loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []
    for epoch in range(num_epoch):
        print('epoch:' + str(epoch))
        avg_training_loss = 0
        wrong_clas1 = 0
        wrong_clas2 = 0

        model.train()

        for step, (x, y) in enumerate(tr_loader):
            count = step
            x, y = Variable(x), Variable(y)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            y = y.unsqueeze(1)
            prediction = model(x)
            # print(prediction.shape)
            # print(y.shape)
            loss = loss_func(prediction, y.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_training_loss += loss.item()
            pre_class = torch.gt(prediction, 0.5)
            wrong = float(torch.sum(y.float() - pre_class.float()))
            wrong_clas1 += wrong
        train_accuracy = round(100 * (1 - wrong_clas1 / (count + 1) / batch_size), 4)
        avg_training_loss = avg_training_loss / (count + 1)
        loss_history.append(avg_training_loss)
        train_accuracy_history.append(train_accuracy)
        print('training loss:' + str(avg_training_loss))
        print('training accuracy: ' + str(train_accuracy) + '%')

        with torch.no_grad():
            model.eval()
            for step, (x, y) in enumerate(val_loader):
                count = step
                x, y = Variable(x), Variable(y)
                if torch.cuda.is_available():
                    x, y = x.cuda(), y.cuda()
                prediction = model(x)
                prediction = torch.gt(prediction, 0.5)
                loss = loss_func(prediction.float(), y.float())
                wrong_clas2 += abs(loss.item())
            val_accuracy = round(100 * (1 - wrong_clas2 / (count + 1)), 4)
            val_accuracy_history.append(val_accuracy)
            print('val accuracy: ' + str(val_accuracy) + '%')


if __name__ == '__main__':
    train(num_epoch=100, batch_size=4, lr=0.005, img_size=128)
