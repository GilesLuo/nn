import Q3.model
from Q3.datagenerator import ReadDataSource
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
import xlwt

if torch.cuda.is_available():
    print('available gpu detected')
else:
    print('available gpu not found, use cpu instead')


def train(num_epoch=1, batch_size=1, lr=0.001, img_size=256, is_two_output=False, decay_lr=None, PCA=False):
    tr_dir = './group_4/train/'
    tr_data = ReadDataSource(tr_dir, img_size, two_output=is_two_output, PCA=PCA)
    tr_loader = DataLoader(dataset=tr_data, batch_size=batch_size, shuffle=True, drop_last=True)

    val_dir = './group_4/val/'
    val_data = ReadDataSource(val_dir, img_size, two_output=is_two_output, PCA=PCA)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, drop_last=True)
    model = Q3.model.ResNet()

    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if decay_lr is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=decay_lr)
    loss_func = torch.nn.CrossEntropyLoss()

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

            # y = y.unsqueeze(1)
            x = x.float()
            y = y.long()
            prediction = model(x)
            # print(prediction)
            loss = loss_func(prediction, y.long())
            # print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            avg_training_loss += loss.item()

            _, pre_class = prediction.max(1)

            wrong = abs(float(torch.sum(y.float() - pre_class.float())))
            # wrong classification number in a batch

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
                y = y.unsqueeze(1)
                y = y.long()
                x = x.float()
                prediction = model(x)

                _, pre_class = prediction.max(1)

                wrong = abs(float(torch.sum(y.float() - pre_class.float())))
                # wrong classification number in a batch
                wrong_clas2 += wrong
            val_accuracy = round(100 * (1 - wrong_clas2 / (count + 1)), 4)
            val_accuracy_history.append(val_accuracy)
            print('val accuracy: ' + str(val_accuracy) + '%: '
                  + str(int(count + 1 - wrong_clas2)) + '/' + str(count + 1))
    return train_accuracy_history, val_accuracy_history, loss_history


class write_excel:
    def __init__(self, path=None):
        self.sheet_name = [0]
        if path is None:
            self.path = './' + str(self.sheet_name[0]) + '.xls'
        else:
            self.path = path

    def write(self, value):
        index = len(value)
        workbook = xlwt.Workbook()
        sheet = workbook.add_sheet(str(self.sheet_name[0]))
        for i in range(0, index):
            for j in range(0, len(value[i])):
                sheet.write(i, j, value[i][j])  # 像表格中写入数据（对应的行和列）
        workbook.save(self.path)
        self.sheet_name[0] += 1
        self.path = './' + str(self.sheet_name[0]) + '.xls'


if __name__ == '__main__':
    writer = write_excel()
    tr_ac, val_ac, loss = train(num_epoch=200, batch_size=8, lr=0.0001, img_size=256,
                                decay_lr=0.9, PCA=False)
    print('max val accuracy is: ' + str(max(val_ac)))
    print('max train accuracy is: ' + str(max(tr_ac)))
    hist = np.array([tr_ac, val_ac])
    writer.write(hist)
