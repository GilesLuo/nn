from torch.utils.data import Dataset
import math
from torch.utils.data import DataLoader
import numpy as np


class ReadDataSource(Dataset):
    def __init__(self, x, y, ):
        self.x_list = x
        self.y_list = y

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, item):
        x = self.x_list[item]  # data

        y = self.y_list[item]  # label
        return x, y


if __name__ == "__main__":
    x_list = np.linspace(-1, 1, 100)
    y_list = [1.2 * math.sin(math.pi * x_list[i]) - math.cos(2.4 * math.pi * x_list[i])
              for i in range(len(x_list))]
    tr = ReadDataSource(x_list, y_list)
    trld = DataLoader(dataset=tr, batch_size=4, shuffle=False, drop_last=True)
    for idx, (x, y) in enumerate(trld):
        print('batch_', idx)
        print('x.shape:', x.shape)
        print('y.shape:', y.shape)
