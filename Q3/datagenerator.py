from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
# to import PIL, please make sure your PILLOW is pinned at version 6.2.1!!!
import numpy as np

class ReadDataSource(Dataset):
    def __init__(self, x_dir, img_size, two_output=False, PCA=True):
        self.PCA = PCA
        self.x_dir = x_dir  # folder '/joints_val_img', '/joints_test_img' or '/joints_train_img'
        self.img_size = img_size
        self.file_list = os.listdir(x_dir)  # return a list of file names
        self.two_output = two_output

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        x_loc = self.file_list[item]  # hand number

        x = torch.zeros((1, self.img_size, self.img_size))

        img = Image.open(self.x_dir + x_loc)
        img = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])(img)
        x[0, :, :] = img
        # print(x.shape)
        if self.PCA:

            x = x.numpy()
            T = x - x.mean(axis=0)
            C = np.cov(x.T)
            C = (C+C.T)/2
            w, v = np.linalg.eig(C)
            v_ = np.mat(v[:, 0])
            v_ = v_.T
            x = T * v_
            x = x.real
            x = torch.from_numpy(x)
        # print(x.shape)
        # x = x.reshape(-1, 1)

        # x = x.squeeze()
        y = float(x_loc.split('_')[1])
        y = torch.tensor(y)
        return x, y


if __name__ == "__main__":

    x_dir = './group_4/train/'
    tr = ReadDataSource(x_dir, img_size=256, PCA=False)

    trld = DataLoader(dataset=tr, batch_size=10, shuffle=True, drop_last=True)
    for idx, (x, y) in enumerate(trld):
        print('batch_', idx)
        print('x.shape:', x.shape)
        print('y.shape:', y.shape)

