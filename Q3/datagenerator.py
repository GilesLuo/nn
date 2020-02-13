from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader


# to import PIL, please make sure your PILLOW is pinned at version 6.2.1!!!

class ReadDataSource(Dataset):
    def __init__(self, x_dir, img_size, two_output=False):
        self.x_dir = x_dir  # folder '/joints_val_img', '/joints_test_img' or '/joints_train_img'
        self.img_size = img_size
        self.file_list = os.listdir(x_dir)  # return a list of file names
        self.two_output = two_output

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        x_loc = self.file_list[item]  # hand number

        x = torch.zeros((self.img_size * self.img_size, 1))

        img = Image.open(self.x_dir + x_loc)
        img = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.Resize((self.img_size * self.img_size, 1)),
            transforms.ToTensor()
        ])(img)
        x[:, :] = img
        x = x.squeeze()
        if not self.two_output:
            y = float(x_loc.split('_')[1])
        else:
            y = [float(x_loc.split('_')[1]), 1 - float(x_loc.split('_')[1])]
        y = torch.tensor(y)
        return x, y


if __name__ == "__main__":

    x_dir = './group_4/train/'
    tr = ReadDataSource(x_dir)

    trld = DataLoader(dataset=tr, batch_size=4, shuffle=True, drop_last=True)
    for idx, (x, y) in enumerate(trld):
        print('batch_', idx)
        print('x.shape:', x.shape)
        print('y.shape:', y.shape)
