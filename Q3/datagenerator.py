from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader


class ReadDataSource(Dataset):
    def __init__(self, x_dir):
        self.x_dir = x_dir  # folder '/joints_val_img', '/joints_test_img' or '/joints_train_img'

        self.file_list = os.listdir(x_dir)  # return a list of file names

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        x_loc = self.file_list[item]  # hand number
        y = x_loc.split('_', 1)[1]
        x = torch.zeros((256, 256))

        img = Image.open(x_loc)
        img = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])(img)
        x[:, :] = img

        return x, y


if __name__ == "__main__":

    x_dir = './group_4/train/'
    tr = ReadDataSource(x_dir)

    trld = DataLoader(dataset=tr, batch_size=4, shuffle=False)
    for idx, (x, y) in enumerate(trld):
        print('batch_', idx)
        print('x.shape:', x.shape)
        print('y.shape:', y.shape)
