import argparse
import torch
# import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from tqdm import tqdm
from datagenerator import ReadDataSource

from model import Model
import pdb


# setting gpu
# CUDA_VISIBLE_DEVICES = 1


# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../BA_dataset')
    args = parser.parse_args()

    test_dir = args.data_dir + '/joints_test_img'
    test_csv = args.data_dir + '/test_set.csv'
    checkpoint = torch.load('./models/model_ep11.pth.tar')

    # todo: change check point epoch num

    print(checkpoint['epoch'])
    trained_model = checkpoint['model_state_dict']

    print('loading testing dataset ...')

    test_dataset = ReadDataSource(x_dir=test_dir, y_file=test_csv)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    print('loading model ...')

    model = Model()
    model.load_state_dict(trained_model)
    model.cuda()

    print('------ start testing ------')

    model.eval()
    count = 0
    error = 0.

    for idx, (x, y) in enumerate(tqdm(test_loader)):
        x, y = x.cuda(), y.cuda()  # x: [batch_size, 14, 48, 48]
        outs = model(x)

        count += x.size(0)
        _, preds = outs.max(1)
        y = y.float()
        preds = preds.float()
        # MAE
        error += float(abs(preds - y).sum())
    mae = error / count

    print('MAE = {:10.6f}'.format(mae))


if __name__ == '__main__':
    main()
