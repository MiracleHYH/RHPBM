import json
import os

import numpy as np
import torch
import torch.utils.data
from torch import nn, optim

from model import RHPBM

batch = 64
epoch = 200
lr = 1e-4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

d = 20
beta = 0.8
SparsityLoss = nn.L1Loss(reduction='sum')

video = 'Video_003'
proj_dir = os.path.abspath('.')
dataset_dir = os.path.join(proj_dir, 'data', 'dataset')
data_path = os.path.join(dataset_dir, '%s.npy' % video)
info_path = os.path.join(dataset_dir, '%s.json' % video)
model_path = os.path.join(proj_dir, 'checkpoint', '%s.pth' % video)


def loss(_x, x, mean_z, logvar_z):
    l1 = SparsityLoss(_x, x)
    kld = -0.5 * torch.sum(1 + logvar_z - mean_z.pow(2) - logvar_z.exp())
    return l1 + beta * kld


def train():
    imgs = np.load(data_path)
    with open(info_path) as f:
        info = json.load(f)

    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(imgs / 256))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True)

    model = RHPBM(d)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for step in range(epoch):
        model_loss = 0
        for _, ([data]) in enumerate(loader):
            x = data.to(device)
            optimizer.zero_grad()
            _x, mean_z, logvar_z = model.forward(x)
            batch_loss = loss(_x, x, mean_z, logvar_z)
            batch_loss.backward()
            model_loss += batch_loss.item()
            optimizer.step()
        model_loss /= len(dataset)
        print('====> Epoch: %03d Loss : %0.8f' % ((step + 1), model_loss))
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    train()
