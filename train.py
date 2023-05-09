import json
import os

import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from tensorboardX import SummaryWriter

from model import RHPBM

batch = 100
epoch = 200
lr = 0.01
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

d = 10
beta = 0.9
SparsityLoss = nn.L1Loss(reduction='sum')

video = 'Video_003'
proj_dir = os.path.abspath('.')
dataset_dir = os.path.join(proj_dir, 'data', 'dataset')
data_path = os.path.join(dataset_dir, '%s.npy' % video)
info_path = os.path.join(dataset_dir, '%s.json' % video)
model_dir = os.path.join(proj_dir, 'checkpoint')


def loss(_x, x, mean_z, logvar_z):
    l1 = SparsityLoss(_x, x)
    kld = -0.5 * torch.sum(1 + logvar_z - mean_z.pow(2) - logvar_z.exp())
    return l1 + beta * kld


def train():
    imgs = np.load(data_path)
    with open(info_path) as f:
        info = json.load(f)
    loss_writer = SummaryWriter('./log')

    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(imgs / 256))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True)

    model = RHPBM(d)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch, eta_min=0.0001)
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
        scheduler.step()
        model_loss /= len(dataset)
        print('====> Epoch: %03d Loss : %0.8f' % ((step + 1), model_loss))
        loss_writer.add_scalar('Loss', model_loss, step + 1)
        if step and step % 10 == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, '%s_%d.pth' % (video, step)))

    torch.save(model.state_dict(), os.path.join(model_dir, '%s_%d.pth' % (video, epoch)))
    loss_writer.close()


if __name__ == '__main__':
    train()
