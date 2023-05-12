import json
import math
import os

import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split

from model import RHPBM

batch = 64
epoch = 200
lr = 0.02
lr_min = 0.0005
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

d = 20
beta = 0.6
beta_max_epoch = 64
beta_iter = 0
beta_list = np.linspace(beta, 1, beta_max_epoch)
test_size = 0.3

SparsityLoss = nn.L1Loss(reduction='sum')

video = 'Video_003'
MIX = True
MIX_LIST = ['Video_002', 'Video_003', 'Video_004']
proj_dir = os.path.abspath('.')
dataset_dir = os.path.join(proj_dir, 'data', 'dataset')
data_path = os.path.join(dataset_dir, '%s.npy' % video)
info_path = os.path.join(dataset_dir, '%s.json' % video)
model_dir = os.path.join(proj_dir, 'checkpoint')

save_log = False


def update_beta():
    global beta, beta_iter
    if beta_iter < beta_max_epoch:
        beta = beta_list[beta_iter]
        beta_iter += 1


def loss(_x, x, mean_z, logvar_z):
    l1 = SparsityLoss(_x, x)
    kld = -0.5 * torch.sum(1 + logvar_z - mean_z.pow(2) - logvar_z.exp())
    return l1 + beta * kld


def train():
    global video
    imgs = None
    if not MIX:
        imgs = np.load(data_path)
    else:
        video = 'mixed'
        for vid in MIX_LIST:
            vid_imgs = np.load(os.path.join(dataset_dir, '%s.npy' % vid))
            if imgs is None:
                imgs = vid_imgs
            else:
                imgs = np.append(imgs, vid_imgs, axis=0)

    # with open(info_path) as f:
    #     info = json.load(f)
    print(imgs.shape)
    if save_log:
        loss_writer = SummaryWriter('./log')

    # dataset, _ = train_test_split(imgs, test_size=test_size)
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(imgs / 256))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True)

    model = RHPBM(d)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch, eta_min=lr_min)
    for step in range(epoch):
        model_loss = 0
        update_beta()
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
        if save_log:
            loss_writer.add_scalar('Loss', model_loss, step + 1)
        if step and step % 50 == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, '%s_%d.pth' % (video, step)))

    torch.save(model.state_dict(), os.path.join(model_dir, '%s_%d.pth' % (video, epoch)))
    if save_log:
        loss_writer.close()


if __name__ == '__main__':
    train()
