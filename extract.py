import json
import os

import cv2
import numpy as np
import torch

from model import RHPBM
from rich.progress import track

d = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

video = 'Video_003'
proj_dir = os.path.abspath('.')
dataset_dir = os.path.join(proj_dir, 'data', 'dataset')
data_path = os.path.join(dataset_dir, '%s.npy' % video)
info_path = os.path.join(dataset_dir, '%s.json' % video)
model_path = os.path.join(proj_dir, 'checkpoint', '%s_200.pth' % video)


def extract():
    imgs = np.load(data_path)
    with open(info_path) as f:
        info = json.load(f)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter('%s_ori_rec.mp4' % video, fourcc, info['fps'], (299, 299 * 2))

    model = RHPBM(d)
    model.load_state_dict(torch.load(model_path))
    model.train(False)
    model.to(device)

    for img in track(imgs):
        x = torch.FloatTensor(img / 256).unsqueeze(0).to(device)
        mu_z, logvar_z = model.encode(x)
        _z = RHPBM.reparam(mu_z, logvar_z)
        rec_img = model.decode(_z)
        # rec_img, _, _ = model.forward(x)
        rec_img = rec_img.cpu().data.numpy()[0].transpose(1, 2, 0)
        rec_img = (rec_img * 256).astype(np.uint8)
        ori_img = img.transpose(1, 2, 0).astype(np.uint8)

        # rec_img = cv2.resize(rec_img, (info['width'], info['height']))
        # ori_img = cv2.resize(ori_img, (info['width'], info['height']))
        video_writer.write(np.concatenate([ori_img, rec_img], axis=0))
    video_writer.release()


if __name__ == '__main__':
    extract()
