import json
import os
from glob import glob

import cv2
import numpy as np
import torch

from model import RHPBM
from rich.progress import track

d = 20
ckpt = '200'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

video = 'Video_002'
proj_dir = os.path.abspath('.')
frames = os.path.join(proj_dir, 'data', 'frame', video, '*')
model_path = os.path.join(proj_dir, 'checkpoint', '%s_%s.pth' % (video, ckpt))


def test():
    model = RHPBM(d)
    model.load_state_dict(torch.load(model_path))
    model.train(False)
    model.to(device)

    frame = None
    x = None

    output = None
    for frame_path in glob(frames):
        frame = cv2.imread(frame_path)
        frame = cv2.resize(frame, (299, 299))
        show_img = frame
        x = np.array([np.transpose(frame, (2, 0, 1))])
        x = torch.FloatTensor(x / 256).to(device)
        mu_z, logvar_z = model.encode(x)
        _z = RHPBM.reparam(mu_z, logvar_z)
        rec_img = model.decode(_z)
        rec_img = rec_img.cpu().data.numpy()[0].transpose(1, 2, 0)
        rec_img = (rec_img * 256).astype(np.uint8)
        cv2.imshow('rec_img', rec_img)
        cv2.waitKey(0)


def sample():
    model = RHPBM(d)
    model.load_state_dict(torch.load(model_path))
    model.train(False)
    model.to(device)

    while True:
        z = torch.randn(1, 20)
        print(z)
        z = z.to(device)
        rec_img = model.decode(z)
        rec_img = rec_img.cpu().data.numpy()[0].transpose(1, 2, 0)
        rec_img = (rec_img * 256).astype(np.uint8)
        cv2.imshow('sample', rec_img)
        cv2.waitKey(0)


if __name__ == '__main__':
    test()
    # sample()