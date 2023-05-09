import os
import cv2
import numpy as np
from rich.progress import Progress
import json

RESIZE = True
video_name = 'Video_003'
proj_dir = os.path.abspath('.')

video_path = os.path.join(proj_dir, 'data', 'video', '%s.avi' % video_name)
dataset_path = os.path.join(proj_dir, 'data', 'dataset', '%s.npy' % video_name)
info_path = os.path.join(proj_dir, 'data', 'dataset', '%s.json' % video_name)

cap = cv2.VideoCapture(video_path)
info = {
    'fps': int(cap.get(cv2.CAP_PROP_FPS)),
    'length': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    'channel': 3,
    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
}

if info['length'] <= 2000:
    dataset = np.empty([info['length'], info['channel'], 299, 299])  # [样本,通道(BGR),高,宽]
    with Progress() as progress:
        task = progress.add_task("loading %s" % video_name, total=info['length'])
        frame_idx = 0
        while cap.isOpened() and frame_idx < info['length']:
            ret, frame = cap.read()
            if ret:
                if RESIZE:
                    frame = cv2.resize(frame, (299, 299))
                dataset[frame_idx, :, :, :] = np.transpose(frame, (2, 0, 1))
                frame_idx += 1
                progress.advance(task, 1)
            else:
                progress.update(task, completed=info['length'])
                break
    np.save(dataset_path, dataset)

cap.release()
with open(info_path, 'w') as f:
    json.dump(info, f)
