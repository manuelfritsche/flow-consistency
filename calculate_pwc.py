from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import sys
import yaml
import argparse
import numpy as np
import time
import re
import cv2
import os
import torch
from semseg.loader import get_loader


def optical_flow(loader, viz, cont):
    n_images = loader.__len__()

    for index in range(n_images):
        print("Estimating flow " + str(index+1) + " / " + str(n_images), end="\r")
        images, _, _ = loader.__getitem__(index)
        img_path = loader.files[loader.split][index].rstrip()
        flow_path = re.sub('leftImg8bit.png$', 'flow.npy', img_path)
        flow_path = re.sub('leftImg8bit', 'flow', flow_path)
        if cont and os.path.isfile(flow_path):
            continue
        flow_dir = flow_path.rpartition("/")[0]
        if not os.path.exists(flow_dir):
            os.makedirs(flow_dir)

        first = torch.FloatTensor(np.array(images[:3, :, :]).astype(np.float32))
        second = torch.FloatTensor(np.array(images[3:, :, :]).astype(np.float32))

        flow = run.estimate(first, second).permute(1, 2, 0).numpy()
        np.save(flow_path, flow)

        if viz:
            first = np.array(np.swapaxes(np.swapaxes(images[:3, :, :], 0, 1), 1, 2), dtype=float, order='c') * 255
            first_img_path = re.sub('_flow.npy$', '.png', flow_path)
            cv2.imwrite(first_img_path, first)
            hsv = np.zeros(first.shape, dtype=np.uint8)
            hsv[:, :, 0] = 255
            hsv[:, :, 1] = 255
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            flow_img_path = re.sub('.npy$', '.png', flow_path)
            cv2.imwrite(flow_img_path, rgb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        help="Configuration file to use"
    )
    parser.add_argument(
        '-cont', dest='cont', action='store_true',
        help='Continue instead of overwrite'
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    data_path = cfg['data']['path']

    train_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['train_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        n_img_after=1)

    val_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['val_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        n_img_after=1)

    test_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['test_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        n_img_after=1)

    # this is a hack to avoid changing the pytorch-pwc code
    os.chdir(os.path.join(os.getcwd(), 'pytorch-pwc'))
    sys.path[0] = os.getcwd()
    import run

    optical_flow(train_loader, viz=True, cont=args.cont)
    optical_flow(val_loader, viz=True, cont=args.cont)
    optical_flow(test_loader, viz=True, cont=args.cont)



