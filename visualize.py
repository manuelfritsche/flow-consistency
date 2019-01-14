import os
import yaml
import torch
import torch.nn.functional as F
import random
import argparse
import numpy as np
import cv2

from torch.utils import data

from semseg.models import get_model
from semseg.loader import get_loader
from semseg.loss.loss import shift
from semseg.utils import convert_state_dict


LABEL_COLORS = np.array([
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32),
    (0, 0, 142),
    (0, 0, 0),
])

# switch bgr to rgb
LABEL_COLORS = LABEL_COLORS[:, [2, 1, 0]]


def visualize(cfg):
    base_dir = os.path.join('visualization', cfg["training"]["logdir"])
    for task in cfg['training']["tasks"]:
        if not os.path.exists(base_dir + "/" + task):
            os.makedirs(base_dir + "/" + task)

    # Setup seeds
    torch.manual_seed(cfg.get('seed', 1337))
    torch.cuda.manual_seed(cfg.get('seed', 1337))
    np.random.seed(cfg.get('seed', 1337))
    random.seed(cfg.get('seed', 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    data_path = cfg['data']['path']

    loader = data_loader(
        data_path,
        split=cfg['data']['val_split'],
        is_transform=True,
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        get_flow=True,
        discard_flow_bottom=cfg['data']['discard_flow_bottom'],
    )

    n_classes = loader.n_classes
    vizloader = data.DataLoader(loader,
                                batch_size=1,
                                shuffle=False)

    # Setup Model
    model = get_model(cfg['model'], n_classes).to(device)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    i = 0
    while i <= cfg['training']['train_iters']:
        for (images, labels, flows) in vizloader:
            i += 1
            model.train()
            images = images.to(device)
            target = labels.to(device)
            flow = flows.to(device)
            pred = model(images)
            if 'input' in cfg['training']['tasks']:
                visualize_input(images, i, base_dir)
            if 'output' in cfg['training']['tasks']:
                visualize_output(pred, i, base_dir)
            if 'ground_truth' in cfg['training']['tasks']:
                visualize_ground_truth(target, i, base_dir)
            if 'flow' in cfg['training']['tasks']:
                visualize_flow(flow, i, base_dir)
            if 'fl_weights' in cfg['training']['tasks']:
                visualize_fl_weights(pred, flow, i, base_dir)
            print("done with " + str(i) + "...")
            if i > cfg['training']['train_iters']:
                break


def visualize_input(input, index, base_dir):
    input = input.permute(0, 2, 3, 1).detach().cpu().numpy()[0, ...] * 256
    cv2.imwrite(base_dir + "/input/" + str(index) + ".png", input)


def visualize_output(output, index, base_dir):
    output = output.detach().cpu().numpy()[0, ...]
    output = np.argmax(output, axis=0)
    output = LABEL_COLORS[output]
    cv2.imwrite(base_dir + "/output/" + str(index) + ".png", output)


def visualize_ground_truth(target, index, base_dir):
    target = target.detach().cpu().numpy()[0, ...]
    target[target == -1] = -2
    target[target == 250] = -1
    target = LABEL_COLORS[target]
    cv2.imwrite(base_dir + "/ground_truth/" + str(index) + ".png", target)


def visualize_flow(flow, index, base_dir):
    # flow_test = f_flow.permute(1, 2, 3, 0).cpu().numpy()[0, ...]
    flow = flow.permute(0, 2, 3, 1).cpu().numpy()[0, ...]
    s_x, s_y, _ = np.shape(flow)
    hsv = np.zeros([s_x, s_y, 3], dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(base_dir + "/flow/" + str(index) + ".png", rgb)


def visualize_fl_weights(output, flow, i, base_dir):
    output = F.softmax(output, dim=1).permute(1, 0, 2, 3)
    flow = flow.permute(1, 0, 2, 3)
    shifted_pred, shifted_flow = shift(output.detach(), flow, 1, 0)
    norm = torch.norm(flow - shifted_flow, dim=0, keepdim=True)
    ignore = torch.norm(shifted_flow, dim=0, keepdim=True) == 0
    norm[ignore] = 0
    norm = norm.cpu().numpy()[0, 0, ...]
    weight = np.maximum(0, norm - np.max(norm) / 10)
    weight = cv2.normalize(weight, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite(base_dir + "/fl_weights/" + str(i) + ".png", weight)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        help="Configuration file to use"
    )
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        help="Path to the saved model",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    visualize(cfg)