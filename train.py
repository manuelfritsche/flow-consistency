import os
import sys
import yaml
import time
import shutil
import torch
import random
import argparse
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.utils import data
from tqdm import tqdm

from semseg.models import get_model
from semseg.loss import get_loss_function
from semseg.loader import get_loader
from semseg.utils import get_logger
from semseg.metrics import runningScore, averageMeter
from semseg.augmentations import get_composed_augmentations
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from tensorboardX import SummaryWriter


def train(cfg, writer, logger):
    # Setup seeds
    torch.manual_seed(cfg.get('seed', 1337))
    torch.cuda.manual_seed(cfg.get('seed', 1337))
    np.random.seed(cfg.get('seed', 1337))
    random.seed(cfg.get('seed', 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Augmentations
    augmentations = cfg['training'].get('augmentations', None)
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    data_path = cfg['data']['path']

    t_loader_lbl = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['train_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        augmentations=data_aug,
        frac_img=cfg['data']['frac_img'],
        get_flow=cfg['data']['get_flow'],
        discard_flow_bottom=cfg['data']['discard_flow_bottom'])

    if cfg['training']['batch_size_flow'] > 0:
        t_loader_flow = data_loader(
            data_path,
            is_transform=True,
            split=cfg['data']['train_split'],
            img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
            augmentations=data_aug,
            frac_img=1.0,
            frac_lbl=0.0,
            get_flow=cfg['data']['get_flow'],
            discard_flow_bottom=cfg['data']['discard_flow_bottom'])

    v_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['val_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),)

    n_classes = t_loader_lbl.n_classes
    trainloader_lbl = data.DataLoader(t_loader_lbl,
                                  batch_size=cfg['training']['batch_size_lbl'],
                                  num_workers=cfg['training']['n_workers'],
                                  shuffle=True)
    if cfg['training']['batch_size_flow'] > 0:
        trainloader_flow = data.DataLoader(t_loader_lbl,
                                    batch_size=cfg['training']['batch_size_flow'],
                                    num_workers=cfg['training']['n_workers'],
                                    shuffle=True)
        iterator_flow = iter(trainloader_flow)

    valloader = data.DataLoader(v_loader,
                                batch_size=cfg['training']['batch_size_lbl'],
                                num_workers=cfg['training']['n_workers'])

    # Retrieve Frequencies:
    if cfg['training']['frequency_weighting']:
        with open("frequencies.yml", 'r') as stream:
            frequency_data = yaml.load(stream)
        av_freq_labels = []
        for i in range(1, len(t_loader_lbl.class_names)):
            av_freq_labels.append(frequency_data[t_loader_lbl.class_names[i]])
        print("class frequencies: ")
        for index in range(n_classes):
            print("   " + str(t_loader_lbl.class_names[index + 1]) + ": " + "{0:.3f}".format(av_freq_labels[index]))
        weight_labels = 1.0 / (torch.FloatTensor(av_freq_labels).float().to(device)  + 0.01)

    # Setup Metrics
    running_metrics_val = runningScore(n_classes)

    # Setup Model
    model = get_model(cfg['model'], n_classes).to(device)

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg['training']['optimizer'].items()
                        if k != 'name'}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg['training']['lr_schedule'])

    loss_fn = get_loss_function(cfg)
    logger.info("Using loss {}".format(loss_fn))

    start_iter = 0
    if cfg['training']['resume'] is not None:
        if os.path.isfile(cfg['training']['resume']):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg['training']['resume'])
            )
            checkpoint = torch.load(cfg['training']['resume'])
            model.load_state_dict(checkpoint["model_state"])
            if not cfg["training"]["reset_optimizer"]:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg['training']['resume'], checkpoint["epoch"]
                )
            )
            del checkpoint
        else:
            logger.info("No checkpoint found at '{}'".format(cfg['training']['resume']))

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -100.0
    i = start_iter
    flag = True

    while i <= cfg['training']['train_iters'] and flag:
        for (t_images, t_labels, t_flows) in trainloader_lbl:
            if cfg['training']['batch_size_flow'] > 0:
                try:
                    (f_images, f_labels, f_flows) = next(iterator_flow)
                except StopIteration:
                    iterator_flow = iter(trainloader_flow)
                    (f_images, f_labels, f_flows) = next(iterator_flow)

                images = torch.cat((t_images, f_images), 0)
                labels = torch.cat((t_labels, f_labels), 0)
                flows = torch.cat((t_flows, f_flows), 0)
            else:
                images = t_images
                labels = t_labels
                flows = t_flows

            i += 1
            start_ts = time.time()
            scheduler.step()
            model.train()
            images = images.to(device)
            labels = labels.to(device)
            flows = flows.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            if cfg['training']['loss']['name'] == 'cross_flow':
                loss = loss_fn(input=outputs, target=labels, flow=flows)
            else:
                if cfg['training']['frequency_weighting']:
                    loss = loss_fn(input=outputs, target=labels, weight=weight_labels)
                else:
                    loss = loss_fn(input=outputs, target=labels)

            loss.backward()
            optimizer.step()

            time_meter.update(time.time() - start_ts)

            if (i + 1) % cfg['training']['print_interval'] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(i + 1,
                                           cfg['training']['train_iters'],
                                           loss.item(),
                                           time_meter.avg / (cfg['training']['batch_size_lbl'] + cfg['training']['batch_size_flow']))

                print(print_str)
                logger.info(print_str)
                writer.add_scalar('loss/train_loss', loss.item(), i + 1)
                time_meter.reset()

            if (i + 1) % cfg['training']['val_interval'] == 0 or \
                    (i + 1) == cfg['training']['train_iters']:
                model.eval()
                with torch.no_grad():
                    for i_val, (images_val, labels_val, _) in tqdm(enumerate(valloader)):
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)

                        outputs = model(images_val)
                        val_loss = loss_fn(input=outputs, target=labels_val)

                        pred = outputs.data.max(1)[1].cpu().numpy()
                        gt = labels_val.data.cpu().numpy()

                        running_metrics_val.update(gt, pred)
                        val_loss_meter.update(val_loss.item())

                writer.add_scalar('loss/val_loss', val_loss_meter.avg, i + 1)
                logger.info("Iter %d Loss: %.4f" % (i + 1, val_loss_meter.avg))

                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(k, v)
                    logger.info('{}: {}'.format(k, v))
                    writer.add_scalar('val_metrics/{}'.format(k), v, i + 1)

                for k, v in class_iou.items():
                    logger.info('{}: {}'.format(k, v))
                    writer.add_scalar('val_metrics/cls_{}'.format(k), v, i + 1)

                val_loss_meter.reset()
                running_metrics_val.reset()

                if score["Mean IoU : \t"] >= best_iou:
                    best_iou = score["Mean IoU : \t"]
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join(writer.file_writer.get_logdir(),
                                             "{}_{}_best_model.pkl".format(
                                                 cfg['model']['arch'],
                                                 cfg['data']['dataset']))
                    torch.save(state, save_path)
                print("Best mIoU for " + cfg['training']['logdir'] + ": " + str(best_iou))

            if (i + 1) == cfg['training']['train_iters']:
                flag = False
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        help="Configuration file to use"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    index = 0
    # logdir = os.path.join('runs', os.path.basename(args.config)[:-4], cfg["training"]["logdir"] + "_" + str(index))
    logdir = os.path.join('runs', cfg["training"]["logdir"] + "_" + str(index))
    while True:
        index += 1
        if os.path.exists(logdir):
            logdir = logdir[:-len(str(index-1))] + str(index)
        else:
            break
    writer = SummaryWriter(log_dir=logdir)

    print('RUNDIR: {}'.format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info('Let the games begin')

    train(cfg, writer, logger)
